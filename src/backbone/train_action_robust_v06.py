"""Build and fit a corruption-trained S4 action student for backbone v0.6."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from src.backbone.actions import ActionModelSession
from src.backbone.contracts import Observation, TrackingFrame, WorldInput
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.source import SoccerTrackSource
from src.backbone.train_action import _metrics, load_split
from src.belief_observation import encode_observations
from src.models.partial_observation_student import PartialObservationStudent
from src.player_centric_dataset import ACTION_LABELS, stable_seed
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "outputs/backbone_v0_5_20261020/corpus/training_index.json"
DEFAULT_BASELINE = ROOT / "outputs/backbone_v0_5_20261020/training/best.pt"
FIT_MATCHES = ["117092", "117093", "118576", "118578", "128057", "132831"]
DEV_MATCHES = ["132877"]
EVAL_MATCHES = ["118577", "128058"]
AGE_SCALE = 8.0 / 5.0
CORRUPTION = {
    "version": "football.action-observation-corruption.0.1",
    "provider_dropout": True,
    "provider_dropout_description": "Causal 0.6-2.2s synthetic visibility gaps before memory construction",
    "additional_correlated_gap_probability": 0.45,
    "additional_correlated_gap_samples": [2, 5],
    "identity_break_probability": 0.20,
    "identity_break_policy": "One non-actor segment changes at first model-history sample, when available",
    "coordinate_noise_std_m": 0.15,
    "seed": 62026,
}
HIGH_STRESS_CORRUPTION = {
    "version": "football.action-observation-corruption-high.0.1",
    "provider_dropout": True,
    "provider_dropout_description": "Same causal provider dropout as training; additional gaps/noise/identity breaks intensified for frozen stress evaluation.",
    "additional_correlated_gap_probability": 0.80,
    "additional_correlated_gap_samples": [3, 7],
    "identity_break_probability": 0.50,
    "identity_break_policy": "One non-actor segment changes at first model-history sample, when available",
    "coordinate_noise_std_m": 0.30,
    "seed": 62027,
}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _corrupt_history(
    history: list[Observation], sample_id: str, *, seed: int,
    corruption: dict | None = None,
) -> tuple[list[Observation], dict]:
    if len(history) != 11 or any(observation.actor is None for observation in history):
        raise ValueError(f"{sample_id}: corruption requires 11 actor-observed samples")
    rng = np.random.default_rng(stable_seed(sample_id, seed))
    policy = CORRUPTION if corruption is None else corruption
    corrupted = list(history)
    actor_track = corrupted[-1].actor_track
    identity_track = None
    first_measurements = {
        measurement.track: measurement
        for measurement in corrupted[0].measurements
        if measurement.track != actor_track
    }
    candidates = []
    for track, first in first_measurements.items():
        if all(
            measurement.segment == first.segment
            for observation in corrupted
            for measurement in observation.measurements
            if measurement.track == track
        ):
            candidates.append((track, first.segment))
    if candidates and rng.random() < policy["identity_break_probability"]:
        identity_track, old_segment = candidates[int(rng.integers(len(candidates)))]
        for index, observation in enumerate(corrupted):
            measurements = tuple(
                replace(measurement, segment=old_segment + 1)
                if measurement.track == identity_track and measurement.segment == old_segment
                else measurement
                for measurement in observation.measurements
            )
            corrupted[index] = replace(observation, measurements=measurements)

    gap_track = None
    removed = 0
    gap_len = 0
    tracks = sorted({
        measurement.track
        for observation in corrupted
        for measurement in observation.measurements
        if measurement.track != actor_track and measurement.track != identity_track
    })
    gap_candidates = [
        track for track in tracks
        if sum(any(m.track == track for m in obs.measurements) for obs in corrupted) >= 3
    ]
    if gap_candidates and rng.random() < policy["additional_correlated_gap_probability"]:
        gap_track = gap_candidates[int(rng.integers(len(gap_candidates)))]
        gap_len = int(rng.integers(
            policy["additional_correlated_gap_samples"][0],
            policy["additional_correlated_gap_samples"][1] + 1,
        ))
        gap_start = int(rng.integers(0, len(corrupted) - gap_len + 1))
        gap_indices = set(range(gap_start, gap_start + gap_len))
        for index in gap_indices:
            observation = corrupted[index]
            measurements = tuple(m for m in observation.measurements if m.track != gap_track)
            removed += len(observation.measurements) - len(measurements)
            corrupted[index] = replace(observation, measurements=measurements)
    return corrupted, {
        "sample_id": sample_id,
        "identity_break_track": identity_track,
        "correlated_gap_track": gap_track,
        "correlated_gap_samples": gap_len if gap_track is not None else 0,
        "additional_detections_removed": removed,
    }


def _encode_history(history: list[Observation], sample_id: str) -> dict[str, np.ndarray]:
    tokens: list[tuple[str, int]] = []
    for observation in history:
        for measurement in observation.measurements:
            token = (measurement.track, measurement.segment)
            if token not in tokens:
                tokens.append(token)
            if measurement.team not in (1, 2) or measurement.goalkeeper is None:
                raise ValueError(f"{sample_id}: unknown team or role in visible input")
    if len(tokens) > 22:
        raise ValueError(f"{sample_id}: more than 22 seen identities")
    actor = history[-1].actor
    if actor is None:
        raise ValueError(f"{sample_id}: actor missing at query")
    actor_index = tokens.index((actor.track, actor.segment))
    shape = (11, len(tokens))
    xy = np.zeros((*shape, 2), np.float32)
    observed = np.zeros(shape, bool)
    segments = np.zeros(shape, np.int32)
    teams = np.zeros(shape, np.int32)
    roles = np.zeros(shape, np.int32)
    for time, observation in enumerate(history):
        for measurement in observation.measurements:
            slot = tokens.index((measurement.track, measurement.segment))
            xy[time, slot] = measurement.xy
            observed[time, slot] = True
            segments[time, slot] = measurement.segment
            teams[time, slot] = measurement.team
            roles[time, slot] = int(measurement.goalkeeper)
    encoded = encode_observations(
        xy, observed, segments, teams, roles, actor_index, observed.copy(),
    )
    encoded["x"][..., 6] = np.clip(encoded["x"][..., 6] * AGE_SCALE, 0.0, 1.0)
    return {key: encoded[key] for key in ("x", "mask", "visible")}


def _corrupted_history(
    source: SoccerTrackSource, row: dict, sample_id: str,
    *, corruption: dict | None = None,
) -> tuple[list[Observation], dict]:
    policy = CORRUPTION if corruption is None else corruption
    frames = [int(value) for value in row["input_frame_indices"]]
    query = int(row["query_frame"])
    if len(frames) != 11 or frames[-1] != query or frames != list(range(frames[0], query + 1, 5)):
        raise ValueError(f"{sample_id}: indexed causal history does not match the frozen 11x5 grid")
    actor_track = str(row["actor_track"])
    provider = SyntheticObservationProvider(
        actor_track, seed=42, fov=120, dropout=bool(policy["provider_dropout"]),
    )
    noise_rng = np.random.default_rng(stable_seed(sample_id, int(policy["seed"])))
    warmup_start = max(0, frames[0] - 25)
    observations = []
    for frame_index in range(warmup_start, query + 1, 5):
        frame = source.frame(frame_index)
        noisy_measurements = []
        for measurement in frame.measurements:
            noise = noise_rng.normal(
                0.0, float(policy["coordinate_noise_std_m"]), size=2,
            )
            xy = np.asarray(measurement.xy, dtype=np.float64) + noise
            noisy_measurements.append(replace(
                measurement, xy=(float(xy[0]), float(xy[1])),
            ))
        noisy_frame: TrackingFrame = replace(frame, measurements=tuple(noisy_measurements))
        observations.append(provider.observe(noisy_frame))
    history = observations[-11:]
    effective_frames = [item.index for item in history]
    if (len(history) != 11 or any(item.actor is None for item in history)
            or effective_frames[-1] > query
            or any(right - left != 5 for left, right in zip(effective_frames, effective_frames[1:]))):
        raise ValueError(f"{sample_id}: observation builder failed the causal 11-sample contract")
    return history, {
        "query_frame": query,
        "effective_history_frames": effective_frames,
        "query_lag_frames": query - effective_frames[-1],
    }


def build_corrupted_split(index_path: Path, split: str, output: Path) -> dict:
    if split not in ("fitting", "development"):
        raise ValueError("Corruption dataset generation is restricted to fitting/development")
    original = load_split(index_path, split)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    metadata_by_id: dict[str, tuple[dict, dict]] = {}
    for shard in index["shards"]:
        if shard["split"] != split:
            continue
        folder = Path(shard["folder"])
        for row in _read_jsonl(folder / "student_metadata.jsonl"):
            sample_id = str(row["sample_id"])
            if sample_id in metadata_by_id:
                raise ValueError(f"Duplicate metadata sample ID: {sample_id}")
            metadata_by_id[sample_id] = (row, shard)
    sample_ids = original["sample_ids"].astype(str).tolist()
    if set(sample_ids) != set(metadata_by_id):
        raise ValueError(f"Indexed input/metadata ID mismatch for {split}")

    sources: dict[tuple[str, str], SoccerTrackSource] = {}
    xs = np.empty((len(sample_ids), 11, 22, 10), dtype=np.float32)
    masks = np.empty((len(sample_ids), 11, 22), dtype=bool)
    visibles = np.empty_like(masks)
    corruption_rows = []
    for row_index, sample_id in enumerate(sample_ids):
        metadata, shard = metadata_by_id[sample_id]
        key = (str(shard["match"]), str(shard["half"]))
        source = sources.get(key)
        if source is None:
            source = SoccerTrackSource(ROOT, *key)
            sources[key] = source
        history, alignment = _corrupted_history(
            source, metadata, sample_id, corruption=CORRUPTION,
        )
        history, corruption_record = _corrupt_history(
            history, sample_id, seed=int(CORRUPTION["seed"]), corruption=CORRUPTION,
        )
        corruption_record.update(alignment)
        encoded = _encode_history(history, sample_id)
        xs[row_index] = encoded["x"]
        masks[row_index] = encoded["mask"]
        visibles[row_index] = encoded["visible"]
        corruption_rows.append(corruption_record)

    if not np.isfinite(xs).all():
        raise FloatingPointError(f"Non-finite corruption-trained inputs in {split}")
    if np.any(xs[..., 6] > 0.40001):
        raise ValueError(f"Corruption-trained age normalization is out of range in {split}")
    target = output / "robust_data" / f"{split}.npz"
    _write_npz(
        target, x=xs, mask=masks, visible=visibles,
        sample_ids=np.asarray(sample_ids, dtype=str),
        y=original["y"].astype(np.int64),
    )
    _write_jsonl(output / "robust_data" / f"{split}_corruptions.jsonl", corruption_rows)
    manifest = {
        "schema_version": "football.action-robustness-dataset.0.1",
        "status": "COMPLETE",
        "split": split,
        "samples": len(sample_ids),
        "matches": sorted({str(value) for value in original["matches"]}),
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "index_sha256": file_digest(index_path),
        "corruption": CORRUPTION,
        "corruption_counts": {
            "identity_breaks": sum(row["identity_break_track"] is not None for row in corruption_rows),
            "correlated_gaps": sum(row["correlated_gap_track"] is not None for row in corruption_rows),
            "removed_detections": sum(row["additional_detections_removed"] for row in corruption_rows),
            "provider_correlated_dropout": bool(CORRUPTION["provider_dropout"]),
            "coordinate_noise_std_m": CORRUPTION["coordinate_noise_std_m"],
            "query_aligned_samples": sum(row["query_lag_frames"] == 0 for row in corruption_rows),
            "causal_lagged_samples": sum(row["query_lag_frames"] > 0 for row in corruption_rows),
            "maximum_query_lag_frames": max(row["query_lag_frames"] for row in corruption_rows),
        },
        "data_sha256": file_digest(target),
        "corruption_log_sha256": file_digest(output / "robust_data" / f"{split}_corruptions.jsonl"),
    }
    write_json(output / "robust_data" / f"{split}_manifest.json", manifest)
    return manifest


def _save_torch(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _robust_data(path: Path, expected: dict) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if data["sample_ids"].astype(str).tolist() != expected["sample_ids"].astype(str).tolist():
        raise ValueError(f"Robust dataset sample IDs differ from indexed split: {path}")
    if not np.array_equal(data["y"], expected["y"]):
        raise ValueError(f"Robust dataset labels differ from indexed targets: {path}")
    if data["x"].shape != (len(data["y"]), 11, 22, 10):
        raise ValueError(f"Robust dataset has an unexpected feature shape: {path}")
    if data["mask"].shape != (len(data["y"]), 11, 22) or data["visible"].shape != data["mask"].shape:
        raise ValueError(f"Robust dataset mask shape mismatch: {path}")
    if not np.isfinite(data["x"]).all():
        raise FloatingPointError(f"Robust dataset contains non-finite values: {path}")
    return data


def validate_robust_data(index_path: Path, output: Path) -> dict:
    expected_matches = {"fitting": FIT_MATCHES, "development": DEV_MATCHES}
    split_results = {}
    for split, matches in expected_matches.items():
        expected = load_split(index_path, split)
        folder = output / "robust_data"
        manifest = json.loads((folder / f"{split}_manifest.json").read_text(encoding="utf-8"))
        data = _robust_data(folder / f"{split}.npz", expected)
        log = _read_jsonl(folder / f"{split}_corruptions.jsonl")
        if (manifest.get("schema_version") != "football.action-robustness-dataset.0.1"
                or manifest.get("split") != split
                or manifest.get("index_sha256") != file_digest(index_path)
                or manifest.get("samples") != len(expected["y"])
                or manifest.get("matches") != matches
                or manifest.get("data_sha256") != file_digest(folder / f"{split}.npz")
                or manifest.get("corruption_log_sha256")
                != file_digest(folder / f"{split}_corruptions.jsonl")):
            raise ValueError(f"Corruption dataset manifest mismatch for {split}")
        if len(log) != len(expected["y"]) or [row["sample_id"] for row in log] != expected["sample_ids"].astype(str).tolist():
            raise ValueError(f"Corruption log/sample alignment mismatch for {split}")
        if (np.any(data["y"] < 0) or np.any(data["y"] >= len(ACTION_LABELS))
                or np.any(~np.isfinite(data["x"])) or np.any(data["x"][..., 6] > 0.40001)):
            raise ValueError(f"Corruption dataset has invalid labels/features for {split}")
        actor_flags = (data["x"][..., 9] > 0.5) & data["mask"]
        actor_visible = (data["visible"][:, -1, :] & actor_flags[:, -1, :]).sum(axis=-1)
        if not np.all(actor_flags.sum(axis=-1) == 1) or not np.all(actor_visible == 1):
            raise ValueError(f"Corruption dataset actor coverage failed for {split}")
        if any(
            len(row["effective_history_frames"]) != 11
            or row["effective_history_frames"][-1] > row["query_frame"]
            or row["query_lag_frames"] != row["query_frame"] - row["effective_history_frames"][-1]
            or row["query_lag_frames"] not in range(5)
            or any(b - a != 5 for a, b in zip(row["effective_history_frames"], row["effective_history_frames"][1:]))
            for row in log
        ):
            raise ValueError(f"Corruption dataset includes unsupported or future-aligned history for {split}")
        counts = manifest["corruption_counts"]
        if (not counts["provider_correlated_dropout"] or not counts["identity_breaks"]
                or not counts["correlated_gaps"] or not counts["removed_detections"]):
            raise ValueError(f"Expected pre-memory dropout, gaps and identity breaks were not observed in {split}")
        split_results[split] = {
            "samples": len(expected["y"]),
            "matches": matches,
            "sample_ids_sha256": manifest["sample_ids_sha256"],
            "data_sha256": manifest["data_sha256"],
            "identity_breaks": counts["identity_breaks"],
            "correlated_gaps": counts["correlated_gaps"],
            "additional_detections_removed": counts["removed_detections"],
            "query_aligned_samples": counts["query_aligned_samples"],
            "causal_lagged_samples": counts["causal_lagged_samples"],
            "maximum_query_lag_frames": counts["maximum_query_lag_frames"],
            "history_causality": "PASS",
        }
    report = {
        "schema_version": "football.action-robustness-data-validation.0.1",
        "status": "PASS",
        "training_index_sha256": file_digest(index_path),
        "retrospective_evaluation_split_read_for_training": False,
        "splits": split_results,
    }
    write_json(output / "robust_data_validation.json", report)
    return report


@torch.inference_mode()
def _predict(model: PartialObservationStudent, data: dict, device: torch.device) -> np.ndarray:
    model.eval()
    logits: list[np.ndarray] = []
    for start in range(0, len(data["y"]), 256):
        stop = min(start + 256, len(data["y"]))
        x = torch.from_numpy(data["x"][start:stop]).to(device)
        mask = torch.from_numpy(data["mask"][start:stop]).to(device)
        visible = torch.from_numpy(data["visible"][start:stop]).to(device)
        output = model(x, mask, visible, memory_mode="memory")["action"].float()
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite action logits")
        logits.append(output.cpu().numpy())
    return np.concatenate(logits)


def train_robust(index_path: Path, output: Path, *, resume: bool = False) -> dict:
    fit_expected = load_split(index_path, "fitting")
    dev_expected = load_split(index_path, "development")
    fit = _robust_data(output / "robust_data/fitting.npz", fit_expected)
    dev = _robust_data(output / "robust_data/development.npz", dev_expected)
    fit_manifest = json.loads((output / "robust_data/fitting_manifest.json").read_text(encoding="utf-8"))
    dev_manifest = json.loads((output / "robust_data/development_manifest.json").read_text(encoding="utf-8"))
    if any(
        manifest.get("schema_version") != "football.action-robustness-dataset.0.1"
        or manifest.get("split") != split
        or manifest.get("index_sha256") != file_digest(index_path)
        for split, manifest in (("fitting", fit_manifest), ("development", dev_manifest))
    ):
        raise ValueError("Corrupted inputs are incomplete or were built from a different training index")
    if fit_manifest["matches"] != FIT_MATCHES or dev_manifest["matches"] != DEV_MATCHES:
        raise ValueError("Robust training split provenance mismatch")
    if set(fit_manifest["matches"]) & set(dev_manifest["matches"]):
        raise ValueError("Robust fitting/development split overlap")
    if set(EVAL_MATCHES) & (set(fit_manifest["matches"]) | set(dev_manifest["matches"])):
        raise ValueError("Retrospective evaluation match entered robust training data")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    source_provenance = {}
    for shard in index["shards"]:
        if shard["split"] not in ("fitting", "development"):
            continue
        source = SoccerTrackSource(ROOT, str(shard["match"]), shard["half"])
        source_provenance[f"{shard['match']}_{shard['half']}"] = source.provenance

    baseline_path = DEFAULT_BASELINE
    initial = torch.load(baseline_path, map_location="cpu", weights_only=False)
    initial_state = initial["model"] if isinstance(initial, dict) and "model" in initial else initial
    if not isinstance(initial_state, dict):
        raise ValueError("Baseline action checkpoint does not contain a model state dictionary")
    initial_hash = file_digest(baseline_path)
    config = {
        "schema_version": "football.action-training-robust.0.1",
        "index_path": str(index_path.resolve()),
        "index_sha256": file_digest(index_path),
        "initialization_path": str(baseline_path.resolve()),
        "initialization_sha256": initial_hash,
        "fitting_matches": FIT_MATCHES,
        "development_matches": DEV_MATCHES,
        "retrospective_evaluation_matches": EVAL_MATCHES,
        "class_mapping": list(ACTION_LABELS),
        "source_provenance": source_provenance,
        "model": {"variant": "S4", "fov_index": 1, "width": 96, "layers": 2, "heads": 4, "dropout": 0.1},
        "optimization": {
            "seed": 62026, "learning_rate": 0.0001, "weight_decay": 0.0001,
            "batch_size": 256, "max_epochs": 5, "patience": 2,
            "gradient_clip_norm": 1.0, "loss": "action-only cross entropy",
            "mixed_precision": True,
        },
        "corruption": CORRUPTION,
        "data_sha256": {
            "fitting": file_digest(output / "robust_data/fitting.npz"),
            "development": file_digest(output / "robust_data/development.npz"),
        },
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    model_dir = output / "robust_training"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "variant": "S4",
        "fov_index": 1,
        "fitting_matches": FIT_MATCHES,
        "development_matches": DEV_MATCHES,
        "config": {"seed": 62026, "width": 96, "layers": 2, "heads": 4, "dropout": 0.1},
    }
    write_json(model_dir / "config.json", model_config)
    write_json(model_dir / "run_config.json", config)

    seed = int(config["optimization"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartialObservationStudent(width=96, layers=2, heads=4, dropout=0.1, belief=True).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_path = model_dir / "best.pt"
    state_path = model_dir / "last.pt"
    next_epoch, best_macro_f1, best_epoch, stale, history = 0, -1.0, 0, 0, []
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"No robust-training safe checkpoint to resume: {state_path}")
        saved = torch.load(state_path, map_location=device, weights_only=False)
        if saved.get("fingerprint") != config_fingerprint:
            raise ValueError("Robust training checkpoint fingerprint mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        next_epoch, best_macro_f1 = int(saved["next_epoch"]), float(saved["best_macro_f1"])
        best_epoch, stale, history = int(saved["best_epoch"]), int(saved["stale"]), list(saved["history"])
        _restore_rng(saved["rng"])
    elif state_path.exists():
        raise FileExistsError(f"Existing robust checkpoint requires --resume: {state_path}")

    opt = config["optimization"]
    while next_epoch < int(opt["max_epochs"]) and stale < int(opt["patience"]):
        epoch = next_epoch
        order = np.random.default_rng(seed + epoch).permutation(len(fit["y"]))
        model.train()
        loss_sum = 0.0
        for start in range(0, len(order), int(opt["batch_size"])):
            indices = order[start:start + int(opt["batch_size"])]
            x = torch.from_numpy(fit["x"][indices]).to(device)
            mask = torch.from_numpy(fit["mask"][indices]).to(device)
            visible = torch.from_numpy(fit["visible"][indices]).to(device)
            target = torch.from_numpy(fit["y"][indices]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and bool(opt["mixed_precision"]),
            ):
                logits = model(x, mask, visible, memory_mode="memory")["action"].float()
                loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite robust action training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(opt["gradient_clip_norm"]), error_if_nonfinite=True,
            )
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)

        dev_logits = _predict(model, dev, device)
        dev_metrics = _metrics(dev["y"], dev_logits.argmax(axis=-1))
        train_loss = loss_sum / max(1, len(order))
        improved = dev_metrics["macro_f1"] > best_macro_f1 + 0.0005
        if improved:
            best_macro_f1, best_epoch, stale = dev_metrics["macro_f1"], epoch + 1, 0
            temporary = best_path.with_suffix(".pt.tmp")
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, best_path)
        else:
            stale += 1
        history.append({
            "epoch": epoch + 1,
            "training_loss": train_loss,
            "development_corrupted": dev_metrics,
            "best_epoch": best_epoch,
            "stale_epochs": stale,
        })
        next_epoch = epoch + 1
        _save_torch(state_path, {
            "fingerprint": config_fingerprint,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "next_epoch": next_epoch,
            "best_macro_f1": best_macro_f1,
            "best_epoch": best_epoch,
            "stale": stale,
            "history": history,
            "rng": _rng_state(),
        })
        print(json.dumps({"epoch": next_epoch, "loss": train_loss,
                          "development_corrupted_macro_f1": dev_metrics["macro_f1"],
                          "best_epoch": best_epoch, "stale": stale}), flush=True)

    if not best_path.is_file():
        raise RuntimeError("Robust training ended without a best checkpoint")
    best_state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state, strict=True)
    baseline_state = initial_state
    baseline = PartialObservationStudent(width=96, layers=2, heads=4, dropout=0.1, belief=True).to(device)
    baseline.load_state_dict(baseline_state, strict=True)

    clean_dev = {
        "x": dev_expected["x"], "mask": dev_expected["mask"],
        "visible": dev_expected["visible"], "y": dev_expected["y"],
    }
    clean_logits = _predict(model, clean_dev, device)
    robust_logits = _predict(model, dev, device)
    baseline_clean_logits = _predict(baseline, clean_dev, device)
    baseline_robust_logits = _predict(baseline, dev, device)
    comparison = {
        "common_sample_ids": int(len(dev["sample_ids"])),
        "common_ids_sha256": dev_manifest["sample_ids_sha256"],
        "coverage": {
            "baseline_clean": 1.0,
            "robust_clean": 1.0,
            "baseline_corrupted": 1.0,
            "robust_corrupted": 1.0,
        },
        "metrics": {
            "baseline_clean_development": _metrics(dev["y"], baseline_clean_logits.argmax(-1)),
            "robust_clean_development": _metrics(dev["y"], clean_logits.argmax(-1)),
            "baseline_corrupted_development": _metrics(dev["y"], baseline_robust_logits.argmax(-1)),
            "robust_corrupted_development": _metrics(dev["y"], robust_logits.argmax(-1)),
        },
        "protocol": "Same 2,158 development IDs and labels in all cells; fitting-only optimization; checkpoint selected on corrupted development macro-F1.",
        "retrospective_evaluation_used": False,
    }
    write_json(model_dir / "comparison.json", comparison)
    result = {
        "schema_version": "football.action-training-robust-result.0.1",
        "status": "COMPLETE",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "training_examples": len(fit["y"]),
        "development_examples": len(dev["y"]),
        "epochs_completed": next_epoch,
        "best_epoch": best_epoch,
        "best_development_corrupted_macro_f1": best_macro_f1,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": file_digest(best_path),
        "baseline_checkpoint_sha256": initial_hash,
        "training_index_sha256": file_digest(index_path),
        "fit_matches": FIT_MATCHES,
        "development_matches": DEV_MATCHES,
        "retrospective_evaluation_matches": EVAL_MATCHES,
        "class_mapping": list(ACTION_LABELS),
        "source_provenance": source_provenance,
        "optimization_history": history,
        "comparison": comparison,
    }
    write_json(model_dir / "training_result.json", result)
    return result


def evaluate_retrospective(index_path: Path, checkpoint: Path, output: Path) -> dict:
    data = load_split(index_path, "retrospective_evaluation")
    if set(data["matches"].tolist()) != set(EVAL_MATCHES):
        raise ValueError("Retrospective evaluation does not contain the declared held-out matches")
    config_record = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    training_record = json.loads(
        (checkpoint.parent / "training_result.json").read_text(encoding="utf-8")
    )
    if (config_record.get("variant") != "S4" or config_record.get("fov_index") != 1
            or config_record.get("fitting_matches") != FIT_MATCHES
            or config_record.get("development_matches") != DEV_MATCHES):
        raise ValueError("Retrospective checkpoint split/architecture provenance mismatch")
    if (training_record.get("status") != "COMPLETE"
            or training_record.get("best_checkpoint_sha256") != file_digest(checkpoint)
            or training_record.get("training_index_sha256") != file_digest(index_path)
            or training_record.get("class_mapping") != list(ACTION_LABELS)
            or training_record.get("retrospective_evaluation_matches") != EVAL_MATCHES):
        raise ValueError("Retrospective checkpoint is not a validated, provenance-matched training result")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robust = PartialObservationStudent(width=96, layers=2, heads=4, dropout=0.1, belief=True).to(device)
    robust.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    baseline_record = torch.load(DEFAULT_BASELINE, map_location=device, weights_only=False)
    baseline_state = baseline_record["model"] if "model" in baseline_record else baseline_record
    baseline = PartialObservationStudent(width=96, layers=2, heads=4, dropout=0.1, belief=True).to(device)
    baseline.load_state_dict(baseline_state, strict=True)

    robust_prediction = _predict(robust, data, device).argmax(axis=-1)
    baseline_prediction = _predict(baseline, data, device).argmax(axis=-1)
    by_match = {}
    for match in sorted(EVAL_MATCHES):
        selected = data["matches"] == match
        if not np.any(selected):
            raise ValueError(f"No retrospective examples for held-out match {match}")
        by_match[match] = {
            "samples": int(selected.sum()),
            "baseline": _metrics(data["y"][selected], baseline_prediction[selected]),
            "robust": _metrics(data["y"][selected], robust_prediction[selected]),
        }
    predictions = [
        {
            "sample_id": str(sample_id),
            "match": str(data["matches"][index]),
            "target": ACTION_LABELS[int(data["y"][index])],
            "baseline_prediction": ACTION_LABELS[int(baseline_prediction[index])],
            "robust_prediction": ACTION_LABELS[int(robust_prediction[index])],
        }
        for index, sample_id in enumerate(data["sample_ids"])
    ]
    prediction_path = output / "heldout_action_predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    comparison = {
        "common_sample_ids": len(data["y"]),
        "common_sample_ids_sha256": hashlib.sha256(
            "\n".join(data["sample_ids"].astype(str).tolist()).encode()
        ).hexdigest(),
        "coverage": {"baseline": 1.0, "robust": 1.0},
        "baseline": _metrics(data["y"], baseline_prediction),
        "robust": _metrics(data["y"], robust_prediction),
        "by_match": by_match,
        "protocol": "Frozen checkpoints on the same indexed retrospective IDs, labels, taxonomy and input arrays.",
        "retrospective_targets_used_for_training_or_selection": False,
        "scores_semantics": "Uncalibrated classifier scores are not confidence, tactical value, or a best-action ranking.",
    }
    report = {
        "schema_version": "football.backbone-v0.6-heldout-action-evaluation.1",
        "status": "COMPLETE",
        "training_index_sha256": file_digest(index_path),
        "robust_checkpoint_sha256": file_digest(checkpoint),
        "baseline_checkpoint_sha256": file_digest(DEFAULT_BASELINE),
        "retrospective_matches": sorted(EVAL_MATCHES),
        "evaluation_examples": len(data["y"]),
        "class_mapping": list(ACTION_LABELS),
        "comparison": comparison,
        "prediction_artifacts": {
            "path": str(prediction_path.resolve()),
            "sha256": file_digest(prediction_path),
            "rows": len(predictions),
        },
        "training_performed": False,
    }
    write_json(output / "heldout_action_evaluation.json", report)
    return report


def validate_retrospective(index_path: Path, checkpoint: Path, output: Path) -> dict:
    report_path = output / "heldout_action_evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prediction_path = Path(report["prediction_artifacts"]["path"])
    data = load_split(index_path, "retrospective_evaluation")
    rows = _read_jsonl(prediction_path)
    if report.get("status") != "COMPLETE" or report.get("training_performed") is not False:
        raise ValueError("Held-out report is incomplete or claims training was performed")
    if (report.get("training_index_sha256") != file_digest(index_path)
            or report.get("robust_checkpoint_sha256") != file_digest(checkpoint)
            or report.get("baseline_checkpoint_sha256") != file_digest(DEFAULT_BASELINE)
            or report.get("evaluation_examples") != len(data["y"])
            or report.get("class_mapping") != list(ACTION_LABELS)
            or report.get("retrospective_matches") != sorted(EVAL_MATCHES)):
        raise ValueError("Held-out report provenance does not match current indexed inputs")
    if (report["prediction_artifacts"].get("sha256") != file_digest(prediction_path)
            or report["prediction_artifacts"].get("rows") != len(rows)
            or len(rows) != len(data["y"])):
        raise ValueError("Held-out prediction artifact count or hash mismatch")
    expected_ids = data["sample_ids"].astype(str).tolist()
    if [row.get("sample_id") for row in rows] != expected_ids:
        raise ValueError("Held-out prediction rows are not aligned to indexed sample IDs")
    label_to_index = {label: index for index, label in enumerate(ACTION_LABELS)}
    target = np.empty(len(rows), dtype=np.int64)
    baseline_prediction = np.empty(len(rows), dtype=np.int64)
    robust_prediction = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        if row.get("match") != str(data["matches"][index]):
            raise ValueError(f"Held-out prediction match mismatch at row {index}")
        try:
            target[index] = label_to_index[row["target"]]
            baseline_prediction[index] = label_to_index[row["baseline_prediction"]]
            robust_prediction[index] = label_to_index[row["robust_prediction"]]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Unknown/missing action class in held-out prediction row {index}") from error
    if not np.array_equal(target, data["y"]):
        raise ValueError("Held-out prediction artifact labels differ from the indexed evaluation targets")
    by_match = {}
    for match in sorted(EVAL_MATCHES):
        selected = data["matches"] == match
        by_match[match] = {
            "samples": int(selected.sum()),
            "baseline": _metrics(target[selected], baseline_prediction[selected]),
            "robust": _metrics(target[selected], robust_prediction[selected]),
        }
    expected_comparison = {
        "common_sample_ids": len(target),
        "common_sample_ids_sha256": hashlib.sha256("\n".join(expected_ids).encode()).hexdigest(),
        "coverage": {"baseline": 1.0, "robust": 1.0},
        "baseline": _metrics(target, baseline_prediction),
        "robust": _metrics(target, robust_prediction),
        "by_match": by_match,
        "protocol": "Frozen checkpoints on the same indexed retrospective IDs, labels, taxonomy and input arrays.",
        "retrospective_targets_used_for_training_or_selection": False,
        "scores_semantics": "Uncalibrated classifier scores are not confidence, tactical value, or a best-action ranking.",
    }
    if report.get("comparison") != expected_comparison:
        raise ValueError("Held-out comparison metrics do not reproduce from saved predictions")
    validation = {
        "schema_version": "football.backbone-v0.6-heldout-action-validation.1",
        "status": "PASS",
        "evaluation_examples": len(rows),
        "matches": sorted(EVAL_MATCHES),
        "prediction_rows_aligned": True,
        "metrics_recomputed_from_saved_predictions": True,
        "evaluation_data_used_for_training_or_selection": False,
    }
    write_json(output / "heldout_evaluation_validation.json", validation)
    return validation


def validate_artifacts(index_path: Path, output: Path) -> dict:
    data_validation = validate_robust_data(index_path, output)
    expected_by_split = {
        "fitting": load_split(index_path, "fitting"),
        "development": load_split(index_path, "development"),
    }
    manifests = {}
    robust_data = {}
    for split, expected in expected_by_split.items():
        manifest_path = output / "robust_data" / f"{split}_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data = _robust_data(output / "robust_data" / f"{split}.npz", expected)
        robust_data[split] = data
        if manifest["data_sha256"] != file_digest(output / "robust_data" / f"{split}.npz"):
            raise ValueError(f"Robust input hash mismatch: {split}")
        logs = _read_jsonl(output / "robust_data" / f"{split}_corruptions.jsonl")
        if len(logs) != len(expected["y"]) or [row["sample_id"] for row in logs] != expected["sample_ids"].astype(str).tolist():
            raise ValueError(f"Corruption log/sample alignment mismatch: {split}")
        manifests[split] = manifest
        if not len(data["y"]):
            raise ValueError(f"Empty robust dataset: {split}")
    model_dir = output / "robust_training"
    checkpoint = model_dir / "best.pt"
    model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    run_config = json.loads((model_dir / "run_config.json").read_text(encoding="utf-8"))
    if model_config["fitting_matches"] != FIT_MATCHES or model_config["development_matches"] != DEV_MATCHES:
        raise ValueError("Inference checkpoint split provenance mismatch")
    expected_sources = {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for shard in index["shards"]:
        if shard["split"] in ("fitting", "development"):
            expected_sources[f"{shard['match']}_{shard['half']}"] = (
                str(shard["match"]), shard["half"],
            )
    if set(run_config["source_provenance"]) != set(expected_sources):
        raise ValueError("Training source provenance does not cover exactly the fitting/development shards")
    for source_key, (match, half) in expected_sources.items():
        current = SoccerTrackSource(ROOT, match, half).provenance
        recorded = run_config["source_provenance"][source_key]
        if any(
            recorded.get(field) != current.get(field)
            for field in ("cache_sha256", "association_sha256", "source_id")
        ):
            raise ValueError(f"Training source provenance changed for {source_key}")
    result = json.loads((model_dir / "training_result.json").read_text(encoding="utf-8"))
    if result["status"] != "COMPLETE" or result["best_checkpoint_sha256"] != file_digest(checkpoint):
        raise ValueError("Robust training result/checkpoint status mismatch")
    if result["training_examples"] != len(expected_by_split["fitting"]["y"]):
        raise ValueError("Robust training example count differs from indexed fitting data")
    resume_path = model_dir / "last.pt"
    resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
    expected_fingerprint = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    required_resume_keys = {
        "fingerprint", "model", "optimizer", "next_epoch", "best_macro_f1",
        "best_epoch", "stale", "history", "rng",
    }
    if (not required_resume_keys.issubset(resume_state)
            or resume_state["fingerprint"] != expected_fingerprint
            or resume_state["next_epoch"] != result["epochs_completed"]
            or len(resume_state["history"]) != result["epochs_completed"]
            or not resume_state["optimizer"].get("state")
            or not {"python", "numpy", "torch", "cuda"}.issubset(resume_state["rng"])
            or not isinstance(resume_state["rng"]["torch"], torch.Tensor)):
        raise ValueError("Safe-boundary optimizer/RNG resume checkpoint is incomplete or mismatched")
    comparison = result["comparison"]
    if (comparison["common_sample_ids"] != len(expected_by_split["development"]["y"])
            or any(value != 1.0 for value in comparison["coverage"].values())
            or comparison["retrospective_evaluation_used"] is not False):
        raise ValueError("Development comparison is not fully covered on common IDs or touched evaluation data")
    session = ActionModelSession(checkpoint)
    inputs = robust_data["development"]
    raw_x = inputs["x"][0].astype(np.float32).copy()
    raw_x[..., 6] = np.clip(raw_x[..., 6] / AGE_SCALE, 0.0, 0.25001)
    world_input = WorldInput(
        raw_x, inputs["mask"][0], np.zeros(2, np.float32),
        np.zeros((22, 2), np.float32), {},
    )
    inference = session.predict(world_input)
    if inference.status != "supported" or session.load_count != 1 or session.forward_calls != 1:
        raise ValueError("Action checkpoint failed normal ActionModelSession inference")
    if set(inference.scores or {}) != set(ACTION_LABELS):
        raise ValueError("Action adapter output class mapping mismatch")
    model_x = raw_x.copy()
    model_x[..., 6] = np.clip(model_x[..., 6] * AGE_SCALE, 0.0, 1.0)
    with torch.inference_mode():
        direct_logits = session.model(
            torch.from_numpy(model_x[None]),
            torch.from_numpy(inputs["mask"][0][None]),
            torch.from_numpy(inputs["visible"][0][None]),
            memory_mode="memory",
        )["action"][0].float()
        direct_scores = direct_logits.softmax(-1).cpu().numpy()
    adapter_scores = np.asarray([inference.scores[label] for label in ACTION_LABELS])
    parity_difference = float(np.max(np.abs(adapter_scores - direct_scores)))
    if parity_difference > 1e-7:
        raise ValueError(f"Action adapter/direct model parity failed: {parity_difference}")
    report = {
        "schema_version": "football.action-robustness-validation.0.1",
        "status": "PASS",
        "data_validation": data_validation,
        "split_manifests": manifests,
        "fitting_examples": len(expected_by_split["fitting"]["y"]),
        "development_examples": len(expected_by_split["development"]["y"]),
        "retrospective_evaluation_examples_used": 0,
        "development_coverage": comparison["coverage"],
        "normal_action_adapter": {
            "status": inference.status,
            "load_count": session.load_count,
            "forward_calls": session.forward_calls,
            "classes": list(inference.scores or {}),
            "action_type": inference.output.action_type,
            "direct_forward_max_abs_score_difference": parity_difference,
        },
        "checkpoint_sha256": file_digest(checkpoint),
        "resume_checkpoint_sha256": file_digest(resume_path),
        "resume_checkpoint": {
            "status": "PASS",
            "completed_epochs": resume_state["next_epoch"],
            "optimizer_state_present": True,
            "rng_states_present": True,
        },
        "source_provenance_shards": len(expected_sources),
        "training_result": str((model_dir / "training_result.json").resolve()),
    }
    write_json(output / "robustness_validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=(
            "build", "validate-data", "train", "validate", "evaluate", "validate-evaluation",
        ),
        required=True,
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    index_path = args.index if args.index.is_absolute() else ROOT / args.index
    if args.phase == "build":
        results = {
            split: build_corrupted_split(index_path, split, output)
            for split in ("fitting", "development")
        }
        print(json.dumps(results, indent=2))
    elif args.phase == "validate-data":
        print(json.dumps(validate_robust_data(index_path, output), indent=2))
    elif args.phase == "train":
        print(json.dumps(train_robust(index_path, output, resume=args.resume), indent=2))
    elif args.phase == "evaluate":
        checkpoint = args.checkpoint or (output.parent / "stage_c" / "robust_training" / "best.pt")
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        output.mkdir(parents=True, exist_ok=True)
        print(json.dumps(evaluate_retrospective(index_path, checkpoint, output), indent=2))
    elif args.phase == "validate-evaluation":
        checkpoint = args.checkpoint or (output.parent / "stage_c" / "robust_training" / "best.pt")
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        print(json.dumps(validate_retrospective(index_path, checkpoint, output), indent=2))
    else:
        print(json.dumps(validate_artifacts(index_path, output), indent=2))


if __name__ == "__main__":
    main()
