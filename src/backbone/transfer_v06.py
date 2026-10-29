"""Match-excluded action transfer using the indexed causal corpus and compact S4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.backbone.actions import ActionModelSession
from src.backbone.corpus import _write_npz
from src.backbone.source import SoccerTrackSource
from src.backbone.train_action import (
    DEV_MATCHES, FIT_MATCHES, _forward, _metrics, _read_jsonl, _restore_rng,
    _rng_state, _save_torch, _seed_all, load_split,
)
from src.backbone.train_action_robust_v06 import DEFAULT_BASELINE, _robust_data
from src.models.partial_observation_student import PartialObservationStudent
from src.player_centric_dataset import ACTION_LABELS, _entity_features, _frame_tensor
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "outputs/backbone_v0_5_20261020/corpus/training_index.json"
ROBUST_DATA = ROOT / "outputs/backbone_v0_6_20261021/stage_c/robust_data"
FOLDS = (
    ("117092", "117093"),
    ("118576", "118578"),
    ("128057", "132831"),
)
MODEL = {"width": 96, "layers": 2, "heads": 4, "dropout": 0.1}
RECIPE = {
    "seed": 42, "batch_size": 256, "learning_rate": 0.0001,
    "weight_decay": 0.0001, "teacher_max_epochs": 8,
    "student_max_epochs": 6, "patience": 2, "minimum_improvement": 0.0005,
    "temperature": 2.0, "distillation_weight": 0.5,
    "mixed_precision": True, "gradient_clip": 1.0,
}


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _history_features(source: SoccerTrackSource, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    anchor = int(metadata["query_frame"])
    actor = int(metadata["actor_track"].removeprefix("track_"))
    frames = list(map(int, metadata["input_frame_indices"]))
    if frames != list(range(anchor - 50, anchor + 1, 5)) or frames[0] < 0:
        raise ValueError(f"Unsupported teacher history: {metadata['sample_id']}")
    _, slots, xy, observed, teams, roles, local_actor = _frame_tensor(
        source.raw, anchor, actor, source.half, 2, 5,
    )
    if np.any(~np.isin(teams, [1, 2])) or np.any(~np.isin(roles, [0, 1])):
        raise ValueError(f"Unknown teacher metadata: {metadata['sample_id']}")
    if not observed[:, local_actor].all():
        raise ValueError(f"Teacher actor absent inside history: {metadata['sample_id']}")
    key = hashlib.sha256(metadata["sample_id"].encode()).digest()
    anonymous = np.random.default_rng(int.from_bytes(key[:8], "little")).permutation(len(slots))
    x, mask, _, _, _ = _entity_features(
        xy, observed, teams, roles, local_actor, observed, anonymous,
    )
    if not np.isfinite(x).all() or not mask[:, anonymous[local_actor]].all():
        raise ValueError(f"Invalid teacher features: {metadata['sample_id']}")
    return x, mask


def build_teacher_inputs(index_path: Path, split: str, data: dict, output: Path) -> dict:
    """Reconstruct privileged histories by keyed metadata, never historical tensor globs."""
    path = output / f"{split}_exo.npz"
    manifest_path = path.with_suffix(".json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["index_sha256"] != file_digest(index_path)
                or manifest["inputs_sha256"] != file_digest(path)):
            raise ValueError("Teacher input resume provenance changed")
        with np.load(path, allow_pickle=False) as saved:
            if saved["sample_ids"].tolist() != data["sample_ids"].tolist():
                raise ValueError("Teacher resume IDs changed")
            return {**data, "x": saved["x"].astype(np.float32),
                    "mask": saved["mask"], "visible": saved["mask"]}
    rows = {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    sources = {}
    for shard in index["shards"]:
        if shard["split"] != split:
            continue
        match, half = str(shard["match"]), shard["half"]
        source = SoccerTrackSource(ROOT, match, half)
        sources[(match, half)] = source
        for row in _read_jsonl(Path(shard["folder"]) / "student_metadata.jsonl"):
            if (row["split"] != split or row["match"] != match or row["half"] != half
                    or row["sample_id"] in rows):
                raise ValueError("Teacher metadata provenance/uniqueness mismatch")
            rows[row["sample_id"]] = row
    if set(rows) != set(data["sample_ids"].tolist()):
        raise ValueError("Teacher metadata IDs do not match indexed labels")
    x = np.empty((len(data["y"]), 11, 22, 10), np.float32)
    mask = np.empty(x.shape[:-1], bool)
    for i, sample_id in enumerate(data["sample_ids"].tolist()):
        row = rows[sample_id]
        x[i], mask[i] = _history_features(sources[(row["match"], row["half"])], row)
    _write_npz(path, x=x.astype(np.float16), mask=mask, sample_ids=data["sample_ids"])
    manifest = {
        "split": split, "samples": len(x), "index_sha256": file_digest(index_path),
        "inputs_sha256": file_digest(path), "class_mapping": list(ACTION_LABELS),
        "features": "Existing _frame_tensor and _entity_features; privileged observed history; no receiver/action labels as features.",
        "normalization": "Fixed pitch dimensions 52.5/34m, velocity 12m/s, age 5s; no fitted population statistics.",
        "timing": "11 samples at 5 native-frame stride ending at causal indexed query.",
        "source_provenance": {f"{m}_{h}": s.provenance for (m, h), s in sources.items()},
    }
    write_json(manifest_path, manifest)
    return {**data, "x": x.astype(np.float16).astype(np.float32), "mask": mask, "visible": mask}


def _subset(data: dict, selected: np.ndarray) -> dict:
    return {key: value[selected] for key, value in data.items()
            if isinstance(value, np.ndarray)}


def validate_probabilities(probabilities: np.ndarray, count: int) -> None:
    if (probabilities.shape != (count, len(ACTION_LABELS))
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(-1), 1, atol=1e-5)):
        raise ValueError("Teacher probabilities are invalid or misaligned")


def validate_fold(fitting: list[str], predicted: list[str]) -> None:
    if (set(fitting) & set(predicted)
            or set(fitting) | set(predicted) != FIT_MATCHES
            or tuple(sorted(predicted)) not in FOLDS):
        raise ValueError("Teacher fold leaks prediction matches or has wrong membership")


def _stop(output: Path, deadline: float) -> None:
    if (output / "STOP_REQUESTED").exists() or time.time() >= deadline:
        raise InterruptedError("Safe checkpoint retained; continuation stop/deadline requested")


def fit(
    fitting: dict, development: dict, directory: Path, specification: dict,
    *, teacher: bool, root_output: Path, deadline: float,
    soft_targets: np.ndarray | None = None,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(specification)
    config_path = directory / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != specification:
        raise ValueError(f"Fit specification changed: {directory}")
    write_json(config_path, specification)
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["best_sha256"] != file_digest(directory / "best.pt"):
            raise ValueError("Completed fit checkpoint changed")
        return result
    if soft_targets is not None:
        validate_probabilities(soft_targets, len(fitting["y"]))
    _seed_all(RECIPE["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartialObservationStudent(**MODEL, belief=not teacher).to(device)
    if not teacher:
        initial = torch.load(DEFAULT_BASELINE, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"] if "model" in initial else initial, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"],
    )
    state = {"epoch": 0, "cursor": 0, "steps": 0, "best": -1.0, "best_epoch": 0,
             "stale": 0, "history": [], "loss_sum": 0.0}
    last = directory / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Fit resume fingerprint changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        _restore_rng(saved["rng"])

    def save() -> None:
        _save_torch(last, {"fingerprint": fingerprint, "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(), "state": state, "rng": _rng_state()})
        write_json(directory / "progress.json", {
            "epoch": state["epoch"], "cursor": state["cursor"], "steps": state["steps"],
            "best_epoch": state["best_epoch"], "safe_checkpoint": str(last.resolve()),
        })

    maximum = RECIPE["teacher_max_epochs" if teacher else "student_max_epochs"]
    while state["epoch"] < maximum and state["stale"] < RECIPE["patience"]:
        order = np.random.default_rng(RECIPE["seed"] + state["epoch"]).permutation(len(fitting["y"]))
        model.train()
        for start in range(state["cursor"], len(order), RECIPE["batch_size"]):
            if (root_output / "STOP_REQUESTED").exists() or time.time() >= deadline:
                save()
                _stop(root_output, deadline)
            chosen = order[start:start + RECIPE["batch_size"]]
            x = torch.from_numpy(fitting["x"][chosen]).to(device)
            mask = torch.from_numpy(fitting["mask"][chosen]).to(device)
            visible = torch.from_numpy(fitting["visible"][chosen]).to(device)
            y = torch.from_numpy(fitting["y"][chosen]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x, mask, visible, memory_mode="memory")["action"].float()
                loss = F.cross_entropy(logits, y)
                if soft_targets is not None:
                    t = float(RECIPE["temperature"])
                    target = torch.from_numpy(soft_targets[chosen]).to(device).clamp_min(1e-8)
                    target = (target.log() / t).softmax(-1)
                    loss = loss + RECIPE["distillation_weight"] * t * t * F.kl_div(
                        (logits / t).log_softmax(-1), target, reduction="batchmean",
                    )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite training loss: {directory}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["gradient_clip"],
                                          error_if_nonfinite=True)
            optimizer.step()
            state["cursor"] = start + len(chosen)
            state["steps"] += 1
            state["loss_sum"] += float(loss.detach()) * len(chosen)
            if state["steps"] % 16 == 0:
                save()
        logits, targets = _forward(model, development, device, RECIPE["batch_size"])
        metric = _metrics(targets, logits.argmax(-1))
        if metric["macro_f1"] > state["best"] + RECIPE["minimum_improvement"]:
            state["best"], state["best_epoch"], state["stale"] = metric["macro_f1"], state["epoch"] + 1, 0
            _save_torch(directory / "best.pt", model.state_dict())
        else:
            state["stale"] += 1
        state["history"].append({
            "epoch": state["epoch"] + 1, "loss": state["loss_sum"] / len(order),
            "development": metric,
        })
        state["epoch"], state["cursor"], state["loss_sum"] = state["epoch"] + 1, 0, 0.0
        save()
        print(json.dumps({"fit": directory.name, "epoch": state["epoch"],
                          "steps": state["steps"], "macro_f1": metric["macro_f1"]}), flush=True)
    result = {
        "status": "COMPLETE", "fitting_examples": len(fitting["y"]),
        "steps": state["steps"], "epochs": state["epoch"], "best_epoch": state["best_epoch"],
        "best_sha256": file_digest(directory / "best.pt"),
        "resume_sha256": file_digest(last), "specification": specification,
        "history": state["history"],
    }
    if not teacher:
        write_json(directory / "config.json", {
            "variant": "S4", "fov_index": 1, "fitting_matches": sorted(FIT_MATCHES),
            "development_matches": sorted(DEV_MATCHES), "config": {**MODEL, "seed": 42},
        })
    write_json(result_path, result)
    return result


def _load_model(checkpoint: Path, teacher: bool) -> PartialObservationStudent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartialObservationStudent(**MODEL, belief=not teacher).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    return model.eval()


def run(index: Path, output: Path, deadline: float) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    specification = {
        "index_sha256": file_digest(index), "recipe": RECIPE, "model": MODEL,
        "folds": [list(fold) for fold in FOLDS],
        "initialization_sha256": file_digest(DEFAULT_BASELINE),
        "normalization": "Fixed physical constants; no cross-match fitted normalization.",
        "development_reuse": "132877 used repeatedly for bounded stopping; not independent evaluation.",
        "student_input_policy": "C1/C2 use identical pre-memory corruption inputs and seed-42 order.",
        "retrospective_used_for_training_or_selection": False,
        "corrupted_inputs": {split: file_digest(ROBUST_DATA / f"{split}.npz")
                             for split in ("fitting", "development")},
    }
    plan_path = output / "specification.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != specification:
        raise ValueError("Transfer program specification changed on resume")
    write_json(plan_path, specification)
    fit_clean, dev_clean = load_split(index, "fitting"), load_split(index, "development")
    fit_exo = build_teacher_inputs(index, "fitting", fit_clean, output)
    dev_exo = build_teacher_inputs(index, "development", dev_clean, output)
    oof = np.zeros((len(fit_clean["y"]), 7), np.float32)
    assignments = np.full(len(oof), -1, np.int16)
    lineage = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold_number, heldout in enumerate(FOLDS):
        fitting_matches = sorted(FIT_MATCHES - set(heldout))
        validate_fold(fitting_matches, list(heldout))
        selected = np.isin(fit_exo["matches"], heldout)
        folder = output / f"teacher_fold_{fold_number}"
        fold_spec = {**specification, "role": "exo_teacher", "initialization": "scratch",
                     "fitting_matches": fitting_matches, "prediction_matches": list(heldout)}
        fit(_subset(fit_exo, ~selected), dev_exo, folder, fold_spec,
            teacher=True, root_output=output, deadline=deadline)
        _stop(output, deadline)
        model = _load_model(folder / "best.pt", True)
        logits, _ = _forward(model, _subset(fit_exo, selected), device, 256)
        probabilities = torch.from_numpy(logits).softmax(-1).numpy()
        validate_probabilities(probabilities, int(selected.sum()))
        oof[selected] = probabilities
        assignments[selected] = fold_number
        lineage.append({**fold_spec, "checkpoint": str((folder / "best.pt").resolve()),
                        "checkpoint_sha256": file_digest(folder / "best.pt"),
                        "prediction_examples": int(selected.sum())})
        del model
    validate_probabilities(oof, len(fit_clean["y"]))
    _write_npz(output / "oof_targets.npz", sample_ids=fit_clean["sample_ids"],
               probabilities=oof, fold=assignments, matches=fit_clean["matches"])
    write_json(output / "oof_lineage.json", {
        "class_mapping": list(ACTION_LABELS), "folds": lineage,
        "target_sha256": file_digest(output / "oof_targets.npz"),
    })
    reference_spec = {**specification, "role": "exo_reference", "initialization": "scratch",
                      "fitting_matches": sorted(FIT_MATCHES), "prediction_matches": sorted(DEV_MATCHES)}
    fit(fit_exo, dev_exo, output / "teacher_reference", reference_spec,
        teacher=True, root_output=output, deadline=deadline)
    reference = _load_model(output / "teacher_reference/best.pt", True)
    dev_logits, _ = _forward(reference, dev_exo, device, 256)
    _write_npz(output / "development_teacher_reference.npz",
               sample_ids=dev_clean["sample_ids"],
               probabilities=torch.from_numpy(dev_logits).softmax(-1).numpy())
    del reference, fit_exo, dev_exo
    student_fit = _robust_data(ROBUST_DATA / "fitting.npz", fit_clean)
    student_dev = _robust_data(ROBUST_DATA / "development.npz", dev_clean)
    cells = {}
    for name, targets in (("C1", None), ("C2", oof)):
        fit_spec = {**specification, "role": name,
                    "teacher_targets_sha256": file_digest(output / "oof_targets.npz") if name == "C2" else None}
        fit(student_fit, student_dev, output / name, fit_spec,
            teacher=False, root_output=output, deadline=deadline, soft_targets=targets)
        model = _load_model(output / name / "best.pt", False)
        for condition, data in (("clean", dev_clean), ("corrupted", student_dev)):
            logits, _ = _forward(model, data, device, 256, mixed_precision=False)
            cells[f"{name}_{condition}"] = _metrics(data["y"], logits.argmax(-1))
            _write_npz(output / f"{name}_{condition}_development.npz",
                       sample_ids=data["sample_ids"], logits=logits)
        del model
    baseline_record = torch.load(DEFAULT_BASELINE, map_location="cpu", weights_only=False)
    baseline = PartialObservationStudent(**MODEL, belief=True).to(device)
    baseline.load_state_dict(baseline_record["model"], strict=True)
    for condition, data in (("clean", dev_clean), ("corrupted", student_dev)):
        logits, _ = _forward(baseline, data, device, 256, mixed_precision=False)
        cells[f"original_{condition}"] = _metrics(data["y"], logits.argmax(-1))
        _write_npz(output / f"original_{condition}_development.npz",
                   sample_ids=data["sample_ids"], logits=logits)
    report = {"status": "COMPLETE", "cells": cells,
              "fitting_examples": len(fit_clean["y"]), "development_examples": len(dev_clean["y"]),
              "default_model": "Original v0.5 S4 retained; no automatic deployment change.",
              "comparison": "Matched seed-42 C1/C2; same initialization/order/corruption and recipe except KL loss.",
              "reference_teacher": "Six-fit teacher reference evaluated on development only in this phase.",
              "retrospective_evaluation": "NOT_RUN; untouched for training and selection."}
    write_json(output / "comparison.json", report)
    return validate(index, output)


def validate(index: Path, output: Path) -> dict:
    fit_data, dev_data = load_split(index, "fitting"), load_split(index, "development")
    lineage = json.loads((output / "oof_lineage.json").read_text(encoding="utf-8"))
    if lineage["target_sha256"] != file_digest(output / "oof_targets.npz"):
        raise ValueError("OOF target hash changed")
    with np.load(output / "oof_targets.npz", allow_pickle=False) as saved:
        if (saved["sample_ids"].tolist() != fit_data["sample_ids"].tolist()
                or saved["matches"].tolist() != fit_data["matches"].tolist()):
            raise ValueError("OOF keyed target alignment mismatch")
        validate_probabilities(saved["probabilities"], len(fit_data["y"]))
        for number, fold in enumerate(lineage["folds"]):
            validate_fold(fold["fitting_matches"], fold["prediction_matches"])
            selected = np.isin(fit_data["matches"], fold["prediction_matches"])
            if not np.all(saved["fold"][selected] == number):
                raise ValueError("OOF assignment mismatch")
            if fold["checkpoint_sha256"] != file_digest(Path(fold["checkpoint"])):
                raise ValueError("OOF checkpoint hash changed")
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    for name in ("original", "C1", "C2"):
        for condition in ("clean", "corrupted"):
            with np.load(output / f"{name}_{condition}_development.npz", allow_pickle=False) as saved:
                if saved["sample_ids"].tolist() != dev_data["sample_ids"].tolist():
                    raise ValueError("Matched comparison IDs changed")
                if _metrics(dev_data["y"], saved["logits"].argmax(-1)) != report["cells"][f"{name}_{condition}"]:
                    raise ValueError("Matched metrics do not recompute")
    parity = {}
    from src.backbone.contracts import WorldInput
    for name in ("C1", "C2"):
        session = ActionModelSession(output / name / "best.pt")
        x = dev_data["x"][0].copy()
        x[..., 6] *= 5 / 8
        inputs = WorldInput(x, dev_data["mask"][0], np.zeros(2, np.float32),
                            np.zeros((22, 2), np.float32), {})
        prediction = session.predict(inputs)
        if prediction.status != "supported":
            raise ValueError("Matched student does not load in normal action adapter")
        model_x = x.copy()
        model_x[..., 6] = np.clip(model_x[..., 6] * 8 / 5, 0, 1)
        with torch.inference_mode():
            direct = session.model(torch.from_numpy(model_x[None]),
                                   torch.from_numpy(dev_data["mask"][:1]),
                                   torch.from_numpy(dev_data["visible"][:1]))["action"].softmax(-1)[0].numpy()
        difference = float(np.max(np.abs(direct - [prediction.scores[label] for label in ACTION_LABELS])))
        if difference > 1e-6:
            raise ValueError("Matched student adapter/direct parity mismatch")
        parity[name] = difference
    result = {"status": "PASS", "oof_examples": len(fit_data["y"]), "folds": 3,
              "match_exclusion": "PASS", "keyed_join": "PASS", "metrics_recomputed": True,
              "adapter_parity": parity, "default_model": "Original v0.5 S4 unchanged",
              "source_code_sha256": file_digest(Path(__file__))}
    write_json(output / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "validate"), required=True)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args()
    from datetime import datetime
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")).timestamp()
    result = run(args.index, args.output, deadline) if args.phase == "run" else validate(args.index, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
