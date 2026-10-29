"""Resumable, split-safe event-corpus export from verified local tracking caches."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import traceback

import numpy as np

from src.backbone.event_time import resolve_event_time
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import SoccerTrackSource, configured_path
from src.player_centric_dataset import (
    ACTION_ALIASES,
    ACTION_LABELS,
    _anonymous_slots,
)
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
VERSION = "football.action-corpus.0.4"
AGE_INPUT_SCALE = 8.0
AGE_TRAINING_SCALE = 5.0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _class_for_label(label: object) -> str | None:
    if not isinstance(label, str):
        return None
    return ACTION_ALIASES.get(label.upper().replace(" ", "_"))


def _split_map(config: dict) -> dict[str, str]:
    result = {}
    for split, matches in config["split_matches"].items():
        for match in matches:
            if match in result:
                raise ValueError(f"Match {match} occurs in more than one configured split")
            result[str(match)] = split
    expected = {
        "fitting": {"117092", "117093", "118576", "118578", "128057", "132831"},
        "development": {"132877"},
        "retrospective_evaluation": {"118577", "128058"},
    }
    actual = {key: set(map(str, value)) for key, value in config["split_matches"].items()}
    if actual != expected:
        raise ValueError(f"v0.4 match split must exactly match the declared split; got {actual}")
    return result


def _source_paths(root: Path, match: str, half: str, configured: dict | None = None) -> tuple[Path, Path, Path]:
    configured = configured or {}
    source_root = configured_path(
        root, configured.get("tracking_cache"),
        Path("outputs") / "trajectory_forecasting_20261016" / "cache",
    )
    association_root = configured_path(
        root, configured.get("associations"), Path("outputs") / "trajectory_audit_20261016",
    )
    return (
        source_root / f"{match}_{half}.npz",
        source_root / f"{match}_{half}.json",
        association_root / f"associations_{match}_{half}.json",
    )


def _feature_input(
    source: SoccerTrackSource,
    actor_track: str,
    anchor: int,
    slot_to_player: dict[int, str],
    models: ModelSession,
    *,
    observation_config: dict | None = None,
    include_passing_assessment: bool = False,
    audit_legacy: bool = False,
) -> tuple[dict | None, dict]:
    raw = source.raw
    model_indices = list(range(anchor - 50, anchor + 1, 5))
    if len(model_indices) != 11 or model_indices[0] < 0 or model_indices[-1] != anchor:
        return None, {"reason": "insufficient_causal_history"}
    try:
        actor_slot = int(actor_track.removeprefix("track_"))
    except ValueError:
        return None, {"reason": "invalid_actor_track"}
    if actor_slot < 0 or actor_slot >= raw["xy"].shape[1] or not raw["observed"][anchor, actor_slot]:
        return None, {"reason": "actor_missing_at_query"}
    segment = int(raw["segments"][anchor, actor_slot])
    if not all(
        raw["observed"][frame_index, actor_slot]
        and int(raw["segments"][frame_index, actor_slot]) == segment
        for frame_index in model_indices
    ):
        return None, {"reason": "actor_missing_or_discontinuous_inside_history"}

    # Five causal warm-up samples give the heading provider the same six
    # positions as a chronological replay at the first model-input sample.
    observation_config = observation_config or {}
    warmup_samples = int(observation_config.get("causal_warmup_samples", 5))
    if warmup_samples < 0:
        raise ValueError("causal_warmup_samples must be nonnegative")
    first_frame = max(0, model_indices[0] - 5 * warmup_samples)
    stream_frames = list(range(first_frame, anchor + 1, 5))
    engine = ReplayEngine(models, samples=16)
    provider = SyntheticObservationProvider(
        actor_track,
        seed=int(observation_config.get("seed", 42)),
        fov=int(observation_config.get("fov_degrees", 120)),
        dropout=bool(observation_config.get("dropout", False)),
    )
    observations = []
    result = None
    for frame_index in stream_frames:
        frame = source.frame(frame_index)
        observation = provider.observe(frame)
        observations.append(observation)
        result = engine.step(
            observation,
            include_passing_assessment=include_passing_assessment and frame_index == anchor,
        )
    inputs = engine.memory.current_input
    if inputs is None or result is None:
        unknown = any(
            measurement.team not in (1, 2) or measurement.goalkeeper is None
            for observation in observations[-11:] for measurement in observation.measurements
        )
        return None, {
            "reason": "unknown_team_or_role_in_causal_observations"
            if unknown else "causal_memory_contract_not_met",
        }
    current_slots = np.flatnonzero(raw["observed"][anchor])
    if len(current_slots) > 22:
        return None, {"reason": "more_than_22_current_players"}
    current_teams = raw["teams"][anchor, current_slots].astype(np.int64)
    current_roles = raw["roles"][anchor, current_slots].astype(np.int64)
    anonymous_key = f"{source.provenance['source_id']}:{source.half}:{actor_track}:{anchor}"
    anonymous_by_source = _anonymous_slots(current_slots.tolist(), slot_to_player, anonymous_key)

    x = inputs.x.copy()
    x[..., 6] = np.clip(x[..., 6] * (AGE_INPUT_SCALE / AGE_TRAINING_SCALE), 0.0, 1.0)
    if (x.shape != (11, 22, 10) or inputs.mask.shape != (11, 22)
            or np.any(~np.isfinite(x))):
        raise ValueError(
            f"Unexpected causal action input for actor={actor_track}, anchor={anchor}, shape={x.shape}"
        )

    exo_xy = np.zeros((22, 2), dtype=np.float32)
    exo_mask = np.zeros(22, dtype=bool)
    exo_team = np.zeros(22, dtype=np.int8)
    exo_goalkeeper = np.zeros(22, dtype=bool)
    for source_slot, anonymous_slot in anonymous_by_source.items():
        exo_xy[int(anonymous_slot)] = raw["xy"][anchor, int(source_slot)]
        exo_mask[int(anonymous_slot)] = True
        local = int(np.flatnonzero(current_slots == source_slot)[0])
        exo_team[int(anonymous_slot)] = int(current_teams[local])
        exo_goalkeeper[int(anonymous_slot)] = bool(current_roles[local]) if current_roles[local] in (0, 1) else False
    return {
        "x": x,
        "mask": inputs.mask.copy(),
        "visible": inputs.x[..., 4] > 0.5,
        "exo_xy": exo_xy,
        "exo_mask": exo_mask,
        "exo_team": exo_team,
        "exo_goalkeeper": exo_goalkeeper,
        "reliable_heading_samples": sum(
            "Motion heading unreliable" not in warning for observation in observations
            for warning in observation.warnings
        ),
        "history_frames": model_indices,
        "warmup_frames": stream_frames[:max(0, len(stream_frames) - len(model_indices))],
        "actor_xy": [float(value) for value in raw["xy"][anchor, actor_slot]],
        "exo_track_slot_order": [
            int(source_slot)
            for source_slot, _ in sorted(anonymous_by_source.items(), key=lambda pair: pair[1])
        ],
        "legacy_audit": {
            "status": "NOT_COMPARABLE_CHANGED_CAUSAL_OBSERVATION",
            "observation_builder": "SyntheticObservationProvider + CausalMemory",
            "visibility_seed_semantics": "source/half/actor-segment/actor/frame; annotation independent",
            "roster_policy": "first-sight memory; no query-time roster truncation",
        } if audit_legacy else None,
        "model_input": inputs,
        "engine": engine,
        "observations": observations[-11:],
        "replay_result": result,
    }, {"reason": None}


def _event_rows(event_file: dict, match: str, half: str) -> list[dict]:
    rows = event_file.get("actions") or event_file.get("annotations")
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported BAS event file for match {match}")
    expected_period = "1" if half == "1st" else "2"
    return [
        {"annotation_index": index, "event": event}
        for index, event in enumerate(rows)
        if str(event.get("gameTime", "")).split("-", 1)[0].strip() == expected_period
    ]


def _shard_fingerprint(
    config_sha: str, code_sha: str, cache_sha: str, metadata_sha: str,
    association_sha: str, bas_sha: str,
) -> str:
    import hashlib

    value = "|".join((config_sha, code_sha, cache_sha, metadata_sha, association_sha, bas_sha))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_cache_shard(folder: Path, fingerprint: str) -> dict | None:
    completion = folder / "completion.json"
    if not completion.is_file():
        return None
    try:
        record = json.loads(completion.read_text(encoding="utf-8"))
        if record.get("status") != "COMPLETE" or record.get("fingerprint") != fingerprint:
            return None
        if not all((folder / name).is_file()
                   and file_digest(folder / name) == digest
                   for name, digest in record["artifacts"].items()):
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _build_half_shard(
    config: dict,
    source_spec: dict,
    output: Path,
    root: Path,
    split: str,
    models: ModelSession,
    resume: bool,
    config_sha: str,
    code_sha: str,
) -> dict:
    match, half = str(source_spec["match"]), str(source_spec["half"])
    source_settings = config.get("source_paths", {})
    cache_path, cache_meta_path, association_path = _source_paths(
        root, match, half, source_settings,
    )
    event_root = configured_path(root, source_settings.get("event_data"), Path("data"))
    bas_path = event_root / f"{match}_12_class_events.json"
    source = SoccerTrackSource(
        root, match, half,
        cache_root=source_settings.get("tracking_cache"),
        association_root=source_settings.get("associations"),
    )
    with cache_path.open("rb"):
        pass
    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    association = json.loads(association_path.read_text(encoding="utf-8"))
    cache_sha, association_sha, bas_sha = (
        file_digest(cache_path), file_digest(association_path), file_digest(bas_path),
    )
    if association.get("cache_sha256") != cache_sha or cache_meta.get("cache_sha256") != cache_sha:
        raise ValueError(f"Cache hash provenance mismatch for {match}_{half}")
    if (int(source_spec["frame_start"]) != 0
            or int(source_spec["frame_end_exclusive"]) != len(source.raw["xy"])):
        raise ValueError(
            f"Configured explicit interval does not match verified source length for {match}_{half}: "
            f"{len(source.raw['xy'])}"
        )
    fingerprint = _shard_fingerprint(
        config_sha, code_sha, cache_sha, file_digest(cache_meta_path), association_sha, bas_sha,
    )
    folder = output / "shards" / match / half
    folder.mkdir(parents=True, exist_ok=True)
    if resume:
        cached = _read_cache_shard(folder, fingerprint)
        if cached is not None:
            return cached
    if (folder / "completion.json").exists():
        (folder / "completion.json").unlink()

    slot_to_player = {int(slot): str(player) for player, slot in association["slots"].items()}
    player_to_slot = {player: slot for slot, player in slot_to_player.items()}
    event_file = json.loads(bas_path.read_text(encoding="utf-8"))
    event_rows = _event_rows(event_file, match, half)
    stats = Counter()
    class_counts = Counter()
    exclusions = Counter()
    student_rows = []
    privileged_rows = []
    target_rows = []
    metadata_rows = []
    parity_records = []
    sample_ids = []
    features = []
    masks = []
    visible_masks = []
    exo_xy = []
    exo_masks = []
    exo_teams = []
    exo_roles = []
    supported_count = 0

    for offset, item in enumerate(event_rows):
        event = item["event"]
        annotation_index = int(item["annotation_index"])
        stats["raw_events"] += 1
        action = _class_for_label(event.get("label"))
        event_id = f"{match}_p{1 if half == '1st' else 2}_f{int(event.get('frame', -1))}_{annotation_index}"
        metadata = {
            "sample_id": None,
            "event_id": event_id,
            "annotation_index": annotation_index,
            "split": split,
            "match": match,
            "half": half,
            "original_bas_label": event.get("label"),
            "mapped_action_label": action,
            "actor_player_id": str(event.get("player_id", "")),
            "stable_id_frame": int(event.get("frame", -1)),
            "position_ms": event.get("position"),
            "query_anchor_frame": None,
            "eligibility": "EXCLUDED",
            "exclusion_reason": None,
            "cache_sha256": cache_sha,
            "association_sha256": association_sha,
            "bas_sha256": bas_sha,
        }
        if action is None:
            metadata["exclusion_reason"] = "unsupported_or_context_label"
            exclusions[metadata["exclusion_reason"]] += 1
            metadata_rows.append(metadata)
            continue
        supported_count += 1
        stats["supported_labeled_events"] += 1
        class_counts[f"raw:{action}"] += 1
        resolution = resolve_event_time(
            event.get("position"),
            event.get("frame"),
            clock=event_file.get("clock"),
            gsr_info=cache_meta.get("gsr_info", {}),
            source_frame_count=len(source.raw["xy"]),
        )
        metadata["time_resolution"] = resolution.to_dict()
        if resolution.status != "supported":
            metadata["exclusion_reason"] = resolution.reason or "unsupported_event_time"
            exclusions[metadata["exclusion_reason"]] += 1
            metadata_rows.append(metadata)
            continue
        position_ms = resolution.position_ms
        stable_id_frame = int(event["frame"])
        query_frame = int(resolution.native_zero_based_frame)
        metadata["position_ms"] = position_ms
        metadata["query_anchor_frame"] = query_frame
        if query_frame < int(source_spec["frame_start"]) or query_frame >= int(source_spec["frame_end_exclusive"]):
            metadata["exclusion_reason"] = "outside_configured_source_interval"
            exclusions[metadata["exclusion_reason"]] += 1
            metadata_rows.append(metadata)
            continue
        actor_player = str(event.get("player_id", ""))
        actor_slot = player_to_slot.get(actor_player)
        if actor_slot is None:
            metadata["exclusion_reason"] = "actor_identity_missing"
            exclusions[metadata["exclusion_reason"]] += 1
            metadata_rows.append(metadata)
            continue
        actor_track = f"track_{actor_slot}"
        sample_id = f"{match}_{half}:{actor_track}:{query_frame}:event:{annotation_index}"
        tensor, failure = _feature_input(
            source, actor_track, query_frame, slot_to_player, models,
            observation_config=config["observation"],
            audit_legacy=not parity_records,
        )
        if tensor is None:
            reason = failure["reason"] or "unknown_input_exclusion"
            metadata["exclusion_reason"] = reason
            exclusions[reason] += 1
            metadata_rows.append(metadata)
            continue
        student_rows.append({
            "sample_id": sample_id,
            "event_id": event_id,
            "match": match,
            "half": half,
            "split": split,
            "actor_track": actor_track,
            "query_frame": query_frame,
            "query_time_s": query_frame / 25.0,
            "input_frame_indices": tensor["history_frames"],
            "source_kind": "annotation_backed_tracking_synthetic_player_observation",
            "target_availability": {
                "action": False, "event": False, "receiver": False,
                "meaning": "LABELS STORED ONLY IN THE JOINED OPTIONAL-TARGET FILE",
            },
        })
        privileged_rows.append({
            "sample_id": sample_id,
            "event_id": event_id,
            "match": match,
            "half": half,
            "query_frame": query_frame,
            "track_slot_order": tensor["exo_track_slot_order"],
            "content": "Evaluation/reference-only full tracked query state; never consumed by action model",
        })
        target_rows.append({
            "sample_id": sample_id,
            "event_id": event_id,
            "annotation_index": annotation_index,
            "target_kind": "event_conditioned_action",
            "original_bas_label": event["label"],
            "mapped_action_label": action,
            "action_class_index": ACTION_LABELS.index(action),
            "mapping_provenance": {
                "mapping_file": "src/recode_primary_actions.py",
                "mapping_sha256": file_digest(root / "src" / "recode_primary_actions.py"),
                "canonical_aliases_file": "src/player_centric_dataset.py",
                "source_label_normalization": "uppercase; spaces converted to underscores",
                "join_method": "stable BAS event ID plus annotation index; frame/period and original label revalidated",
            },
            "actor": {
                "source_player_id": actor_player,
                "source_track": actor_track,
                "association_path": str(association_path.resolve()),
                "association_sha256": association_sha,
            },
            "timing": {
                **resolution.to_dict(),
                "stable_id_frame": stable_id_frame,
                "query_anchor_frame": query_frame,
                "input_frame_indices": tensor["history_frames"],
                "input_timestamps_s": [frame / 25.0 for frame in tensor["history_frames"]],
                "warmup_frames": tensor["warmup_frames"],
                "no_post_event_frames": True,
                "timing_semantics": (
                    "Causal query is floor(position_ms * native_fps / 1000); stable BAS frame is an independent "
                    "rounded-frame/index-origin consistency check. Logical 25 Hz model grid; not certified physical contact."
                ),
            },
            "source": {
                "cache_path": str(cache_path.resolve()),
                "cache_sha256": cache_sha,
                "match": match,
                "half": half,
                "split": split,
            },
        })
        metadata.update(
            sample_id=sample_id,
            actor_track=actor_track,
            eligibility="ELIGIBLE",
            exclusion_reason=None,
            input_frame_indices=tensor["history_frames"],
            reliable_heading_samples=tensor["reliable_heading_samples"],
        )
        if tensor["legacy_audit"] is not None:
            parity_records.append({"event_id": event_id, **tensor["legacy_audit"]})
        features.append(tensor["x"].astype(np.float16))
        masks.append(tensor["mask"])
        visible_masks.append(tensor["visible"])
        exo_xy.append(tensor["exo_xy"])
        exo_masks.append(tensor["exo_mask"])
        exo_teams.append(tensor["exo_team"])
        exo_roles.append(tensor["exo_goalkeeper"])
        sample_ids.append(sample_id)
        class_counts[f"eligible:{action}"] += 1
        stats["eligible_examples"] += 1
        metadata_rows.append(metadata)
        if (offset + 1) % 100 == 0:
            write_json(output / "progress.json", {
                "status": "RUNNING",
                "active_shard": f"{match}_{half}",
                "processed_raw_events": offset + 1,
                "raw_events_in_half": len(event_rows),
                "eligible_examples": len(sample_ids),
                "exclusions": dict(exclusions),
            })

    feature_array = np.stack(features) if features else np.zeros((0, 11, 22, 10), np.float16)
    mask_array = np.stack(masks) if masks else np.zeros((0, 11, 22), bool)
    visible_array = np.stack(visible_masks) if visible_masks else np.zeros((0, 11, 22), bool)
    ids_array = np.asarray(sample_ids, dtype=str)
    _write_npz(
        folder / "student_inputs.npz",
        sample_ids=ids_array,
        x=feature_array,
        entity_mask=mask_array,
        visible=visible_array,
    )
    _write_npz(
        folder / "privileged_exo.npz",
        sample_ids=ids_array,
        xy=np.stack(exo_xy) if exo_xy else np.zeros((0, 22, 2), np.float32),
        mask=np.stack(exo_masks) if exo_masks else np.zeros((0, 22), bool),
        team=np.stack(exo_teams) if exo_teams else np.zeros((0, 22), np.int8),
        goalkeeper=np.stack(exo_roles) if exo_roles else np.zeros((0, 22), bool),
    )
    _write_jsonl(folder / "student_metadata.jsonl", student_rows)
    _write_jsonl(folder / "privileged_metadata.jsonl", privileged_rows)
    _write_jsonl(folder / "optional_targets.jsonl", target_rows)
    _write_jsonl(folder / "event_eligibility.jsonl", metadata_rows)
    write_json(folder / "summary.json", {
        "match": match,
        "half": half,
        "split": split,
        "source_interval": source_spec,
        "frames_available": len(source.raw["xy"]),
        "source_hashes": {
            "cache": cache_sha, "cache_metadata": file_digest(cache_meta_path),
            "association": association_sha, "bas": bas_sha,
        },
        "raw_events": len(event_rows),
        "supported_labeled_events": supported_count,
        "eligible_examples": len(sample_ids),
        "excluded_events_by_reason": dict(sorted(exclusions.items())),
        "raw_supported_class_counts": {
            key.removeprefix("raw:"): value for key, value in sorted(class_counts.items())
            if key.startswith("raw:")
        },
        "eligible_class_counts": {
            key.removeprefix("eligible:"): value for key, value in sorted(class_counts.items())
            if key.startswith("eligible:")
        },
        "feature_builder_parity": parity_records,
        "training_loss_scope": "Action CE targets are in optional_targets.jsonl; exo states are separate and not model inputs.",
    })
    artifacts = {
        name: file_digest(folder / name)
        for name in (
            "student_inputs.npz", "privileged_exo.npz", "student_metadata.jsonl",
            "privileged_metadata.jsonl", "optional_targets.jsonl",
            "event_eligibility.jsonl", "summary.json",
        )
    }
    completion = {
        "schema_version": VERSION,
        "status": "COMPLETE",
        "fingerprint": fingerprint,
        "artifacts": artifacts,
        "sample_count": len(sample_ids),
        "raw_event_count": len(event_rows),
        "excluded_event_counts": dict(sorted(exclusions.items())),
    }
    write_json(folder / "completion.json", completion)
    return completion


def _build_unlabeled_windows(
    config: dict, sources: list[dict], output: Path, root: Path, models: ModelSession,
) -> dict:
    settings = config["unlabeled_windows"]
    if not settings.get("enabled"):
        return {"status": "DISABLED", "windows": 0}
    all_ids, all_x, all_mask, all_visible, metadata = [], [], [], [], []
    excluded = Counter()
    for source_spec in sources:
        match, half = str(source_spec["match"]), str(source_spec["half"])
        source_settings = config.get("source_paths", {})
        source = SoccerTrackSource(
            root, match, half,
            cache_root=source_settings.get("tracking_cache"),
            association_root=source_settings.get("associations"),
        )
        actor_track = str(settings["actor_track"])
        for anchor in range(
            int(settings["first_query_frame"]),
            int(source_spec["frame_end_exclusive"]),
            int(settings["stride_frames"]),
        ):
            sample_id = f"{match}_{half}:{actor_track}:{anchor}:unlabeled"
            tensor, failure = _feature_input(
                source, actor_track, anchor, {}, models,
                observation_config=config["observation"],
                audit_legacy=False,
            )
            if tensor is None:
                excluded[failure["reason"]] += 1
                continue
            all_ids.append(sample_id)
            all_x.append(tensor["x"].astype(np.float16))
            all_mask.append(tensor["mask"])
            all_visible.append(tensor["visible"])
            metadata.append({
                "sample_id": sample_id,
                "match": match,
                "half": half,
                "split": next(
                    name for name, values in config["split_matches"].items()
                    if match in set(map(str, values))
                ),
                "actor_track": actor_track,
                "query_frame": anchor,
                "query_time_s": anchor / 25.0,
                "input_frame_indices": tensor["history_frames"],
                "target_availability": {
                    "action": False, "event": False,
                    "meaning": "UNANNOTATED ARBITRARY WINDOW; NOT A NEGATIVE LABEL OR NO_ACTION",
                },
            })
    folder = output / "unlabeled"
    folder.mkdir(parents=True, exist_ok=True)
    _write_npz(
        folder / "student_windows.npz",
        sample_ids=np.asarray(all_ids, dtype=str),
        x=np.stack(all_x) if all_x else np.zeros((0, 11, 22, 10), np.float16),
        entity_mask=np.stack(all_mask) if all_mask else np.zeros((0, 11, 22), bool),
        visible=np.stack(all_visible) if all_visible else np.zeros((0, 11, 22), bool),
    )
    _write_jsonl(folder / "metadata.jsonl", metadata)
    summary = {
        "status": "COMPLETE",
        "windows": len(all_ids),
        "excluded_windows_by_reason": dict(sorted(excluded.items())),
        "target_policy": settings["label_policy"],
        "source": "Explicit fixed-interval queries from configured source halves; no BAS lookup",
        "artifacts": {
            "student_windows.npz": file_digest(folder / "student_windows.npz"),
            "metadata.jsonl": file_digest(folder / "metadata.jsonl"),
        },
    }
    write_json(folder / "summary.json", summary)
    return summary


def export_corpus(config_path: Path, output: Path, *, resume: bool = False, root: Path = ROOT) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != VERSION:
        raise ValueError(f"Expected corpus schema {VERSION}")
    split_map = _split_map(config)
    match_halves = {(str(row["match"]), str(row["half"])) for row in config["sources"]}
    expected_halves = {(match, half) for match in split_map for half in ("1st", "2nd")}
    if match_halves != expected_halves:
        raise ValueError("Source configuration must include exactly both local halves for every split match")
    output.mkdir(parents=True, exist_ok=True)
    config_sha = file_digest(config_path)
    code_files = (
        Path(__file__),
        Path(__file__).with_name("event_time.py"),
        Path(__file__).with_name("actions.py"),
        Path(__file__).with_name("contracts.py"),
        Path(__file__).with_name("memory.py"),
        Path(__file__).with_name("observation.py"),
        Path(__file__).with_name("pipeline.py"),
        Path(__file__).with_name("source.py"),
        ROOT / "src" / "player_centric_dataset.py",
        ROOT / "src" / "player_centric_visibility.py",
        ROOT / "src" / "belief_observation.py",
        ROOT / "src" / "models" / "partial_observation_student.py",
    )
    code_sha = "|".join(file_digest(root / file.relative_to(ROOT)) for file in code_files)
    model_path = configured_path(
        root, config.get("source_paths", {}).get("belief_experiment"),
        Path("outputs") / "dynamic_gnn_tactical_20261019",
    )
    models = ModelSession(model_path, backend="last_seen")
    shard_records = []
    totals = Counter()
    split_class_counts = defaultdict(Counter)
    for source_spec in config["sources"]:
        match, half = str(source_spec["match"]), str(source_spec["half"])
        split = split_map[match]
        try:
            record = _build_half_shard(
                config, source_spec, output, root, split, models, resume, config_sha, code_sha,
            )
        except Exception as error:
            write_json(output / "failure.json", {
                "status": "FAILED",
                "active_source": f"{match}_{half}",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            })
            raise
        summary = json.loads((output / "shards" / match / half / "summary.json").read_text(encoding="utf-8"))
        record = {
            "match": match,
            "half": half,
            "split": split,
            "folder": str((output / "shards" / match / half).resolve()),
            "fingerprint": record["fingerprint"],
            "samples": record["sample_count"],
            "raw_events": record["raw_event_count"],
            "excluded_events_by_reason": record["excluded_event_counts"],
            "artifact_hashes": record["artifacts"],
        }
        shard_records.append(record)
        totals["raw_events"] += summary["raw_events"]
        totals["supported_labeled_events"] += summary["supported_labeled_events"]
        totals["eligible_examples"] += summary["eligible_examples"]
        for reason, count in summary["excluded_events_by_reason"].items():
            totals[f"excluded:{reason}"] += count
        for label, count in summary["raw_supported_class_counts"].items():
            split_class_counts[split][f"raw:{label}"] += count
        for label, count in summary["eligible_class_counts"].items():
            split_class_counts[split][f"eligible:{label}"] += count
        write_json(output / "progress.json", {
            "status": "RUNNING",
            "completed_shards": len(shard_records),
            "total_shards": len(config["sources"]),
            "active_source": f"{match}_{half}",
            "totals": dict(totals),
        })
    unlabeled_summary = _build_unlabeled_windows(config, config["sources"], output, root, models)
    totals["unlabeled_windows"] = unlabeled_summary["windows"]
    report = {
        "schema_version": VERSION,
        "status": "COMPLETE",
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "split_matches": config["split_matches"],
        "source_intervals": config["sources"],
        "class_mapping": list(ACTION_LABELS),
        "class_mapping_source": "src/recode_primary_actions.py + src/player_centric_dataset.py ACTION_ALIASES",
        "history": "11 observations every 0.2 seconds, ending exactly at floor(BAS position_ms/40)",
        "visibility": config["observation"],
        "age_conversion": "Student NPZ stores S4 training scale min(age,5s)/5; v0.2 memory's min(age,8s)/8 is converted at the boundary.",
        "counts": {
            "raw_events": totals["raw_events"],
            "supported_labeled_events": totals["supported_labeled_events"],
            "eligible_examples": totals["eligible_examples"],
            "excluded_events_by_reason": {
                key.removeprefix("excluded:"): value
                for key, value in sorted(totals.items()) if key.startswith("excluded:")
            },
            "unlabeled_windows": unlabeled_summary["windows"],
            "eligible_by_split_class": {
                split: dict(sorted(values.items())) for split, values in split_class_counts.items()
            },
            "eligible_by_match": {
                match: sum(row["samples"] for row in shard_records if row["match"] == match)
                for match in split_map
            },
        },
        "unlabeled_stream": unlabeled_summary,
        "shards": shard_records,
        "training_contract": {
            "inputs": "student_inputs.npz keyed by sample_ids; x, entity_mask and visible only",
            "targets": "optional_targets.jsonl keyed by sample_id; action labels are separate",
            "privileged": "privileged_exo.npz keyed by sample_ids; query-frame full state is separate and must not be used as action input",
            "unlabeled": "unlabeled/student_windows.npz; no target records and no negative action labels",
        },
        "resume_policy": "Per match-half completion hashes; only complete hash-matched indexed shards are skipped; incomplete shards are rebuilt, never appended.",
    }
    write_json(output / "corpus_manifest.json", report)
    index = {
        "schema_version": VERSION,
        "status": "COMPLETE",
        "warning": "Consume only these indexed shards, not filesystem globs.",
        "shards": shard_records,
        "unlabeled": {
            "folder": str((output / "unlabeled").resolve()),
            **unlabeled_summary,
        },
    }
    write_json(output / "training_index.json", index)
    write_json(output / "progress.json", {
        "status": "COMPLETE",
        "completed_shards": len(shard_records),
        "total_shards": len(shard_records),
        "totals": report["counts"],
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = export_corpus(config.resolve(), output.resolve(), resume=args.resume)
    print(json.dumps({
        "status": result["status"],
        "output": str(output.resolve()),
        "counts": result["counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
