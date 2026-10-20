"""Build an isolated, fitting-only v0.3 action inference and export bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.backbone.actions import ActionModelSession
from src.backbone.batch import student_record
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import SoccerTrackSource
from src.player_centric_dataset import ACTION_ALIASES, ACTION_LABELS, ROW_ID, stable_seed
from src.recode_primary_actions import SOURCE_TO_PRIMARY_ACTION
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "outputs/partial_observation_20261017/students/final/S4/fov120/seed42/final.pt"
TRAINING_INDEX = ROOT / "outputs/backbone_v0_2_20261020_1558/batch/training_index.json"
TEMPORAL_LABELS = ROOT / "data/train_temporal_7class.jsonl"
VALIDATION_MATCHES = {"118577", "128058"}
AGE_SCALE_V0_2_SECONDS = 8.0
AGE_SCALE_S4_SECONDS = 5.0
SAMPLE_STRIDE = 5
HISTORY_STEPS = 11


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def indexed_sources(index_path: Path, fitting_matches: set[str]) -> tuple[set[tuple[str, str]], list[dict]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("status") != "COMPLETE" or not index.get("shards"):
        raise ValueError("Current v0.2 training_index.json is not complete")
    sources: set[tuple[str, str]] = set()
    shard_records = []
    for shard in index["shards"]:
        folder = Path(shard["folder"])
        path = folder / "student.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Indexed student shard is missing: {path}")
        sample_ids = []
        for row in read_jsonl(path):
            sample_id = row.get("sample_id", "")
            source_id = sample_id.split(":", 1)[0]
            match, separator, half = source_id.partition("_")
            if not separator or half not in ("1st", "2nd") or match not in fitting_matches:
                raise ValueError(f"Indexed training sample is outside the action checkpoint fitting split: {sample_id}")
            sample_ids.append(sample_id)
            sources.add((match, half))
        if len(sample_ids) != int(shard["samples"]):
            raise ValueError(f"Indexed shard count mismatch for {shard['job']}")
        shard_records.append({
            "job": shard["job"],
            "samples": len(sample_ids),
            "source_ids": sorted({sample_id.split(":", 1)[0] for sample_id in sample_ids}),
            "student_sha256": file_digest(path),
            "existing_optional_targets_sha256": file_digest(folder / "optional_targets.jsonl"),
        })
    if not sources:
        raise ValueError("The indexed v0.2 shards did not identify any training sources")
    return sources, shard_records


def event_candidates(
    temporal_path: Path,
    root: Path,
    sources: set[tuple[str, str]],
) -> tuple[list[dict], dict[str, str]]:
    events_by_match: dict[str, dict] = {}
    event_hashes: dict[str, str] = {}
    candidates = []
    for row in read_jsonl(temporal_path):
        match = str(row.get("match_id", ""))
        parsed = ROW_ID.fullmatch(str(row.get("id", "")))
        if parsed is None or parsed.group("match") != match:
            raise ValueError(f"Malformed stable event ID: {row.get('id')}")
        period = int(parsed.group("period"))
        half = "1st" if period == 1 else "2nd"
        if (match, half) not in sources:
            continue
        if match in VALIDATION_MATCHES:
            raise ValueError(f"Development/evaluation match cannot enter the fitting export: {match}")
        if match not in events_by_match:
            path = root / "data" / f"{match}_12_class_events.json"
            event_file = json.loads(path.read_text(encoding="utf-8"))
            events = event_file.get("actions") or event_file.get("annotations")
            if not isinstance(events, list):
                raise ValueError(f"Unsupported BAS event file: {path}")
            events_by_match[match] = {"file": event_file, "events": events}
            event_hashes[str(path.resolve())] = file_digest(path)
        event_file = events_by_match[match]["file"]
        events = events_by_match[match]["events"]
        annotation_index = int(parsed.group("index"))
        if annotation_index >= len(events):
            raise ValueError(f"BAS annotation index out of range: {row['id']}")
        event = events[annotation_index]
        event_period = str(event.get("gameTime", "")).split("-", 1)[0].strip()
        if event_period and event_period != str(period):
            raise ValueError(f"Period mismatch for stable event ID {row['id']}")
        if int(event.get("frame", -1)) != int(parsed.group("frame")):
            raise ValueError(f"BAS frame mismatch for stable event ID {row['id']}")
        source_label = str(event.get("label", ""))
        mapped = ACTION_ALIASES.get(source_label.upper().replace(" ", "_"))
        recoded_label = SOURCE_TO_PRIMARY_ACTION.get(str(row.get("source_label", "")))
        if mapped is None or mapped != row.get("response") or recoded_label != mapped:
            raise ValueError(f"BAS/recode label mismatch for stable event ID {row['id']}")
        if not event_file.get("clock", "").startswith("ms from the first frame"):
            raise ValueError(f"Unsupported BAS event clock for {row['id']}")
        position_ms = float(event["position"])
        anchor = int(math.floor(position_ms / 40.0))
        if anchor < 50:
            continue
        candidates.append({
            "event_id": row["id"],
            "match": match,
            "period": period,
            "half": half,
            "annotation_index": annotation_index,
            "stable_id_frame": int(parsed.group("frame")),
            "anchor_frame": anchor,
            "position_ms": position_ms,
            "source_label": source_label,
            "mapped_label": mapped,
            "actor_player_id": str(event.get("player_id", "")),
            "event": event,
            "temporal_row": row,
            "bas_path": str((root / "data" / f"{match}_12_class_events.json").resolve()),
        })
    candidates.sort(key=lambda row: (
        ACTION_LABELS.index(row["mapped_label"]), row["match"], row["period"],
        row["anchor_frame"], row["annotation_index"],
    ))
    return candidates, event_hashes


def build_candidate(
    candidate: dict,
    source: SoccerTrackSource,
    slot_by_player: dict[str, int],
    models: ModelSession,
    action_model: ActionModelSession,
) -> dict | None:
    actor_slot = slot_by_player.get(candidate["actor_player_id"])
    if actor_slot is None:
        return None
    anchor = candidate["anchor_frame"]
    first = anchor - (HISTORY_STEPS - 1) * SAMPLE_STRIDE
    indices = list(range(first, anchor + 1, SAMPLE_STRIDE))
    raw = source.raw
    if len(indices) != HISTORY_STEPS or indices[0] < 0 or indices[-1] != anchor:
        return None
    if not all(bool(raw["observed"][frame, actor_slot]) for frame in indices):
        return None
    segments = {int(raw["segments"][frame, actor_slot]) for frame in indices}
    if len(segments) != 1:
        return None
    history_observed = raw["observed"][indices]
    at_query = raw["observed"][anchor]
    previously_observed = history_observed.any(axis=0)
    if np.any(previously_observed & ~at_query):
        return None
    for slot in np.flatnonzero(at_query):
        observed_frames = history_observed[:, slot]
        if (len(set(raw["teams"][indices, slot][observed_frames].tolist())) > 1
                or len(set(raw["roles"][indices, slot][observed_frames].tolist())) > 1):
            return None
    actor_track = f"track_{actor_slot}"
    provider = SyntheticObservationProvider(
        actor_track,
        seed=stable_seed(candidate["event_id"], 42),
        fov=120,
        dropout=False,
    )
    engine = ReplayEngine(models, samples=16, action_model=action_model)
    result = None
    last_observation = None
    for frame_index in indices:
        frame = source.frame(frame_index)
        observation = provider.observe(frame)
        if observation.actor is None:
            return None
        last_observation = observation
        result = engine.step(observation, include_passing_assessment=True)
    if result is None or last_observation is None:
        return None
    inference = result.get("action_inference")
    if not inference or inference["status"] != "supported":
        return None
    student = student_record(last_observation, engine, result)
    if student["world_input"] is None:
        return None
    return {
        "student": student,
        "replay": result,
        "actor_track": actor_track,
        "input_frame_indices": indices,
        "source_provenance": source.provenance,
        "slot": actor_slot,
    }


def event_target(candidate: dict, source: SoccerTrackSource, slot: int, anchor: int) -> dict:
    source_label = candidate["source_label"]
    return {
        "schema_version": "football.action-target.0.1",
        "sample_id": f"{source.provenance['source_id']}:track_{slot}:{anchor}",
        "event_id": candidate["event_id"],
        "event_annotation_index": candidate["annotation_index"],
        "original_source_label": source_label,
        "mapped_action_label": candidate["mapped_label"],
        "mapping_provenance": {
            "mapping_file": "src/recode_primary_actions.py",
            "mapping_sha256": file_digest(ROOT / "src/recode_primary_actions.py"),
            "mapping_key": source_label.upper().replace(" ", "_"),
            "mapping_value": SOURCE_TO_PRIMARY_ACTION.get(source_label)
            or ACTION_ALIASES[source_label.upper().replace(" ", "_")],
            "mapping_normalization": "uppercase; spaces converted to underscores before alias lookup",
            "temporal_recode_source_label": candidate["temporal_row"]["source_label"],
            "temporal_record_sha256": file_digest(TEMPORAL_LABELS),
            "bas_file": candidate["bas_path"],
            "bas_file_sha256": file_digest(Path(candidate["bas_path"])),
            "join_method": "stable event ID match + parsed annotation index into BAS actions; not row order",
        },
        "timing": {
            "position_ms": candidate["position_ms"],
            "query_anchor_frame": anchor,
            "stable_id_frame": candidate["stable_id_frame"],
            "history_frame_indices": list(range(anchor - 50, anchor + 1, 5)),
            "causal_convention": "11 samples every 5 frames, inclusive of floor(position_ms/40); no post-event frames",
            "alignment_status": "SUPPORTED_POSITION_FLOOR_40MS",
        },
        "source_provenance": {
            "match": candidate["match"],
            "half": candidate["half"],
            "cache_path": source.provenance["cache_path"],
            "cache_sha256": source.provenance["cache_sha256"],
            "association_sha256": source.provenance["association_sha256"],
        },
    }


def build_bundle(output: Path, root: Path = ROOT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing v0.3 output directory: {output}")
    output.mkdir(parents=True)
    checkpoint = root / CHECKPOINT.relative_to(ROOT)
    index_path = root / TRAINING_INDEX.relative_to(ROOT)
    temporal_path = root / TEMPORAL_LABELS.relative_to(ROOT)
    action_model = ActionModelSession(checkpoint)
    fitting_matches = set(action_model.provenance["fitting_matches"])
    sources, shard_records = indexed_sources(index_path, fitting_matches)
    candidates, bas_hashes = event_candidates(temporal_path, root, sources)
    runtime_models = ModelSession(
        root / "outputs" / "dynamic_gnn_tactical_20261019", backend="last_seen",
    )

    source_objects: dict[tuple[str, str], SoccerTrackSource] = {}
    slot_maps: dict[tuple[str, str], dict[str, int]] = {}
    for match, half in sorted(sources):
        source = SoccerTrackSource(root, match, half)
        association_path = root / "outputs" / "trajectory_audit_20261016" / f"associations_{match}_{half}.json"
        association = json.loads(association_path.read_text(encoding="utf-8"))
        if association.get("cache_sha256") != source.provenance["cache_sha256"]:
            raise ValueError(f"Source association does not match verified cache: {match}_{half}")
        source_objects[(match, half)] = source
        slot_maps[(match, half)] = {str(player): int(slot) for player, slot in association["slots"].items()}

    selected_by_label: dict[str, dict] = {}
    rejected_counts: dict[str, int] = {}
    for candidate in candidates:
        label = candidate["mapped_label"]
        if label in selected_by_label:
            continue
        key = (candidate["match"], candidate["half"])
        result = build_candidate(
            candidate, source_objects[key], slot_maps[key], runtime_models, action_model,
        )
        if result is None:
            rejected_counts[label] = rejected_counts.get(label, 0) + 1
            continue
        selected_by_label[label] = {**candidate, **result}

    student_path = output / "student.jsonl"
    target_path = output / "optional_targets.jsonl"
    replay_path = output / "mixed_action_replay.json"
    with student_path.open("w", encoding="utf-8", newline="\n") as students, \
            target_path.open("w", encoding="utf-8", newline="\n") as targets:
        replay_examples = []
        for label in ACTION_LABELS:
            selected = selected_by_label.get(label)
            if selected is None:
                continue
            source = source_objects[(selected["match"], selected["half"])]
            target = event_target(selected, source, selected["slot"], selected["anchor_frame"])
            targets.write(json.dumps(target, allow_nan=False) + "\n")
            students.write(json.dumps(selected["student"], allow_nan=False) + "\n")
            replay = selected["replay"]
            replay_examples.append({
                "sample_id": replay["sample_id"],
                "query": {
                    "event_id": selected["event_id"],
                    "actor_track": selected["actor_track"],
                    "query_time_s": selected["anchor_frame"] / 25.0,
                    "input_frame_indices": selected["input_frame_indices"],
                    "conditioning": "event-conditioned demonstration; actor and query time supplied",
                },
                "player_facing_prediction": {
                    "action_output": replay["action"],
                    "model_scores": replay["action_inference"]["scores"],
                    "model_score_semantics": replay["action_inference"]["score_semantics"],
                },
                "exo_reference_prediction": {
                    "status": "UNAVAILABLE",
                    "reason": "No compatible exo-reference action checkpoint is configured",
                },
                "recorded_annotation_evaluation_only": {
                    "event_id": selected["event_id"],
                    "original_source_label": selected["source_label"],
                    "mapped_action_label": selected["mapped_label"],
                    "annotation_index": selected["annotation_index"],
                },
                "optional_passing_geometry": replay["modules"]["passing"],
                "ball_state": replay["ball_state"],
                "restart_context": replay["restart_context"],
                "outcome": replay["outcome"],
                "unimplemented_action_branches": {
                    "status": "UNSUPPORTED",
                    "reason": "This checkpoint supports only its seven trained classes; no fallback or inferred label is supplied outside that taxonomy",
                },
            })
    if len(selected_by_label) < 3 or sum(label != "PASS" for label in selected_by_label) < 2:
        raise RuntimeError(
            "Indexed fitting sources did not yield the required three-class/two-non-PASS coverage; "
            f"supported={sorted(selected_by_label)}"
        )

    source_records = [source_objects[key].provenance for key in sorted(source_objects)]
    replay_document = {
        "schema_version": "football.action-replay.0.3",
        "selection_policy": "First stable event ID per class with BAS label and complete causal input coverage, stable roster and team/role metadata; no prediction-based selection",
        "trained_classes": list(ACTION_LABELS),
        "unsupported_action_branches": "All action types outside the fixed seven-class taxonomy are unsupported.",
        "examples": replay_examples,
    }
    write_json(replay_path, replay_document)
    manifest = {
        "schema_version": "football.backbone.integration-manifest.0.3",
        "status": "PLAYER_FACING_ACTION_INFERENCE_SUPPORTED_EVENT_CONDITIONED",
        "checkpoint": action_model.provenance,
        "fitting_matches": action_model.provenance["fitting_matches"],
        "development_matches_recorded_separately": [],
        "held_out_validation_matches": ["118577", "128058"],
        "development_note": (
            "The selected final seed config records seven fitting matches and fixed_epochs=17; "
            "the original experiment reports held-out validation on 118577 and 128058. "
            "The v0.3 fitting export is restricted to matches/sources present in the current v0.2 training index."
        ),
        "input_contract": {
            "actor": "Any selected, continuously observed player; possession is not assumed",
            "training_conditioning": "The classifier was trained at annotated BAS event times; this does not make it a continuous action detector",
            "v0_2_source_support": "Compatible with the synthetic pitch-coordinate tracking observation stream. Pixel-only video remains unsupported without accepted pitch projection, synchronization, identities and team assignments.",
            "history": "11 samples, 0.2 seconds apart, at offsets -2.0s through query time inclusive",
            "query_time": "Event-conditioned demonstration query; inputs end at floor(BAS position_ms / 40); no post-event frames",
            "features": "22 anonymous slots x 10 relative player features in order: dx/52.5m, dy/34m, vx/12m/s, vy/12m/s (clipped to [-1,1]), visible, ever-seen, min(age,5s)/5, team relation (+1/-1), goalkeeper, selected actor",
            "visibility": "120-degree synthetic FOV, 45m range; causal motion-heading fallback, no artificial dropout; actor must remain visible. Seed is deterministic per event but the random fallback draw is not required to match the training cache's seed.",
            "memory": "First-sight causal memory; entity mask is model memory mask; visible mask is x[...,4] > 0.5",
            "age_normalization": "v0.2 input uses min(age,8s)/8; adapter scales that feature by 8/5 and clips to [0,1], matching S4 min(age,5s)/5",
            "identity_convention": "v0.2 first-sight slots; S4 has no slot embeddings and is permutation-equivariant",
            "roster_compatibility": "Event samples with a track seen earlier but absent at query time, or changing team/role metadata inside the history, are excluded; this matches the original event builder's current-roster feature contract.",
            "label_independence": "Labels, BAS IDs, global hidden coordinates and full-state order are never passed to the player-facing model",
        },
        "output_contract": {
            "action_output": "football.action-output.0.1; predicted class only, confidence null, support_status partial",
            "scores": "Seven-way softmax values retained as uncalibrated classifier scores, not tactical value",
            "other_fields": "Target, location, direction, timing, ball/possession, restart and outcome remain unset/unknown unless independently supported",
            "unimplemented_branches": "Unsupported outside the trained seven-class taxonomy",
        },
        "loading_and_inference": {
            "loader": "src.backbone.actions.ActionModelSession(checkpoint, device='cpu')",
            "predictor": "ActionModelSession.predict(WorldInput)",
            "replay_integration": "src.backbone.pipeline.ReplayEngine(..., action_model=session).step(observation)",
            "loads_per_session": action_model.load_count,
            "original_forward": "model(x, entity_mask, visible, memory_mode='memory') as in train/run_partial_observation_experiments.py",
        },
        "indexed_training_sources": {
            "training_index": str(index_path.resolve()),
            "training_index_sha256": file_digest(index_path),
            "shards": shard_records,
            "resolved_sources": source_records,
            "source_hashes": bas_hashes,
            "temporal_labels_path": str(temporal_path.resolve()),
            "temporal_labels_sha256": file_digest(temporal_path),
        },
        "selection": {
            "selected_classes": [label for label in ACTION_LABELS if label in selected_by_label],
            "rejected_input_coverage_by_class": rejected_counts,
            "examples": len(selected_by_label),
        },
        "artifacts": {
            "student": {"path": student_path.name, "sha256": file_digest(student_path)},
            "optional_targets": {"path": target_path.name, "sha256": file_digest(target_path)},
            "replay": {"path": replay_path.name, "sha256": file_digest(replay_path)},
        },
    }
    write_json(output / "integration_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/backbone_v0_3_20261020"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = build_bundle(output.resolve())
    print(json.dumps({
        "status": manifest["status"],
        "output": str(output.resolve()),
        "examples": manifest["selection"]["examples"],
        "classes": manifest["selection"]["selected_classes"],
    }, indent=2))


if __name__ == "__main__":
    main()
