"""Build configured v0.4 action queries through the shared replay engine."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from src.backbone.actions import ActionModelSession
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import SoccerTrackSource
from src.player_centric_dataset import ACTION_LABELS, stable_seed
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
FIT_MATCHES = {"117092", "117093", "118576", "118578", "128057", "132831"}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_fitting_examples(index_path: Path) -> dict[str, list[dict]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != "football.action-corpus.0.4" or index.get("status") != "COMPLETE":
        raise ValueError("Replay requires the completed v0.4 indexed corpus")
    by_class: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for shard in index["shards"]:
        if shard["split"] != "fitting":
            continue
        match = str(shard["match"])
        if match not in FIT_MATCHES:
            raise ValueError(f"Non-fitting match found in training replay index: {match}")
        folder = Path(shard["folder"])
        for name, digest in shard["artifact_hashes"].items():
            path = folder / name
            if not path.is_file() or file_digest(path) != digest:
                raise ValueError(f"Indexed replay artifact is missing or changed: {path}")
        with np.load(folder / "student_inputs.npz", allow_pickle=False) as archive:
            sample_ids = archive["sample_ids"].astype(str).tolist()
        metadata = {row["sample_id"]: row for row in _read_jsonl(folder / "student_metadata.jsonl")}
        targets = {row["sample_id"]: row for row in _read_jsonl(folder / "optional_targets.jsonl")}
        if len(metadata) != len(sample_ids) or len(targets) != len(sample_ids):
            raise ValueError(f"Keyed replay join is incomplete in {match}_{shard['half']}")
        for sample_id in sample_ids:
            if sample_id in seen:
                raise ValueError(f"Duplicate indexed training sample: {sample_id}")
            seen.add(sample_id)
            meta, target = metadata[sample_id], targets[sample_id]
            if (meta["match"] != match or meta["split"] != "fitting"
                    or target["source"]["match"] != match or target["source"]["split"] != "fitting"
                    or target["sample_id"] != sample_id):
                raise ValueError(f"Replay target provenance mismatch for {sample_id}")
            label = target["mapped_action_label"]
            if label not in ACTION_LABELS or target["action_class_index"] != ACTION_LABELS.index(label):
                raise ValueError(f"Unexpected trained class mapping for {sample_id}")
            by_class[label].append({
                "metadata": meta,
                "target": target,
                "match": match,
                "half": str(shard["half"]),
            })
    for examples in by_class.values():
        examples.sort(key=lambda row: (
            row["match"], row["half"], row["metadata"]["query_frame"],
            row["target"]["annotation_index"],
        ))
    return by_class


def _run_query(
    example: dict,
    root: Path,
    action_model: ActionModelSession,
    world_models: ModelSession,
    *,
    passing: bool,
) -> dict:
    meta = example["metadata"]
    match = str(example["match"])
    half = str(example["half"])
    actor_track = str(meta["actor_track"])
    source = SoccerTrackSource(root, match, half)
    provider = SyntheticObservationProvider(
        actor_track,
        seed=stable_seed(meta["sample_id"], 42),
        fov=120,
        dropout=False,
    )
    engine = ReplayEngine(world_models, samples=16, action_model=action_model)
    result = None
    for frame_index in meta["input_frame_indices"]:
        result = engine.step(
            provider.observe(source.frame(int(frame_index))),
            include_passing_assessment=passing,
        )
    if result is None or result.get("action_inference", {}).get("status") != "supported":
        raise RuntimeError(f"Configured replay query did not produce action inference: {meta['sample_id']}")
    return {
        "sample_id": meta["sample_id"],
        "query": {
            "actor_track": actor_track,
            "query_time_s": float(meta["query_time_s"]),
            "query_frame": int(meta["query_frame"]),
            "input_frame_indices": list(meta["input_frame_indices"]),
            "conditioning": (
                "event-conditioned demonstration; actor and query time supplied"
                if example.get("target") else
                "exploratory arbitrary observation window; not a detector evaluation"
            ),
        },
        "player_facing_prediction": result["action_inference"],
        "optional_passing_geometry": result["modules"]["passing"],
        "ball_state": result["ball_state"],
        "restart_context": result["restart_context"],
        "outcome": result["outcome"],
        "recorded_annotation_evaluation_only": (
            {
                "event_id": example["target"]["event_id"],
                "annotation_index": example["target"]["annotation_index"],
                "original_bas_label": example["target"]["original_bas_label"],
                "mapped_action_label": example["target"]["mapped_action_label"],
                "mapping_provenance": example["target"]["mapping_provenance"],
            }
            if example.get("target") else None
        ),
        "unsupported_branches": {
            "status": "UNSUPPORTED",
            "reason": "The adapter only supports its seven trained classes; this output is not a best-action recommendation.",
        },
        "source_provenance": {
            "match": match,
            "half": half,
            "cache_path": source.provenance["cache_path"],
            "cache_sha256": source.provenance["cache_sha256"],
        },
    }


def build_replay(index_path: Path, checkpoint: Path, output: Path, root: Path = ROOT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing v0.4 replay directory: {output}")
    output.mkdir(parents=True)
    action_model = ActionModelSession(
        checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    world_models = ModelSession(root / "outputs" / "dynamic_gnn_tactical_20261019", backend="last_seen")
    candidates = _load_fitting_examples(index_path)
    event_examples = []
    selected_classes = []
    for label in ACTION_LABELS:
        choices = candidates.get(label, [])
        if not choices:
            continue
        selected = choices[0]
        event_examples.append(_run_query(
            selected, root, action_model, world_models, passing=True,
        ))
        selected_classes.append(label)
        if len(event_examples) >= 3 and sum(name != "PASS" for name in selected_classes) >= 2:
            break
    if len(event_examples) < 3 or sum(name != "PASS" for name in selected_classes) < 2:
        raise RuntimeError(f"Indexed fitting corpus does not cover the required mixed classes: {selected_classes}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    unlabeled_folder = Path(index["unlabeled"]["folder"])
    unlabeled_summary = json.loads((unlabeled_folder / "summary.json").read_text(encoding="utf-8"))
    if unlabeled_summary.get("target_policy", "").find("not a negative") < 0:
        raise ValueError("Unlabeled replay windows do not declare missing action targets as unlabeled")
    arbitrary_meta = _read_jsonl(unlabeled_folder / "metadata.jsonl")
    arbitrary = [
        {
            "metadata": row,
            "match": str(row["match"]),
            "half": str(row["half"]),
        }
        for row in arbitrary_meta
        if row.get("target_availability", {}).get("action") is False
        and row.get("split") == "fitting"
    ][:3]
    if not arbitrary:
        raise ValueError("No fitting-only arbitrary observation windows are available for exploratory replay")
    exploratory_examples = [
        _run_query(row, root, action_model, world_models, passing=False) for row in arbitrary
    ]
    replay = {
        "schema_version": "football.action-replay.0.4",
        "status": "COMPLETE",
        "notice": (
            "Event-conditioned classifier demonstrations only. Exploratory arbitrary-window scores are not "
            "continuous detection, calibrated confidence, tactical value, or best-action recommendations."
        ),
        "trained_classes": list(ACTION_LABELS),
        "selection_policy": (
            "First fitting-indexed example with complete causal input coverage per class, in stable class/match/time order; "
            "selection does not depend on model correctness."
        ),
        "action_model": action_model.provenance,
        "action_model_load_count": action_model.load_count,
        "player_facing_event_queries": event_examples,
        "exploratory_unlabeled_queries": exploratory_examples,
        "target_policy": "Labels are evaluation-only; arbitrary windows have no action target and are not negative labels.",
    }
    write_json(output / "shared_replay.json", replay)
    manifest = {
        "schema_version": "football.backbone.integration-manifest.0.4",
        "status": "PLAYER_FACING_ACTION_REPLAY_SUPPORTED",
        "training_index": str(index_path.resolve()),
        "training_index_sha256": file_digest(index_path),
        "checkpoint": action_model.provenance,
        "replay_path": "shared_replay.json",
        "replay_sha256": file_digest(output / "shared_replay.json"),
        "replay_engine": "src.backbone.pipeline.ReplayEngine(..., action_model=ActionModelSession(...))",
        "event_classes_demonstrated": selected_classes,
        "exploratory_unlabeled_queries": len(exploratory_examples),
        "labels_in_player_facing_input": False,
        "ball_possession": "Unknown unless independently evidenced; not inferred from actor identity.",
    }
    write_json(output / "integration_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = args.index if args.index.is_absolute() else ROOT / args.index
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = build_replay(index.resolve(), checkpoint.resolve(), output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
