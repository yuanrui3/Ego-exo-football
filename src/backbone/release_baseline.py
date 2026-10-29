"""Run the frozen tracking-based G1/S4 baseline into a new output directory."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import torch

from src.backbone.actions import ActionModelSession
from src.backbone.external import ExternalObservationSource
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import EvaluationReference, SoccerTrackSource
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("PASS", "HIGH_PASS", "DRIVE", "CROSS", "SHOT", "HEADER", "DEFENSIVE_INTERVENTION")


def _verified_queries(config: dict) -> tuple[dict[int, dict], dict]:
    source = config["source"]
    if source["adapter"] == "external_jsonl":
        queries = {}
        for query in config.get("event_queries", []):
            frame = query["frame_index"]
            event_id = query.get("event_id")
            annotation_index = query.get("annotation_index")
            if (not isinstance(frame, int) or isinstance(frame, bool)
                    or not isinstance(event_id, str) or not event_id
                    or not isinstance(annotation_index, int) or isinstance(annotation_index, bool)):
                raise ValueError("External action queries require frame_index, event_id and annotation_index")
            if "mapped_action_label" in query or "original_bas_label" in query:
                raise ValueError("External annotations must not be displayed without a verified label join")
            if frame in queries:
                raise ValueError(f"Duplicate external action query frame: {frame}")
            queries[frame] = {
                "event_id": event_id,
                "annotation_index": annotation_index,
                "conditioning": "event-conditioned demonstration; actor and query time supplied",
                "annotation_status": "UNVERIFIED/NOT_SUPPLIED",
            }
        return queries, {"labels": "No annotation targets joined or supplied"}
    if source["adapter"] != "soccertrack_cache":
        raise ValueError(f"Unsupported baseline source adapter: {source['adapter']}")
    root = ROOT / source["training_index"]
    index = json.loads(root.read_text(encoding="utf-8"))
    shards = [row for row in index["shards"]
              if str(row["match"]) == str(source["match"]) and row["half"] == source["half"]]
    if len(shards) != 1 or shards[0]["split"] != "retrospective_evaluation":
        raise ValueError("Configured event labels require exactly one retrospective indexed shard")
    shard = shards[0]
    folder = Path(shard["folder"])
    loaded = {}
    for name in ("student_metadata.jsonl", "optional_targets.jsonl"):
        path = folder / name
        if file_digest(path) != shard["artifact_hashes"][name]:
            raise ValueError(f"Indexed event artifact failed hash validation: {name}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        loaded[name] = {row["sample_id"]: row for row in rows}
        if len(loaded[name]) != len(rows):
            raise ValueError(f"Duplicate sample IDs in indexed event artifact: {name}")
    metadata, targets = loaded["student_metadata.jsonl"], loaded["optional_targets.jsonl"]
    if metadata.keys() != targets.keys():
        raise ValueError("Student metadata and optional action targets do not join by sample ID")
    by_actor_frame = {}
    for sample_id, row in metadata.items():
        key = (row["actor_track"], int(row["query_frame"]))
        if key in by_actor_frame:
            raise ValueError(f"Ambiguous multiple annotations at actor/query time: {key}")
        by_actor_frame[key] = (sample_id, targets[sample_id])
    actor = source["actor"]
    expected = {row["frame_index"]: row for row in config["event_queries"]}
    if len(expected) != len(config["event_queries"]):
        raise ValueError("Configured event query frames must be unique")
    actual = {}
    for frame, wanted in expected.items():
        if frame % int(source["stride_native_frames"]):
            raise ValueError(f"Event query frame is not on the configured source grid: {frame}")
        joined = by_actor_frame.get((actor, frame))
        if joined is None:
            continue
        sample_id, target = joined
        wanted = expected[frame]
        if any(target.get(key) != wanted.get(key) for key in (
                "event_id", "annotation_index", "original_bas_label", "mapped_action_label")):
            raise ValueError(f"Configured event provenance does not match current indexed target at frame {frame}")
        actual[frame] = {
            "sample_id": sample_id,
            "event_id": target["event_id"],
            "annotation_index": target["annotation_index"],
            "original_bas_label": target["original_bas_label"],
            "mapped_action_label": target["mapped_action_label"],
            "mapping_provenance": target["mapping_provenance"],
            "conditioning": "event-conditioned demonstration; actor and query time supplied",
        }
    if set(actual) != set(expected):
        raise ValueError(f"Indexed event query mismatch; missing frames: {sorted(set(expected) - set(actual))}")
    provenance = {
        "training_index_sha256": file_digest(root),
        "shard_fingerprint": shard["fingerprint"],
        "shard_artifacts": {
            name: {"path": str(folder / name), "sha256": file_digest(folder / name)}
            for name in ("student_metadata.jsonl", "optional_targets.jsonl")
        },
        "selection": "Fixed release segment and actor; event targets joined by stable sample ID, event ID and annotation index.",
        "labels": "Evaluation-only; not passed to provider, memory, G1 or S4",
    }
    return actual, provenance


def run(config_path: Path, output: Path) -> dict:
    config_path = config_path.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Release outputs are immutable; choose an empty, new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "football.tracking-backbone-release.1":
        raise ValueError("Unsupported release baseline configuration")
    release_manifest_path = ROOT / config["release_manifest"]
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    if release_manifest["canonical_configuration_sha256"] != file_digest(config_path):
        raise ValueError("Canonical configuration does not match its pinned release manifest")
    world_model = release_manifest["models"]["default_world"]
    action_model_manifest = release_manifest["models"]["default_action"]
    world_key = world_model["checkpoint"]
    if (config["world_backend"] != world_model["name"]
            or config["pinned_files"].get(world_key) != world_model["sha256"]
            or config["action_checkpoint"] != action_model_manifest["checkpoint"]
            or config["pinned_files"].get(config["action_checkpoint"]) != action_model_manifest["sha256"]):
        raise ValueError("Release manifest and configuration disagree on default checkpoint hashes")
    torch.set_num_threads(4)

    for relative_path, expected_sha256 in config.get("pinned_files", {}).items():
        path = ROOT / relative_path
        if not path.is_file() or file_digest(path) != expected_sha256:
            raise ValueError(f"Release baseline prerequisite hash mismatch: {relative_path}")
    experiment = ROOT / config["experiment"]
    action_path = ROOT / config["action_checkpoint"]
    models = ModelSession(experiment, config["world_backend"], device="cpu")
    action_model = ActionModelSession(action_path, device="cpu")
    world_record = models.provenance[config["world_backend"]]
    world_checkpoint_key = str((experiment / world_record["path"]).relative_to(ROOT)).replace("\\", "/")
    if world_record["sha256"] != config["pinned_files"][world_checkpoint_key]:
        raise ValueError("Loaded world-model provenance differs from the pinned release model")
    if action_model.sha256 != config["pinned_files"][config["action_checkpoint"]]:
        raise ValueError("Loaded action-model provenance differs from the pinned release model")
    source_config = config["source"]
    adapter = source_config["adapter"]
    if adapter == "soccertrack_cache":
        source = SoccerTrackSource(
            ROOT, str(source_config["match"]), source_config["half"],
            cache_root=source_config["tracking_cache"],
            association_root=source_config["associations"],
        )
        actor_track = source_config["actor"]
    elif adapter == "external_jsonl":
        source = ExternalObservationSource(ROOT / source_config["input_path"])
        actor_track = source_config.get("actor", source.header.get("actor_track"))
        if not actor_track:
            raise ValueError("External source must declare actor in configuration or player-visible metadata")
        if (source.match != str(source_config["match"]) or source.half != source_config["half"]):
            raise ValueError("External source match/half metadata does not match the release configuration")
        if source.header["observation_kind"] == "player_visible" and actor_track != source.header["actor_track"]:
            raise ValueError("Configured actor must match the source's player-visible actor")
    else:
        raise ValueError(f"Unsupported baseline source adapter: {adapter}")
    frames = list(source.frames(
        int(source_config["start_frame"]), int(source_config["samples"]),
        int(source_config["stride_native_frames"]),
    ))
    if len(frames) != int(source_config["samples"]):
        raise ValueError("Source did not produce the configured segment length")
    queries, event_selection = _verified_queries(config)
    valid_frames = {frame.index for frame in frames}
    if not set(queries) <= valid_frames:
        raise ValueError("An event query falls outside the configured source segment")

    provider = SyntheticObservationProvider(
        actor_track, seed=int(source_config["seed"]),
        fov=int(source_config["field_of_view_degrees"]),
        dropout=bool(source_config["dropout"]),
    )
    engine = ReplayEngine(
        models,
        expiry_s=float(config["replay"]["memory_expiry_seconds"]),
        fallback=config["replay"]["fallback"],
        action_model=action_model,
    )
    reference = EvaluationReference(source) if isinstance(source, SoccerTrackSource) else None
    rows = []
    for frame in frames:
        observation = (source.observe(frame, provider) if isinstance(source, ExternalObservationSource)
                       else provider.observe(frame))
        annotation = queries.get(frame.index)
        result = engine.step(
            observation,
            include_passing_assessment=bool(config["replay"]["passing_assessment"]),
            action_query=annotation is not None,
        )
        if annotation is not None:
            result["action_query_conditioning"] = annotation["conditioning"]
            if "mapped_action_label" in annotation:
                result["recorded_annotation_evaluation_only"] = annotation
        rows.append(result)

    sequence = {
        "id": f"{source_config['match']}_{source_config['half']}_f{source_config['start_frame']}",
        "source": source.provenance,
        "dataset_relation": "RETROSPECTIVE: previously evaluated; not an independent test",
        "selection": "Fixed source segment and actor; any supplied event targets are evaluation-only.",
        "actors": {actor_track: rows},
        "evaluation_only": ([reference.overlay(frame) for frame in frames] if reference else [
            {"label": "SOURCE OBSERVATIONS ONLY; incomplete inventory is not privileged truth",
             "players": [{"source_track": m.track, "xy": list(m.xy), "team": m.team}
                         for m in frame.measurements],
             "absent_cache_slots": frame.missing_count, "privileged_available": False}
            for frame in frames
        ]),
    }
    bundle = {
        "schema_version": "backbone.0.1",
        "notice": "Synthetic player visibility; tracking-based prototype.",
        "description": "Tracking observations -> causal memory/G1 estimates and separate event-conditioned S4 action queries.",
        "units": "metres",
        "sampling_seconds": 0.2,
        "truth_overlay_default": False,
        "exo_reference_default": False,
        "model_provenance": models.provenance,
        "action_model_provenance": action_model.provenance,
        "config": config,
        "sequences": [sequence],
    }
    bundle_path = output / "replay.json"
    viewer_path = output / "viewer.html"
    write_json(bundle_path, bundle)
    payload = json.dumps(bundle, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = (ROOT / "src" / "backbone" / "viewer.html").read_text(encoding="utf-8")
    viewer_path.write_text(template.replace("__REPLAY_JSON__", payload), encoding="utf-8")

    query_rows = [row for row in rows if row["frame_index"] in queries]
    supported = [row for row in query_rows
                 if row.get("action_inference", {}).get("status") == "supported"]
    labels = {row["recorded_annotation_evaluation_only"]["mapped_action_label"] for row in supported
              if row.get("recorded_annotation_evaluation_only")}
    needs_mixed_annotations = adapter == "soccertrack_cache" and bool(queries)
    if (len(supported) != len(queries) or (needs_mixed_annotations and len(labels) < 3)):
        raise ValueError(
            f"Mixed-action release replay needs all event queries supported and >=3 labels; "
            f"got {len(supported)}/{len(queries)} supported, labels={sorted(labels)}"
        )
    if isinstance(source, SoccerTrackSource):
        source_requirements = {
            "tracking_cache": str(Path(source.provenance["cache_path"])),
            "tracking_cache_metadata": str(Path(source.provenance["cache_path"]).with_suffix(".json")),
            "association_manifest": str(
                ROOT / source_config["associations"] /
                f"associations_{source_config['match']}_{source_config['half']}.json"
            ),
            "training_index": str(ROOT / source_config["training_index"]),
        }
    else:
        source_requirements = {"external_observation_jsonl": str(source.path)}
    result = {
        "status": "PASS",
        "segment": f"{source_config['match']}_{source_config['half']}:{source_config['start_frame']}"
                   f"+{len(frames)}x{source_config['stride_native_frames']}",
        "actor": actor_track,
        "frames": len(rows),
        "event_queries": len(queries),
        "supported_queries": len(supported),
        "annotation_classes": sorted(labels),
        "predicted_classes": sorted({row["action"]["action_type"] for row in supported}),
        "model_load_counts": {"G1": models.load_counts["world"], "S4": action_model.load_count},
        "g1_to_s4_feature_connection": False,
        "annotations_passed_to_inference": False,
        "optional_passing_module": "DISABLED",
        "ball_possession": "UNKNOWN",
        "source_sha256": source.provenance.get("cache_sha256", source.provenance.get("input_sha256")),
        "source_association_sha256": source.provenance.get("association_sha256"),
        "event_selection": event_selection,
        "source_requirements": source_requirements,
        "files": {
            "replay.json": file_digest(bundle_path),
            "viewer.html": file_digest(viewer_path),
        },
    }
    write_json(output / "replay_validation.json", result)
    g1_checkpoint = Path(models.provenance["G1"]["resolved_path"])
    g1_files = (
        g1_checkpoint,
        g1_checkpoint.parent / "config.json",
        g1_checkpoint.parent / "result.json",
        experiment / "integration_manifest.json",
    )
    if isinstance(source, SoccerTrackSource):
        input_files = (
            Path(source.provenance["cache_path"]),
            Path(source.provenance["cache_path"]).with_suffix(".json"),
            ROOT / source_config["associations"] / f"associations_{source_config['match']}_{source_config['half']}.json",
            ROOT / source_config["training_index"],
            *(Path(item["path"]) for item in event_selection["shard_artifacts"].values()),
        )
    else:
        input_files = (source.path,)
    manifest = {
        "status": "EXPORTED",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": file_digest(config_path),
        "release_manifest_path": str(release_manifest_path),
        "release_manifest_sha256": file_digest(release_manifest_path),
        "source_files": {},
        "model_provenance": models.provenance,
        "action_model_provenance": action_model.provenance,
        "verification": result,
        "no_training_or_source_mutation": True,
    }
    for path in (action_path, action_path.parent / "config.json",     release_manifest_path,
    *g1_files, *input_files):
        try:
            key = str(path.resolve().relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            key = str(path.resolve())
        manifest["source_files"][key] = file_digest(path)
    write_json(output / "run_manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "backbone_release_baseline.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="A new or empty directory; existing nonempty output is never overwritten")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
