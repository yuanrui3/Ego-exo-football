"""Precomputed sequential replays and training-only student/teacher samples."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from src.backbone import SCHEMA_VERSION
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import EvaluationReference, load_source
from src.dynamic_football_data import FIT_MATCHES, RETROSPECTIVE
from src.trajectory_dataset import atomic_replace, write_json


def dataset_relation(match, models):
    fitting = {str(m) for model in models.provenance.values() for m in model["fitting_matches"]}
    development = {str(m) for model in models.provenance.values() for m in model["development_only"]}
    if str(match) in fitting:
        return "FITTING_MATCH: demo is NOT out-of-match evaluation"
    if str(match) in development:
        return "DEVELOPMENT_ONLY: used for selection, not an unseen test"
    if str(match) in RETROSPECTIVE:
        return "RETROSPECTIVE: previously evaluated, not a fresh prospective test"
    return "OUTSIDE_FITTING_LIST: independence and domain compatibility not established"


def sequence_coverage(observations):
    first_seen, last_visible, returned = set(), set(), set()
    disappeared = set()
    visible_teammates = []
    for observation in observations:
        visible = {(m.track, m.segment) for m in observation.measurements if m.track != observation.actor_track}
        disappeared |= last_visible - visible
        returned |= visible & disappeared & first_seen
        first_seen |= visible
        last_visible = visible
        actor = observation.actor
        visible_teammates.append(sum(m.team == actor.team and m.track != actor.track
                                      for m in observation.measurements) if actor is not None else 0)
    return {"observed_nonactor_tracks": len(first_seen), "disappeared_then_reappeared": len(returned),
            "visible_teammate_frames": sum(count > 0 for count in visible_teammates),
            "actor_observed_fraction": float(np.mean([obs.actor is not None for obs in observations]))}


def export_replays(root: Path, output: Path, config: dict, models: ModelSession):
    source_cache = {}
    sequences, summary = [], []
    for spec in config["sequences"]:
        adapter = spec.get("source_adapter", config.get("source_adapter", "soccertrack_cache"))
        key = (adapter, str(spec["match"]), spec["half"])
        if key not in source_cache:
            source_cache[key] = load_source(root, spec, adapter)
        source = source_cache[key]
        reference = EvaluationReference(source)
        frames = list(source.frames(spec["start_frame"], spec["samples"]))
        sequence = {"id": spec["id"], "source": source.provenance,
                    "dataset_relation": dataset_relation(spec["match"], models), "actors": {},
                    "selection": spec["selection"], "evaluation_only": []}
        for frame in frames:
            sequence["evaluation_only"].append(reference.overlay(frame))
        for actor in spec["actors"]:
            provider = SyntheticObservationProvider(actor, seed=config["seed"], fov=config["fov"])
            observations = [provider.observe(frame) for frame in frames]
            coverage = sequence_coverage(observations)
            engine = ReplayEngine(models, config["expiry_s"], config["fallback"], config["mc_samples"])
            rows = []
            for frame, obs in zip(frames, observations):
                # Full frame and evaluation reference are NOT passed to the engine.
                row = engine.step(obs)
                if models.p0 is not None:
                    supported = reference.receiver_states(frame.index, actor) if obs.actor is not None else None
                    row["exo_reference"] = (models.exo_reference(*supported) if supported is not None else
                                            {"status": "UNAVAILABLE", "reason": "Actor missing or unsupported full-state capacity"})
                rows.append(row)
            sequence["actors"][actor] = rows
            all_entities = [e for row in rows for e in row["belief"]["entities"]]
            hidden_entities = [e for e in all_entities if not e["visible"]]
            fallback = sum(e["backend"] in ("last_seen", "constant_velocity") for e in hidden_entities)
            summary.append({
                "sequence": spec["id"], "actor": actor, "frames": len(rows), "failed_frames": [],
                "coverage": coverage,
                "latency_ms_p50": float(np.median([row["latency_ms"] for row in rows])),
                "latency_ms_p95": float(np.quantile([row["latency_ms"] for row in rows], .95)),
                "frames_with_absent_cache_slots": sum(frame.missing_count > 0 for frame in frames),
                "absent_cache_slots_mean": float(np.mean([frame.missing_count for frame in frames])),
                "missingness_warning": "Absent cache slots may be unused roster slots; no physical missing-player count inferred",
                "actor_missing_frames": sum(obs.actor is None for obs in observations),
                "learned_hidden_entity_estimates": sum(e["backend"] in ("G1", "G3", "G4") for e in hidden_entities),
                "hidden_entity_estimates": len(hidden_entities),
                "fallback_entity_estimates": fallback,
                "fallback_fraction_hidden_estimates": fallback / len(hidden_entities) if hidden_entities else None,
                "warmup_or_unsupported_frames": sum(not row["belief"]["model_contract_met"] for row in rows),
                "available_lane_assessments": sum(c["status"] == "AVAILABLE" for row in rows
                                                  for c in row["assessment"]["candidates"]),
                "unavailable_uncertain_candidate_lanes": sum(not c["visible"] for row in rows
                                                             for c in row["assessment"]["candidates"]),
            })
            print(f"Exported {spec['id']}/{actor}: {len(rows)} sequential frames, coverage={coverage}", flush=True)
        sequences.append(sequence)
    bundle = {
        "schema_version": SCHEMA_VERSION, "notice": "Synthetic player visibility; tracking-based prototype.",
        "description": "Tracking -> simulated observations -> causal memory -> G1 belief -> Gaussian geometry. NOT raw-video-to-ego.",
        "units": "metres", "sampling_seconds": .2, "config": config,
        "model_provenance": models.provenance, "model_load_counts": models.load_counts,
        "seeking": "Viewer indexes states precomputed by chronological prefix replay from each segment start",
        "truth_overlay_default": False, "exo_reference_default": False, "sequences": sequences,
    }
    write_json(output / "replay.json", bundle)
    write_json(output / "replay_validation.json", {
        "status": "PASSED", "sequences": len(sequences), "actor_runs": summary,
        "models_loaded_once": models.load_counts,
        "has_disappearance_reappearance": any(row["coverage"]["disappeared_then_reappeared"] for row in summary),
        "no_tactical_accuracy_claim": True,
    })
    return bundle


def export_training_sample(root, output, models, spec, count=8):
    """Student and teacher files are deliberately separate and keyed by sample ID."""
    if str(spec["match"]) not in FIT_MATCHES:
        raise ValueError("Future-learning export may use ONLY the six permitted fitting matches")
    if count < 1 or count > 128:
        raise ValueError("This small integration export supports 1..128 samples, not a full dataset rebuild")
    source = load_source(root, spec, spec.get("source_adapter", "soccertrack_cache"))
    provider = SyntheticObservationProvider(spec["actor"], seed=42)
    engine = ReplayEngine(models)
    reference = EvaluationReference(source)
    students, teachers, optional = [], [], []
    for index, frame in enumerate(source.frames(spec["start_frame"], count + 10)):
        obs = provider.observe(frame)
        result = engine.step(obs)
        if index < 10:
            continue
        inputs = engine.memory.current_input
        record = {
            "schema_version": SCHEMA_VERSION, "sample_id": result["sample_id"],
            "source_id": obs.source_id, "timestamp_s": obs.timestamp_s, "frame_index": obs.index,
            "units": "metres", "observation": dataclasses.asdict(obs),
            "causal_memory": result["belief"], "world_input": ({
                key: getattr(inputs, key).tolist() for key in ("x", "mask", "actor_xy", "last_xy")}
                if inputs is not None else None),
            "missingness": {"actor_missing": obs.actor is None, "world_contract_met": inputs is not None},
        }
        students.append(record)
        teachers.append({
            "schema_version": SCHEMA_VERSION, "sample_id": result["sample_id"],
            "privileged_exo": reference.overlay(frame),
            "supervision_confidence": "TRACKING_PROXY_ONLY: quantized/missing GSR, no physical uncertainty calibration",
            "source_missing_count": frame.missing_count,
        })
        optional.append({"sample_id": result["sample_id"], "event_target": None, "receiver_proxy_target": None,
                         "status": "NO_EVENT_JOIN: arbitrary replay timestamp, not asserted pass decision"})
    folder = output / "training_sample"
    folder.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("student.jsonl", students), ("privileged_exo.jsonl", teachers),
                           ("optional_targets.jsonl", optional)):
        path = folder / filename
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
        atomic_replace(temporary, path)
    write_json(folder / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "samples": len(students), "match": str(spec["match"]),
        "source": source.provenance, "models": models.provenance,
        "dataset_relation": dataset_relation(spec["match"], models),
        "permitted_fitting_matches": list(FIT_MATCHES),
        "separation": {"student": "student.jsonl", "privileged_supervision": "privileged_exo.jsonl",
                       "optional_proxy_targets": "optional_targets.jsonl"},
        "warning": "These are fitting-match examples, not a held-out evaluation; no new teacher training performed",
    })


def main():
    import argparse
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Export a small, fitting-match-only student/teacher sample; no training")
    parser.add_argument("--config", type=Path, default=root / "configs" / "backbone_v0_1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", help="One training_export job from a v0.2 batch configuration")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") == "backbone.batch.0.2":
        from src.backbone.batch import run_batch
        selected = next((job for job in config["jobs"] if job["id"] == args.only), None)
        if selected is None:
            parser.error("v0.2 training export requires --only TRAINING_JOB")
        if {**config.get("defaults", {}), **selected}["intended_use"] != "training_export":
            parser.error("Selected job is not training_export; replay/evaluation cannot become training data")
        result = run_batch(args.config.resolve(), args.output.resolve(), args.resume, only=args.only)
        if result["status"] != "COMPLETE":
            raise SystemExit(1)
        return
    if args.only or args.resume:
        parser.error("--only/--resume require a v0.2 batch configuration")
    experiment = root / config["experiment"]
    if json.loads((experiment / "progress_summary.json").read_text())["status"] != "PAUSED_FOR_INTEGRATION":
        raise ValueError("Research must stay paused")
    if (args.output / "training_sample" / "manifest.json").exists():
        raise FileExistsError("Use a new sample-export directory")
    import torch
    torch.set_num_threads(4)
    models = ModelSession(experiment, config["backend"])
    export_training_sample(root, args.output, models, config["training_sample"], config["training_sample"]["count"])
    print(args.output / "training_sample" / "manifest.json")


if __name__ == "__main__":
    main()
