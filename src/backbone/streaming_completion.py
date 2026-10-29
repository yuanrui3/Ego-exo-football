"""Complete 5 Hz tracking halves with persistent causal state and bounded shards."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.backbone.actions import ActionModelSession
from src.backbone import SCHEMA_VERSION
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import SoccerTrackSource
from src.backbone.train_action import _restore_rng, _rng_state, _save_torch
from src.trajectory_dataset import atomic_replace, file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "outputs" / "backbone_v0_5_20261020" / "corpus" / "training_index.json"
ACTION = ROOT / "outputs" / "backbone_v0_5_20261020" / "training" / "model" / "final.pt"
HALVES = (("118577", "1st"), ("128058", "2nd"))


def state_of(provider: SyntheticObservationProvider, engine: ReplayEngine) -> dict:
    return deepcopy({
        "provider": vars(provider),
        "memory": {key: value for key, value in vars(engine.memory).items() if key != "models"},
    })


def restore(state: dict, models: ModelSession, action: ActionModelSession):
    provider = SyntheticObservationProvider(state["provider"]["actor_track"])
    provider.__dict__.update(deepcopy(state["provider"]))
    engine = ReplayEngine(models, action_model=action)
    engine.memory.__dict__.update(deepcopy(state["memory"]))
    return provider, engine


def compact(row: dict) -> dict:
    row = deepcopy(row)
    row.pop("latency_ms")
    occupancy = row["belief"]["anonymous_occupancy"]
    for name in ("all_hidden", "never_seen_defender"):
        values = occupancy[name]
        if values is not None:
            array = np.asarray(values)
            occupancy[name] = {"shape": list(array.shape), "sum": float(array.sum()),
                               "maximum": float(array.max()), "dense_array_stored": False}
    occupancy["storage_note"] = "Dense occupancy omitted in stream; bounded viewer retains full arrays."
    return row


def event_schedule(match: str, half: str) -> tuple[list[str], dict, dict]:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    shards = [s for s in index["shards"] if s["match"] == match and s["half"] == half]
    if len(shards) != 1 or shards[0]["split"] != "retrospective_evaluation":
        raise ValueError("Full-half replay must use one indexed retrospective shard")
    shard = shards[0]
    folder = Path(shard["folder"])
    loaded = {}
    for name in ("student_metadata.jsonl", "optional_targets.jsonl"):
        if file_digest(folder / name) != shard["artifact_hashes"][name]:
            raise ValueError(f"Changed current indexed shard: {name}")
        rows = [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").splitlines()]
        loaded[name] = {row["sample_id"]: row for row in rows}
        if len(loaded[name]) != len(rows):
            raise ValueError("Duplicate event key")
    metadata, targets = loaded["student_metadata.jsonl"], loaded["optional_targets.jsonl"]
    if metadata.keys() != targets.keys():
        raise ValueError("Event target key mismatch")
    # Selection uses label availability only, never model scores or prediction correctness.
    remaining = sorted({row["actor_track"] for row in metadata.values()})
    actors, labels = [], set()
    for _ in range(4):
        actor = min(remaining, key=lambda track: (
            -len({targets[key]["mapped_action_label"] for key, row in metadata.items()
                  if row["actor_track"] == track} - labels),
            -sum(row["actor_track"] == track for row in metadata.values()), track))
        actors.append(actor)
        remaining.remove(actor)
        labels |= {targets[key]["mapped_action_label"] for key, row in metadata.items()
                   if row["actor_track"] == actor}
    schedule, excluded = {}, Counter()
    for key, row in metadata.items():
        if row["actor_track"] not in actors:
            continue
        frame = int(row["query_frame"])
        if frame % 5:
            excluded["unsupported_exact_alignment_to_continuous_5hz_grid"] += 1
            continue
        schedule.setdefault((row["actor_track"], frame), []).append(targets[key])
    provenance = {"index_sha256": file_digest(INDEX), "shard_fingerprint": shard["fingerprint"],
                  "selection": "Greedy trained-class annotation coverage; count then track tie-break, no score selection",
                  "available_annotated_classes": sorted(labels), "excluded_queries": dict(excluded),
                  "query_timing": "Only exact original annotation query frames on 0,5,... grid; no rounding or future frames",
                  "scheduled_event_annotations": sum(map(len, schedule.values())),
                  "labels": "Evaluation-only; not passed to provider, memory, G1 or action model"}
    return actors, schedule, provenance


def resume_parity(source: SoccerTrackSource, actor: str, models, action, output: Path) -> dict:
    provider, engine = SyntheticObservationProvider(actor), ReplayEngine(models, action_model=action)
    for frame in source.frames(0, 300):
        engine.step(provider.observe(frame), include_passing_assessment=False)
    checkpoint = output / "boundary_parity.pt"
    _save_torch(checkpoint, {"state": state_of(provider, engine), "rng": _rng_state()})
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _restore_rng(saved["rng"])
    other_provider, other_engine = restore(saved["state"], models, action)
    before = len(engine.memory.history)
    for frame in source.frames(1500, 25):
        left = compact(engine.step(provider.observe(frame), include_passing_assessment=False, action_query=True))
        right = compact(other_engine.step(other_provider.observe(frame),
                                         include_passing_assessment=False, action_query=True))
        if left != right:
            raise ValueError("Persisted chunk state changed replay numerics or identities")
    result = {"status": "PASS", "boundary_frame": 1500, "prefix_samples": 300,
              "compared_actor_steps": 25, "maximum_numerical_difference": 0.0,
              "history_length_before_boundary": before, "explicit_state_checkpoint": str(checkpoint),
              "rng_preserved": True, "comparison": "All stored fields except wall-clock latency"}
    write_json(output / "boundary_parity.json", result)
    return result


def run(output: Path, *, stop_after_chunks: int | None = None, device: str = "cuda",
        deadline: float | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    models = ModelSession(ROOT / "outputs" / "dynamic_gnn_tactical_20261019" , device=device)
    action = ActionModelSession(ACTION, device=device)
    manifests = []
    invocation_started = time.perf_counter()
    completed_this_invocation = 0
    for match, half in HALVES:
        folder = output / f"{match}_{half}"
        folder.mkdir(exist_ok=True)
        source = SoccerTrackSource(ROOT, match, half)
        actors, queries, event_provenance = event_schedule(match, half)
        specification = {
            "source": source.provenance, "actors": actors, "sampling_stride": 5, "start": 0,
            "native_frames": len(source.raw["xy"]), "chunk_source_samples": 300,
            "action": action.provenance, "world": models.provenance,
            "events": event_provenance,
            "implementation_sha256": file_digest(Path(__file__)),
            "observation_implementation_sha256": file_digest(Path(__file__).with_name("observation.py")),
            "selected_viewer": "First 40 and last 40 samples; first 20 exact aligned event queries",
        }
        fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        total = len(range(0, len(source.raw["xy"]), 5))
        path = folder / "state.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["fingerprint"] != fingerprint:
                raise ValueError("Source/models/implementation changed since stream checkpoint")
            for shard in saved["shards"]:
                if file_digest(folder / shard["file"]) != shard["sha256"]:
                    raise ValueError("Changed committed stream shard")
            _restore_rng(saved["rng"])
            streams = {actor: restore(saved["states"][actor], models, action) for actor in actors}
        else:
            saved = {"fingerprint": fingerprint, "cursor": 0, "shards": [],
                     "stats": {}, "viewer": {actor: [] for actor in actors}, "resumes": 0,
                     "runtime_s": 0.0}
            streams = {actor: (SyntheticObservationProvider(actor), ReplayEngine(models, action_model=action))
                       for actor in actors}
            write_json(folder / "specification.json", specification)
            resume_parity(source, actors[0], models, action, folder)
        saved["resumes"] += int(saved["cursor"] not in (0, total))
        stats = Counter(saved["stats"])
        viewer_query_frames = set(sorted({frame for _, frame in queries})[:20])
        for start in range(saved["cursor"], total, 300):
            tick = time.perf_counter()
            end = min(start + 300, total)
            filename = f"chunk_{start:06d}_{end:06d}.jsonl.gz"
            target = folder / filename
            temporary = folder / (filename + ".tmp")
            if target.exists():
                atomic_replace(target, target.with_name(target.name + f".orphan_{time.time_ns()}"))
            calls_before = models.world_forward_calls, action.forward_calls
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as writer:
                for sample in range(start, end):
                    frame = source.frame(sample * 5)
                    for actor, (provider, engine) in streams.items():
                        observation = provider.observe(frame)
                        annotations = queries.get((actor, frame.index))
                        row = engine.step(observation, include_passing_assessment=False,
                                          action_query=annotations is not None)
                        if annotations:
                            row["recorded_annotations_evaluation_only"] = annotations
                        if sample < 40 or sample >= total - 40 or frame.index in viewer_query_frames:
                            saved["viewer"][actor].append(deepcopy(row))
                        writer.write(json.dumps(compact(row), allow_nan=False, separators=(",", ":")) + "\n")
                        stats["actor_steps"] += 1
                        stats["actor_missing"] += int(observation.actor is None)
                        stats["model_contract_met"] += int(row["belief"]["model_contract_met"])
                        stats["scheduled_action_queries"] += int(annotations is not None)
                        stats["supported_action_queries"] += int(row["action"]["support_status"] != "unsupported")
                        for entity in row["belief"]["entities"]:
                            if not entity["visible"]:
                                stats[f"hidden_backend_{entity['backend']}"] += 1
                        stats["max_memory_entries"] = max(stats["max_memory_entries"], len(engine.memory.entries))
                        stats["max_history_samples"] = max(stats["max_history_samples"], len(engine.memory.history))
                    stats["source_samples"] += 1
            atomic_replace(temporary, target)
            stats["world_calls"] += models.world_forward_calls - calls_before[0]
            stats["action_calls"] += action.forward_calls - calls_before[1]
            saved["shards"].append({"file": filename, "sha256": file_digest(target),
                                     "first_sample": start, "end_sample_exclusive": end,
                                     "records": (end - start) * len(actors)})
            saved.update(cursor=end, stats=dict(stats),
                         states={actor: state_of(*stream) for actor, stream in streams.items()}, rng=_rng_state())
            saved["runtime_s"] += time.perf_counter() - tick
            _save_torch(path, saved)
            metrics = {"status": "COMPLETE" if end == total else "RUNNING",
                       "native_duration_minutes": len(source.raw["xy"]) / 25 / 60,
                       "first_sampled_native_frame": 0, "last_sampled_native_frame": (end - 1) * 5,
                       "trailing_unsampled_native_frames": len(source.raw["xy"]) - 1 - (total - 1) * 5,
                       "sampled_source_minutes": end / 5 / 60, "requested_source_samples": total,
                       "four_actors": actors, "stats": dict(stats), "chunks": len(saved["shards"]),
                       "runtime_s": saved["runtime_s"], "actor_steps_per_s": stats["actor_steps"] / saved["runtime_s"],
                       "resume_count": saved["resumes"], "fingerprint": fingerprint}
            write_json(folder / "progress.json", metrics)
            print(json.dumps({"half": f"{match}_{half}", **metrics}), flush=True)
            completed_this_invocation += 1
            if ((stop_after_chunks is not None and completed_this_invocation >= stop_after_chunks)
                    or (output / "STOP_REQUESTED").exists() or (deadline is not None and time.time() >= deadline)):
                return {"status": "PAUSED_AT_SAVED_CHUNK_BOUNDARY", "half": f"{match}_{half}",
                        "cursor": end, "elapsed_s": time.perf_counter() - invocation_started}
        metrics = json.loads((folder / "progress.json").read_text(encoding="utf-8"))
        write_json(folder / "metrics.json", metrics)
        write_viewer(folder, source, saved, specification)
        manifests.append(metrics)
    result = {"status": "COMPLETE", "halves": manifests, "default_action_model": str(ACTION),
              "total_native_minutes": sum(m["native_duration_minutes"] for m in manifests),
              "total_actor_steps": sum(m["stats"]["actor_steps"] for m in manifests)}
    write_json(output / "manifest.json", result)
    return result


def write_viewer(folder: Path, source, saved: dict, specification: dict) -> None:
    bundle = {"schema_version": SCHEMA_VERSION, "units": "metres", "sampling_seconds": .2,
              "notice": "Bounded selections only from complete streamed halves; gaps are explicit timestamps.",
              "truth_overlay_default": False, "exo_reference_default": False,
              "model_provenance": specification["world"], "action_model_provenance": specification["action"],
              "config": specification,
              "sequences": [{"id": source.provenance["source_id"], "source": source.provenance,
                             "selection": "First/last samples and deterministic event query subset",
                             "actors": saved["viewer"], "evaluation_only": []}]}
    write_json(folder / "selected_replay.json", bundle)
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    payload = json.dumps(bundle, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (folder / "viewer.html").write_text(template.replace("__REPLAY_JSON__", payload), encoding="utf-8")


def validate(output: Path) -> dict:
    results = []
    for match, half in HALVES:
        folder = output / f"{match}_{half}"
        saved = torch.load(folder / "state.pt", map_location="cpu", weights_only=False)
        spec = json.loads((folder / "specification.json").read_text(encoding="utf-8"))
        source = SoccerTrackSource(ROOT, match, half)
        total = len(range(0, len(source.raw["xy"]), 5))
        if saved["cursor"] != total or spec["source"] != source.provenance:
            raise ValueError("Full native half was not consumed or source changed")
        cursor, records = 0, 0
        actors = spec["actors"]
        for shard in saved["shards"]:
            if shard["first_sample"] != cursor or file_digest(folder / shard["file"]) != shard["sha256"]:
                raise ValueError("Stream gap, duplicate chunk or changed artifact")
            with gzip.open(folder / shard["file"], "rt", encoding="utf-8") as reader:
                for number, line in enumerate(reader):
                    row = json.loads(line)
                    sample, actor_index = divmod(number, len(actors))
                    frame = (cursor + sample) * 5
                    if (row["frame_index"] != frame or row["actor"] != actors[actor_index]
                            or row["sample_id"] != f"{match}_{half}:{actors[actor_index]}:{frame}"):
                        raise ValueError("Duplicate, missing or out-of-order actor/time record")
                    if row["ball_state"]["possession_status"] != "unknown":
                        raise ValueError("Unobserved possession was invented")
                    records += 1
            cursor = shard["end_sample_exclusive"]
            if number + 1 != shard["records"]:
                raise ValueError("Shard record count mismatch")
        if cursor != total or records != total * 4:
            raise ValueError("Missing complete-half actor steps")
        parity = json.loads((folder / "boundary_parity.json").read_text(encoding="utf-8"))
        if parity["status"] != "PASS" or parity["maximum_numerical_difference"] != 0:
            raise ValueError("State restoration parity failed")
        results.append({"match": match, "half": half, "complete_source_samples": total,
                        "actor_steps": records, "duplicates": 0, "memory_restore_parity": "PASS",
                        "native_source_minutes": len(source.raw["xy"]) / 25 / 60,
                        "resume_count": saved["resumes"]})
    result = {"status": "PASS", "halves": results, "full_native_extent_at_5hz": True,
              "labels": "Evaluation only, off-grid annotations never rounded",
              "source_gaps": "Retained as missing observations, not trimmed"}
    write_json(output / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "validate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after-chunks", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline-utc")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")).timestamp() if args.deadline_utc else None
    result = (validate(args.output) if args.phase == "validate" else
              run(args.output, stop_after_chunks=args.stop_after_chunks, device=args.device, deadline=deadline))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
