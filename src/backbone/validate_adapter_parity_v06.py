"""Compare cache and external-JSONL replay on the same chronological observations."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

from src.backbone.external import ExternalObservationSource
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.source import SoccerTrackSource
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]


def _compare(actual: object, expected: object, path: str, max_error: list[float]) -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        if actual is not expected:
            raise ValueError(f"Source-adapter parity mismatch at {path}")
        return
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        difference = abs(float(actual) - float(expected))
        max_error[0] = max(max_error[0], difference)
        if not math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=2e-5):
            raise ValueError(f"Source-adapter numeric mismatch at {path}: {actual} vs {expected}")
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        ignore = {"latency_ms", "warnings"}
        actual_keys, expected_keys = set(actual) - ignore, set(expected) - ignore
        if actual_keys != expected_keys:
            raise ValueError(f"Source-adapter output keys differ at {path}")
        for key in actual_keys:
            _compare(actual[key], expected[key], f"{path}.{key}", max_error)
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Source-adapter list size differs at {path}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _compare(left, right, f"{path}[{index}]", max_error)
        return
    if actual != expected:
        raise ValueError(f"Source-adapter value differs at {path}")


def validate(
    *,
    match: str,
    half: str,
    start_frame: int,
    samples: int,
    actors: list[str],
    cache_batch_summary: Path,
    external_batch_summary: Path,
    external_jsonl: Path,
    output_path: Path,
) -> dict:
    cache_batch_summary = cache_batch_summary.resolve()
    external_batch_summary = external_batch_summary.resolve()
    external_jsonl = external_jsonl.resolve()
    cache_summary = json.loads(cache_batch_summary.read_text(encoding="utf-8"))
    external_summary = json.loads(external_batch_summary.read_text(encoding="utf-8"))
    if cache_summary["status"] != "COMPLETE" or external_summary["status"] != "COMPLETE":
        raise ValueError("Both source-adapter replays must be COMPLETE")
    cache_jobs = {job["id"]: job for job in cache_summary["jobs"]}
    external_jobs = {job["id"]: job for job in external_summary["jobs"]}
    cache_job = cache_jobs.get("sustained_pilot_118577_1st")
    external_job = external_jobs.get("sustained_pilot_external")
    if not cache_job or not external_job or cache_job["status"] != "COMPLETE" or external_job["status"] != "COMPLETE":
        raise ValueError("Expected completed SoccerTrack and JSONL replay jobs")
    cache_replay_path = Path(cache_job["output"]) / "replay.json"
    external_replay_path = Path(external_job["output"]) / "replay.json"
    cache_replay = json.loads(cache_replay_path.read_text(encoding="utf-8"))
    external_replay = json.loads(external_replay_path.read_text(encoding="utf-8"))
    cache_sequence, external_sequence = cache_replay["sequences"][0], external_replay["sequences"][0]
    if list(cache_sequence["actors"]) != actors or list(external_sequence["actors"]) != actors:
        raise ValueError("Actor order differs between source adapters")

    cache_source = SoccerTrackSource(ROOT, match, half)
    external_source = ExternalObservationSource(external_jsonl)
    cache_frames = list(cache_source.frames(start_frame, samples))
    external_frames = list(external_source.frames(start_frame, samples))
    if len(cache_frames) != samples or len(external_frames) != samples:
        raise ValueError("Source adapters did not supply the full requested sample count")
    released_observation_comparisons = 0
    for actor in actors:
        cache_provider = SyntheticObservationProvider(actor, seed=42, fov=120, dropout=False)
        external_provider = SyntheticObservationProvider(actor, seed=42, fov=120, dropout=False)
        for source_frame, external_frame in zip(cache_frames, external_frames, strict=True):
            if (source_frame.index != external_frame.index
                    or source_frame.timestamp_s != external_frame.timestamp_s
                    or source_frame.measurements != external_frame.measurements):
                raise ValueError(f"Released source frame differs at {actor}:{source_frame.index}")
            left = cache_provider.observe(source_frame)
            right = external_source.observe(external_frame, external_provider)
            left_record = dataclasses.asdict(left)
            right_record = dataclasses.asdict(right)
            left_record.pop("warnings", None)
            right_record.pop("warnings", None)
            if left_record != right_record:
                raise ValueError(f"Released player-view observations differ at {actor}:{source_frame.index}")
            released_observation_comparisons += 1

    max_error = [0.0]
    prediction_digests = {}
    for actor in actors:
        left_rows = cache_sequence["actors"][actor]
        right_rows = external_sequence["actors"][actor]
        if len(left_rows) != samples or len(right_rows) != samples:
            raise ValueError(f"Replay rows incomplete for {actor}")
        digest = hashlib.sha256()
        for index, (left, right) in enumerate(zip(left_rows, right_rows, strict=True)):
            if left["frame_index"] != right["frame_index"] or left["timestamp_s"] != right["timestamp_s"]:
                raise ValueError(f"Replay time grid differs for {actor} at row {index}")
            for key in ("belief", "assessment", "action", "action_inference", "ball_state",
                        "restart_context", "outcome", "modules"):
                _compare(left.get(key), right.get(key), f"{actor}[{index}].{key}", max_error)
            digest.update(_canonical_prediction(right))
        prediction_digests[actor] = digest.hexdigest()
    cache_metrics = cache_job["metrics"]
    external_metrics = external_job["metrics"]
    cache_action_calls = sum(
        row.get("action_inference", {}).get("status") == "supported"
        for rows in cache_sequence["actors"].values() for row in rows
    )
    cache_call_counts = {
        "world_model_forward_calls": cache_metrics.get(
            "world_model_forward_calls", cache_metrics["world_supported_frames"],
        ),
        "world_model_forward_batch_elements": cache_metrics.get(
            "world_model_forward_batch_elements", cache_metrics["world_supported_frames"],
        ),
        "action_model_forward_calls": cache_metrics.get(
            "action_model_forward_calls", cache_action_calls,
        ),
        "action_model_forward_batch_elements": cache_metrics.get(
            "action_model_forward_batch_elements", cache_action_calls,
        ),
    }
    for name, expected in cache_call_counts.items():
        if expected != external_metrics.get(name):
            raise ValueError(f"Forward-use counters differ between adapters: {name}")
    result = {
        "schema_version": "football.source-adapter-parity.0.6",
        "status": "PASS",
        "claim_scope": "Cache-to-JSONL interoperability on an identical annotation-backed tracking observation stream; not video-only inference or independent generalization.",
        "source": {
            "match": match,
            "half": half,
            "start_frame": start_frame,
            "samples": samples,
            "unique_source_frames": len(cache_frames),
            "actors": actors,
            "source_cache_sha256": cache_source.provenance["cache_sha256"],
            "external_jsonl": str(external_jsonl),
            "external_jsonl_sha256": file_digest(external_jsonl),
            "external_source_metadata": external_source.provenance,
        },
        "identical_released_observation_comparisons": released_observation_comparisons,
        "prediction_rows_compared": samples * len(actors),
        "max_numeric_absolute_difference": max_error[0],
        "model_use_counters": {
            name: {
                "cache_adapter": cache_call_counts[name],
                "external_adapter": external_metrics[name],
                "cache_evidence": "Instrumented counter" if name in cache_metrics
                else "Reconciled from supported 1-item model-contract inputs / supported scheduled query rows",
            }
            for name in cache_call_counts
        },
        "actor_prediction_stream_sha256": prediction_digests,
        "cache_replay_sha256": file_digest(cache_replay_path),
        "external_replay_sha256": file_digest(external_replay_path),
        "tolerance": {"relative": 1e-5, "absolute": 2e-5},
        "decision": "PASS_AND_ADVANCE",
    }
    write_json(output_path.resolve(), result)
    return result


def _canonical_prediction(row: dict) -> bytes:
    stable = {key: row.get(key) for key in (
        "frame_index", "timestamp_s", "belief", "assessment", "action", "action_inference",
        "ball_state", "restart_context", "outcome", "modules",
    )}
    stable["belief"].pop("warnings", None)
    return json.dumps(stable, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True)
    parser.add_argument("--half", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--actors", nargs="+", required=True)
    parser.add_argument("--cache-summary", type=Path, required=True)
    parser.add_argument("--external-summary", type=Path, required=True)
    parser.add_argument("--external-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        match=args.match,
        half=args.half,
        start_frame=args.start_frame,
        samples=args.samples,
        actors=args.actors,
        cache_batch_summary=args.cache_summary if args.cache_summary.is_absolute()
        else ROOT / args.cache_summary,
        external_batch_summary=args.external_summary if args.external_summary.is_absolute()
        else ROOT / args.external_summary,
        external_jsonl=args.external_jsonl if args.external_jsonl.is_absolute()
        else ROOT / args.external_jsonl,
        output_path=args.output if args.output.is_absolute() else ROOT / args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "released_observation_comparisons": result["identical_released_observation_comparisons"],
        "prediction_rows_compared": result["prediction_rows_compared"],
        "max_numeric_absolute_difference": result["max_numeric_absolute_difference"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
