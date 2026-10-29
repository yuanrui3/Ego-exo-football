"""Development-only causal failure checks, including identity changes after memory formation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from src.backbone.actions import ActionModelSession
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import SoccerTrackSource
from src.backbone.streaming_completion import ACTION, INDEX, ROOT, compact
from src.trajectory_dataset import file_digest, write_json


CONDITIONS = ("baseline", "short_source_gap", "long_source_gap", "position_noise",
              "unknown_team_role", "camera_discontinuity", "identity_after_memory")


def identity_target(observations: tuple) -> str | None:
    return next((m.track for m in observations[50].measurements
                 if m.track != observations[50].actor_track
                 and any(old.track == m.track for obs in observations[30:50] for old in obs.measurements)
                 and any(old.track == m.track for obs in observations[60:70] for old in obs.measurements)), None)


def transform(observations: tuple, condition: str) -> tuple:
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported stress condition: {condition}")
    target = identity_target(observations)
    result = []
    for index, observation in enumerate(observations):
        measurements = observation.measurements
        if ((condition == "short_source_gap" and 98 <= index <= 100)
                or (condition == "long_source_gap" and 87 <= index <= 100)):
            measurements = ()
        if condition == "position_noise":
            rng = np.random.default_rng(42 + index)
            measurements = tuple(replace(m, xy=tuple(np.asarray(m.xy) + rng.normal(0, 1, 2)))
                                 for m in measurements)
        if condition == "unknown_team_role" and 87 <= index <= 110:
            measurements = tuple(replace(m, team=0, goalkeeper=None) for m in measurements)
        if condition == "identity_after_memory" and index >= 60 and target is not None:
            measurements = tuple(replace(m, segment=m.segment + 100000) if m.track == target else m
                                 for m in measurements)
        result.append(replace(observation, measurements=measurements,
                              discontinuity=condition == "camera_discontinuity" and index == 60))
    if condition == "identity_after_memory" and target is None:
        raise ValueError("Fixed development sample lacks a previously observed non-actor identity")
    return tuple(result)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Preserve development stress audit")
    torch.set_num_threads(4)
    models = ModelSession(ROOT / "outputs" / "dynamic_gnn_tactical_20261019", device="cpu")
    action = ActionModelSession(ACTION, device="cpu")
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    selected, sources, exclusions = {}, {}, Counter()
    for shard in sorted(index["shards"], key=lambda row: (row["match"], row["half"])):
        if shard["split"] != "development":
            continue
        folder = Path(shard["folder"])
        for filename in ("student_metadata.jsonl", "optional_targets.jsonl"):
            if file_digest(folder / filename) != shard["artifact_hashes"][filename]:
                raise ValueError("Development metadata changed")
        targets = {row["sample_id"]: row for row in
                   map(json.loads, (folder / "optional_targets.jsonl").read_text(encoding="utf-8").splitlines())}
        for row in map(json.loads, (folder / "student_metadata.jsonl").read_text(encoding="utf-8").splitlines()):
            if row["query_frame"] >= 500:
                label = targets[row["sample_id"]]["mapped_action_label"]
                if label in selected:
                    continue
                key = row["match"], row["half"]
                if key not in sources:
                    sources[key] = SoccerTrackSource(ROOT, *key)
                provider = SyntheticObservationProvider(row["actor_track"])
                stream = tuple(provider.observe(frame) for frame in
                               sources[key].frames(int(row["query_frame"]) - 500, 121))
                if identity_target(stream) is None:
                    exclusions["no_nonactor_observed_before_and_after_identity_boundary"] += 1
                    continue
                selected[label] = (row, targets[row["sample_id"]], stream)
    if len(selected) != 7:
        raise ValueError("Fixed development selection does not cover seven trained classes")
    summaries = []
    for label, (metadata, annotation, observations) in sorted(selected.items()):
        anchor = int(metadata["query_frame"])
        for condition in CONDITIONS:
            stream = transform(observations, condition)
            engine = ReplayEngine(models, action_model=action)
            stats = Counter()
            query, boundary = None, None
            rows = []
            for number, observation in enumerate(stream):
                if number == 60:
                    boundary = {"history_before": len(engine.memory.history),
                                "entries_before": len(engine.memory.entries)}
                row = engine.step(observation, include_passing_assessment=False, action_query=number == 100)
                if number == 60:
                    boundary.update(history_after=len(engine.memory.history), warnings=row["belief"]["warnings"])
                if number == 100:
                    query = compact(row)
                stats["actor_missing"] += int(observation.actor is None)
                stats["supported_world_inputs"] += int(row["belief"]["model_contract_met"])
                stats["unsupported_action_queries"] += int(number == 100 and row["action"]["support_status"] == "unsupported")
                if number >= 55:
                    rows.append(compact(row))
            if query is None or query["frame_index"] != anchor:
                raise ValueError("Stress action query changed annotated timing")
            if condition in ("short_source_gap", "long_source_gap", "unknown_team_role"):
                if query["action"]["support_status"] != "unsupported":
                    raise ValueError("Severely unsupported input produced an action claim")
            if condition == "camera_discontinuity" and boundary["history_after"] != 1:
                raise ValueError("Camera discontinuity failed to reset history")
            if condition == "identity_after_memory" and not any(
                    "OBSERVED_IDENTITY_DISCONTINUITY" in warning
                    for row in rows for warning in row["belief"]["warnings"]):
                raise ValueError("Post-memory identity break was not exercised")
            if query["ball_state"]["possession_status"] != "unknown":
                raise ValueError("Stress runner invented possession")
            record = {"sample_id": metadata["sample_id"], "label_evaluation_only": label,
                      "annotation_evaluation_only": annotation, "condition": condition,
                      "query": query, "stats": dict(stats), "boundary": boundary,
                      "temporal_rows": rows}
            write_json(output / f"{label}_{condition}.json", record)
            summaries.append({key: value for key, value in record.items()
                              if key not in ("temporal_rows", "annotation_evaluation_only")})
    result = {"status": "PASS", "conditions": list(CONDITIONS), "development_match": "132877",
              "selection": "First indexed eligible development example per trained class, query>=500, with a non-actor observed before/after the identity boundary; no prediction-based selection",
              "explicit_selection_exclusions": dict(exclusions),
              "streams": len(summaries), "actor_steps": len(summaries) * 121,
              "default_and_selected": "Same original S4; no automatic specialist switching",
              "conditions_units": "Source gaps 0.6s/2.8s ending at query; observed-coordinate Gaussian sigma1m; identity/camera breaks after12s memory",
              "events_only": "Exactly one original annotated query per stream; later observations never enter that prediction",
              "summaries": summaries, "source_provenance": {f"{m}_{h}": s.provenance for (m, h), s in sources.items()}}
    write_json(output / "result.json", result)
    return {"status": result["status"], "streams": result["streams"], "actor_steps": result["actor_steps"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
