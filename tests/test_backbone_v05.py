from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from src.backbone.actions import ActionInference
from src.backbone.batch import validate_job
from src.backbone.contracts import ActionOutput, Measurement, Observation
from src.backbone.event_time import resolve_event_time
from src.backbone.memory import CausalMemory
from src.backbone.pipeline import ReplayEngine
from src.backbone.replay_config_v05 import build_config
from src.backbone.train_action import _restore_rng, _rng_state
from src.trajectory_dataset import file_digest, write_json


def test_event_time_uses_causal_floor_and_checks_rounded_stable_frame():
    resolved = resolve_event_time(
        "1513", 39,
        clock="ms from the first frame of the half's released video",
        gsr_info={"frame_rate": 25, "clip_start": 0},
        source_frame_count=10_000,
    )

    assert resolved.status == "supported"
    assert resolved.native_zero_based_frame == 37
    assert resolved.nearest_native_frame == 38
    assert resolved.stable_frame_delta_from_nearest == 1
    assert resolved.causal_query_timestamp_ms == "1480"
    assert resolved.timing_error_ms == "33"


def test_event_time_rejects_inconsistent_stable_id_without_using_it_as_query():
    resolved = resolve_event_time(
        "1513", 44,
        clock="ms from the first frame of the half's released video",
        gsr_info={"frame_rate": 25, "clip_start": 0},
        source_frame_count=10_000,
    )

    assert resolved.status == "unsupported"
    assert resolved.reason == "stable_frame_inconsistent_with_rounded_source_time"
    assert resolved.native_zero_based_frame == 37


def test_causal_memory_keeps_a_previously_seen_query_absent_teammate():
    memory = CausalMemory(SimpleNamespace(world=None, backend="last_seen"))
    for sample in range(11):
        actor = Measurement("actor", 1, (float(sample), 0.0), 1, False)
        measurements = (actor,)
        if sample == 0:
            measurements = (*measurements, Measurement("teammate", 1, (5.0, 3.0), 1, False))
        memory.step(Observation(
            "fixture", "1st", sample * 5, sample * 0.2,
            "actor", measurements,
        ))

    inputs = memory.current_input
    assert inputs is not None
    teammate_identity = memory.entries[("teammate", 1)].identity
    teammate_slot = next(
        slot for slot, identity in inputs.identities.items() if identity == teammate_identity
    )
    assert inputs.mask[-1, teammate_slot]
    assert inputs.x[-1, teammate_slot, 4] == 0.0
    assert np.isclose(inputs.x[-1, teammate_slot, 6], 0.25)


def test_replay_runs_action_model_only_at_an_explicit_query():
    class FakeActionModel:
        def __init__(self):
            self.calls = 0

        def predict(self, inputs):
            self.calls += 1
            assert inputs is not None
            return ActionInference(
                ActionOutput(
                    action_type="PASS",
                    support_status="partial",
                    reason="Event-conditioned test inference",
                ),
                {"PASS": 1.0},
                "supported",
            )

    action_model = FakeActionModel()
    engine = ReplayEngine(
        SimpleNamespace(world=None, backend="last_seen"),
        samples=16,
        action_model=action_model,
    )
    result = None
    for sample in range(11):
        actor = Measurement("actor", 1, (float(sample), 0.0), 1, False)
        result = engine.step(
            Observation("fixture", "1st", sample * 5, sample * 0.2, "actor", (actor,)),
            include_passing_assessment=False,
            action_query=sample == 10,
        )
        if sample < 10:
            assert result["action_inference"]["status"] == "not_queried"

    assert action_model.calls == 1
    assert result is not None
    assert result["action_inference"]["status"] == "supported"
    assert result["action"]["action_type"] == "PASS"


def test_retrospective_batch_selection_uses_indexed_labels_and_stable_ids(tmp_path):
    labels = (
        ("PASS", "HIGH_PASS", "DRIVE"),
        ("CROSS", "SHOT", "HEADER"),
        ("DEFENSIVE_INTERVENTION", "PASS", "DRIVE"),
        ("PASS", "HIGH_PASS", "CROSS"),
    )
    match_halves = (
        ("118577", "1st"), ("118577", "2nd"),
        ("128058", "1st"), ("128058", "2nd"),
    )
    shards = []
    for (match, half), half_labels in zip(match_halves, labels, strict=True):
        folder = tmp_path / f"{match}_{half}"
        folder.mkdir()
        metadata_rows, target_rows = [], []
        for index, label in enumerate(half_labels):
            query_frame = 500 + index * 100
            sample_id = f"{match}_{half}:track_{index}:{query_frame}:event:{index}"
            event_id = f"{match}_p{1 if half == '1st' else 2}_f{query_frame + 1}_{index}"
            metadata_rows.append({
                "sample_id": sample_id,
                "split": "retrospective_evaluation",
                "match": match,
                "actor_track": f"track_{index}",
                "query_frame": query_frame,
            })
            target_rows.append({
                "sample_id": sample_id,
                "event_id": event_id,
                "annotation_index": index,
                "original_bas_label": label,
                "mapped_action_label": label,
                "mapping_provenance": {"source": "test"},
                "source": {"match": match, "split": "retrospective_evaluation"},
            })
        metadata_path = folder / "student_metadata.jsonl"
        target_path = folder / "optional_targets.jsonl"
        metadata_path.write_text(
            "".join(json.dumps(row) + "\n" for row in metadata_rows), encoding="utf-8",
        )
        target_path.write_text(
            "".join(json.dumps(row) + "\n" for row in target_rows), encoding="utf-8",
        )
        shards.append({
            "match": match,
            "half": half,
            "split": "retrospective_evaluation",
            "samples": 3,
            "folder": str(folder),
            "artifact_hashes": {
                metadata_path.name: file_digest(metadata_path),
                target_path.name: file_digest(target_path),
            },
        })
    index_path = tmp_path / "training_index.json"
    write_json(index_path, {
        "schema_version": "football.action-corpus.0.4",
        "status": "COMPLETE",
        "shards": shards,
    })

    config = build_config(index_path, "outputs/backbone_v0_5_20261020/training/model/final.pt")

    assert len(config["jobs"]) == 12
    selected = {
        query["mapped_action_label"]
        for job in config["jobs"]
        for query in job["action_queries"]
    }
    assert selected == {
        "PASS", "HIGH_PASS", "DRIVE", "CROSS", "SHOT", "HEADER", "DEFENSIVE_INTERVENTION",
    }
    assert all(job["samples"] == 11 for job in config["jobs"])
    validate_job(config["jobs"][0])
    invalid = dict(config["jobs"][0])
    invalid["action_queries"] = [dict(invalid["action_queries"][0], frame_index=999)]
    with pytest.raises(ValueError, match="exactly sampled"):
        validate_job(invalid)


def test_cuda_checkpoint_rng_state_restores_from_mapped_device():
    state = _rng_state()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this test environment")

    state["torch"] = state["torch"].to("cuda")
    state["cuda"] = [value.to("cuda") for value in state["cuda"]]

    _restore_rng(state)

    assert torch.get_rng_state().device.type == "cpu"
