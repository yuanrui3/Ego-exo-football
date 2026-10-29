"""Focused acceptance tests for the checkpoint-backed v0.3 action adapter."""

import json
from pathlib import Path

import numpy as np
import torch

from src.backbone.actions import ActionModelSession
from src.backbone.contracts import WorldInput
from src.backbone.v0_3 import (
    CHECKPOINT,
    ROOT,
    TRAINING_INDEX,
    build_bundle,
    event_candidates,
    indexed_sources,
)
from src.player_centric_dataset import ACTION_LABELS


def make_world_input(seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .1, size=(11, 22, 10)).astype(np.float32)
    mask = np.ones((11, 22), dtype=bool)
    x[..., 4] = 1
    x[..., 5] = 1
    x[..., 6] = rng.uniform(0, .25, size=(11, 22))
    x[..., 9] = 0
    x[:, 3, 9] = 1
    x[:, 3, 4] = 1
    return WorldInput(x, mask, np.zeros(2, np.float32), np.zeros((22, 2), np.float32), {})


def test_checkpoint_wrapper_matches_original_s4_forward():
    torch.set_num_threads(2)
    session = ActionModelSession(CHECKPOINT)
    inputs = make_world_input()
    actual = session.predict(inputs)

    model_x = inputs.x.copy()
    model_x[..., 6] = np.clip(model_x[..., 6] * (8.0 / 5.0), 0, 1)
    with torch.inference_mode():
        expected_logits = session.model(
            torch.from_numpy(model_x[None]),
            torch.from_numpy(inputs.mask[None]),
            torch.from_numpy((model_x[..., 4] > .5)[None]),
            memory_mode="memory",
        )["action"][0]
        expected_scores = expected_logits.float().softmax(-1).numpy()
    assert actual.status == "supported"
    assert actual.output.confidence is None
    assert session.load_count == 1
    np.testing.assert_allclose(
        [actual.scores[label] for label in ACTION_LABELS],
        expected_scores,
        atol=1e-7,
        rtol=1e-6,
    )


def test_incompatible_inputs_are_explicitly_unsupported():
    session = ActionModelSession(CHECKPOINT)
    invalid = WorldInput(
        np.zeros((10, 22, 10), np.float32),
        np.zeros((10, 22), bool),
        np.zeros(2, np.float32),
        np.zeros((22, 2), np.float32),
        {},
    )
    result = session.predict(invalid)
    assert result.status == "unsupported"
    assert result.output.support_status == "unsupported"
    assert "expected x (11, 22, 10)" in result.reason
    assert session.predict(None).output.support_status == "unsupported"


def test_stable_event_id_joins_to_annotation_index_and_mapping_provenance():
    sources, _ = indexed_sources(TRAINING_INDEX, {
        "117092", "117093", "118576", "118578", "128057", "132831", "132877",
    })
    candidates, _ = event_candidates(ROOT / "data/train_temporal_7class.jsonl", ROOT, sources)
    example = next(row for row in candidates if row["event_id"] == "117092_p1_f1022_10")
    bas = json.loads((ROOT / "data/117092_12_class_events.json").read_text())["actions"]
    assert example["annotation_index"] == 10
    assert example["event"] == bas[10]
    assert example["source_label"] == "PASS"
    assert example["mapped_label"] == "PASS"
    assert example["anchor_frame"] == 1021


def test_end_to_end_bundle_keeps_labels_and_replay_views_separate(tmp_path):
    torch.set_num_threads(2)
    bundle = tmp_path / "backbone_v0_3"
    manifest = build_bundle(bundle)
    targets = [json.loads(line) for line in (bundle / "optional_targets.jsonl").read_text().splitlines()]
    students = [json.loads(line) for line in (bundle / "student.jsonl").read_text().splitlines()]
    replay = json.loads((bundle / "mixed_action_replay.json").read_text())

    assert len(targets) >= 3
    assert sum(row["mapped_action_label"] != "PASS" for row in targets) >= 2
    assert manifest["selection"]["examples"] == len(targets) == len(students)
    assert {
        source_id.split("_", 1)[0]
        for shard in manifest["indexed_training_sources"]["shards"]
        for source_id in shard["source_ids"]
    } == {"117092", "117093"}
    assert {row["source_provenance"]["match"] for row in targets} <= {"117092", "117093"}
    assert not {row["source_provenance"]["match"] for row in targets} & {"118577", "128058"}

    for student, target in zip(students, targets):
        assert target["sample_id"] == student["sample_id"]
        assert "mapped_action_label" not in student
        assert student["target_availability"]["action"] is False
        timing = target["timing"]
        assert timing["history_frame_indices"][-1] == timing["query_anchor_frame"]
        assert all(b - a == 5 for a, b in zip(
            timing["history_frame_indices"], timing["history_frame_indices"][1:]
        ))
        assert max(timing["history_frame_indices"]) == timing["query_anchor_frame"]

    for example in replay["examples"]:
        assert example["player_facing_prediction"]["action_output"]["support_status"] == "partial"
        assert example["player_facing_prediction"]["action_output"]["confidence"] is None
        assert example["player_facing_prediction"]["model_scores"]
        assert example["exo_reference_prediction"]["status"] == "UNAVAILABLE"
        assert example["recorded_annotation_evaluation_only"]["event_id"] == example["query"]["event_id"]
        assert example["ball_state"]["possession_status"] == "unknown"
        assert example["unimplemented_action_branches"]["status"] == "UNSUPPORTED"
        assert "mapped_action_label" not in example["player_facing_prediction"]["action_output"]

    index = json.loads(TRAINING_INDEX.read_text())
    for shard in index["shards"]:
        optional = Path(shard["folder"]) / "optional_targets.jsonl"
        old_rows = [json.loads(line) for line in optional.read_text().splitlines()]
        assert old_rows and all(row["event_target"] is None for row in old_rows)
        assert all(row["target_availability"]["action"] is False for row in old_rows)
        shard_manifest = next(
            item for item in manifest["indexed_training_sources"]["shards"]
            if item["job"] == shard["job"]
        )
        from src.trajectory_dataset import file_digest
        assert file_digest(optional) == shard_manifest["existing_optional_targets_sha256"]


def test_prediction_does_not_mutate_or_depend_on_external_targets():
    session = ActionModelSession(CHECKPOINT)
    inputs = make_world_input(9)
    before = inputs.x.copy()
    first = session.predict(inputs)
    annotation = {"mapped_action_label": "SHOT", "hidden_xy": np.full((22, 2), 1e6)}
    changed_reference_only = WorldInput(
        inputs.x.copy(),
        inputs.mask.copy(),
        np.full(2, -1e9, np.float32),
        np.full((22, 2), 1e9, np.float32),
        {"0": "different-reference-identity"},
    )
    second = session.predict(changed_reference_only)
    assert annotation["mapped_action_label"] == "SHOT"
    assert first.scores == second.scores
    np.testing.assert_array_equal(inputs.x, before)
