"""Acceptance checks for the split-safe v0.4 action integration."""

import json
import os
from pathlib import Path

from src.backbone.train_action import load_split
from src.backbone.video_tracklets import associate_tracklets, pixel_to_normalized


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "BACKBONE_V0_4_OUTPUT", str(ROOT / "outputs" / "backbone_v0_4_20261020"),
))
INDEX = OUTPUT / "training_index.json"
TRAINING = OUTPUT / "training"
REPLAY = OUTPUT / "replay"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_indexed_splits_are_disjoint_and_labels_stay_out_of_student_inputs():
    expected = {
        "fitting": {"117092", "117093", "118576", "118578", "128057", "132831"},
        "development": {"132877"},
        "retrospective_evaluation": {"118577", "128058"},
    }
    loaded = {split: load_split(INDEX, split) for split in expected}
    assert all(set(data["matches"]) <= expected[split] for split, data in loaded.items())
    assert set(loaded["development"]["matches"]) == {"132877"}
    assert set(loaded["retrospective_evaluation"]["matches"]) == {"118577", "128058"}
    assert not (
        set(loaded["fitting"]["matches"]) & set(loaded["development"]["matches"])
        or set(loaded["fitting"]["matches"]) & set(loaded["retrospective_evaluation"]["matches"])
    )

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    shard = next(row for row in index["shards"] if row["split"] == "fitting" and row["samples"])
    folder = Path(shard["folder"])
    students = _jsonl(folder / "student_metadata.jsonl")
    targets = _jsonl(folder / "optional_targets.jsonl")
    student_by_id = {row["sample_id"]: row for row in students}
    target_by_id = {row["sample_id"]: row for row in targets}
    assert set(student_by_id) == set(target_by_id)
    sample_id = next(iter(student_by_id))
    student = student_by_id[sample_id]
    target = target_by_id[sample_id]
    assert student["target_availability"]["action"] is False
    assert "mapped_action_label" not in student
    assert target["sample_id"] == sample_id
    assert target["source"]["split"] == "fitting"
    assert max(student["input_frame_indices"]) == student["query_frame"]


def test_unlabeled_windows_remain_unlabeled():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    folder = Path(index["unlabeled"]["folder"])
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    rows = _jsonl(folder / "metadata.jsonl")
    assert summary["windows"] == len(rows) > 0
    assert all(row["target_availability"]["action"] is False for row in rows)
    assert all("not a negative" in row["target_availability"]["meaning"].lower() for row in rows)
    assert all(max(row["input_frame_indices"]) == row["query_frame"] for row in rows)


def test_training_checkpoint_adapter_parity_and_shared_replay_separation():
    evaluation = json.loads((TRAINING / "retrospective_evaluation.json").read_text(encoding="utf-8"))
    parity = evaluation["reloaded_checkpoint_prediction_parity"]
    assert evaluation["status"] == "COMPLETE"
    assert parity["status"] == "PASS"
    assert parity["maximum_absolute_probability_difference"] <= 1e-6

    replay = json.loads((REPLAY / "shared_replay.json").read_text(encoding="utf-8"))
    assert replay["status"] == "COMPLETE"
    assert len(replay["player_facing_event_queries"]) >= 3
    assert sum(
        row["recorded_annotation_evaluation_only"]["mapped_action_label"] != "PASS"
        for row in replay["player_facing_event_queries"]
    ) >= 2
    assert len(replay["exploratory_unlabeled_queries"]) >= 1
    for row in replay["player_facing_event_queries"]:
        assert row["player_facing_prediction"]["status"] == "supported"
        assert row["player_facing_prediction"]["scores"]
        assert "mapped_action_label" not in row["player_facing_prediction"]["output"]
        assert row["ball_state"]["possession_status"] == "unknown"
        assert row["recorded_annotation_evaluation_only"]
        assert row["optional_passing_geometry"]["status"] in ("AVAILABLE", "UNAVAILABLE")
    for row in replay["exploratory_unlabeled_queries"]:
        assert row["recorded_annotation_evaluation_only"] is None
        assert row["query"]["conditioning"].startswith("exploratory arbitrary")


def test_pixel_candidate_association_stays_within_shot_and_conversion_is_image_only():
    records = [
        {
            "frame_index": frame,
            "time_from_video_start_s": frame / 25,
            "detections": [{
                "bbox_xyxy_px": [10 + offset, 20, 30 + offset, 60],
                "detector_confidence": 0.8,
                "tile": 0,
            }],
        }
        for frame, offset in ((1500, 0), (1525, 4), (1550, 8))
    ]
    linked, tracklets = associate_tracklets(records, [False, False, True])
    assert linked[0]["detections"][0]["tracklet_id"] == linked[1]["detections"][0]["tracklet_id"]
    assert linked[2]["detections"][0]["tracklet_id"] != linked[1]["detections"][0]["tracklet_id"]
    assert all(track["status"] == "UNVERIFIED_IMAGE_SPACE_CANDIDATE" for track in tracklets)
    assert pixel_to_normalized(1920, 953, 3840, 1906) == (0.5, 0.5)


def test_video_extension_keeps_pitch_adapter_blocked_and_emits_overlays():
    extension = OUTPUT / "video_pilot_extension"
    manifest = json.loads((extension / "manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads((extension / "pixel_calibration_sidecar.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PIXEL_TRACKLET_CANDIDATES_READY_PITCH_ADAPTER_BLOCKED"
    assert manifest["frames"] == 12
    assert manifest["detections"] == 25
    assert manifest["detector_rerun"] is False
    assert manifest["no_new_downloads"] is True
    assert manifest["identity_claim"] == "NONE; image-space candidate tracklets only."
    assert manifest["overlay_paths"]
    assert all((extension / name).is_file() for name in manifest["overlay_paths"])
    assert calibration["eligible_for_pitch_backbone"] is False
    assert calibration["pitch_coordinates"]["transform"] is None
