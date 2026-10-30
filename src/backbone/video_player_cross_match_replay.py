"""Build experimental external observations and replay the frozen G1/S4 backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import ijson
import numpy as np
import torch

from src.backbone.video_player_cross_match import (
    ROOT,
    _maximum_matches,
    interval_by_name,
    load_local_projector,
    load_protocol,
    prediction_path,
    project_footpoint,
    source_by_id,
    validate_final_prediction_set,
)
from src.backbone.external import ExternalObservationSource, VERSION as EXTERNAL_SCHEMA_VERSION
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.actions import ActionModelSession
from src.backbone.models import ModelSession
from src.trajectory_dataset import file_digest, write_json


ATTRIBUTE_MODEL = (
    ROOT / "outputs" / "overnight_video_player_20261028"
    / "attribute_model" / "appearance_knn.npz"
)
ATTRIBUTE_PROVENANCE = ATTRIBUTE_MODEL.with_name(
    "appearance_classifier_provenance.json",
)
PERSON_WEIGHTS = ROOT / "yolo11n.pt"
SAMPLE_STRIDE = 5
NATIVE_FPS = 25
DETECTION_CONFIDENCE = 0.20
APPEARANCE_CONFIDENCE = 0.05
ATTRIBUTE_MATCH_TOLERANCE_PX = 80.0
VISUAL_ANCHOR_SAMPLES = 100
S4_QUERY_SECONDS = (20, 40, 60, 80, 100)


def external_frames(source: ExternalObservationSource, start: int, count: int):
    """Batch adapter reads without resetting replay engines or runtime IDs."""
    for offset in range(0, count, 1500):
        yield from source.frames(start + offset * SAMPLE_STRIDE,
                                 min(1500, count - offset), stride=SAMPLE_STRIDE)


def _read_predictions(path: Path, expected_count: int) -> tuple[list[dict], dict]:
    metadata_path = path.with_suffix(".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Prediction and sidecar required: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("prediction_jsonl_sha256") != file_digest(path):
        raise ValueError(f"Prediction hash does not match frozen sidecar: {path}")
    with path.open(encoding="utf-8") as stream:
        frames = [json.loads(line) for line in stream if line.strip()]
    if len(frames) != expected_count:
        raise ValueError(f"Expected {expected_count} prediction rows, found {len(frames)}")
    return frames, metadata


def _read_video_sample(
    capture: cv2.VideoCapture,
    frame_index: int,
) -> np.ndarray:
    ok, image = capture.read()
    if not ok or image is None:
        raise RuntimeError(f"Could not decode frame {frame_index} for appearance crops")
    for _ in range(SAMPLE_STRIDE - 1):
        if not capture.grab():
            raise RuntimeError(f"Could not advance past sampled frame {frame_index}")
    return image


def _read_attribute_annotations(source: dict, frame_indices: set[int]) -> dict[int, list[dict]]:
    annotation_path = ROOT / source["annotation"]
    image_zero = int(source["image_id_frame_zero"])
    labels: dict[int, list[dict]] = {}
    last_frame = -1
    with annotation_path.open("rb") as stream:
        for item in ijson.items(stream, "annotations.item"):
            try:
                frame_index = int(item["image_id"]) - image_zero
            except (KeyError, TypeError, ValueError):
                continue
            if frame_index < last_frame:
                raise ValueError(f"Annotation frames are not ordered: {annotation_path}")
            last_frame = frame_index
            if frame_index > max(frame_indices):
                break
            if frame_index not in frame_indices or item.get("category_id") not in (1, 2):
                continue
            box = item.get("bbox_image") or {}
            attrs = item.get("attributes") or {}
            if not all(key in box for key in ("x", "y", "w", "h")):
                continue
            x, y, width, height = (float(box[key]) for key in ("x", "y", "w", "h"))
            labels.setdefault(frame_index, []).append({
                "footpoint_xy_px": [x + width / 2.0, y + height],
                "team": str(attrs.get("team", "UNKNOWN")),
                "role": str(attrs.get("role", "UNKNOWN")),
            })
    return labels


def _attribute_model():
    if not ATTRIBUTE_MODEL.is_file() or not ATTRIBUTE_PROVENANCE.is_file():
        raise FileNotFoundError("The frozen source-trained appearance classifier is missing")
    archive = np.load(ATTRIBUTE_MODEL, allow_pickle=False)
    metadata = json.loads(ATTRIBUTE_PROVENANCE.read_text(encoding="utf-8"))
    return (
        archive["features"].astype(np.float32),
        archive["teams"].astype(str).tolist(),
        archive["roles"].astype(str).tolist(),
        metadata,
    )


def build_external_observations(
    protocol: dict,
    output: Path,
    match_id: str,
    interval_name: str,
    detector_route: str,
    *,
    assisted_attributes: bool,
    device: str = "cpu",
) -> dict:
    source = source_by_id(protocol, match_id)
    start_s, end_s = interval_by_name(protocol, match_id, interval_name)
    count = (end_s - start_s) * (NATIVE_FPS // SAMPLE_STRIDE)
    predictions_path = prediction_path(
        output, detector_route, match_id, source["half"], interval_name,
    )
    frames, prediction_meta = _read_predictions(predictions_path, count)
    if (
        prediction_meta.get("route") != detector_route
        or prediction_meta.get("match_id") != match_id
        or prediction_meta.get("interval") != interval_name
        or prediction_meta.get("source_video_sha256") != source["video_sha256"]
    ):
        raise ValueError(f"Prediction metadata does not match the requested segment: {predictions_path}")
    expected_frame_indices = [
        start_s * NATIVE_FPS + index * SAMPLE_STRIDE
        for index in range(count)
    ]
    if [int(frame["frame_index"]) for frame in frames] != expected_frame_indices:
        raise ValueError("Prediction frames do not match the locked source timestamps")
    if file_digest(ROOT / source["video"]) != source["video_sha256"]:
        raise ValueError("Source video no longer matches the locked protocol hash")
    if interval_name.startswith("final"):
        final_set = validate_final_prediction_set(protocol, output)
        if final_set["status"] != "ALL_FINAL_ROUTES_AND_INTERVALS_FROZEN":
            raise RuntimeError("Final predictions are not sealed")
    else:
        final_set = None

    projector = load_local_projector(protocol, output, match_id)
    frame_indices = {int(frame["frame_index"]) for frame in frames}
    if assisted_attributes and file_digest(ROOT / source["annotation"]) != source["annotation_sha256"]:
        raise ValueError("Assisted-attribute annotation source differs from the locked hash")
    annotations = (
        _read_attribute_annotations(source, frame_indices)
        if assisted_attributes else {}
    )

    train_features = train_teams = train_roles = attribute_metadata = None
    person_model = None
    capture = None
    if not assisted_attributes:
        from ultralytics import YOLO

        from src.backbone.video_player_attributes import (
            ROLE_CLASSES,
            TEAM_CLASSES,
            _knn_scores,
            _maximum_footpoint_matches,
            appearance_feature,
        )
        from src.backbone.video_player_overnight import detect_native

        train_features, train_teams, train_roles, attribute_metadata = _attribute_model()
        if not PERSON_WEIGHTS.is_file():
            raise FileNotFoundError(f"Appearance crop detector is missing: {PERSON_WEIGHTS}")
        person_model = YOLO(str(PERSON_WEIGHTS))
        capture = cv2.VideoCapture(str(ROOT / source["video"]), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source video for appearance crops: {source['video']}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_s * NATIVE_FPS)
    else:
        from src.backbone.video_player_attributes import ROLE_CLASSES

    rows = []
    counts = {
        "video_detections": 0,
        "projected_experimental_positions": 0,
        "unprojected_detections": 0,
        "appearance_crop_matches": 0,
        "attribute_assisted_matches": 0,
        "known_team": 0,
        "unknown_team": 0,
        "known_role": 0,
        "unknown_role": 0,
    }
    try:
        for frame in frames:
            image = None
            appearance_boxes = []
            if capture is not None and person_model is not None:
                image = _read_video_sample(capture, int(frame["frame_index"]))
                if image.shape[1::-1] != tuple(source["dimensions_px"]):
                    raise ValueError("Appearance crop frame dimensions differ from locked source")
                if any(
                    project_footpoint(item["footpoint_xy_px"], projector)[0] is not None
                    for item in frame["detections"]
                ):
                    appearance_boxes = detect_native(
                        person_model,
                        image,
                        confidence=APPEARANCE_CONFIDENCE,
                        device=device,
                    )

            projected_items = []
            for detection_index, detection in enumerate(frame["detections"]):
                pitch, projection_status = project_footpoint(
                    detection["footpoint_xy_px"], projector,
                )
                projected_items.append({
                    "detection_index": detection_index,
                    "detection": detection,
                    "pitch": pitch,
                    "projection_status": projection_status,
                    "team": 0,
                    "team_confidence": None,
                    "team_prediction": "UNKNOWN",
                    "goalkeeper": None,
                    "role_confidence": None,
                    "role_prediction": "UNKNOWN",
                    "attribute_source": "UNKNOWN",
                    "attribute_match_distance_px": None,
                })

            if assisted_attributes:
                truth = annotations.get(int(frame["frame_index"]), [])
                matches = _maximum_matches(
                    [
                        {"footpoint_xy_px": item["detection"]["footpoint_xy_px"]}
                        for item in projected_items
                    ],
                    truth,
                    40.0 * source["dimensions_px"][0] / 4096.0,
                )
                for detection_index, annotation_index, distance in matches:
                    item = projected_items[detection_index]
                    annotation = truth[annotation_index]
                    item["team"] = {"left": 1, "right": 2}.get(annotation["team"], 0)
                    item["team_prediction"] = annotation["team"]
                    item["goalkeeper"] = (
                        annotation["role"] == "goalkeeper"
                        if annotation["role"] in ROLE_CLASSES else None
                    )
                    item["role_prediction"] = annotation["role"]
                    item["attribute_source"] = "MATCHED_ANNOTATION_ASSISTED_DIAGNOSTIC"
                    item["attribute_match_distance_px"] = float(distance)
                    counts["attribute_assisted_matches"] += 1
            elif image is not None and appearance_boxes and projected_items:
                from src.backbone.video_player_attributes import (
                    ROLE_CLASSES,
                    TEAM_CLASSES,
                    _knn_scores,
                    _maximum_footpoint_matches,
                    appearance_feature,
                )

                projectable = [
                    item for item in projected_items if item["pitch"] is not None
                ]
                matches = _maximum_footpoint_matches(
                    [
                        {"bbox_xyxy_px": item["detection"]["bbox_xyxy_px"]}
                        for item in projectable
                    ],
                    [
                        {
                            "footpoint_xy_px": [
                                (box["bbox_xyxy_px"][0] + box["bbox_xyxy_px"][2]) / 2.0,
                                box["bbox_xyxy_px"][3],
                            ],
                            "box": box,
                        }
                        for box in appearance_boxes
                    ],
                    ATTRIBUTE_MATCH_TOLERANCE_PX,
                )
                for detection_index, appearance_index, distance in matches:
                    item = projectable[detection_index]
                    crop_box = appearance_boxes[appearance_index]["bbox_xyxy_px"]
                    feature = appearance_feature(image, crop_box)[None]
                    team_scores = _knn_scores(
                        train_features, train_teams, feature, TEAM_CLASSES,
                    )[0]
                    role_scores = _knn_scores(
                        train_features, train_roles, feature, ROLE_CLASSES,
                    )[0]
                    team_index, role_index = int(team_scores.argmax()), int(role_scores.argmax())
                    team_confidence = float(team_scores[team_index])
                    role_confidence = float(role_scores[role_index])
                    team_threshold = attribute_metadata.get("team_confidence_threshold")
                    role_threshold = attribute_metadata.get("role_confidence_threshold")
                    item["team"] = (
                        team_index + 1
                        if team_threshold is not None and team_confidence >= team_threshold
                        else 0
                    )
                    item["team_confidence"] = team_confidence
                    item["team_prediction"] = TEAM_CLASSES[team_index]
                    item["goalkeeper"] = (
                        role_index == 1
                        if role_threshold is not None and role_confidence >= role_threshold
                        else None
                    )
                    item["role_confidence"] = role_confidence
                    item["role_prediction"] = ROLE_CLASSES[role_index]
                    item["attribute_source"] = "SOURCE_TRAINED_APPEARANCE_MODEL_OR_ABSTENTION"
                    item["attribute_match_distance_px"] = float(distance)
                    counts["appearance_crop_matches"] += 1

            measurements = []
            for item in projected_items:
                detection = item["detection"]
                counts["video_detections"] += 1
                has_position = item["pitch"] is not None
                if has_position:
                    counts["projected_experimental_positions"] += 1
                else:
                    counts["unprojected_detections"] += 1
                counts["known_team"] += int(item["team"] in (1, 2))
                counts["unknown_team"] += int(item["team"] not in (1, 2))
                counts["known_role"] += int(item["goalkeeper"] is not None)
                counts["unknown_role"] += int(item["goalkeeper"] is None)
                measurements.append({
                    "track": str(detection["track"]),
                    "segment": 0,
                    "team": item["team"] if item["team"] in (1, 2) else "UNKNOWN",
                    "goalkeeper": item["goalkeeper"],
                    "position_status": "observed" if has_position else "missing",
                    "xy": item["pitch"],
                    "quality": {
                        "status": "UNVALIDATED_EXPERIMENTAL",
                        "projection": item["projection_status"],
                        "detector_confidence": float(detection["detector_confidence"]),
                        "attribute_source": item["attribute_source"],
                        "team_prediction": item["team_prediction"],
                        "team_confidence": item["team_confidence"],
                        "role_prediction": item["role_prediction"],
                        "role_confidence": item["role_confidence"],
                        "attribute_match_distance_px": item["attribute_match_distance_px"],
                        "position_accuracy_m": None,
                    },
                })
            rows.append({
                "record_type": "frame",
                "frame_index": int(frame["frame_index"]),
                "timestamp_s": float(frame["timestamp_s"]),
                "segment_id": f"{match_id}_{source['half']}_{interval_name}",
                "identity_reset": False,
                "camera_cut": False,
                "measurements": measurements,
                "absent_count": sum(
                    item["pitch"] is None for item in projected_items
                ),
            })
    finally:
        if capture is not None:
            capture.release()

    route_name = "attribute_assisted" if assisted_attributes else "video_estimated"
    external_path = output / "external" / route_name / (
        f"{detector_route}_{match_id}_{source['half']}_{interval_name}.jsonl"
    )
    if external_path.exists():
        raise FileExistsError(f"Refusing to overwrite external observations: {external_path}")
    external_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "record_type": "metadata",
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "source_id": f"cross-match-{detector_route}-{match_id}-{interval_name}-{route_name}",
        "match": match_id,
        "half": source["half"],
        "observation_kind": "exo_source",
        "sampling_seconds": 0.2,
        "missing_count_semantics": "unprojected video detections are explicit missing-position records; inventory is incomplete",
        "coordinates": {
            "system": "pitch_top_left",
            "units": "metres",
            "axes": "x_length_y_width",
            "pitch_length_m": 105,
            "pitch_width_m": 68,
        },
        "origin": {
            "type": "native_resolution_video_detector_footpoints",
            "status": "UNVALIDATED_EXPERIMENTAL",
            "source_video_sha256": source["video_sha256"],
            "detector_sha256": prediction_meta["detector_sha256"],
            "prediction_sha256": prediction_meta["prediction_jsonl_sha256"],
        },
        "calibration": {
            "type": "source_specific_setup_landmark_projection",
            "status": "GLOBAL_GATE_FAILED_LOCAL_PROJECTOR_UNVALIDATED",
            "projector_sha256": file_digest(
                output / "calibration" / f"local_projector_{match_id}.json",
            ),
            "position_accuracy_m": "UNVALIDATED",
            "fitting_and_validation_landmarks": "separate source-specific setup correspondences; no player coordinates used",
        },
        "synchronization": {
            "status": "native_frame_clock",
            "native_fps": NATIVE_FPS,
            "sample_stride_native_frames": SAMPLE_STRIDE,
        },
        "quality": {
            "position_accuracy": "UNVALIDATED",
            "attributes": (
                "post-inference matched annotation assistance; separate diagnostic only"
                if assisted_attributes
                else "frozen source-trained appearance estimate with confidence abstention"
            ),
            "visibility": "simulated later by the existing player-facing provider; not measured gaze",
            "ball_possession": "unknown",
        },
    }
    with external_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(header, allow_nan=False, separators=(",", ":")) + "\n")
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    adapter = ExternalObservationSource(external_path)
    adapted_frames = list(external_frames(adapter, start_s * NATIVE_FPS, count))
    if len(adapted_frames) != count:
        raise ValueError(f"External adapter returned {len(adapted_frames)}/{count} frames")
    result = {
        "schema_version": "football.video-player-cross-match-external-build.1",
        "status": "UNVALIDATED_EXPERIMENTAL_EXTERNAL_OBSERVATIONS",
        "match_id": match_id,
        "interval": interval_name,
        "detector_route": detector_route,
        "attribute_route": route_name,
        "continuous_samples": len(rows),
        "continuous_seconds": end_s - start_s,
        "sampling_hz": NATIVE_FPS / SAMPLE_STRIDE,
        "prediction_sha256": prediction_meta["prediction_jsonl_sha256"],
        "external_observation_path": str(external_path.relative_to(ROOT)),
        "external_observation_sha256": file_digest(external_path),
        "annotation_attributes_used": assisted_attributes,
        "all_final_predictions_frozen": final_set,
        "counts": counts,
        "limitations": [
            "Experimental projections do not pass the source-specific calibration gate.",
            "Unprojected detections remain explicit missing positions; they are not outside the player's true view.",
            "Autonomous team and role predictions may abstain; unknown attributes are not filled from hidden truth.",
            "The assisted branch changes attributes only; positions and runtime tracks remain video-derived.",
        ],
    }
    result_path = external_path.with_suffix(".meta.json")
    write_json(result_path, result)
    return result


def _select_video_actors(
    candidate_frames: list[dict],
    selection_frames: list,
) -> list[dict]:
    if len(candidate_frames) != len(selection_frames):
        raise ValueError("Actor selection inputs must cover the same video samples")

    counts: dict[str, int] = {}
    runs: dict[str, int] = {}
    longest_runs: dict[str, int] = {}
    anchors: dict[str, dict] = {}
    for sample_index, (frame, selection_frame) in enumerate(
        zip(candidate_frames[:VISUAL_ANCHOR_SAMPLES],
            selection_frames[:VISUAL_ANCHOR_SAMPLES]),
    ):
        if int(frame["frame_index"]) != selection_frame.index:
            raise ValueError("Actor selection frames do not match detector timestamps")
        usable_tracks = {
            measurement.track
            for measurement in selection_frame.measurements
            if measurement.team in (1, 2) and measurement.goalkeeper is not None
        }
        for track in counts.keys() | usable_tracks:
            runs[track] = runs.get(track, 0) + 1 if track in usable_tracks else 0
            longest_runs[track] = max(longest_runs.get(track, 0), runs[track])
        for track in usable_tracks:
            counts[track] = counts.get(track, 0) + 1
            if track in anchors:
                continue
            detection = next(
                (
                    item for item in frame["detections"]
                    if str(item.get("track", "")) == track
                ),
                None,
            )
            if detection is None:
                raise ValueError(
                    f"Usable observation track {track} has no detector anchor"
                )
            anchors[track] = {
                "track": track,
                "sample_index": sample_index,
                "frame_index": int(frame["frame_index"]),
                "footpoint_xy_px": list(detection["footpoint_xy_px"]),
            }
    ordered = sorted(
        counts,
        key=lambda track: (-longest_runs[track], -counts[track], track),
    )
    return [
        {
            **anchors[track],
            "usable_observation_samples_in_anchor_prefix": counts[track],
            "longest_usable_run_samples": longest_runs[track],
            "selection_basis": "video_estimated_position_and_confidence_qualified_attributes",
        }
        for track in ordered[:2]
    ]


def _pair_frozen_actors(
    actors: list[dict],
    frozen_frames: list[dict],
    source_width: int,
) -> list[dict]:
    paired = []
    for actor in actors:
        sample_index = actor["sample_index"]
        predictions = frozen_frames[sample_index]["detections"]
        candidates = [
            (
                float(np.hypot(
                    item["footpoint_xy_px"][0] - actor["footpoint_xy_px"][0],
                    item["footpoint_xy_px"][1] - actor["footpoint_xy_px"][1],
                )),
                str(item.get("track", "")),
            )
            for item in predictions
            if item.get("track")
        ]
        candidates = [
            candidate for candidate in candidates
            if candidate[0] <= 80.0 * source_width / 4096.0
        ]
        paired.append({
            **actor,
            "frozen_track": min(candidates)[1] if candidates else None,
            "frozen_anchor_distance_px": min(candidates)[0] if candidates else None,
        })
    return paired


def replay_route(
    protocol: dict,
    output: Path,
    match_id: str,
    interval_name: str,
    detector_route: str,
    *,
    assisted_attributes: bool,
) -> dict:
    source = source_by_id(protocol, match_id)
    start_s, end_s = interval_by_name(protocol, match_id, interval_name)
    count = (end_s - start_s) * (NATIVE_FPS // SAMPLE_STRIDE)
    route_name = "attribute_assisted" if assisted_attributes else "video_estimated"
    external_path = output / "external" / route_name / (
        f"{detector_route}_{match_id}_{source['half']}_{interval_name}.jsonl"
    )
    source_adapter = ExternalObservationSource(external_path)
    tracking_frames = list(external_frames(source_adapter, start_s * NATIVE_FPS, count))
    if len(tracking_frames) != count:
        raise ValueError("External input failed continuous 5Hz frame validation")

    candidate_path = prediction_path(
        output, "adapted", match_id, source["half"], interval_name,
    )
    frozen_path = prediction_path(
        output, "frozen", match_id, source["half"], interval_name,
    )
    candidate_frames, candidate_meta = _read_predictions(candidate_path, count)
    frozen_frames, frozen_meta = _read_predictions(frozen_path, count)
    expected_indices = [
        start_s * NATIVE_FPS + index * SAMPLE_STRIDE for index in range(count)
    ]
    for route, metadata, frames in (
        ("adapted", candidate_meta, candidate_frames),
        ("frozen", frozen_meta, frozen_frames),
    ):
        if (
            metadata.get("route") != route
            or metadata.get("match_id") != match_id
            or metadata.get("interval") != interval_name
            or metadata.get("source_video_sha256") != source["video_sha256"]
            or [int(frame["frame_index"]) for frame in frames] != expected_indices
        ):
            raise ValueError(f"{route} predictions do not match the replay segment")
    load_local_projector(protocol, output, match_id)
    selection_path = output / "external" / "video_estimated" / (
        f"adapted_{match_id}_{source['half']}_{interval_name}.jsonl"
    )
    selection_metadata_path = selection_path.with_suffix(".meta.json")
    if not selection_path.is_file() or not selection_metadata_path.is_file():
        raise FileNotFoundError(
            "Video-only actor selection requires the adapted autonomous "
            f"external stream: {selection_path}"
        )
    selection_metadata = json.loads(
        selection_metadata_path.read_text(encoding="utf-8"),
    )
    if (
        selection_metadata.get("annotation_attributes_used") is not False
        or selection_metadata.get("attribute_route") != "video_estimated"
        or selection_metadata.get("detector_route") != "adapted"
        or selection_metadata.get("match_id") != match_id
        or selection_metadata.get("interval") != interval_name
        or selection_metadata.get("prediction_sha256")
        != candidate_meta["prediction_jsonl_sha256"]
        or selection_metadata.get("external_observation_sha256")
        != file_digest(selection_path)
    ):
        raise ValueError("Video-only actor-selection stream failed provenance checks")
    selection_source = ExternalObservationSource(selection_path)
    if (
        selection_source.header["origin"].get("source_video_sha256")
        != source["video_sha256"]
        or selection_source.header["origin"].get("prediction_sha256")
        != candidate_meta["prediction_jsonl_sha256"]
    ):
        raise ValueError("Actor-selection stream is not tied to the requested video predictions")
    selection_frames = list(external_frames(selection_source, start_s * NATIVE_FPS, count))
    if len(selection_frames) != count:
        raise ValueError("Video-only actor-selection stream is not continuous")
    actors = _pair_frozen_actors(
        _select_video_actors(candidate_frames, selection_frames),
        frozen_frames,
        source["dimensions_px"][0],
    )

    baseline_path = ROOT / "configs" / "backbone_release_baseline.json"
    if file_digest(baseline_path) != protocol["preservation"]["release_config_sha256"]:
        raise ValueError("Frozen backbone release configuration hash changed")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    experiment = ROOT / baseline["experiment"]
    expected_g1 = protocol["preservation"]["G1_checkpoint_sha256"]
    expected_s4 = protocol["preservation"]["original_v0_5_S4_checkpoint_sha256"]
    torch.set_num_threads(4)
    world = ModelSession(experiment, "G1", device="cpu")
    action = ActionModelSession(ROOT / baseline["action_checkpoint"], device="cpu")
    if world.provenance.get("G1", {}).get("sha256") != expected_g1:
        raise ValueError("Loaded G1 hash differs from the locked frozen checkpoint")
    if action.sha256 != expected_s4:
        raise ValueError("Loaded S4 hash differs from the locked original v0.5 checkpoint")
    engine = ReplayEngine(
        world, expiry_s=6.0, fallback="last_seen", action_model=action,
    )

    actor_results = []
    all_rows = []
    for actor in actors:
        track_id = (
            actor["frozen_track"] if detector_route == "frozen"
            else actor["track"]
        )
        if track_id is None:
            track_id = "__NO_COMMON_VIDEO_ANCHOR__"
        provider = SyntheticObservationProvider(
            track_id, seed=42, fov=120, dropout=True,
        )
        engine.reset()
        g1_calls_before = world.world_forward_calls
        g1_batch_before = world.world_forward_batch_elements
        s4_calls_before = action.forward_calls
        query_indices = {
            seconds * (NATIVE_FPS // SAMPLE_STRIDE)
            for seconds in S4_QUERY_SECONDS
        }
        actor_rows = []
        input_observations = []
        for sample_index in range(count):
            frame = tracking_frames[sample_index]
            observation = source_adapter.observe(frame, provider)
            input_observations.append(observation)
            row = engine.step(
                observation,
                include_passing_assessment=False,
                action_query=sample_index in query_indices,
            )
            if sample_index in query_indices:
                actor_rows.append({
                    "sample_index": sample_index,
                    "frame_index": frame.index,
                    "timestamp_s": frame.timestamp_s,
                    "status": row.get("action_inference", {}).get("status", "not_queried"),
                    "reason": row.get("action_inference", {}).get("reason"),
                    "neural_forward_calls_so_far": (
                        action.forward_calls - s4_calls_before
                    ),
                    "output": row.get("action_inference"),
                })
            row["video_anchor"] = actor
            row["runtime_actor_track"] = track_id
            row["scheduled_s4_query"] = sample_index in query_indices
            all_rows.append(row)

        belief_rows = [
            row for row in all_rows[-count:]
        ]
        actor_results.append({
            "video_anchor": actor,
            "runtime_actor_track": track_id,
            "actor_observation_samples": sum(
                observation.actor is not None for observation in input_observations
            ),
            "actor_continuity_seconds": sum(
                observation.actor is not None for observation in input_observations
            ) / (NATIVE_FPS / SAMPLE_STRIDE),
            "g1_calls": world.world_forward_calls - g1_calls_before,
            "g1_contract_unavailable_samples": sum(
                "WORLD_CONTRACT_UNAVAILABLE" in warning
                for row in belief_rows for warning in row["belief"]["warnings"]
            ),
            "g1_hidden_estimates_used": sum(
                entity["backend"] == "G1" and not entity["visible"]
                for row in belief_rows
                for entity in row["belief"]["entities"]
            ),
            "fallback_entity_states": sum(
                entity["backend"] != "G1" and not entity["visible"]
                for row in belief_rows
                for entity in row["belief"]["entities"]
            ),
            "scheduled_s4_queries": len(query_indices),
            "s4_neural_forward_calls": action.forward_calls - s4_calls_before,
            "g1_forward_batch_elements": (
                world.world_forward_batch_elements - g1_batch_before
            ),
            "s4_query_results": actor_rows,
        })

    result = {
        "schema_version": "football.video-player-cross-match-backbone-replay.1",
        "status": "UNVALIDATED_EXPERIMENTAL_BACKBONE_REPLAY",
        "match_id": match_id,
        "half": source["half"],
        "interval": interval_name,
        "interval_seconds": [start_s, end_s],
        "detector_route": detector_route,
        "attribute_route": route_name,
        "external_observation_sha256": file_digest(external_path),
        "g1_checkpoint_sha256": world.provenance["G1"]["sha256"],
        "s4_checkpoint_sha256": action.sha256,
        "g1_checkpoint_load_count": world.load_counts["world"],
        "g1_forward_calls": world.world_forward_calls,
        "g1_forward_batch_elements": world.world_forward_batch_elements,
        "s4_checkpoint_load_count": action.load_count,
        "s4_neural_forward_calls": action.forward_calls,
        "actors_selected_by_video_only_rule": actors,
        "actors": actor_results,
        "actual_visibility": "Synthetic provider, FOV 120, deterministic dropout enabled; not measured gaze.",
        "all_final_predictions_frozen": (
            validate_final_prediction_set(protocol, output)
            if interval_name.startswith("final") else None
        ),
        "limitations": [
            "Position coordinates remain experimental because source calibration validation failed.",
            "Unknown team or goalkeeper labels block G1/S4 input; no hidden attributes are imputed.",
            "S4 queries are supplied-time diagnostics without action labels or tactical-accuracy claims.",
            "G1 uncertainty does not include detector or calibration error.",
            "Ball and possession remain unknown.",
        ],
    }
    result_path = output / "backbone_replay_usable_video_selection" / route_name / (
        f"{detector_route}_{match_id}_{source['half']}_{interval_name}.json"
    )
    status_path = result_path.with_suffix(".jsonl")
    if result_path.exists() or status_path.exists():
        raise FileExistsError(f"Refusing to overwrite replay outputs: {result_path}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    write_json(result_path, result)
    return result


def render_video_comparison(
    protocol: dict,
    output: Path,
    match_id: str,
    interval_name: str,
    detector_route: str,
    duration_seconds: int,
) -> dict:
    source = source_by_id(protocol, match_id)
    start_s, end_s = interval_by_name(protocol, match_id, interval_name)
    if duration_seconds <= 0 or duration_seconds > end_s - start_s:
        raise ValueError("Render duration must fit within the locked interval")
    count = duration_seconds * (NATIVE_FPS // SAMPLE_STRIDE)
    predictions_path = prediction_path(
        output, detector_route, match_id, source["half"], interval_name,
    )
    frames, metadata = _read_predictions(predictions_path, (end_s - start_s) * 5)
    expected_indices = [
        start_s * NATIVE_FPS + index * SAMPLE_STRIDE
        for index in range((end_s - start_s) * 5)
    ]
    if (
        metadata.get("route") != detector_route
        or metadata.get("match_id") != match_id
        or metadata.get("interval") != interval_name
        or metadata.get("source_video_sha256") != source["video_sha256"]
        or file_digest(ROOT / source["video"]) != source["video_sha256"]
        or [int(frame["frame_index"]) for frame in frames] != expected_indices
    ):
        raise ValueError("Render inputs do not match the locked prediction/source provenance")
    projector = load_local_projector(protocol, output, match_id)
    video_path = ROOT / source["video"]
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video for rendering: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != tuple(source["dimensions_px"]):
        capture.release()
        raise ValueError("Render video dimensions differ from locked source metadata")

    ffmpeg = (
        ROOT / "data" / "video_review_tools" / "portable"
        / "ffmpeg-9.0.2-essentials_build" / "bin" / "ffmpeg.exe"
    )
    if not ffmpeg.is_file():
        capture.release()
        raise FileNotFoundError(f"Existing portable ffmpeg is required: {ffmpeg}")
    render_dir = output / "video_comparisons"
    render_dir.mkdir(parents=True, exist_ok=True)
    destination = render_dir / (
        f"{detector_route}_{match_id}_{source['half']}_{interval_name}_"
        f"{duration_seconds}s.mp4"
    )
    if destination.exists():
        capture.release()
        raise FileExistsError(f"Refusing to overwrite rendered video: {destination}")
    process = subprocess.Popen(
        [
            str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080",
            "-r", "5", "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "24",
            "-movflags", "+faststart", str(destination),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        capture.release()
        process.kill()
        raise RuntimeError("Could not create ffmpeg raw-frame input stream")

    start_frame = start_s * NATIVE_FPS
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    preview = None
    try:
        for sample_index, item in enumerate(frames[:count]):
            frame_index = start_frame + sample_index * SAMPLE_STRIDE
            image = _read_video_sample(capture, frame_index)
            canvas = np.full((1080, 1920, 3), (24, 24, 24), dtype=np.uint8)
            cv2.putText(
                canvas,
                "VIDEO-ESTIMATED TRACKS | EXPERIMENTAL, UNVALIDATED POSITIONS",
                (32, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 220, 255), 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{match_id} {source['half']} | {item['timestamp_s']:.1f}s | "
                f"{detector_route} detector",
                (32, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1,
                cv2.LINE_AA,
            )
            video_height = round(1200 * height / width)
            video_display = cv2.resize(
                image, (1200, video_height), interpolation=cv2.INTER_AREA,
            )
            scale_x, scale_y = 1200 / width, video_height / height
            pitch_points = []
            for detection in item["detections"]:
                x1, y1, x2, y2 = detection["bbox_xyxy_px"]
                pitch, _ = project_footpoint(
                    detection["footpoint_xy_px"], projector,
                )
                color = (0, 210, 255) if pitch is not None else (145, 145, 145)
                cv2.rectangle(
                    video_display,
                    (round(x1 * scale_x), round(y1 * scale_y)),
                    (round(x2 * scale_x), round(y2 * scale_y)),
                    color, 1,
                )
                cv2.putText(
                    video_display,
                    str(detection.get("track", ""))[-6:],
                    (round(x1 * scale_x), max(12, round(y1 * scale_y) - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA,
                )
                if pitch is not None:
                    pitch_points.append((pitch, str(detection.get("track", ""))))
            canvas[110:110 + video_height, 32:1232] = video_display
            cv2.putText(
                canvas, "DETECTIONS / LOCAL TRACK IDS",
                (32, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1,
            )
            pitch_left, pitch_top, pitch_width, pitch_height = 1305, 245, 550, 357
            cv2.rectangle(
                canvas,
                (pitch_left, pitch_top),
                (pitch_left + pitch_width, pitch_top + pitch_height),
                (170, 170, 170), 2,
            )
            cv2.line(
                canvas, (pitch_left + pitch_width // 2, pitch_top),
                (pitch_left + pitch_width // 2, pitch_top + pitch_height),
                (120, 120, 120), 1,
            )
            cv2.circle(
                canvas,
                (pitch_left + pitch_width // 2, pitch_top + pitch_height // 2),
                40, (120, 120, 120), 1,
            )
            for pitch, track_id in pitch_points:
                px = pitch_left + round(pitch[0] / 105 * pitch_width)
                py = pitch_top + round(pitch[1] / 68 * pitch_height)
                cv2.circle(canvas, (px, py), 5, (0, 210, 255), -1)
                cv2.putText(
                    canvas, track_id[-5:], (px + 6, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (240, 240, 240), 1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                canvas, "EXPERIMENTAL PROJECTION; NOT VERIFIED PLAYER LOCATIONS",
                (1270, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 255), 1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"projected {len(pitch_points)}/{len(item['detections'])} | "
                "global calibration gate failed",
                (32, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1,
            )
            cv2.putText(
                canvas,
                "team/role, ball, possession, gaze, and tactical value are unknown",
                (32, 820), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1,
            )
            if preview is None:
                preview = canvas.copy()
            process.stdin.write(canvas.tobytes())
    except BaseException:
        capture.release()
        process.stdin.close()
        process.kill()
        process.communicate()
        raise
    else:
        process.stdin.close()
        _, stderr = process.communicate()
        capture.release()
    if process.returncode:
        raise RuntimeError(
            f"ffmpeg could not encode render: {stderr.decode(errors='replace')[-2000:]}"
        )
    if preview is None:
        raise RuntimeError("No frames were rendered")
    preview_path = destination.with_suffix(".jpg")
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Could not save rendered preview: {preview_path}")
    result = {
        "status": "EXPERIMENTAL_VIDEO_RENDER",
        "match_id": match_id,
        "interval": interval_name,
        "detector_route": detector_route,
        "duration_seconds": duration_seconds,
        "sample_count": count,
        "frame_rate": 5,
        "source_video_sha256": source["video_sha256"],
        "prediction_sha256": metadata["prediction_jsonl_sha256"],
        "mp4_path": str(destination.relative_to(ROOT)),
        "mp4_sha256": file_digest(destination),
        "preview_path": str(preview_path.relative_to(ROOT)),
        "projected_positions_are_unvalidated": True,
    }
    write_json(destination.with_suffix(".meta.json"), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--match", required=True)
    build.add_argument("--interval", required=True)
    build.add_argument("--detector-route", choices=["frozen", "adapted"], required=True)
    build.add_argument("--assisted-attributes", action="store_true")
    build.add_argument("--device", choices=["cpu", "0"], default="cpu")

    replay = subparsers.add_parser("replay")
    replay.add_argument("--match", required=True)
    replay.add_argument("--interval", required=True)
    replay.add_argument("--detector-route", choices=["frozen", "adapted"], required=True)
    replay.add_argument("--assisted-attributes", action="store_true")

    render = subparsers.add_parser("render")
    render.add_argument("--match", required=True)
    render.add_argument("--interval", required=True)
    render.add_argument("--detector-route", choices=["frozen", "adapted"], required=True)
    render.add_argument("--duration-seconds", type=int, default=60)

    args = parser.parse_args()
    protocol = load_protocol(args.protocol.resolve())
    output = args.output.resolve()
    if args.command == "build":
        result = build_external_observations(
            protocol,
            output,
            args.match,
            args.interval,
            args.detector_route,
            assisted_attributes=args.assisted_attributes,
            device=args.device,
        )
    elif args.command == "replay":
        result = replay_route(
            protocol,
            output,
            args.match,
            args.interval,
            args.detector_route,
            assisted_attributes=args.assisted_attributes,
        )
    else:
        result = render_video_comparison(
            protocol,
            output,
            args.match,
            args.interval,
            args.detector_route,
            args.duration_seconds,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
