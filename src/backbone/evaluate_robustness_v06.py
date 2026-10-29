"""Run a frozen multi-condition action robustness evaluation on held-out matches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.backbone.source import SoccerTrackSource
from src.backbone.train_action import _metrics, load_split
from src.backbone.train_action_robust_v06 import (
    CORRUPTION,
    DEFAULT_BASELINE,
    DEV_MATCHES,
    EVAL_MATCHES,
    FIT_MATCHES,
    HIGH_STRESS_CORRUPTION,
    _corrupt_history,
    _corrupted_history,
    _encode_history,
    _predict,
    _read_jsonl,
    _write_jsonl,
)
from src.models.partial_observation_student import PartialObservationStudent
from src.player_centric_dataset import ACTION_LABELS
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "outputs/backbone_v0_5_20261020/corpus/training_index.json"
DEFAULT_CHECKPOINT = ROOT / "outputs/backbone_v0_6_20261021/stage_c/robust_training/best.pt"
EVALUATION_CORRUPTIONS = {
    "training_regimen": CORRUPTION,
    "intensified_stress": HIGH_STRESS_CORRUPTION,
    "provider_dropout_only": {
        **CORRUPTION,
        "version": "football.action-observation-provider-dropout-only.0.1",
        "additional_correlated_gap_probability": 0.0,
        "identity_break_probability": 0.0,
        "coordinate_noise_std_m": 0.0,
        "seed": 62028,
    },
    "correlated_gap_only": {
        **CORRUPTION,
        "version": "football.action-observation-correlated-gap-only.0.1",
        "provider_dropout": False,
        "additional_correlated_gap_probability": 1.0,
        "identity_break_probability": 0.0,
        "coordinate_noise_std_m": 0.0,
        "seed": 62029,
    },
    "coordinate_noise_only": {
        **CORRUPTION,
        "version": "football.action-observation-coordinate-noise-only.0.1",
        "provider_dropout": False,
        "additional_correlated_gap_probability": 0.0,
        "identity_break_probability": 0.0,
        "coordinate_noise_std_m": 0.15,
        "seed": 62030,
    },
    "identity_break_only": {
        **CORRUPTION,
        "version": "football.action-observation-identity-break-only.0.1",
        "provider_dropout": False,
        "additional_correlated_gap_probability": 0.0,
        "identity_break_probability": 1.0,
        "coordinate_noise_std_m": 0.0,
        "seed": 62031,
    },
}


def _sample_ids_hash(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()


def _build_condition(
    index_path: Path,
    data: dict,
    policy: dict,
    condition_output: Path,
) -> tuple[dict, dict]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    metadata_by_id: dict[str, tuple[dict, dict]] = {}
    for shard in index["shards"]:
        if shard["split"] != "retrospective_evaluation":
            continue
        folder = Path(shard["folder"])
        for filename, digest in shard["artifact_hashes"].items():
            artifact = folder / filename
            if not artifact.is_file() or file_digest(artifact) != digest:
                raise ValueError(f"Indexed retrospective artifact changed: {artifact}")
        for row in _read_jsonl(folder / "student_metadata.jsonl"):
            sample_id = str(row["sample_id"])
            if sample_id in metadata_by_id:
                raise ValueError(f"Duplicate retrospective metadata sample ID: {sample_id}")
            metadata_by_id[sample_id] = (row, shard)

    sample_ids = data["sample_ids"].astype(str).tolist()
    if set(sample_ids) != set(metadata_by_id):
        raise ValueError("Retrospective metadata does not match the indexed evaluation IDs")
    x = np.empty((len(sample_ids), 11, 22, 10), dtype=np.float32)
    masks = np.empty((len(sample_ids), 11, 22), dtype=bool)
    visible = np.empty_like(masks)
    corruption_rows = []
    sources: dict[tuple[str, str], SoccerTrackSource] = {}
    for row_index, sample_id in enumerate(sample_ids):
        metadata, shard = metadata_by_id[sample_id]
        key = (str(shard["match"]), str(shard["half"]))
        source = sources.get(key)
        if source is None:
            source = SoccerTrackSource(ROOT, *key)
            sources[key] = source
        if (metadata["split"] != "retrospective_evaluation"
                or metadata["match"] != key[0] or metadata["half"] != key[1]):
            raise ValueError(f"{sample_id}: held-out source provenance mismatch")
        history, alignment = _corrupted_history(
            source, metadata, sample_id, corruption=policy,
        )
        history, corruption_record = _corrupt_history(
            history, sample_id, seed=int(policy["seed"]), corruption=policy,
        )
        encoded = _encode_history(history, sample_id)
        x[row_index] = encoded["x"]
        masks[row_index] = encoded["mask"]
        visible[row_index] = encoded["visible"]
        corruption_record.update(alignment)
        corruption_rows.append(corruption_record)
    if not np.isfinite(x).all() or np.any(x[..., 6] > 0.40001):
        raise ValueError("Retrospective corruption inputs contain invalid normalized features")
    actor_flags = (x[..., 9] > 0.5) & masks
    actor_visible = (visible[:, -1, :] & actor_flags[:, -1, :]).sum(axis=-1)
    if not np.all(actor_flags.sum(axis=-1) == 1) or not np.all(actor_visible == 1):
        raise ValueError("Retrospective corruption inputs do not preserve the selected actor")

    condition_output.mkdir(parents=True, exist_ok=True)
    inputs_path = condition_output / "inputs.npz"
    temporary = inputs_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, x=x, mask=masks, visible=visible,
            sample_ids=np.asarray(sample_ids, dtype=str),
        )
    os.replace(temporary, inputs_path)
    log_path = condition_output / "corruptions.jsonl"
    _write_jsonl(log_path, corruption_rows)
    manifest = {
        "schema_version": "football.backbone-v0.6-heldout-corruption-inputs.1",
        "status": "COMPLETE",
        "condition": policy["version"],
        "split": "retrospective_evaluation",
        "matches": sorted(EVAL_MATCHES),
        "samples": len(sample_ids),
        "sample_ids_sha256": _sample_ids_hash(sample_ids),
        "training_index_sha256": file_digest(index_path),
        "corruption": policy,
        "corruption_counts": {
            "identity_breaks": sum(row["identity_break_track"] is not None for row in corruption_rows),
            "correlated_gaps": sum(row["correlated_gap_track"] is not None for row in corruption_rows),
            "removed_detections": sum(row["additional_detections_removed"] for row in corruption_rows),
            "query_aligned_samples": sum(row["query_lag_frames"] == 0 for row in corruption_rows),
            "causal_lagged_samples": sum(row["query_lag_frames"] > 0 for row in corruption_rows),
            "maximum_query_lag_frames": max(row["query_lag_frames"] for row in corruption_rows),
        },
        "source_provenance": {
            f"{match}_{half}": source.provenance
            for (match, half), source in sources.items()
        },
        "inputs_sha256": file_digest(inputs_path),
        "corruption_log_sha256": file_digest(log_path),
        "targets_in_input_artifact": False,
    }
    write_json(condition_output / "manifest.json", manifest)
    condition_data = {
        "x": x, "mask": masks, "visible": visible,
        "y": data["y"], "sample_ids": data["sample_ids"],
        "matches": data["matches"],
    }
    return condition_data, manifest


def _metrics_by_match(data: dict, prediction: np.ndarray) -> dict:
    result = {}
    for match in EVAL_MATCHES:
        selected = data["matches"] == match
        if not np.any(selected):
            raise ValueError(f"No retrospective samples for held-out match {match}")
        result[match] = {
            "samples": int(selected.sum()),
            "metrics": _metrics(data["y"][selected], prediction[selected]),
        }
    return result


def evaluate(
    index_path: Path,
    checkpoint: Path,
    output: Path,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing Stage E artifacts; choose a new output directory: {output}"
        )
    if (output / "heldout_robustness_evaluation.json").exists():
        raise FileExistsError("Refusing to overwrite an existing Stage E evaluation")
    data = load_split(index_path, "retrospective_evaluation")
    if set(data["matches"].tolist()) != set(EVAL_MATCHES):
        raise ValueError("Evaluation input contains unexpected held-out matches")
    model_config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    training_result = json.loads(
        (checkpoint.parent / "training_result.json").read_text(encoding="utf-8")
    )
    if (model_config.get("variant") != "S4" or model_config.get("fov_index") != 1
            or model_config.get("fitting_matches") != FIT_MATCHES
            or model_config.get("development_matches") != DEV_MATCHES
            or training_result.get("status") != "COMPLETE"
            or training_result.get("best_checkpoint_sha256") != file_digest(checkpoint)
            or training_result.get("training_index_sha256") != file_digest(index_path)
            or training_result.get("class_mapping") != list(ACTION_LABELS)
            or training_result.get("retrospective_evaluation_matches") != EVAL_MATCHES):
        raise ValueError("Checkpoint is not a complete, provenance-matched S4 training result")

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robust = PartialObservationStudent(width=96, layers=2, heads=4, dropout=0.1, belief=True).to(device)
    robust.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    baseline_record = torch.load(DEFAULT_BASELINE, map_location=device, weights_only=False)
    baseline_state = baseline_record["model"] if "model" in baseline_record else baseline_record
    baseline = PartialObservationStudent(width=96, layers=2, heads=4, dropout=0.1, belief=True).to(device)
    baseline.load_state_dict(baseline_state, strict=True)

    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    target_independence = {}
    for model_name, model in (("baseline", baseline), ("robust", robust)):
        logits = _predict(model, data, device)
        changed_targets = np.roll(data["y"], 1)
        if np.array_equal(changed_targets, data["y"]):
            raise ValueError("Target-independence probe did not change the target array")
        altered_target_data = {**data, "y": changed_targets}
        altered_target_logits = _predict(model, altered_target_data, device)
        difference = float(np.max(np.abs(logits - altered_target_logits)))
        if difference != 0.0:
            raise ValueError(f"{model_name} predictions depend on held-out labels: {difference}")
        target_independence[model_name] = difference
        prediction = logits.argmax(axis=-1)
        key = f"clean_{model_name}"
        predictions[key] = prediction
        metrics[key] = {
            "overall": _metrics(data["y"], prediction),
            "by_match": _metrics_by_match(data, prediction),
        }

    condition_manifests = {}
    for condition, policy in EVALUATION_CORRUPTIONS.items():
        condition_data, manifest = _build_condition(
            index_path, data, policy, output / "corrupted_inputs" / condition,
        )
        condition_manifests[condition] = manifest
        for model_name, model in (("baseline", baseline), ("robust", robust)):
            prediction = _predict(model, condition_data, device).argmax(axis=-1)
            key = f"{condition}_{model_name}"
            predictions[key] = prediction
            metrics[key] = {
                "overall": _metrics(data["y"], prediction),
                "by_match": _metrics_by_match(data, prediction),
            }

    sample_ids = data["sample_ids"].astype(str).tolist()
    prediction_rows = []
    for index, sample_id in enumerate(sample_ids):
        prediction_rows.append({
            "sample_id": sample_id,
            "match": str(data["matches"][index]),
            "target": ACTION_LABELS[int(data["y"][index])],
            "predictions": {
                name: ACTION_LABELS[int(values[index])]
                for name, values in predictions.items()
            },
        })
    predictions_path = output / "heldout_robustness_predictions.jsonl"
    _write_jsonl(predictions_path, prediction_rows)
    report = {
        "schema_version": "football.backbone-v0.6-heldout-robustness-evaluation.1",
        "status": "COMPLETE",
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "evaluation_matches": sorted(EVAL_MATCHES),
        "evaluation_examples": len(data["y"]),
        "training_index_sha256": file_digest(index_path),
        "robust_checkpoint_sha256": file_digest(checkpoint),
        "baseline_checkpoint_sha256": file_digest(DEFAULT_BASELINE),
        "evaluator_code_sha256": file_digest(Path(__file__)),
        "corruption_builder_code_sha256": file_digest(
            Path(__file__).with_name("train_action_robust_v06.py")
        ),
        "class_mapping": list(ACTION_LABELS),
        "common_sample_ids_sha256": _sample_ids_hash(sample_ids),
        "evaluation_target_independence": {
            "status": "PASS",
            "changed_target_array": True,
            "observation_arrays_unchanged": True,
            "maximum_logit_difference": target_independence,
        },
        "conditions": {
            "clean": {
                "input_source": "Indexed v0.5 retrospective student_inputs.npz",
                "metrics": {key: value for key, value in metrics.items() if key.startswith("clean_")},
            },
            **{
                condition: {
                    "input_manifest": str(
                        (output / "corrupted_inputs" / condition / "manifest.json").resolve()
                    ),
                    "input_manifest_sha256": file_digest(
                        output / "corrupted_inputs" / condition / "manifest.json"
                    ),
                    "corruption": manifest["corruption"],
                    "corruption_counts": manifest["corruption_counts"],
                    "metrics": {
                        key: value for key, value in metrics.items()
                        if key.startswith(f"{condition}_")
                    },
                }
                for condition, manifest in condition_manifests.items()
            },
        },
        "comparison_protocol": "Frozen baseline and robust S4 checkpoints use identical held-out sample IDs and labels in clean, training-regimen, intensified-corruption, and four isolated corruption-component conditions; no retrospective result is used for training or checkpoint selection.",
        "task_semantics": "Event-conditioned action classification at indexed actor/query times; not continuous action detection or best-action recommendation.",
        "score_semantics": "Reported values are categorical metrics. Classifier scores are not calibrated confidence, tactical value, or a best-action ranking.",
        "prediction_artifact": {
            "path": str(predictions_path.resolve()),
            "sha256": file_digest(predictions_path),
            "rows": len(prediction_rows),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "heldout_robustness_evaluation.json", report)
    return report


def validate(index_path: Path, checkpoint: Path, output: Path) -> dict:
    report_path = output / "heldout_robustness_evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    data = load_split(index_path, "retrospective_evaluation")
    artifact = report["prediction_artifact"]
    predictions_path = Path(artifact["path"])
    rows = _read_jsonl(predictions_path)
    expected_ids = data["sample_ids"].astype(str).tolist()
    if (report.get("status") != "COMPLETE" or report.get("training_performed") is not False
            or report.get("checkpoint_selection_performed") is not False
            or report.get("evaluation_target_independence", {}).get("status") != "PASS"
            or report.get("evaluation_target_independence", {}).get("changed_target_array") is not True
            or report.get("evaluation_target_independence", {}).get("observation_arrays_unchanged") is not True
            or report.get("evaluation_target_independence", {}).get("maximum_logit_difference")
            != {"baseline": 0.0, "robust": 0.0}
            or report.get("training_index_sha256") != file_digest(index_path)
            or report.get("robust_checkpoint_sha256") != file_digest(checkpoint)
            or report.get("baseline_checkpoint_sha256") != file_digest(DEFAULT_BASELINE)
            or report.get("evaluator_code_sha256") != file_digest(Path(__file__))
            or report.get("corruption_builder_code_sha256") != file_digest(
                Path(__file__).with_name("train_action_robust_v06.py")
            )
            or report.get("class_mapping") != list(ACTION_LABELS)
            or report.get("evaluation_examples") != len(data["y"])
            or report.get("evaluation_matches") != sorted(EVAL_MATCHES)
            or report.get("task_semantics")
            != "Event-conditioned action classification at indexed actor/query times; not continuous action detection or best-action recommendation."
            or report.get("common_sample_ids_sha256") != _sample_ids_hash(expected_ids)):
        raise ValueError("Held-out evaluation provenance/status mismatch")
    if (artifact.get("sha256") != file_digest(predictions_path)
            or artifact.get("rows") != len(rows) or len(rows) != len(expected_ids)):
        raise ValueError("Held-out prediction artifact count/hash mismatch")
    names = {"clean_baseline", "clean_robust"} | {
        f"{condition}_{model}"
        for condition in EVALUATION_CORRUPTIONS
        for model in ("baseline", "robust")
    }
    if any(set(row.get("predictions", {})) != names for row in rows):
        raise ValueError("Held-out prediction rows have missing/extra evaluation conditions")
    label_to_index = {label: index for index, label in enumerate(ACTION_LABELS)}
    prediction_arrays = {
        name: np.empty(len(rows), dtype=np.int64) for name in names
    }
    for index, row in enumerate(rows):
        if row.get("sample_id") != expected_ids[index] or row.get("match") != str(data["matches"][index]):
            raise ValueError(f"Held-out prediction row alignment mismatch at row {index}")
        try:
            target = label_to_index[row["target"]]
            for name in names:
                prediction_arrays[name][index] = label_to_index[row["predictions"][name]]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Unknown/missing class in held-out prediction row {index}") from error
        if target != int(data["y"][index]):
            raise ValueError(f"Held-out target mismatch at row {index}")

    for condition, policy in EVALUATION_CORRUPTIONS.items():
        manifest_path = Path(report["conditions"][condition]["input_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inputs_path = manifest_path.parent / "inputs.npz"
        log_path = manifest_path.parent / "corruptions.jsonl"
        if (manifest.get("status") != "COMPLETE"
                or manifest.get("split") != "retrospective_evaluation"
                or manifest.get("condition") != policy["version"]
                or manifest.get("corruption") != policy
                or manifest.get("targets_in_input_artifact") is not False
                or manifest.get("samples") != len(expected_ids)
                or manifest.get("sample_ids_sha256") != _sample_ids_hash(expected_ids)
                or manifest.get("training_index_sha256") != file_digest(index_path)
                or report["conditions"][condition].get("input_manifest_sha256") != file_digest(manifest_path)
                or manifest.get("inputs_sha256") != file_digest(inputs_path)
                or manifest.get("corruption_log_sha256") != file_digest(log_path)):
            raise ValueError(f"Held-out corruption artifacts failed provenance checks: {condition}")
        with np.load(inputs_path, allow_pickle=False) as archive:
            if set(archive.files) != {"x", "mask", "visible", "sample_ids"}:
                raise ValueError(f"Held-out corruption input contains unexpected target fields: {condition}")
            if archive["sample_ids"].astype(str).tolist() != expected_ids:
                raise ValueError(f"Held-out corrupted input IDs are not aligned: {condition}")
            x = archive["x"]
            mask = archive["mask"].astype(bool)
            visible = archive["visible"].astype(bool)
            if (x.shape != (len(expected_ids), 11, 22, 10)
                    or mask.shape != (len(expected_ids), 11, 22)
                    or visible.shape != mask.shape
                    or not np.isfinite(x).all() or np.any(x[..., 6] > 0.40001)):
                raise ValueError(f"Held-out corrupted feature/mask arrays are invalid: {condition}")
            actor_flags = (x[..., 9] > 0.5) & mask
            if (not np.all(actor_flags.sum(axis=-1) == 1)
                    or not np.all((visible[:, -1, :] & actor_flags[:, -1, :]).sum(axis=-1) == 1)):
                raise ValueError(f"Held-out corrupted inputs do not preserve visible actors: {condition}")
        corruption_rows = _read_jsonl(log_path)
        if (len(corruption_rows) != len(expected_ids)
                or [row.get("sample_id") for row in corruption_rows] != expected_ids
                or any(
            len(row["effective_history_frames"]) != 11
            or row["effective_history_frames"][-1] > row["query_frame"]
            or row["query_lag_frames"] != row["query_frame"] - row["effective_history_frames"][-1]
            or row["query_lag_frames"] not in range(5)
            or any(b - a != 5 for a, b in zip(row["effective_history_frames"], row["effective_history_frames"][1:]))
            for row in corruption_rows
        )):
            raise ValueError(f"Held-out corruption log violates causal 11x5 history: {condition}")
        expected_counts = {
            "identity_breaks": sum(row["identity_break_track"] is not None for row in corruption_rows),
            "correlated_gaps": sum(row["correlated_gap_track"] is not None for row in corruption_rows),
            "removed_detections": sum(row["additional_detections_removed"] for row in corruption_rows),
            "query_aligned_samples": sum(row["query_lag_frames"] == 0 for row in corruption_rows),
            "causal_lagged_samples": sum(row["query_lag_frames"] > 0 for row in corruption_rows),
            "maximum_query_lag_frames": max(row["query_lag_frames"] for row in corruption_rows),
        }
        if manifest.get("corruption_counts") != expected_counts:
            raise ValueError(f"Held-out corruption counts do not reproduce: {condition}")
        for source_key, provenance in manifest.get("source_provenance", {}).items():
            match, half = source_key.rsplit("_", 1)
            current = SoccerTrackSource(ROOT, match, half).provenance
            if any(
                provenance.get(field) != current.get(field)
                for field in ("cache_sha256", "association_sha256", "source_id")
            ):
                raise ValueError(f"Held-out source provenance changed: {source_key}")

    expected_metrics = {}
    for name, prediction in prediction_arrays.items():
        expected_metrics[name] = {
            "overall": _metrics(data["y"], prediction),
            "by_match": _metrics_by_match(data, prediction),
        }
    for condition, prefix in (
        ("clean", "clean"),
        *((name, name) for name in EVALUATION_CORRUPTIONS),
    ):
        for model_name in ("baseline", "robust"):
            name = f"{prefix}_{model_name}"
            if report["conditions"][condition]["metrics"].get(name) != expected_metrics[name]:
                raise ValueError(f"Held-out metrics do not reproduce: {name}")
    validation = {
        "schema_version": "football.backbone-v0.6-heldout-robustness-validation.1",
        "status": "PASS",
        "examples": len(rows),
        "matches": sorted(EVAL_MATCHES),
        "conditions": ["clean", *EVALUATION_CORRUPTIONS],
        "predictions_aligned": True,
        "metrics_recomputed": True,
        "corruption_inputs_target_free": True,
        "causal_history_validation": "PASS",
        "training_or_checkpoint_selection_on_retrospective_data": False,
    }
    write_json(output / "heldout_robustness_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("evaluate", "validate"), required=True)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.index if args.index.is_absolute() else ROOT / args.index
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = (
        evaluate(index_path, checkpoint, output)
        if args.phase == "evaluate"
        else validate(index_path, checkpoint, output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
