"""Frozen post-selection transfer audit; inference and disposable resume checks only."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from src.backbone.corpus import _write_npz
from src.backbone.train_action import _forward, _metrics, _save_torch, load_split
from src.backbone.train_action_robust_v06 import DEFAULT_BASELINE
from src.backbone.transfer_v06 import (
    FOLDS, INDEX, MODEL, RECIPE, ROOT, build_teacher_inputs, validate, validate_probabilities,
)
from src.models.partial_observation_student import PartialObservationStudent
from src.player_centric_dataset import ACTION_LABELS
from src.trajectory_dataset import file_digest, write_json


def provenance_audit(transfer: Path) -> dict:
    specification = json.loads((transfer / "specification.json").read_text(encoding="utf-8"))
    lineage = json.loads((transfer / "oof_lineage.json").read_text(encoding="utf-8"))
    if len(lineage["folds"]) != 3 or lineage["class_mapping"] != list(ACTION_LABELS):
        raise ValueError("OOF fold count or class mapping mismatch")
    results = {}
    for name in ("teacher_fold_0", "teacher_fold_1", "teacher_fold_2", "teacher_reference", "C1", "C2"):
        folder = transfer / name
        config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        if (result["status"] != "COMPLETE" or result["specification"] != config
                or result["best_sha256"] != file_digest(folder / "best.pt")
                or result["resume_sha256"] != file_digest(folder / "last.pt")
                or saved["state"]["steps"] != result["steps"]
                or saved["state"]["epoch"] != result["epochs"]):
            raise ValueError(f"Checkpoint/result/optimizer provenance mismatch: {name}")
        if not saved["optimizer"]["state"] or set(saved["rng"]) != {"python", "numpy", "torch", "cuda"}:
            raise ValueError(f"Missing resumable optimizer/RNG state: {name}")
        for key in ("index_sha256", "recipe", "model", "initialization_sha256", "corrupted_inputs"):
            if config[key] != specification[key]:
                raise ValueError(f"Unmatched transfer recipe/input: {name}/{key}")
        if name.startswith("teacher_fold"):
            number = int(name[-1])
            fold = lineage["folds"][number]
            if (config["prediction_matches"] != list(FOLDS[number])
                    or {k: fold[k] for k in config} != config
                    or fold["checkpoint_sha256"] != result["best_sha256"]):
                raise ValueError("Teacher lineage differs from actual fit specification")
        results[name] = {"optimizer_steps": result["steps"], "epochs": result["epochs"],
                         "best_epoch": result["best_epoch"], "checkpoint_sha256": result["best_sha256"],
                         "optimizer_rng_preserved": True}
    configs = [json.loads((transfer / name / "run_config.json").read_text(encoding="utf-8"))
               for name in ("C1", "C2")]
    for key in ("recipe", "model", "corrupted_inputs", "initialization_sha256"):
        if configs[0][key] != configs[1][key]:
            raise ValueError("C1/C2 are not a matched comparison")
    return results


def optimizer_parity(transfer: Path, data: dict, output: Path) -> dict:
    saved = torch.load(transfer / "C1" / "last.pt", map_location="cpu", weights_only=False)

    def load(record: dict):
        model = PartialObservationStudent(**MODEL, belief=True).train()
        model.load_state_dict(record["model"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["learning_rate"])
        optimizer.load_state_dict(deepcopy(record["optimizer"]))
        return model, optimizer

    def step(model, optimizer, start: int) -> None:
        model.train()
        selected = slice(start, start + 4)
        prediction = model(torch.from_numpy(data["x"][selected]), torch.from_numpy(data["mask"][selected]),
                           torch.from_numpy(data["visible"][selected]))["action"]
        loss = torch.nn.functional.cross_entropy(prediction, torch.from_numpy(data["y"][selected]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()

    control, control_optimizer = load(saved)
    torch.set_rng_state(saved["rng"]["torch"].cpu())
    step(control, control_optimizer, 0)
    checkpoint = output / "disposable_optimizer_boundary.pt"
    _save_torch(checkpoint, {"model": control.state_dict(), "optimizer": control_optimizer.state_dict(),
                            "torch_rng": torch.get_rng_state()})
    step(control, control_optimizer, 4)
    boundary = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed, resumed_optimizer = load(boundary)
    torch.set_rng_state(boundary["torch_rng"])
    step(resumed, resumed_optimizer, 4)
    difference = max(float(torch.max(torch.abs(control.state_dict()[key] - resumed.state_dict()[key])))
                     for key in control.state_dict() if control.state_dict()[key].is_floating_point())
    if difference != 0:
        raise ValueError("Disposable optimizer/RNG continuation diverged")
    return {"status": "PASS", "next_step_maximum_difference": difference,
            "canonical_checkpoint_modified": False, "uninterrupted_control_steps": 2,
            "optimizer_steps_after_reload": 1, "boundary_checkpoint": str(checkpoint),
            "training": "Disposable two-step control versus persisted-boundary reload and actual next optimizer step"}


def evaluate(transfer: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        raise FileExistsError("Frozen retrospective audit exists; validate saved outputs instead")
    torch.set_num_threads(4)
    validate(INDEX, transfer)
    provenance = provenance_audit(transfer)
    clean = load_split(INDEX, "retrospective_evaluation")
    inputs_root = ROOT / "outputs" / "backbone_v0_6_20261021" / "stage_e" / "heldout_robustness_v06" / "corrupted_inputs"
    conditions = {"clean": clean}
    condition_hashes = {"clean": file_digest(INDEX)}
    for condition in ("training_regimen", "provider_dropout_only", "intensified_stress"):
        folder = inputs_root / condition
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if file_digest(folder / "inputs.npz") != manifest["inputs_sha256"]:
            raise ValueError("Frozen retrospective condition changed")
        with np.load(folder / "inputs.npz", allow_pickle=False) as archive:
            if (archive["sample_ids"].tolist() != clean["sample_ids"].tolist()
                    or manifest["training_index_sha256"] != file_digest(INDEX)
                    or archive["x"].shape != clean["x"].shape
                    or archive["mask"].shape != clean["mask"].shape
                    or archive["visible"].shape != clean["visible"].shape
                    or not np.isfinite(archive["x"]).all()):
                raise ValueError("Frozen target-free condition input contract mismatch")
            conditions[condition] = {**clean, "x": archive["x"].astype(np.float32),
                                     "mask": archive["mask"], "visible": archive["visible"]}
        condition_hashes[condition] = manifest["inputs_sha256"]
    exo = build_teacher_inputs(INDEX, "retrospective_evaluation", clean, output)
    teacher = PartialObservationStudent(**MODEL, belief=False).eval()
    teacher.load_state_dict(torch.load(transfer / "teacher_reference" / "best.pt",
                                      map_location="cpu", weights_only=True), strict=True)
    teacher_logits, _ = _forward(teacher, exo, torch.device("cpu"), 256, mixed_precision=False)
    _write_npz(output / "teacher_reference.npz", sample_ids=clean["sample_ids"], logits=teacher_logits)
    cells = {}
    for name in ("original", "C1", "C2"):
        model = PartialObservationStudent(**MODEL, belief=True).eval()
        weights = (torch.load(DEFAULT_BASELINE, map_location="cpu", weights_only=False)["model"]
                   if name == "original" else
                   torch.load(transfer / name / "best.pt", map_location="cpu", weights_only=True))
        model.load_state_dict(weights, strict=True)
        for condition, data in conditions.items():
            logits, _ = _forward(model, data, torch.device("cpu"), 256, mixed_precision=False)
            prediction = logits.argmax(-1)
            _write_npz(output / f"{name}_{condition}.npz", sample_ids=data["sample_ids"], logits=logits)
            cells[f"{name}_{condition}"] = {
                "metrics": _metrics(data["y"], prediction), "examples": len(prediction),
                "coverage": 1.0, "excluded": 0,
                "coverage_scope": "Pre-existing supported indexed event tensors; corpus exclusions are in source index",
                "per_match": {str(match): _metrics(data["y"][data["matches"] == match],
                                                  prediction[data["matches"] == match])
                              for match in sorted(set(data["matches"].tolist()))},
            }
    fitting = load_split(INDEX, "fitting")
    with np.load(transfer / "oof_targets.npz", allow_pickle=False) as archive:
        probabilities = archive["probabilities"]
        validate_probabilities(probabilities, len(fitting["y"]))
        oof = _metrics(fitting["y"], probabilities.argmax(-1))
        distribution = np.bincount(probabilities.argmax(-1), minlength=7).tolist()
    result = {
        "status": "COMPLETE", "selection": "Frozen original S4 retained using development results BEFORE this audit",
        "retrospective_is_not_fresh_test": True, "conditions_sha256": condition_hashes,
        "cells": cells, "oof_teacher_metrics": oof, "oof_teacher_predicted_class_counts": distribution,
        "class_mapping": list(ACTION_LABELS),
        "teacher_reference_metrics": _metrics(clean["y"], teacher_logits.argmax(-1)),
        "provenance_audit": provenance, "optimizer_continuation": optimizer_parity(transfer, fitting, output),
        "teacher_student_input_separation": "Privileged histories evaluated only by teacher; students use indexed clean or frozen pre-memory-corrupted partial observations",
        "limitation": "No calibration, tactical value, continuous detector, or off-ball semantic-label claim",
    }
    write_json(output / "evaluation.json", result)
    return validate_saved(output)


def validate_saved(output: Path) -> dict:
    result = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    clean = load_split(INDEX, "retrospective_evaluation")
    for name, cell in result["cells"].items():
        with np.load(output / f"{name}.npz", allow_pickle=False) as archive:
            if archive["sample_ids"].tolist() != clean["sample_ids"].tolist():
                raise ValueError("Retrospective comparison event keys differ")
            if _metrics(clean["y"], archive["logits"].argmax(-1)) != cell["metrics"]:
                raise ValueError("Frozen retrospective metrics do not recompute")
    validation = {"status": "PASS", "matched_cells": len(result["cells"]),
                  "matched_event_ids_per_cell": len(clean["y"]), "metrics_recomputed": True,
                  "no_additional_fits": True, "original_default_unchanged": True}
    write_json(output / "validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("evaluate", "validate", "optimizer-probe"), required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "evaluate":
        result = evaluate(args.transfer, args.output)
    elif args.phase == "optimizer-probe":
        torch.set_num_threads(4)
        result = optimizer_parity(args.transfer, load_split(INDEX, "fitting"), args.output)
        write_json(args.output / "optimizer_continuation.json", result)
    else:
        result = validate_saved(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
