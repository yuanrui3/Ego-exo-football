"""Causal pre-memory dose response with frozen release models and keyed targets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import difflib
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from src.backbone.actions import ActionModelSession
from src.backbone.association_audit import run_association, synthetic_reproduction
from src.backbone.identity_program import ExclusiveLock
from src.backbone.memory import CausalMemory
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.source import SoccerTrackSource
from src.player_centric_dataset import ACTION_LABELS
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "outputs" / "backbone_v0_5_20261020" / "corpus" / "training_index.json"
SOURCES = ("src/backbone/association_audit.py", "src/backbone/perception_sensitivity.py",
           "tests/test_association_sensitivity.py")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_for(*values) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, values)).encode()).digest()[:8], "little")


def conditions() -> list[dict]:
    result = [{"family": "clean", "severity": 0.}]
    for magnitude in (.2, .4, 1., 2.):
        result.append({"family": "focal_gap", "severity": magnitude, "rate": .1})
    for family in ("focal_fragment", "same_team_swap", "opposing_team_swap", "nonfocal_fragment"):
        for rate in (.02, .1, .25):
            result.append({"family": family, "severity": rate, "rate": rate})
    for family in ("noise", "translation", "drift"):
        for magnitude in (.25, .5, 1., 2., 4.):
            result.append({"family": family, "severity": magnitude})
    return result


def condition_name(condition: dict) -> str:
    return f"{condition['family']}_{condition['severity']:g}"


def corrupt(stream: tuple, condition: dict, seed: int) -> tuple[tuple, dict]:
    """Only observations enter here. No reference positions, action labels or predictions."""
    family, severity = condition["family"], condition["severity"]
    if family == "clean" or severity == 0:
        return stream, {}
    if family == "rename":
        return tuple(replace(o, actor_track="renamed_" + o.actor_track,
                             measurements=tuple(replace(m, track="renamed_" + m.track) for m in o.measurements))
                     for o in stream), {}
    if family == "permutation":
        return tuple(replace(o, measurements=tuple(reversed(o.measurements))) for o in stream), {}
    result, counts = [], Counter()
    gap_until, fragments, swaps = -1., {}, {}
    drift = np.zeros(2)
    if not stream:
        return (), {}
    rng = np.random.default_rng(seed_for(seed, stream[0].source_id, stream[0].actor_track, "direction"))
    angle = rng.uniform(-np.pi, np.pi)
    translation = severity * np.array([np.cos(angle), np.sin(angle)])
    starts = []
    first_runtime_loss = None
    for observation in stream:
        now = observation.timestamp_s
        measurements = observation.measurements
        actor = observation.actor_track
        # Same uniform variates and eligible opportunities across rates/severities.
        generator = np.random.default_rng(seed_for(seed, observation.source_id, actor, observation.index))
        uniform = {m.track: float(generator.random()) for m in sorted(measurements, key=lambda x: x.track)}
        counts["eligible_focal_player_milliseconds"] += int(observation.actor is not None) * 200
        counts["eligible_nonfocal_player_milliseconds"] += sum(m.track != actor for m in measurements) * 200
        rate = condition.get("rate", 0.)
        if family in ("focal_gap", "focal_fragment") and observation.actor is not None:
            if uniform[actor] < 1 - math.exp(-rate * .2):
                counts["event_starts"] += 1
                starts.append(now)
                if family == "focal_gap":
                    gap_until = max(gap_until, now + severity)
                elif actor not in fragments:
                    fragments[actor] = f"fragment_{actor}_{observation.index}"
        if family == "nonfocal_fragment":
            for m in measurements:
                if m.track != actor and uniform[m.track] < 1 - math.exp(-rate * .2):
                    fragments[m.track] = f"fragment_{m.track}_{observation.index}"
                    counts["event_starts"] += 1
        if family in ("same_team_swap", "opposing_team_swap"):
            swaps = {key: value for key, value in swaps.items() if now < value[1] - 1e-6}
            reserved = set(swaps)
            available = [m for m in sorted(measurements, key=lambda m: seed_for(seed, observation.index, m.track))
                         if m.track != actor and m.track not in reserved]
            used = set()
            for a in available:
                if a.track in used:
                    continue
                b = next((b for b in available if b.track != a.track and b.track not in used
                          and (a.team == b.team if family == "same_team_swap" else a.team != b.team)
                          and a.team in (1, 2) and b.team in (1, 2)), None)
                if b is None:
                    continue
                used.update((a.track, b.track))
                counts["eligible_disjoint_pair_milliseconds"] += 200
                u = np.random.default_rng(seed_for(seed, observation.index, *sorted((a.track, b.track)), "pair")).random()
                if u < 1 - math.exp(-rate * .2):
                    swaps[a.track], swaps[b.track] = (b.track, now + 1.), (a.track, now + 1.)
                    counts["event_starts"] += 1
        if family == "drift":
            rho = math.exp(-.2 / 5)
            drift = rho * drift + generator.normal(0, severity / math.sqrt(2) * math.sqrt(1 - rho * rho), 2)
        changed = []
        displacements = {}
        for m in measurements:
            if family == "focal_gap" and m.track == actor and now < gap_until - 1e-6:
                counts["removed_focal_observations"] += 1
                continue
            track = fragments.get(m.track, m.track)
            if m.track in swaps:
                track = swaps[m.track][0]
            xy = np.asarray(m.xy, float)
            displacement = np.zeros(2)
            if family == "noise":
                displacement = np.random.default_rng(seed_for(seed, observation.index, m.track, "xy")).normal(
                    0, severity / math.sqrt(2), 2)
            elif family == "translation":
                displacement = translation
            elif family == "drift":
                displacement = drift
            counts["affected_observations"] += int(track != m.track or np.any(displacement))
            counts["available_observations"] += 1
            counts["position_squared_error_m2"] += float(np.dot(displacement, displacement))
            counts["position_samples"] += 1
            displacements[m.track] = displacement
            changed.append(replace(m, track=track, xy=tuple(xy + displacement)))
        # A swap remains bijective even when one endpoint is temporarily unobserved.
        if len({m.track for m in changed}) != len(changed):
            raise AssertionError("Corruption collapsed identities")
        altered = replace(observation, measurements=tuple(changed))
        if observation.actor is not None and altered.actor is None and first_runtime_loss is None:
            first_runtime_loss = now - stream[0].timestamp_s
        if actor in displacements:
            for displacement in displacements.values():
                relative = displacement - displacements[actor]
                counts["relative_position_squared_error_m2"] += float(np.dot(relative, relative))
                counts["relative_position_samples"] += 1
        result.append(altered)
    counts["first_event_time_s"] = starts[0] if starts else None
    counts["first_runtime_loss_elapsed_s"] = first_runtime_loss
    counts["prefix_duration_s"] = stream[-1].timestamp_s - stream[0].timestamp_s
    return tuple(result), dict(counts)


class FinalInputMemory(CausalMemory):
    """Skip unused intermediate encoding; state updates remain the released implementation."""

    def __init__(self):
        super().__init__(SimpleNamespace(world=None, backend="last_seen"))
        self.encode_now = False

    def encode(self):
        return super().encode() if self.encode_now else None


def prepare(stream: tuple, *, optimized: bool = True):
    memory = FinalInputMemory() if optimized else CausalMemory(SimpleNamespace(world=None, backend="last_seen"))
    warnings = ()
    for i, observation in enumerate(stream):
        if optimized:
            memory.encode_now = i == len(stream) - 1
        row = memory.step(observation)
        warnings = row["warnings"]
    value = memory.current_input
    reason = None
    if value is None:
        if not stream or stream[-1].actor is None:
            reason = "FOCAL_RUNTIME_TOKEN_UNAVAILABLE"
        elif len(memory.history) < 11:
            reason = "HISTORY_WARMUP"
        elif any(o.actor is None for o in memory.history):
            reason = "DISCONTINUOUS_FOCAL_HISTORY"
        elif any(m.team not in (1, 2) or m.goalkeeper is None for o in memory.history for m in o.measurements):
            reason = "UNKNOWN_TEAM_ROLE"
        elif len({(m.track, m.segment) for o in memory.history for m in o.measurements}) > 22:
            reason = "SLOT_OVERFLOW"
        else:
            reason = "OTHER_MODEL_CONTRACT"
    return value, memory, reason, warnings


def action_input(inputs):
    """Release adapter validation without a forward pass; matched against ActionModelSession."""
    x, mask = inputs.x, inputs.mask
    if x.shape != (11, 22, 10) or mask.shape != (11, 22) or not np.isfinite(x).all():
        return None, "SHAPE_OR_NONFINITE"
    visible = x[..., 4] > .5
    actor = (x[..., 9] > .5) & mask
    if np.any(visible & ~mask) or not np.array_equal(mask, x[..., 5] > .5):
        return None, "MASK_CONTRACT"
    if not np.all(actor.sum(1) == 1):
        return None, "ACTOR_SLOT_CONTRACT"
    slots = actor.argmax(1)
    if not np.all(slots == slots[-1]) or not np.all(x[:, slots[-1], 4] > .5):
        return None, "ACTOR_HISTORY_CONTRACT"
    if np.any(x[..., 6] < 0) or np.any(x[..., 6] > .25001):
        return None, "AGE_CONTRACT"
    value = x.copy()
    value[..., 6] = np.clip(value[..., 6] * 8 / 5, 0, 1)
    return value, None


class FrozenBatch:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA frozen inference required; no silent CPU fallback")
        release = read(ROOT / "configs" / "backbone_release_baseline.json")
        self.world = ModelSession(ROOT / release["experiment"], "G1", device="cuda")
        self.action = ActionModelSession(ROOT / release["action_checkpoint"], device="cuda")
        self.elements, self.calls = Counter(), Counter()

    @torch.inference_mode()
    def predict(self, inputs: list):
        result = [{"world": None, "action": None, "reason": "MEMORY_UNSUPPORTED"} for _ in inputs]
        selected = [(i, v) for i, v in enumerate(inputs) if v is not None]
        for start in range(0, len(selected), 128):
            chunk = selected[start:start + 128]
            if not chunk:
                continue
            batch = {key: torch.from_numpy(np.stack([getattr(v, key) for _, v in chunk])).cuda()
                     for key in ("x", "mask", "actor_xy", "last_xy")}
            output = self.world.world(**batch)
            means = output["mean"].float().cpu().numpy()
            std = output["log_std"].float().exp().cpu().numpy()
            if not np.isfinite(means).all() or not np.isfinite(std).all():
                raise FloatingPointError("Nonfinite frozen G1 output")
            self.elements["G1"] += len(chunk)
            self.calls["G1"] += 1
            action_rows = []
            for n, (index, value) in enumerate(chunk):
                result[index]["world"] = {"mean": means[n], "std": std[n]}
                converted, reason = action_input(value)
                result[index]["reason"] = reason
                if converted is not None:
                    action_rows.append((index, value, converted))
            if action_rows:
                x = torch.from_numpy(np.stack([v for _, _, v in action_rows])).cuda()
                mask = torch.from_numpy(np.stack([v.mask for _, v, _ in action_rows])).cuda()
                logits = self.action.model(x, mask, x[..., 4] > .5, memory_mode="memory")["action"].float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite frozen S4 output")
                probabilities = logits.softmax(-1).cpu().numpy()
                for (index, _, _), probabilities_i in zip(action_rows, probabilities, strict=True):
                    result[index]["action"] = probabilities_i
                self.elements["S4"] += len(action_rows)
                self.calls["S4"] += 1
        return result

    def parity(self, inputs: list) -> dict:
        selected = [v for v in inputs if v is not None][:8]
        if not selected:
            raise RuntimeError("No supported development inputs for GPU parity")
        batched = self.predict(selected)
        differences = []
        for v, result in zip(selected, batched, strict=True):
            world = self.world.predict(v)
            action = self.action.predict(v)
            differences.append(float(np.max(np.abs(world["mean"] - result["world"]["mean"]))))
            differences.append(float(np.max(np.abs(world["std"] - result["world"]["std"]))))
            if result["action"] is None or action.status != "supported":
                raise AssertionError("Batched/sequential support parity failed")
            differences.append(float(np.max(np.abs(np.asarray(list(action.scores.values())) - result["action"]))))
        # Also compare the released sequential CPU path, not only CUDA batch sizes.
        cpu_world = ModelSession(ROOT / "outputs" / "dynamic_gnn_tactical_20261019", "G1", device="cpu")
        cpu_action = ActionModelSession(self.action.checkpoint, device="cpu")
        for v, result in zip(selected[:2], batched[:2], strict=True):
            reference = cpu_world.predict(v)
            action = cpu_action.predict(v)
            differences.append(float(np.max(np.abs(reference["mean"] - result["world"]["mean"]))))
            differences.append(float(np.max(np.abs(reference["std"] - result["world"]["std"]))))
            differences.append(float(np.max(np.abs(np.asarray(list(action.scores.values())) - result["action"]))))
        maximum = max(differences)
        if maximum > .0002:
            raise AssertionError(f"CPU/sequential/CUDA batched parity failed: {maximum}")
        return {"status": "PASS", "inputs": len(selected), "maximum_absolute_difference": maximum,
                "tolerance": .0002, "gpu": torch.cuda.get_device_name(0),
                "precision": "FP32 inference_mode, no autocast, frozen eval models"}


def population() -> tuple[list, dict]:
    index = read(INDEX)
    result, evidence = [], {"index_sha256": file_digest(INDEX), "shards": []}
    for shard in index["shards"]:
        if shard["match"] not in ("132877", "118577", "128058"):
            continue
        folder = Path(shard["folder"])
        loaded = {}
        for name in ("student_metadata.jsonl", "optional_targets.jsonl"):
            if file_digest(folder / name) != shard["artifact_hashes"][name]:
                raise ValueError(f"Indexed population changed: {folder / name}")
            rows = [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").splitlines()]
            loaded[name] = {row["sample_id"]: row for row in rows}
            if len(rows) != len(loaded[name]):
                raise ValueError("Duplicate evaluation key")
        metadata, targets = loaded["student_metadata.jsonl"], loaded["optional_targets.jsonl"]
        if metadata.keys() != targets.keys():
            raise ValueError("Stable annotation/metadata join mismatch")
        for key, row in metadata.items():
            target = targets[key]
            label = target["mapped_action_label"]
            if label not in ACTION_LABELS or target["actor"]["source_track"] != row["actor_track"]:
                raise ValueError("Untrained class or intended actor mismatch")
            result.append({"id": key, "match": str(row["match"]), "half": row["half"],
                           "actor": row["actor_track"], "query": int(row["query_frame"]),
                           "label": ACTION_LABELS.index(label), "weight": 1.})
        evidence["shards"].append({"match": shard["match"], "half": shard["half"],
                                   "count": len(metadata), "artifact_hashes": shard["artifact_hashes"]})
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate cross-shard sample key")
    return sorted(result, key=lambda r: (r["match"] != "132877", r["match"], r["half"], r["query"], r["id"])), evidence


def select_population(rows: list, cap: int) -> list:
    groups = defaultdict(list)
    for row in rows:
        groups[row["match"], row["half"], row["label"]].append(row)
    result = []
    for group in groups.values():
        number = min(len(group), max(1, round(cap * len(group) / len(rows))))
        chosen = sorted(group, key=lambda r: seed_for("population", r["id"]))[:number]
        result.extend({**r, "weight": len(group) / len(chosen), "stratum_population": len(group),
                       "stratum_selected": len(chosen)} for r in chosen)
    return sorted(result, key=lambda r: (r["match"] != "132877", r["match"], r["half"], r["query"], r["id"]))


def clean_stream(source, row: dict) -> tuple[tuple, dict]:
    # Exact native query offset, no query rounding or post-query samples.
    start = max(row["query"] % 5, row["query"] - 200)
    provider = SyntheticObservationProvider(row["actor"], seed=42, fov=120, dropout=True)
    stream = tuple(provider.observe(source.frame(i)) for i in range(start, row["query"] + 1, 5))
    reference = source.frame(row["query"])
    targets = {(m.track, m.segment): m.xy for m in reference.measurements}
    return stream, targets


def target_set(stream: tuple, memory: CausalMemory, reference: dict) -> list:
    if not stream:
        return []
    visible = {(m.track, m.segment) for m in stream[-1].measurements}
    now = stream[-1].timestamp_s
    return [token for token, entry in memory.entries.items() if token not in visible
            and now - entry.last_seen <= 2 + 1e-6 and token in reference
            and token[0] != stream[-1].actor_track]


def score_prediction(row, stream, memory, inputs, prediction, reference, targets, reason, counts):
    by_identity = {entry.identity: token for token, entry in memory.entries.items()}
    slots = ({by_identity[identity]: slot for slot, identity in inputs.identities.items()
              if identity in by_identity} if inputs is not None else {})
    now = stream[-1].timestamp_s
    reference_actor = next((np.asarray(xy) for (track, _), xy in reference.items() if track == row["actor"]), None)
    observed_actor = np.asarray(stream[-1].actor.xy) if stream[-1].actor is not None else None
    evaluated = []
    for token in targets:
        entry = memory.entries.get(token)
        truth = np.asarray(reference[token], float)
        target = {"key": f"{token[0]}:{token[1]}", "error": None, "last_seen_error": None,
                  "velocity_error": None, "covered_95": None}
        if entry is not None:
            age = now - entry.last_seen
            last = np.asarray(entry.measurement.xy)
            target["last_seen_error"] = float(np.linalg.norm(last - truth))
            target["velocity_error"] = float(np.linalg.norm(last + entry.velocity * age - truth))
        if prediction["world"] is not None and token in slots:
            slot = slots[token]
            mean, std = prediction["world"]["mean"][slot], prediction["world"]["std"][slot]
            target["error"] = float(np.linalg.norm(mean - truth))
            target["relative_error"] = (float(np.linalg.norm((mean - observed_actor) - (truth - reference_actor)))
                                        if observed_actor is not None and reference_actor is not None else None)
            target["covered_95"] = bool(np.sum(((mean - truth) / np.maximum(std, 1e-6)) ** 2) <= 5.991464547)
            target["sigma_mean_m"] = float(np.mean(std))
        evaluated.append(target)
    action = prediction["action"]
    return {"id": row["id"], "match": row["match"], "half": row["half"], "query": row["query"],
            "block": f"{row['half']}:{int(row['query'] / 1500)}", "label": row["label"],
            "weight": row["weight"], "prediction": int(np.argmax(action)) if action is not None else None,
            "reason": reason if inputs is None else prediction["reason"],
            "g1": evaluated, "corruption_counts": counts,
            "focal_available": bool(stream[-1].actor is not None)}


def f1_metrics(labels, predictions, weights=None) -> dict:
    labels, predictions = np.asarray(labels, int), np.asarray(predictions, int)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights)
    matrix = np.zeros((7, 7), float)
    for a, b, weight in zip(labels, predictions, weights, strict=True):
        matrix[a, b] += weight
    f1 = []
    for i in range(7):
        denominator = matrix[i].sum() + matrix[:, i].sum()
        f1.append(float(2 * matrix[i, i] / denominator) if denominator else 0.)
    return {"accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else None,
            "macro_f1_fixed7": float(np.mean(f1)), "confusion_weighted": matrix.tolist(),
            "per_class_f1": dict(zip(ACTION_LABELS, f1, strict=True))}


def summarize_pair(clean: list, corrupt_rows: list, *, bootstrap: bool = True) -> dict:
    if [r["id"] for r in clean] != [r["id"] for r in corrupt_rows]:
        raise ValueError("Clean/corrupt query keys differ")
    supported = [r for r in corrupt_rows if r["prediction"] is not None]
    paired = [(a, b) for a, b in zip(clean, corrupt_rows, strict=True)
              if a["prediction"] is not None and b["prediction"] is not None]
    clean_supported = sum(r["prediction"] is not None for r in clean)
    def performance(rows):
        return f1_metrics([r["label"] for r in rows], [r["prediction"] for r in rows],
                          [r["weight"] for r in rows])
    correct = sum(r["prediction"] == r["label"] for r in supported)
    total_weight = sum(r["weight"] for r in corrupt_rows)
    clean_targets, available, errors, paired_errors, last_errors, cv_errors, covered = 0, 0, [], [], [], [], []
    relative_errors = []
    for a, b in zip(clean, corrupt_rows, strict=True):
        if [t["key"] for t in a["g1"]] != [t["key"] for t in b["g1"]]:
            raise ValueError("Immutable hidden target set changed")
        clean_targets += sum(t["error"] is not None for t in a["g1"])
        for x, y in zip(a["g1"], b["g1"], strict=True):
            if y["last_seen_error"] is not None:
                last_errors.append(y["last_seen_error"])
                cv_errors.append(y["velocity_error"])
            if y["error"] is not None:
                available += 1
                errors.append(y["error"])
                covered.append(y["covered_95"])
                if y.get("relative_error") is not None:
                    relative_errors.append(y["relative_error"])
                if x["error"] is not None:
                    paired_errors.append((x["error"], y["error"]))
    clean_paired = performance([a for a, _ in paired])
    corrupt_paired = performance([b for _, b in paired])
    clean_mean = float(np.mean([x for x, _ in paired_errors])) if paired_errors else None
    corrupt_mean = float(np.mean([y for _, y in paired_errors])) if paired_errors else None
    s4_ratio = len(paired) / clean_supported if clean_supported else None
    g1_ratio = len(paired_errors) / clean_targets if clean_targets else None
    delta = corrupt_paired["macro_f1_fixed7"] - clean_paired["macro_f1_fixed7"] if paired else None
    ratio = corrupt_mean / clean_mean if clean_mean and corrupt_mean is not None else None
    totals = Counter()
    for r in corrupt_rows:
        totals.update({k: v for k, v in r["corruption_counts"].items()
                       if isinstance(v, (int, float)) and k not in ("first_event_time_s", "prefix_duration_s",
                                                                    "first_runtime_loss_elapsed_s")})
    loss_times = [r["corruption_counts"].get("first_runtime_loss_elapsed_s") for r in corrupt_rows]
    loss_times = [v for v in loss_times if v is not None]
    summary = {"scheduled_queries": len(clean), "clean_supported_queries": clean_supported,
               "corrupt_supported_queries": len(supported), "paired_supported_queries": len(paired),
               "unsupported_by_reason": dict(Counter(r["reason"] for r in corrupt_rows if r["prediction"] is None)),
               "conditional": performance(supported), "paired_clean": clean_paired, "paired_corrupt": corrupt_paired,
               "paired_macro_f1_delta": delta, "s4_availability_ratio": s4_ratio,
               "correct_and_supported": correct, "correct_and_supported_fraction_all": correct / len(clean) if clean else None,
               "weighted_correct_and_supported_fraction_all": sum(r["weight"] for r in supported if r["prediction"] == r["label"]) / total_weight if total_weight else None,
               "g1_fixed_target_opportunities": sum(len(r["g1"]) for r in clean),
               "g1_clean_supported_targets": clean_targets, "g1_corrupt_supported_targets": available,
               "g1_availability_ratio": g1_ratio, "g1_mean_error_m": float(np.mean(errors)) if errors else None,
               "g1_error_p50_p90_p95_m": np.quantile(errors, [.5, .9, .95]).tolist() if errors else None,
               "g1_paired_target_count": len(paired_errors), "g1_paired_clean_mean_m": clean_mean,
               "g1_paired_corrupt_mean_m": corrupt_mean, "g1_paired_error_ratio": ratio,
               "g1_empirical_95ellipse_coverage": float(np.mean(covered)) if covered else None,
               "g1_actor_relative_mean_error_m": float(np.mean(relative_errors)) if relative_errors else None,
               "last_seen_mean_m": float(np.mean(last_errors)) if last_errors else None,
               "constant_velocity_mean_m": float(np.mean(cv_errors)) if cv_errors else None,
               "baseline_supported_targets": len(last_errors),
               "corruption_exposure_window_counts_not_unique_fullhalf_time": dict(totals),
               "original_actor_available_at_query": sum(r["focal_available"] for r in corrupt_rows),
               "windows_with_original_actor_loss": len(loss_times),
               "time_to_first_original_actor_loss_p10_p50_p90_s": np.quantile(loss_times, [.1, .5, .9]).tolist() if loss_times else None,
               "measured_observation_position_radial_rms_m": math.sqrt(totals["position_squared_error_m2"] / totals["position_samples"]) if totals["position_samples"] else 0.,
               "measured_actor_relative_position_radial_rms_m": math.sqrt(totals["relative_position_squared_error_m2"] / totals["relative_position_samples"]) if totals["relative_position_samples"] else 0.,
               "s4_tolerance_pass": bool(s4_ratio is not None and s4_ratio >= .9 and delta is not None and delta >= -.03),
               "g1_tolerance_pass": bool(g1_ratio is not None and g1_ratio >= .9 and ratio is not None and ratio <= 1.2)}
    if bootstrap and clean:
        groups = defaultdict(list)
        for i, row in enumerate(clean):
            groups[row["block"]].append(i)
        blocks = sorted(groups)
        rng = np.random.default_rng(814)
        vectors = []
        for block in blocks:
            stats = np.zeros(7 + 49 * 2)
            for i in groups[block]:
                a, b = clean[i], corrupt_rows[i]
                stats[0] += a["prediction"] is not None
                if a["prediction"] is not None and b["prediction"] is not None:
                    stats[1] += 1
                    stats[7 + a["label"] * 7 + a["prediction"]] += a["weight"]
                    stats[56 + b["label"] * 7 + b["prediction"]] += b["weight"]
                for x, y in zip(a["g1"], b["g1"], strict=True):
                    stats[2] += x["error"] is not None
                    if x["error"] is not None and y["error"] is not None:
                        stats[3] += 1
                        stats[4] += x["error"]
                        stats[5] += y["error"]
            vectors.append(stats)
        vectors = np.stack(vectors)
        def macro(matrix):
            matrix = matrix.reshape(7, 7)
            denominator = matrix.sum(0) + matrix.sum(1)
            return float(np.mean(np.divide(2 * matrix.diagonal(), denominator,
                                          out=np.zeros(7), where=denominator > 0)))
        values = []
        for _ in range(200):
            sampled = vectors[rng.integers(0, len(blocks), len(blocks))].sum(0)
            values.append([macro(sampled[56:]) - macro(sampled[7:56]) if sampled[1] else None,
                           sampled[1] / sampled[0] if sampled[0] else None,
                           sampled[5] / sampled[4] if sampled[4] else None,
                           sampled[3] / sampled[2] if sampled[2] else None])
        summary["paired_temporal_block_bootstrap_95"] = {
            key: np.quantile([v[i] for v in values if v[i] is not None], [.025, .975]).tolist()
            if any(v[i] is not None for v in values) else None
            for i, key in enumerate(("s4_f1_delta", "s4_availability_ratio", "g1_error_ratio", "g1_availability_ratio"))}
        summary["temporal_blocks_60s"] = len(blocks)
    return summary


def gz_write(path: Path, value) -> None:
    temporary = path.with_suffix(".partial")
    with gzip.open(temporary, "wt", encoding="utf-8") as file:
        json.dump(value, file, allow_nan=False, separators=(",", ":"))
    temporary.replace(path)


def gz_read(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def sensitivity(output: Path, protocol: dict, check, status) -> dict:
    rows, evidence = population()
    write_json(output / "population_candidates.json", {"rows": rows, "evidence": evidence})
    models = FrozenBatch()
    source_cache = {}
    def source_for(row):
        key = row["match"], row["half"]
        if key not in source_cache:
            source_cache[key] = SoccerTrackSource(ROOT, *key)
        return source_cache[key]
    pilot_rows = [r for r in rows if r["match"] == "132877" and r["query"] >= 500][:16]
    started = time.monotonic()
    streams = [clean_stream(source_for(r), r)[0] for r in pilot_rows]
    values = [prepare(s)[0] for s in streams]
    parity = models.parity(values)
    write_json(output / "gpu_parity.json", parity)
    control = {}
    for name in ("clean", "rename", "permutation", "translation"):
        inputs = [prepare(corrupt(s, {"family": name, "severity": 1. if name != "clean" else 0.}, 42)[0])[0] for s in streams]
        outputs = models.predict(inputs)
        control[name] = outputs
    maxima = {}
    for name in ("rename", "permutation", "translation"):
        differences = []
        for a, b in zip(control["clean"], control[name], strict=True):
            if a["action"] is None or b["action"] is None:
                if (a["action"] is None) != (b["action"] is None):
                    raise AssertionError(f"{name} changed contract support")
                continue
            differences.append(float(np.max(np.abs(a["action"] - b["action"]))))
        maxima[name] = max(differences, default=0.)
    # Translation/permutation are diagnostics, not assumed invariant architectures.
    if maxima["rename"] > .0002:
        raise AssertionError("Consistent one-time token rename failed")
    write_json(output / "invariance_controls.json", {"max_S4_probability_difference": maxima,
               "meaning": "Renaming required invariant. Slot-order and relative-translation differences measured, not silently equated to temporal swaps.",
               "absolute_translation_m": 1., "relative_translation_error_m": 0.})
    for stream in streams[:4]:
        a = prepare(stream, optimized=True)[0]
        b = prepare(stream, optimized=False)[0]
        if (a is None) != (b is None) or a is not None and not np.array_equal(a.x, b.x):
            raise AssertionError("Optimized causal encoding differs from released memory")
    trial_condition = {"family": "noise", "severity": 1.}
    timer = time.monotonic()
    for stream in streams:
        prepare(corrupt(stream, trial_condition, 42)[0])
    seconds_per_window = (time.monotonic() - timer) / len(streams)
    available = max(0., (datetime.fromisoformat(protocol["reserve_utc"]) - datetime.now(timezone.utc)).total_seconds())
    # Leave 25% for scoring/bootstrap/charts and protect final verification reserve.
    cost = max(.005, seconds_per_window) * len(conditions()) * len(protocol["seeds"])
    cap = min(len(rows), max(42, int(available * .75 / cost)))
    population_path = output / "population_lock.json"
    if population_path.exists():
        locked = read(population_path)
        selected = locked["rows"]
        if locked["candidate_hash"] != file_digest(output / "population_candidates.json"):
            raise ValueError("Population changed across resume")
    else:
        selected = select_population(rows, cap)
        write_json(population_path, {"rows": selected, "candidate_count": len(rows),
                   "candidate_hash": file_digest(output / "population_candidates.json"),
                   "pilot_seconds_per_corrupt_prepare": seconds_per_window, "selected_count": len(selected),
                   "pilot_total_s": time.monotonic() - started, "before_full_predictions": True,
                   "absent_classes": [c for i, c in enumerate(ACTION_LABELS) if not any(r["label"] == i for r in selected)],
                   "rule": "Hash-ranked per match/half/class, proportional inclusion weights; independent of predictions."})
    directory = output / "query_metrics"
    directory.mkdir(exist_ok=True)
    by_source = defaultdict(list)
    for row in selected:
        by_source[row["match"], row["half"]].append(row)
    for (match, half), group in by_source.items():
        check()
        source = source_for(group[0])
        for start in range(0, len(group), 128):
            check()
            batch = group[start:start + 128]
            clean_data = [clean_stream(source, row) for row in batch]
            clean_prepared = [prepare(stream) for stream, _ in clean_data]
            targets = [target_set(stream, state[1], reference)
                       for (stream, reference), state in zip(clean_data, clean_prepared, strict=True)]
            base_path = directory / f"{match}_{half}_{start:05d}_clean.json.gz"
            if base_path.exists():
                base = gz_read(base_path)
            else:
                prediction = models.predict([state[0] for state in clean_prepared])
                base = [score_prediction(row, data[0], state[1], state[0], pred, data[1], target, state[2], {})
                        for row, data, state, pred, target in zip(batch, clean_data, clean_prepared, prediction, targets, strict=True)]
                gz_write(base_path, base)
            for seed in protocol["seeds"]:
                for condition in conditions()[1:]:
                    check()
                    name = condition_name(condition)
                    path = directory / f"{match}_{half}_{start:05d}_{name}_s{seed}.json.gz"
                    if path.exists():
                        continue
                    changed = [corrupt(stream, condition, seed) for stream, _ in clean_data]
                    states = [prepare(stream) for stream, _ in changed]
                    predictions = models.predict([state[0] for state in states])
                    record = [score_prediction(row, transformed[0], state[1], state[0], pred,
                                               data[1], target, state[2], transformed[1])
                              for row, transformed, state, pred, data, target in
                              zip(batch, changed, states, predictions, clean_data, targets, strict=True)]
                    gz_write(path, record)
            status(current_job="sensitivity", progress={"match": match, "half": half, "completed_queries": start + len(batch),
                   "source_queries": len(group), "gpu_elements": dict(models.elements)})
        # Sources are large; do not retain all raw arrays when moving to next half.
        source_cache.pop((match, half), None)
    workload = {"status": "INFERENCE_COMPLETE", "query_count": len(selected),
                "conditions": conditions(), "seeds": protocol["seeds"],
                "actual_gpu_batch_elements": dict(models.elements), "actual_gpu_forward_calls": dict(models.calls),
                "device": "cuda:0", "gpu": torch.cuda.get_device_name(0),
                "initial_parity": parity, "index_sha256": file_digest(INDEX)}
    write_json(output / "inference_workload.json", workload)
    del models
    torch.cuda.empty_cache()
    return workload


def report(output: Path, protocol: dict, check) -> dict:
    population_lock = read(output / "population_lock.json")
    directory = output / "query_metrics"
    summary = {}
    envelope = []
    for match in ("132877", "118577", "128058"):
        clean = []
        for half in ("1st", "2nd"):
            for path in sorted(directory.glob(f"{match}_{half}_*_clean.json.gz")):
                clean.extend(gz_read(path))
        if not clean:
            continue
        clean.sort(key=lambda r: r["id"])
        summary[f"{match}/clean"] = summarize_pair(clean, clean)
        for seed in protocol["seeds"]:
            for condition in conditions()[1:]:
                check()
                name = condition_name(condition)
                changed = []
                for half in ("1st", "2nd"):
                    for path in sorted(directory.glob(f"{match}_{half}_*_{name}_s{seed}.json.gz")):
                        changed.extend(gz_read(path))
                changed.sort(key=lambda r: r["id"])
                if [r["id"] for r in clean] != [r["id"] for r in changed]:
                    summary[f"{match}/{name}/s{seed}"] = {"status": "PARTIAL_MISSING_BATCHES",
                                                       "completed": len(changed), "expected": len(clean)}
                    continue
                metrics = summarize_pair(clean, changed)
                summary[f"{match}/{name}/s{seed}"] = metrics
                envelope.append({"match": match, "seed": seed, **condition,
                                 **{key: metrics[key] for key in ("s4_tolerance_pass", "g1_tolerance_pass",
                                     "s4_availability_ratio", "g1_availability_ratio", "paired_macro_f1_delta",
                                     "g1_paired_error_ratio")}})
            write_json(output / "sensitivity_metrics.json", summary)
    write_json(output / "tolerance_table.json", {"rows": envelope,
               "criteria": protocol["tolerance"], "population_selected": population_lock["selected_count"],
               "warning": "Empirical discrete tested levels only; do not interpolate safety thresholds. Seed variation and paired block intervals in sensitivity_metrics.json."})
    ranges = {}
    for family in sorted({r["family"] for r in envelope}):
        ranges[family] = {}
        for match in ("132877", "118577", "128058"):
            observed = [r for r in envelope if r["family"] == family and r["match"] == match]
            levels = sorted({r["severity"] for r in observed})
            ranges[family][match] = {
                model: [level for level in levels if len([r for r in observed if r["severity"] == level]) == 3
                        and all(r[model + "_tolerance_pass"] for r in observed if r["severity"] == level)]
                for model in ("s4", "g1")}
    focal = [r for r in envelope if r["family"] in ("focal_gap", "focal_fragment")
             and r["match"] in ("118577", "128058")]
    noise = [r for r in envelope if r["family"] in ("noise", "drift")
             and r["severity"] <= .5 and r["match"] in ("118577", "128058")]
    priority = ("focal continuity / detection coverage" if any(not r["s4_tolerance_pass"] for r in focal)
                else "projection / position accuracy" if any(not r["g1_tolerance_pass"] for r in noise)
                else "downstream robustness and clean-baseline adequacy")
    write_json(output / "decision.json", {
        "priority": priority, "tested_levels_passing_all_three_seeds": ranges,
        "basis": "Prioritize original-focal serviceability if tested focal losses violate the predeclared availability/F1 envelope; otherwise small-position-error G1 sensitivity, otherwise clean downstream quality. No automatic release change.",
        "clean": {k: {"S4_supported": v["clean_supported_queries"], "S4_scheduled": v["scheduled_queries"],
                      "S4_conditional_accuracy": v["conditional"]["accuracy"],
                      "S4_conditional_macro_f1": v["conditional"]["macro_f1_fixed7"],
                      "G1_mean_error_m": v["g1_mean_error_m"],
                      "last_seen_mean_m": v["last_seen_mean_m"],
                      "constant_velocity_mean_m": v["constant_velocity_mean_m"]}
                  for k, v in summary.items() if k.endswith("/clean")},
        "caution": "Retaining inadequate clean performance is not coaching adequacy. Same-kit retrieval gains alone did not establish tracking improvement. No G1-to-S4 link or tactical-value claim."})
    charts(output, summary)
    result = {"status": "COMPLETE" if all("status" not in v for v in summary.values()) and len(envelope) == 3 * 3 * (len(conditions()) - 1) else "PARTIAL",
              "metric_groups": len(summary), "table_rows": len(envelope),
              "optional_source_level": "NOT_RUN; fixed-visibility injection is the declared main study",
              "not_first_stress_study": True}
    write_json(output / "sensitivity_summary.json", result)
    return result


def charts(output: Path, summary: dict) -> None:
    # SVG charts need no optional plotting dependency.
    families = sorted({c["family"] for c in conditions() if c["family"] != "clean"})
    for family in families:
        levels = sorted({c["severity"] for c in conditions() if c["family"] == family})
        pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="600">',
                  '<rect width="100%" height="100%" fill="white"/>',
                  f'<text x="20" y="24" font-size="18">{family}: frozen visibility, pre-memory corruption</text>']
        for panel, metric in enumerate(("s4_availability_ratio", "paired_macro_f1_delta", "g1_availability_ratio", "g1_paired_error_ratio")):
            left, top = 60 + (panel % 2) * 510, 70 + (panel // 2) * 260
            points_by_match = {}
            for match in ("132877", "118577", "128058"):
                points_by_match[match] = []
                for level in levels:
                    values = [summary.get(f"{match}/{family}_{level:g}/s{seed}", {}).get(metric)
                              for seed in (42, 137, 901)]
                    values = [v for v in values if v is not None]
                    if values:
                        points_by_match[match].append((level, float(np.mean(values)), min(values), max(values)))
            all_values = [v for points in points_by_match.values() for _, mean, lo, hi in points for v in (mean, lo, hi)]
            lo, hi = min([0., *all_values]), max([1., *all_values])
            if hi == lo:
                hi += 1
            pieces += [f'<text x="{left}" y="{top-15}" font-size="13">{metric}</text>',
                       f'<path d="M{left},{top} v180 h400" fill="none" stroke="black"/>']
            for match, color in zip(points_by_match, ("#1462a0", "#bf5420", "#308530"), strict=True):
                coords = []
                for level, mean, lower, upper in points_by_match[match]:
                    x = left + 10 + 380 * (level - levels[0]) / max(1e-8, levels[-1] - levels[0])
                    y = top + 180 - 175 * (mean - lo) / (hi - lo)
                    y1 = top + 180 - 175 * (lower - lo) / (hi - lo)
                    y2 = top + 180 - 175 * (upper - lo) / (hi - lo)
                    coords.append(f"{x:.2f},{y:.2f}")
                    pieces.append(f'<path d="M{x},{y1} V{y2}" stroke="{color}" stroke-width="2"/>')
                    pieces.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
                joined = " ".join(coords)
                pieces.append(f'<polyline points="{joined}" fill="none" stroke="{color}"/>')
            pieces += [f'<text x="{left}" y="{top+200}" font-size="11">severity {levels}; y[{lo:.3g},{hi:.3g}]</text>']
        pieces += ['<text x="20" y="575" font-size="12">Blue132877(dev), orange118577, green128058(retrospective). Bars=3-seed range; block CIs in JSON. No camera-to-ego certification.</text>', '</svg>']
        (output / f"curve_{family}.svg").write_text("\n".join(pieces), encoding="utf-8")


class Worker:
    def __init__(self, output: Path):
        self.output = output
        self.protocol = read(output / "protocol.json")
        self.protocol_hash = file_digest(output / "protocol.json")
        path = output / "task_state.json"
        self.state = read(path) if path.exists() else {"status": "IMPLEMENTING", "jobs": {}, "blockers": [],
                     "started_utc": self.protocol["started_utc"], "protocol_sha256": self.protocol_hash}
        if self.state["protocol_sha256"] != self.protocol_hash:
            raise ValueError("Locked protocol changed")

    def update(self, **values):
        self.state.update(values, heartbeat_utc=utc(), pid=__import__("os").getpid(),
                          elapsed_seconds=(datetime.now(timezone.utc) - datetime.fromisoformat(self.protocol["started_utc"])).total_seconds())
        write_json(self.output / "task_state.json", self.state)

    def check(self, association=False, final=False):
        key = "deadline_utc" if final else "association_cutoff_utc" if association else "reserve_utc"
        if datetime.now(timezone.utc) >= datetime.fromisoformat(self.protocol[key]):
            raise TimeoutError(key)

    def stage(self, name, function, *, association=False):
        job = self.state["jobs"].get(name, {})
        if job.get("status") == "COMPLETE":
            path = self.output / job["result"]
            if file_digest(path) != job["sha256"]:
                raise ValueError("Completed stage evidence changed")
            return read(path)
        for attempt in range(job.get("attempts", 0), 3):
            self.check(association=association)
            self.state["jobs"][name] = {"status": "RUNNING", "attempts": attempt + 1}
            self.update(status="RUNNING", current_job=name)
            try:
                value = function()
                path = self.output / f"{name}_result.json"
                write_json(path, value)
                self.state["jobs"][name].update(status="COMPLETE", result=path.name, sha256=file_digest(path))
                self.update()
                return value
            except TimeoutError as error:
                self.state["jobs"][name].update(status="PARTIAL", reason=str(error))
                self.update()
                if association:
                    return None
                raise
            except Exception as error:
                (self.output / f"{name}_error_{attempt+1}.txt").write_text(traceback.format_exc(), encoding="utf-8")
                self.update(last_error={"stage": name, "attempt": attempt + 1, "error": str(error)})
                # Retry transient I/O once; deterministic failures preserve evidence and advance.
                if not isinstance(error, OSError) or attempt == 2:
                    self.state["jobs"][name].update(status="BLOCKED", reason=str(error))
                    self.state["blockers"].append({"stage": name, "error": str(error)})
                    self.update()
                    return None
        return None

    def pins(self) -> dict:
        manifest_path = ROOT / "configs" / "backbone_release_manifest.json"
        manifest = read(manifest_path)
        pins = {str(manifest_path.relative_to(ROOT)): file_digest(manifest_path)}
        for name in ("default_world", "default_action"):
            item = manifest["models"][name]
            if file_digest(ROOT / item["checkpoint"]) != item["sha256"]:
                raise ValueError("Protected release checkpoint mismatch")
            pins[item["checkpoint"]] = item["sha256"]
        paths = ["configs/backbone_release_baseline.json", "outputs/ip_20261029_202245/selection.json",
                 "outputs/ip_20261029_202245/summary.json", "src/backbone/identity_program.py"]
        for path in paths:
            pins[path] = file_digest(ROOT / path)
        prior = ROOT / "outputs" / "ip_20261029_202245"
        for path in (prior / "R1" / "best.pt", prior / "R2" / "best.pt",
                     ROOT / "outputs" / "pc_20261029_144726" / "selected.pt"):
            pins[str(path.relative_to(ROOT))] = file_digest(path)
        index = read(INDEX)
        for shard in index["shards"]:
            if shard["match"] in ("132877", "118577", "128058"):
                for name in ("student_metadata.jsonl", "optional_targets.jsonl"):
                    path = Path(shard["folder"]) / name
                    pins[str(path.relative_to(ROOT))] = file_digest(path)
        return pins

    def verify(self):
        self.check(final=True)
        pins = read(self.output / "protected_before.json")
        if any(file_digest(ROOT / name) != digest for name, digest in pins.items()):
            raise ValueError("Protected models/annotations/history changed")
        result = subprocess.run([sys.executable, "-m", "pytest", "-q",
                  "tests/test_association_sensitivity.py", "tests/test_actor_continuity.py",
                  "tests/test_identity_program.py", "tests/test_backbone_release.py",
                  "--junitxml=" + str(self.output / "tests.xml")],
                  cwd=ROOT, capture_output=True, text=True, timeout=300)
        (self.output / "tests.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise AssertionError("Focused verification failed")
        return {"status": "PASS", "protected_unchanged": True, "tests": result.stdout.strip().splitlines()[-1]}

    def package(self):
        self.check(final=True)
        complete = all(self.state["jobs"].get(name, {}).get("status") == "COMPLETE"
                       for name in ("association", "sensitivity", "report", "verify"))
        if (self.output / "sensitivity_summary.json").exists():
            complete = complete and read(self.output / "sensitivity_summary.json")["status"] == "COMPLETE"
        terminal = "COMPLETE" if complete else "PARTIAL"
        destination = Path.home() / "Downloads" / f"return_tolerance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        patch = []
        for name in SOURCES:
            path = ROOT / name
            target = self.output / "changed" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            patch.extend(difflib.unified_diff([], path.read_text().splitlines(keepends=True),
                                             fromfile="/dev/null", tofile="b/" + name))
        (self.output / "source.patch").write_text("".join(patch), encoding="utf-8")
        command = f"& .\\.venv\\Scripts\\python.exe -u -m src.backbone.perception_sensitivity --output {self.output.relative_to(ROOT)}"
        association = read(self.output / "association_development.json") if (self.output / "association_development.json").exists() else {}
        metrics = read(self.output / "sensitivity_metrics.json") if (self.output / "sensitivity_metrics.json").exists() else {}
        clean = {k: v for k, v in metrics.items() if k.endswith("/clean")}
        decision = read(self.output / "decision.json") if (self.output / "decision.json").exists() else {
            "priority": "BLOCKED: inspect preserved failures before making a perception investment"}
        handoff = ["# Association and downstream tolerance", f"Execution: {terminal}. No training/default changes.",
                   "Scientific outcome is distinct from execution completion. Previous identity negative results remain unchanged.",
                   f"Association development selected experimental comparison: {association.get('selected', 'NOT_COMPLETED')}. Production baseline unchanged.",
                   "See association_summary.json for lifecycle repair and identical-mechanics comparisons; reference labels never enter assignment.",
                   "See tolerance_table.json for discrete empirical pass/fail levels and sensitivity_metrics.json for per-match/seed paired scores, failures and60s block CIs.",
                   "Clean baseline:", json.dumps({k: {m: v.get(m) for m in ('scheduled_queries', 'clean_supported_queries',
                       'conditional', 'g1_clean_supported_targets', 'g1_mean_error_m')} for k, v in clean.items()}),
                   "Bounded8s prefix, fixed simulated visibility, all conditions injected before memory. Reference tracking is not certified physical truth.",
                   "Fixed original focal token. Fragmented focal IDs never rebound. G1 scoring uses fixed original token/segment, no nearest-person rescue. G1 never feeds S4.",
                   "Exposure counts sum overlapping query windows, not unique full-half player-time. No conversion to calibrated real-video swap probabilities.",
                   "Common translation is a relative-coordinate diagnostic; shared drift/noise are controlled synthetic conditions, not source calibration.",
                   "Measured next decision: " + json.dumps(decision),
                   f"Blockers: {self.state['blockers']}. Required unrun branches remain PARTIAL.",
                   "## Commands", "```powershell", command + " run", command + " resume", command + " verify", "```",
                   "Resume only with stopped worker; unchanged deadlines, deterministic per-batch caches. Terminal status saved BEFORE ZIP. No survival across reboot/sleep promised."]
        (self.output / "handoff.md").write_text("\n\n".join(handoff), encoding="utf-8")
        self.update(status=terminal, current_job=None, return_zip=str(destination), finished_utc=utc(),
                    scientific_outcome="See matched association and per-match tolerance metrics; execution is not improvement")
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.output.rglob("*")):
                if path.is_file() and (path.suffix in (".json", ".md", ".txt", ".xml", ".svg", ".patch", ".py")
                                       and "query_metrics" not in path.parts):
                    archive.write(path, str(path.relative_to(self.output)))
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None or read(self.output / "task_state.json")["status"] != terminal:
                raise AssertionError("ZIP verification failed")
        write_json(self.output / "package_verification.json", {"status": "PASS", "zip": str(destination),
                   "sha256": file_digest(destination), "bytes": destination.stat().st_size,
                   "terminal_status_before_zip": terminal})

    def run(self):
        with ExclusiveLock(self.output / "worker.lock"):
            contract = {"protocol_sha256": self.protocol_hash,
                        "source_sha256": {name: file_digest(ROOT / name) for name in SOURCES}}
            contract_path = self.output / "worker_contract.json"
            if contract_path.exists() and read(contract_path) != contract:
                raise ValueError("Worker source/protocol changed; do not reuse cached results silently")
            write_json(contract_path, contract)
            self.update(status="RUNNING", command=" ".join(sys.argv))
            before = self.output / "protected_before.json"
            if not before.exists():
                write_json(before, self.pins())
            write_json(self.output / "missing_feature_reproduction.json", synthetic_reproduction())
            try:
                self.stage("association", lambda: run_association(self.output, self.check), association=True)
                self.stage("sensitivity", lambda: sensitivity(self.output, self.protocol, self.check, self.update))
                if (self.output / "population_lock.json").exists():
                    self.stage("report", lambda: report(self.output, self.protocol, self.check))
                self.update(stop_reason="REQUIRED_USEFUL_WORK_EXHAUSTED")
            except TimeoutError:
                self.update(stop_reason="FINAL_VERIFICATION_RESERVE")
            except Exception as error:
                (self.output / "worker_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                self.state["blockers"].append({"stage": "orchestrator", "error": str(error)})
            try:
                result = self.verify()
                path = self.output / "verify_result.json"
                write_json(path, result)
                self.state["jobs"]["verify"] = {"status": "COMPLETE", "result": path.name, "sha256": file_digest(path)}
            except Exception as error:
                self.state["blockers"].append({"stage": "verify", "error": str(error)})
                (self.output / "verify_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            self.package()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", choices=("run", "resume", "verify", "reproduce"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2 = __import__("cv2")
    cv2.setNumThreads(2)
    worker = Worker(args.output.resolve())
    if args.command == "reproduce":
        write_json(worker.output / "missing_feature_reproduction.json", synthetic_reproduction())
        worker.update(status="IMPLEMENTING")
    elif args.command == "verify":
        with ExclusiveLock(worker.output / "worker.lock"):
            write_json(worker.output / "verify_result.json", worker.verify())
    else:
        worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
