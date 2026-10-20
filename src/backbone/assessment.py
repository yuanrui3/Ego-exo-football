"""Known-entity Gaussian geometry, not a pass-success or optimal-action policy."""

from __future__ import annotations

import numpy as np

from src.hidden_defender_geometry import probability_from_samples


class GaussianAssessor:
    def __init__(self, samples=128, seed=42):
        if samples < 16:
            raise ValueError("At least 16 Gaussian samples are required")
        self.samples, self.seed = samples, seed

    def assess(self, belief, sample_id):
        result = {
            "status": "UNAVAILABLE",
            "unavailable_reason": "No candidate has the required observed endpoint and defender evidence",
            "label": "Provisional geometric ordering—not a trained optimal-pass policy.",
            "thresholds_m": [2, 3, 5], "endpoint_exclusion_m": 2,
            "coverage": "Visible/remembered defenders only; anonymous occupancy displayed separately, NOT assumed empty",
            "candidates": [], "warnings": ["A clear estimated lane is not proof of pass success."],
        }
        if not belief["actor_available"] or belief["actor_team"] not in (1, 2):
            result["warnings"].append("No observed actor with known team: candidate assessments unavailable")
            result["unavailable_reason"] = "No observed actor with a known team"
            return result
        actor = np.asarray(belief["actor_xy"], np.float32)
        entities = belief["entities"]
        defenders = [entity for entity in entities if entity["team"] != belief["actor_team"] and entity["team"] != 0]
        for candidate in entities:
            if candidate["team"] != belief["actor_team"] or candidate["actor"]:
                continue
            row = {"id": candidate["id"], "xy": candidate["xy"], "visible": candidate["visible"],
                   "std_m": candidate["std_m"], "status": "UNAVAILABLE"}
            if not candidate["visible"]:
                row["reason"] = "Uncertain candidate endpoint unsupported by fixed-endpoint geometry"
            elif np.linalg.norm(np.asarray(candidate["xy"]) - actor) <= 4:
                row["reason"] = "Lane too short for 2m endpoint exclusions"
            elif not defenders:
                row["reason"] = "No observed/retained defender evidence; do not infer safety"
            else:
                from src.player_centric_dataset import stable_seed
                rng = np.random.default_rng(stable_seed(f"{sample_id}:{candidate['id']}", self.seed))
                means = np.asarray([d["xy"] for d in defenders], np.float32)
                sigma = np.asarray([d["std_m"] for d in defenders], np.float32)
                draws = means[None] + rng.standard_normal((self.samples, len(defenders), 2)).astype(np.float32) * sigma[None]
                valid = np.ones(len(defenders), bool)
                hidden = np.asarray([not d["visible"] for d in defenders], bool)
                prediction = probability_from_samples(actor, np.asarray(candidate["xy"], np.float32),
                                                       draws, valid, hidden)
                row.update(status="AVAILABLE", lane_probability=prediction[:3].tolist(),
                           hidden_threat_probability=prediction[3:6].tolist(),
                           expected_nearest_defender_m=float(prediction[6]), defender_evidence=len(defenders))
            result["candidates"].append(row)
        available = sorted((c for c in result["candidates"] if c["status"] == "AVAILABLE"),
                           key=lambda c: (c["lane_probability"][1], -c["expected_nearest_defender_m"], c["id"]))
        result["status"] = "AVAILABLE" if available else "UNAVAILABLE"
        if available:
            result.pop("unavailable_reason")
        result["ordering"] = [c["id"] for c in available]
        result["ordering_rule"] = "Ascending 3m obstruction probability, descending expected defender separation, anonymous ID tie-break"
        return result
