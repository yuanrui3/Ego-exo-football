"""Load completed checkpoints once, preserving existing forward contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.backbone.contracts import WorldInput
from src.models.candidate_ranker_receiver import CandidateRankerReceiver
from src.models.dynamic_football_gnn import GraphWorldModel
from src.trajectory_dataset import file_digest


class ModelSession:
    def __init__(self, experiment: Path, backend: str = "G1", exo_p0: bool = False, device: str = "cpu"):
        keys = {"G1": "world_baseline", "G3": "world_gnn_alternative", "G4": "belief_gnn_alternative"}
        if backend not in (*keys, "last_seen", "constant_velocity"):
            raise ValueError("Unknown belief backend")
        self.backend, self.device = backend, device
        self.world = self.p0 = None
        self.provenance = {}
        self.load_counts = {"world": 0, "p0": 0}
        manifest = json.loads((experiment / "integration_manifest.json").read_text())
        for name, key in ((backend, keys.get(backend)), ("P0", "receiver_baseline" if exo_p0 else None)):
            if key is None:
                continue
            record = manifest["selected_checkpoints"][key]
            path = experiment / record["path"]
            if not path.is_file() or file_digest(path) != record["sha256"]:
                raise ValueError(f"Missing or changed selected checkpoint: {path}")
            if not (path.parent / "result.json").exists():
                raise ValueError("An unfinished checkpoint cannot serve the prototype")
            config = json.loads((path.parent / "config.json").read_text())
            if config["fitting_matches"] != record["fitting_matches"]:
                raise ValueError("Checkpoint fitting provenance mismatch")
            saved = torch.load(path, weights_only=False, map_location="cpu")
            self.provenance[name] = {**record, "resolved_path": str(path),
                                     "config": config["config"], "development_only": config["development_matches"]}
            if name == "P0":
                statistics = {k: saved["model"][k] for k in (
                    "feature_mean", "feature_std", "scene_mean", "scene_std")}
                self.p0 = CandidateRankerReceiver(statistics).to(device).eval()
                self.p0.load_state_dict(saved["model"])
                self.load_counts["p0"] += 1
            else:
                if config["config"]["seconds"] != 2:
                    raise ValueError("Only completed 2s world models supported")
                self.world = GraphWorldModel(config["config"]).to(device).eval()
                self.world.load_state_dict(saved["model"])
                self.load_counts["world"] += 1

    @torch.inference_mode()
    def predict(self, inputs: WorldInput):
        if self.world is None:
            raise ValueError("No learned backend selected")
        batch = {key: value.to(self.device) for key, value in inputs.tensors().items()}
        output = self.world(**batch)
        result = {key: value.float().cpu().numpy()[0] for key, value in (
            ("mean", output["mean"]), ("std", output["log_std"].exp()),
            ("occupancy", output["occupancy_logits"].sigmoid()),
            ("anonymous_defender", output["anonymous_defender_logits"].sigmoid()))}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Non-finite learned belief; replay stopped, no silent fallback")
        return result

    @torch.inference_mode()
    def exo_reference(self, states: np.ndarray, mapping: dict):
        if self.p0 is None:
            return {"status": "DISABLED"}
        value = torch.from_numpy(states[None].astype(np.float32)).to(self.device)
        probabilities = self.p0(value).float().softmax(-1).cpu().numpy()[0]
        ranking = [{"source_track": mapping[f"TEAMMATE_{i+1}"], "probability": float(probabilities[i])}
                   for i in np.argsort(-probabilities) if f"TEAMMATE_{i+1}" in mapping]
        return {"label": "Full-state next-event receiver proxy.", "role": "EXO REFERENCE ONLY, not player's recommendation",
                "ranking": ranking, "status": "AVAILABLE"}
