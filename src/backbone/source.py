"""SoccerTrack adapter and strictly separated full-state reference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.backbone.contracts import Measurement, TrackingFrame
from src.trajectory_dataset import file_digest


class SoccerTrackSource:
    def __init__(self, root: Path, match: str, half: str):
        if half not in ("1st", "2nd"):
            raise ValueError("SoccerTrack half must be 1st or 2nd")
        self.match, self.half = str(match), half
        path = root / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{match}_{half}.npz"
        association_path = root / "outputs" / "trajectory_audit_20261016" / f"associations_{match}_{half}.json"
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        association = json.loads(association_path.read_text(encoding="utf-8"))
        digest = file_digest(path)
        if digest != metadata["cache_sha256"] or digest != association["cache_sha256"]:
            raise ValueError(f"Unverified tracking cache: {path}")
        with np.load(path, allow_pickle=False) as archive:
            self.raw = {key: archive[key] for key in ("xy", "observed", "segments", "teams", "roles")}
        self.provenance = {
            "adapter": "SoccerTrackSource", "source_id": f"{match}_{half}", "match": self.match,
            "half": half, "cache_path": str(path), "cache_sha256": digest,
            "association_sha256": file_digest(association_path), "original": metadata["source"],
            "units": "metres", "coordinate_system": "pitch-centred x[-52.5,52.5], y[-34,34]",
            "fps": 25, "missingness": "observed=False, no interpolation; discontinuities use segment IDs",
            "missing_count_semantics": "Absent cache slots, possibly unused roster slots; NOT a count of physically missing players",
            "quality": "Quantized tracking, not guaranteed physical full state; actor is explicitly configured",
        }

    def frame(self, index: int) -> TrackingFrame:
        if not 0 <= index < len(self.raw["xy"]):
            raise IndexError(f"Frame outside source: {index}")
        valid = self.raw["observed"][index]
        measurements = tuple(Measurement(
            f"track_{slot}", int(self.raw["segments"][index, slot]),
            tuple(float(v) for v in self.raw["xy"][index, slot]),
            int(self.raw["teams"][index, slot]), bool(self.raw["roles"][index, slot]))
            for slot in np.flatnonzero(valid))
        return TrackingFrame(self.provenance["source_id"], self.half, index, index / 25,
                             measurements, int((~valid).sum()))

    def frames(self, start: int, count: int, stride: int = 5):
        if count < 1 or stride != 5:
            raise ValueError("This adapter requires positive count and 5-frame/0.2s sampling")
        for index in range(start, start + count * stride, stride):
            yield self.frame(index)


class EvaluationReference:
    """Privileged states never cross into ReplayEngine.step(Observation)."""

    def __init__(self, source: SoccerTrackSource):
        self.source = source

    def overlay(self, frame: TrackingFrame):
        return {"label": "EVALUATION ONLY: complete tracked state, not the player's map",
                "players": [{"source_track": m.track, "xy": list(m.xy), "team": m.team}
                            for m in frame.measurements],
                "absent_cache_slots": frame.missing_count}

    def receiver_states(self, index: int, actor_track: str):
        from src.receiver_forecast_adapter import asof_base
        slot = int(actor_track.removeprefix("track_"))
        built = asof_base(self.source.raw, {actor_track: slot},
                          {"passer_id": actor_track, "receiver_id": "NO_PROXY_TARGET"}, index)
        if built is None:
            return None
        states, mapping, _ = built
        return states, {label: f"track_{value}" for label, value in mapping.items()}


def load_source(root: Path, specification: dict, adapter: str = "soccertrack_cache"):
    if adapter == "external_jsonl":
        from src.backbone.external import ExternalObservationSource
        return ExternalObservationSource(root / specification["input_path"])
    if adapter != "soccertrack_cache":
        raise ValueError(f"Source adapter not implemented: {adapter}; no fabricated replacement")
    return SoccerTrackSource(root, str(specification["match"]), specification["half"])
