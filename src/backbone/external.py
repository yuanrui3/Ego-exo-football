"""Streaming public observation envelope around the existing backbone records."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

from src.backbone.contracts import Measurement, Observation, TrackingFrame
from src.backbone.observation import SyntheticObservationProvider
from src.trajectory_dataset import atomic_replace, file_digest

VERSION = "backbone.observations.0.2"
ROOT = Path(__file__).resolve().parents[2]


def read_object(line: str, number: int) -> dict:
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    try:
        value = json.loads(line, parse_constant=reject_constant)
    except ValueError as error:
        raise ValueError(f"JSONL line {number}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSONL line {number}: expected an object")
    return value


class ExternalObservationSource:
    """No interpolation; each packet contains only positions actually supplied."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        with self.path.open(encoding="utf-8") as stream:
            header = read_object(stream.readline(), 1)
        if header.get("schema_version") != VERSION or header.get("record_type") != "metadata":
            raise ValueError(f"Expected {VERSION} metadata as first JSONL record")
        coordinates = header.get("coordinates", {})
        if not isinstance(coordinates, dict):
            raise ValueError("coordinates metadata must be an object")
        if coordinates.get("units") != "metres":
            raise ValueError("External input requires explicit units=metres; pixels cannot enter the pitch backbone")
        if coordinates.get("system") not in ("pitch_centred", "pitch_top_left"):
            raise ValueError("Unsupported coordinate system; no implicit camera-to-pitch projection")
        if coordinates.get("pitch_length_m") != 105 or coordinates.get("pitch_width_m") != 68:
            raise ValueError("Frozen model supports a declared 105m x 68m pitch only; no implicit rescaling")
        if coordinates.get("axes") != "x_length_y_width":
            raise ValueError("Declare axes=x_length_y_width (positive right/down for top-left coordinates)")
        if header.get("observation_kind") not in ("exo_source", "player_visible"):
            raise ValueError("Observation kind must be exo_source or player_visible; privileged data requires a separate export")
        for key in ("source_id", "match", "half", "origin", "calibration", "synchronization", "quality"):
            if not header.get(key):
                raise ValueError(f"Missing external metadata: {key}")
        for key in ("source_id", "half"):
            if not isinstance(header[key], str):
                raise ValueError(f"{key} must be a nonempty string")
        for key in ("origin", "calibration", "synchronization", "quality"):
            if not isinstance(header[key], dict):
                raise ValueError(f"{key} provenance must be an object")
        if header.get("sampling_seconds") != .2:
            raise ValueError("Frozen G1 requires declared 0.2-second sampling; no future resampling")
        if header["observation_kind"] == "player_visible" and not header.get("actor_track"):
            raise ValueError("Player-visible files require the provider's actor_track")
        self.header = header
        self.match, self.half = str(header["match"]), header["half"]
        self.provenance = {
            **header, "adapter": "ExternalObservationSource", "input_path": str(self.path),
            "input_sha256": file_digest(self.path),
            "warning": "Source-supplied observations, not complete truth; schema validity is not physical validity",
        }

    def frames(self, start: int, count: int, stride: int = 5):
        """25Hz logical frame indices; sparse packets preserve their actual times."""
        if start < 0 or count < 1 or count > 1500 or stride != 5:
            raise ValueError("External selections require start>=0, 1..1500 samples and stride=5")
        previous_time, previous_index, previous_segment = -1., -1, None
        selected, generation = 0, -1
        with self.path.open(encoding="utf-8") as stream:
            next(stream)
            for number, line in enumerate(stream, 2):
                row = read_object(line, number)
                try:
                    timestamp, index = row["timestamp_s"], row["frame_index"]
                    segment = row["segment_id"]
                    if (not isinstance(timestamp, (float, int)) or isinstance(timestamp, bool)
                            or not math.isfinite(timestamp) or timestamp < 0):
                        raise ValueError("Timestamp must be finite nonnegative seconds")
                    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                        raise ValueError("frame_index must be a nonnegative integer on the 25Hz logical grid")
                    if not isinstance(segment, str) or not segment:
                        raise ValueError("segment_id must be a nonempty string")
                    if timestamp <= previous_time or index <= previous_index:
                        raise ValueError("Timestamp/frame order must be strictly increasing; duplicates are rejected")
                    if previous_index >= 0 and ((index - previous_index) % 5 != 0
                            or abs((index - previous_index) / 25 - (timestamp - previous_time)) > 1e-4):
                        raise ValueError("Packets must follow a 0.2s/5-index logical grid; explicit gaps allowed, no resampling")
                    if row.get("record_type") != "frame":
                        raise ValueError("Expected frame record; privileged records cannot be mixed into observations")
                    if any(not isinstance(row.get(key, False), bool) for key in ("identity_reset", "camera_cut")):
                        raise ValueError("identity_reset/camera_cut must be boolean")
                    reset = ((previous_segment is not None and segment != previous_segment)
                             or row.get("identity_reset", False) or row.get("camera_cut", False))
                    if generation < 0 or reset:
                        generation += 1
                    previous_time, previous_index, previous_segment = timestamp, index, segment
                    measurements, tracks, missing = [], set(), 0
                    if not isinstance(row["measurements"], list):
                        raise ValueError("measurements must be a list")
                    for value in row["measurements"]:
                        track = value["track"]
                        if not isinstance(track, str) or not track or track in tracks:
                            raise ValueError("Duplicate or invalid source-local track")
                        tracks.add(track)
                        team = value["team"]
                        if team == "UNKNOWN":
                            team = 0
                        if isinstance(team, bool) or team not in (0, 1, 2):
                            raise ValueError("team must be 1, 2 or UNKNOWN")
                        role = value["goalkeeper"]
                        if role is not None and not isinstance(role, bool):
                            raise ValueError("goalkeeper must be boolean or null (UNKNOWN)")
                        if value["position_status"] == "missing" and value.get("xy") is None:
                            missing += 1
                            continue
                        if value["position_status"] != "observed" or not isinstance(value.get("xy"), list):
                            raise ValueError("Position must be observed xy or explicit missing/null; no imputation")
                        xy = value["xy"]
                        if len(xy) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                               or not math.isfinite(v) for v in xy):
                            raise ValueError("xy must contain two finite numbers")
                        if self.header["coordinates"]["system"] == "pitch_top_left":
                            xy = [xy[0] - 52.5, xy[1] - 34.]
                        association = value.get("segment", 0)
                        if not isinstance(association, int) or isinstance(association, bool) or association < 0:
                            raise ValueError("Measurement segment must be a nonnegative integer")
                        if association >= 1_000_000:
                            raise ValueError("Measurement segment exceeds the supported association range")
                        quality = value.get("quality")
                        if quality is not None and not isinstance(quality, dict):
                            raise ValueError("quality must be an object or null; confidence is not metre accuracy")
                        measurements.append(Measurement(track, association + generation * 1_000_000,
                                                        tuple(xy), team, role, quality))
                    absent = row.get("absent_count", missing)
                    if not isinstance(absent, int) or isinstance(absent, bool) or absent < missing:
                        raise ValueError("absent_count must be an integer >= explicit missing positions")
                    if start <= index < start + count * stride and (index - start) % stride == 0:
                        selected += 1
                        yield TrackingFrame(self.header["source_id"], self.half, index, float(timestamp),
                                            tuple(measurements), absent, reset,
                                            missing_count_semantics=self.header.get("missing_count_semantics",
                                                "source-declared absent positions (not a complete inventory)"))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{self.path.name}, line {number}: {error}") from error
        if not selected:
            raise ValueError("No external frames in requested interval")

    def observe(self, frame: TrackingFrame, provider: SyntheticObservationProvider) -> Observation:
        if self.header["observation_kind"] == "exo_source":
            return provider.observe(frame)
        if provider.actor_track != self.header["actor_track"]:
            raise ValueError("Configured actor differs from player-visible source actor")
        return Observation(frame.source_id, frame.half, frame.index, frame.timestamp_s,
                           provider.actor_track, frame.measurements,
                           ("External player-visible measurements; source completeness not established",),
                           frame.discontinuity)


def export_public(source, path: Path, start: int, count: int):
    """Annotation-backed interoperability fixture, NOT detector-generated evidence."""
    if path.exists():
        raise FileExistsError(f"Public export exists: {path}")
    header = {
        "record_type": "metadata", "schema_version": VERSION,
        "source_id": source.provenance["source_id"], "match": source.match, "half": source.half,
        "observation_kind": "exo_source", "sampling_seconds": .2,
        "missing_count_semantics": "absent cache slots (not a missing-player count)",
        "coordinates": {"system": "pitch_centred", "units": "metres", "axes": "x_length_y_width",
                        "pitch_length_m": 105, "pitch_width_m": 68},
        "origin": {"type": "SoccerTrack_annotations", "test": "annotation-backed interoperability only",
                   "source": source.provenance},
        "quality": {"position_accuracy": "UNVALIDATED", "detector_confidence": "NOT_APPLICABLE"},
        "calibration": {"status": "ANNOTATION_COORDINATES_ONLY", "physical_validation": False},
        "synchronization": {"status": "ANNOTATION_TIMEBASE", "rgb_validated": False},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header, allow_nan=False) + "\n")
        for frame in source.frames(start, count):
            measurements = [{**dataclasses.asdict(m), "position_status": "observed"}
                            for m in frame.measurements]
            stream.write(json.dumps({
                "record_type": "frame", "frame_index": frame.index, "timestamp_s": frame.timestamp_s,
                "segment_id": source.half, "identity_reset": False, "measurements": measurements,
                "absent_count": frame.missing_count,
            }, allow_nan=False) + "\n")
    atomic_replace(temporary, path)
    return header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-match")
    parser.add_argument("--half", default="1st")
    parser.add_argument("--start-frame", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate:
        source = ExternalObservationSource(args.validate)
        frames = sum(1 for _ in source.frames(0, 1500))
        print(json.dumps({"status": "VALIDATED_SCHEMA_ONLY", "selected_frames": frames,
                          "source": source.provenance}, indent=2))
    elif args.export_match and args.output:
        from src.backbone.source import SoccerTrackSource
        export_public(SoccerTrackSource(ROOT, args.export_match, args.half), args.output,
                      args.start_frame, args.samples)
        print(args.output)
    else:
        parser.error("Supply --validate INPUT or --export-match MATCH --output FILE")


if __name__ == "__main__":
    main()
