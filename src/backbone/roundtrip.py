"""Verify actual public-file inference parity against its original local annotations."""

import argparse
import dataclasses
import json
from pathlib import Path
import time

import torch

from src.backbone.external import ExternalObservationSource
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import SoccerTrackSource
from src.trajectory_dataset import write_json

ROOT = Path(__file__).resolve().parents[2]


def verify(path: Path, output: Path, actor: str, start: int, samples: int):
    started = time.perf_counter()
    torch.set_num_threads(4)
    external = ExternalObservationSource(path)
    original = SoccerTrackSource(ROOT, external.match, external.half)
    models = ModelSession(ROOT / "outputs" / "dynamic_gnn_tactical_20261019")
    engines = [ReplayEngine(models), ReplayEngine(models)]
    providers = [SyntheticObservationProvider(actor) for _ in range(2)]
    compared, learned, lanes = 0, 0, 0
    for a, b in zip(original.frames(start, samples), external.frames(start, samples), strict=True):
        if a != b:
            raise AssertionError(f"Source round-trip mismatch at frame {a.index}")
        oa, ob = providers[0].observe(a), external.observe(b, providers[1])
        if dataclasses.asdict(oa) != dataclasses.asdict(ob):
            raise AssertionError(f"Observation round-trip mismatch at frame {a.index}")
        ra, rb = engines[0].step(oa), engines[1].step(ob)
        ra.pop("latency_ms")
        rb.pop("latency_ms")
        if ra != rb:
            raise AssertionError(f"Inference round-trip mismatch at frame {a.index}")
        compared += 1
        learned += sum(e["backend"] == "G1" for e in rb["belief"]["entities"] if not e["visible"])
        lanes += sum(c["status"] == "AVAILABLE" for c in rb["assessment"]["candidates"])
    if compared != samples or learned == 0 or lanes == 0:
        raise AssertionError("Round trip did not exercise actual learned hidden inference and geometry")
    result = {
        "status": "PASSED", "frames": compared, "actor": actor, "source": external.provenance,
        "observation_parity": "EXACT", "prediction_parity": "EXACT_CPU_EXCLUDING_LATENCY",
        "documented_tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "learned_hidden_estimates": learned, "available_lanes": lanes,
        "label": "Annotation-backed interoperability test; not detector-generated evidence",
        "execution_seconds": time.perf_counter() - started,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor", default="track_1")
    parser.add_argument("--start-frame", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    verify(args.input, args.output, args.actor, args.start_frame, args.samples)


if __name__ == "__main__":
    main()
