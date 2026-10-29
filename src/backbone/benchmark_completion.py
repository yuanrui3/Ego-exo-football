"""Before/after benchmark for repeated causal dropout draws, with inference parity."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import time

import numpy as np
import torch

from src.backbone.actions import ActionModelSession
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import SoccerTrackSource
from src.backbone.streaming_completion import ACTION, ROOT, compact
from src.trajectory_dataset import write_json


class UncachedProvider(SyntheticObservationProvider):
    @staticmethod
    def _dropout_duration(seed: int) -> float | None:
        rng = np.random.default_rng(seed)
        return float(rng.uniform(.6, 2.2)) if rng.random() < .04 else None


def benchmark(output: Path, samples: int = 600) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    source = SoccerTrackSource(ROOT, "118577", "1st")
    models = ModelSession(ROOT / "outputs" / "dynamic_gnn_tactical_20261019", device="cpu")
    action = ActionModelSession(ACTION, device="cpu")
    frames = tuple(source.frames(3000, samples))
    times, hashes, observation_times = {}, {}, {}
    for name, provider_type in (("uncached", UncachedProvider), ("cached", SyntheticObservationProvider)):
        SyntheticObservationProvider._dropout_duration.cache_clear()
        provider = provider_type("track_11")
        engine = ReplayEngine(models, action_model=action)
        digest = hashlib.sha256()
        started, observation_s = time.perf_counter(), 0.0
        for number, frame in enumerate(frames):
            tick = time.perf_counter()
            observation = provider.observe(frame)
            observation_s += time.perf_counter() - tick
            row = engine.step(observation, include_passing_assessment=False, action_query=number % 50 == 0)
            digest.update(json.dumps(compact(row), sort_keys=True, allow_nan=False).encode())
        times[name] = time.perf_counter() - started
        hashes[name], observation_times[name] = digest.hexdigest(), observation_s
    if hashes["uncached"] != hashes["cached"]:
        raise ValueError("Dropout optimization changed observations or belief/action predictions")
    profile = cProfile.Profile()
    provider = UncachedProvider("track_11")
    profile.enable()
    for frame in frames[:200]:
        provider.observe(frame)
    profile.disable()
    profile.dump_stats(str(output / "uncached_observation.prof"))
    stats = pstats.Stats(profile)
    hotspots = sorted(
        ({"function": f"{key[0]}:{key[1]}:{key[2]}", "calls": value[1],
          "self_seconds": value[2], "cumulative_seconds": value[3]}
         for key, value in stats.stats.items()),
        key=lambda row: row["cumulative_seconds"], reverse=True)[:15]
    result = {
        "status": "PASS", "samples_per_variant": samples, "device": "cpu",
        "matched_source": "118577_1st native frames 3000 onward; actor track_11",
        "full_replay_seconds": times, "observation_seconds": observation_times,
        "end_to_end_speedup": times["uncached"] / times["cached"],
        "observation_speedup": observation_times["uncached"] / observation_times["cached"],
        "output_sha256": hashes, "exact_stored_prediction_parity": True,
        "profile": hotspots, "bounded_cache": SyntheticObservationProvider._dropout_duration.cache_info()._asdict(),
        "scope": "Repeated deterministic RNG construction in the observation path; not a claim of global optimality",
        "behavior": "Same seed keys, random draw order, interval comparisons and causal observations; capped at 4096 draws",
    }
    write_json(output / "benchmark.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.output), indent=2))


if __name__ == "__main__":
    main()
