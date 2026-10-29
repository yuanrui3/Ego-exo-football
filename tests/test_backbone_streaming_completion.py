import numpy as np
import pytest

from src.backbone.contracts import Measurement, Observation, TrackingFrame
from src.backbone.lazy_windows import select_windows
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.stress_completion import transform


class UncachedProvider(SyntheticObservationProvider):
    @staticmethod
    def _dropout_duration(seed: int) -> float | None:
        rng = np.random.default_rng(seed)
        return float(rng.uniform(.6, 2.2)) if rng.random() < .04 else None


def test_cached_dropout_exactly_matches_uncached_measurements_and_is_bounded():
    cached = SyntheticObservationProvider("track_0")
    uncached = UncachedProvider("track_0")
    for step in range(100):
        index = step * 5
        measurements = tuple(
            Measurement(f"track_{i}", int(step >= 50 and i == 2),
                        (step / 5 + i * 3, float(i)), 1 if i < 3 else 2, False)
            for i in range(6) if not (i == 4 and 20 <= step < 60)
        )
        frame = TrackingFrame("fixture", "1st", index, index / 25,
                              measurements, 0)
        assert cached.observe(frame) == uncached.observe(frame)
    assert cached._dropout_duration.cache_info().currsize <= 4096


def test_lazy_window_selection_is_distinct_causal_and_bounded():
    query, actor = select_windows(1000, 24, 400)
    assert len(set(zip(query.tolist(), actor.tolist()))) == 400
    assert query.min() >= 75 and query.max() < 1000
    assert actor.min() >= 0 and actor.max() < 24
    assert np.all((query - 75) % 25 == 0)
    assert np.all(np.diff(query) >= 0)
    assert len(select_windows(100, 2, 999)[0]) == 2
    with pytest.raises(ValueError):
        select_windows(50, 22, 10)


def test_stress_switch_occurs_after_memory_and_gaps_preserve_times():
    stream = tuple(Observation(
        "fixture", "1st", i * 5, i / 5, "track_0",
        (Measurement("track_0", 1, (0., 0.), 1, False),
         Measurement("track_1", 1, (5., 0.), 2, False))) for i in range(121))
    switched = transform(stream, "identity_after_memory")
    assert switched[:60] == stream[:60]
    assert switched[60].measurements[1].segment == 100001
    assert switched[60].measurements[1].xy == stream[60].measurements[1].xy
    for name, missing in (("short_source_gap", 3), ("long_source_gap", 14)):
        corrupted = transform(stream, name)
        assert sum(obs.actor is None for obs in corrupted) == missing
        assert [obs.index for obs in corrupted] == [obs.index for obs in stream]
        assert corrupted[100].actor is None
    unknown = transform(stream, "unknown_team_role")
    assert all(m.team == 0 and m.goalkeeper is None for m in unknown[100].measurements)
    camera = transform(stream, "camera_discontinuity")
    assert [i for i, obs in enumerate(camera) if obs.discontinuity] == [60]
