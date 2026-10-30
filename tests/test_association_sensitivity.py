from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.backbone.association_audit import SharedAssociation, synthetic_reproduction
from src.backbone.contracts import Measurement, Observation
from src.backbone.identity_program import LearnedAssociation
from src.backbone.perception_sensitivity import (
    action_input, conditions, corrupt, prepare, select_population, summarize_pair,
)


def frame(t, xs=(100.,), cut=False):
    return {"timestamp_s": t, "frame_index": round(t * 25), "camera_cut": cut,
            "detections": [{"track": f"input{j}", "footpoint_xy_px": [x, 100.],
                            "bbox_xyxy_px": [x - 5, 50., x + 5, 100.], "detector_confidence": .9}
                           for j, x in enumerate(xs)]}


def observations(count=41):
    return tuple(Observation("synthetic", "1st", i * 5, i * .2, "focal", (
        Measurement("focal", 0, (i * .1, 0.), 1, False),
        Measurement("friend", 0, (5. + i * .1, 2.), 1, False),
        Measurement("other_friend", 0, (6. + i * .1, 3.), 1, False),
        Measurement("opponent", 0, (7. + i * .1, -3.), 2, False),
    )) for i in range(count))


def test_missing_feature_reproduced_and_justified_repair():
    assert synthetic_reproduction()["results"]["retained_missing_True"]["survives_gap"]


def test_repair_does_not_refresh_missing_feature_timestamp_and_expires():
    tracker = SharedAssociation(4096, "R1", original_scale=True)
    for t in (0., .2, .4, .6, .8):
        tracker.step(frame(t), [[1., 0.]])
    key = next(iter(tracker.tracks))
    for t in (1., 1.2, 1.4):
        tracker.step(frame(t), [None])
        assert tracker.tracks[key]["feature_time"] == .8
    tracker.step(frame(1.6), [None])
    tracker.step(frame(1.8), [None])
    assert tracker.tracks[key]["feature"] is None
    result = tracker.step(frame(2.2), [[1., 0.]])
    assert result["detections"][0]["track"] != key


def test_missing_current_gap_still_abstains_and_camera_cut_resets():
    tracker = SharedAssociation(4096, "R1", original_scale=True)
    for t in (0., .2, .4, .6, .8, 1.):
        old = tracker.step(frame(t), [[1., 0.]])["detections"][0]["track"]
    new = tracker.step(frame(1.4), [None])["detections"][0]["track"]
    assert old != new
    assert tracker.step(frame(1.6, cut=True), [[1., 0.]])["detections"][0]["track"] != new


def test_ambiguous_samekit_crossing_and_irregular_time():
    tracker = SharedAssociation(4096, "R1")
    for t in (0., .2, .4, .6, .8):
        tracker.step(frame(t, (100., 120.)), [[1., 0.], [1., 0.]])
    result = tracker.step(frame(1.2, (110.,)), [[1., 0.]])
    assert result["detections"][0]["association_quality"]["status"] == "AMBIGUOUS_ABSTAIN"
    with pytest.raises(ValueError):
        tracker.step(frame(1.1), [[1., 0.]])


def test_original_scale_no_retention_matches_original_ids():
    a = SharedAssociation(4096, "R1", original_scale=True, retain=False)
    b = LearnedAssociation(4096, .35)
    for i, t in enumerate((0., .2, .4, .6, .8, 1., 1.4)):
        feature = [None if i == 5 else [1., 0.]]
        assert a.step(frame(t), feature)["detections"][0]["track"] == b.step(frame(t), feature)["detections"][0]["track"]


def test_clean_zero_and_label_isolation():
    stream = observations()
    zero, counts = corrupt(stream, {"family": "noise", "severity": 0.}, 42)
    assert zero == stream and counts == {}
    value, _, reason, _ = prepare(stream)
    assert value is not None and reason is None
    converted, reason = action_input(value)
    assert converted is not None and reason is None
    # Targets/labels are deliberately not accepted by corruption or preparation APIs.
    with pytest.raises(TypeError):
        corrupt(stream, {"family": "noise", "severity": 1.}, 42, targets={})


def test_optimized_memory_matches_release_path():
    for condition in ({"family": "clean", "severity": 0.},
                      {"family": "noise", "severity": 1.},
                      {"family": "focal_gap", "severity": .4, "rate": .4}):
        stream = corrupt(observations(), condition, 42)[0]
        a = prepare(stream)[0]
        b = prepare(stream, optimized=False)[0]
        assert (a is None) == (b is None)
        if a is not None:
            assert np.array_equal(a.x, b.x) and np.array_equal(a.mask, b.mask)
            assert np.array_equal(a.actor_xy, b.actor_xy)


def test_consistent_rename_keeps_focal_and_identical_tensor():
    original = observations()
    renamed = corrupt(original, {"family": "rename", "severity": 1.}, 42)[0]
    assert all(o.actor is not None for o in renamed)
    a, b = prepare(original)[0], prepare(renamed)[0]
    assert np.array_equal(a.x, b.x)


def test_translation_relative_input_and_absolute_actor():
    original = observations()
    changed = corrupt(original, {"family": "translation", "severity": 2.}, 42)[0]
    a, b = prepare(original)[0], prepare(changed)[0]
    np.testing.assert_allclose(a.x, b.x, atol=2e-6)
    assert np.linalg.norm(a.actor_xy - b.actor_xy) == pytest.approx(2, abs=1e-5)


@pytest.mark.parametrize("family", ["focal_gap", "focal_fragment", "same_team_swap",
                                   "opposing_team_swap", "nonfocal_fragment", "noise", "translation", "drift"])
def test_future_suffix_never_changes_prefix(family):
    condition = {"family": family, "severity": 1., "rate": .25}
    full = observations()
    a = corrupt(full, condition, 137)[0]
    b = corrupt(full[:20], condition, 137)[0]
    assert a[:20] == b


def test_fragmented_focal_is_not_rebound_and_needs_genuine_gap_warmup():
    stream = observations()
    condition = {"family": "focal_fragment", "severity": 100., "rate": 100.}
    changed, _ = corrupt(stream, condition, 42)
    assert changed[-1].actor is None
    assert prepare(changed)[2] == "FOCAL_RUNTIME_TOKEN_UNAVAILABLE"
    broken = list(stream)
    broken[-5] = replace(broken[-5], measurements=tuple(m for m in broken[-5].measurements if m.track != "focal"))
    assert prepare(tuple(broken))[2] == "DISCONTINUOUS_FOCAL_HISTORY"
    assert prepare(tuple(broken) + tuple(replace(o, index=o.index + 205, timestamp_s=o.timestamp_s + 8.2)
                                       for o in stream[:11]))[0] is not None


@pytest.mark.parametrize("family", ["same_team_swap", "opposing_team_swap"])
def test_swaps_bijective_no_focal_and_attributes_travel(family):
    original = observations()
    changed, counts = corrupt(original, {"family": family, "severity": 10., "rate": 10.}, 42)
    assert counts["event_starts"] > 0
    for a, b in zip(original, changed, strict=True):
        assert len({m.track for m in b.measurements}) == len(b.measurements)
        assert b.actor == a.actor
        assert [m.team for m in b.measurements] == [m.team for m in a.measurements]
    if family == "opposing_team_swap":
        assert any(next(m for m in o.measurements if m.track == "opponent").team == 1 for o in changed)


def test_hash_population_stratified_inclusion_weights_not_scores():
    rows = [{"id": str(i), "match": "132877", "half": "1st", "query": i, "label": i % 7}
            for i in range(140)]
    selected = select_population(rows, 70)
    assert len(selected) == 70 and all(r["weight"] == 2 for r in selected)
    assert selected == select_population(list(reversed(rows)), 70)


def test_unsupported_kept_in_denominator_and_fixed_targets_not_rescued():
    clean = [{"id": "a", "label": 0, "prediction": 0, "weight": 1., "reason": None,
              "block": "1st:0", "focal_available": True, "g1": [], "corruption_counts": {}}]
    changed = [{**clean[0], "prediction": None, "reason": "FOCAL_RUNTIME_TOKEN_UNAVAILABLE"}]
    result = summarize_pair(clean, changed)
    assert result["scheduled_queries"] == 1 and result["corrupt_supported_queries"] == 0
    assert result["correct_and_supported_fraction_all"] == 0
    assert not result["s4_tolerance_pass"]


def test_newly_supported_examples_cannot_offset_lost_clean_support():
    base = {"label": 0, "weight": 1., "reason": None, "block": "1st:0",
            "focal_available": True, "g1": [], "corruption_counts": {}}
    clean = [{**base, "id": "a", "prediction": 0}, {**base, "id": "b", "prediction": None}]
    altered = [{**base, "id": "a", "prediction": None}, {**base, "id": "b", "prediction": 0}]
    metrics = summarize_pair(clean, altered)
    assert metrics["clean_supported_queries"] == metrics["corrupt_supported_queries"] == 1
    assert metrics["s4_availability_ratio"] == 0.
    assert not metrics["s4_tolerance_pass"]


def test_required_severity_grid_present():
    grid = conditions()
    assert len(grid) == 32
    assert {c["severity"] for c in grid if c["family"] == "focal_gap"} == {.2, .4, 1., 2.}
