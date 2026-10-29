"""Integration acceptance: causal boundaries, lifecycle and frozen-model parity."""

import dataclasses
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.backbone.assessment import GaussianAssessor
from src.backbone.contracts import Measurement, Observation, TrackingFrame, WorldInput
from src.backbone.export import dataset_relation, export_training_sample
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine, replay_prefix
from src.belief_observation import SCALE
from src.hidden_defender_geometry import lane_distances
from train.dynamic_football_training import ArrayWorldDataset, infer_decision, infer_world

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs" / "dynamic_gnn_tactical_20261019"


@pytest.fixture(scope="module")
def models():
    torch.set_num_threads(2)
    return ModelSession(EXPERIMENT, exo_p0=True)


def observations(count=35):
    result = []
    for index in range(count):
        members = [Measurement("actor", 1, (index * .1, 0), 1, False),
                   Measurement("friend", 1, (index * .1 + 20, 0), 1, False)]
        if index < 8 or index >= 15:
            members.append(Measurement("defender", 1, (index * .1 + 10, 1), 2, False))
        result.append(Observation("test", "1st", index * 5, index * .2, "actor", tuple(members)))
    return result


def without_latency(value):
    return {key: item for key, item in value.items() if key != "latency_ms"}


def test_prefix_consistency_with_genuine_model(models):
    stream = observations()
    short, long = ReplayEngine(models), ReplayEngine(models)
    prefix = [short.step(obs) for obs in stream[:13]]
    full = [long.step(obs) for obs in stream]
    assert without_latency(prefix[-1]) == without_latency(full[12])
    assert any(e["backend"] == "G1" for e in prefix[-1]["belief"]["entities"])


def test_hidden_truth_and_future_are_not_student_inputs(models):
    stream = observations()
    evaluation = {"hidden_xy": np.ones((35, 22, 2))}
    engine = ReplayEngine(models)
    first = [engine.step(obs) for obs in stream[:13]]
    evaluation["hidden_xy"][:] = 1e9
    engine.reset()
    second = [engine.step(obs) for obs in stream[:13]]
    assert without_latency(first[-1]) == without_latency(second[-1])
    with pytest.raises(TypeError):
        engine.step(evaluation)
    changed = stream[:13] + [dataclasses.replace(obs, measurements=()) for obs in stream[13:]]
    engine.reset()
    third = [engine.step(obs) for obs in changed]
    assert without_latency(first[-1]) == without_latency(third[12])


def test_first_sight_never_invents_named_players(models):
    engine = ReplayEngine(models)
    result = engine.step(observations()[0])
    assert len(result["belief"]["entities"]) == 3
    assert set(c["id"] for c in result["assessment"]["candidates"]) <= set(
        e["id"] for e in result["belief"]["entities"])
    first = dataclasses.replace(observations()[0], measurements=observations()[0].measurements[:1])
    assert len(ReplayEngine(models).step(first)["belief"]["entities"]) == 1
    assert not ReplayEngine(models).step(first)["assessment"]["candidates"]


def test_wrapper_world_parity_and_normalization_contract(models):
    engine = ReplayEngine(models)
    for obs in observations()[:13]:
        engine.step(obs)
    inputs = engine.memory.current_input
    assert isinstance(inputs, WorldInput)
    assert inputs.x.shape == (11, 22, 10) and inputs.mask.shape == (11, 22)
    assert not inputs.x[0, :, 2:4].any()
    actor = observations()[12].actor
    assert actor is not None
    np.testing.assert_allclose(inputs.actor_xy, actor.xy)
    # Model input relative coordinates correspond to last observations in metres.
    np.testing.assert_allclose(inputs.x[-1, inputs.mask[-1], :2] * SCALE + inputs.actor_xy,
                               inputs.last_xy[inputs.mask[-1]], atol=2e-6)
    actual = models.predict(inputs)
    arrays = {key: getattr(inputs, key)[None] for key in ("x", "mask", "actor_xy", "last_xy")}
    expected = infer_world(EXPERIMENT / "world" / "G1_2s_seed42_excludenone",
                           ArrayWorldDataset(arrays, [0]), time.monotonic() + 30, device="cpu")
    for key, old_key in (("mean", "mean"), ("std", "std"), ("occupancy", "occupancy"),
                         ("anonymous_defender", "anonymous")):
        np.testing.assert_allclose(actual[key], expected[old_key][0], atol=1e-5, rtol=1e-5)


def test_p0_saved_statistics_parity_and_single_load(models):
    from src.belief_tracking_dataset import load_npz
    data = load_npz(EXPERIMENT / "data" / "receiver_asof.npz")
    rows = np.asarray([0, 1])
    expected = infer_decision(EXPERIMENT / "receiver" / "P0_seed42_excludenone", data, None, rows,
                              "receiver", time.monotonic() + 60, device="cpu")
    for index, row in enumerate(rows):
        mapping = {f"TEAMMATE_{i+1}": f"source_{i}" for i in range(10)
                   if data["states"][row, -1, i+1, 2] > .5}
        actual = models.exo_reference(data["states"][row], mapping)
        for candidate in actual["ranking"]:
            slot = int(candidate["source_track"].split("_")[1])
            assert candidate["probability"] == pytest.approx(expected[index, slot], abs=1e-5, rel=1e-5)
    assert models.load_counts == {"world": 1, "p0": 1}


def test_geometry_thresholds_units_uncertain_candidate_and_missing_evidence(models):
    def entity(name, xy, team, visible=True, sigma=0):
        return {"id": name, "xy": xy, "team": team, "actor": name == "actor",
                "visible": visible, "std_m": [sigma, sigma]}
    belief = {"actor_available": True, "actor_xy": [0, 0], "actor_team": 1,
              "entities": [entity("actor", [0, 0], 1), entity("candidate", [20, 0], 1),
                           entity("defender", [10, 2.5], 2)]}
    assessor = GaussianAssessor()
    row = assessor.assess(belief, "same")["candidates"][0]
    assert row["lane_probability"] == [0., 1., 1.]
    assert row["expected_nearest_defender_m"] == pytest.approx(np.hypot(10, 2.5))
    belief["entities"][1]["visible"] = False
    assert assessor.assess(belief, "same")["candidates"][0]["status"] == "UNAVAILABLE"
    belief["entities"][1]["visible"] = True
    belief["entities"] = belief["entities"][:2]
    no_defender_evidence = assessor.assess(belief, "same")
    assert no_defender_evidence["status"] == "UNAVAILABLE"
    assert no_defender_evidence["candidates"][0]["reason"].startswith("No observed")
    distance, interior = lane_distances(np.zeros(2), np.asarray([20, 0]), np.asarray([[1, 0], [10, 0]]))
    assert interior.tolist() == [False, True]
    assert distance.tolist() == [0, 0]
    belief["actor_team"] = 0
    assert not assessor.assess(belief, "same")["candidates"]
    actor_only = Observation(
        "test", "1st", 0, 0, "actor",
        (Measurement("actor", 1, (0, 0), 1, False),),
    )
    passing_module = ReplayEngine(models).step(actor_only)["modules"]["passing"]
    assert passing_module["status"] == "UNAVAILABLE"
    assert passing_module["reason"] == "No candidate has the required observed endpoint and defender evidence"


def test_lifecycle_warmup_gap_reappearance_expiry_and_seek(models):
    engine = ReplayEngine(models, expiry_s=3)
    stream = observations()
    first = engine.step(stream[0])
    assert not first["belief"]["model_contract_met"]
    defender_id = next(e["id"] for e in first["belief"]["entities"] if e["team"] == 2)
    results = [first] + [engine.step(obs) for obs in stream[1:]]
    assert next(e["id"] for e in results[15]["belief"]["entities"] if e["team"] == 2) == defender_id
    hidden_candidate = dataclasses.replace(stream[-1], index=180, timestamp_s=7.2,
                                         measurements=stream[-1].measurements[:1])
    gap = engine.step(hidden_candidate)
    assert not gap["belief"]["model_contract_met"]
    assert any("TIME_GAP" in w for w in gap["belief"]["warnings"])
    missing = dataclasses.replace(hidden_candidate, index=185, timestamp_s=7.4, measurements=())
    assert not engine.step(missing)["assessment"]["candidates"]
    later = dataclasses.replace(missing, index=260, timestamp_s=10.4)
    assert not engine.step(later)["belief"]["entities"]
    with pytest.raises(ValueError, match="Backwards seeking"):
        engine.step(stream[0])
    recovered = replay_prefix(engine, stream, 12)
    assert without_latency(recovered) == without_latency(results[12])


def test_identity_discontinuity_and_half_reset(models):
    engine = ReplayEngine(models)
    stream = observations()
    for obs in stream[:12]:
        before = engine.step(obs)
    old_ids = {e["id"] for e in before["belief"]["entities"]}
    obs = stream[12]
    members = tuple(dataclasses.replace(m, segment=2) if m.track == "friend" else m for m in obs.measurements)
    changed = engine.step(dataclasses.replace(obs, measurements=members))
    assert not changed["belief"]["model_contract_met"]
    assert any(e["id"] not in old_ids for e in changed["belief"]["entities"])
    half = engine.step(dataclasses.replace(stream[0], half="2nd"))
    assert not half["belief"]["model_contract_met"]
    assert any("HALF_RESET" in warning for warning in half["belief"]["warnings"])


def test_synthetic_visibility_prefix_and_hidden_measurement_removal():
    frames = []
    for i in range(20):
        frames.append(TrackingFrame("source", "1st", i * 5, i * .2, (
            Measurement("a", 1, (i, 0), 1, False), Measurement("never", 1, (-40, 30), 2, False)), 0))
    one, two = SyntheticObservationProvider("a"), SyntheticObservationProvider("a")
    assert [one.observe(f) for f in frames[:12]] == [two.observe(f) for f in frames][:12]
    provider = SyntheticObservationProvider("a", fov=90, dropout=False)
    visible = [provider.observe(f) for f in frames]
    assert all(m.track != "never" for obs in visible[2:] for m in obs.measurements)
    assert all(obs.actor is not None for obs in visible)


def test_older_memory_is_explicit_fallback_not_six_second_g1(models):
    stream = observations(32)
    stream = [dataclasses.replace(obs, measurements=tuple(m for m in obs.measurements
               if m.track != "defender" or index < 8)) for index, obs in enumerate(stream)]
    engine = ReplayEngine(models)
    result = [engine.step(obs) for obs in stream][-1]
    hidden = next(e for e in result["belief"]["entities"] if e["team"] == 2)
    assert hidden["age_s"] > 2
    assert hidden["backend"] == "last_seen"
    assert hidden["fallback_reason"] == "older_than_2s_model_context"


def test_uncertainty_changes_gaussian_assessment():
    belief = {"actor_available": True, "actor_xy": [0, 0], "actor_team": 1,
              "entities": [
                  {"id": "candidate", "team": 1, "actor": False, "xy": [20, 0],
                   "std_m": [0, 0], "visible": True},
                  {"id": "defender", "team": 2, "actor": False, "xy": [10, 0],
                   "std_m": [5, 5], "visible": False}]}
    risk = GaussianAssessor(samples=1024).assess(belief, "fixed")["candidates"][0]
    assert 0 < risk["lane_probability"][1] < 1
    assert risk["hidden_threat_probability"] == risk["lane_probability"]


def test_provenance_and_training_export_restriction(models, tmp_path):
    assert dataset_relation("117092", models).startswith("FITTING")
    assert dataset_relation("118577", models).startswith("RETROSPECTIVE")
    assert dataset_relation("132877", models).startswith("DEVELOPMENT")
    with pytest.raises(ValueError, match="ONLY"):
        export_training_sample(ROOT, tmp_path, models, {"match": "118577"})
    assert not list(tmp_path.iterdir())


def test_external_adapter_invalid_measurements_surface_errors():
    with pytest.raises(ValueError, match="finite"):
        Measurement("track", 1, (float("nan"), 1), 1, False)
    item = Measurement("track", 1, (0, 0), 1, False)
    with pytest.raises(ValueError, match="Duplicate"):
        Observation("test", "1st", 0, 0, "track", (item, item))
