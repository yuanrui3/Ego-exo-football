"""External packets, causal separation and hash-safe batch acceptance."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.backbone.external import ExternalObservationSource, export_public
from src.backbone.contracts import ActionOutput, ActionOutcome, BallState, RestartContext
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import SoccerTrackSource
from src.trajectory_dataset import write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def models():
    torch.set_num_threads(2)
    return ModelSession(ROOT / "outputs" / "dynamic_gnn_tactical_20261019")


@pytest.fixture
def public(tmp_path):
    path = tmp_path / "example.jsonl"
    source = SoccerTrackSource(ROOT, "118577", "1st")
    export_public(source, path, 1500, 100)
    return path


def rewrite(path, mutate):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_real_roundtrip_g1_parity(tmp_path, models):
    original = SoccerTrackSource(ROOT, "118577", "1st")
    path = tmp_path / "real.jsonl"
    export_public(original, path, 1500, 100)
    external = ExternalObservationSource(path)
    providers = [SyntheticObservationProvider("track_1") for _ in range(2)]
    engines = [ReplayEngine(models) for _ in range(2)]
    learned, lanes, compared = 0, 0, 0
    for a, b in zip(original.frames(1500, 100), external.frames(1500, 100), strict=True):
        assert a == b
        oa, ob = providers[0].observe(a), external.observe(b, providers[1])
        assert dataclasses.asdict(oa) == dataclasses.asdict(ob)
        ra, rb = engines[0].step(oa), engines[1].step(ob)
        ra.pop("latency_ms")
        rb.pop("latency_ms")
        assert ra == rb
        learned += sum(e["backend"] == "G1" for e in rb["belief"]["entities"] if not e["visible"])
        lanes += sum(c["status"] == "AVAILABLE" for c in rb["assessment"]["candidates"])
        compared += 1
    assert compared == 100 and learned > 0 and lanes > 0


@pytest.mark.parametrize("change,match", [
    (lambda rows: rows[0]["coordinates"].pop("units"), "units"),
    (lambda rows: rows[0]["coordinates"].update(system="pixels"), "coordinate"),
    (lambda rows: rows[0]["coordinates"].update(units="pixels"), "pixels"),
    (lambda rows: rows[0].update(observation_kind="privileged"), "kind"),
    (lambda rows: rows[2].update(timestamp_s=rows[1]["timestamp_s"]), "order"),
    (lambda rows: rows[2].update(frame_index=rows[1]["frame_index"]), "order"),
    (lambda rows: rows[1]["measurements"].append(rows[1]["measurements"][0]), "Duplicate"),
])
def test_invalid_external(public, change, match):
    rewrite(public, change)
    with pytest.raises(ValueError, match=match):
        list(ExternalObservationSource(public).frames(1500, 100))


def test_declared_conversion(public):
    baseline = list(ExternalObservationSource(public).frames(1500, 100))
    def convert(rows):
        rows[0]["coordinates"]["system"] = "pitch_top_left"
        for row in rows[1:]:
            for measurement in row["measurements"]:
                measurement["xy"][0] += 52.5
                measurement["xy"][1] += 34
    rewrite(public, convert)
    converted = list(ExternalObservationSource(public).frames(1500, 100))
    for a, b in zip(baseline, converted):
        np.testing.assert_allclose([m.xy for m in a.measurements],
                                   [m.xy for m in b.measurements], atol=1e-12)


def test_missing_unknown_and_no_future_fill(public, models):
    def change(rows):
        rows[1]["measurements"][1].update(xy=None, position_status="missing")
        for row in rows[1:]:
            row["measurements"][0].update(team="UNKNOWN", goalkeeper=None,
                                           quality={"detector_confidence": .99})
    rewrite(public, change)
    source = ExternalObservationSource(public)
    frames = list(source.frames(1500, 100))
    assert len(frames[0].measurements) < len(frames[1].measurements)
    assert frames[0].measurements[0].team == 0
    assert frames[0].measurements[0].goalkeeper is None
    engine = ReplayEngine(models)
    provider = SyntheticObservationProvider("track_0", fov=360, dropout=False)
    results = [engine.step(source.observe(frame, provider)) for frame in frames[:15]]
    assert not any(row["belief"]["model_contract_met"] for row in results)
    assert any(e["team"] == 0 for e in results[0]["belief"]["entities"])
    first = dataclasses.asdict(source.observe(frames[0], SyntheticObservationProvider("track_1")))
    rewrite(public, lambda rows: rows[-1]["measurements"][0].update(xy=[30, 20]))
    changed = ExternalObservationSource(public)
    frame = next(changed.frames(1500, 100))
    assert first == dataclasses.asdict(changed.observe(frame, SyntheticObservationProvider("track_1")))


@pytest.mark.parametrize("reset", ["segment", "identity", "camera_cut"])
def test_camera_segment_reset_even_missing_actor(public, models, reset):
    def change(rows):
        if reset == "segment":
            for row in rows[13:]:
                row["segment_id"] = "camera_2"
        elif reset == "identity":
            rows[13]["identity_reset"] = True
        else:
            rows[13]["camera_cut"] = True
        rows[13]["measurements"] = []
    rewrite(public, change)
    source = ExternalObservationSource(public)
    engine = ReplayEngine(models)
    provider = SyntheticObservationProvider("track_1")
    results = [engine.step(source.observe(frame, provider)) for frame in source.frames(1500, 20)]
    assert not results[12]["belief"]["entities"]
    assert "EXPLICIT_SOURCE_DISCONTINUITY" in " ".join(results[12]["belief"]["warnings"])
    assert not results[13]["belief"]["model_contract_met"]


def test_player_visible_bypasses_fov_and_privilege(public):
    rewrite(public, lambda rows: rows[0].update(observation_kind="player_visible", actor_track="track_1"))
    source = ExternalObservationSource(public)
    frame = next(source.frames(1500, 1))
    obs = source.observe(frame, SyntheticObservationProvider("track_1", fov=90))
    assert obs.measurements == frame.measurements
    with pytest.raises(ValueError, match="actor"):
        source.observe(frame, SyntheticObservationProvider("track_2"))


def batch_configuration(path, public, jobs=None):
    value = {
        "schema_version": "backbone.batch.0.2",
        "defaults": {
            "source_adapter": "external_jsonl", "input_path": str(public), "match": "118577", "half": "1st",
            "start_frame": 1500, "samples": 20, "actors": ["track_1"],
            "intended_use": "retrospective_replay", "backend": "G1", "exo_p0": False,
            "experiment": "outputs/dynamic_gnn_tactical_20261019",
            "selection": "Fixed fixture prefix, no prediction-based selection",
            "observation": {"seed": 42, "fov": 120},
        },
        "jobs": jobs if jobs is not None else [{"id": "first"}, {"id": "second", "start_frame": 1600}],
    }
    write_json(path, value)
    return value


def test_batch_resume_no_duplicates_reuse_and_invalidation(public, tmp_path, monkeypatch, models):
    import src.backbone.batch as batch
    loads = []
    def load(*args):
        loads.append(args)
        return models
    monkeypatch.setattr(batch, "ModelSession", load)
    config_path, folder = tmp_path / "config.json", tmp_path / "batch"
    config = batch_configuration(config_path, public)
    initial = batch.run_batch(config_path, folder)
    assert initial["counts"] == {"COMPLETE": 2}
    assert len(loads) == 1 and initial["model_sessions_loaded"] == 1
    assert initial["totals"]["frames"] == 40
    assert all(len(Path(job["output"]).name) == 16 for job in initial["jobs"])
    resumed = batch.run_batch(config_path, folder, resume=True)
    assert resumed["counts"] == {"SKIPPED_HASH_MATCH": 2} and resumed["model_sessions_loaded"] == 0
    assert len(list(folder.glob("jobs/*/*/replay.json"))) == 2
    config["jobs"][0]["observation"] = {"seed": 42, "fov": 180}
    write_json(config_path, config)
    changed = batch.run_batch(config_path, folder, resume=True)
    assert changed["counts"] == {"COMPLETE": 1, "SKIPPED_HASH_MATCH": 1}
    assert changed["jobs"][0]["fingerprint"] != initial["jobs"][0]["fingerprint"]
    rewrite(public, lambda rows: rows[0]["quality"].update(test_revision="changed"))
    source_changed = batch.run_batch(config_path, folder, resume=True)
    assert source_changed["counts"] == {"COMPLETE": 2}
    assert source_changed["jobs"][0]["fingerprint"] != changed["jobs"][0]["fingerprint"]
    viewer = Path(source_changed["jobs"][0]["output"]) / "viewer.html"
    viewer.write_text("deliberately corrupt")
    recovered = batch.run_batch(config_path, folder, resume=True)
    assert recovered["counts"] == {"COMPLETE": 1, "SKIPPED_HASH_MATCH": 1}
    assert viewer.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_batch_malformed_job_and_training_rejection(public, tmp_path, monkeypatch, models):
    import src.backbone.batch as batch
    monkeypatch.setattr(batch, "ModelSession", lambda *args: models)
    path = tmp_path / "config.json"
    batch_configuration(path, public, [
        {"id": "valid"}, {"id": "malformed", "samples": "bad"},
        {"id": "illegal_training", "intended_use": "training_export"},
        {"id": "wrong_metadata", "match": "117092"},
    ])
    summary = batch.run_batch(path, tmp_path / "batch")
    assert summary["status"] == "FAILED"
    assert summary["counts"] == {"COMPLETE": 1, "FAILED": 3}
    for job in summary["jobs"][1:]:
        assert (Path(job["output"]) / "failure.json").exists()
        assert not (Path(job["output"]) / "completion.json").exists()
    assert "fitting matches" in summary["jobs"][2]["error"]
    assert "metadata" in summary["jobs"][3]["error"]


def test_training_records_separation_and_resume(tmp_path, monkeypatch, models):
    import src.backbone.batch as batch
    monkeypatch.setattr(batch, "ModelSession", lambda *args: models)
    path = tmp_path / "config.json"
    config = batch_configuration(path, Path("unused"), [
        {"id": "train", "source_adapter": "soccertrack_cache", "match": "117092",
         "intended_use": "training_export"},
    ])
    summary = batch.run_batch(path, tmp_path / "batch")
    assert summary["status"] == "COMPLETE"
    folder = Path(summary["jobs"][0]["output"]) / "training_sample"
    student = [json.loads(line) for line in (folder / "student.jsonl").read_text().splitlines()]
    teacher = [json.loads(line) for line in (folder / "privileged_exo.jsonl").read_text().splitlines()]
    optional = [json.loads(line) for line in (folder / "optional_targets.jsonl").read_text().splitlines()]
    assert len(student) == 10
    assert [row["sample_id"] for row in student] == [row["sample_id"] for row in teacher]
    assert len({row["sample_id"] for row in student}) == 10
    assert all(row["teacher_scores"] is None and row["receiver_proxy_target"] is None for row in optional)
    for row in student:
        assert "privileged_exo" not in row and "exo_reference" not in row
        assert "source_track" not in json.dumps(row)
        assert "track_1" not in json.dumps({key: row[key] for key in ("observation", "causal_memory", "world_input")})
    before = (folder / "student.jsonl").read_bytes()
    assert batch.run_batch(path, tmp_path / "batch", resume=True)["counts"] == {"SKIPPED_HASH_MATCH": 1}
    assert (folder / "student.jsonl").read_bytes() == before
    index = json.loads((tmp_path / "batch" / "training_index.json").read_text(encoding="utf-8"))
    assert len(index["shards"]) == 1 and index["shards"][0]["samples"] == 10
    assert not list((tmp_path / "batch").rglob("*.tmp"))


def test_reference_poison_does_not_change_player_path(public, tmp_path, monkeypatch, models):
    import src.backbone.batch as batch
    config_path = tmp_path / "config.json"
    config = batch_configuration(config_path, public, [{"id": "poison"}])
    job = {**config["defaults"], **config["jobs"][0]}
    folders = [tmp_path / "normal", tmp_path / "poisoned"]
    for folder in folders:
        folder.mkdir()
    batch.execute_job(ROOT, folders[0], job, models)
    monkeypatch.setattr(batch, "source_reference", lambda *_: {
        "label": "poisoned reference", "players": [{"source_track": "NEVER_SEEN", "xy": [1e6, 1e6], "team": 2}]})
    batch.execute_job(ROOT, folders[1], job, models)
    rows = [json.loads((folder / "replay.json").read_text())["sequences"][0]["actors"]["track_1"]
            for folder in folders]
    for a, b in zip(*rows):
        assert a["belief"] == b["belief"]
        assert a["assessment"] == b["assessment"]


def test_unknown_role_alone_preserved_and_disables_world(public, models):
    rewrite(public, lambda rows: [row["measurements"][0].update(goalkeeper=None) for row in rows[1:]])
    source = ExternalObservationSource(public)
    provider = SyntheticObservationProvider("track_1", fov=360, dropout=False)
    engine = ReplayEngine(models)
    rows = [engine.step(source.observe(frame, provider)) for frame in source.frames(1500, 15)]
    assert all(e["team"] in (1, 2) for e in rows[-1]["belief"]["entities"])
    assert any(e["goalkeeper"] is None for e in rows[-1]["belief"]["entities"])
    assert not rows[-1]["belief"]["model_contract_met"]
    assert any("unknown team/role" in warning for warning in rows[-1]["belief"]["warnings"])


@pytest.mark.parametrize("module,job_id", [("run", "first"), ("export", "train")])
def test_existing_entrypoints_route_external_and_training(public, tmp_path, monkeypatch, module, job_id):
    import importlib
    import sys
    import src.backbone.batch as batch
    path = tmp_path / "config.json"
    batch_configuration(path, public, [{"id": "first"}, {"id": "train", "intended_use": "training_export"}])
    calls = []
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "COMPLETE"}
    monkeypatch.setattr(batch, "run_batch", run)
    monkeypatch.setattr(sys, "argv", [
        "backbone", "--config", str(path), "--output", str(tmp_path / "output"), "--only", job_id, "--resume"])
    importlib.import_module(f"src.backbone.{module}").main()
    assert len(calls) == 1 and calls[0][1]["only"] == job_id
    assert calls[0][0][2] is True


def test_export_entrypoint_rejects_replay(public, tmp_path, monkeypatch):
    import sys
    from src.backbone.export import main
    path = tmp_path / "config.json"
    batch_configuration(path, public)
    monkeypatch.setattr(sys, "argv", [
        "export", "--config", str(path), "--output", str(tmp_path / "output"), "--only", "first"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_action_general_contract_does_not_assume_possession(public, models):
    source = ExternalObservationSource(public)
    frame = next(source.frames(1500, 1))
    observation = source.observe(frame, SyntheticObservationProvider("track_1"))
    result = ReplayEngine(models).step(observation, include_passing_assessment=False)
    assert result["assessment"] is None
    assert result["modules"]["passing"]["status"] == "UNSUPPORTED"
    assert result["action"] == {
        "schema_version": "football.action-output.0.1", "action_type": None,
        "teammate_target": None, "opponent_target": None, "location_xy": None,
        "direction_xy": None, "timing_start_s": None, "timing_end_s": None,
        "confidence": None, "support_status": "unsupported",
        "reason": "No action model is configured",
    }
    assert result["ball_state"] == {
        "position_status": "unknown", "xy": None, "possession_status": "unknown",
        "possessor_track": None, "possessor_team": None,
    }
    assert result["restart_context"]["status"] == result["outcome"]["status"] == "unknown"
    # A current tracking actor is merely the selected viewpoint; possession remains unknown.
    assert observation.actor is not None and result["ball_state"]["possessor_track"] is None
    ball = BallState("observed", (1., 2.), "unknown")
    assert ball.xy == (1., 2.) and ball.possessor_track is None
    estimated = BallState("estimated", (3., 4.), "estimated", "source_actor", 1)
    assert estimated.position_status == estimated.possession_status == "estimated"
    action = ActionOutput(action_type="carry", location_xy=(4., 5.), direction_xy=(1., 0.),
                          timing_start_s=1., timing_end_s=2., confidence=.7, support_status="partial")
    assert action.action_type == "carry" and action.support_status == "partial"
    with pytest.raises(ValueError, match="Unknown possession"):
        BallState("unknown", None, "unknown", "track_1")
    with pytest.raises(ValueError, match="cannot assert"):
        RestartContext("corner", "unknown")
    with pytest.raises(ValueError, match="cannot assert"):
        ActionOutcome("goal", "unknown")
    with pytest.raises(ValueError, match="cannot target both"):
        ActionOutput(teammate_target="a", opponent_target="b")
    # Any configured actor is eligible; the input has no possession requirement.
    off_ball = source.observe(frame, SyntheticObservationProvider("track_2"))
    assert off_ball.actor is not None
    off_ball_result = ReplayEngine(models).step(off_ball, include_passing_assessment=False)
    assert off_ball_result["ball_state"]["possession_status"] == "unknown"


def test_batch_output_can_omit_passing_module(public, tmp_path, monkeypatch, models):
    import src.backbone.batch as batch
    monkeypatch.setattr(batch, "ModelSession", lambda *args: models)
    path = tmp_path / "config.json"
    config = batch_configuration(path, public, [{"id": "no_passing", "passing_assessment": False}])
    summary = batch.run_batch(path, tmp_path / "batch")
    assert summary["status"] == "COMPLETE"
    rows = json.loads((Path(summary["jobs"][0]["output"]) / "replay.json").read_text())["sequences"][0]["actors"]["track_1"]
    assert all(row["assessment"] is None for row in rows)
    assert all(row["modules"]["passing"]["status"] == "UNSUPPORTED" for row in rows)
    assert all(row["action"]["support_status"] == "unsupported" for row in rows)
