"""Verify replay contracts, artifact hashes and produce the integration handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from src.analyze_dynamic_football import atomic_text
from src.trajectory_dataset import file_digest, write_json


def validate(root: Path, output: Path):
    run = json.loads((output / "run_manifest.json").read_text())
    for name, expected in run["files"].items():
        if file_digest(output / name) != expected:
            raise ValueError(f"Export changed: {name}")
    for name, expected in run["source_hashes"].items():
        if file_digest(root / name) != expected:
            raise ValueError(f"Integration source changed after export: {name}; export a fresh demo")
    experiment = root / "outputs" / "dynamic_gnn_tactical_20261019"
    if file_digest(experiment / "progress_summary.json") != run["research_queue_unchanged_sha256"]:
        raise ValueError("Paused research queue changed")
    frozen = json.loads((experiment / "provenance.json").read_text())["sha256"]
    for name, expected in frozen.items():
        if file_digest(root / name) != expected:
            raise ValueError(f"Historical source/data changed: {name}")
    replay = json.loads((output / "replay.json").read_text())
    counts = json.loads((output / "replay_validation.json").read_text())
    if len(replay["sequences"]) < 3 or len({s["source"]["match"] for s in replay["sequences"]}) < 2:
        raise ValueError("Three sequences from at least two matches are required")
    if not counts["has_disappearance_reappearance"]:
        raise ValueError("No declared disappearance/reappearance case")
    learned, assessed, hidden = 0, 0, 0
    for sequence in replay["sequences"]:
        for actor, rows in sequence["actors"].items():
            seen, previous = set(), -1
            for row in rows:
                if row["timestamp_s"] <= previous:
                    raise ValueError("Replay is not chronological")
                previous = row["timestamp_s"]
                for entity in row["belief"]["entities"]:
                    if entity["id"] not in seen and not entity["visible"]:
                        raise ValueError("Named memory entity appeared before first sight")
                    seen.add(entity["id"])
                    if not entity["visible"]:
                        hidden += 1
                    if entity["backend"] in ("G1", "G3", "G4"):
                        learned += 1
                        if entity["age_s"] > 2.0001 or not row["belief"]["model_contract_met"]:
                            raise ValueError("Learned estimate outside supported history")
                for candidate in row["assessment"]["candidates"]:
                    if candidate["id"] not in seen:
                        raise ValueError("Never-observed named candidate")
                    if candidate["status"] == "AVAILABLE":
                        assessed += 1
                        if not candidate["visible"]:
                            raise ValueError("Uncertain candidate treated as exact endpoint")
                if "evaluation_only" in row["belief"] or "exo_reference" in row["assessment"]:
                    raise ValueError("Reference and student records are not separated")
    if not learned or not assessed:
        raise ValueError("Demo lacks genuine learned hidden inference or geometric assessments")
    students = [json.loads(line) for line in (output / "training_sample" / "student.jsonl").read_text().splitlines()]
    teachers = [json.loads(line) for line in (output / "training_sample" / "privileged_exo.jsonl").read_text().splitlines()]
    if [row["sample_id"] for row in students] != [row["sample_id"] for row in teachers]:
        raise ValueError("Student/teacher sample IDs mismatch")
    if any("privileged_exo" in row or "receiver_proxy_target" in row for row in students):
        raise ValueError("Student export contains privileged targets")
    if counts["models_loaded_once"]["world"] != 1:
        raise ValueError("World model not loaded exactly once")
    test_xml = output / "acceptance_tests.xml"
    if not test_xml.exists():
        raise FileNotFoundError("Run pytest with --junitxml into the demo output before validation")
    tree = ET.parse(test_xml).getroot()
    suites = [tree] if tree.tag == "testsuite" else list(tree.findall("testsuite"))
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
             for key in ("tests", "errors", "failures", "skipped")}
    if tests["errors"] or tests["failures"]:
        raise ValueError("Acceptance suite has failures")
    result = {"status": "PASSED", "tests": tests, "source_and_protected_hashes": "UNCHANGED",
              "named_memory_first_sight": "PASSED", "model_history_boundary": "PASSED",
              "reference_separation": "PASSED", "actual_learned_hidden_estimates": learned,
              "actual_geometric_assessments": assessed, "hidden_memory_estimates": hidden,
              "training_samples": len(students), "research_status": "PAUSED_FOR_INTEGRATION"}
    write_json(output / "acceptance.json", result)
    report(root, output, run, counts, result)
    print(json.dumps(result, indent=2))
    return result


def report(root, output, run, counts, result):
    relative = output.relative_to(root).as_posix()
    tests = result["tests"]
    rows = counts["actor_runs"]
    estimates = sum(row["hidden_entity_estimates"] for row in rows)
    fallback = sum(row["fallback_entity_estimates"] for row in rows)
    lines = [
        "# October 31 backbone: integration sprint 1", "",
        "**Tracking-driven offline prototype: connected and acceptance-verified. Not a raw-video-to-ego system.**",
        "Dynamic-GNN research remains **PAUSED_FOR_INTEGRATION**. No new training, architecture search, "
        "annotation edits, environment upgrades, commits or pushes.",
        "", "## Run the demo", "", "From the repository root in PowerShell (choose a new output directory):", "",
        "```powershell",
        "& 'C:\\Users\\flexadmin\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe' -m src.backbone.run "
        "--config configs\\backbone_v0_1.json --output outputs\\backbone_v0_1_my_demo",
        "Start-Process .\\outputs\\backbone_v0_1_my_demo\\viewer.html",
        "```", "",
        f"Actual [self-contained replay viewer](../{relative}/viewer.html), "
        f"[replay records](../{relative}/replay.json), [acceptance checks](../{relative}/acceptance.json), "
        f"[provenance manifest](../{relative}/run_manifest.json). No server, downloads or synchronized RGB required.",
        "", "## Components actually connected", "",
        "| Module | Status | Contract / implementation |", "|---|---|---|",
        "| Match source | Working | Verified SoccerTrack cache adapter; 25Hz source, replay every 5 frames/0.2s; pitch-centred metres; observed mask and segment IDs, no interpolation |",
        "| Synthetic observations | Working, not real ego perception | Existing causal motion-heading/FOV helpers, 120deg/45m; streaming seeded occlusions; full-state positions stop at this boundary |",
        "| First-sight memory | Working | Anonymous identities only after sight; source IDs for association only, not learned features; gaps/reset/expiry explicit |",
        "| G1 learned belief | Working frozen checkpoint | Exact 11 samples/2s, reinitialized per-window encoder, fixed coordinate/velocity/age scales; mean, diagonal sigma, separate occupancy |",
        "| Geometric assessment | Working | Existing Gaussian sampler, 2/3/5m lane thresholds, 2m endpoint exclusions; known defenders only |",
        "| Exo P0 | Working optional reference | Saved statistics loaded once; supported full-state inputs; distinct from player panel; no masked/imputed P0 policy |",
        "| Replay | Working | Play/pause, step/seek, configured actor selection, memory toggle, uncertainty, warnings; truth/teacher overlays OFF by default |",
        "| Future training export | Working example | Eight fitting-match samples; student, privileged teacher and optional targets in separate keyed files; NO teacher fitting |",
        "| GNN backends | Optional wrappers | G3/G4 accepted by the same world interface; not required or promoted over G1 |",
        "| Raw video, calibrated facing, elite broadcast learning | NOT implemented | Replaceable future modules, not fabricated |",
        "| Memory-aware receiver policy / real outcome labels | NOT implemented | Provisional geometric ordering only; next-event P0 is not intention or pass-success evidence |",
        "", "## Inference and lifecycle policy", "",
        "- `ReplayEngine.step(Observation)` cannot accept a full tracking frame or privileged reference. "
        "`EvaluationReference` and full-state P0 stay outside the engine.",
        "- G1 weights were learned from exo tracking, transferred here to simulated partial observations. "
        "The wrapper matches the existing CPU `infer_world` path within absolute/relative tolerance 1e-5.",
        "- Learned inputs `[1,11,22,10]`: actor-relative x/52.5,y/34; vx/12,vy/12; current visibility, ever-seen; "
        "min(age,8)/8; known team relation, goalkeeper, actor. Bool first-sight mask, absolute actor/last-seen xy in metres.",
        "- First sample velocities are zero. No more than 22 first-sight entities; capacity overflow or unknown team "
        "disables learned inference explicitly. Actor must be continuously observed across all 11 samples.",
        "- Warm-up is 10 samples. Missing actor/time gaps/observed identity discontinuities reset the learned context. "
        "Observed segment changes retire prior anonymous memory. Half/source changes reset memory.",
        "- Memories older than the 2s model window use the configured **last-seen** baseline, with sigma "
        "`1.5 + 1.1*age` metres, then expire after **6s**. Constant velocity is an explicitly selectable fallback. "
        "These heuristic fallback sigmas are not measured calibrated perception uncertainty.",
        "- Backwards engine stepping is rejected. `replay_prefix` reconstructs preceding observations; "
        "viewer seeking indexes states already computed sequentially from the segment start.",
        "- Hidden remembered teammates are displayed with sigma and age, but lane risk is **unavailable**: "
        "the current implementation does not integrate uncertain candidate endpoints.",
        "- Gaussian lane risk uses visible/remembered defenders only. Anonymous occupancy is shown separately "
        "and never treated as a set of named players or fused with known-memory risk. The occupancy head's "
        "'never seen' means unseen within the current **2s window**, not necessarily never seen in a long replay.",
        "- Partial geometry is not a completeness claim. No known defenders, unknown actor team and short lanes "
        "produce explicit unavailable assessments, not a success-shaped safe score.",
        "- P0 normalization buffers are loaded directly from its completed checkpoint, once per session; "
        "parity with existing `infer_decision` on identical inputs passed at tolerance 1e-5.",
        "", "## Actual replay acceptance cases", "",
        "Three declared segments from retrospective 118577 and 128058; two configured actors each, "
        "360 sequential player frames total. Selection: fixed starts 60s/120s, first two outfield slots with "
        ">=95% actor coverage, >=20/60 frames with a visible teammate, >=1 disappearance/reappearance. "
        "No prediction-success criterion. These are retrospective functional checks, not tactical-accuracy tests.",
        "", "| Segment / actor | G1 hidden estimates | Fallback / hidden | Available lanes | p50 / p95 ms | Returned tracks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['sequence']} / {row['actor']} | {row['learned_hidden_entity_estimates']} | "
                     f"{row['fallback_entity_estimates']}/{row['hidden_entity_estimates']} | "
                     f"{row['available_lane_assessments']} | {row['latency_ms_p50']:.1f} / "
                     f"{row['latency_ms_p95']:.1f} | {row['coverage']['disappeared_then_reappeared']} |")
    lines += ["", f"Hidden fallback frequency: **{fallback}/{estimates} = {100*fallback/max(1,estimates):.1f}%**. "
              "Failed replay frames: **0** (processing stops visibly on inference errors; none occurred). "
              "All actor samples observed in selected real sequences; synthetic lifecycle tests also cover actor gaps.",
              "All selected source frames contain at least one absent cache slot. This can include unused roster slots "
              "and is **not** interpreted as a physically missing-player count. Counts and warnings are preserved.",
              "Latency covers memory + belief + candidate assessment only; it excludes source/model loading, "
              "file export, viewer rendering and the optional P0 reference. CPU observations are not a real-time guarantee.",
              "", "## Tests and UI verification", "",
              f"**{tests['tests'] - tests['skipped']} passed; {tests['failures']} failed; "
              f"{tests['errors']} errors; {tests['skipped']} skipped.**",
              "Focused checks cover prefix consistency, fixed-observation hidden-truth isolation, first-sight identities, "
              "world/P0 inference parity, fixed scaling, lane geometry and uncertainty, unavailable hidden candidates, "
              "warm-up/gaps/reappearance/expiry/discontinuities/half reset/seeking, fitting provenance and train-only export.",
              "Browser acceptance checks are recorded separately in the demo's `browser_validation.json` "
              "(play/pause, timeline/step, actor/sequence selection, memory, truth/teacher toggles).",
              "", "## Versioned adapter / learning-export contract", "",
              "Common record version: `backbone.0.1`. A `TrackingFrame` has source ID, half, chronological frame index, "
              "timestamp seconds, finite pitch-centred metre coordinates for valid tracks, known team 1/2 or explicit "
              "unknown 0, goalkeeper flag, association token and observed segment token. Missing detections are absent, "
              "never interpolated from future data. Source provenance carries hashes, fps and quality caveats.",
              "An `Observation` carries only actor-visible measurements; no hidden positions or future labels. "
              "Source track associations may persist internally, but become anonymous first-sight slots before learning. "
              "A broadcast/ego adapter must establish coordinate calibration, timestamps, identity continuity and "
              "measurement validity. Unknown team or unsupported sample cadence must remain explicit; no detector/"
              "calibration or measured facing is implemented in this sprint.",
              "Configuration selects adapter, match/half, segment, explicit actors, model, synthetic FOV and memory policy. "
              "`load_source` currently supports only `soccertrack_cache`; unsupported adapters raise errors.",
              "Student export holds allowed observations, causal memory, exact model inputs and missingness. "
              "Privileged exo supervision is a separate file with source-confidence caveats. Optional event/receiver "
              "targets are null here because arbitrary replay timestamps are not asserted pass decisions.",
              "", "Small sample-only command (new output directory, permitted fitting matches enforced):", "",
              "```powershell",
              "& 'C:\\Users\\flexadmin\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe' -m src.backbone.export "
              "--config configs\\backbone_v0_1.json --output outputs\\backbone_training_sample",
              "```", "",
              "Checkpoint provenance: all selected models fit 117092,117093,118576,118578,128057,132831; "
              "132877 development-only. Retrospective demos are labeled accordingly; the example training export "
              "uses fitting match 117092 and is never labeled held-out.",
              "", "## Next integration priority", "",
              "Replace synthetic visibility with a validated observation adapter, while preserving this tested "
              "student/evaluation boundary and record schema. Prioritize camera-to-pitch tracking/identity continuity "
              "and time alignment before real ego claims. A trained memory-aware pass policy and independent "
              "outcome labels remain later work; no more architecture ablations are needed for this prototype."]
    atomic_text(root / "results" / "backbone_v0_1_handoff.md", "\n".join(lines) + "\n")


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "batch" / "batch_summary.json").exists():
        from src.backbone.validate_v02 import validate as validate_ingestion
        validate_ingestion(args.output.resolve())
        return
    validate(root, args.output.resolve())


if __name__ == "__main__":
    main()
