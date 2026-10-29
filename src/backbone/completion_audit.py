"""Package and verify completion evidence without promoting blocked capabilities."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from src.backbone.lazy_windows import validate as validate_lazy
from src.backbone.regional_bridge_completion import validate as validate_bridge
from src.backbone.source import EvaluationReference, SoccerTrackSource
from src.backbone.streaming_completion import HALVES, ROOT, validate as validate_halves
from src.backbone.transfer_completion_audit import validate_saved as validate_transfer_audit
from src.backbone.train_action import _metrics
from src.player_centric_dataset import ACTION_LABELS
from src.trajectory_dataset import file_digest, write_json


def run_regressions(output: Path) -> None:
    files = [
        "tests\\test_backbone.py", "tests\\test_backbone_ingestion.py", "tests\\test_backbone_v0_3.py",
        "tests\\test_backbone_v0_4.py", "tests\\test_backbone_v05.py",
        "tests\\test_backbone_v06_robust_action.py", "tests\\test_backbone_v06_video.py",
        "tests\\test_backbone_transfer_completion.py", "tests\\test_backbone_streaming_completion.py",
    ]
    command = [sys.executable, "-m", "pytest", *files, "-q"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (output / "regression_output.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)
    matched = re.search(r"(\d+) passed", completed.stdout)
    if matched is None:
        raise ValueError("Successful pytest command did not report passed test count")
    write_json(output / "regression_validation.json", {
        "status": "PASS", "passed": int(matched.group(1)), "returncode": completed.returncode,
        "files": files, "scope": "Fixture tests plus explicit local-checkpoint/corpus integration tests",
        "stdout_sha256": file_digest(output / "regression_output.txt"),
    })


def package_viewer(output: Path) -> dict:
    sequences = []
    model_provenance, action_provenance = {}, {}
    for match, half in HALVES:
        folder = output / "full_halves" / f"{match}_{half}"
        replay = json.loads((folder / "selected_replay.json").read_text(encoding="utf-8"))
        sequence = replay["sequences"][0]
        actors = list(sequence["actors"])
        indices = [row["frame_index"] for row in sequence["actors"][actors[0]]]
        if any([row["frame_index"] for row in sequence["actors"][actor]] != indices for actor in actors):
            raise ValueError("Selected actor timelines differ; source overlays cannot be positionally joined")
        source = SoccerTrackSource(ROOT, match, half)
        reference = EvaluationReference(source)
        sequence["evaluation_only"] = [reference.overlay(source.frame(index)) for index in indices]
        sequence["dataset_relation"] = "RETROSPECTIVE: previously evaluated; not a new independent test"
        sequences.append(sequence)
        model_provenance, action_provenance = replay["model_provenance"], replay["action_model_provenance"]
    bundle = {"schema_version": "backbone.0.1", "units": "metres", "sampling_seconds": .2,
              "notice": "Selected rows from two complete halves; timeline gaps retain full-prefix memory. Truth and annotations are evaluation-only.",
              "description": "G1 belief plus original S4 event-conditioned actions; passing disabled",
              "truth_overlay_default": False, "exo_reference_default": False,
              "model_provenance": model_provenance, "action_model_provenance": action_provenance,
              "config": {"full_stream": "full_halves", "viewer_selections_only": True},
              "sequences": sequences}
    write_json(output / "selected_replay.json", bundle)
    template = ROOT / "src" / "backbone" / "viewer.html"
    payload = json.dumps(bundle, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (output / "viewer.html").write_text(template.read_text(encoding="utf-8").replace("__REPLAY_JSON__", payload),
                                      encoding="utf-8")
    return {"sequences": len(sequences), "actors_per_half": [len(row["actors"]) for row in sequences],
            "viewer_sha256": file_digest(output / "viewer.html")}


def package_review(output: Path) -> None:
    import cv2
    from src.backbone.regional_bridge_completion import PRIOR

    folder = output / "review_only"
    folder.mkdir(exist_ok=True)
    review = json.loads((output / "video_bridge" / "review_only.json").read_text(encoding="utf-8"))
    clip = PRIOR / "video_corpus" / "fitting" / "117092_1st_00240s_1200s"
    metadata = json.loads((clip / "clip_manifest.json").read_text(encoding="utf-8"))
    video = cv2.VideoCapture(str(ROOT / metadata["source_video"]), cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_N_THREADS, 4])
    if not video.isOpened():
        raise RuntimeError("Could not open the source for bounded rejected-region review")
    images = []
    try:
        for row in review["records"]:
            index = row["native_frame"]
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, image = video.read()
            if not success:
                raise RuntimeError(f"Could not decode review-only native frame {index}")
            cv2.line(image, (1510, 0), (1510, image.shape[0] - 1), (0, 0, 255), 3)
            for detection in row["detections"]:
                x1, y1, x2, y2 = map(round, detection["bbox_xyxy_px"])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(image, f"REJECTED REGION | {row['timestamp_s']:.1f}s | NO PITCH PROJECTION",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            image = cv2.resize(image, (1920, round(image.shape[0] * 1920 / image.shape[1])))
            name = f"rejected_region_frame_{index}.jpg"
            if not cv2.imwrite(str(folder / name), image):
                raise RuntimeError("Could not write bounded review image")
            images.append((name, row["timestamp_s"], row["reason"]))
    finally:
        video.release()
    entries = []
    stress = json.loads((output / "development_stress_v2" / "result.json").read_text(encoding="utf-8"))
    for row in stress["summaries"]:
        if row["condition"] not in ("identity_after_memory", "unknown_team_role"):
            continue
        label, condition = row["label_evaluation_only"], row["condition"]
        entries.append(f'<li><a href="../development_stress_v2/{html.escape(label)}_{condition}.json">'
                       f'{html.escape(label)}: {condition}, event {html.escape(row["sample_id"])}</a></li>')
    contents = ('<!doctype html><meta charset="utf-8"><title>Review-only failure pack</title>'
                '<h1>Review only: rejected calibration, identity changes and unknown roles</h1>'
                '<p>No human labels were filled. Red line bounds the proposed ROI, not a physical camera.'
                ' Yellow boxes are unverified detector outputs. No accepted pitch positions.</p>')
    for name, timestamp, reason in images:
        contents += f'<h2>{timestamp:.1f} source seconds</h2><p>{html.escape(reason)}</p><img width="960" src="{name}">'
    contents += '<h2>Post-memory identity / unknown-metadata cases</h2><ul>' + "".join(entries) + "</ul>"
    (folder / "index.html").write_text(contents, encoding="utf-8")


def prefix_parity(output: Path) -> dict:
    filename = "chunk_000000_000300.jsonl.gz"
    pilot = output / "full_halves_pilot" / "118577_1st" / filename
    resumed = output / "full_halves" / "118577_1st" / filename
    digests = []
    for path in (pilot, resumed):
        digest = hashlib.sha256()
        with gzip.open(path, "rb") as reader:
            for line in reader:
                digest.update(line)
        digests.append(digest.hexdigest())
    if digests[0] != digests[1]:
        raise ValueError("Worker restart changed the previously committed four-actor prefix")
    return {"status": "PASS", "compared_prefix_actor_steps": 1200, "sha256": digests[0],
            "duplicates": 0, "comparison": "Decompressed numerical rows, timestamps, identities and scores"}


def validate_stress(output: Path) -> dict:
    folder = output / "development_stress_v2"
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if result["development_match"] != "132877" or len(result["summaries"]) != 49:
        raise ValueError("Missing fixed development stress cases")
    cases, gap_abstentions, identity_breaks = set(), 0, 0
    for summary in result["summaries"]:
        label, condition = summary["label_evaluation_only"], summary["condition"]
        row = json.loads((folder / f"{label}_{condition}.json").read_text(encoding="utf-8"))
        if row["query"] != summary["query"] or row["stats"] != summary["stats"]:
            raise ValueError("Development stress summary differs from actual saved query")
        query = row["query"]
        if query["frame_index"] != row["annotation_evaluation_only"]["timing"]["query_anchor_frame"]:
            raise ValueError("Development action input/query timestamp changed")
        if condition in ("short_source_gap", "long_source_gap", "unknown_team_role"):
            if query["action"]["support_status"] != "unsupported":
                raise ValueError("Unsupported source state received a model-supported action")
            gap_abstentions += 1
        if condition == "identity_after_memory":
            warnings = [warning for step in row["temporal_rows"] for warning in step["belief"]["warnings"]]
            if not any("OBSERVED_IDENTITY_DISCONTINUITY" in warning for warning in warnings):
                raise ValueError("Post-memory identity switch absent from actual replay")
            identity_breaks += 1
        cases.add((label, condition))
    if len(cases) != 49:
        raise ValueError("Duplicate development stress cases")
    return {"status": "PASS", "cases": len(cases), "source_failure_abstentions": gap_abstentions,
            "post_memory_identity_break_cases": identity_breaks, "actor_steps": result["actor_steps"]}


def score_queries(output: Path) -> dict:
    examples, truth, prediction, excluded = [], [], [], []
    for match, half in HALVES:
        folder = output / "full_halves" / f"{match}_{half}"
        state = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        chunks = sorted(folder.glob("chunk_*.jsonl.gz"))
        if len(chunks) != state["chunks"]:
            raise ValueError("Unexpected full-half query-scoring shard count")
        for path in chunks:
            with gzip.open(path, "rt", encoding="utf-8") as reader:
                for line in reader:
                    if '"recorded_annotations_evaluation_only":' not in line:
                        continue
                    row = json.loads(line)
                    annotations = row["recorded_annotations_evaluation_only"]
                    if len(annotations) != 1 or row["action_inference"]["status"] != "supported":
                        excluded.append({"sample_id": row["sample_id"],
                                         "reason": "Multiple annotations or unsupported input; no forced single target"})
                        continue
                    annotation = annotations[0]
                    if annotation["timing"]["query_anchor_frame"] != row["frame_index"]:
                        raise ValueError("Scheduled annotation moved relative to action query")
                    label = annotation["mapped_action_label"]
                    predicted = row["action"]["action_type"]
                    truth.append(ACTION_LABELS.index(label))
                    prediction.append(ACTION_LABELS.index(predicted))
                    examples.append({"sample_id": row["sample_id"], "event_id": annotation["event_id"],
                                     "annotation_index": annotation["annotation_index"], "recorded": label,
                                     "predicted": predicted, "correct": label == predicted})
    if not examples:
        raise ValueError("Full-half demonstration returned no scoreable real action outputs")
    result = {"status": "PASS", "scored_queries": len(examples), "explicit_exclusions": excluded,
              "metrics": _metrics(np.asarray(truth), np.asarray(prediction)), "examples": examples,
              "protocol": "Complete-prefix continuous memory, exact native-grid event queries only",
              "comparison_warning": "NOT directly compared to separately initialized event-tensor benchmark scores",
              "taxonomy": list(ACTION_LABELS)}
    write_json(output / "full_halves" / "scheduled_action_metrics.json", result)
    return {key: value for key, value in result.items() if key != "examples"}


def requirement_ledger(output: Path, evidence: dict) -> None:
    original = output / "original_assignment.md"
    if file_digest(original) != "4875b886856b29dfe3a98ba400896b61b2145397acf34f9994e64b48c0befcb8":
        raise ValueError("Original assignment wording changed")
    rules = [
        ("Plan for 16–19 hours", "FAIL", "Original v0.6 stopped after4h2m; this continuation is separate and does not fulfill the original wall-clock target."),
        ("Run the integrated system on two complete", "PASS", "93.45 native source minutes,28035 source samples,112140 actor steps; full_halves/validation.json"),
        ("Generate keyed out-of-fold probabilities", "PASS", "12038 keyed OOF examples,three match-excluded folds,six-fit development/retrospective reference; transfer_audit/validation.json"),
        ("Each teacher fits only the other four matches", "PASS", "Scratch initialization,exact actual fit specifications,checkpoint/optimizer/RNG hashes independently checked; transfer_audit/evaluation.json"),
        ("C1: action supervision", "PASS", "Matched original-initialized seed42 control executed192 optimizer steps; teacher/student provenance registry."),
        ("C2: the same setup", "PASS", "Matched C2 executed192 optimizer steps with keyed OOF KL targets; no improvement claimed."),
        ("Run browser assertions on real outputs", "PASS", "Actual Play/Pause,8 actor choices,two long seeks,uncertainty and reference invariance; browser_validation.json"),
        ("Force one controlled worker interruption", "PASS", "Safe worker exit after first300 samples,actual restart,1200-prefix actor-step equality,zero duplicates; optimizer two-step control versus actual one-step continuation."),
        ("Evaluate a fixed development-only subset under", "PASS", "49 class/condition streams,5929 actor steps; all7 trained classes; missing/unknown metadata abstains; post-memory identity changes exercised."),
        ("- Baseline observations.", "PASS", "Seven fixed development-class baseline streams."),
        ("- Short and long source gaps.", "PASS", "Fourteen streams:0.6s/2.8s source gaps; every affected query abstained."),
        ("- Position noise.", "PASS", "Seven streams:independent observed-coordinate Gaussian sigma1m, applied before memory."),
        ("- Unknown team/role metadata.", "PASS", "Seven streams:team0/roleNone preserved; every affected query abstained."),
        ("- Camera/identity discontinuities.", "PASS", "Fourteen streams:explicit camera reset and previously observed non-actor identity changes after12s."),
        ("2. Build a lazy index", "PASS", "100000 distinct fitting-only windows across12 indexed halves,63267 compressed bytes,11932 missing/unknown actor windows retained."),
        ("3. Remove the single worst", "FAIL", "A profiled observation hotspot was removed:4.40x observation and1.39x full-replay speedup with exact output parity. Global-worst claim not established."),
        ("Feed accepted detector-derived pitch positions", "BLOCKED", "New narrow region failed:2 within-hull withheld points,median8.35m/p9010.99m. Zero accepted pitch records."),
        ("If a mapping passes its frozen gate", "BLOCKED", "No mapping passed; detector footpoint and identity accuracy also unverified."),
        ("1. Expand successful video ingestion", "NOT_RUN", "Further uncalibrated footage expansion deprioritized by user's explicit correction; no additional pixel-inference volume counted."),
        ("4. Run the resulting source-independent adapter", "PASS", "Unmodified public adapter/roundtrip CLI,128058 second half,600 source samples,exact observation/prediction parity; same-dataset interoperability only."),
        ("5. Build a small review-only diagnostic pack", "PASS", "review_only/index.html:3 original-time rejected-region images plus14 identity/unknown-role case links; no human labels filled."),
        ("Run relevant existing regressions and new behavioral tests.", "PASS", "73 local regression tests;9 portable source-only fixture tests separately reported. Local-artifact tests intentionally require explicit caches/checkpoints, not a data-free installation."),
    ]
    rows = []
    for number, text in enumerate(original.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip() or text.startswith("#"):
            continue
        matched = next((rule for rule in rules if rule[0] in text), None)
        rows.append({"original_line": number, "original_wording": text,
                     "status": matched[1] if matched else "NOT_RUN",
                     "measured_result": matched[2] if matched else
                     "Not separately re-audited in this continuation; no PASS inferred from historical stage labels.",
                     "scope": "Original wording retained; inherited original v0.6 remains PARTIAL"})
    write_json(output / "requirement_ledger.json", {"original_sha256": file_digest(original),
               "policy": "Every non-heading nonempty original line preserved. NOT_RUN means not separately re-audited, not necessarily absent historically.",
               "requirements": rows, "continuation_gates": evidence["gates"]})


def record_decisions(output: Path, gates: dict, now: datetime, remaining: float) -> None:
    journal = output / "decision_log.jsonl"
    prior = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    stages = {
        "transfer": ("transfer_audit\\evaluation.json", "PASS_AND_ADVANCE", "Valid matched transfer; poor OOF teacher and harmful C2. Keep original S4."),
        "full_halves": ("full_halves\\manifest.json", "PASS_AND_ADVANCE", "Both complete native extents,93.45 minutes,112140 actor steps; no duplicated records."),
        "lazy_index": ("lazy_windows\\windows.npz", "PASS_AND_ADVANCE", "100000 target-free fitting windows,missing actor/metadata retained."),
        "video_bridge": ("video_bridge\\result.json", "BLOCK_DEPENDENT_BRANCH", "Rejected regional mapping; no accepted video-derived pitch observations or video action-inference claim."),
        "browser": ("viewer.html", "REPAIR_AND_RETEST", "Real pointer interaction and reference toggles passed after verified missing-reference and sticky-header repairs."),
        "interoperability": ("interoperability\\validation.json", "PASS_AND_ADVANCE", "Another recording through unchanged public adapter; exact600-sample parity,not cross-dataset generalization."),
    }
    with journal.open("a", encoding="utf-8") as writer:
        for stage, (path, decision, explanation) in stages.items():
            fingerprint = file_digest(output / path)
            if any(row.get("stage") == stage and row.get("evidence_sha256") == fingerprint for row in prior):
                continue
            writer.write(json.dumps({"observed_at_utc": now.isoformat(), "stage": stage,
                         "decision": decision, "measured_evidence": gates.get(stage),
                         "evidence_path": path, "evidence_sha256": fingerprint, "explanation": explanation,
                         "remaining_seconds": remaining,
                         "next_cost": "Verification/packaging only; no additional fits or uncalibrated-video volume"}) + "\n")


def audit(output: Path) -> dict:
    halves = validate_halves(output / "full_halves")
    transfer = validate_transfer_audit(output / "transfer_audit")
    lazy = validate_lazy(output / "lazy_windows")
    bridge = validate_bridge(output / "video_bridge")
    bridge.update(validation_status=bridge["status"], status="BLOCKED")
    prefix = prefix_parity(output)
    stress = validate_stress(output)
    queries = score_queries(output)
    browser = json.loads((output / "browser_validation.json").read_text(encoding="utf-8"))
    if (browser["viewer_sha256"] != file_digest(output / "viewer.html") or browser["browserErrors"]
            or not all(browser[key] for key in ("initialTruthOff", "playPause", "longSeek", "actorSwitching"))):
        raise ValueError("Browser evidence is absent, failed or belongs to a different generated viewer")
    regressions = json.loads((output / "regression_validation.json").read_text(encoding="utf-8"))
    if (regressions["returncode"] != 0
            or regressions["stdout_sha256"] != file_digest(output / "regression_output.txt")):
        raise ValueError("Regression process failed or output changed")
    interoperability = json.loads((output / "interoperability" / "validation.json").read_text(encoding="utf-8"))
    if (interoperability["status"] != "PASSED" or interoperability["frames"] != 600
            or interoperability["observation_parity"] != "EXACT"
            or interoperability["prediction_parity"] != "EXACT_CPU_EXCLUDING_LATENCY"
            or interoperability["source"]["input_sha256"] !=
            file_digest(output / "interoperability" / "128058_2nd.jsonl")):
        raise ValueError("Second-recording public-adapter parity failed or source changed")
    optimizer = json.loads((output / "transfer_audit" / "optimizer_continuation.json").read_text(encoding="utf-8"))
    if (optimizer["next_step_maximum_difference"] != 0 or optimizer["uninterrupted_control_steps"] != 2
            or optimizer["optimizer_steps_after_reload"] != 1):
        raise ValueError("Disposable interrupted/uninterrupted optimizer continuation failed")
    queue_path = ROOT / "outputs" / "dynamic_gnn_tactical_20261019" / "progress_summary.json"
    if json.loads(queue_path.read_text(encoding="utf-8"))["status"] != "PAUSED_FOR_INTEGRATION":
        raise ValueError("Research GNN queue is not paused")
    program = json.loads((output / "program.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    started = datetime.fromisoformat(program["started_at_utc"].replace("Z", "+00:00"))
    deadline = datetime.fromisoformat(program["continuation_deadline_utc"].replace("Z", "+00:00"))
    registry = {}
    for name in ("teacher_fold_0", "teacher_fold_1", "teacher_fold_2", "teacher_reference", "C1", "C2"):
        folder = output / "transfer" / name
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if (file_digest(folder / "best.pt") != result["best_sha256"]
                or file_digest(folder / "last.pt") != result["resume_sha256"]):
            raise ValueError("Preserved trained checkpoint changed")
        registry[name] = {"checkpoint": str(folder / "best.pt"), "sha256": result["best_sha256"],
                          "resume_checkpoint": str(folder / "last.pt"),
                          "executed_optimizer_steps": result["steps"], "specification": result["specification"],
                          "deployment": "EXPERIMENTAL; original default unchanged"}
    source_registry = json.loads((output / "lazy_windows" / "manifest.json").read_text(encoding="utf-8"))["sources"]
    full_spec = json.loads((output / "full_halves" / "118577_1st" / "specification.json").read_text(encoding="utf-8"))
    write_json(output / "model_source_registry.json", {"models": registry, "sources": source_registry,
               "default": full_spec["action"], "world_belief": full_spec["world"],
               "score_semantics": "Uncalibrated seven-class event-conditioned probabilities; no tactical value",
               "normalization_and_inputs": "Fixed pitch/velocity scales, 11 causal samples over2s; see each exo input manifest and original S4 adapter config",
               "ball_and_possession": "UNKNOWN; selected actor need not possess ball"})
    result = {
        "schema_version": "football.backbone-completion-acceptance.1", "status": "PARTIAL",
        "reason": "Transfer and full-half/indexing workloads completed; accepted video-to-pitch bridge remains BLOCKED",
        "original_v06_status": "PARTIAL, unchanged; continuation does not retroactively reset its expired deadline",
        "checked_at_utc": now.isoformat(), "elapsed_seconds": (now - started).total_seconds(),
        "remaining_seconds": max(0, (deadline - now).total_seconds()),
        "gates": {"transfer": transfer, "full_halves": halves, "lazy_index": lazy,
                  "video_bridge": bridge, "resumed_four_actor_prefix": prefix, "development_stress": stress,
                  "browser": browser, "regressions": regressions, "interoperability": interoperability},
        "scheduled_full_half_actions": queries,
        "optimizer_continuation": optimizer,
        "gnn_queue": "PAUSED_FOR_INTEGRATION",
        "no_commit_push_or_human_annotation_changes": "Policy retained; no such operations executed",
        "not_run_or_not_established": [
            "Accepted detector-derived pitch observations or video-based action inference",
            "General improved teacher/student accuracy or calibrated confidence",
            "Global-worst bottleneck claim (one profiled observation hotspot was removed)",
            "Additional uncalibrated footage expansion, deprioritized by user's explicit observation-bridge correction",
            "Real ego-view perception or broad tactical recommendations",
        ],
    }
    write_json(output / "acceptance_record.json", result)
    program["status"] = "PARTIAL"
    program["last_verified_at_utc"] = now.isoformat()
    for priority in program["priorities"]:
        if priority["id"] == "full_halves":
            priority.update(status="PASS", evidence="full_halves/validation.json",
                            decision="Complete native halves verified; no duration trimming or chunk-memory reset.",
                            result="Two complete halves,93.45 native minutes,28035 source samples,112140 actor steps;95 supported exact-grid event queries.")
        elif priority["id"] == "transfer":
            priority["evidence"] = "transfer/validation.json; transfer_audit/validation.json; transfer_audit/optimizer_continuation.json"
            priority["decision"] = "Valid transfer path and matched comparison completed. C2 harmful on development and retrospective conditions; original S4 retained. No further fits."
    write_json(output / "program.json", program)
    write_json(output / "progress.json", {"status": "PARTIAL", "checked_at_utc": now.isoformat(),
               "elapsed_seconds": result["elapsed_seconds"], "remaining_seconds": result["remaining_seconds"],
               "completed_workloads": ["matched_teacher_student_path", "two_complete_halves", "100000_lazy_windows",
                                       "post_memory_stress", "interrupted_optimizer_continuation"],
               "blocked": ["accepted_video_to_pitch_observations"],
               "next_decision": "BLOCK_DEPENDENT_BRANCH; retain original S4 and rejected-region evidence"})
    write_json(output / "stage_plan.json", {"source": "program.json and original_assignment.md",
               "original_deadline_not_reset": True, "priorities": program["priorities"],
               "overall_status": "PARTIAL; no substitution of fallback results for blocked capability"})
    requirement_ledger(output, result)
    record_decisions(output, result["gates"], now, result["remaining_seconds"])
    return result


def resume(output: Path) -> None:
    program = json.loads((output / "program.json").read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(program["continuation_deadline_utc"].replace("Z", "+00:00"))
    deadline -= timedelta(hours=2)
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("Continuation computation deadline expired; use --phase verify. Deadline not reset.")

    def command(module: str, *arguments: str) -> None:
        if (output / "STOP_REQUESTED").exists():
            raise InterruptedError("Continuation pause marker present; preserved outputs not modified")
        subprocess.run([sys.executable, "-m", f"src.backbone.{module}", *arguments], cwd=ROOT, check=True)

    transfer_phase = "validate" if (output / "transfer" / "comparison.json").exists() else "run"
    command("transfer_v06", "--phase", transfer_phase, "--output", str(output / "transfer"),
            "--deadline-utc", deadline.isoformat())
    stages = [
        ("full_halves\\manifest.json", "streaming_completion",
         ["--phase", "run", "--output", str(output / "full_halves"), "--deadline-utc", deadline.isoformat()]),
        ("lazy_windows\\manifest.json", "lazy_windows", ["--phase", "build", "--output", str(output / "lazy_windows")]),
        ("transfer_audit\\evaluation.json", "transfer_completion_audit",
         ["--phase", "evaluate", "--transfer", str(output / "transfer"), "--output", str(output / "transfer_audit")]),
        ("transfer_audit\\optimizer_continuation.json", "transfer_completion_audit",
         ["--phase", "optimizer-probe", "--transfer", str(output / "transfer"), "--output", str(output / "transfer_audit")]),
        ("development_stress_v2\\result.json", "stress_completion", ["--output", str(output / "development_stress_v2")]),
        ("video_bridge\\result.json", "regional_bridge_completion", ["--output", str(output / "video_bridge")]),
    ]
    for artifact, module, arguments in stages:
        if not (output / Path(artifact)).exists():
            command(module, *arguments)
    package_viewer(output)
    package_review(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("package", "verify", "resume"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "resume":
        resume(args.output)
        run_regressions(args.output)
        result = audit(args.output)
    elif args.phase == "package":
        result = package_viewer(args.output)
        package_review(args.output)
    else:
        run_regressions(args.output)
        result = audit(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
