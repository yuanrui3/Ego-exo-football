"""Acceptance checks for persistent v0.2 outputs, not physical source certification."""

import argparse
import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from src.backbone.batch import completed_matches, implementation_hashes
from src.dynamic_football_data import FIT_MATCHES
from src.trajectory_dataset import file_digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate(output: Path):
    protected = read(output / "protected_before.json")
    for name, expected in protected.items():
        require(file_digest(ROOT / name) == expected, f"Protected artifact changed: {name}")
    tree = ET.parse(output / "acceptance_tests.xml").getroot()
    suites = [tree] if tree.tag == "testsuite" else tree.findall("testsuite")
    tests = {name: sum(int(s.attrib.get(name, 0)) for s in suites)
             for name in ("tests", "errors", "failures", "skipped")}
    require(tests["tests"] > 0 and tests["errors"] == 0 and tests["failures"] == 0, "Regression suite failed")
    initial = read(output / "batch" / "last_executed_summary.json")
    resumed = read(output / "batch" / "batch_summary.json")
    require(initial["status"] == resumed["status"] == "COMPLETE", "Batch has failures")
    require(len(initial["jobs"]) == 10 and resumed["counts"] == {"SKIPPED_HASH_MATCH": 10}, "Actual 10-job resume not verified")
    require(initial["model_sessions_loaded"] == 1 and resumed["model_sessions_loaded"] == 0,
            "Compatible models were not reused or resume loaded unnecessary models")
    require(initial["implementation"] == implementation_hashes(), "Implementation changed since actual batch")
    source_matches, training_matches, sample_ids = set(), set(), set()
    frames, learned, available, student_count = 0, 0, 0, 0
    for job in resumed["jobs"]:
        folder = Path(job["output"])
        require(completed_matches(folder, job["fingerprint"]), f"Changed batch output: {job['id']}")
        spec = read(folder / "resolved_job.json")
        bundle = read(folder / "replay.json")
        source_matches.add(str(spec["match"]))
        require(not bundle["truth_overlay_default"] and not bundle["exo_reference_default"], "Reference overlays enabled by default")
        sequence = bundle["sequences"][0]
        for rows in sequence["actors"].values():
            previous, seen = -1., set()
            for row in rows:
                frames += 1
                require(row["timestamp_s"] > previous, "Nonchronological replay")
                previous = row["timestamp_s"]
                if any("DISCONTINUITY" in warning or "RESET" in warning for warning in row["belief"]["warnings"]):
                    seen.clear()
                for entity in row["belief"]["entities"]:
                    require(entity["id"] in seen or entity["visible"], "Named entity before first sight")
                    seen.add(entity["id"])
                    if entity["backend"] in ("G1", "G3", "G4"):
                        learned += 1
                        require(entity["age_s"] <= 2.0001 and row["belief"]["model_contract_met"],
                                "Learned estimate outside the two-second contract")
                require("exo_reference" not in row["belief"] and
                        (row["assessment"] is None or "exo_reference" not in row["assessment"]),
                        "Privileged reference entered player path")
                require(row["action"]["support_status"] == "unsupported"
                        and row["action"]["action_type"] is None,
                        "Unimplemented action branch must remain explicitly unsupported")
                require(row["ball_state"]["position_status"] == "unknown"
                        and row["ball_state"]["possession_status"] == "unknown"
                        and row["ball_state"]["possessor_track"] is None,
                        "Unknown ball state must not assume the selected actor has possession")
                if row["assessment"] is None:
                    require(row["modules"]["passing"]["status"] == "UNSUPPORTED",
                            "Disabled passing module must be explicitly unsupported")
                    continue
                require(row["assessment"]["source_coverage"] == "PARTIAL_OR_UNKNOWN", "Missing scene-coverage caveat")
                for candidate in row["assessment"]["candidates"]:
                    require(candidate["id"] in seen, "Never-seen candidate")
                    if candidate["status"] == "AVAILABLE":
                        available += 1
                        require(candidate["visible"], "Uncertain receiver endpoint silently assessed")
        training = folder / "training_sample"
        if spec["intended_use"] != "training_export":
            require(not training.exists(), "Replay/evaluation job produced training records")
            continue
        training_matches.add(str(spec["match"]))
        require(str(spec["match"]) in FIT_MATCHES, "Nonfitting match in training output")
        streams = [(training / f"{name}.jsonl").open(encoding="utf-8")
                   for name in ("student", "privileged_exo", "optional_targets")]
        try:
            for student_line, teacher_line, optional_line in zip(*streams, strict=True):
                student, teacher, optional = map(json.loads, (student_line, teacher_line, optional_line))
                identity = student["sample_id"]
                require(identity == teacher["sample_id"] == optional["sample_id"] and identity not in sample_ids,
                        "Duplicated/misaligned training record")
                sample_ids.add(identity)
                require("privileged_exo" not in student and "exo_reference" not in student, "Privileged student record")
                features = json.dumps({key: student[key] for key in ("observation", "causal_memory", "world_input")})
                require("source_track" not in features and "track_" not in features, "Source identity used as learning feature")
                require(optional["teacher_scores"] is None and optional["event_target"] is None
                        and optional["receiver_proxy_target"] is None, "Unsupported target/OOF claim")
                require(optional["target_availability"]["action"] is False
                        and "not a negative" in optional["target_availability"]["meaning"],
                        "Missing action label must not be interpreted as a negative")
                student_count += 1
        finally:
            for stream in streams:
                stream.close()
    require(frames == 1000 and learned > 0 and available > 0 and student_count == 180, "Actual export shape mismatch")
    require(any(job["metrics"].get("unavailable_assessments", 0) == 0 and
                job["id"] == "retro118577_t60" for job in resumed["jobs"]),
            "The explicitly passing-disabled job must omit its passing candidate assessments")
    require({"118577", "128058"} <= source_matches and training_matches == {"117092", "117093"}, "Wrong dataset split")
    roundtrip = read(output / "roundtrip_validation.json")
    require(roundtrip["status"] == "PASSED" and roundtrip["frames"] == 100, "Real round trip missing")
    pilot = read(output / "video_pilot" / "manifest.json")
    require(pilot["pixel_observations"]["frames"] == 12 and pilot["pixel_observations"]["detections"] > 0,
            "No genuine image detections")
    require(pilot["camera_to_pitch_projection"]["status"] == "BLOCKED"
            and pilot["backbone_rejection"]["status"] == "PASSED", "Unsupported pixel projection accepted")
    malformed = read(output / "malformed_batch" / "batch_summary.json")
    require(malformed["status"] == "FAILED" and malformed["counts"].get("FAILED", 0) > 0,
            "Malformed job falsely marked successful")
    browser = read(output / "browser_validation.json")
    require(browser["status"] == "PASSED", "Browser acceptance not verified")
    elapsed = (datetime.datetime.now(datetime.timezone.utc)
               - datetime.datetime.fromisoformat("2026-10-20T19:58:13.774+00:00")).total_seconds()
    result = {
        "status": "PASSED", "tests": tests, "browser": browser,
        "protected_artifacts_unchanged": len(protected), "research_status": "PAUSED_FOR_INTEGRATION",
        "replay_frames": frames, "learned_hidden_estimates": learned, "available_assessments": available,
        "training_records": student_count, "training_matches": sorted(training_matches),
        "batch_execution_seconds": initial["execution_seconds"], "resume_seconds": resumed["execution_seconds"],
        "video_detection_seconds": pilot["execution_seconds"],
        "wall_seconds_since_assignment": elapsed, "no_source_validity_claim": True,
    }
    write_json(output / "acceptance.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.output.resolve())


if __name__ == "__main__":
    main()
