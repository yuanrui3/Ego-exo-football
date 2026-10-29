"""Run release regressions and produce a fresh mixed-action baseline replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from src.backbone.release_baseline import ROOT, run
from src.trajectory_dataset import file_digest, write_json


TESTS = (
    "tests/test_backbone.py",
    "tests/test_backbone_ingestion.py",
    "tests/test_backbone_v0_3.py",
    "tests/test_backbone_v0_4.py",
    "tests/test_backbone_v05.py",
    "tests/test_backbone_v06_robust_action.py",
    "tests/test_backbone_v06_video.py",
    "tests/test_backbone_transfer_completion.py",
    "tests/test_backbone_streaming_completion.py",
    "tests/test_backbone_release.py",
)

SOURCE_BUNDLE_FILES = (
    "src/backbone/__init__.py",
    "src/backbone/actions.py",
    "src/backbone/assessment.py",
    "src/backbone/contracts.py",
    "src/backbone/external.py",
    "src/backbone/memory.py",
    "src/backbone/models.py",
    "src/backbone/observation.py",
    "src/backbone/pipeline.py",
    "src/backbone/release_baseline.py",
    "src/backbone/source.py",
    "src/backbone/verify_release_baseline.py",
    "src/belief_observation.py",
    "src/extract_soccertrack_events.py",
    "src/hidden_defender_geometry.py",
    "src/models/__init__.py",
    "src/models/belief_world_model.py",
    "src/models/candidate_ranker_receiver.py",
    "src/models/dynamic_football_gnn.py",
    "src/models/entity_transformer_receiver.py",
    "src/models/partial_observation_student.py",
    "src/player_centric_dataset.py",
    "src/player_centric_visibility.py",
    "src/recode_primary_actions.py",
    "src/trajectory_dataset.py",
)


def verify_source_only_imports() -> dict:
    with tempfile.TemporaryDirectory(prefix="football-backbone-source-check-") as temporary:
        bundle = Path(temporary)
        for relative in SOURCE_BUNDLE_FILES:
            target = bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        (bundle / "configs").mkdir()
        shutil.copy2(ROOT / "configs" / "backbone_release_baseline.json",
                     bundle / "configs" / "backbone_release_baseline.json")
        script = (
            "import importlib, json, pathlib, sys; "
            "root=pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root)); "
            "names=['src.backbone.release_baseline','src.backbone.verify_release_baseline',"
            "'src.backbone.models','src.backbone.actions','src.backbone.external','src.backbone.source',"
            "'src.backbone.observation','src.backbone.memory','src.backbone.pipeline']; "
            "mods=[importlib.import_module(name) for name in names]; "
            "paths=sorted({str(pathlib.Path(mod.__file__).resolve().relative_to(root)) "
            "for mod in sys.modules.values() if getattr(mod,'__file__',None) and "
            "pathlib.Path(mod.__file__).resolve().is_relative_to(root / 'src')}); "
            "assert all((root / path).is_file() for path in paths), paths; "
            "print(json.dumps({'status':'PASS','modules':names,'all_from_source_copy':True,"
            "'source_files':paths}))"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script, str(bundle)],
            cwd=bundle, capture_output=True, text=True, check=True,
        )
        return json.loads(completed.stdout)


def verify(config: Path, output: Path) -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Verification outputs are immutable; choose a new empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", *TESTS, "-q"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    test_output = output / "pytest_output.txt"
    test_output.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)
    source_only = verify_source_only_imports()
    replay = run(config, output / "replay")
    result = {
        "status": "PASS",
        "tests": list(TESTS),
        "test_output_sha256": file_digest(test_output),
        "test_summary": completed.stdout.strip().splitlines()[-1],
        "source_only_imports": source_only,
        "mixed_action_replay": replay,
        "browser_smoke": "HTML produced; open replay/viewer.html and verify sequence, actor, event labels and Play/Pause controls in a browser.",
    }
    write_json(output / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "backbone_release_baseline.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(verify(args.config.resolve(), args.output))


if __name__ == "__main__":
    main()
