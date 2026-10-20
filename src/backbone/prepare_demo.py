"""Export only the declared local annotation-backed public-format fixtures."""

import argparse
import json
from pathlib import Path

from src.backbone.external import ExternalObservationSource, export_public
from src.backbone.source import SoccerTrackSource
from src.trajectory_dataset import file_digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protection = args.output / "protected_before.json"
    if not protection.exists():
        experiment = ROOT / "outputs" / "dynamic_gnn_tactical_20261019"
        names = set(json.loads((experiment / "provenance.json").read_text())["sha256"])
        names.update(str(path.relative_to(ROOT)) for path in experiment.rglob("*") if path.is_file())
        old = ROOT / "outputs" / "backbone_v0_1_20261020_demo"
        names.update(str(path.relative_to(ROOT)) for path in old.rglob("*") if path.is_file())
        write_json(protection, {name: file_digest(ROOT / name) for name in sorted(names)})
    for spec in config["jobs"]:
        job = {**config.get("defaults", {}), **spec}
        if job["source_adapter"] != "external_jsonl":
            continue
        path = ROOT / job["input_path"]
        if path.exists():
            source = ExternalObservationSource(path)
            if source.match != str(job["match"]) or source.half != job["half"]:
                raise ValueError(f"Existing fixture mismatch: {path}")
            list(source.frames(job["start_frame"], job["samples"]))
        else:
            export_public(SoccerTrackSource(ROOT, str(job["match"]), job["half"]),
                          path, job["start_frame"], job["samples"])
        print(path)
    write_json(args.output / "fixture_provenance.json", {
        "label": "Annotation-backed interoperability only, NOT a new dataset or image-derived positions",
        "config_sha256": file_digest(args.config), "no_training": True,
        "selection": config["defaults"]["selection"],
    })


if __name__ == "__main__":
    main()
