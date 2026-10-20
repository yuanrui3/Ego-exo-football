"""One command exports checkpoint-backed player replays and a self-contained viewer."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import torch

from src.backbone.export import export_replays, export_training_sample
from src.backbone.models import ModelSession
from src.trajectory_dataset import atomic_replace, file_digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def run(config_path: Path, output: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != "backbone.0.1":
        raise ValueError("Unsupported backbone configuration schema")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_manifest.json").exists():
        raise FileExistsError("Use a new output directory; completed replay exports are immutable")
    experiment = ROOT / config["experiment"]
    paused_path = experiment / "progress_summary.json"
    paused_hash = file_digest(paused_path)
    if json.loads(paused_path.read_text())["status"] != "PAUSED_FOR_INTEGRATION":
        raise ValueError("The research queue must remain paused")
    torch.set_num_threads(4)
    models = ModelSession(experiment, config["backend"], config["exo_p0"], config.get("device", "cpu"))
    bundle = export_replays(ROOT, output, config, models)
    export_training_sample(ROOT, output, models, config["training_sample"], config["training_sample"]["count"])
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    # Embedded data is inert JSON; escape HTML/script delimiters for offline opening.
    payload = json.dumps(bundle, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = template.replace("__REPLAY_JSON__", payload)
    temporary = output / "viewer.tmp.html"
    temporary.write_text(html, encoding="utf-8")
    atomic_replace(temporary, output / "viewer.html")
    if file_digest(paused_path) != paused_hash:
        raise RuntimeError("Research progress changed during integration")
    source = {str(path.relative_to(ROOT)): file_digest(path)
              for path in Path(__file__).parent.glob("*") if path.is_file()}
    write_json(output / "resolved_config.json", config)
    write_json(output / "run_manifest.json", {
        "status": "EXPORTED", "schema_version": config["schema_version"],
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "configuration": {"path": str(config_path), "sha256": file_digest(config_path)},
        "source_hashes": source, "model_provenance": models.provenance,
        "research_queue_unchanged_sha256": paused_hash, "no_training": True,
        "files": {name: file_digest(output / name) for name in (
            "replay.json", "viewer.html", "replay_validation.json", "training_sample/student.jsonl",
            "training_sample/privileged_exo.jsonl", "training_sample/optional_targets.jsonl")},
    })
    print(f"Replay: {output / 'viewer.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "backbone_v0_1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / (
        "backbone_v0_1_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--only", help="One job from a v0.2 batch configuration")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") == "backbone.batch.0.2":
        from src.backbone.batch import run_batch
        if not args.only:
            parser.error("Single external replay requires --only JOB; use src.backbone.batch for all jobs")
        result = run_batch(args.config.resolve(), args.output.resolve(), args.resume, only=args.only)
        if result["status"] != "COMPLETE":
            raise SystemExit(1)
        return
    if args.only or args.resume:
        parser.error("--only/--resume require a v0.2 batch configuration; v0.1 exports remain immutable")
    run(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
