"""Report source, Python, data and checkpoint readiness for backbone v0.5."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path

from src.trajectory_dataset import file_digest


ROOT = Path(__file__).resolve().parents[2]
ENTRY_MODULES = (
    "src.backbone.corpus",
    "src.backbone.train_action",
    "src.backbone.batch",
    "src.backbone.video_pilot",
    "src.backbone.video_tracklets",
    "src.backbone.video_ingest",
)
OPTIONAL_VIDEO_PACKAGES = ("ultralytics",)


def _module_path(root: Path, module: str) -> Path:
    return root / Path(*module.split(".")).with_suffix(".py")


def _local_source_dependencies(root: Path, entry_modules: tuple[str, ...]) -> dict:
    pending = list(entry_modules)
    visited: set[str] = set()
    missing = []
    resolved = []
    while pending:
        module = pending.pop()
        if module in visited or not module.startswith("src."):
            continue
        visited.add(module)
        path = _module_path(root, module)
        if not path.is_file():
            package_init = root / Path(*module.split(".")) / "__init__.py"
            if package_init.is_file():
                path = package_init
            else:
                missing.append({"module": module, "expected_path": str(path)})
                continue
        resolved.append(str(path.relative_to(root)))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names if alias.name.startswith("src."))
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module and node.module.startswith("src."):
                    pending.append(node.module)
    return {
        "entry_modules": list(entry_modules),
        "resolved_source_files": sorted(set(resolved)),
        "missing_source_modules": missing,
        "status": "PASS" if not missing else "FAIL",
    }


def _checkpoint_record(root: Path, experiment_path: Path, key: str) -> dict:
    manifest_path = experiment_path / "integration_manifest.json"
    if not manifest_path.is_file():
        return {"status": "BLOCKED", "reason": "integration_manifest_missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("selected_checkpoints", {}).get(key)
    if not isinstance(record, dict):
        return {"status": "BLOCKED", "reason": f"selected_checkpoint_missing:{key}"}
    checkpoint = experiment_path / record["path"]
    config = checkpoint.parent / "config.json"
    result = checkpoint.parent / "result.json"
    if not checkpoint.is_file() or not config.is_file() or not result.is_file():
        return {
            "status": "BLOCKED",
            "reason": "checkpoint_or_provenance_missing",
            "checkpoint_relative_path": str(checkpoint.relative_to(root)),
        }
    digest = file_digest(checkpoint)
    if digest != record.get("sha256"):
        return {"status": "FAIL", "reason": "checkpoint_hash_mismatch"}
    config_value = json.loads(config.read_text(encoding="utf-8"))
    return {
        "status": "PASS",
        "checkpoint_relative_path": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": digest,
        "config_sha256": file_digest(config),
        "fitting_matches": config_value.get("fitting_matches", record.get("fitting_matches")),
        "development_matches": config_value.get("development_matches", []),
        "input_contract": config_value.get("config", {}),
    }


def preflight(config_path: Path, *, root: Path = ROOT) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "football.backbone-integration.0.5":
        raise ValueError("Unsupported v0.5 preflight configuration")
    declared_root = Path(config["source_root"])
    repo = (root / declared_root).resolve() if not declared_root.is_absolute() else declared_root.resolve()
    source_report = _local_source_dependencies(repo, ENTRY_MODULES)
    package_report = {}
    for package in ("numpy", "torch", "ijson", "cv2"):
        package_report[package] = importlib.util.find_spec(package) is not None
    optional_packages = {
        package: importlib.util.find_spec(package) is not None
        for package in OPTIONAL_VIDEO_PACKAGES
    }

    paths = config["source_paths"]
    event_root = repo / paths["event_data"]
    cache_root = repo / paths["tracking_cache"]
    association_root = repo / paths["associations"]
    split_matches = sorted({
        str(match) for values in config["split_manifest"].values() for match in values
    })
    sources = []
    missing_sources = []
    for match in split_matches:
        bas_path = event_root / f"{match}_12_class_events.json"
        for half in ("1st", "2nd"):
            cache = cache_root / f"{match}_{half}.npz"
            metadata = cache.with_suffix(".json")
            association = association_root / f"associations_{match}_{half}.json"
            record = {
                "match": match,
                "half": half,
                "bas": bas_path.is_file(),
                "cache": cache.is_file(),
                "cache_metadata": metadata.is_file(),
                "association": association.is_file(),
            }
            record["status"] = "PASS" if all(record[key] for key in ("bas", "cache", "cache_metadata", "association")) else "BLOCKED"
            if record["status"] != "PASS":
                missing_sources.append(record)
            sources.append(record)

    experiment = repo / paths["belief_experiment"]
    g1 = _checkpoint_record(repo, experiment, "world_baseline")
    s4_path = repo / paths["s4_initialization"]
    s4 = {"status": "BLOCKED", "reason": "initialization_checkpoint_missing"}
    if s4_path.is_file() and s4_path.with_name("config.json").is_file():
        s4_config = json.loads(s4_path.with_name("config.json").read_text(encoding="utf-8"))
        digest = file_digest(s4_path)
        declared_hash = config["training"]["initialization_sha256"]
        match_split = sorted(s4_config.get("fitting_matches", []))
        expected_fits = sorted(map(str, config["split_manifest"]["fitting"]))
        s4 = {
            "status": "PASS" if digest == declared_hash and match_split == expected_fits else "FAIL",
            "checkpoint_relative_path": str(s4_path.relative_to(repo)),
            "checkpoint_sha256": digest,
            "declared_sha256": declared_hash,
            "fitting_matches": match_split,
            "development_matches": s4_config.get("development_matches", []),
            "variant": s4_config.get("variant"),
            "fov_index": s4_config.get("fov_index"),
        }

    video_root = repo / paths["video_root"]
    video_assets = sorted(str(path.relative_to(repo)) for path in video_root.rglob("*.mp4")) if video_root.is_dir() else []
    video_weights = repo / paths["video_weights"]
    required_ok = (
        source_report["status"] == "PASS"
        and all(package_report.values())
        and not missing_sources
        and g1["status"] == "PASS"
        and s4["status"] == "PASS"
    )
    video_ok = bool(video_assets) and video_weights.is_file() and all(optional_packages.values())
    return {
        "schema_version": "football.backbone-preflight.0.5",
        "status": "PASS" if required_ok else "BLOCKED",
        "repository_root": str(repo),
        "source_tree": source_report,
        "required_python_packages": package_report,
        "optional_video_packages": optional_packages,
        "configured_sources": sources,
        "missing_required_sources": missing_sources,
        "checkpoints": {"G1": g1, "S4_initialization": s4},
        "video_prerequisites": {
            "status": "PASS" if video_ok else "BLOCKED",
            "available_video_count": len(video_assets),
            "video_assets": video_assets,
            "detector_weights_present": video_weights.is_file(),
            "optional_packages": optional_packages,
        },
        "checkpoint_contracts": {
            "G1": "selected completed two-second world_baseline from the configured experiment manifest",
            "S4": "S4/120-degree, width 96, 2 layers, 4 heads, 11x22x10 input; 8s input age converted to /5s checkpoint age",
            "separation": "G1 predicted hidden positions are not S4 action-model inputs",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/backbone_v0_5.json"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.root / args.config
    result = preflight(config_path.resolve(), root=args.root.resolve())
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else args.root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
