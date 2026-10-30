"""Disposable weights-only fitting-data AMP stability check, never continuation.

The epoch-based schedule advances even when AMP skips, matching installed
Ultralytics. Warmup and accumulation use minibatch indices, not commit counts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import warnings
import zipfile

import numpy as np
import torch

from src.backbone.amp_guard import AmpGuard, AmpSkipLimit

ROOT = Path(__file__).resolve().parents[2]
SELECTED_SHA = "3dc49dd000e00d66c3c48b3882c31dd57eee50896d2d5c95ef03b141c88fb28b"
FILES = ["src/backbone/amp_guard.py", "src/backbone/gpu_perception.py",
         "src/backbone/detector_stability.py", "tests/test_amp_guard.py"]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def protected(source: Path) -> dict[str, str]:
    release = json.loads((ROOT / "configs/backbone_release_manifest.json").read_text())
    paths = [source / name for name in (
        "selected.pt", "recipe.json", "resolved_trainer.json", "training_summary.json",
        "data_manifest.json", "historical_data_audit.json", "data_v1/fit.txt",
        "data_v1/dev.txt", "data_v1/dataset.yaml")]
    paths += [ROOT / "configs/backbone_release_baseline.json"]
    paths += [ROOT / row["checkpoint"] for key, row in release["models"].items()
              if key in ("default_world", "default_action")]
    recipe = json.loads((source / "recipe.json").read_text())
    paths.append(ROOT / recipe["initialization"])
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def fit_worker(args: argparse.Namespace) -> dict:
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import ModelEMA
    from src.backbone.gpu_perception import enable_detector_gradients

    output, source = args.output, args.source_run
    started = time.monotonic()
    stop_at = started + args.max_seconds
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED; no CPU fallback")
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    resolved = json.loads((source / "resolved_trainer.json").read_text())["args"]
    recipe = json.loads((source / "recipe.json").read_text())
    manifest = json.loads((source / "data_manifest.json").read_text())
    if digest(source / "selected.pt") != SELECTED_SHA:
        raise RuntimeError("Protected selected checkpoint hash mismatch")
    fit_list = source / "data_v1/fit.txt"
    if digest(fit_list) != manifest["fit_loader"]["list_sha256"]:
        raise RuntimeError("Audited fitting list changed")
    paths = fit_list.read_text().splitlines()
    for path in paths:
        p = Path(path).resolve()
        if not any(p.is_relative_to(source / f"data_v1/images/{match}_fit")
                   for match in ("117093", "117092")):
            raise RuntimeError(f"Non-fitting image in audited loader: {p}")
    config = get_cfg(overrides={key: value for key, value in resolved.items() if key != "save_dir"})
    config.cache = False
    config.workers = 0
    config.project = str(output)
    config.name = "disposable"
    config.save_dir = str(output)
    config.data = str(output / "fitting_only.yaml")
    (output / "fitting_only.yaml").write_text(
        f"train: {fit_list.as_posix()}\nnames:\n  0: player_footpoint\n", encoding="utf-8")
    write(output / "resolved_stability.json", {
        "args": vars(config), "status": "DISPOSABLE WEIGHTS-ONLY STABILITY RUN",
        "optimizer_scaler_scheduler": "Fresh; not exact continuation",
        "accumulation": "Native nbs/batch warmup; helper runs only after accumulation",
        "scheduler": "Epoch schedule including skipped updates; warmup uses minibatches",
        "original_recipe": recipe, "limits": {"attempts": args.max_attempts, "seconds": args.max_seconds},
        "augmentations": {key: resolved[key] for key in (
            "hsv_h", "hsv_s", "hsv_v", "translate", "scale", "fliplr", "mosaic",
            "mixup", "degrees", "shear", "perspective", "flipud", "bgr")},
    })
    # Native loader scans only fitting images; all new label caches are redirected.
    class LocalCacheDataset(YOLODataset):
        def _load_or_scan_cache(self, cache_path, cache_hash):
            return super()._load_or_scan_cache(output / "fitting_labels.cache", cache_hash)

    dataset = LocalCacheDataset(
        img_path=str(fit_list), imgsz=config.imgsz, batch_size=config.batch,
        augment=True, hyp=config, rect=False, cache=False, stride=32,
        data={"names": {0: "player_footpoint"}, "nc": 1, "channels": 3},
    )
    if len(dataset) != manifest["fit_loader"]["images"] or sum(len(row["cls"]) for row in dataset.labels) != manifest["fit_loader"]["labels"]:
        raise RuntimeError("Audited fitting population does not match actual loader")
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=config.batch, shuffle=True, num_workers=0,
                        collate_fn=dataset.collate_fn, generator=generator, pin_memory=True)
    model = YOLO(str(source / "selected.pt")).model.float().to("cuda:0").train()
    model.args = config
    enable_detector_gradients(model)
    trainer = SimpleNamespace(args=config, device=torch.device("cuda:0"), stride=32)
    optimizer = BaseTrainer.build_optimizer(trainer, model, name="AdamW", lr=config.lr0,
                                           momentum=config.momentum, decay=config.weight_decay)
    lf = lambda epoch: max(1 - epoch / config.epochs, 0) * (1 - config.lrf) + config.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = torch.amp.GradScaler("cuda", init_scale=recipe["amp_initial_scale"], enabled=True)
    ema = ModelEMA(model)
    pending_samples: list[str] = []
    def sink(event):
        with (output / "step_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        if event.get("first_abnormal"):
            write(output / "first_abnormal.json", event)
            torch.save({
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "loader_generator": generator.get_state(),
                "gradients": {name: p.grad.detach().cpu() for name, p in model.named_parameters()
                              if name in event.get("affected_parameters", []) and p.grad is not None},
            }, output / "first_abnormal_local.pt")
    guard = AmpGuard(event_sink=sink)
    checkpoint_path = output / "last_finite.pt"
    def checkpoint():
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError("Refusing to overwrite finite checkpoint with invalid parameters")
        temporary = output / "last_finite.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
                    "guard": guard.state_dict(), "ema": ema.ema.state_dict(), "ema_updates": ema.updates,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "loader_generator": generator.get_state(),
                    "sampler_limitation": "No exact sampler cursor; disposable weights-only run"},
                   temporary)
        temporary.replace(checkpoint_path)
    checkpoint()
    stop_reason, epoch, ni, last_boundary = "", 0, 0, -1
    warmup = max(round(config.warmup_epochs * len(loader)), 100)
    optimizer.zero_grad(set_to_none=True)
    try:
        while not stop_reason:
            if time.monotonic() >= stop_at:
                stop_reason = "MAX_SECONDS"
                break
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step")
                scheduler.step()
            for batch in loader:
                if guard.counters.attempts >= args.max_attempts:
                    stop_reason = "MAX_ATTEMPTS"
                    break
                if time.monotonic() >= stop_at:
                    stop_reason = "MAX_SECONDS"
                    break
                samples = list(batch["im_file"])
                pending_samples.extend(samples)
                accumulate = max(1, round(config.nbs / config.batch))
                if ni < warmup:
                    accumulate = max(1, int(np.interp(ni, [0, warmup], [1, config.nbs/config.batch]).round()))
                    for group in optimizer.param_groups:
                        group["lr"] = float(np.interp(ni, [0, warmup], [
                            config.warmup_bias_lr if group.get("param_group") == "bias" else 0.,
                            group["initial_lr"] * lf(epoch)]))
                context = {"fitting_samples": pending_samples.copy(), "epoch": epoch,
                           "minibatch_index": ni, "accumulate": accumulate,
                           "augmentation": "Native augmented YOLODataset; resolved_stability.json",
                           "rng": "Seed42; first abnormal RNG/gradients retained locally",
                           "input_shape": list(batch["img"].shape)}
                with torch.autocast("cuda", dtype=torch.float16):
                    batch = DetectionTrainer.preprocess_batch(trainer, batch)
                    loss, _ = model(batch)
                    loss = loss.sum()
                guard.note_minibatch(loss, context, scaler=scaler)
                scaler.scale(loss).backward()
                if time.monotonic() >= stop_at:
                    stop_reason = "MAX_SECONDS"
                    optimizer.zero_grad(set_to_none=True)
                    break
                if ni - last_boundary >= accumulate:
                    event = guard.update(model, optimizer, scaler, loss, ema=ema, context=context)
                    last_boundary = ni
                    pending_samples.clear()
                    if event["status"] == "COMMITTED" and guard.counters.committed % 100 == 0:
                        checkpoint()
                    if guard.counters.attempts % 25 == 0:
                        write(output / "progress.json", {"counters": asdict(guard.counters),
                              "elapsed_seconds": time.monotonic()-started, "scale": scaler.get_scale()})
                    if guard.counters.attempts >= args.max_attempts:
                        stop_reason = "MAX_ATTEMPTS"
                        break
                ni += 1
            epoch += 1
    except AmpSkipLimit:
        stop_reason = "AMP_SKIP_LIMIT"
    except Exception:
        (output / "worker_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write(output / "worker_summary.json", {
            "status": "FAILED", "stop_reason": "HARD_FAILURE", "counters": asdict(guard.counters),
            "elapsed_seconds": time.monotonic()-started,
            "first_abnormal": guard.first_abnormal, "error": traceback.format_exc(),
        })
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
    checkpoint()
    result = {
        "status": "PASS_BOUNDED_STABILITY" if stop_reason in ("MAX_ATTEMPTS", "MAX_SECONDS") and guard.counters.committed > 0 else "STOPPED_SKIP_LIMIT",
        "label": "DISPOSABLE WEIGHTS-ONLY STABILITY RUN", "stop_reason": stop_reason,
        "counters": asdict(guard.counters), "elapsed_seconds": time.monotonic()-started,
        "scale": scaler.get_scale(), "first_abnormal": guard.first_abnormal,
        "historical_failure_reproduced": False,
        "historical_failure_note": "No causal reproduction asserted; AMP skip events, if any, are not proof of the old cause.",
        "checkpoint": "last_finite.pt (local only; excluded from return ZIP)",
        "evaluation": "Finite-state and exact boundary accounting only; no retrospective evaluation",
    }
    write(output / "worker_summary.json", result)
    return result


def package(output: Path) -> None:
    patch = []
    for name in FILES:
        path = ROOT / name
        destination = output / "changed" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        # These experimental files are untracked in the published baseline.
        patch.extend(difflib.unified_diff([], path.read_text().splitlines(keepends=True),
                                         fromfile="/dev/null", tofile="b/" + name))
    (output / "amp_fix.patch").write_text("".join(patch), encoding="utf-8")
    allowed = {".json", ".jsonl", ".txt", ".xml", ".md", ".patch", ".py"}
    with zipfile.ZipFile(output / "return_amp_fix.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix in allowed and path.stat().st_size < 8_000_000:
                archive.write(path, path.relative_to(output))
    with zipfile.ZipFile(output / "return_amp_fix.zip") as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Corrupt return archive: {bad}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--max-seconds", type=int, default=2700)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source_run = args.source_run.resolve()
    args.output = args.output.resolve()
    if not 1 <= args.max_attempts <= 1000 or not 1 <= args.max_seconds <= 2700:
        parser.error("Diagnostic caps are 1000 attempts and 2700 seconds")
    if args.worker:
        fit_worker(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output
    started = datetime.now(timezone.utc)
    write(output / "deadline.json", {"started_utc": started.isoformat(),
          "hard_deadline_utc": (started+timedelta(hours=2)).isoformat(),
          "verification_reserve_utc": (started+timedelta(hours=1, minutes=45)).isoformat(),
          "first_inspection_utc": "2026-10-29T17:43:52.590984+00:00",
          "assignment_hard_deadline_utc": "2026-10-29T19:43:52.590984+00:00"})
    before = protected(args.source_run)
    write(output / "protected_before.json", before)
    if digest(args.source_run / "selected.pt") != SELECTED_SHA:
        raise RuntimeError("Selected hash differs from explicit protected hash")
    from ultralytics.engine.trainer import BaseTrainer
    import ultralytics
    parent = inspect.getsource(BaseTrainer.optimizer_step)
    parent_path = Path(inspect.getsourcefile(BaseTrainer))
    scaler_path = Path(inspect.getsourcefile(torch.amp.GradScaler))
    write(output / "environment.json", {
        "interpreter": sys.executable, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parent_path": str(parent_path), "parent_sha256": digest(parent_path),
        "scaler_path": str(scaler_path), "scaler_sha256": digest(scaler_path),
        "parent_optimizer_step": parent,
        "scaler_step": inspect.getsource(torch.amp.GradScaler.step),
        "scaler_unscale": inspect.getsource(torch.amp.GradScaler.unscale_),
    })
    summary = {"guard_corrected": True, "controlled_overflow_passed": False,
               "real_stability_passed": False, "historical_failure_reproduced": False}
    try:
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA",
            "tests/test_amp_guard.py", "tests/test_gpu_perception.py",
            "--junitxml=" + str(output / "focused_tests.xml")],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        (output / "focused_tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
        if tests.returncode:
            raise RuntimeError("Focused tests failed; detector not launched")
        import xml.etree.ElementTree as ET
        suites = ET.parse(output / "focused_tests.xml").getroot()
        summary["tests"] = {key: sum(int(s.get(key, 0)) for s in suites.iter("testsuite"))
                            for key in ("tests", "failures", "errors", "skipped")}
        overflow = [case for case in suites.iter("testcase") if case.get("name", "").startswith("test_real_cuda_overflow")]
        summary["controlled_overflow_passed"] = bool(overflow) and all(len(list(case)) == 0 for case in overflow)
        if not summary["controlled_overflow_passed"]:
            raise RuntimeError("Real CUDA overflow test did not pass; detector not launched")
        with (output / "worker_stdout.log").open("w", encoding="utf-8") as stdout, (
                output / "worker_stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run([
                sys.executable, "-m", "src.backbone.detector_stability", "--worker",
                "--source-run", str(args.source_run), "--output", str(output),
                "--max-attempts", str(args.max_attempts), "--max-seconds", str(args.max_seconds)],
                cwd=ROOT, stdout=stdout, stderr=stderr, timeout=args.max_seconds + 120)
        if completed.returncode:
            raise RuntimeError(f"Detector worker failed with exit {completed.returncode}; see worker_error.txt/stderr")
        worker = json.loads((output / "worker_summary.json").read_text())
        summary.update(worker=worker, real_stability_passed=worker["status"] == "PASS_BOUNDED_STABILITY")
    except Exception as exc:
        summary.update(blocker=str(exc), error=traceback.format_exc())
        if (output / "worker_summary.json").is_file():
            summary["worker"] = json.loads((output / "worker_summary.json").read_text())
        (output / "complete_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        after = protected(args.source_run)
        write(output / "protected_after.json", after)
        summary["protected_hashes_unchanged"] = before == after
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(output / "summary.json", summary)
        text = (
            "# Bounded AMP guard verification\n\n"
            f"Guard corrected: {summary['guard_corrected']}. Controlled CUDA overflow: {summary['controlled_overflow_passed']}. "
            f"Fitting stability passed: {summary['real_stability_passed']}. Historical failure reproduced: False.\n\n"
            f"Worker stop: {summary.get('worker', {}).get('stop_reason', summary.get('blocker'))}. "
            f"Counters: {summary.get('worker', {}).get('counters', {})}.\n\n"
            "Only fitting tiles and native training augmentations were used. Optimizer, scaler and scheduler were fresh. "
            "The helper unscales once after accumulation, skips invalid AMP updates without clipping or EMA update, and fails hard "
            "on nonfinite loss/parameters, FP32 gradients, missing gradients, clipping norm or device errors. "
            "Epoch scheduling and minibatch warmup advance independently of commits. Skip limits are configurable engineering limits. "
            "No overflow is evidence of stability only, not a causal explanation of the historical epoch16 failure. "
            "Local checkpoint and abnormal tensors are excluded from return_amp_fix.zip. "
            f"Protected hashes unchanged: {before == after}. No release changes, commits or pushes.\n"
        )
        (output / "handoff.md").write_text(text, encoding="utf-8")
        package(output)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary.get("real_stability_passed") and before == after else 2


if __name__ == "__main__":
    raise SystemExit(main())
