"""Deadline-bounded detector, fixed-detection association and frozen replay worker.

Run/resume advances completed stages, never launches a second fit in the same root.
All association choices are frozen on development before retrospective scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import difflib
import inspect
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import cv2
import numpy as np
import torch

from src.backbone.amp_guard import AmpGuard
from src.backbone.detector_stability import protected, SELECTED_SHA
from src.backbone.gpu_perception import (
    ROOT, DEFAULT_OUTPUT, HISTORICAL, aggregate_scores, development_scores,
    predict_video, score_predictions, source_records, require_cuda,
)
from src.backbone.gpu_perception_post import associate, render_diagnostics
from src.backbone.video_player_cross_match import (
    _read_annotations, _read_frame, load_protocol,
)
from src.trajectory_dataset import file_digest, write_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def frames(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def journal(output: Path, **values) -> dict:
    state = read(output / "task_state.json")
    state.update(values)
    state["updated_utc"] = now()
    state["remaining_seconds"] = max(0., (
        datetime.fromisoformat(state["hard_deadline_utc"]) - datetime.now(timezone.utc)).total_seconds())
    write_json(output / "task_state.json", state)
    return state


def before_reserve(output: Path) -> bool:
    return datetime.now(timezone.utc) < datetime.fromisoformat(read(output / "task_state.json")["verification_reserve_utc"])


class FitBudgetReached(RuntimeError):
    pass


def train_candidate(output: Path) -> dict:
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils.torch_utils import unwrap_model

    require_cuda()
    if (output / "training_started.json").exists():
        raise RuntimeError("One fit already started; no automatic second fit")
    if file_digest(DEFAULT_OUTPUT / "selected.pt") != SELECTED_SHA:
        raise ValueError("Protected selected checkpoint changed")
    write_json(output / "training_started.json", {"started_utc": now(), "pid": __import__("os").getpid()})
    shutil.copy2(DEFAULT_OUTPUT / "data_manifest.json", output / "data_manifest.json")
    baseline = development_scores(output, DEFAULT_OUTPUT / "selected.pt", save_name="preserved_development.json")
    raw = torch.load(DEFAULT_OUTPUT / "selected.pt", map_location="cpu", weights_only=False)
    runtime = torch.load(DEFAULT_OUTPUT / "last_runtime_state.pt", map_location="cpu", weights_only=False)
    resolved = read(DEFAULT_OUTPUT / "resolved_trainer.json")["args"]
    start_epoch = raw["epoch"] + 1
    native_epochs = start_epoch + 20
    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(
        f"train: {(DEFAULT_OUTPUT / 'data_v1/fit.txt').as_posix()}\n"
        f"val: {(DEFAULT_OUTPUT / 'data_v1/dev.txt').as_posix()}\n"
        "names:\n  0: player_footpoint\n", encoding="utf-8")
    criterion = read(output / "task_state.json")["acceptance"]["detector_selection"]
    training = {
        "status": "RUNNING", "initialization_sha256": SELECTED_SHA,
        "initialization": str((DEFAULT_OUTPUT / "selected.pt").relative_to(ROOT)),
        "start_epoch": start_epoch, "maximum_additional_epochs": 20, "maximum_seconds": 10800,
        "selection_rule": criterion, "epochs": [], "selected": str((DEFAULT_OUTPUT / "selected.pt").relative_to(ROOT)),
        "selected_sha256": SELECTED_SHA, "selected_mean_f1": baseline["mean_per_match_f1"],
        "candidate_best": None, "epochs_without_gain": 0, "optimizer_steps": 0,
        "restoration": "Native EMA weights plus compatible optimizer/scaler/scheduler state; NOT exact continuation: original live model and sampler cursor unavailable.",
        "precision": "CUDA FP16 AMP with shared guard; epoch scheduler independent of skips",
    }
    write_json(output / "training_summary.json", training)
    started = time.monotonic()
    last_context = {}
    def log_event(event):
        nonlocal last_context
        last_context = event
        with (output / "step_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        if event.get("first_abnormal"):
            write_json(output / "first_abnormal.json", event)
            torch.save({"python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
                       output / "first_abnormal_rng.pt")
    guard = AmpGuard(event_sink=log_event)
    def budget():
        if time.monotonic() - started >= 10800 or not before_reserve(output):
            raise FitBudgetReached("TRAINING_TIME_CAP")

    class GuardedTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            class CachedDataset(YOLODataset):
                def _load_or_scan_cache(self, cache_path, cache_hash):
                    return super()._load_or_scan_cache(output / f"{mode}_labels.cache", cache_hash)
            stride = max(int(unwrap_model(self.model).stride.max()), 32)
            return CachedDataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
                augment=mode == "train", hyp=self.args, rect=self.args.rect or mode == "val",
                cache=False, single_cls=False, stride=stride, pad=0 if mode == "train" else .5,
                prefix=f"{mode}: ", task=self.args.task, classes=self.args.classes,
                data=self.data, fraction=1.)

        def preprocess_batch(self, batch):
            budget()
            if self.batch_size != resolved["batch"]:
                raise RuntimeError("Automatic batch-size changes are prohibited")
            self.guard_samples = getattr(self, "guard_samples", []) + list(batch.get("im_file", []))
            guard.note_minibatch(torch.zeros((), device=self.device))
            return super().preprocess_batch(batch)

        def optimizer_step(self):
            budget()
            guard.update(self.model, self.optimizer, self.scaler, self.loss, ema=self.ema,
                context={"fitting_samples": self.guard_samples, "epoch": self.epoch,
                         "augmentation": "Resolved native YOLO training transforms",
                         "rng": "Local first_abnormal_rng.pt", "accumulate": self.accumulate})
            self.guard_samples = []
            training["optimizer_steps"] = guard.counters.committed
            training["numerics"] = guard.state_dict()
            if guard.counters.attempts % 100 == 0:
                write_json(output / "training_progress.json", training)

        def _handle_nan_recovery(self, epoch):
            if not torch.isfinite(self.loss).all():
                raise RuntimeError("Nonfinite loss; no native retry")
            return False

        def _setup_scheduler(self):
            # Keep the original 40-epoch LR horizon while limiting this run to20 extra epochs.
            self.lf = lambda epoch: max(1 - epoch / resolved["epochs"], 0) * (1-self.args.lrf) + self.args.lrf
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def started_callback(trainer):
        if trainer.device.type != "cuda" or type(trainer.optimizer).__name__ != "AdamW":
            raise RuntimeError("Resolved CUDA AdamW contract violated")
        if runtime["epoch"] != raw["epoch"]:
            raise RuntimeError("Optimizer/scheduler checkpoint epochs disagree")
        trainer._load_checkpoint_state(copy.deepcopy(raw))
        trainer.start_epoch = start_epoch
        trainer.scheduler.load_state_dict(runtime["scheduler"])
        if len(trainer.optimizer.state) != len(raw["optimizer"]["state"]):
            raise RuntimeError("Optimizer state restore incomplete")
        if trainer.scaler.state_dict() != raw["scaler"]:
            raise RuntimeError("AMP scaler state restore incomplete")
        trainer.stopper.best_epoch = start_epoch
        write_json(output / "restoration.json", {
            "optimizer_parameter_states": len(trainer.optimizer.state), "scaler": trainer.scaler.state_dict(),
            "scheduler": trainer.scheduler.state_dict(), "restored_epoch": raw["epoch"],
            "not_exact": training["restoration"], "actual_device": str(trainer.device),
        })
        write_json(output / "resolved_trainer.json", {
            "args": vars(trainer.args), "device": str(trainer.device),
            "optimizer": type(trainer.optimizer).__name__, "initial_lrs": [g["initial_lr"] for g in trainer.optimizer.param_groups],
            "gpu": torch.cuda.get_device_name(), "train_batches": len(trainer.train_loader)})

    def save_callback(trainer):
        budget()
        if not all(torch.isfinite(p).all() for p in trainer.model.parameters()):
            raise RuntimeError("Refusing to save invalid model")
        checkpoint = output / "last_training.pt"
        shutil.copy2(trainer.last, checkpoint)
        torch.save({"scheduler": trainer.scheduler.state_dict(), "scaler": trainer.scaler.state_dict(),
            "guard": guard.state_dict(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "epoch": trainer.epoch, "sampler_limitation": "No exact live model/sampler restoration"},
            output / "last_runtime_state.pt")
        metrics = development_scores(output, checkpoint)
        eligible = all(metrics[m]["thresholds"]["40"]["recall"] >= baseline[m]["thresholds"]["40"]["recall"]-.02
                       for m in ("117093", "117092"))
        row = {"epoch": trainer.epoch+1, "additional_epoch": trainer.epoch-start_epoch+1,
               "eligible": eligible, "mean_f1": metrics["mean_per_match_f1"],
               "matches": {m: metrics[m]["thresholds"] for m in ("117093", "117092")},
               "native_box_metrics": trainer.metrics, "counters": asdict(guard.counters),
               "elapsed_seconds": time.monotonic()-started}
        training["epochs"].append(row)
        if training["candidate_best"] is None or row["mean_f1"] > training["candidate_best_mean_f1"]:
            candidate = output / "candidate_best.pt"
            shutil.copy2(checkpoint, candidate)
            training.update(candidate_best=str(candidate.relative_to(ROOT)), candidate_best_mean_f1=row["mean_f1"])
        if eligible and row["mean_f1"] > training["selected_mean_f1"]+.0001:
            selected = output / "selected.pt"
            shutil.copy2(checkpoint, selected)
            training.update(selected=str(selected.relative_to(ROOT)), selected_sha256=file_digest(selected),
                            selected_mean_f1=row["mean_f1"], epochs_without_gain=0)
            write_json(output / "selected_development.json", metrics)
        else:
            training["epochs_without_gain"] += 1
        write_json(output / "training_summary.json", training)
        print(f"DEVELOPMENT epoch={row['epoch']} mean_f1={row['mean_f1']:.5f} eligible={eligible}", flush=True)
        if training["epochs_without_gain"] >= 5:
            trainer.stop = True
        budget()

    def batch_end(trainer):
        budget()
        if not torch.isfinite(trainer.loss).all():
            raise RuntimeError("Nonfinite minibatch loss; stop without native recovery")

    model = YOLO(str(DEFAULT_OUTPUT / "selected.pt"))
    model.add_callback("on_train_start", started_callback)
    model.add_callback("on_model_save", save_callback)
    model.add_callback("on_train_batch_end", batch_end)
    overrides = {key: value for key, value in resolved.items()
                 if key not in ("save_dir", "model", "data", "epochs", "project", "name", "resume")}
    overrides.update(data=str(dataset_yaml), epochs=native_epochs, project=str(output), name="fit",
                     resume=False, plots=False, verbose=False, save_period=-1, exist_ok=False)
    try:
        model.train(trainer=GuardedTrainer, **overrides)
        training["stop_reason"] = "DEVELOPMENT_EARLY_STOP" if training["epochs_without_gain"] >= 5 else "MAX_ADDITIONAL_EPOCHS"
        training["status"] = "FINITE_TRAINING_COMPLETE"
    except FitBudgetReached:
        training.update(status="FINITE_TIME_BOUNDED", stop_reason="TRAINING_TIME_CAP")
    except Exception as exc:
        (output / "training_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        training.update(status="TRAINING_BRANCH_STOPPED", stop_reason=str(exc))
        # A safety failure never promotes a new model from this branch.
        training.update(selected=str((DEFAULT_OUTPUT / "selected.pt").relative_to(ROOT)),
                        selected_sha256=SELECTED_SHA, selected_mean_f1=baseline["mean_per_match_f1"])
    training.update(numerics=guard.state_dict(), optimizer_steps=guard.counters.committed,
                    elapsed_seconds=time.monotonic()-started, finished_utc=now())
    write_json(output / "training_summary.json", training)
    return training


def valid_cache(record: dict, weights: Path, source: dict, scheme: str) -> bool:
    path = ROOT / record["path"]
    return (record.get("detector_sha256") == file_digest(weights)
            and record.get("source_video_sha256") == source["video_sha256"]
            and record.get("configuration_sha256") == prediction_config_hash(scheme)
            and record.get("scheme") == scheme and record.get("sampling_hz") == 5
            and path.is_file() and file_digest(path) == record["prediction_jsonl_sha256"])


def prediction_config_hash(scheme: str) -> str:
    import hashlib
    from src.backbone.gpu_perception import merged_detections, tile_bounds, restore_boxes
    config={"scheme":scheme,"sampling_hz":5,"imgsz":1280,"confidence":.20,"native_nms_iou":.5,
            "implementation":[inspect.getsource(function) for function in (merged_detections,tile_bounds,restore_boxes)]}
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def evaluate_detectors(output: Path) -> dict:
    training = read(output / "training_summary.json") if (output / "training_summary.json").is_file() else {}
    selected = ROOT / training.get("selected", str((DEFAULT_OUTPUT / "selected.pt").relative_to(ROOT)))
    candidate = training.get("candidate_best")
    preserved = DEFAULT_OUTPUT / "selected.pt"
    choices = {"preserved": preserved}
    if candidate and (ROOT / candidate).is_file():
        choices["new_candidate"] = ROOT / candidate
    if file_digest(selected) not in [file_digest(path) for path in choices.values()]:
        choices["new_selected"] = selected
    # Freeze all choices before any retrospective reads/scoring.
    lock = {"created_utc": now(), "selected": str(selected.relative_to(ROOT)),
            "selected_sha256": file_digest(selected), "criterion": read(output / "task_state.json")["acceptance"]["detector_selection"],
            "routes": {name: {"path": str(path.relative_to(ROOT)), "sha256": file_digest(path)} for name,path in choices.items()},
            "exposure": "All six D/E windows previously examined: retrospective"}
    write_json(output / "detector_lock.json", lock)
    dev_path = output / "selected_final_development.json"
    development_scores(output, selected, save_name=dev_path.name)
    old_seal = read(DEFAULT_OUTPUT / "evaluation_seal.json")["files"]
    sources = source_records()
    protocol = load_protocol(HISTORICAL / "protocol.json")
    records = []
    for route, weights in choices.items():
        for match in ("117092", "118576"):
            for interval in protocol["split"]["final_intervals_seconds_by_match"][match]:
                cached = next((record for record in old_seal
                    if record["route"] == "gpu_selected" and record["match_id"] == match
                    and record["interval_seconds"] == interval and valid_cache(record, weights, sources[match], "consistent")), None)
                if cached is not None:
                    records.append({**cached, "route": route, "cache_reused": True,
                        "cache_contract": "checkpoint+video+consistent geometry+5Hz+sealed prediction SHA; confidence.20/NMS.5"})
                else:
                    record=predict_video(output, sources[match], interval, route, weights, scheme="consistent")
                    record["configuration_sha256"]=prediction_config_hash("consistent")
                    records.append(record)
    write_json(output / "detector_evaluation_seal.json", {"created_utc": now(), "files": records})
    scores = []
    for record in records:
        score = score_predictions(output, record, sources[record["match_id"]])
        path = output / "detector_scores" / record["route"] / f"{record['match_id']}_{record['interval_seconds'][0]}.json"
        write_json(path, score)
        scores.append({"path": str(path.relative_to(ROOT)), "route": record["route"], "match_id":record["match_id"],
                       "interval":record["interval_seconds"], "thresholds":score["thresholds"],"projection":score["projection"]})
    selected_route = next(route for route, weight in choices.items() if file_digest(weight) == file_digest(selected))
    result = {"status": "RETROSPECTIVE_DETECTOR_EVALUATION_COMPLETE", "selected_route":selected_route,
              "lock": lock, "scores":scores}
    write_json(output / "detector_evaluation.json", result)
    return result


def appearance_descriptor(image: np.ndarray, point: list[float], width: int) -> list[float] | None:
    scale = width/4096
    x,y = point
    x1,x2 = max(0,round(x-12*scale)), min(image.shape[1],round(x+12*scale))
    y1,y2 = max(0,round(y-46*scale)), min(image.shape[0],round(y-7*scale))
    crop = image[y1:y2,x1:x2]
    if x2<=x1 or y2<=y1 or crop.size < 48:
        return None
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[8,4],[0,180,0,256]).ravel()
    return (hist/max(float(hist.sum()),1.)).astype(float).tolist()


def associate_focused(saved: list[dict], width: int, method: str,
                      features: list[list[list[float] | None]] | None = None) -> list[dict]:
    if method == "baseline":
        return associate(saved,width,"greedy")
    if method not in ("damped_recent", "damped_appearance"):
        raise ValueError("Unknown frozen association method")
    result, tracks, next_id = copy.deepcopy(saved), {}, 1
    for sample,frame in enumerate(result):
        detections = frame["detections"]
        for detection in detections:
            detection.pop("track",None)
        candidates = []
        for track_id,track in tracks.items():
            age=sample-track["last_sample"]
            if not 1<=age<=5:
                continue
            predicted=np.asarray(track["point"])+np.asarray(track["velocity"])*age
            for j,detection in enumerate(detections):
                distance=float(np.linalg.norm(predicted-np.asarray(detection["footpoint_xy_px"])))
                if distance>45*width/4096*min(age,2):
                    continue
                app=0.
                feature = features[sample][j] if features is not None else None
                if method=="damped_appearance" and feature is not None and track["feature"] is not None:
                    app=float(np.abs(np.asarray(feature)-np.asarray(track["feature"])).sum())/2
                    if app>.65:
                        continue
                candidates.append((distance+8*(age-1)*width/4096+30*app*width/4096,track_id,j,app))
        assigned_tracks,assigned_detections=set(),set()
        for cost,track_id,j,app in sorted(candidates):
            if track_id in assigned_tracks or j in assigned_detections:
                continue
            track=tracks[track_id]
            point=detections[j]["footpoint_xy_px"]
            age=sample-track["last_sample"]
            displacement=(np.asarray(point)-np.asarray(track["point"]))/age
            track["velocity"]=(.35*displacement+.65*np.asarray(track["velocity"])).clip(-40,40).tolist()
            track.update(point=point,last_sample=sample)
            if features is not None and features[sample][j] is not None:
                track["feature"]=features[sample][j]
            detections[j]["track"]=track_id
            detections[j]["association_quality"]={"status":"OBSERVED_ASSOCIATED", "cost":cost,
                "age_samples":age,"appearance_distance":app,"uncertainty":"UNVALIDATED_ASSOCIATION"}
            assigned_tracks.add(track_id)
            assigned_detections.add(j)
        for j,detection in enumerate(detections):
            if j in assigned_detections:
                continue
            track_id=f"video-player-{next_id:04d}"
            next_id+=1
            tracks[track_id]={"point":detection["footpoint_xy_px"],"velocity":[0.,0.],
                "last_sample":sample,"feature":features[sample][j] if features is not None else None}
            detection["track"]=track_id
            detection["association_quality"]={"status":"NEW_OBSERVED_TRACK","uncertainty":"UNVALIDATED_ASSOCIATION"}
        tracks={key:track for key,track in tracks.items() if sample-track["last_sample"]<=5}
    return result


def feature_stream(saved: list[dict], source: dict) -> list:
    cap=cv2.VideoCapture(str(ROOT/source["video"]),cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise OSError("Cannot open appearance source video")
    result,previous=[],None
    try:
        for frame in saved:
            image=_read_frame(cap,frame["frame_index"],previous)
            previous=frame["frame_index"]
            result.append([appearance_descriptor(image,row["footpoint_xy_px"],source["dimensions_px"][0])
                           for row in frame["detections"]])
    finally:
        cap.release()
    return result


def same_detections(original: list[dict], associated: list[dict]) -> None:
    if len(original)!=len(associated):
        raise ValueError("Tracker changed frame coverage")
    for before,after in zip(original,associated,strict=True):
        if before["frame_index"]!=after["frame_index"] or len(before["detections"])!=len(after["detections"]):
            raise ValueError("Tracker omitted frames or detections")
        for a,b in zip(before["detections"],after["detections"],strict=True):
            for key in ("bbox_xyxy_px","footpoint_xy_px","detector_confidence"):
                if a.get(key)!=b.get(key):
                    raise ValueError("Association changed an observed detection")


def tracker_eligible(baseline: dict, alternative: dict) -> bool:
    old_rates,new_rates=[],[]
    for match in ("117093","117092"):
        old,new=(r[match]["thresholds"]["40"] for r in (baseline,alternative))
        if old["switch_rate"] is None or new["switch_rate"] is None:
            return False
        old_rates.append(old["switch_rate"])
        new_rates.append(new["switch_rate"])
        if (new["truth_instance_complete_history_fraction"] or 0)<(old["truth_instance_complete_history_fraction"] or 0)-.01:
            return False
        if new["fragmentation_extra_tracks_per_identity"]>old["fragmentation_extra_tracks_per_identity"]*1.05+1:
            return False
    return float(np.mean(new_rates))<float(np.mean(old_rates))


def evaluate_trackers(output: Path) -> dict:
    lock={"created_utc":now(),"alternatives":["damped_recent","damped_appearance"],
          "gate_width4096_px":45,"velocity_smoothing":.35,"max_gap_samples":5,
          "stale_penalty_px":8,"appearance_weight_px":30,"appearance_max_l1_distance":.65,
          "selection":read(output/"task_state.json")["acceptance"]["tracker_selection"],
          "appearance":"Non-trained HSV8x4 native video crop above detected footpoint; no reference identity at runtime"}
    write_json(output/"tracker_lock.json",lock)
    saved=read(output/"selected_final_development_predictions.json")
    sources=source_records()
    metrics={method:{} for method in ("baseline","damped_recent","damped_appearance")}
    for match in ("117093","117092"):
        features=feature_stream(saved[match],sources[match])
        labels=_read_annotations(sources[match],{r["frame_index"] for r in saved[match]})
        for method in metrics:
            associated=associate_focused(saved[match],sources[match]["dimensions_px"][0],method,features)
            same_detections(saved[match],associated)
            metrics[method][match]=aggregate_scores(associated,labels,sources[match]["dimensions_px"][0])
            write_json(output/f"development_tracks_{method}_{match}.json",{"frames":associated})
    selected="baseline"
    for method in ("damped_recent","damped_appearance"):
        if tracker_eligible(metrics["baseline"],metrics[method]):
            old=np.mean([metrics[selected][m]["thresholds"]["40"]["switch_rate"] for m in ("117093","117092")])
            new=np.mean([metrics[method][m]["thresholds"]["40"]["switch_rate"] for m in ("117093","117092")])
            if new<old:
                selected=method
    # Freeze setting before reading any retrospective identities.
    write_json(output/"tracker_selection.json",{"selected":selected,"metrics":metrics,"lock":lock,
               "negative_result":selected=="baseline","selected_utc":now()})
    evaluation=read(output/"detector_evaluation.json")
    seal=read(output/"detector_evaluation_seal.json")["files"]
    records=[r for r in seal if r["route"]==evaluation["selected_route"]]
    derived,scores=[],[]
    for record in records:
        original=frames(ROOT/record["path"])
        features=feature_stream(original,sources[record["match_id"]]) if selected=="damped_appearance" else None
        for method in sorted({"baseline",selected}):
            associated=associate_focused(original,sources[record["match_id"]]["dimensions_px"][0],method,features)
            same_detections(original,associated)
            path=output/"tracking"/method/Path(record["path"]).name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text("".join(json.dumps(row,separators=(",",":"))+"\n" for row in associated),encoding="utf-8")
            new={**record,"route":method,"path":str(path.relative_to(ROOT)),
                 "prediction_jsonl_sha256":file_digest(path),"source_detection_sha256":record["prediction_jsonl_sha256"]}
            derived.append(new)
            score=score_predictions(output,new,sources[record["match_id"]])
            score["ambiguity_related_detection_abstentions"]=0
            score["runtime_policy"]="No detections removed; unassigned detections start new observed tracks; no interpolation"
            path=output/"tracker_scores"/method/f"{record['match_id']}_{record['interval_seconds'][0]}.json"
            write_json(path,score)
            scores.append({"method":method,"match":record["match_id"],"path":str(path.relative_to(ROOT)),"thresholds":score["thresholds"]})
    result={"status":"FIXED_DETECTION_TRACKER_EVALUATION_COMPLETE","selected":selected,"records":derived,"scores":scores}
    write_json(output/"tracking_summary.json",result)
    return result


def build_replay(destination: Path, match: str, interval: list[int],
                 original: dict, selected: dict) -> dict:
    from src.backbone.video_player_cross_match_replay import (
        build_external_observations, replay_route,
    )
    if destination.exists():
        raise FileExistsError("Replay destination already exists; preserve partial evidence for repair")
    destination.mkdir(parents=True)
    protocol=load_protocol(HISTORICAL/"protocol.json")
    protocol=copy.deepcopy(protocol)
    protocol["split"]["development_match"]=match
    protocol["split"]["development_interval_seconds"]=interval
    write_json(destination/"protocol.json",protocol)
    (destination/"calibration").mkdir()
    shutil.copy2(HISTORICAL/"calibration"/f"local_projector_{match}.json",
                 destination/"calibration"/f"local_projector_{match}.json")
    source=source_records()[match]
    for route,record in (("frozen",original),("adapted",selected)):
        path=ROOT/record["path"]
        if file_digest(path)!=record["prediction_jsonl_sha256"]:
            raise ValueError("Sealed replay source predictions changed")
        prediction=destination/"predictions"/route/f"{match}_{source['half']}_development.jsonl"
        prediction.parent.mkdir(parents=True)
        with prediction.open("w",encoding="utf-8") as stream:
            for row in frames(path):
                row["interval"]="development"
                stream.write(json.dumps(row,separators=(",",":"))+"\n")
        write_json(prediction.with_suffix(".meta.json"),{
            **record,"route":route,"interval":"development","sample_count":(interval[1]-interval[0])*5,
            "prediction_jsonl_sha256":file_digest(prediction),
            "derived_from_sealed_prediction_sha256":record["prediction_jsonl_sha256"]})
    builds=[build_external_observations(protocol,destination,match,"development",route,
                assisted_attributes=False,device="0") for route in ("adapted","frozen")]
    replays=[replay_route(protocol,destination,match,"development",route,
                assisted_attributes=False) for route in ("adapted","frozen")]
    result={"match":match,"interval_seconds":interval,"builds":builds,"replays":replays,
            "comparison":"Preserved finite detector+baseline tracker versus development-selected detector+tracker",
            "actor_anchors":"Same video-only candidate prefix anchors paired to baseline detections; fixed query20/40/60/80/100s",
            "global_calibration":"FAILED/UNCHANGED; local projection experimental"}
    write_json(destination/"replay_summary.json",result)
    return result


def replay_comparison(output: Path) -> dict:
    evaluation=read(output/"detector_evaluation_seal.json")["files"]
    tracker_path=output/"tracking_summary.json"
    tracking=read(tracker_path) if tracker_path.is_file() else {"selected":"baseline","records":[]}
    selected_route=read(output/"detector_evaluation.json")["selected_route"]
    results=[]
    for match in ("117092","118576"):
        original=next(r for r in evaluation if r["route"]=="preserved" and r["match_id"]==match and r["interval_seconds"]==[300,420])
        selected=next((r for r in tracking["records"] if r["route"]==tracking["selected"]
            and r["match_id"]==match and r["interval_seconds"]==[300,420]),None)
        if selected is None:
            selected=next(r for r in evaluation if r["route"]==selected_route and r["match_id"]==match and r["interval_seconds"]==[300,420])
        results.append(build_replay(output/f"r{match}",match,[300,420],original,selected))
        write_json(output/"integration_progress.json",{"results":results})
    write_json(output/"integration_summary.json",{"status":"SCOPED_REPLAYS_COMPLETE","results":results})
    # Existing enhanced renderer exposes native video, unprojected detections,
    # actor states, actual query outputs, and small experimental pitch maps.
    render_diagnostics(output,review=True)
    return {"status":"TWO_MATCHED_120S_COMPARISONS_COMPLETE","results":results}


def expansion(output: Path) -> dict:
    if not before_reserve(output):
        result={"status":"SKIPPED_VERIFICATION_RESERVE"}
        write_json(output/"expansion.json",result)
        return result
    # Freeze this continuous interval from setup, never select on its results.
    match,interval="118576",[600,1200]
    write_json(output/"expansion_lock.json",{
        "match":match,"interval_seconds":interval,"sampling_hz":5,
        "exposure":"Retrospective recording E; no selection on this window",
        "objective":"Ten-minute identity drift, causal-memory continuity, failed frames and model availability"})
    weights=ROOT/read(output/"detector_lock.json")["selected"]
    source=source_records()[match]
    method=read(output/"tracker_selection.json")["selected"] if (output/"tracker_selection.json").is_file() else "baseline"
    if (output/"continuous_seal.json").is_file():
        selected=read(output/"continuous_seal.json")
        # Reuse the exact sealed native detections; no second fitting or selection.
        candidates=list((output/"predictions"/"continuous").glob("*.meta.json"))
        record=next(({**read(p),"path":str(p.with_suffix("").with_suffix(".jsonl").relative_to(ROOT))} for p in candidates
                     if read(p).get("prediction_jsonl_sha256")==selected["source_detection_sha256"]),None)
        if record is None:
            raise ValueError("Continuous source prediction sidecar is missing")
        prediction=ROOT/record["path"]
        if (file_digest(prediction)!=selected["source_detection_sha256"]
                or file_digest(ROOT/selected["path"])!=selected["prediction_jsonl_sha256"]
                or selected["detector_sha256"]!=file_digest(weights)
                or selected["source_video_sha256"]!=source["video_sha256"]
                or selected["association"]!=method):
            raise ValueError("Continuous cache provenance changed")
        original=frames(prediction)
        same_detections(original,frames(ROOT/selected["path"]))
    else:
        record=predict_video(output,source,interval,"continuous",weights,scheme="consistent")
        original=frames(ROOT/record["path"])
        feature=feature_stream(original,source) if method=="damped_appearance" else None
        associated=associate_focused(original,source["dimensions_px"][0],method,feature)
        same_detections(original,associated)
        path=output/"continuous_tracks.jsonl"
        path.write_text("".join(json.dumps(row,separators=(",",":"))+"\n" for row in associated),encoding="utf-8")
        selected={**record,"path":str(path.relative_to(ROOT)),"prediction_jsonl_sha256":file_digest(path),
                  "source_detection_sha256":record["prediction_jsonl_sha256"],"association":method}
        write_json(output/"continuous_seal.json",selected)
    score=score_predictions(output,selected,source)
    write_json(output/"continuous_score.json",score)
    baseline=copy.deepcopy(record)
    destination=output/"c118576"
    if destination.exists():
        destination=next((output/f"c118576_repair{i}" for i in (1,2)
                          if not (output/f"c118576_repair{i}").exists()),None)
        if destination is None:
            raise RuntimeError("Two continuous replay repairs exhausted")
    replay=build_replay(destination,match,interval,baseline,selected)
    replay["comparison"]="Fixed selected-detector detections: baseline greedy versus development-selected association"
    availability=[]
    for route in ("adapted","frozen"):
        rows=frames(destination/"backbone_replay_usable_video_selection/video_estimated"/f"{route}_{match}_{source['half']}_development.jsonl")
        for minute in range(10):
            part=[r for r in rows if 600+minute*60<=r["timestamp_s"]<660+minute*60]
            availability.append({
                "route":route,"minute":minute,"actor_samples":len(part),
                "actor_available_samples":sum(r["belief"]["actor_available"] for r in part),
                "g1_entity_states":sum(e["backend"]=="G1" for r in part for e in r["belief"]["entities"]),
                "scheduled_queries":sum(r["scheduled_s4_query"] for r in part),
                "supported_queries":sum(r["scheduled_s4_query"] and r["action_inference"]["status"]=="supported" for r in part),
            })
    result={"status":"TEN_MINUTE_CONTINUOUS_COMPLETE","match":match,"interval_seconds":interval,
            "samples":len(original),"expected_samples":3000,"failed_frames":3000-len(original),
            "tracking":score["thresholds"],"replay":replay,"per_minute_memory_availability":availability,
            "limitations":"Fixed video actors may leave view/lose track; missing availability is counted, not filled. Queries remain fixed prefix schedule."}
    write_json(output/"expansion.json",result)
    return result


def verify(output: Path) -> dict:
    before=read(output/"protected_before.json")
    after={path:file_digest(ROOT/path) for path in before}
    write_json(output/"protected_after.json",after)
    if before!=after:
        raise ValueError("Protected historical artifacts changed")
    for seal_name in ("detector_evaluation_seal.json",):
        if not (output/seal_name).is_file():
            continue
        for record in read(output/seal_name)["files"]:
            if file_digest(ROOT/record["path"])!=record["prediction_jsonl_sha256"]:
                raise ValueError("Sealed predictions changed")
    task=read(output/"task_state.json")
    tests=subprocess.run([sys.executable,"-m","pytest","-q","-rA",
        "tests/test_amp_guard.py","tests/test_gpu_perception.py",
        "tests/test_perception_continuation.py","tests/test_video_player_cross_match_replay.py",
        "tests/test_backbone_release.py","--junitxml="+str(output/"final_tests.xml")],
        cwd=ROOT,text=True,capture_output=True,timeout=300)
    (output/"final_tests.txt").write_text(tests.stdout+tests.stderr,encoding="utf-8")
    if tests.returncode:
        raise RuntimeError("Focused verification failed")
    result={"status":"PASS","protected_hashes_unchanged":True,
        "regressions":tests.stdout.strip().splitlines()[-1],"finished_utc":now(),
        "within_deadline":datetime.now(timezone.utc)<datetime.fromisoformat(task["hard_deadline_utc"])}
    write_json(output/"verification.json",result)
    return result


def package(output: Path) -> dict:
    task=read(output/"task_state.json")
    training=read(output/"training_summary.json") if (output/"training_summary.json").is_file() else {}
    evaluation=read(output/"detector_evaluation.json") if (output/"detector_evaluation.json").is_file() else {}
    tracking=read(output/"tracking_summary.json") if (output/"tracking_summary.json").is_file() else {}
    integration=read(output/"integration_summary.json") if (output/"integration_summary.json").is_file() else {}
    verification=read(output/"verification.json") if (output/"verification.json").is_file() else {}
    tables=[]
    for route in sorted({row["route"] for row in evaluation.get("scores",[])}):
        for match in ("117092","118576"):
            totals=Counter()
            for row in evaluation["scores"]:
                if row["route"]==route and row["match_id"]==match:
                    for key in ("matches","detections","truth_instances","misses","duplicates","false_positives_including_duplicates"):
                        totals[key]+=row["thresholds"]["40"][key]
            p=totals["matches"]/totals["detections"] if totals["detections"] else 0
            r=totals["matches"]/totals["truth_instances"] if totals["truth_instances"] else 0
            tables.append({"route":route,"match":match,**dict(totals),"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0})
    calls=[{"match":match["match"],"route":replay["detector_route"],
             "g1":replay["g1_forward_calls"],"s4":replay["s4_neural_forward_calls"],
             "actors":replay["actors"],"attribute_counts":next(b["counts"] for b in match["builds"] if b["detector_route"]==replay["detector_route"])}
           for match in integration.get("results",[]) for replay in match["replays"]]
    core_complete=all(task["stages"].get(stage,{}).get("status")=="COMPLETE" for stage in ("detectors","tracking","replay"))
    result={"execution_status":"COMPLETE_CORE_EXPERIMENT" if core_complete else "PARTIAL_BLOCKED",
        "training_status":training.get("status"),"training_counters":training.get("numerics",{}).get("counters",{}),
        "detector_improved_on_development":training.get("selected_sha256")!=SELECTED_SHA if training else False,
        "tracker_improved_on_development":tracking.get("selected","baseline")!="baseline",
        "retrospective_pixel_table":tables,"actual_model_calls":calls,"verification":verification,
        "calibration":"FAILED_UNCHANGED; all local projected observations experimental",
        "historical_failure_reproduced":"Not established by this experiment","finished_utc":now()}
    write_json(output/"completion.json",result)
    lines=["# Guarded GPU perception and continuity",
        f"Execution: {result['execution_status']}. Training: {training.get('status')}; {training.get('stop_reason')}.",
        f"Committed/skipped updates: {result['training_counters']}.",
        "Initialized from protected finite detector, never disposable stability weights. Native optimizer/scaler/scheduler restored when compatible; not exact live-model/sampler continuation.",
        f"Detector development improvement: {result['detector_improved_on_development']}; tracker development improvement: {result['tracker_improved_on_development']}.",
        "Detector and tracker choices were locked on development before retrospective evaluation. Negative alternatives remain reported.",
        "| Detector | Match | Matches/truth | Precision40 | Recall40 | F1-40 |",
        "|---|---|---|---:|---:|---:|"]
    for row in tables:
        lines.append(f"|{row['route']}|{row['match']}|{row['matches']}/{row['truth_instances']}|{row['precision']:.3f}|{row['recall']:.3f}|{row['f1']:.3f}|")
    lines+=["Identity/fragmentation/history denominators are in tracking_summary.json and tracker_scores; detector boxes/footpoints are unchanged across associations.",
        "Actual forward calls: "+str([{k:v for k,v in row.items() if k not in ("actors","attribute_counts")} for row in calls]),
        "Unknown team/role remains unknown. Local calibration is still FAILED; camera support is assistance, not certification. Visibility is simulated, ball/possession unknown; G1 hidden estimates do not feed S4.",
        "The appearance feature is non-trained video HSV, not reference identity. Calls and action softmax are not tactical accuracy.",
        "See expansion.json for the ten-minute drift and real memory-availability diagnostic.",
        "Frozen models, annotations and historical artifacts remain unchanged. No commits, pushes or promotion.",
        "```powershell",
        f"& .\\.venv\\Scripts\\python.exe -m src.backbone.perception_continuation --output '{output.relative_to(ROOT)}' resume",
        f"& .\\.venv\\Scripts\\python.exe -m src.backbone.perception_continuation --output '{output.relative_to(ROOT)}' verify",
        f"& .\\.venv\\Scripts\\python.exe -m src.backbone.video_review_server --directory '{output.relative_to(ROOT)}' --port 8767",
        "```",
        "Resume advances unfinished stages only; it never restarts a fit. Deadlines are persistent. Logs/errors and exact source snapshots are local; weights/media excluded from return ZIP."]
    (output/"handoff.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    videos=read(output/"comparison_videos_review.json").get("videos",[]) if (output/"comparison_videos_review.json").is_file() else []
    (output/"index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Perception continuation</title>"
        "<style>body{background:#181818;color:white;font:16px system-ui}video{width:min(95%,1280px);display:block}</style>"
        "<h1>Preserved baseline / development-selected detector and tracker</h1>"
        "<p>Experimental projection; calibration FAILED. Simulated visibility; unknown attributes retained. Not tactical accuracy.</p>"+
        "".join(f"<video controls preload='metadata' src='{Path(v['path']).name}'></video>" for v in videos)+
        "<a href='handoff.md'>Handoff</a> | <a href='completion.json'>Measured results</a>",encoding="utf-8")
    names=["src/backbone/perception_continuation.py","src/backbone/amp_guard.py",
           "src/backbone/video_player_cross_match_replay.py","tests/test_perception_continuation.py"]
    patch=[]
    for name in names:
        path=ROOT/name
        target=output/"changed"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        tracked=subprocess.run(["git","show","HEAD:"+name],cwd=ROOT,capture_output=True,text=True)
        patch.extend(difflib.unified_diff(tracked.stdout.splitlines(keepends=True) if tracked.returncode==0 else [],
            path.read_text().splitlines(keepends=True),fromfile="a/"+name if tracked.returncode==0 else "/dev/null",tofile="b/"+name))
    (output/"source.patch").write_text("".join(patch),encoding="utf-8")
    destination=output/"return_perception.zip"
    if destination.exists():
        destination=output/f"return_perception_{datetime.now().strftime('%H%M%S')}.zip"
    with zipfile.ZipFile(destination,"x",zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path.suffix not in (".json",".jsonl",".txt",".xml",".md",".py",".patch",".html"):
                continue
            if path.stat().st_size>8_000_000 or any(folder in path.relative_to(output).parts for folder in ("predictions","tracking","fit")):
                continue
            # Replay frame streams are bulky; compact summaries/call counts stay included.
            if path.suffix==".jsonl" and path.name!="step_events.jsonl":
                continue
            archive.write(path,path.relative_to(output))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Return ZIP verification failed")
    return {**result,"return_zip":str(destination.relative_to(ROOT))}


def preservation_pins() -> dict:
    pins=protected(DEFAULT_OUTPUT)
    manifest=read(DEFAULT_OUTPUT/"data_manifest.json")
    for source in manifest["sources"].values():
        pins[source["annotation"]]=source["annotation_sha256"]
        for name,sha_name in (("calibration_keypoints","keypoints_sha256"),
                              ("camera_intrinsics","camera_intrinsics_sha256")):
            if name in source:
                pins[source[name]]=source[sha_name]
    return pins


def supervise(output: Path, *, resume: bool) -> dict:
    state=read(output/"task_state.json")
    if not resume and state.get("status") not in ("IMPLEMENTING","READY"):
        raise RuntimeError("Use resume for a previously started root")
    if not (output/"protected_before.json").exists():
        write_json(output/"protected_before.json",preservation_pins())
    snapshot=output/"executed_source"
    if not snapshot.exists():
        from ultralytics.engine.trainer import BaseTrainer
        for name in ("src/backbone/perception_continuation.py","src/backbone/amp_guard.py","tests/test_perception_continuation.py"):
            path=snapshot/name
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,path)
        write_json(output/"environment.json",{
            "interpreter":sys.executable,"torch":torch.__version__,"cuda":torch.version.cuda,
            "source_sha256":{str(p.relative_to(snapshot)):file_digest(p) for p in snapshot.rglob("*.py")},
            "worker_lifetime":"Session-attached local subprocesses; not claimed to survive host termination",
            "fit_restore":"Native EMA+optimizer/scaler/scheduler where compatible; not exact sampler/live-model continuation",
            "installed_parent":str(inspect.getfile(BaseTrainer)),
            "installed_parent_sha256":file_digest(Path(inspect.getfile(BaseTrainer))),
            "parent_optimizer_step":inspect.getsource(BaseTrainer.optimizer_step),
            "scaler_step":inspect.getsource(torch.amp.GradScaler.step),
            "scaler_update":inspect.getsource(torch.amp.GradScaler.update)})
    for stage,minutes in (("train",185),("detectors",100),("tracking",40),("replay",60),("expand",70),("verify",30)):
        state=read(output/"task_state.json")
        if state["stages"].get(stage,{}).get("status")=="COMPLETE":
            continue
        remaining=(datetime.fromisoformat(state["verification_reserve_utc" if stage!="verify" else "hard_deadline_utc"])-datetime.now(timezone.utc)).total_seconds()
        if remaining<=180:
            state["stages"][stage]={"status":"SKIPPED_DEADLINE","utc":now()}
            journal(output,stages=state["stages"])
            continue
        if stage=="train" and (output/"training_started.json").exists():
            if not (output/"training_summary.json").exists():
                write_json(output/"training_summary.json",{"status":"TRAINING_INTERRUPTED_RETAIN_PRESERVED",
                    "selected":str((DEFAULT_OUTPUT/"selected.pt").relative_to(ROOT)),"selected_sha256":SELECTED_SHA})
            state["stages"][stage]={"status":"BLOCKED_NO_SECOND_FIT","utc":now()}
            journal(output,stages=state["stages"])
            continue
        dependencies={"tracking":"detector_evaluation.json","replay":"detector_evaluation.json","expand":"detector_lock.json"}
        if stage in dependencies and not (output/dependencies[stage]).is_file():
            state["stages"][stage]={"status":"BLOCKED_DEPENDENCY","missing":dependencies[stage]}
            journal(output,stages=state["stages"])
            continue
        journal(output,status="RUNNING",next_action=stage)
        with (output/f"{stage}.stdout.log").open("w",encoding="utf-8") as stdout, (output/f"{stage}.stderr.log").open("w",encoding="utf-8") as stderr:
            process=subprocess.Popen([sys.executable,"-m","src.backbone.perception_continuation",
                                      "--output",str(output),stage],cwd=ROOT,stdout=stdout,stderr=stderr)
            try:
                code=process.wait(timeout=min(minutes*60,remaining-120))
                outcome={"status":"COMPLETE" if code==0 else "BLOCKED","exit_code":code,"finished_utc":now()}
            except subprocess.TimeoutExpired:
                # Only this owned stage and its explicit descendants may be stopped.
                subprocess.run(["powershell","-NoProfile","-Command",
                    f"$children=Get-CimInstance Win32_Process -Filter 'ParentProcessId={process.pid}'; "
                    "foreach($p in $children){Stop-Process -Id $p.ProcessId -ErrorAction SilentlyContinue}; "
                    f"Stop-Process -Id {process.pid} -ErrorAction SilentlyContinue"],check=True,timeout=30)
                outcome={"status":"BLOCKED_TIMEOUT","finished_utc":now()}
        state=read(output/"task_state.json")
        state["stages"][stage]=outcome
        if stage!="verify":
            state["stages"].pop("verify",None)
        journal(output,stages=state["stages"],next_action="Advance after stage evidence; failed optional branches do not stop independent work")
        if stage=="train" and not (output/"training_summary.json").exists():
            write_json(output/"training_summary.json",{"status":"BLOCKED_RETAIN_PRESERVED",
                "selected":str((DEFAULT_OUTPUT/"selected.pt").relative_to(ROOT)),"selected_sha256":SELECTED_SHA})
    result=package(output)
    journal(output,status=result["execution_status"],next_action="Handoff; no further training or release promotion",return_zip=result["return_zip"])
    print(json.dumps({key:value for key,value in result.items() if key!="actual_model_calls"},indent=2),flush=True)
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("command",choices=("run","resume","train","detectors","tracking","replay","expand","verify","package"))
    args=parser.parse_args()
    output=args.output.resolve()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if args.command in ("run","resume"):
        supervise(output,resume=args.command=="resume")
    else:
        try:
            {"train":train_candidate,"detectors":evaluate_detectors,"tracking":evaluate_trackers,
             "replay":replay_comparison,"expand":expansion,"verify":verify,"package":package}[args.command](output)
        except Exception:
            (output/f"{args.command}_error.txt").write_text(traceback.format_exc(),encoding="utf-8")
            raise
    return 0


if __name__=="__main__":
    raise SystemExit(main())
