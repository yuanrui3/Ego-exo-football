"""Causal actor-session experiment on sealed detections; no training or promotion."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import copy
from datetime import datetime, timedelta, timezone
import difflib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.backbone.perception_continuation import (
    ROOT, frames, read, same_detections, preservation_pins,
    associate_focused, tracker_eligible,
)
from src.backbone.gpu_perception import source_records, aggregate_scores
from src.backbone.video_player_cross_match import (
    _read_annotations, _maximum_matches, _read_frame,
)
from src.backbone.video_player_cross_match_replay import (
    external_frames, ExternalObservationSource, ModelSession, ActionModelSession,
    ReplayEngine, SyntheticObservationProvider,
)
from src.trajectory_dataset import file_digest, write_json

SOURCE = ROOT / "outputs/pc_20261029_144726"
EXTERNAL = SOURCE / "c118576_repair1/external/video_estimated/adapted_118576_2nd_development.jsonl"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def state(output: Path, **values) -> dict:
    result = read(output / "task_state.json")
    result.update(values, updated_utc=utc())
    write_json(output / "task_state.json", result)
    return result


def budget(output: Path) -> None:
    if datetime.now(timezone.utc) >= datetime.fromisoformat(read(output / "task_state.json")["work_deadline_utc"]):
        raise TimeoutError("Verification reserve reached")


def inputs() -> tuple[list[dict], dict, list[dict], dict]:
    seal = read(SOURCE / "continuous_seal.json")
    prediction = ROOT / seal["path"]
    if file_digest(prediction) != seal["prediction_jsonl_sha256"]:
        raise ValueError("Historical continuous predictions changed")
    saved = frames(prediction)
    source = source_records()["118576"]
    rows = frames(EXTERNAL)
    header, external = rows[0], rows[1:]
    meta = read(EXTERNAL.with_suffix(".meta.json"))
    derived = SOURCE / "c118576_repair1/predictions/adapted/118576_2nd_development.jsonl"
    derived_meta = read(derived.with_suffix(".meta.json"))
    if (header["origin"]["prediction_sha256"] != file_digest(derived)
            or derived_meta["derived_from_sealed_prediction_sha256"] != file_digest(prediction)
            or meta["external_observation_sha256"] != file_digest(EXTERNAL)):
        raise ValueError("External stream is not tied to continuous predictions")
    if len(saved) != 3000 or len(external) != 3000:
        raise ValueError("Expected 3000 chronological samples")
    for frame, row in zip(saved, external, strict=True):
        if (frame["frame_index"] != row["frame_index"]
                or [d["track"] for d in frame["detections"]] != [m["track"] for m in row["measurements"]]):
            raise ValueError("Detection-to-external index alignment failed")
    return saved, source, external, header


def label_matches(saved: list[dict], source: dict) -> tuple[dict, list[dict]]:
    """Post-inference diagnostics only. Never called by the association API."""
    truth = _read_annotations(source, {f["frame_index"] for f in saved})
    matched = []
    for frame in saved:
        labels = truth.get(frame["frame_index"], [])
        pairs = _maximum_matches(frame["detections"], labels, 40 * source["dimensions_px"][0] / 4096)
        matched.append({pi: labels[gi] for pi, gi, _ in pairs})
    return truth, matched


def diagnose(output: Path) -> dict:
    saved, source, external, header = inputs()
    expansion = read(SOURCE / "expansion.json")
    anchors = expansion["replay"]["replays"][0]["actors_selected_by_video_only_rule"]
    truth, matches = label_matches(saved, source)
    historical = frames(SOURCE / "c118576_repair1/backbone_replay_usable_video_selection/video_estimated/adapted_118576_2nd_development.jsonl")
    minute_rows, traces, pictures = [], [], []
    for anchor_index, anchor in enumerate(anchors):
        sample = next(i for i, f in enumerate(saved) if f["frame_index"] == anchor["frame_index"])
        pi = next(i for i, d in enumerate(saved[sample]["detections"]) if d["track"] == anchor["track"])
        reference = matches[sample].get(pi)
        votes = Counter(m[j]["track_id"] for f,m in zip(saved[:100],matches[:100],strict=True)
                        for j,d in enumerate(f["detections"]) if d["track"]==anchor["track"] and j in m)
        majority = votes.most_common(1)
        identity = majority[0][0] if majority and majority[0][1]/sum(votes.values())>=.9 else None
        anchor["post_inference_prefix_reference_votes"] = dict(votes)
        anchor["post_inference_reference"] = identity
        anchor["exact_anchor_reference_match"] = reference["track_id"] if reference else None
        anchor["reference_warning"] = "Dominant first100-sample annotation match for diagnosis, NOT certified original identity or runtime input"
        last_present, first_lost, reappearance = None, False, False
        per_minute = [Counter() for _ in range(10)]
        for i, (frame, row) in enumerate(zip(saved, external, strict=True)):
            minute = min(9, int((frame["timestamp_s"] - 600) / 60))
            counts = per_minute[minute]
            counts["opportunities"] += 1
            actor_index = next((j for j, d in enumerate(frame["detections"]) if d["track"] == anchor["track"]), None)
            target = next((a for a in truth.get(frame["frame_index"], []) if a["track_id"] == identity), None) if identity else None
            candidate = next((j for j, a in matches[i].items() if a["track_id"] == identity), None) if identity else None
            cause = "RUNTIME_TRACK_ABSENT_REFERENCE_UNRESOLVED"
            if actor_index is not None:
                counts["image_track_present"] += 1
                last_present = i
                measurement = row["measurements"][actor_index]
                cause = "PROJECTED_ACTOR_PRESENT" if measurement["xy"] is not None else "OUTSIDE_SUPPORTED_PROJECTION"
                counts["actor_projected"] += int(measurement["xy"] is not None)
                counts["unknown_team"] += int(measurement["team"] == "UNKNOWN")
                counts["unknown_role"] += int(measurement["goalkeeper"] is None)
                actual = matches[i].get(actor_index)
                if actual and identity:
                    counts["reference_matched_actor_samples"] += 1
                    counts["wrong_reference_identity"] += int(actual["track_id"] != identity)
            elif candidate is not None:
                cause = "REFERENCE_MATCHED_DETECTION_UNDER_DIFFERENT_TRACK"
                if last_present is not None and not reappearance:
                    pictures.append((anchor_index, i, candidate, "reference person under replacement ID"))
                    reappearance = True
            elif target:
                x, y = target["footpoint_xy_px"]
                cause = ("REFERENCE_POINT_OUTSIDE_IMAGE" if not (0 <= x < source["dimensions_px"][0] and 0 <= y < source["dimensions_px"][1])
                         else "REFERENCE_IN_IMAGE_NO_MATCHED_DETECTION")
            elif identity:
                cause = "REFERENCE_NOT_ANNOTATED_PHYSICAL_VISIBILITY_UNRESOLVED"
            counts[cause] += 1
            counts["reference_detection_present"] += int(candidate is not None)
            if candidate is not None:
                m = row["measurements"][candidate]
                counts["reference_detection_unprojected"] += int(m["xy"] is None)
                counts["reference_detection_unknown_attributes"] += int(m["team"] == "UNKNOWN" or m["goalkeeper"] is None)
            if last_present is not None and actor_index is None and not first_lost:
                pictures.extend([(anchor_index, last_present, None, "last original runtime track"),
                                 (anchor_index, i, candidate, "first original runtime ID loss")])
                first_lost = True
            traces.append({"actor":anchor["track"],"frame_index":frame["frame_index"],"timestamp_s":frame["timestamp_s"],
                "cause":cause,"reference_track":identity,"reference_detection_index":candidate,
                "replacement_runtime_track":frame["detections"][candidate]["track"] if candidate is not None else None})
        for minute, count in enumerate(per_minute):
            replay = [r for r in historical if r["runtime_actor_track"] == anchor["track"]
                      and 600 + minute * 60 <= r["timestamp_s"] < 660 + minute * 60]
            count["historical_actor_available"] = sum(r["belief"]["actor_available"] for r in replay)
            count["historical_g1_contract_ready"] = sum(r["belief"]["model_contract_met"] for r in replay)
            count["historical_queries"] = sum(r["scheduled_s4_query"] for r in replay)
            minute_rows.append({"actor":anchor["track"],"minute":minute,**dict(count)})
    (output / "anchor_trace.jsonl").write_text("".join(json.dumps(r)+"\n" for r in traces), encoding="utf-8")
    contact_sheet(output, saved, source, anchors, truth, pictures)
    availability = [sum(r.get("historical_actor_available", 0) for r in minute_rows if r["minute"] == m) for m in range(10)]
    if availability != [359] + [0] * 9:
        raise ValueError("Historical availability observation did not reproduce")
    result = {"status":"DIAGNOSED","samples":len(saved),"sampling_hz":5,
        "historical_availability_per_600_opportunities":availability,"per_minute":minute_rows,
        "anchors":anchors,"provenance":{"detector_sha256":header["origin"]["detector_sha256"],
        "selected_tracker":read(SOURCE/"tracker_selection.json")["selected"],
        "historical_frozen_route":"Greedy association of SAME selected-detector detections, not preserved detector",
        "historical_adapted_route":"Development-selected damped_appearance association",
        "association_gap_limit_samples":read(SOURCE/"tracker_lock.json")["max_gap_samples"],
        "historical_queries_seconds":[20,40,60,80,100],"runtime_actor_policy":"Fixed original ID, no relink"},
        "limits":"Reference matches at40px support post-inference diagnosis, not certified real identity/physical visibility; no annotation use at runtime"}
    write_json(output/"diagnosis.json",result)
    print(json.dumps({"stage":"diagnosis","availability":availability,
        "causes":dict(sum((Counter({k:v for k,v in r.items() if isinstance(v,int) and k.isupper()}) for r in minute_rows),Counter()))}),flush=True)
    return result


def contact_sheet(output: Path, saved: list[dict], source: dict, anchors: list,
                  truth: dict, pictures: list) -> None:
    cap = cv2.VideoCapture(str(ROOT/source["video"]))
    if not cap.isOpened():
        raise OSError("Cannot read diagnostic video")
    panels = []
    try:
        for ai, index, candidate, title in pictures[:12]:
            frame = saved[index]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame["frame_index"])
            ok, image = cap.read()
            if not ok:
                raise OSError("Contact-sheet source frame failed")
            actor_detection = next((d for d in frame["detections"] if d["track"]==anchors[ai]["track"]),None)
            point = (frame["detections"][candidate]["footpoint_xy_px"] if candidate is not None
                     else actor_detection["footpoint_xy_px"] if actor_detection else anchors[ai]["footpoint_xy_px"])
            x,y = map(int,point)
            x0,y0 = max(0,x-220),max(0,y-140)
            crop=image[y0:min(image.shape[0],y0+280),x0:min(image.shape[1],x0+440)].copy()
            for d in frame["detections"]:
                px,py=d["footpoint_xy_px"]
                if x0<=px<x0+440 and y0<=py<y0+280:
                    cv2.circle(crop,(round(px-x0),round(py-y0)),4,(0,255,255),-1)
                    cv2.putText(crop,d["track"][-4:],(round(px-x0),round(py-y0)-8),0,.4,(0,255,255),1)
            panel=np.zeros((330,440,3),np.uint8)
            panel[:280]=cv2.resize(crop,(440,280))
            cv2.putText(panel,f"Actor{ai} t={frame['timestamp_s']:.1f} retrospective",(5,299),0,.45,(255,255,255),1)
            cv2.putText(panel,title[:58],(5,320),0,.4,(255,255,255),1)
            panels.append(panel)
    finally:
        cap.release()
    if panels:
        while len(panels)%3:
            panels.append(np.zeros_like(panels[0]))
        image=np.concatenate([np.concatenate(panels[i:i+3],axis=1) for i in range(0,len(panels),3)])
        if not cv2.imwrite(str(output/"first_failure_contact.jpg"),image):
            raise OSError("Contact-sheet write failed")


class SessionAssociation:
    """Global image-space association with reciprocal ambiguity abstention."""
    def __init__(self, width: int):
        self.scale = width / 4096
        self.sessions: dict[str, dict] = {}
        self.previous: float | None = None
        self.next_id = 1

    def step(self, frame: dict, features: list | None = None) -> dict:
        timestamp = frame.get("timestamp_s", frame["frame_index"]/25)
        if not math.isfinite(timestamp) or (self.previous is not None and timestamp <= self.previous):
            raise ValueError("Association requires strictly increasing finite timestamps")
        if frame.get("camera_cut") or frame.get("identity_reset"):
            self.sessions.clear()
        self.previous = timestamp
        self.sessions = {key:value for key,value in self.sessions.items() if timestamp-value["time"] <= 1.+1e-6}
        result = copy.deepcopy(frame)
        detections = result["detections"]
        keys = sorted(self.sessions)
        n,m = len(keys),len(detections)
        cost = np.full((n,m),np.inf)
        details = {}
        for i,key in enumerate(keys):
            old = self.sessions[key]
            dt = timestamp-old["time"]
            predicted = old["point"]+old["velocity"]*dt
            for j,d in enumerate(detections):
                point = np.asarray(d["footpoint_xy_px"],float)
                distance = float(np.linalg.norm(point-predicted))/self.scale
                if distance > 45*min(dt/.2,2):
                    continue
                feature = features[j] if features is not None else None
                app = None
                if feature is not None and old["feature"] is not None:
                    app = float(np.abs(np.asarray(feature)-old["feature"]).sum())/2
                    if app > .65:
                        continue
                gap = dt > .2001
                # Recovery needs both a mature motion trajectory AND appearance
                # agreement. Same kit alone cannot authorize a session relink.
                if gap and (old["observations"] < 5 or app is None or app > .15 or distance > 20):
                    continue
                cost[i,j] = distance+8*(dt/.2-1)+30*(app or 0)
                details[i,j] = {"elapsed_s":dt,"motion_residual_px":distance,
                                "appearance_distance":app,"relinked":gap}
        accepted = {}
        ambiguous = set()
        if n and m:
            augmented = np.concatenate([np.where(np.isfinite(cost),cost,1e9),np.full((n,n),100.)],axis=1)
            rows, cols = linear_sum_assignment(augmented)
            for i,j in zip(rows,cols,strict=True):
                if j>=m or not np.isfinite(cost[i,j]):
                    continue
                row_other = np.delete(cost[i],j)
                col_other = np.delete(cost[:,j],i)
                gap = details[i,j]["relinked"]
                margin = 12 if gap else 4
                if (row_other.size and row_other.min()-cost[i,j] < margin
                        or col_other.size and col_other.min()-cost[i,j] < margin):
                    ambiguous.add(int(j))
                    continue
                accepted[int(j)] = (keys[int(i)],details[i,j])
        for j,d in enumerate(detections):
            point = np.asarray(d["footpoint_xy_px"],float)
            feature = np.asarray(features[j],float) if features is not None and features[j] is not None else None
            d["tracklet_id"] = d.get("track")
            if j in accepted:
                session, detail = accepted[j]
                old = self.sessions[session]
                dt = timestamp-old["time"]
                velocity = (.35*(point-old["point"])/dt + .65*old["velocity"]).clip(-200*self.scale,200*self.scale)
                observations = old["observations"]+1
                status = "JUSTIFIED_GAP_RELINK" if detail["relinked"] else "GLOBAL_OBSERVED_ASSOCIATION"
            else:
                session = f"session-{self.next_id:06d}"
                self.next_id += 1
                velocity = np.zeros(2)
                observations = 1
                detail = {"elapsed_s":None,"relinked":False}
                status = "AMBIGUOUS_ABSTAIN_NEW_SESSION" if j in ambiguous else "NEW_OBSERVED_SESSION"
            d["track"] = session
            d["session_id"] = session
            d["association_quality"] = {"status":status,**detail,
                "uncertainty":"UNVALIDATED; no identity labels, no pitch prediction"}
            self.sessions[session] = {"point":point,"velocity":velocity,"time":timestamp,
                                     "feature":feature,"observations":observations}
        return result


def associate_sessions(saved: list[dict], features: list, width: int) -> list[dict]:
    tracker = SessionAssociation(width)
    result = [tracker.step(frame,feature) for frame,feature in zip(saved,features,strict=True)]
    same_detections(saved,result)
    return result


def extract_features(output: Path, saved: list[dict], source: dict, name: str) -> tuple[list, list[dict]]:
    """Video-only native HSV features and coarse, conservative scene-cut flags."""
    cache = output/f"features_{name}.json"
    clean_hash = __import__("hashlib").sha256(json.dumps([
        [r["frame_index"],[d["footpoint_xy_px"] for d in r["detections"]]] for r in saved],sort_keys=True).encode()).hexdigest()
    if cache.exists():
        value = read(cache)
        if value["input_hash"] != clean_hash or value["video_sha256"] != source["video_sha256"]:
            raise ValueError("Feature cache input changed")
        features,cuts = value["features"],value["cuts"]
    else:
        from src.backbone.perception_continuation import appearance_descriptor
        cap = cv2.VideoCapture(str(ROOT/source["video"]))
        if not cap.isOpened():
            raise OSError("Native feature video failed")
        features,cuts,previous,last_image = [],[],None,None
        try:
            for i,frame in enumerate(saved):
                if i%100==0:
                    budget(output)
                image = _read_frame(cap,frame["frame_index"],previous)
                previous = frame["frame_index"]
                tiny = cv2.resize(image,(128,36)).astype(np.float32)/255
                change = float(np.abs(tiny-last_image).mean()) if last_image is not None else 0
                cuts.append(change>.35)
                last_image = tiny
                features.append([appearance_descriptor(image,d["footpoint_xy_px"],source["dimensions_px"][0])
                                 for d in frame["detections"]])
        finally:
            cap.release()
        write_json(cache,{"input_hash":clean_hash,"video_sha256":source["video_sha256"],
            "features":features,"cuts":cuts,"cut_rule":"Mean absolute128x36 RGB change>.35; unvalidated automatic cut cue"})
    clean = copy.deepcopy(saved)
    for frame,cut in zip(clean,cuts,strict=True):
        frame["timestamp_s"] = frame.get("timestamp_s",frame["frame_index"]/25)
        frame["camera_cut"] = bool(cut or frame.get("camera_cut",False))
    return features,clean


def select_development(output: Path) -> dict:
    write_json(output/"session_lock.json",{
        "change":"One global reciprocal-margin actor-session association, no larger gap/distance gate",
        "max_gap_seconds":1.,"native_motion_gate_px":45,"gap_motion_residual_px":20,
        "gap_requires_observations":5,"gap_max_appearance_l1":.15,
        "reciprocal_margin_px":4,"gap_reciprocal_margin_px":12,
        "settings":"Predeclared once; no tuning on retrospective continuous window",
        "selection":"Same two-source development gate as preceding experiment; no increase in pooled false relinks",
        "session":"Stable image-space session distinct from source tracklet. No projection/label input."})
    saved = read(SOURCE/"selected_final_development_predictions.json")
    sources = source_records()
    metrics = {"baseline":{},"experimental":{}}
    relinks = {"baseline":Counter(),"experimental":Counter()}
    for match in ("117093","117092"):
        budget(output)
        features, clean = extract_features(output,saved[match],sources[match],f"dev_{match}")
        old = associate_focused(clean,sources[match]["dimensions_px"][0],"damped_appearance",features)
        new = associate_sessions(old,features,sources[match]["dimensions_px"][0])
        labels = _read_annotations(sources[match],{r["frame_index"] for r in clean})
        for method,stream in (("baseline",old),("experimental",new)):
            metrics[method][match] = aggregate_scores(stream,labels,sources[match]["dimensions_px"][0])
            relinks[method].update(identity_audit(stream,labels,sources[match]["dimensions_px"][0])["relinks"])
    chosen = (tracker_eligible(metrics["baseline"],metrics["experimental"])
              and relinks["experimental"]["false"] <= relinks["baseline"]["false"])
    result = {"selected":"experimental" if chosen else "baseline","metrics":metrics,
        "gap_relink_diagnostics":{k:dict(v) for k,v in relinks.items()},
        "selected_utc":utc(),"retrospective_used_for_selection":False,
        "negative_result":not chosen}
    write_json(output/"development_selection.json",result)
    print(json.dumps({"stage":"development_selection","selected":result["selected"]}),flush=True)
    return result


def identity_audit(saved: list[dict], labels: dict, width: int) -> dict:
    prior, counts = {},Counter()
    per_minute = {}
    for frame in saved:
        truth = labels.get(frame["frame_index"],[])
        pairs = _maximum_matches(frame["detections"],truth,40*width/4096)
        current = {i:truth[j]["track_id"] for i,j,_ in pairs}
        for j,d in enumerate(frame["detections"]):
            key = d["track"]
            identity = None
            if key in prior:
                then,identity = prior[key]
                if frame["frame_index"]-then>5:
                    counts["attempts"] += 1
                    if j in current and identity is not None:
                        counts["reference_scorable"] += 1
                        counts["false"] += int(identity!=current[j])
                        counts["consistent"] += int(identity==current[j])
                    else:
                        counts["unscorable"] += 1
            prior[key] = (frame["frame_index"],current.get(j,identity))
    start = saved[0]["frame_index"]
    for minute in range(math.ceil(len(saved)/300)):
        part = [r for r in saved if start+minute*1500<=r["frame_index"]<start+(minute+1)*1500]
        if part:
            per_minute[str(minute)] = aggregate_scores(part,labels,width)["thresholds"]["40"]
    return {"relinks":dict(counts),"per_minute":per_minute,
            "warning":"Post-inference40px one-to-one label matches; unscorable relinks not assumed correct"}


def materialize(output: Path) -> dict:
    saved,source,external,header = inputs()
    features,clean = extract_features(output,saved,source,"continuous")
    alternative = associate_sessions(clean,features,source["dimensions_px"][0])
    outputs = {}
    labels = _read_annotations(source,{f["frame_index"] for f in saved})
    for method,stream in (("baseline",clean),("experimental",alternative)):
        same_detections(saved,stream)
        path = output/f"{method}_sessions.jsonl"
        path.write_text("".join(json.dumps(f,separators=(",",":"))+"\n" for f in stream),encoding="utf-8")
        external_path = output/f"{method}_external.jsonl"
        derived_header = copy.deepcopy(header)
        derived_header["origin"].update(prediction_sha256=file_digest(path),
            parent_external_sha256=file_digest(EXTERNAL),association=method)
        # Keep identical source_id for matched synthetic visibility seeds.
        with external_path.open("w",encoding="utf-8") as file:
            file.write(json.dumps(derived_header)+"\n")
            for frame,row in zip(stream,external,strict=True):
                row = remap_external_row(frame,row)
                file.write(json.dumps(row,separators=(",",":"))+"\n")
        outputs[method] = {"predictions":str(path.relative_to(ROOT)),"prediction_sha256":file_digest(path),
            "external":str(external_path.relative_to(ROOT)),"external_sha256":file_digest(external_path),
            "metrics":aggregate_scores(stream,labels,source["dimensions_px"][0])["thresholds"]["40"],
            "audit":identity_audit(stream,labels,source["dimensions_px"][0]),
            "cuts":sum(bool(f["camera_cut"]) for f in stream),
            "ambiguity_abstentions":sum(d.get("association_quality",{}).get("status")=="AMBIGUOUS_ABSTAIN_NEW_SESSION"
                for f in stream for d in f["detections"])}
    result = {"status":"MATCHED_FIXED_DETECTIONS_SEALED","methods":outputs,
        "selected":read(output/"development_selection.json")["selected"],
        "attributes":"Exact historical per-detection projections and qualified attributes, never filled across gaps"}
    write_json(output/"association_comparison.json",result)
    return result


def remap_external_row(frame: dict, row: dict) -> dict:
    if frame["frame_index"]!=row["frame_index"]:
        raise ValueError("Identity resolution requires aligned source samples")
    result = copy.deepcopy(row)
    result["camera_cut"] = frame.get("camera_cut",False)
    for d,m in zip(frame["detections"],result["measurements"],strict=True):
        m["track"] = d["track"]
    return result


def due_queries(previous: float, current: float, every: float) -> list[float]:
    if every<=0 or not math.isfinite(every) or current<previous:
        raise ValueError("Finite positive query interval and forward clock required")
    return [k*every for k in range(max(1,math.floor(previous/every)+1),math.floor((current+1e-8)/every)+1)]


class Cohort:
    def __init__(self, slots: int = 2):
        self.slots: list[str | None] = [None]*slots
        self.last_seen: dict[str,float] = {}
        self.runs: Counter = Counter()
        self.previous: float | None = None
        self.admissions: list[dict] = []

    def step(self, frame) -> list[str | None]:
        time_s = frame.timestamp_s
        if frame.discontinuity:
            self.slots = [None]*len(self.slots)
            self.last_seen.clear()
            self.runs.clear()
        if self.previous is not None and abs(time_s-self.previous-.2)>1e-4:
            self.runs.clear()
        self.previous = time_s
        usable = {m.track for m in frame.measurements if m.team in (1,2) and m.goalkeeper is not None}
        self.runs = Counter({key:self.runs[key]+1 for key in usable})
        for key in usable:
            self.last_seen[key] = time_s
        for i,key in enumerate(self.slots):
            if key is not None and time_s-self.last_seen.get(key,-math.inf)>2.+1e-6:
                self.slots[i] = None
        for i,key in enumerate(self.slots):
            if key is None:
                candidates = sorted((k for k in usable if self.runs[k]>=11 and k not in self.slots),
                                    key=lambda k:(-self.runs[k],k))
                if candidates:
                    self.slots[i] = candidates[0]
                    self.admissions.append({"slot":i,"session":candidates[0],"timestamp_s":time_s,
                                            "past_usable_run_samples":self.runs[candidates[0]]})
        return self.slots.copy()


def contract_causes(engine: ReplayEngine, observation, row: dict) -> list[str]:
    causes = []
    history = list(engine.memory.history)
    if observation.actor is None:
        causes.append("ACTOR_NO_SUPPORTED_POSITION")
    if len(history)<11 or any(o.actor is None for o in history):
        causes.append("INSUFFICIENT_OR_DISCONTINUOUS_ACTOR_HISTORY")
    if any(m.team not in (1,2) for o in history for m in o.measurements):
        causes.append("UNKNOWN_TEAM_IN_HISTORY")
    if any(m.goalkeeper is None for o in history for m in o.measurements):
        causes.append("UNKNOWN_ROLE_IN_HISTORY")
    if len({(m.track,m.segment) for o in history for m in o.measurements})>22:
        causes.append("MORE_THAN_22_HISTORY_IDENTITIES")
    if not row["belief"]["model_contract_met"] and not causes:
        causes.append("OTHER_MODEL_CONTRACT_FAILURE")
    return causes


def replay(output: Path, every: float = 20.) -> dict:
    comparison = read(output/"association_comparison.json")
    anchors = read(output/"diagnosis.json")["anchors"]
    release = read(ROOT/"configs/backbone_release_baseline.json")
    world = ModelSession(ROOT/release["experiment"],"G1",device="cpu")
    action = ActionModelSession(ROOT/release["action_checkpoint"],device="cpu")
    results = {}
    for method,record in comparison["methods"].items():
        predictions = frames(ROOT/record["predictions"])
        adapter = ExternalObservationSource(ROOT/record["external"])
        source_frames = list(external_frames(adapter,predictions[0]["frame_index"],len(predictions)))
        fixed = []
        for a in anchors:
            index = next(i for i,f in enumerate(predictions) if f["frame_index"]==a["frame_index"])
            original = frames(ROOT/read(SOURCE/"continuous_seal.json")["path"])[index]
            j = next(j for j,d in enumerate(original["detections"]) if d["track"]==a["track"])
            fixed.append(predictions[index]["detections"][j]["track"])
        for mode in ("FIXED_ACTOR","OBSERVABLE_COHORT"):
            budget(output)
            engines,providers = {},{}
            cohort = Cohort()
            per_minute = [Counter() for _ in range(10)]
            events,session_stats = [],{}
            before_g1,before_s4 = world.world_forward_calls,action.forward_calls
            previous = -1e-6
            stream_path = output/f"replay_{method}_{mode}.jsonl"
            with stream_path.open("w",encoding="utf-8") as file:
                for i,(frame,prediction) in enumerate(zip(source_frames,predictions,strict=True)):
                    if i%100==0:
                        budget(output)
                    if frame.discontinuity:
                        for engine in engines.values():
                            engine.reset()
                        for provider in providers.values():
                            provider.reset()
                    elapsed = frame.timestamp_s-source_frames[0].timestamp_s
                    minute = min(9,int(elapsed/60))
                    counts = per_minute[minute]
                    ids = ([key if frame.index>=anchor["frame_index"] else None
                            for key,anchor in zip(fixed,anchors,strict=True)]
                           if mode=="FIXED_ACTOR" else cohort.step(frame))
                    due = due_queries(previous,elapsed,every)
                    previous = elapsed
                    for slot,key in enumerate(ids):
                        counts["slot_opportunities"] += 1
                        counts["scheduled_query_attempts"] += len(due)
                        if key is None:
                            counts["NO_COHORT_SESSION" if mode=="OBSERVABLE_COHORT" else "ANCHOR_NOT_YET_OBSERVED"] += 1
                            counts["unsupported_queries"] += len(due)
                            for q in due:
                                events.append({"slot":slot,"session":None,"scheduled_elapsed_s":q,
                                    "timestamp_s":frame.timestamp_s,"status":"unsupported","reason":"No current causal actor session"})
                            continue
                        if key not in engines:
                            engines[key] = ReplayEngine(world,expiry_s=6.,fallback="last_seen",action_model=action)
                            providers[key] = SyntheticObservationProvider(key,seed=42,fov=120,dropout=True)
                            session_stats[key] = {"first_sample_s":frame.timestamp_s,"samples":0,"available":0,
                                                  "g1_calls":0,"s4_calls":0,"query_attempts":0}
                        engine = engines[key]
                        observation = adapter.observe(frame,providers[key])
                        g1_before,s4_before = world.world_forward_calls,action.forward_calls
                        query_now = bool(due and abs(due[-1]-elapsed)<1e-4)
                        row = engine.step(observation,include_passing_assessment=False,action_query=query_now)
                        causes = contract_causes(engine,observation,row)
                        counts["actor_available_samples"] += int(observation.actor is not None)
                        counts["history_contract_ready_samples"] += int(engine.memory.current_input is not None)
                        counts["g1_calls"] += world.world_forward_calls-g1_before
                        counts["s4_calls"] += action.forward_calls-s4_before
                        counts["memory_fallback_states"] += sum(not e["visible"] and e["backend"]!="G1" for e in row["belief"]["entities"])
                        counts["g1_hidden_states"] += sum(not e["visible"] and e["backend"]=="G1" for e in row["belief"]["entities"])
                        counts.update(causes)
                        found = next((d for d in prediction["detections"] if d["track"]==key),None)
                        if found is None:
                            counts["RUNTIME_SESSION_NOT_DETECTED_IDENTITY_UNRESOLVED"] += 1
                        elif observation.actor is None:
                            counts["IMAGE_SESSION_OUTSIDE_PROJECTION"] += 1
                        session_stats[key]["samples"] += 1
                        session_stats[key]["available"] += int(observation.actor is not None)
                        session_stats[key]["g1_calls"] += world.world_forward_calls-g1_before
                        session_stats[key]["s4_calls"] += action.forward_calls-s4_before
                        session_stats[key]["query_attempts"] += len(due)
                        for q in due:
                            inference = row["action_inference"] if query_now and q==due[-1] else {
                                "status":"unsupported","reason":"No observation at supplied query timestamp"}
                            counts["supported_queries" if inference["status"]=="supported" else "unsupported_queries"] += 1
                            events.append({"slot":slot,"session":key,"scheduled_elapsed_s":q,"timestamp_s":frame.timestamp_s,**inference})
                        # Compact local stream includes genuine uncertainty, no labels.
                        file.write(json.dumps({"timestamp_s":frame.timestamp_s,"frame_index":frame.index,
                            "slot":slot,"session":key,"actor_available":observation.actor is not None,
                            "causes":causes,"belief":row["belief"],"action_inference":row["action_inference"],
                            "scheduled_queries":due},separators=(",",":"))+"\n")
            totals = sum(per_minute,Counter())
            result = {"mode":mode,"association":method,"samples":len(source_frames),"query_every_s":every,
                "fixed_anchor_sessions":fixed if mode=="FIXED_ACTOR" else None,
                "slot_opportunities":len(source_frames)*2,"totals":dict(totals),
                "g1_calls":world.world_forward_calls-before_g1,"s4_calls":action.forward_calls-before_s4,
                "per_minute":[{"minute":i,**dict(c)} for i,c in enumerate(per_minute)],
                "sessions":session_stats,"cohort_admissions":cohort.admissions if mode=="OBSERVABLE_COHORT" else [],
                "query_events":events,"runtime_stream":str(stream_path.relative_to(ROOT)),
                "limitations":"Cohort admissions are different causal sessions, NOT original actor recovery. Synthetic visibility; experimental pitch; unknown attributes preserved."}
            results[f"{method}_{mode}"] = result
            write_json(output/f"metrics_{method}_{mode}.json",result)
            print(json.dumps({"stage":"replay","method":method,"mode":mode,"g1":result["g1_calls"],
                "s4":result["s4_calls"],"available":totals["actor_available_samples"],
                "queries":totals["scheduled_query_attempts"]}),flush=True)
    write_json(output/"replay_comparison.json",{"status":"CHRONOLOGICAL_COMPARISON_COMPLETE","results":results})
    return results


def score_replay_identities(output: Path) -> dict:
    """Audit completed autonomous sessions; these labels never reach replay."""
    saved,source,_,_ = inputs()
    labels = _read_annotations(source,{f["frame_index"] for f in saved})
    anchors = read(output/"diagnosis.json")["anchors"]
    comparison = read(output/"association_comparison.json")
    result = {}
    for method,record in comparison["methods"].items():
        predictions = frames(ROOT/record["predictions"])
        matched = {}
        for frame in predictions:
            truth = labels.get(frame["frame_index"],[])
            pairs = _maximum_matches(frame["detections"],truth,40*source["dimensions_px"][0]/4096)
            matched[frame["frame_index"]] = {frame["detections"][j]["track"]:truth[k]["track_id"] for j,k,_ in pairs}
        for mode in ("FIXED_ACTOR","OBSERVABLE_COHORT"):
            metrics = read(output/f"metrics_{method}_{mode}.json")
            counters,last_identity,runs,projected_runs = Counter(),{},{},{}
            identity_sessions: dict[str,set] = {}
            visible_runs = Counter()
            previous_sample = {}
            duration_runs = []
            per_minute = [Counter() for _ in range(10)]
            fixed = metrics["fixed_anchor_sessions"]
            for row in frames(ROOT/metrics["runtime_stream"]):
                counts = per_minute[min(9,int((row["timestamp_s"]-600)/60))]
                session = row["session"]
                truth_id = matched[row["frame_index"]].get(session)
                key = (session,row["slot"]) if mode=="FIXED_ACTOR" else session
                if key in previous_sample and row["frame_index"]-previous_sample[key]!=5 and visible_runs[key]:
                    duration_runs.append(visible_runs.pop(key))
                previous_sample[key] = row["frame_index"]
                if row["actor_available"]:
                    visible_runs[key] += 1
                elif visible_runs[key]:
                    duration_runs.append(visible_runs.pop(key))
                if truth_id:
                    identity_sessions.setdefault(truth_id,set()).add(session)
                    counts["reference_matched_actor_samples"] += 1
                    continuous = key in last_identity and row["frame_index"]-last_identity[key][0]==5
                    same_identity = key in last_identity and truth_id==last_identity[key][1]
                    if continuous:
                        counts["reference_adjacent_transitions"] += 1
                        counts["identity_switches"] += not same_identity
                        runs[key] = runs.get(key,0)+1 if same_identity else 1
                    else:
                        runs[key] = 1
                        if key in last_identity:
                            counts["reference_scorable_gap_returns"] += 1
                            counts["reference_inconsistent_gap_returns"] += not same_identity
                    counts["identity_consistent_11sample_image_observations"] += int(runs[key]>=11)
                    projected_runs[key] = (projected_runs.get(key,0)+1 if continuous and same_identity else 1) if row["actor_available"] else 0
                    counts["identity_consistent_11sample_actor_observations"] += int(projected_runs[key]>=11)
                    last_identity[key] = (row["frame_index"],truth_id)
                    if mode=="FIXED_ACTOR":
                        expected = anchors[row["slot"]]["post_inference_reference"]
                        if expected:
                            counts["fixed_reference_scorable"] += 1
                            counts["not_original_prefix_reference"] += truth_id!=expected
                else:
                    runs[key] = 0
                    projected_runs[key] = 0
                    counts["unscorable_actor_samples"] += 1
            counters = sum(per_minute,Counter())
            duration_runs.extend(visible_runs.values())
            result[f"{method}_{mode}"] = {"totals":dict(counters),
                "per_minute":[{"minute":i,**dict(c)} for i,c in enumerate(per_minute)],
                "fragmentation_extra_sessions_per_reference_identity":sum(max(0,len(ids)-1) for ids in identity_sessions.values()),
                "observed_projected_actor_runs":len(duration_runs),
                "observed_projected_actor_run_median_seconds":float(np.median(duration_runs))*.2 if duration_runs else None,
                "false_relink_scope":"Reference-inconsistent same-session gap returns, not every identity switch; missing reference evidence unscorable",
                "limits":"Annotation one-to-one40px matches after runtime; prefix reference imperfect. Unscorable is not assumed correct. No labels given to cohort admission."}
    write_json(output/"post_inference_identity_scores.json",result)
    return result


def render(output: Path) -> dict:
    ffmpeg = ROOT/"data/video_review_tools/portable/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe"
    if not ffmpeg.is_file():
        raise FileNotFoundError("Existing local ffmpeg missing; no download")
    saved,source,external,_ = inputs()
    streams = {method:frames(output/f"{method}_sessions.jsonl") for method in ("baseline","experimental")}
    replay_rows = {}
    for method in streams:
        for mode in ("FIXED_ACTOR","OBSERVABLE_COHORT"):
            data = {}
            for row in frames(output/f"replay_{method}_{mode}.jsonl"):
                data.setdefault(row["frame_index"],[]).append(row)
            replay_rows[method,mode] = data
    cap = cv2.VideoCapture(str(ROOT/source["video"]))
    if not cap.isOpened():
        raise OSError("Replay rendering source unavailable")
    destination = output/"continuity_118576_10min.mp4"
    error_path = output/"render.stderr.log"
    with error_path.open("wb") as error:
        process = subprocess.Popen([str(ffmpeg),"-nostdin","-y","-f","rawvideo","-pix_fmt","bgr24",
            "-s","1280x720","-r","5","-i","pipe:0","-an","-c:v","libx264","-preset","veryfast",
            "-crf","25","-pix_fmt","yuv420p","-movflags","+faststart",str(destination)],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=error)
        if process.stdin is None:
            raise RuntimeError("Renderer input missing")
        previous = None
        trails = {m:{} for m in streams}
        latest = {}
        def text(image,value,xy,color=(235,235,235),scale=.4):
            cv2.putText(image,str(value),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
        try:
            for i,frame in enumerate(saved):
                if i%100==0:
                    budget(output)
                image = _read_frame(cap,frame["frame_index"],previous)
                previous = frame["frame_index"]
                canvas = np.zeros((720,1280,3),np.uint8)
                text(canvas,f"118576 {frame['timestamp_s']:.1f}s | Same sealed detections | LOCAL PROJECTION EXPERIMENTAL / FAILED calibration",(10,20))
                for column,method in enumerate(streams):
                    left = column*640
                    text(canvas,f"{method}: {'prior selected tracker' if column==0 else 'one experimental global session association'}",(left+8,45))
                    canvas[60:229,left:left+640] = cv2.resize(image,(640,169))
                    current = set()
                    for j,d in enumerate(streams[method][i]["detections"]):
                        key = d["track"]
                        current.add(key)
                        pt = (left+round(d["footpoint_xy_px"][0]*640/4096),
                              60+round(d["footpoint_xy_px"][1]*169/1080))
                        color = (0,220,220) if external[i]["measurements"][j]["xy"] is not None else (150,150,150)
                        history = trails[method].setdefault(key,deque(maxlen=11))
                        history.append(pt)
                        if len(history)>1:
                            cv2.polylines(canvas,[np.asarray(history,np.int32)],False,color,1)
                        cv2.circle(canvas,pt,2,color,-1)
                        text(canvas,key[-4:],pt,color,.28)
                    trails[method] = {k:v for k,v in trails[method].items() if k in current}
                    y = 250
                    for mode in ("FIXED_ACTOR","OBSERVABLE_COHORT"):
                        rows = replay_rows[method,mode].get(frame["frame_index"],[])
                        text(canvas,mode+" (separate causal memories)",(left+8,y),(0,220,255))
                        y += 22
                        for slot in (0,1):
                            row = next((r for r in rows if r["slot"]==slot),None)
                            if row is None:
                                text(canvas,f"slot{slot}: no admitted/current anchor session",(left+8,y))
                                y += 68
                                continue
                            if row["scheduled_queries"]:
                                latest[method,mode,slot] = (row["timestamp_s"],row["action_inference"])
                            text(canvas,f"slot{slot} {row['session']} present={row['actor_available']} ready={row['belief']['model_contract_met']}",(left+8,y))
                            text(canvas,"; ".join(row["causes"])[:87] or "G1 contract satisfied",(left+8,y+18),scale=.32)
                            query = latest.get((method,mode,slot))
                            if query:
                                inference = query[1]
                                output_action = (inference.get("output") or {}).get("action_type")
                                text(canvas,f"query@{query[0]:.1f}: {inference['status']} {output_action or ''}",(left+8,y+36),scale=.34)
                            text(canvas,"std_m: "+str([e["std_m"] for e in row["belief"]["entities"] if not e["visible"]][:2])[:88],(left+8,y+52),scale=.31)
                            y += 68
                        y += 8
                    text(canvas,"Gray=detected/unprojected. Trails only observed image points.",(left+8,648),scale=.35)
                    text(canvas,"Cohort switching is NOT fixed-person recovery. Unknown stays unknown.",(left+8,668),scale=.33)
                text(canvas,"Simulated FOV/dropout, not gaze. G1 hidden estimates do not feed S4. Calls are not tactical accuracy.",(10,704),scale=.43)
                if i in (0,300,1500,2999):
                    cv2.imwrite(str(output/f"replay_frame_{i:04d}.jpg"),canvas)
                process.stdin.write(canvas.tobytes())
        except BaseException:
            process.kill()
            raise
        finally:
            cap.release()
            process.stdin.close()
            process.wait(timeout=60)
    if process.returncode:
        raise RuntimeError("Local media render failed; see render.stderr.log")
    check = cv2.VideoCapture(str(destination))
    count,fps = int(check.get(cv2.CAP_PROP_FRAME_COUNT)),float(check.get(cv2.CAP_PROP_FPS))
    check.release()
    if count!=3000 or abs(fps-5)>.001:
        raise ValueError("Rendered media does not cover3000 samples at5Hz")
    result = {"path":str(destination.relative_to(ROOT)),"frames":count,"fps":fps,"seconds":count/fps,
              "sha256":file_digest(destination)}
    write_json(output/"media.json",result)
    return result


def verify(output: Path) -> dict:
    before = read(output/"protected_before.json")
    after = {path:file_digest(ROOT/path) for path in before}
    write_json(output/"protected_after.json",after)
    if before!=after:
        raise ValueError("Protected historical artifacts changed")
    saved,_,_,_ = inputs()
    comparison = read(output/"association_comparison.json")
    for record in comparison["methods"].values():
        path = ROOT/record["predictions"]
        if file_digest(path)!=record["prediction_sha256"] or file_digest(ROOT/record["external"])!=record["external_sha256"]:
            raise ValueError("Current sealed association changed")
        same_detections(saved,frames(path))
    tests = subprocess.run([sys.executable,"-m","pytest","-q","-rA",
        "tests/test_actor_continuity.py","tests/test_perception_continuation.py",
        "tests/test_video_player_cross_match_replay.py","tests/test_backbone_release.py",
        "--junitxml="+str(output/"tests.xml")],cwd=ROOT,capture_output=True,text=True,timeout=300)
    (output/"tests.txt").write_text(tests.stdout+tests.stderr,encoding="utf-8")
    if tests.returncode:
        raise RuntimeError("Focused tests failed; complete output preserved")
    result = {"status":"PASS","protected_artifacts_unchanged":True,"identical_detections":True,
        "tests":tests.stdout.strip().splitlines()[-1],"verified_utc":utc()}
    write_json(output/"verification.json",result)
    return result


def loss_summary(output: Path) -> dict:
    diagnosis = read(output/"diagnosis.json")
    saved,_,_,_ = inputs()
    trace = frames(output/"anchor_trace.jsonl")
    actors = []
    for anchor in diagnosis["anchors"]:
        relevant = [r for r in trace if r["actor"]==anchor["track"]]
        present = [f for f in saved if any(d["track"]==anchor["track"] for d in f["detections"])]
        end = present[-1]["timestamp_s"] if present else None
        after = [r for r in relevant if end is not None and r["timestamp_s"]>end]
        actors.append({"actor":anchor["track"],"last_runtime_detection_s":end,
            "last_runtime_detection_elapsed_s":end-600 if end is not None else None,
            "remaining_opportunities":len(after),
            "after_retirement_loss_causes":dict(Counter(r["cause"] for r in after)),
            "all_opportunity_causes":dict(Counter(r["cause"] for r in relevant)),
            "prefix_reference_votes":anchor["post_inference_prefix_reference_votes"],
            "exact_anchor_reference":anchor["exact_anchor_reference_match"]})
    totals = sum((Counter(a["all_opportunity_causes"]) for a in actors),Counter())
    return {"actors":actors,"all_6000_opportunities":dict(totals),
        "physical_visibility":"Reference-matched detections support visible replacements; an absent runtime ID alone never establishes physical absence.",
        "projection_vs_attributes":"Per-minute unprojected, unknown attribute and model-contract counts remain in diagnosis/replay metrics; these may overlap and are not additive causes.",
        "original_actor_identity":"Post-inference dominant prefix reference only; original runtime IDs already show identity errors and one exact anchor lacks a40px reference match."}


def package(output: Path) -> dict:
    task = read(output/"task_state.json")
    complete = all(task["stages"].get(s,{}).get("status")=="COMPLETE" for s in (
        "diagnose","select","associate","replay","score","render","verify"))
    terminal = "COMPLETE" if complete else "PARTIAL_BLOCKED"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = Path.home()/"Downloads"/f"return_continuity_{stamp}.zip"
    diagnosis = read(output/"diagnosis.json")
    comparison = read(output/"association_comparison.json") if (output/"association_comparison.json").is_file() else {}
    replay_result = read(output/"replay_comparison.json") if (output/"replay_comparison.json").is_file() else {}
    scores = read(output/"post_inference_identity_scores.json") if (output/"post_inference_identity_scores.json").is_file() else {}
    losses = loss_summary(output)
    write_json(output/"loss_causes.json",losses)
    summary = {"status":terminal,"historical_availability":diagnosis["historical_availability_per_600_opportunities"],
        "selected_method":comparison.get("selected"),"methods":{},
        "runtime_seconds":(datetime.now(timezone.utc)-datetime.fromisoformat(task["started_utc"])).total_seconds(),
        "verification":read(output/"verification.json") if (output/"verification.json").is_file() else None,
        "media":read(output/"media.json") if (output/"media.json").is_file() else None,
        "no_training":True,"finished_utc":utc(),"return_zip":str(destination),"loss_causes":losses}
    for key,value in replay_result.get("results",{}).items():
        summary["methods"][key] = {k:value[k] for k in ("g1_calls","s4_calls","totals","slot_opportunities")}
        summary["methods"][key]["post_inference_identity"] = scores.get(key,{}).get("totals")
    write_json(output/"summary.json",summary)
    command = f"& .\\.venv\\Scripts\\python.exe -m src.backbone.actor_continuity --output '{output.relative_to(ROOT)}'"
    lines = ["# Actor continuity / sustained replay",f"Status: {terminal}. No training, promotion, commits or pushes.",
        "Original availability reproduced:359/600 actor-opportunities in minute0, zero in minutes1-9. Original actors were held at fixed runtime IDs; absence alone is not physical disappearance.",
        "Post-inference40px annotation matches diagnose replacements, missed detections and unsupported projection. Prefix reference identities are imperfect; one exact anchor was not reference-matched.",
        "One experimental change: global one-to-one image-space session association with actual elapsed time, reciprocal ambiguity margins, mature-motion+appearance gap recovery. Gap and distance caps were not enlarged; color alone cannot relink. Cuts reset identities. Uncertain matches start separate sessions.",
        f"Development-selected method: {comparison.get('selected')}; both methods are reported regardless of selection. Fixed anchors map through identical detection indices before observation construction.",
        "FIXED_ACTOR never switches viewpoints. OBSERVABLE_COHORT causally admits up to two currently usable sessions after11 past samples; each session owns separate history/memory. Cohort switching is not recovery of original actors.",
        "Queries occur every20s across the full duration, including unsupported/empty slots. Unknown attributes stay unknown; unsupported pitch creates no observed position. Genuine gaps require fresh history warm-up.",
        "Global switch/fragmentation/history denominators: association_comparison.json. Fixed/cohort scores and query outputs: metrics_*.json. Identity reference audits are retrospective only. Synthetic visibility seeds depend on session IDs, so model calls are not a controlled tactical-accuracy comparison.",
        "See diagnosis.json, anchor_trace.jsonl, first_failure_contact.jpg and post_inference_identity_scores.json for failures. Local ten-minute replay media path is in media.json; footage excluded from ZIP.",
        f"Runtime: {summary['runtime_seconds']/60:.1f}min. Verification: {summary['verification']}.",
        "## Commands", "```powershell",command+" run",command+" resume",command+" verify",
        "```","Run/resume skips completed stages, never trains, and preserves absolute deadlines. Terminal task status is written BEFORE ZIP creation. Frozen/release artifacts and earlier packages are unchanged."]
    lines[3:3] = [
        "Matched retrospective loss counts across6000 opportunities: "+str(losses["all_6000_opportunities"]),
        "Last observed original runtime IDs: "+str([(a["actor"],a["last_runtime_detection_s"]) for a in losses["actors"]]),
        "| Association / mode | Available/6000 | G1 calls | S4 calls /58 supplied queries |",
        "|---|---:|---:|---:|",
        *[f"|{key}|{value['totals'].get('actor_available_samples',0)}/6000|{value['g1_calls']}|{value['s4_calls']}/58|"
          for key,value in summary["methods"].items()],
        "Experimental association was rejected on development. Fewer false relinks do NOT compensate for worse fragmentation/coverage; fixed-actor continuity was not solved."]
    (output/"handoff.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    names = ("src/backbone/actor_continuity.py","tests/test_actor_continuity.py")
    patch = []
    for name in names:
        path = ROOT/name
        target = output/"changed"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        patch.extend(difflib.unified_diff([],path.read_text().splitlines(keepends=True),fromfile="/dev/null",tofile="b/"+name))
    (output/"source.patch").write_text("".join(patch),encoding="utf-8")
    state(output,status=terminal,next_action="STOP; scoped continuity experiment finished",
          return_zip=str(destination),finished_utc=utc())
    with zipfile.ZipFile(destination,"x",zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path.suffix not in (".json",".jpg",".py",".patch",".md",".xml",".txt"):
                continue
            if path.name.startswith(("features_","replay_frame_")) or path.stat().st_size>4_000_000:
                continue
            archive.write(path,path.relative_to(output))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or json.loads(archive.read("task_state.json"))["status"]!=terminal:
            raise RuntimeError("ZIP or packaged terminal status verification failed")
    print(json.dumps({"status":terminal,"zip":str(destination),"runtime_minutes":summary["runtime_seconds"]/60}),flush=True)
    return summary


def run(output: Path, every: float) -> None:
    task = read(output/"task_state.json")
    if "query_every_s" in task and task["query_every_s"]!=every:
        raise ValueError("Query schedule is sealed; do not change it during resume")
    state(output,query_every_s=every)
    functions = {"diagnose":diagnose,"select":select_development,"associate":materialize,
                 "replay":lambda p:replay(p,every),"score":score_replay_identities,
                 "render":render,"verify":verify}
    for name,function in functions.items():
        task = read(output/"task_state.json")
        if task["stages"].get(name,{}).get("status")=="COMPLETE":
            continue
        if name!="verify":
            try:
                budget(output)
            except TimeoutError:
                state(output,status="PAUSED_RESERVE",next_action="Verify and package partial evidence")
                break
        state(output,status="RUNNING",next_action=name)
        attempts = task["stages"].get(name,{}).get("attempts",0)+1
        if attempts>3:
            raise RuntimeError(f"Two focused repairs exhausted for stage {name}")
        source_snapshot = output/"executed_source"/name
        source_snapshot.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__),source_snapshot/"actor_continuity.py")
        try:
            function(output)
            outcome = {"status":"COMPLETE","attempts":attempts,"finished_utc":utc()}
        except Exception as exc:
            (output/f"{name}_error.txt").write_text(traceback.format_exc(),encoding="utf-8")
            outcome = {"status":"BLOCKED","attempts":attempts,"error":str(exc),"finished_utc":utc()}
            task["stages"][name] = outcome
            state(output,stages=task["stages"],status="PAUSED_REVIEW",
                  next_action=f"Inspect {name}_error.txt; at most two focused repairs; completed stages stay sealed")
            # Dependent stages cannot fabricate inputs; caller inspects failures.
            raise
        task["stages"][name] = outcome
        state(output,stages=task["stages"],next_action="Advance")
    package(output)


def initialize(output: Path) -> None:
    output.mkdir(parents=True,exist_ok=False)
    start=datetime.fromisoformat("2026-10-29T23:50:56.967+00:00")
    write_json(output/"task_state.json",{"status":"RUNNING","started_utc":start.isoformat(),
        "hard_deadline_utc":(start+timedelta(hours=4)).isoformat(),
        "work_deadline_utc":(start+timedelta(hours=3,minutes=30)).isoformat(),
        "stages":{},"next_action":"diagnose","objective":"Fixed original actor continuity and separate causal observable cohort; no training"})
    pins=preservation_pins()
    pins.update({str(p.relative_to(ROOT)):file_digest(p) for p in (SOURCE/"continuous_seal.json",
        SOURCE/"detector_lock.json",SOURCE/"tracker_selection.json",SOURCE/"tracker_lock.json",
        SOURCE/"tracking_summary.json",SOURCE/"expansion.json",EXTERNAL,ROOT/read(SOURCE/"continuous_seal.json")["path"],
        ROOT/read(SOURCE/"detector_lock.json")["selected"])})
    write_json(output/"protected_before.json",pins)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--query-every",type=float,default=20.)
    parser.add_argument("command",choices=("init","diagnose","run","resume","select","associate","replay","score","render","verify","package"))
    args=parser.parse_args()
    output=args.output.resolve()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if args.command=="init":
        initialize(output)
    elif args.command in ("run","resume"):
        run(output,args.query_every)
    else:
        snapshot = output/"executed_source"/f"final_{args.command}"
        snapshot.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__),snapshot/"actor_continuity.py")
        {"diagnose":diagnose,"select":select_development,"associate":materialize,
         "replay":lambda p:replay(p,args.query_every),"score":score_replay_identities,
         "render":render,"verify":verify,"package":package}[args.command](output)
        if args.command in ("score","verify"):
            task = read(output/"task_state.json")
            task["stages"][args.command] = {**task["stages"].get(args.command,{}),
                "status":"COMPLETE","finalized_utc":utc()}
            state(output,stages=task["stages"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
