"""Split-locked GPU appearance learning and chronological identity evaluation.

All expensive work is local, resumable in bounded chunks and deadline-controlled.
No detector, G1, or S4 training; reference labels enter only fitting/scoring APIs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timedelta, timezone
import difflib
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import cv2
import ijson
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from src.backbone.amp_guard import AmpGuard
from src.backbone.actor_continuity import Cohort, due_queries, contract_causes, identity_audit
from src.backbone.perception_continuation import (
    ROOT, DEFAULT_OUTPUT, HISTORICAL, read, frames, same_detections, preservation_pins,
    associate_focused, appearance_descriptor,
)
from src.backbone.gpu_perception import tile_bounds, merged_detections, aggregate_scores
from src.backbone.video_player_cross_match import (
    _read_frame, _read_annotations, _maximum_matches, project_footpoint, load_local_projector,
)
from src.backbone.video_player_cross_match_replay import (
    ExternalObservationSource, external_frames, ModelSession, ActionModelSession,
    ReplayEngine, SyntheticObservationProvider, _attribute_model,
)
from src.trajectory_dataset import file_digest, write_json

PREDECESSOR = ROOT/"outputs/ac_20261029_195056"
PERCEPTION = ROOT/"outputs/pc_20261029_144726"
BODY = ROOT/"yolo11n.pt"
FIT_MATCHES = {"117092","117093"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def atomic_torch(path: Path, value) -> None:
    temporary = path.with_suffix(".partial")
    torch.save(value,temporary)
    temporary.replace(path)


class Run:
    def __init__(self, output: Path, config: dict):
        self.output,self.config = output,config
        self.status = read(output/"status.json")
        self.start_monotonic = time.monotonic()
        self.base_elapsed = self.status.get("elapsed_seconds",0.)
        self.original_start = datetime.fromisoformat(self.status["started_utc"])

    def elapsed(self) -> float:
        return max(self.base_elapsed+time.monotonic()-self.start_monotonic,
                   (datetime.now(timezone.utc)-self.original_start).total_seconds())

    def remaining(self, final: bool = False) -> float:
        return (self.config["max_hours"]-(0 if final else self.config["reserve_hours"]))*3600-self.elapsed()

    def check(self, final: bool = False) -> None:
        if self.remaining(final)<=0:
            raise TimeoutError("ABSOLUTE_DEADLINE" if final else "FINAL_VERIFICATION_RESERVE")

    def update(self, **values) -> None:
        self.status.update(values,elapsed_seconds=self.elapsed(),heartbeat_utc=utc(),
                           remaining_work_seconds=max(0,self.remaining()),pid=os.getpid())
        write_json(self.output/"status.json",self.status)

    def decision(self, **value) -> None:
        with (self.output/"decisions.jsonl").open("a",encoding="utf-8") as file:
            file.write(json.dumps({"utc":utc(),**value},allow_nan=False)+"\n")

    def stage(self, name: str, function, *, optional: bool = False):
        record = self.status["jobs"].get(name,{})
        if record.get("state")=="COMPLETE":
            path = self.output/record["result"]
            if not path.is_file() or file_digest(path)!=record["result_sha256"]:
                raise ValueError(f"Completed validator evidence changed: {name}")
            return read(path)
        if record.get("attempts",0)>=3:
            self.decision(stage=name,state="BLOCKED",reason="TWO_REPAIRS_EXHAUSTED")
            return None
        self.check()
        record = {"state":"RUNNING","attempts":record.get("attempts",0)+1,"started_utc":utc(),
                  "prerequisites":self.config["prerequisites"].get(name,[])}
        self.status["jobs"][name]=record
        self.update(state="RUNNING",current_job=name,next_action="Worker advances from validator outputs")
        try:
            result = function()
            path = self.output/f"{name}_result.json"
            write_json(path,result)
            record.update(state="COMPLETE",finished_utc=utc(),result=path.name,result_sha256=file_digest(path))
            self.decision(stage=name,state="COMPLETE",outcome=result.get("status"))
            return result
        except TimeoutError:
            record.update(state="DEADLINE_STOP",finished_utc=utc())
            raise
        except Exception as error:
            path = self.output/f"{name}_error_{record['attempts']}.txt"
            path.write_text(traceback.format_exc(),encoding="utf-8")
            record.update(state="BLOCKED",finished_utc=utc(),error=str(error),error_path=path.name)
            self.decision(stage=name,state="BLOCKED",error=str(error),independent_branches_continue=True)
            if not optional:
                self.status["blockers"].append({"stage":name,"error":str(error)})
            return None
        finally:
            self.update()


class ExclusiveLock:
    def __init__(self, path: Path):
        self.path=path

    def __enter__(self):
        import msvcrt
        self.file=self.path.open("a+b")
        if self.file.tell()==0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            self.file.close()
            raise RuntimeError("RUN_ALREADY_ACTIVE: exclusive worker lock held")
        return self

    def __exit__(self, *_):
        import msvcrt
        self.file.seek(0)
        msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
        self.file.close()


def scan_source(match: str, half: str) -> dict:
    annotation = ROOT/f"data/{match}_{half}.json"
    video = ROOT/f"data/video_review_v1_2/videos/{match}/{match}_panorama_{half}_half.mp4"
    with annotation.open("rb") as file:
        iterator=ijson.items(file,"images.item")
        first=next(iterator)
        image_zero=int(first["image_id"])
        count=1
        for item in iterator:
            if int(item["image_id"])!=image_zero+count:
                raise ValueError(f"Noncontiguous annotation clock: {annotation}")
            count+=1
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Missing video: {video}")
    dimensions=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    native_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps!=25 or native_count!=count or dimensions!=[int(first["width"]),int(first["height"])]:
        raise ValueError(f"VIDEO_ANNOTATION_CLOCK_MISMATCH {match} {half}: {native_count}/{count}, {dimensions}")
    return {"id":match,"half":half,"key":f"{match}_{half}","video":str(video.relative_to(ROOT)),
        "video_sha256":file_digest(video),"annotation":str(annotation.relative_to(ROOT)),
        "annotation_sha256":file_digest(annotation),"image_id_frame_zero":image_zero,
        "dimensions_px":dimensions,"fps":fps,"frames":count,"duration_seconds":count/25,
        "fitting_eligible":match in FIT_MATCHES,
        "exposure":("Existing detector fitting match; appearance temporally split" if match in FIT_MATCHES else
                    "Excluded from all local appearance/detector fitting; G1/S4/pretraining exposure not claimed unseen")}


def split_at(source: dict, second: float) -> str:
    if not source["fitting_eligible"]:
        return "evaluation"
    duration=source["duration_seconds"]
    # Existing detector dev blocks remain excluded from appearance fitting.
    exclusions = {"117093_1st":[[1620,1680]],"117092_1st":[[900,1020]]}
    for start,end in exclusions.get(source["key"],[]):
        if start-20<=second<end+20:
            return "development" if start<=second<end else "buffer"
    if duration-240<=second<duration-120:
        return "development"
    if second>=duration-260:
        return "buffer"
    return "fit"


def initialize(output: Path, config_path: Path) -> None:
    config=read(config_path)
    predecessor=read(PREDECESSOR/"task_state.json")
    if predecessor["status"]!="COMPLETE" or not (PREDECESSOR/"handoff.md").is_file():
        raise RuntimeError("Predecessor not terminal with saved handoff")
    started=utc()
    # Starts at acceptance of the full attachment, not at detached launch.
    started=config["started_utc"]
    decision={"route":"A","predecessor":str(PREDECESSOR.relative_to(ROOT)),
        "predecessor_status":predecessor["status"],
        "evidence":{"loss_causes":read(PREDECESSOR/"loss_causes.json")["all_6000_opportunities"],
                   "selected_continuity_baseline":"damped_appearance",
                   "experimental_global_session_rejected":True},
        "learning":"CONDITIONALLY_JUSTIFIED; verify conservative supervised body-crop support before fits",
        "mixed_causes":"Projection and unknown attributes remain independent; image-space evaluation never blocked by pitch.",
        "created_utc":utc()}
    lock=read(PERCEPTION/"detector_lock.json")
    detector=ROOT/lock["selected"]
    if file_digest(detector)!=lock["selected_sha256"]:
        raise ValueError("Selected detector hash changed")
    write_json(output/"route_decision.json",decision)
    pins=preservation_pins()
    for path in (detector,PREDECESSOR/"handoff.md",PREDECESSOR/"summary.json",PREDECESSOR/"loss_causes.json",
                 PERCEPTION/"detector_lock.json",BODY,output/"authorization.md"):
        pins[str(path.relative_to(ROOT))]=file_digest(path)
    write_json(output/"protected_before.json",pins)
    shutil.copy2(config_path,output/"config.json")
    write_json(output/"status.json",{"state":"IMPLEMENTING","started_utc":started,
        "deadline_utc":(datetime.fromisoformat(started)+timedelta(hours=config["max_hours"])).isoformat(),
        "reserve_utc":(datetime.fromisoformat(started)+timedelta(hours=config["max_hours"]-config["reserve_hours"])).isoformat(),
        "jobs":{},"blockers":[],"elapsed_seconds":0.,"authorization_sha256":file_digest(output/"authorization.md"),
        "predecessor":decision["predecessor"],"detector":str(detector.relative_to(ROOT)),
        "detector_sha256":lock["selected_sha256"],"next_action":"Pilot, lock corpus/source manifest, then standalone queue"})


def source_manifest(run: Run) -> dict:
    sources,excluded=[],[]
    for match,half in run.config["sources"]:
        run.check()
        try:
            source=scan_source(match,half)
            sources.append(source)
        except (ValueError,OSError) as exc:
            excluded.append({"match":match,"half":half,"reason":str(exc)})
    manifest={"status":"SPLITS_LOCKED","sources":sources,"excluded":excluded,"created_utc":utc(),
        "fitting_matches":sorted(FIT_MATCHES),"identity_namespace":"recording/half/verified annotation track_id",
        "fit_interval_policy":"Whole eligible halves excluding contiguous dev and20s buffers; no random adjacent-frame split",
        "reserved":"All nonfitting matches excluded from encoder and tracker selection",
        "unique_available_source_minutes":sum(s["duration_seconds"] for s in sources)/60,
        "module_exposure":"118576 and128057 already retrospective; other sources reserved for this perception task, G1/S4/pretraining exposure disclosed unknown unless existing provenance confirms"}
    write_json(run.output/"source_manifest.json",manifest)
    if not all(any(s["key"]==f"{m}_{h}" for s in sources) for m,h in run.config["core_sources"]):
        raise ValueError("Core source clock validation failed; inspect source_manifest.json exclusions")
    pins=read(run.output/"protected_before.json")
    pins.update({s["annotation"]:s["annotation_sha256"] for s in sources})
    write_json(run.output/"protected_before.json",pins)
    return manifest


def body_detections(model, image: np.ndarray) -> list[dict]:
    bounds=tile_bounds(image.shape[1],image.shape[0],scheme="consistent")
    return merged_detections(model,[image[t:b,l:r] for l,t,r,b in bounds],bounds)


def conservative_pairs(bodies: list[dict], labels: list[dict], width: int) -> tuple[list, Counter]:
    accepted,counts=[],Counter()
    if not bodies or not labels:
        counts["no_body_or_label"]+=len(labels)
        return accepted,counts
    distance=np.asarray([[math.dist(b["footpoint_xy_px"],a["footpoint_xy_px"]) for a in labels] for b in bodies])
    for bi,body in enumerate(bodies):
        j=int(distance[bi].argmin())
        x1,y1,x2,y2=body["bbox_xyxy_px"]
        counts["body_candidates"]+=1
        if not (x2-x1>=10 and y2-y1>=28 and y2-y1>=1.15*(x2-x1)):
            counts["small_or_not_body"]+=1
            continue
        if distance[bi,j]>30*width/4096 or int(distance[:,j].argmin())!=bi:
            counts["distance_or_not_reciprocal"]+=1
            continue
        alternative_label=np.delete(distance[bi],j)
        alternative_body=np.delete(distance[:,j],bi)
        if (alternative_label.size and alternative_label.min()-distance[bi,j]<8*width/4096
                or alternative_body.size and alternative_body.min()-distance[bi,j]<8*width/4096):
            counts["ambiguous_footpoint"]+=1
            continue
        overlap=False
        for k,other in enumerate(bodies):
            if k==bi:
                continue
            a,b,c,d=other["bbox_xyxy_px"]
            intersection=max(0,min(c,x2)-max(a,x1))*max(0,min(d,y2)-max(b,y1))
            if intersection/max(1,(x2-x1)*(y2-y1))>.35:
                overlap=True
        if overlap:
            counts["overlapping_bodies"]+=1
            continue
        if not labels[j]["track_id"] or labels[j]["track_id"]=="None":
            counts["missing_identity"]+=1
            continue
        accepted.append((bi,j,float(distance[bi,j])))
        counts["accepted"]+=1
    return accepted,counts


class AppearanceEncoder(nn.Module):
    def __init__(self, classes: int = 0):
        super().__init__()
        native=YOLO(str(BODY)).model
        self.backbone=nn.Sequential(*list(native.model.children())[:10])
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        self.classifier=nn.Linear(256,classes) if classes else None

    def forward(self, image):
        embedding=F.normalize(self.backbone(image).mean((-2,-1)),dim=1)
        return (embedding,self.classifier(embedding)) if self.classifier is not None else embedding


def image_tensor(crop: np.ndarray) -> torch.Tensor:
    resized=cv2.resize(crop,(64,128))
    return torch.from_numpy(np.ascontiguousarray(resized[:,:,::-1])).permute(2,0,1).float()/255


def pilot(run: Run) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED; no CPU fallback")
    manifest=read(run.output/"source_manifest.json")
    source=next(s for s in manifest["sources"] if s["key"]=="117093_1st")
    cap=cv2.VideoCapture(str(ROOT/source["video"]))
    model=YOLO(str(BODY))
    rates=[]
    counts=Counter()
    crops=[]
    indices=set(range(1500*25,1512*25,25))
    labels=_read_annotations(source,indices)
    previous=None
    try:
        start=time.monotonic()
        for index in sorted(indices):
            image=_read_frame(cap,index,previous)
            previous=index
            bodies=body_detections(model,image)
            pairs,counter=conservative_pairs(bodies,labels.get(index,[]),source["dimensions_px"][0])
            counts.update(counter)
            for bi,j,distance in pairs:
                x1,y1,x2,y2=map(int,bodies[bi]["bbox_xyxy_px"])
                crop=image[max(0,y1):min(image.shape[0],y2),max(0,x1):min(image.shape[1],x2)]
                if crop.size:
                    crops.append(image_tensor(crop))
        extraction_s=time.monotonic()-start
    finally:
        cap.release()
    if len(crops)<8:
        raise RuntimeError("LABELS_UNAVAILABLE: pilot has fewer than8 conservative verified body crops")
    torch.manual_seed(42)
    encoder=AppearanceEncoder(4).cuda().train()
    optimizer=torch.optim.AdamW(encoder.parameters(),lr=3e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda",init_scale=128.)
    guard=AmpGuard()
    batch=torch.stack((crops*math.ceil(64/len(crops)))[:64]).cuda()
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    for step in range(12):
        with torch.autocast("cuda",dtype=torch.float16):
            embedding,logits=encoder(batch)
            loss=F.cross_entropy(logits,torch.arange(64,device="cuda")%4)
        guard.note_minibatch(loss,scaler=scaler)
        scaler.scale(loss).backward()
        guard.update(encoder,optimizer,scaler,loss,context={"purpose":"disposable real appearance model pilot"})
    training_s=time.monotonic()-start
    before=[p.detach().clone() for p in encoder.parameters()]
    # Real checkpoint restore + next update parity is tested separately below.
    atomic_torch(run.output/"pilot_disposable.pt",{"model":encoder.state_dict(),"optimizer":optimizer.state_dict(),
        "scaler":scaler.state_dict(),"guard":guard.state_dict()})
    result={"status":"REAL_CUDA_PILOT_PASS","body_seconds_per_source_frame":extraction_s/12,
        "conservative_crops":len(crops),"exclusions":dict(counts),
        "train_seconds_per64_crop_update":training_s/12,"actual_updates":guard.state_dict(),
        "batch":64,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"gpu":torch.cuda.get_device_name(0),
        "encoder":"COCO-pretrained YOLO11n compact backbone0..9, global pool256; NOT pretrained person-ReID",
        "pretraining":"Local official COCO detector yolo11n.pt; broad COCO exposure, football overlap unknown",
        "initialization_sha256":file_digest(BODY),"input_hw":[128,64]}
    unique_fit_seconds=sum(s["duration_seconds"] for s in manifest["sources"] if s["fitting_eligible"])
    estimated_crops=unique_fit_seconds*len(crops)/12
    extraction_hours=unique_fit_seconds*extraction_s/12/3600
    training_hours=min(run.config["fit_max_hours"],estimated_crops/64*run.config["epoch_cap"]*training_s/12/3600)
    core_seconds=sum(s["duration_seconds"] for s in manifest["sources"] if [s["id"],s["half"]] in run.config["core_sources"])
    forecast={"status":"MEASURED_QUEUE_FORECAST","corpus_expected_crops":int(estimated_crops),
        "corpus_estimated_hours":extraction_hours,"two_fit_estimated_hours":training_hours*2,
        "core_detection_estimated_hours":core_seconds*5*extraction_s/12/3600,
        "available_work_hours":run.remaining()/3600,
        "uncertainty":"Pilot12source seconds includes setup. Actual convergence/validity may shorten fits; no minimum runtime or duplicate crop padding.",
        "priority":"Core4halves, then distinct remaining eligible evaluation halves, then one quality stress sweep; deadline always wins",
        "target_hours":[12,16],"planned_fit_max_hours_each":run.config["fit_max_hours"]}
    write_json(run.output/"queue_forecast.json",forecast)
    del encoder,optimizer,scaler,model,batch,before
    torch.cuda.empty_cache()
    run.update(pilot=result,forecast=forecast)
    return result


def crop_corpus(run: Run) -> dict:
    sources=[s for s in read(run.output/"source_manifest.json")["sources"] if s["fitting_eligible"]]
    model=YOLO(str(BODY))
    chunks=run.output/"corpus"
    chunks.mkdir(exist_ok=True)
    counters=Counter()
    all_records=[]
    seen=set()
    audit_count=0
    for source in sources:
        run.check()
        indices=set(range(0,source["frames"],25))
        labels=_read_annotations(source,indices)
        cap=cv2.VideoCapture(str(ROOT/source["video"]))
        if not cap.isOpened():
            raise OSError("Corpus video unavailable")
        previous=None
        try:
            for start in range(0,math.ceil(source["duration_seconds"]),60):
                run.check()
                path=chunks/f"{source['key']}_{start:04d}.json"
                if path.exists():
                    report=read(path)
                    for row in report["records"]:
                        if not (run.output/row["crop"]).is_file():
                            raise ValueError("Completed crop chunk missing image")
                        seen.add(row["pixel_sha256"])
                    all_records.extend(report["records"])
                    counters.update(report["counts"])
                    continue
                rows=[]
                count=Counter()
                for second in range(start,min(start+60,math.ceil(source["duration_seconds"]))):
                    index=second*25
                    if index>=source["frames"]:
                        break
                    split=split_at(source,second)
                    if split=="buffer":
                        count["buffer_source_frames"]+=1
                        continue
                    image=_read_frame(cap,index,previous)
                    previous=index
                    bodies=body_detections(model,image)
                    pairs,stats=conservative_pairs(bodies,labels.get(index,[]),source["dimensions_px"][0])
                    count.update(stats)
                    count["unique_source_frames"]+=1
                    for bi,j,distance in pairs:
                        annotation=labels[index][j]
                        x1,y1,x2,y2=bodies[bi]["bbox_xyxy_px"]
                        left,top,right,bottom=max(0,int(x1)),max(0,int(y1)),min(image.shape[1],int(x2)),min(image.shape[0],int(y2))
                        crop=image[top:bottom,left:right]
                        if crop.size==0:
                            count["empty_crop"]+=1
                            continue
                        crop=cv2.resize(crop,(64,128))
                        digest=hashlib.sha256(crop.tobytes()).hexdigest()
                        if digest in seen:
                            count["duplicate_pixels_rejected"]+=1
                            continue
                        identity=f"{source['key']}:{annotation['track_id']}"
                        name=Path("crops")/source["key"]/f"f{index:07d}_{j:02d}.jpg"
                        target=run.output/name
                        target.parent.mkdir(parents=True,exist_ok=True)
                        if not cv2.imwrite(str(target),crop,[cv2.IMWRITE_JPEG_QUALITY,92]):
                            raise OSError("Crop write failed")
                        seen.add(digest)
                        rows.append({"crop":str(name),"identity":identity,"track_id":annotation["track_id"],
                            "source":source["key"],"frame_index":index,"timestamp_s":second,
                            "box":bodies[bi]["bbox_xyxy_px"],"annotation_footpoint":annotation["footpoint_xy_px"],
                            "body_to_annotation_distance_px":distance,"team":annotation["team"],
                            "role":annotation["role"],"native_height":bottom-top,"native_width":right-left,
                            "split":split,"pixel_sha256":digest,"crop_sha256":file_digest(target)})
                        if audit_count<16:
                            audit=run.output/"audit"
                            audit.mkdir(exist_ok=True)
                            panel=cv2.resize(crop,(128,256))
                            cv2.putText(panel,f"{identity[-15:]} {distance:.1f}px",(2,15),0,.35,(0,255,255),1)
                            cv2.imwrite(str(audit/f"accepted_{audit_count:02d}.jpg"),panel)
                            audit_count+=1
                    if audit_count<20 and count["ambiguous_footpoint"]>0:
                        audit=run.output/"audit"
                        audit.mkdir(exist_ok=True)
                        panel=cv2.resize(image,(1024,round(1024*image.shape[0]/image.shape[1])))
                        cv2.putText(panel,"Rejected ambiguity exists in this source frame; no identity assigned",(5,20),0,.5,(0,255,255),1)
                        cv2.imwrite(str(audit/f"rejected_source_{audit_count:02d}.jpg"),panel)
                        audit_count+=1
                write_json(path,{"source":source["key"],"start_s":start,"end_s":start+60,
                    "counts":dict(count),"records":rows,"body_sha256":file_digest(BODY)})
                all_records.extend(rows)
                counters.update(count)
                run.update(current_job="corpus",progress={"source":source["key"],"seconds_complete":start+60,
                    "stored_crops":len(all_records),"counts":dict(counters)})
        finally:
            cap.release()
    identity_counts=Counter(r["identity"] for r in all_records if r["split"]=="fit")
    eligible={key for key,count in identity_counts.items() if count>=run.config["minimum_crops_per_identity"]}
    filtered=[r for r in all_records if r["identity"] in eligible]
    counters["insufficient_identity_support_rejected"]=len(all_records)-len(filtered)
    manifest=run.output/"corpus.jsonl"
    manifest.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in filtered),encoding="utf-8")
    split_counts=Counter(r["split"] for r in filtered)
    total_minutes=sum(s["duration_seconds"] for s in sources)/60
    result={"status":"CORPUS_VALID" if len(eligible)>=16 and split_counts["fit"]>=2000 else "LABELS_UNAVAILABLE",
        "manifest_sha256":file_digest(manifest),"unique_crops":len(filtered),"unique_source_frames":counters["unique_source_frames"],
        "distinct_source_minutes":total_minutes,"sources":[s["key"] for s in sources],
        "split_counts":dict(split_counts),"identity_counts":dict(Counter(r["identity"] for r in filtered)),
        "team_counts":dict(Counter(r["team"] for r in filtered)),"exclusions":dict(counters),
        "desired_crops":[80000,200000],"shortage":max(0,80000-len(filtered)),
        "supervision":"Verified annotation IDs only, namespaced by half; reciprocal30px body/footpoint match,8px ambiguity margin, overlap and minimum body-size rejection. Never footpoint boxes as person crops.",
        "limitations":"Annotation truth and image-space compatibility checked automatically; audit images supplied, not a new manually certified identity dataset. Track annotation errors remain possible."}
    write_json(run.output/"corpus_summary.json",result)
    del model
    torch.cuda.empty_cache()
    return result


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, output: Path, records: list[dict], classes: dict, augment: str | None):
        self.output,self.records,self.classes,self.augment=output,records,classes,augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        row=self.records[index]
        image=cv2.imread(str(self.output/row["crop"]))
        if image is None:
            raise OSError("Missing corpus crop")
        if self.augment:
            h,w=image.shape[:2]
            margin=random.randrange(0,4)
            image=cv2.resize(image[margin:h-margin or h,margin:w-margin or w],(64,128))
            if random.random()<.5:
                image=image[:,::-1].copy()
            image=np.clip(image.astype(np.float32)*random.uniform(.85,1.15),0,255).astype(np.uint8)
            if self.augment=="R2" and random.random()<.4:
                y,x=random.randrange(0,96),random.randrange(0,48)
                image[y:y+24,x:x+12]=np.mean(image,axis=(0,1))
        return image_tensor(image),self.classes[row["identity"]],min(1.,row["native_height"]/64)


class IdentityBatches(torch.utils.data.Sampler):
    def __init__(self, records: list[dict], *, epoch: int, hard: bool, seed: int = 42):
        self.by_id=defaultdict(list)
        self.groups=defaultdict(list)
        for i,row in enumerate(records):
            self.by_id[row["identity"]].append(i)
        for identity,indices in self.by_id.items():
            r=records[indices[0]]
            self.groups[(r["source"],r["team"])].append(identity)
        self.ids=sorted(self.by_id)
        self.steps=max(1,len(records)//64)
        self.epoch,self.hard,self.seed=epoch,hard,seed

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng=random.Random(self.seed+self.epoch)
        for _ in range(self.steps):
            chosen=[]
            if self.hard:
                candidates=[v for v in self.groups.values() if len(v)>=4]
                if candidates:
                    group=rng.choice(candidates)
                    chosen=rng.sample(group,min(8,len(group)))
            remaining=[key for key in self.ids if key not in chosen]
            chosen+=rng.sample(remaining,min(16-len(chosen),len(remaining)))
            batch=[]
            for identity in chosen:
                values=self.by_id[identity]
                batch+=rng.sample(values,4) if len(values)>=4 else rng.choices(values,k=4)
            yield batch


def metric_loss(embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    distances=1-embedding.float()@embedding.float().T
    same=labels[:,None]==labels[None,:]
    same.fill_diagonal_(False)
    different=labels[:,None]!=labels[None,:]
    valid=same.any(1)&different.any(1)
    if not valid.any():
        raise ValueError("Metric batch needs same-identity positives and different identities")
    positive=distances.masked_fill(~same,-1e4).max(1).values
    negative=distances.masked_fill(~different,1e4).min(1).values
    return F.relu(positive[valid]-negative[valid]+.2).mean()


def retrieval(run: Run, model: nn.Module, records: list[dict], classes: dict) -> dict:
    rows=[r for r in records if r["split"]=="development"]
    groups=defaultdict(list)
    for row in rows:
        groups[row["identity"]].append(row)
    query,gallery=[],[]
    for identity,values in sorted(groups.items()):
        values=sorted(values,key=lambda r:r["timestamp_s"])
        if len(values)<4 or values[-1]["timestamp_s"]-values[0]["timestamp_s"]<10:
            continue
        query+=values[:min(8,len(values)//3)]
        gallery+=values[-min(16,len(values)//3):]
    if not query or not gallery:
        raise ValueError("No temporally disjoint retrieval query/gallery")
    def encode(values):
        features=[]
        model.eval()
        with torch.inference_mode():
            for start in range(0,len(values),128):
                images=torch.stack([CropDataset(run.output,values,classes,None)[i][0]
                                    for i in range(start,min(start+128,len(values)))]).cuda()
                with torch.autocast("cuda",dtype=torch.float16):
                    output=model(images)
                features.append((output[0] if isinstance(output,tuple) else output).float().cpu().numpy())
        return np.concatenate(features)
    q,g=encode(query),encode(gallery)
    similarity=q@g.T
    same_kit_correct,same_kit_n=0,0
    correct=0
    margins=[]
    for i,row in enumerate(query):
        eligible=[j for j,other in enumerate(gallery) if row["source"]==other["source"]]
        if not eligible:
            continue
        best=max(eligible,key=lambda j:similarity[i,j])
        correct+=row["identity"]==gallery[best]["identity"]
        kit=[j for j in eligible if gallery[j]["team"]==row["team"]]
        if kit:
            same_kit_n+=1
            k=max(kit,key=lambda j:similarity[i,j])
            same_kit_correct+=row["identity"]==gallery[k]["identity"]
            positives=[similarity[i,j] for j in kit if gallery[j]["identity"]==row["identity"]]
            negatives=[similarity[i,j] for j in kit if gallery[j]["identity"]!=row["identity"]]
            if positives and negatives:
                margins.append(float(max(positives)-max(negatives)))
    return {"queries":len(query),"gallery":len(gallery),"top1_correct":int(correct),"top1":correct/len(query),
        "same_kit_queries":same_kit_n,"same_kit_correct":int(same_kit_correct),
        "same_kit_top1":same_kit_correct/same_kit_n if same_kit_n else None,
        "positive_minus_hard_samekit_margin_median":float(np.median(margins)) if margins else None,
        "strict_rejection_correct_count":sum(m>.1 for m in margins),"rejection_margin_denominator":len(margins),
        "separation":"Early versus late crops,>=10s same identity; no duplicated source frame; development only"}


def fit_encoder(run: Run, recipe: str) -> dict:
    corpus=read(run.output/"corpus_summary.json")
    if corpus["status"]!="CORPUS_VALID":
        return {"status":"LABELS_UNAVAILABLE","fit":recipe}
    records=frames(run.output/"corpus.jsonl")
    classes={key:i for i,key in enumerate(sorted({r["identity"] for r in records}))}
    train=[r for r in records if r["split"]=="fit"]
    directory=run.output/recipe
    directory.mkdir(exist_ok=True)
    checkpoint=directory/"last.pt"
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    model=AppearanceEncoder(len(classes)).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda",init_scale=128.)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=run.config["epoch_cap"])
    events=directory/"numerical_events.jsonl"
    def emit(event):
        with events.open("a",encoding="utf-8") as file:
            file.write(json.dumps(event,allow_nan=False)+"\n")
    guard=AmpGuard(event_sink=emit)
    epoch,best,bad,training_elapsed=0,-1.,0,0.
    initial_retrieval=None
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location="cuda",weights_only=False)
        if saved["corpus_sha256"]!=corpus["manifest_sha256"] or saved["recipe"]!=recipe:
            raise ValueError("Training resume manifest/recipe mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        scheduler.load_state_dict(saved["scheduler"])
        guard.load_state_dict(saved["guard"])
        epoch,best,bad,training_elapsed=saved["epoch"]+1,saved["best"],saved["bad"],saved["training_elapsed"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
    else:
        initial_retrieval=retrieval(run,model,records,classes)
        write_json(directory/"initial_retrieval.json",initial_retrieval)
    started=time.monotonic()
    forecast=read(run.output/"queue_forecast.json")
    # Reserve measured full-half detection and feature extraction before optional fits.
    evaluation_reserve=forecast["core_detection_estimated_hours"]*1.6*3600
    fit_budget=min(run.config["fit_max_hours"]*3600,
                   max(0.,run.remaining()-evaluation_reserve)/2)
    write_json(directory/"budget.json",{"remaining_work_seconds":run.remaining(),
        "evaluation_reserve_seconds":evaluation_reserve,"fit_budget_seconds":fit_budget,
        "policy":"At most configured fit cap; measured core evaluation has priority over training."})
    stop="EPOCH_CAP"
    epochs=[]
    try:
        for current_epoch in range(epoch,run.config["epoch_cap"]):
            run.check()
            if training_elapsed+time.monotonic()-started>=fit_budget:
                stop="FIT_DEADLINE"
                break
            dataset=CropDataset(run.output,train,classes,recipe)
            batches=IdentityBatches(train,epoch=current_epoch,hard=recipe=="R2")
            generator=torch.Generator().manual_seed(42+current_epoch)
            loader=torch.utils.data.DataLoader(dataset,batch_sampler=batches,
                num_workers=2,pin_memory=True,generator=generator,prefetch_factor=2)
            model.train()
            losses=[]
            for batch_index,(images,labels,quality) in enumerate(loader):
                run.check()
                if training_elapsed+time.monotonic()-started>=fit_budget:
                    raise TimeoutError("FIT_MAX_HOURS")
                images,labels,quality=images.cuda(non_blocking=True),labels.cuda(non_blocking=True),quality.cuda(non_blocking=True)
                with torch.autocast("cuda",dtype=torch.float16):
                    embedding,logits=model(images)
                    classification=F.cross_entropy(logits,labels,reduction="none")
                    loss=(classification*quality).mean() if recipe=="R2" else classification.mean()
                    loss=loss+metric_loss(embedding,labels)
                guard.note_minibatch(loss,context={"recipe":recipe,"epoch":current_epoch,"batch":batch_index},scaler=scaler)
                scaler.scale(loss).backward()
                guard.update(model,optimizer,scaler,loss,context={"recipe":recipe,"epoch":current_epoch,"batch":batch_index})
                losses.append(float(loss.detach()))
                if batch_index%100==0:
                    run.update(current_job=recipe,progress={"epoch":current_epoch,"batch":batch_index,
                        "counters":guard.state_dict()["counters"],"scale":scaler.get_scale(),
                        "gpu_peak_bytes":torch.cuda.max_memory_allocated()})
            scheduler.step()
            if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()
                       if value.is_floating_point()):
                raise FloatingPointError("Nonfinite appearance model state including running buffers")
            score=retrieval(run,model,records,classes)
            metric=score["same_kit_top1"]
            if metric is None:
                raise ValueError("No same-kit retrieval denominator")
            improved=metric>best+.001
            best,bad=(metric,0) if improved else (best,bad+1)
            record={"epoch":current_epoch,"mean_loss":float(np.mean(losses)),"retrieval":score,
                    "counters":guard.state_dict()["counters"]}
            epochs.append(record)
            elapsed=training_elapsed+time.monotonic()-started
            saved={"recipe":recipe,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"guard":guard.state_dict(),
                "epoch":current_epoch,"best":best,"bad":bad,"classes":classes,"corpus_sha256":corpus["manifest_sha256"],
                "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all(),"sampler":{"seed":42,"next_epoch":current_epoch+1},
                "training_elapsed":elapsed,"resume_limitation":"Exact completed-epoch boundary; interrupted partial epoch discarded, not exact mid-batch parity"}
            atomic_torch(checkpoint,saved)
            if improved:
                atomic_torch(directory/"best.pt",saved)
            write_json(directory/"progress.json",{"epochs":epochs,"best_samekit_top1":best,"elapsed_seconds":elapsed})
            if bad>=run.config["patience"]:
                stop="DEVELOPMENT_EARLY_STOP"
                break
    except TimeoutError as error:
        stop=str(error)
    except Exception:
        (directory/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        stop="NUMERICAL_OR_TRAINING_FAILURE_RETAIN_BASELINE"
    result={"status":"FINITE_FIT" if (directory/"best.pt").is_file() and "FAILURE" not in stop else "TRAINING_BRANCH_BLOCKED",
        "recipe":recipe,"stop_reason":stop,"numerics":guard.state_dict(),"epochs_this_process":len(epochs),
        "elapsed_seconds":training_elapsed+time.monotonic()-started,
        "best":str((directory/"best.pt").relative_to(ROOT)) if (directory/"best.pt").exists() else None,
        "best_samekit_top1":best,"presentations":guard.counters.minibatches*64,
        "same_initialization_seed":42,"actual_device":"cuda:0"}
    write_json(directory/"summary.json",result)
    del model,optimizer,scaler
    torch.cuda.empty_cache()
    return result


def load_encoder(path: Path | None) -> nn.Module:
    if path is None:
        return AppearanceEncoder().cuda().eval()
    value=torch.load(path,map_location="cpu",weights_only=False)
    model=AppearanceEncoder(len(value["classes"]))
    model.load_state_dict(value["model"])
    model.classifier=None
    return model.cuda().eval()


def available_encoders(run: Run) -> dict:
    result={"pretrained":None}
    for recipe in ("R1","R2"):
        summary=run.output/recipe/"summary.json"
        if summary.exists() and read(summary)["status"]=="FINITE_FIT":
            result[recipe]=ROOT/read(summary)["best"]
    return result


def crop_assignment(detections: list[dict], bodies: list[dict], width: int) -> list:
    """Autonomous reciprocal body association; no evaluation annotations."""
    result=[None]*len(detections)
    if not detections or not bodies:
        return result
    distance=np.asarray([[math.dist(d["footpoint_xy_px"],b["footpoint_xy_px"]) for b in bodies] for d in detections])
    for i,d in enumerate(detections):
        j=int(distance[i].argmin())
        alternative=np.delete(distance[i],j)
        competing=np.delete(distance[:,j],i)
        if (distance[i,j]<=40*width/4096 and int(distance[:,j].argmin())==i
                and (not alternative.size or alternative.min()-distance[i,j]>=8*width/4096)
                and (not competing.size or competing.min()-distance[i,j]>=8*width/4096)):
            result[i]=bodies[j]["bbox_xyxy_px"]
    return result


def encode_crops(model: nn.Module, image: np.ndarray, boxes: list) -> list:
    indices,tensors=[],[]
    result=[None]*len(boxes)
    for i,box in enumerate(boxes):
        if box is None:
            continue
        x1,y1,x2,y2=box
        crop=image[max(0,int(y1)):min(image.shape[0],int(y2)),max(0,int(x1)):min(image.shape[1],int(x2))]
        if crop.size and crop.shape[0]>=20 and crop.shape[1]>=8:
            indices.append(i)
            tensors.append(image_tensor(crop))
    with torch.inference_mode():
        for start in range(0,len(indices),128):
            batch=torch.stack(tensors[start:start+128]).cuda()
            with torch.autocast("cuda",dtype=torch.float16):
                feature=model(batch)
            if isinstance(feature,tuple):
                feature=feature[0]
            for index,value in zip(indices[start:start+128],feature.float().cpu().numpy(),strict=True):
                result[index]=value.tolist()
    return result


class LearnedAssociation:
    def __init__(self,width: int, maximum_cosine: float):
        self.scale=width/4096
        self.maximum_cosine=maximum_cosine
        self.tracks={}
        self.next_id=1
        self.previous=None

    def step(self, frame: dict, features: list) -> dict:
        timestamp=frame.get("timestamp_s",frame["frame_index"]/25)
        if not math.isfinite(timestamp) or self.previous is not None and timestamp<=self.previous:
            raise ValueError("Association timestamp must strictly increase")
        if frame.get("camera_cut") or frame.get("identity_reset"):
            self.tracks.clear()
        self.previous=timestamp
        self.tracks={k:v for k,v in self.tracks.items() if timestamp-v["time"]<=1.+1e-6}
        result=copy.deepcopy(frame)
        detections=result["detections"]
        keys=sorted(self.tracks)
        cost=np.full((len(keys),len(detections)),np.inf)
        details={}
        for i,key in enumerate(keys):
            track=self.tracks[key]
            dt=timestamp-track["time"]
            predicted=track["point"]+track["velocity"]*dt
            for j,d in enumerate(detections):
                distance=float(np.linalg.norm(np.asarray(d["footpoint_xy_px"])-predicted))/self.scale
                if distance>45*min(dt/.2,2):
                    continue
                cosine=None
                if features[j] is not None and track["embedding"] is not None:
                    cosine=max(0.,1-float(np.dot(features[j],track["embedding"])))
                    if cosine>self.maximum_cosine:
                        continue
                gap=dt>.2001
                if gap and (cosine is None or cosine>min(.15,self.maximum_cosine) or distance>20 or track["observations"]<5):
                    continue
                cost[i,j]=distance+30*(cosine or 0)+8*(dt/.2-1)
                details[i,j]={"gap":gap,"elapsed_s":dt,"motion_residual_px":distance,"cosine_distance":cosine}
        accepted={}
        ambiguous=set()
        if cost.size:
            a=np.concatenate([np.where(np.isfinite(cost),cost,1e9),np.full((len(keys),len(keys)),100.)],1)
            rows,cols=linear_sum_assignment(a)
            for i,j in zip(rows,cols,strict=True):
                if j>=len(detections) or not np.isfinite(cost[i,j]):
                    continue
                others_row=np.delete(cost[i],j)
                others_col=np.delete(cost[:,j],i)
                margin=8 if details[i,j]["gap"] else 2
                if (others_row.size and others_row.min()-cost[i,j]<margin
                        or others_col.size and others_col.min()-cost[i,j]<margin):
                    ambiguous.add(int(j))
                else:
                    accepted[int(j)]=(keys[int(i)],details[i,j])
        for j,d in enumerate(detections):
            point=np.asarray(d["footpoint_xy_px"],float)
            d["tracklet_id"]=d.get("track")
            if j in accepted:
                key,detail=accepted[j]
                track=self.tracks[key]
                velocity=(.35*(point-track["point"])/detail["elapsed_s"]+.65*track["velocity"]).clip(-200*self.scale,200*self.scale)
                observations=track["observations"]+1
                quality={"status":"APPEARANCE_GAP_RELINK" if detail["gap"] else "APPEARANCE_ASSOCIATED",**detail}
            else:
                key=f"appearance-session-{self.next_id:06d}"
                self.next_id+=1
                velocity=np.zeros(2)
                observations=1
                quality={"status":"AMBIGUOUS_ABSTAIN" if j in ambiguous else "NEW_OBSERVED_SESSION"}
            d.update(track=key,session_id=key,association_quality={**quality,"uncertainty":"UNVALIDATED_APPEARANCE_IDENTITY"})
            self.tracks[key]={"point":point,"velocity":velocity,"time":timestamp,
                "embedding":features[j],"observations":observations}
        return result

    def state_dict(self) -> dict:
        return {"scale":self.scale,"maximum_cosine":self.maximum_cosine,"tracks":self.tracks,
                "next_id":self.next_id,"previous":self.previous}

    def load_state_dict(self,value: dict) -> None:
        if value["scale"]!=self.scale or value["maximum_cosine"]!=self.maximum_cosine:
            raise ValueError("Tracker resume settings changed")
        self.tracks=value["tracks"]
        self.next_id=value["next_id"]
        self.previous=value["previous"]


def native_features(run: Run, source: dict, saved: list[dict], name: str, encoders: dict) -> dict:
    directory=run.output/"features"/name
    directory.mkdir(parents=True,exist_ok=True)
    contract={"source_video":source["video_sha256"],"body_sha256":file_digest(BODY),
        "predictions":json_hash([[r["frame_index"],r["detections"]] for r in saved]),
        "encoders":{k:file_digest(p) if p is not None else file_digest(BODY) for k,p in encoders.items()},
        "crop_policy":"reciprocal40px margin8; full-body COCO crop;128x64",
        "source_code":file_digest(Path(__file__))}
    cache=directory/"embeddings.pt"
    meta=directory/"contract.json"
    if cache.exists() and meta.exists() and read(meta)==contract:
        return torch.load(cache,map_location="cpu",weights_only=False)
    models={name:load_encoder(path) for name,path in encoders.items()}
    body=YOLO(str(BODY))
    cap=cv2.VideoCapture(str(ROOT/source["video"]))
    if not cap.isOpened():
        raise OSError("Embedding video cannot open")
    result={"embeddings":{name:[] for name in models},"hsv":[],"body_boxes":[],"cuts":[]}
    previous,last_tiny=None,None
    if saved and saved[0]["frame_index"]>=5:
        preceding=_read_frame(cap,saved[0]["frame_index"]-5,None)
        previous=saved[0]["frame_index"]-5
        last_tiny=cv2.resize(preceding,(128,36)).astype(np.float32)/255
    try:
        for i,frame in enumerate(saved):
            if i%100==0:
                run.check()
                run.update(current_job="embedding",progress={"source":source["key"],"frame":i,"of":len(saved)})
            image=_read_frame(cap,frame["frame_index"],previous)
            previous=frame["frame_index"]
            bodies=body_detections(body,image)
            boxes=crop_assignment(frame["detections"],bodies,source["dimensions_px"][0])
            result["body_boxes"].append(boxes)
            for name,model in models.items():
                result["embeddings"][name].append(encode_crops(model,image,boxes))
            result["hsv"].append([appearance_descriptor(image,d["footpoint_xy_px"],source["dimensions_px"][0]) for d in frame["detections"]])
            tiny=cv2.resize(image,(128,36)).astype(np.float32)/255
            result["cuts"].append(bool(last_tiny is not None and np.abs(tiny-last_tiny).mean()>.35))
            last_tiny=tiny
    finally:
        cap.release()
    atomic_torch(cache,result)
    write_json(meta,contract)
    del models,body
    torch.cuda.empty_cache()
    return result


def apply_association(saved: list[dict], features: dict, width: int, route: dict) -> list:
    clean=copy.deepcopy(saved)
    for row,cut in zip(clean,features["cuts"],strict=True):
        row.update(camera_cut=cut,timestamp_s=row.get("timestamp_s",row["frame_index"]/25))
    if route["encoder"]=="baseline":
        # Preserve causal cut resets even though the historical helper lacked a cut API.
        result=[]
        start=0
        for i in range(len(clean)+1):
            if i==len(clean) or i>start and clean[i]["camera_cut"]:
                segment=associate_focused(clean[start:i],width,"damped_appearance",features["hsv"][start:i])
                for frame in segment:
                    for d in frame["detections"]:
                        d["track"]=f"c{start}:{d['track']}"
                result+=segment
                start=i
    else:
        tracker=LearnedAssociation(width,route["cosine"])
        result=[tracker.step(frame,feature) for frame,feature in zip(clean,features["embeddings"][route["encoder"]],strict=True)]
    same_detections(saved,result)
    return result


def eligible_identity(old: dict, new: dict) -> bool:
    for key in old:
        a,b=old[key]["score"],new[key]["score"]
        if b["truth_instance_complete_history_fraction"] < a["truth_instance_complete_history_fraction"]-.01:
            return False
        if b["fragmentation_extra_tracks_per_identity"]>a["fragmentation_extra_tracks_per_identity"]*1.05+1:
            return False
    old_rate=np.mean([v["score"]["switch_rate"] for v in old.values()])
    new_rate=np.mean([v["score"]["switch_rate"] for v in new.values()])
    return bool(new_rate<old_rate and sum(v["relinks"].get("false",0) for v in new.values())
                <=sum(v["relinks"].get("false",0) for v in old.values()))


def select_tracker(run: Run) -> dict:
    encoders=available_encoders(run)
    grid=[{"name":"baseline","encoder":"baseline","cosine":None}]
    grid += [{"name":f"{encoder}_c{cosine}","encoder":encoder,"cosine":cosine}
             for encoder in encoders for cosine in (.2,.35)]
    write_json(run.output/"association_grid_lock.json",{"grid":grid,"created_utc":utc(),
        "criteria":"Lower mean switch rate, <=1pp history loss each source, <=5%+1 fragmentation, no increase false relinks. No tactical-call selection."})
    development=read(PERCEPTION/"selected_final_development_predictions.json")
    sources=read(run.output/"source_manifest.json")["sources"]
    metrics={route["name"]:{} for route in grid}
    for match in ("117092","117093"):
        source=next(s for s in sources if s["key"]==f"{match}_1st")
        saved=development[match]
        features=native_features(run,source,saved,f"development_{match}",encoders)
        frozen={}
        for route in grid:
            associated=apply_association(saved,features,source["dimensions_px"][0],route)
            path=run.output/f"development_{match}_{route['name']}.jsonl"
            path.write_text("".join(json.dumps(f,separators=(",",":"))+"\n" for f in associated),encoding="utf-8")
            frozen[route["name"]]=(associated,file_digest(path))
        # Scorer only after autonomous predictions have been frozen.
        labels=_read_annotations(source,{f["frame_index"] for f in saved})
        for route in grid:
            stream,digest=frozen[route["name"]]
            metrics[route["name"]][match]={
                "prediction_sha256":digest,"score":aggregate_scores(stream,labels,source["dimensions_px"][0])["thresholds"]["40"],
                "relinks":identity_audit(stream,labels,source["dimensions_px"][0])["relinks"]}
    selected=grid[0]
    best=np.mean([v["score"]["switch_rate"] for v in metrics["baseline"].values()])
    for route in grid[1:]:
        value=metrics[route["name"]]
        rate=np.mean([v["score"]["switch_rate"] for v in value.values()])
        if rate<best and eligible_identity(metrics["baseline"],value):
            selected,best=route,rate
    result={"status":"DEVELOPMENT_SELECTION_FROZEN","selected":selected,"grid":grid,"metrics":metrics,
        "encoders":{k:{"path":str(v.relative_to(ROOT)) if v else str(BODY.relative_to(ROOT)),
                       "sha256":file_digest(v or BODY)} for k,v in encoders.items()},
        "frozen_utc":utc(),"retrospective_or_reserved_used":False,"negative_result":selected["encoder"]=="baseline"}
    write_json(run.output/"selection.json",result)
    return result


def existing_predictions(source: dict, detector_hash: str, start: int, end: int) -> list | None:
    seal=read(PERCEPTION/"detector_evaluation_seal.json")["files"]
    continuous=read(PERCEPTION/"continuous_seal.json")
    seal.append(continuous)
    for record in seal:
        if (record["match_id"]!=source["id"] or record["half"]!=source["half"]
                or record["detector_sha256"]!=detector_hash
                or record["source_video_sha256"]!=source["video_sha256"]
                or record["scheme"]!="consistent" or record["sampling_hz"]!=5):
            continue
        low,high=record["interval_seconds"]
        if low*25<=start and high*25>=end:
            path=ROOT/record["path"]
            if file_digest(path)!=record["prediction_jsonl_sha256"]:
                raise ValueError("Historical cached prediction changed")
            return [r for r in frames(path) if start<=r["frame_index"]<end]
    return None


def predict_half(run: Run, source: dict) -> dict:
    detector_path=ROOT/run.status["detector"]
    contract={"source_video_sha256":source["video_sha256"],"detector_sha256":file_digest(detector_path),
        "configuration":{"scheme":"consistent","imgsz":1280,"confidence":.2,"iou":.5,"stride":5,
                         "prediction_source_sha256":file_digest(ROOT/"src/backbone/gpu_perception.py")}}
    directory=run.output/"detections"/source["key"]
    directory.mkdir(parents=True,exist_ok=True)
    model=None
    cap=cv2.VideoCapture(str(ROOT/source["video"]))
    if not cap.isOpened():
        raise OSError("Half video unavailable")
    chunks=[]
    start_clock=time.monotonic()
    try:
        for start in range(0,source["frames"],15000):
            run.check()
            end=min(start+15000,source["frames"])
            path=directory/f"{start:07d}.jsonl"
            meta=path.with_suffix(".meta.json")
            if meta.exists():
                value=read(meta)
                if value["contract"]!=contract or file_digest(path)!=value["sha256"]:
                    raise ValueError("Full-half prediction cache contract mismatch")
                chunks.append(value)
                continue
            cached=existing_predictions(source,contract["detector_sha256"],start,end)
            temporary=path.with_suffix(".partial")
            samples=0
            previous=None
            with temporary.open("w",encoding="utf-8") as file:
                if cached is not None:
                    for row in cached:
                        file.write(json.dumps(row,separators=(",",":"))+"\n")
                        samples+=1
                else:
                    if model is None:
                        model=YOLO(str(detector_path))
                    for index in range(start,end,5):
                        if samples%100==0:
                            run.check()
                            run.update(current_job="detect_half",progress={"source":source["key"],"frame":index,"of":source["frames"]})
                        image=_read_frame(cap,index,previous)
                        previous=index
                        bounds=tile_bounds(*source["dimensions_px"],scheme="consistent")
                        detections=merged_detections(model,[image[t:b,l:r] for l,t,r,b in bounds],bounds)
                        row={"frame_index":index,"timestamp_s":index/25,"match_id":source["id"],
                             "half":source["half"],"dimensions_px":source["dimensions_px"],
                             "detector_sha256":contract["detector_sha256"],"detections":detections}
                        file.write(json.dumps(row,separators=(",",":"))+"\n")
                        samples+=1
            if samples!=len(range(start,end,5)):
                raise ValueError("Prediction chunk sample gap")
            temporary.replace(path)
            value={"path":str(path.relative_to(ROOT)),"start_frame":start,"end_frame_exclusive":end,
                "samples":samples,"sha256":file_digest(path),"contract":contract,"historical_cache_used":cached is not None}
            write_json(meta,value)
            chunks.append(value)
    finally:
        cap.release()
        del model
        torch.cuda.empty_cache()
    result={"status":"FULL_HALF_DETECTIONS_SEALED","source":source["key"],"chunks":chunks,
        "samples":sum(c["samples"] for c in chunks),"expected_samples":len(range(0,source["frames"],5)),
        "source_minutes":source["duration_seconds"]/60,"elapsed_seconds":time.monotonic()-start_clock}
    write_json(run.output/f"detections_{source['key']}.json",result)
    return result


def baseline_chunk(saved: list[dict], hsv: list, history: dict, width: int) -> tuple[list,dict]:
    # Historical selected tracker has <=5sample memory; retaining six boundary
    # samples and a tracklet-to-session map reproduces continuous history.
    prefix=history.get("prefix",[])
    combined=prefix+saved
    combined_features=history.get("features",[])+hsv
    associated=associate_focused(combined,width,"damped_appearance",combined_features)
    mapping={}
    previous_map=history.get("tail_map",{})
    next_id=history.get("next_id",1)
    if prefix:
        for before,after in zip(prefix,associated[:len(prefix)],strict=True):
            for a,b in zip(before["detections"],after["detections"],strict=True):
                old=previous_map.get((before["frame_index"],a.get("track")))
                if old is not None:
                    mapping[b["track"]]=old
    result=associated[len(prefix):]
    for frame in result:
        if frame.get("camera_cut"):
            mapping.clear()
        for d in frame["detections"]:
            raw=d["track"]
            if raw not in mapping:
                mapping[raw]=f"baseline-session-{next_id:07d}"
                next_id+=1
            d["track"]=mapping[raw]
    # To make exact chunk-boundary parity testable, keep a rolling input prefix
    # mapped by original input detection indices, not retrospective identities.
    tail_original=copy.deepcopy(combined[-6:])
    tail_associated=associated[-6:]
    tail_map={}
    for original,tracked in zip(tail_original,tail_associated,strict=True):
        for d,t in zip(original["detections"],tracked["detections"],strict=True):
            # Rows before the newly emitted suffix still use raw local IDs.
            identity=t["track"] if t["track"].startswith("baseline-session-") else mapping.get(t["track"])
            if identity is not None:
                tail_map[original["frame_index"],d.get("track")]=identity
    return result,{"prefix":tail_original,"features":combined_features[-6:],"tail_map":tail_map,"next_id":next_id}


class CausalBaseline:
    """Same damped-HSV rule as predecessor, with serializable live state."""
    def __init__(self,width: int):
        self.scale=width/4096
        self.tracks={}
        self.next_id=1
        self.previous=None

    def step(self,frame: dict,features: list) -> dict:
        timestamp=frame.get("timestamp_s",frame["frame_index"]/25)
        if not math.isfinite(timestamp) or self.previous is not None and timestamp<=self.previous:
            raise ValueError("Baseline chronological clock violation")
        if frame.get("camera_cut") or frame.get("identity_reset"):
            self.tracks.clear()
        self.previous=timestamp
        result=copy.deepcopy(frame)
        candidates=[]
        for key,track in self.tracks.items():
            dt=timestamp-track["time"]
            if not 0<dt<=1.+1e-6:
                continue
            age=dt/.2
            predicted=track["point"]+track["velocity"]*dt
            for j,d in enumerate(result["detections"]):
                distance=float(np.linalg.norm(predicted-np.asarray(d["footpoint_xy_px"])))
                if distance>45*self.scale*min(age,2):
                    continue
                app=0.
                if features[j] is not None and track["feature"] is not None:
                    app=float(np.abs(np.asarray(features[j])-track["feature"]).sum())/2
                    if app>.65:
                        continue
                candidates.append((distance+8*(age-1)*self.scale+30*app*self.scale,key,j))
        used,assigned=set(),{}
        for cost,key,j in sorted(candidates):
            if key not in used and j not in assigned:
                assigned[j]=key
                used.add(key)
        for j,d in enumerate(result["detections"]):
            point=np.asarray(d["footpoint_xy_px"],float)
            d["tracklet_id"]=d.get("track")
            if j in assigned:
                key=assigned[j]
                track=self.tracks[key]
                dt=timestamp-track["time"]
                velocity=(.35*(point-track["point"])/dt+.65*track["velocity"]).clip(-200,200)
                quality="OBSERVED_ASSOCIATED"
            else:
                key=f"video-player-{self.next_id:04d}"
                self.next_id+=1
                velocity=np.zeros(2)
                quality="NEW_OBSERVED_TRACK"
            d.update(track=key,session_id=key,association_quality={"status":quality,"uncertainty":"UNVALIDATED_ASSOCIATION"})
            self.tracks[key]={"point":point,"time":timestamp,"velocity":velocity,
                "feature":np.asarray(features[j]) if features[j] is not None else None}
        self.tracks={k:v for k,v in self.tracks.items() if timestamp-v["time"]<=1.+1e-6}
        return result

    def state_dict(self):
        return {"scale":self.scale,"tracks":self.tracks,"next_id":self.next_id,"previous":self.previous}

    def load_state_dict(self,state):
        if state["scale"]!=self.scale:
            raise ValueError("Baseline state geometry changed")
        self.tracks,self.next_id,self.previous=state["tracks"],state["next_id"],state["previous"]


def long_routes(run: Run) -> list:
    selection=read(run.output/"selection.json")
    routes=[{"name":"baseline","encoder":"baseline","cosine":None}]
    for encoder in available_encoders(run):
        eligible=[r for r in selection["grid"] if r["encoder"]==encoder]
        # Development-best per encoder, not retrospective tuning.
        measured=[r for r in eligible if selection["metrics"].get(r["name"])]
        if measured:
            route=min(measured,key=lambda r:np.mean([v["score"]["switch_rate"] for v in selection["metrics"][r["name"]].values()]))
        else:
            route=next(r for r in eligible if r["cosine"]==.2)
        if selection["selected"]["encoder"]==encoder:
            route=selection["selected"]
        routes.append(route)
    return routes


def associate_half(run: Run, source: dict) -> dict:
    seal=read(run.output/f"detections_{source['key']}.json")
    routes=long_routes(run)
    encoders=available_encoders(run)
    trackers={r["name"]:(CausalBaseline(source["dimensions_px"][0]) if r["encoder"]=="baseline" else
                        LearnedAssociation(source["dimensions_px"][0],r["cosine"])) for r in routes}
    directory=run.output/"tracks"/source["key"]
    directory.mkdir(parents=True,exist_ok=True)
    records=[]
    for chunk in seal["chunks"]:
        run.check()
        saved=frames(ROOT/chunk["path"])
        start=chunk["start_frame"]
        state_path=directory/f"state_{start:07d}.pt"
        result_path=directory/f"chunk_{start:07d}.json"
        contract={"detection_sha256":chunk["sha256"],"routes":routes,
            "encoder_hashes":{k:file_digest(p or BODY) for k,p in encoders.items()},
            "implementation":file_digest(Path(__file__))}
        if result_path.exists():
            prior=read(result_path)
            if prior["contract"]!=contract or not state_path.exists():
                raise ValueError("Long association resume contract changed")
            state_value=torch.load(state_path,weights_only=False)
            for name,tracker in trackers.items():
                tracker.load_state_dict(state_value[name])
                if file_digest(ROOT/prior["files"][name]["path"])!=prior["files"][name]["sha256"]:
                    raise ValueError("Sealed long route changed")
            records.append(prior)
            continue
        features=native_features(run,source,saved,f"{source['key']}_{start:07d}",encoders)
        files={}
        for route in routes:
            route_frames=[]
            tracker=trackers[route["name"]]
            for i,frame in enumerate(saved):
                frame=copy.deepcopy(frame)
                frame["camera_cut"]=features["cuts"][i]
                vector=features["hsv"][i] if route["encoder"]=="baseline" else features["embeddings"][route["encoder"]][i]
                route_frames.append(tracker.step(frame,vector))
            same_detections(saved,route_frames)
            path=directory/f"{start:07d}_{route['name']}.jsonl"
            temporary=path.with_suffix(".partial")
            temporary.write_text("".join(json.dumps(f,separators=(",",":"))+"\n" for f in route_frames),encoding="utf-8")
            temporary.replace(path)
            files[route["name"]]={"path":str(path.relative_to(ROOT)),"sha256":file_digest(path)}
        atomic_torch(state_path,{k:t.state_dict() for k,t in trackers.items()})
        record={"start_frame":start,"end_frame_exclusive":chunk["end_frame_exclusive"],
                "contract":contract,"files":files,"features_path":str((run.output/"features"/f"{source['key']}_{start:07d}"/"embeddings.pt").relative_to(ROOT))}
        write_json(result_path,record)
        records.append(record)
        run.update(current_job="associate_half",progress={"source":source["key"],"end_frame":chunk["end_frame_exclusive"]})
    result={"status":"FULL_HALF_ASSOCIATIONS_FROZEN","source":source["key"],"routes":routes,"chunks":records}
    write_json(run.output/f"tracks_{source['key']}.json",result)
    return result


def score_half(run: Run, source: dict) -> dict:
    seal=read(run.output/f"tracks_{source['key']}.json")
    # Labels read only after every route is frozen for the complete half.
    indices=set(range(0,source["frames"],5))
    labels=_read_annotations(source,indices)
    result={}
    for route in seal["routes"]:
        saved=[]
        for chunk in seal["chunks"]:
            record=chunk["files"][route["name"]]
            if file_digest(ROOT/record["path"])!=record["sha256"]:
                raise ValueError("Long predictions changed before scorer")
            saved.extend(frames(ROOT/record["path"]))
        score=aggregate_scores(saved,labels,source["dimensions_px"][0])["thresholds"]["40"]
        audit=identity_audit(saved,labels,source["dimensions_px"][0])
        result[route["name"]]={"score":score,"audit":audit,
            "ambiguity_abstentions":sum(d["association_quality"]["status"]=="AMBIGUOUS_ABSTAIN"
                                       for f in saved for d in f["detections"])}
        del saved
    summary={"status":"POST_INFERENCE_FULL_HALF_SCORED","source":source["key"],"routes":result,
             "exposure":source["exposure"],"labels_runtime":False}
    write_json(run.output/f"scores_{source['key']}.json",summary)
    return summary


def build_half_external(run: Run, source: dict) -> dict:
    from src.backbone.video_player_attributes import _knn_scores, TEAM_CLASSES, ROLE_CLASSES, appearance_feature
    selection=read(run.output/"selection.json")
    selected=selection["selected"]["name"]
    # Long routes retain the development-best setting per encoder, including selected.
    routes=read(run.output/f"tracks_{source['key']}.json")
    names=list(dict.fromkeys(["baseline",selected]))
    protocol=read(HISTORICAL/"protocol.json")
    projector=None
    if source["key"] in ("117092_1st","118576_2nd"):
        projector=load_local_projector(protocol,HISTORICAL,source["id"])
    features,teams,roles,attribute_metadata=_attribute_model()
    directory=run.output/"external"/source["key"]
    directory.mkdir(parents=True,exist_ok=True)
    paths={name:directory/f"{name}.jsonl" for name in names}
    complete=directory/"seal.json"
    if complete.exists():
        value=read(complete)
        if all(file_digest(ROOT/r["path"])==r["sha256"] for r in value["files"].values()):
            return value
        raise ValueError("Completed external observations changed")
    handles={name:path.with_suffix(".partial").open("w",encoding="utf-8") for name,path in paths.items()}
    header={"record_type":"metadata","schema_version":"backbone.observations.0.2",
        "source_id":f"identity-program-{source['key']}","match":source["id"],"half":source["half"],
        "observation_kind":"exo_source","sampling_seconds":.2,
        "missing_count_semantics":"unprojected image detections, never imputed complete players",
        "coordinates":{"system":"pitch_top_left","units":"metres","axes":"x_length_y_width",
                       "pitch_length_m":105,"pitch_width_m":68},
        "origin":{"type":"sealed_frozen_detector_and_video_appearance","video_sha256":source["video_sha256"],
                  "detector_sha256":run.status["detector_sha256"]},
        "calibration":{"status":"FAILED_GLOBAL_EXPERIMENTAL_LOCAL" if projector else "UNAVAILABLE_ALL_POSITIONS_UNKNOWN"},
        "synchronization":{"status":"native25fps_frame_clock","stride":5},
        "quality":{"attributes":"Frozen source-trained KNN with unchanged confidence abstention",
                   "visibility":"Simulated later, not measured gaze","ball_possession":"UNKNOWN"}}
    for file in handles.values():
        file.write(json.dumps(header)+"\n")
    cap=cv2.VideoCapture(str(ROOT/source["video"]))
    if not cap.isOpened():
        raise OSError("Attribute video failed")
    previous=None
    counts=Counter()
    try:
        for chunk in routes["chunks"]:
            run.check()
            streams={name:frames(ROOT/chunk["files"][name]["path"]) for name in names}
            values=torch.load(ROOT/chunk["features_path"],weights_only=False)
            for i,base in enumerate(streams[names[0]]):
                if i%100==0:
                    run.check()
                    run.update(current_job="external",progress={"source":source["key"],"frame":base["frame_index"]})
                image=None
                measurements=[]
                for j,d in enumerate(base["detections"]):
                    pitch,status=project_footpoint(d["footpoint_xy_px"],projector) if projector else (None,"NO_APPROVED_SOURCE_PROJECTOR")
                    team,goalkeeper=0,None
                    quality={"status":"UNVALIDATED_EXPERIMENTAL","projection":status,
                             "attribute_source":"UNKNOWN","detector_confidence":d["detector_confidence"],
                             "position_accuracy_m":None}
                    box=values["body_boxes"][i][j]
                    if pitch is not None and box is not None:
                        if image is None:
                            image=_read_frame(cap,base["frame_index"],previous)
                            previous=base["frame_index"]
                        vector=appearance_feature(image,box)[None]
                        team_score=_knn_scores(features,teams,vector,TEAM_CLASSES)[0]
                        role_score=_knn_scores(features,roles,vector,ROLE_CLASSES)[0]
                        ti,ri=int(team_score.argmax()),int(role_score.argmax())
                        team=ti+1 if float(team_score[ti])>=attribute_metadata["team_confidence_threshold"] else 0
                        goalkeeper=bool(ri==1) if float(role_score[ri])>=attribute_metadata["role_confidence_threshold"] else None
                        quality.update(attribute_source="FROZEN_SOURCE_KNN_OR_ABSTENTION",
                            team_confidence=float(team_score[ti]),role_confidence=float(role_score[ri]))
                    measurements.append({"track":d["track"],"segment":0,"team":team or "UNKNOWN",
                        "goalkeeper":goalkeeper,"position_status":"observed" if pitch is not None else "missing",
                        "xy":pitch,"quality":quality})
                    counts["detections"]+=1
                    counts["projected"]+=pitch is not None
                    counts["unknown_team"]+=team==0
                    counts["unknown_role"]+=goalkeeper is None
                for name,file in handles.items():
                    row=streams[name][i]
                    assigned=copy.deepcopy(measurements)
                    for measurement,d in zip(assigned,row["detections"],strict=True):
                        measurement["track"]=d["track"]
                    file.write(json.dumps({"record_type":"frame","frame_index":row["frame_index"],
                        "timestamp_s":row["timestamp_s"],"segment_id":source["key"],
                        "identity_reset":False,"camera_cut":row.get("camera_cut",False),
                        "measurements":assigned,"absent_count":sum(m["xy"] is None for m in assigned)},separators=(",",":"))+"\n")
    finally:
        cap.release()
        for file in handles.values():
            file.close()
    for name,path in paths.items():
        path.with_suffix(".partial").replace(path)
    result={"status":"EXTERNAL_INPUTS_FROZEN","source":source["key"],"selected":selected,
        "files":{name:{"path":str(path.relative_to(ROOT)),"sha256":file_digest(path)} for name,path in paths.items()},
        "counts":dict(counts),"projection":"EXPERIMENTAL_UNVALIDATED" if projector else "UNSUPPORTED_NOT_INVENTED",
        "annotation_attributes_used":False,"same_detector_all_routes":True}
    write_json(complete,result)
    return result


def integrate_half(run: Run, source: dict) -> dict:
    seal=build_half_external(run,source)
    release=read(ROOT/"configs/backbone_release_baseline.json")
    world=ModelSession(ROOT/release["experiment"],"G1",device="cpu")
    action=ActionModelSession(ROOT/release["action_checkpoint"],device="cpu")
    results={}
    fixed_lock=run.output/f"anchors_{source['key']}.json"
    # Anchors are causally chosen from baseline prefix, then mapped via identical
    # detection indices. No future longest-track ranking.
    baseline=ExternalObservationSource(ROOT/seal["files"]["baseline"]["path"])
    prefix=list(external_frames(baseline,0,min(600,len(range(0,source["frames"],5)))))
    cohort=Cohort()
    anchors=[]
    seen=set()
    for frame in prefix:
        for session in cohort.step(frame):
            if session is not None and session not in seen and len(anchors)<2:
                anchors.append({"track":session,"frame_index":frame.index,"timestamp_s":frame.timestamp_s,
                                "rule":"First2causal11-sample qualified baseline sessions in first120s"})
                seen.add(session)
    while len(anchors)<2:
        anchors.append({"track":None,"frame_index":0,"timestamp_s":0.,"rule":"NO_USABLE_PREFIX_ANCHOR"})
    write_json(fixed_lock,{"anchors":anchors,"selected_by_video_only":True,
        "unsupported_projection":seal["projection"]=="UNSUPPORTED_NOT_INVENTED"})
    # Precompute same-detection mapping only at the two observed prefix anchors.
    tracks=read(run.output/f"tracks_{source['key']}.json")
    mappings={name:[] for name in seal["files"]}
    for anchor in anchors:
        if anchor["track"] is None:
            for values in mappings.values():
                values.append(None)
            continue
        chunk=next(c for c in tracks["chunks"] if c["start_frame"]<=anchor["frame_index"]<c["end_frame_exclusive"])
        original=next(r for r in frames(ROOT/chunk["files"]["baseline"]["path"]) if r["frame_index"]==anchor["frame_index"])
        index=next(j for j,d in enumerate(original["detections"]) if d["track"]==anchor["track"])
        for name in mappings:
            row=next(r for r in frames(ROOT/chunk["files"][name]["path"]) if r["frame_index"]==anchor["frame_index"])
            mappings[name].append(row["detections"][index]["track"])
    directory=run.output/"replays"/source["key"]
    directory.mkdir(parents=True,exist_ok=True)
    for name,record in seal["files"].items():
        adapter=ExternalObservationSource(ROOT/record["path"])
        for mode in ("FIXED_ACTOR","OBSERVABLE_COHORT"):
            result_path=directory/f"{name}_{mode}.json"
            if result_path.exists():
                value=read(result_path)
                if value["input_sha256"]!=record["sha256"]:
                    raise ValueError("Replay input hash changed")
                results[name+"_"+mode]=value
                continue
            engines,providers,last_processed={},{},{}
            cohort=Cohort()
            totals=Counter()
            minutes=defaultdict(Counter)
            before_g1,before_s4=world.world_forward_calls,action.forward_calls
            previous=-1e-6
            query_rows=[]
            runtime_path=directory/f"{name}_{mode}.jsonl"
            with runtime_path.with_suffix(".partial").open("w",encoding="utf-8") as file:
                raw_file=(ROOT/record["path"]).open(encoding="utf-8")
                next(raw_file)
                for i,frame in enumerate(external_frames(adapter,0,len(range(0,source["frames"],5)))):
                    raw_row=json.loads(next(raw_file))
                    if raw_row["frame_index"]!=frame.index:
                        raise ValueError("Raw external/adapter alignment changed")
                    if i%100==0:
                        run.check()
                        run.update(current_job="integration",progress={"source":source["key"],"route":name,
                            "mode":mode,"sample":i})
                    if frame.discontinuity:
                        engines.clear()
                        providers.clear()
                        last_processed.clear()
                    elapsed=frame.timestamp_s
                    counts=minutes[int(elapsed/60)]
                    if mode=="FIXED_ACTOR":
                        ids=[session if frame.index>=anchor["frame_index"] else None
                             for session,anchor in zip(mappings[name],anchors,strict=True)]
                    else:
                        ids=cohort.step(frame)
                    due=due_queries(previous,elapsed,run.config["query_every_s"])
                    previous=elapsed
                    for slot,session in enumerate(ids):
                        counts["opportunities"]+=1
                        counts["query_attempts"]+=len(due)
                        if session is None:
                            counts["NO_USABLE_SESSION"]+=1
                            counts["unsupported_queries"]+=len(due)
                            for timestamp in due:
                                query_rows.append({"slot":slot,"session":None,"scheduled_s":timestamp,
                                    "status":"unsupported","reason":"No causal video actor anchor/session"})
                            continue
                        if session not in engines:
                            engines[session]=ReplayEngine(world,expiry_s=6.,fallback="last_seen",action_model=action)
                            providers[session]=SyntheticObservationProvider(session,seed=42,fov=120,dropout=True)
                        last_processed[session]=elapsed
                        engine=engines[session]
                        observation=adapter.observe(frame,providers[session])
                        gb,sb=world.world_forward_calls,action.forward_calls
                        query=bool(due and abs(due[-1]-elapsed)<1e-4)
                        row=engine.step(observation,include_passing_assessment=False,action_query=query)
                        causes=contract_causes(engine,observation,row)
                        measurement=next((m for m in raw_row["measurements"] if m["track"]==session),None)
                        if measurement is None:
                            counts["RUNTIME_SESSION_NOT_DETECTED_VISIBILITY_UNRESOLVED"]+=1
                        elif measurement["xy"] is None:
                            counts["IMAGE_DETECTED_OUTSIDE_PROJECTION"]+=1
                        else:
                            counts["IMAGE_DETECTED_AND_PROJECTED"]+=1
                        counts.update(causes)
                        counts["actor_available"]+=observation.actor is not None
                        counts["g1_calls"]+=world.world_forward_calls-gb
                        counts["s4_calls"]+=action.forward_calls-sb
                        counts["fallback_entity_states"]+=sum(not e["visible"] and e["backend"]!="G1" for e in row["belief"]["entities"])
                        counts["g1_hidden_states"]+=sum(not e["visible"] and e["backend"]=="G1" for e in row["belief"]["entities"])
                        for timestamp in due:
                            inference=row["action_inference"] if query and timestamp==due[-1] else {
                                "status":"unsupported","reason":"No observation at supplied timestamp"}
                            counts["supported_queries" if inference["status"]=="supported" else "unsupported_queries"]+=1
                            query_rows.append({"slot":slot,"session":session,"scheduled_s":timestamp,**inference})
                        file.write(json.dumps({"sample":i,"frame_index":frame.index,"timestamp_s":elapsed,
                            "slot":slot,"session":session,"actor_available":observation.actor is not None,
                            "causes":causes,"belief":row["belief"],"action_inference":row["action_inference"],
                            "scheduled_queries":due},separators=(",",":"))+"\n")
                    # Inactive cohort engines older than expiry contain no valid
                    # memory; explicit retirement prevents full-half RAM growth.
                    expired=[key for key,then in last_processed.items() if elapsed-then>6.+1e-6 and key not in ids]
                    for key in expired:
                        del engines[key],providers[key],last_processed[key]
                        counts["expired_independent_session_memories"]+=1
                raw_file.close()
            runtime_path.with_suffix(".partial").replace(runtime_path)
            totals=sum(minutes.values(),Counter())
            expected=2*len(range(0,source["frames"],5))
            if totals["opportunities"]!=expected:
                raise ValueError("Replay opportunities lost")
            value={"status":"FULL_HALF_REPLAY_EXECUTED","source":source["key"],"route":name,"mode":mode,
                "input_sha256":record["sha256"],"totals":dict(totals),"opportunity_denominator":expected,
                "g1_calls":world.world_forward_calls-before_g1,"s4_calls":action.forward_calls-before_s4,
                "per_minute":[{"minute":m,**dict(c)} for m,c in sorted(minutes.items())],
                "anchors":mappings[name] if mode=="FIXED_ACTOR" else None,
                "cohort_admissions":cohort.admissions,"query_results":query_rows,
                "runtime_path":str(runtime_path.relative_to(ROOT)),"runtime_sha256":file_digest(runtime_path),
                "projection":seal["projection"],"frozen_G1_sha256":world.provenance["G1"]["sha256"],
                "frozen_S4_sha256":action.sha256,
                "limitations":"Cohort sessions never recover fixed actors by substitution; unknown attrs and unprojected positions stay unknown. Simulated visibility, supplied-time actions. No tactical-accuracy claim."}
            write_json(result_path,value)
            results[name+"_"+mode]=value
    summary={"status":"FULL_HALF_INTEGRATION_EXECUTED","source":source["key"],"results":results,
        "selected_equals_baseline":seal["selected"]=="baseline",
        "unavailable_projection_explicit":seal["projection"]=="UNSUPPORTED_NOT_INVENTED"}
    write_json(run.output/f"integration_{source['key']}.json",summary)
    return summary


def interruption_check(run: Run) -> dict:
    directory=run.output/"interruption"
    directory.mkdir(exist_ok=True)
    fixture=[]
    for i in range(40):
        points=[] if i==12 else [[100+i,100],[300-i*2,120]]
        fixture.append({"frame_index":i*5,"timestamp_s":i*.2,"camera_cut":i==30,
            "detections":[{"track":str(j),"footpoint_xy_px":p,"bbox_xyxy_px":[p[0]-3,p[1]-20,p[0]+3,p[1]],
                           "detector_confidence":.9} for j,p in enumerate(points)]})
    write_json(directory/"fixture.json",{"frames":fixture})
    # Actual distinct processes, first stopped at a declared completed boundary.
    command=[sys.executable,"-m","src.backbone.identity_program","--config",str(run.output/"config.json"),
             "--output",str(run.output),"restart-fixture"]
    for mode in ("full","first","resume"):
        completed=subprocess.run(command+["--fixture-mode",mode],cwd=ROOT,capture_output=True,text=True,timeout=60)
        (directory/f"{mode}.log").write_text(completed.stdout+completed.stderr,encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Disposable restart fixture failed: {mode}")
    full=frames(directory/"full.jsonl")
    resumed=frames(directory/"resumed.jsonl")
    if full!=resumed or len({r["frame_index"] for r in resumed})!=len(fixture):
        raise ValueError("Real process restart changed chronological association or duplicated times")
    # Real CUDA training checkpoint optimizer/scaler roundtrip with exact next step.
    torch.manual_seed(42)
    model=AppearanceEncoder(4).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    scaler=torch.amp.GradScaler("cuda",init_scale=128.)
    guard=AmpGuard()
    batch=torch.randn(8,3,128,64,device="cuda")
    target=torch.arange(8,device="cuda")%4
    def update(m,o,s,g):
        with torch.autocast("cuda",dtype=torch.float16):
            _,logits=m(batch)
            loss=F.cross_entropy(logits,target)
        g.note_minibatch(loss,scaler=s)
        s.scale(loss).backward()
        g.update(m,o,s,loss)
    update(model,optimizer,scaler,guard)
    atomic_torch(directory/"cuda_roundtrip.pt",{"model":model.state_dict(),"optimizer":optimizer.state_dict(),
        "scaler":scaler.state_dict(),"guard":guard.state_dict()})
    restored=AppearanceEncoder(4).cuda().train()
    opt=torch.optim.AdamW(restored.parameters(),lr=3e-4)
    scale=torch.amp.GradScaler("cuda",init_scale=128.)
    g=AmpGuard()
    saved=torch.load(directory/"cuda_roundtrip.pt",weights_only=False)
    restored.load_state_dict(saved["model"]);opt.load_state_dict(saved["optimizer"])
    scale.load_state_dict(saved["scaler"]);g.load_state_dict(saved["guard"])
    update(model,optimizer,scaler,guard)
    update(restored,opt,scale,g)
    difference=max(float((a-b).abs().max()) for a,b in zip(model.parameters(),restored.parameters(),strict=True))
    if difference>1e-5 or guard.state_dict()!=g.state_dict():
        raise ValueError("CUDA next-update checkpoint parity failed")
    return {"status":"PROCESS_RESTART_AND_CUDA_NEXT_UPDATE_PASS","chronological_rows":len(resumed),
        "duplicate_frame_rows":0,"max_parameter_difference":difference,
        "scope":"Disposable actual process termination at boundary, then new process resumes tracker. CUDA model/optimizer/scaler next update parity. Main fits resume completed epochs, not exact mid-batch."}


def fixture_process(output: Path, mode: str) -> None:
    directory=output/"interruption"
    fixture=read(directory/"fixture.json")["frames"]
    tracker=LearnedAssociation(4096,.2)
    start,end=0,len(fixture)
    if mode=="first":
        end=17
    if mode=="resume":
        tracker.load_state_dict(torch.load(directory/"tracker.pt",weights_only=False))
        start=17
    path=directory/("full.jsonl" if mode=="full" else "resumed.jsonl")
    with path.open("a" if mode=="resume" else "w",encoding="utf-8") as file:
        for row in fixture[start:end]:
            features=[[1.,0.] if j==0 else [0.,1.] for j in range(len(row["detections"]))]
            file.write(json.dumps(tracker.step(row,features),sort_keys=True)+"\n")
    if mode=="first":
        atomic_torch(directory/"tracker.pt",tracker.state_dict())


def render_examples(run: Run, source: dict) -> dict:
    ffmpeg=ROOT/"data/video_review_tools/portable/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe"
    if not ffmpeg.is_file():
        raise FileNotFoundError("Local ffmpeg unavailable")
    integration=read(run.output/f"integration_{source['key']}.json")
    selection=read(run.output/"selection.json")["selected"]["name"]
    names=list(dict.fromkeys(["baseline",selection]))
    runtime={}
    failure=None
    for name in names:
        for mode in ("FIXED_ACTOR","OBSERVABLE_COHORT"):
            record=integration["results"][name+"_"+mode]
            row_map=defaultdict(list)
            previously=False
            with (ROOT/record["runtime_path"]).open() as file:
                for line in file:
                    row=json.loads(line)
                    row_map[row["frame_index"]].append(row)
                    if name=="baseline" and mode=="FIXED_ACTOR" and row["slot"]==0:
                        if previously and not row["actor_available"] and failure is None:
                            failure=row["timestamp_s"]
                        previously=row["actor_available"]
            runtime[name,mode]=row_map
    starts=[("predetermined",60.),("measured_failure",max(0.,(failure or 120.)-2))]
    chunks=read(run.output/f"tracks_{source['key']}.json")["chunks"]
    results=[]
    for kind,start in starts:
        run.check()
        end=min(start+30,source["duration_seconds"])
        first=int(start*25)//5*5
        last=int(end*25)//5*5
        route_frames={name:{} for name in names}
        for chunk in chunks:
            if chunk["end_frame_exclusive"]<=first or chunk["start_frame"]>=last:
                continue
            for name in names:
                route_frames[name].update({r["frame_index"]:r for r in frames(ROOT/chunk["files"][name]["path"])
                                          if first<=r["frame_index"]<last})
        destination=run.output/f"{source['key']}_{kind}.mp4"
        cap=cv2.VideoCapture(str(ROOT/source["video"]))
        if not cap.isOpened():
            raise OSError("Renderer source unavailable")
        process=subprocess.Popen([str(ffmpeg),"-nostdin","-y","-f","rawvideo","-pix_fmt","bgr24",
            "-s","1280x720","-r","5","-i","pipe:0","-an","-c:v","libx264","-preset","veryfast",
            "-crf","25","-pix_fmt","yuv420p","-movflags","+faststart",str(destination)],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        trails={name:{} for name in names}
        latest={}
        previous=None
        try:
            for index in range(first,last,5):
                image=_read_frame(cap,index,previous)
                previous=index
                canvas=np.zeros((720,1280,3),np.uint8)
                cv2.putText(canvas,f"{source['key']} {index/25:.1f}s | {kind} | Same frozen detections. Experimental/unsupported pitch",(8,20),0,.45,(255,255,255),1)
                for column in range(2):
                    name=names[min(column,len(names)-1)]
                    left=column*640
                    cv2.putText(canvas,name+(" (selected=baseline)" if len(names)==1 and column==1 else ""),(left+5,45),0,.4,(0,220,220),1)
                    canvas[60:229,left:left+640]=cv2.resize(image,(640,169))
                    for d in route_frames[name][index]["detections"]:
                        p=d["footpoint_xy_px"]
                        point=(left+round(p[0]*640/source["dimensions_px"][0]),60+round(p[1]*169/source["dimensions_px"][1]))
                        trail=trails[name].setdefault(d["track"],[])
                        trail.append(point)
                        del trail[:-11]
                        if len(trail)>1:
                            cv2.polylines(canvas,[np.asarray(trail,np.int32)],False,(150,150,150),1)
                        cv2.circle(canvas,point,2,(180,180,180),-1)
                        cv2.putText(canvas,d["track"][-4:],point,0,.26,(220,220,220),1)
                    y=250
                    for mode in ("FIXED_ACTOR","OBSERVABLE_COHORT"):
                        cv2.putText(canvas,mode+" independent causal memories",(left+5,y),0,.4,(0,220,255),1)
                        y+=22
                        rows=runtime[name,mode].get(index,[])
                        for slot in (0,1):
                            row=next((r for r in rows if r["slot"]==slot),None)
                            if row is None:
                                text=f"slot{slot}: no supported actor/session; query attempts retained"
                                cv2.putText(canvas,text,(left+5,y),0,.33,(230,230,230),1)
                            else:
                                if row["scheduled_queries"]:
                                    latest[name,mode,slot]=(row["timestamp_s"],row["action_inference"])
                                lines=[f"slot{slot} {row['session']} present={row['actor_available']}",
                                    ";".join(row["causes"])[:85] or "G1 history contract ready",
                                    "memory uncertainty std_m="+str([e["std_m"] for e in row["belief"]["entities"] if not e["visible"]][:1])[:60]]
                                query=latest.get((name,mode,slot))
                                if query:
                                    lines.append(f"Last supplied query@{query[0]:.1f}: {query[1]['status']} {(query[1].get('output') or {}).get('action_type')}")
                                for line in lines:
                                    cv2.putText(canvas,line,(left+5,y),0,.32,(230,230,230),1)
                                    y+=17
                            y+=18
                        y+=10
                    cv2.putText(canvas,"Gray=observed image detection; no invented pitch outside support",(left+5,640),0,.32,(220,220,220),1)
                    cv2.putText(canvas,"Cohort switching is NOT fixed-person identity recovery",(left+5,660),0,.35,(220,220,220),1)
                cv2.putText(canvas,"Simulated visibility, not gaze. G1 hidden estimates never feed S4. Calls/classifications are not tactical accuracy.",(8,700),0,.4,(220,220,220),1)
                if index==first:
                    cv2.imwrite(str(destination.with_suffix(".jpg")),canvas)
                process.stdin.write(canvas.tobytes())
        except BaseException:
            process.kill()
            raise
        finally:
            cap.release()
            process.stdin.close()
            process.wait(timeout=60)
        if process.returncode:
            raise RuntimeError("Render failed")
        check=cv2.VideoCapture(str(destination))
        count=int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=float(check.get(cv2.CAP_PROP_FPS))
        decoded=0
        for sample in (0,count//2,max(0,count-2)):
            check.set(cv2.CAP_PROP_POS_FRAMES,sample)
            ok,_=check.read()
            decoded+=int(ok)
        check.release()
        if count!=len(range(first,last,5)) or abs(fps-5)>.001 or decoded!=3:
            raise ValueError("Real rendered-video decoding/coverage failed")
        results.append({"path":str(destination.relative_to(ROOT)),"frames":count,"fps":fps,
            "decoded_seek_frames":decoded,"start_s":first/25,"end_s":last/25,"kind":kind,
            "sha256":file_digest(destination),"failure_found":failure is not None})
    write_json(run.output/f"media_{source['key']}.json",{"status":"REAL_VIDEO_DECODE_VERIFIED","files":results})
    return {"status":"REAL_VIDEO_DECODE_VERIFIED","files":results}


def stress_sequence(run: Run) -> dict:
    sources=read(run.output/"source_manifest.json")["sources"]
    source=next(s for s in sources if s["key"]=="118578_1st")
    seal_path=run.output/f"detections_{source['key']}.json"
    if not seal_path.exists():
        return {"status":"BLOCKED_NO_NEW_CHRONOLOGICAL_SOURCE","condition":"Stress never uses fitting/development frames"}
    seal=read(seal_path)
    chunk=next(c for c in seal["chunks"] if c["start_frame"]>=15000)
    saved=frames(ROOT/chunk["path"])[:300]
    chosen=read(run.output/"selection.json")["selected"]
    encoders=available_encoders(run)
    if chosen["encoder"]=="baseline":
        chosen={"name":"pretrained_stress","encoder":"pretrained","cosine":.2}
    path=encoders[chosen["encoder"]]
    model=load_encoder(path)
    cap=cv2.VideoCapture(str(ROOT/source["video"]))
    body=YOLO(str(BODY))
    conditions={"natural":[],"half_resolution":[],"short_observation_gap":[],"partial_occlusion":[]}
    trackers={name:LearnedAssociation(source["dimensions_px"][0],chosen["cosine"]) for name in conditions}
    previous=None
    try:
        for i,frame in enumerate(saved):
            run.check()
            image=_read_frame(cap,frame["frame_index"],previous)
            previous=frame["frame_index"]
            bodies=body_detections(body,image)
            boxes=crop_assignment(frame["detections"],bodies,source["dimensions_px"][0])
            for name in conditions:
                altered=image
                if name=="half_resolution":
                    altered=cv2.resize(cv2.resize(image,(image.shape[1]//2,image.shape[0]//2)),image.shape[1::-1])
                elif name=="partial_occlusion":
                    altered=image.copy()
                    for box in boxes:
                        if box is None:
                            continue
                        x1,y1,x2,y2=map(int,box)
                        y=int(y1+(y2-y1)*.4)
                        altered[max(0,y):max(0,y+int((y2-y1)*.15)),max(0,x1):max(0,x2)]=80
                f=copy.deepcopy(frame)
                if name=="short_observation_gap" and i%100 in (40,41):
                    f["detections"]=[]
                    embedding=[]
                else:
                    embedding=encode_crops(model,altered,boxes)
                conditions[name].append(trackers[name].step(f,embedding))
    finally:
        cap.release()
    # Freeze all variants before retrospective stress labels.
    for name,stream in conditions.items():
        file=run.output/f"stress_{name}.jsonl"
        file.write_text("".join(json.dumps(f,separators=(",",":"))+"\n" for f in stream),encoding="utf-8")
    labels=_read_annotations(source,{f["frame_index"] for f in saved})
    result={"status":"ONE_PREDECLARED_SYNTHETIC_QUALITY_SWEEP_COMPLETE","source":source["key"],
        "new_unique_footage_minutes":0,"conditions_are_not_new_footage":True,
        "samples":len(saved),"scores":{name:aggregate_scores(stream,labels,source["dimensions_px"][0])["thresholds"]["40"]
                                      for name,stream in conditions.items()},
        "distinction":"Natural errors in unchanged route; resampling/occlusion/gaps synthetic; frozen detections except intentionally removed observation fixtures, never claimed detector parity"}
    write_json(run.output/"stress.json",result)
    return result


def verify(run: Run) -> dict:
    run.check(final=True)
    before=read(run.output/"protected_before.json")
    after={name:file_digest(ROOT/name) for name in before}
    write_json(run.output/"protected_after.json",after)
    if before!=after:
        raise ValueError("Protected model/configuration/annotation artifact changed")
    test=subprocess.run([sys.executable,"-m","pytest","-q","-rA",
        "tests/test_identity_program.py","tests/test_actor_continuity.py","tests/test_amp_guard.py",
        "tests/test_perception_continuation.py","tests/test_backbone_release.py",
        "--junitxml="+str(run.output/"tests.xml")],cwd=ROOT,text=True,capture_output=True,timeout=300)
    (run.output/"tests.txt").write_text(test.stdout+test.stderr,encoding="utf-8")
    if test.returncode:
        raise RuntimeError("Focused verification failed; see complete tests.txt")
    parity=[]
    sources=read(run.output/"source_manifest.json")["sources"]
    for source in sources:
        path=run.output/f"tracks_{source['key']}.json"
        if not path.exists():
            continue
        tracks=read(path)
        detections=read(run.output/f"detections_{source['key']}.json")
        for track_chunk,det_chunk in zip(tracks["chunks"],detections["chunks"],strict=True):
            original=frames(ROOT/det_chunk["path"])
            if file_digest(ROOT/det_chunk["path"])!=det_chunk["sha256"]:
                raise ValueError("Sealed detector outputs changed")
            for name,record in track_chunk["files"].items():
                if file_digest(ROOT/record["path"])!=record["sha256"]:
                    raise ValueError("Sealed association outputs changed")
                same_detections(original,frames(ROOT/record["path"]))
        parity.append(source["key"])
    result={"status":"PASS","protected_artifacts_unchanged":True,"parity_sources":parity,
        "tests":test.stdout.strip().splitlines()[-1],"verified_utc":utc()}
    write_json(run.output/"verification.json",result)
    return result


def package(run: Run) -> dict:
    sources=read(run.output/"source_manifest.json")["sources"] if (run.output/"source_manifest.json").exists() else []
    core_keys=[f"{m}_{h}" for m,h in run.config["core_sources"]]
    complete=all(run.status["jobs"].get(f"integration_{key}",{}).get("state")=="COMPLETE" for key in core_keys)
    complete=complete and (run.output/"verification.json").exists()
    training_ok=all((run.output/r/"summary.json").exists() and read(run.output/r/"summary.json")["status"]=="FINITE_FIT" for r in ("R1","R2"))
    terminal="COMPLETE" if complete and training_ok else "PARTIAL"
    destination=Path.home()/"Downloads"/f"return_identity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    completed_sources=[s for s in sources if run.status["jobs"].get(f"scores_{s['key']}",{}).get("state")=="COMPLETE"]
    summary={"status":terminal,"predecessor_consumed":str(PREDECESSOR.relative_to(ROOT)),
        "route":read(run.output/"route_decision.json"),"elapsed_hours":run.elapsed()/3600,
        "full_half_sources":[s["key"] for s in completed_sources],
        "unique_evaluation_minutes":sum(s["duration_seconds"] for s in completed_sources)/60,
        "training":{r:read(run.output/r/"summary.json") if (run.output/r/"summary.json").exists() else {"status":"NOT_EXECUTED"} for r in ("R1","R2")},
        "selection":read(run.output/"selection.json").get("selected") if (run.output/"selection.json").exists() else None,
        "verification":read(run.output/"verification.json") if (run.output/"verification.json").exists() else {"status":"BLOCKED"},
        "stop_reason":run.status.get("stop_reason","USEFUL_AUTHORIZED_QUEUE_EXHAUSTED"),
        "blockers":run.status["blockers"],"return_zip":str(destination),
        "calibration":"FAILED/UNVALIDATED on existing local supports; unavailable sources stay pixel-only",
        "module_exposure":"Retrospective118576/128057 disclosed; G1/S4/external pretraining may have scene exposure, not claimed unseen"}
    write_json(run.output/"summary.json",summary)
    names=("src/backbone/identity_program.py","tests/test_identity_program.py")
    patch=[]
    for name in names:
        path=ROOT/name
        target=run.output/"changed"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        patch+=list(difflib.unified_diff([],path.read_text().splitlines(keepends=True),fromfile="/dev/null",tofile="b/"+name))
    (run.output/"source.patch").write_text("".join(patch),encoding="utf-8")
    command=f"& .\\.venv\\Scripts\\python.exe -m src.backbone.identity_program --config '{run.output.relative_to(ROOT)}\\config.json' --output '{run.output.relative_to(ROOT)}'"
    handoff=["# GPU identity program",f"Status: {terminal}; elapsed{summary['elapsed_hours']:.2f}h. RouteA, predecessor consumed unchanged.",
        "Separate appearance learning only. Local COCO-pretrained compact backbone; no person-ReID exposure certification. No detector/G1/S4 training.",
        f"Full halves: {summary['full_half_sources']}; unique evaluation minutes{summary['unique_evaluation_minutes']:.1f}. Corpus footage overlapping evaluation is not additional unique footage.",
        f"Selected: {summary['selection']}. Negative fits/routes retained in metrics. Learning completion is not identity improvement.",
        "See corpus_summary.json for supervision exclusions and shortages; association_grid_lock.json and selection.json freeze development settings before long scoring.",
        "scores_*.json retain switches/transition, false-relink/scorable, fragmentation/duration and11sample-history denominators. integration_*.json separates fixed anchors and causal cohort, all unsupported20s queries, per-minute availability and losses.",
        "No pitch coordinates are invented outside support. Unknown attributes stay unknown; G1 hidden estimates never feed S4. Visibility simulated; supplied-time action scores are not tactical accuracy.",
        "Large local artifacts: corpus/crops, R1/R2 checkpoints, features, detections, tracks, replays. Small rendered MP4s remain locally listed in media_*.json. Weights, dense predictions, crops and embeddings excluded from ZIP.",
        "Resume uses exclusive lock and unchanged absolute deadline. Main fits restore last completed epoch, not exact mid-batch cursor. Disposable process restart and CUDA next-update tests are separate evidence.",
        f"Blockers: {summary['blockers']}. Workstation sleep interrupts computation; no power settings changed.",
        "## Commands","```powershell",command+" run",command+" resume",command+" verify","```",
        "Terminal status was saved before ZIP creation. No commits/pushes/default changes."]
    (run.output/"handoff.md").write_text("\n".join(handoff)+"\n",encoding="utf-8")
    run.update(state=terminal,current_job=None,next_action="STOP",return_zip=str(destination),finished_utc=utc())
    with zipfile.ZipFile(destination,"x",zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.output.rglob("*")):
            relative=path.relative_to(run.output)
            if not path.is_file() or path.suffix not in (".json",".jpg",".txt",".xml",".md",".py",".patch"):
                continue
            if any(part in ("crops","corpus","features","detections","tracks","external","replays","snapshot") for part in relative.parts):
                continue
            if path.stat().st_size>8_000_000:
                continue
            archive.write(path,relative)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or json.loads(archive.read("status.json"))["state"]!=terminal:
            raise RuntimeError("ZIP or terminal status verification failed")
    return summary


def execute(run: Run) -> None:
    # All steps are local; no agent epoch supervision or competing GPU worker.
    with ExclusiveLock(run.output/"run.lock"):
        run.update(state="RUNNING",command=" ".join(sys.argv),sleep_warning="Workstation sleep interrupts local computation; settings unchanged",
                   launch={"pid":os.getpid(),"python":sys.executable,"cuda_available":torch.cuda.is_available()})
        snapshot=run.output/"snapshot"
        snapshot.mkdir(exist_ok=True)
        shutil.copy2(Path(__file__),snapshot/"identity_program.py")
        write_json(run.output/"environment.json",{"python":sys.version,"torch":torch.__version__,
            "cuda":torch.version.cuda,"ultralytics":__import__("ultralytics").__version__,
            "source_sha256":file_digest(Path(__file__)),"body_sha256":file_digest(BODY),
            "lifetime":"Detached process after verified disposable liveness test, no survival claim across sleep/reboot",
            "reproducibility":"Seed42; finite AMP update helper; complete-epoch restart checkpoint"})
        try:
            manifest=run.stage("manifest",lambda:source_manifest(run))
            if manifest is None:
                raise RuntimeError("No compatible source manifest")
            probe=run.stage("pilot",lambda:pilot(run))
            run.stage("interruption",lambda:interruption_check(run))
            corpus=run.stage("corpus",lambda:crop_corpus(run)) if probe is not None else None
            if corpus is not None and corpus["status"]=="CORPUS_VALID":
                for recipe in ("R1","R2"):
                    run.stage(recipe,lambda r=recipe:fit_encoder(run,r),optional=True)
            else:
                run.decision(stage="training",state="LABELS_UNAVAILABLE" if probe else "BLOCKED_CUDA_PILOT",
                             independent_pixel_evaluation=True)
            selected=run.stage("selection",lambda:select_tracker(run))
            if selected is None:
                # Preserve established HSV baseline, no unvalidated learned promotion.
                fallback={"status":"BASELINE_RETAINED_SELECTION_BLOCKED",
                    "selected":{"name":"baseline","encoder":"baseline","cosine":None},
                    "grid":[{"name":"baseline","encoder":"baseline","cosine":None},
                            *[{"name":f"{n}_c0.2","encoder":n,"cosine":.2} for n in available_encoders(run)]],
                    "metrics":{},"retrospective_used":False,"negative_result":True}
                write_json(run.output/"selection.json",fallback)
            sources=manifest["sources"]
            core=[next(s for s in sources if s["key"]==f"{m}_{h}") for m,h in run.config["core_sources"]]
            remaining=[s for s in sources if s not in core and not s["fitting_eligible"]]
            # Full fitting halves not already core are valuable frozen evaluations.
            remaining += [s for s in sources if s not in core and s["fitting_eligible"]]
            for source in core+remaining:
                run.check()
                key=source["key"]
                if source not in core and run.remaining()<3600:
                    run.decision(stage=key,state="DEFERRED_FINAL_RESERVE",remaining=run.remaining())
                    break
                detection=run.stage(f"detections_{key}",lambda s=source:predict_half(run,s))
                if detection is None:
                    continue
                association=run.stage(f"tracks_{key}",lambda s=source:associate_half(run,s))
                if association is None:
                    continue
                run.stage(f"scores_{key}",lambda s=source:score_half(run,s))
                run.stage(f"integration_{key}",lambda s=source:integrate_half(run,s))
                if source in core:
                    run.stage(f"media_{key}",lambda s=source:render_examples(run,s),optional=True)
            if run.remaining()>1800:
                run.stage("stress",lambda:stress_sequence(run),optional=True)
            run.update(stop_reason="USEFUL_AUTHORIZED_QUEUE_EXHAUSTED")
        except TimeoutError:
            run.update(stop_reason="FINAL_VERIFICATION_RESERVE",state="VERIFYING")
        except Exception as error:
            (run.output/"worker_error.txt").write_text(traceback.format_exc(),encoding="utf-8")
            run.status["blockers"].append({"stage":"orchestrator","error":str(error)})
            run.update(stop_reason="DEPENDENCY_BLOCKED",state="VERIFYING")
        try:
            result=verify(run)
            path=run.output/"verify_result.json"
            write_json(path,result)
            run.status["jobs"]["verify"]={"state":"COMPLETE","result":path.name,"result_sha256":file_digest(path)}
        except Exception as error:
            (run.output/"verification_error.txt").write_text(traceback.format_exc(),encoding="utf-8")
            run.status["blockers"].append({"stage":"verify","error":str(error)})
        package(run)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fixture-mode",choices=("full","first","resume"))
    parser.add_argument("command",choices=("init","pilot","run","resume","verify","restart-fixture","liveness"))
    args=parser.parse_args()
    output=args.output.resolve()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if args.command=="liveness":
        for i in range(8):
            write_json(output/"detached_liveness.json",{"pid":os.getpid(),"step":i,"utc":utc(),
                "status":"COMPLETE" if i==7 else "RUNNING","parent_pid":os.getppid()})
            time.sleep(2)
        return 0
    if args.command=="restart-fixture":
        fixture_process(output,args.fixture_mode)
        return 0
    if args.command=="init":
        initialize(output,args.config.resolve())
        return 0
    config=read(args.config)
    if json_hash(config)!=json_hash(read(output/"config.json")):
        raise ValueError("Configuration changed after initialization")
    run=Run(output,config)
    if args.command=="pilot":
        with ExclusiveLock(output/"run.lock"):
            manifest=run.stage("manifest",lambda:source_manifest(run))
            if manifest is not None:
                run.stage("pilot",lambda:pilot(run))
    elif args.command=="verify":
        with ExclusiveLock(output/"run.lock"):
            verify(run)
    else:
        execute(run)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
