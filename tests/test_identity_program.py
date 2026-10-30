from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.backbone.identity_program import (
    AppearanceEncoder, CausalBaseline, CropDataset, ExclusiveLock, IdentityBatches,
    LearnedAssociation, conservative_pairs, crop_assignment, fixture_process,
    image_tensor, metric_loss, split_at,
)
from src.backbone.perception_continuation import associate_focused, same_detections
from src.trajectory_dataset import write_json


def test_fallback_routes_do_not_require_failed_development_metrics(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.backbone.identity_program as program
    route={"name":"pretrained_c0.2","encoder":"pretrained","cosine":.2}
    write_json(tmp_path/"selection.json",{
        "selected":{"name":"baseline","encoder":"baseline","cosine":None},
        "grid":[route],"metrics":{},
    })
    monkeypatch.setattr(program,"available_encoders",lambda run:{"pretrained":None})
    assert program.long_routes(SimpleNamespace(output=tmp_path))==[
        {"name":"baseline","encoder":"baseline","cosine":None},route,
    ]


def test_source_clock_failure_preserves_exclusion_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.backbone.identity_program as program
    def incompatible(*args):
        raise ValueError("VIDEO_ANNOTATION_CLOCK_MISMATCH")
    monkeypatch.setattr(program,"scan_source",incompatible)
    run=SimpleNamespace(output=tmp_path,check=lambda:None,
        config={"sources":[["m","1st"]],"core_sources":[["m","1st"]]})
    with pytest.raises(ValueError,match="Core source"):
        program.source_manifest(run)
    assert json.loads((tmp_path/"source_manifest.json").read_text())["excluded"][0]["reason"]=="VIDEO_ANNOTATION_CLOCK_MISMATCH"


def frame(i,points,cut=False,time=None):
    return {"frame_index":i*5,"timestamp_s":i*.2 if time is None else time,"camera_cut":cut,
        "detections":[{"track":str(j),"footpoint_xy_px":[x,100.],
            "bbox_xyxy_px":[x-5,60.,x+5,100.],"detector_confidence":.9} for j,x in enumerate(points)]}


def body(x,width=20,height=60):
    return {"bbox_xyxy_px":[x-width/2,100-height,x+width/2,100.],
            "footpoint_xy_px":[x,100.],"detector_confidence":.9}


def label(x,identity="a"):
    return {"footpoint_xy_px":[x,100.],"track_id":identity,"team":"left"}


def test_body_supervision_reciprocal_and_ambiguous_pairs_rejected():
    pairs,counts=conservative_pairs([body(100),body(300)],[label(101),label(299,"b")],4096)
    assert len(pairs)==2 and counts["accepted"]==2
    pairs,counts=conservative_pairs([body(100),body(105)],[label(102)],4096)
    assert not pairs
    pairs,counts=conservative_pairs([body(100,5,10)],[label(100)],4096)
    assert not pairs and counts["small_or_not_body"]==1


def test_person_crop_runtime_never_consumes_annotation_identity():
    predicted=[body(100),body(300)]
    crops=crop_assignment(predicted,[body(101),body(299)],4096)
    assert all(c is not None for c in crops)
    predicted[0]["player_id"]="FORBIDDEN_UNREAD"
    assert crop_assignment(predicted,[body(101),body(299)],4096)==crops
    assert crop_assignment([body(100)],[body(99),body(102)],4096)==[None]


def test_whole_source_and_temporal_buffers_preserve_exclusions():
    source={"fitting_eligible":True,"key":"117093_1st","duration_seconds":2700}
    assert split_at(source,1500)=="fit"
    assert split_at(source,1620)=="development"
    assert split_at(source,1610)=="buffer"
    assert split_at(source,2470)=="development"
    assert split_at(source,2450)=="buffer"
    assert split_at(source,2699)=="buffer"
    source["fitting_eligible"]=False
    assert split_at(source,1500)=="evaluation"


def test_metric_loss_demands_real_positive_and_negative_evidence():
    x=torch.nn.functional.normalize(torch.randn(8,16),dim=1)
    labels=torch.tensor([0,0,1,1,2,2,3,3])
    assert torch.isfinite(metric_loss(x,labels))
    with pytest.raises(ValueError):
        metric_loss(x,torch.arange(8))


def test_matched_sampler_epoch_reproducibility_and_samekit_hard_negatives():
    records=[{"identity":str(i),"source":"s","team":"left" if i<8 else "right"}
             for i in range(20) for _ in range(8)]
    for hard in (False,True):
        first=list(IdentityBatches(records,epoch=3,hard=hard))
        second=list(IdentityBatches(records,epoch=3,hard=hard))
        assert first==second
        assert first!=list(IdentityBatches(records,epoch=4,hard=hard))
        assert len(first[0])==64
        counts={}
        for index in first[0]:
            key=records[index]["identity"]
            counts[key]=counts.get(key,0)+1
        assert all(v==4 for v in counts.values())


def test_baseline_matches_predecessor_regular_clock_and_saved_state():
    saved=[frame(i,[] if i==12 else [100+i*2,400-i]) for i in range(40)]
    hsv=[[[1.,0.],[0.,1.]] if f["detections"] else [] for f in saved]
    reference=associate_focused(saved,4096,"damped_appearance",hsv)
    tracker=CausalBaseline(4096)
    actual=[tracker.step(f,v) for f,v in zip(saved,hsv,strict=True)]
    assert [[d["track"] for d in f["detections"]] for f in actual]==[[d["track"] for d in f["detections"]] for f in reference]
    same_detections(reference,actual)
    before=CausalBaseline(4096)
    prefix=[before.step(f,v) for f,v in zip(saved[:17],hsv[:17],strict=True)]
    restored=CausalBaseline(4096)
    restored.load_state_dict(copy.deepcopy(before.state_dict()))
    after=[restored.step(f,v) for f,v in zip(saved[17:],hsv[17:],strict=True)]
    assert prefix+after==actual


def test_learned_recovery_needs_both_motion_and_nonambiguous_appearance():
    tracker=LearnedAssociation(4096,.2)
    for i in range(7):
        old=tracker.step(frame(i,[100+i]),[[1.,0.]])["detections"][0]["track"]
    tracker.step(frame(7,[]),[])
    row=tracker.step(frame(8,[108]),[[1.,0.]])
    assert row["detections"][0]["track"]==old
    assert row["detections"][0]["association_quality"]["status"]=="APPEARANCE_GAP_RELINK"
    row=tracker.step(frame(10,[180]),[[1.,0.]])
    assert row["detections"][0]["track"]!=old


def test_ambiguous_samekit_crossing_and_camera_cut_abstain():
    tracker=LearnedAssociation(4096,.2)
    first=tracker.step(frame(0,[100,120]),[[1.,0.],[1.,0.]])
    second=tracker.step(frame(1,[110,111]),[[1.,0.],[1.,0.]])
    assert not {d["track"] for d in first["detections"]}&{d["track"] for d in second["detections"]}
    assert all(d["association_quality"]["status"]=="AMBIGUOUS_ABSTAIN" for d in second["detections"])
    cut=tracker.step(frame(2,[110],cut=True),[[1.,0.]])
    assert cut["detections"][0]["track"] not in {d["track"] for d in second["detections"]}
    with pytest.raises(ValueError):
        tracker.step(frame(3,[110],time=.4),[[1.,0.]])


def test_prefix_and_label_isolation_and_fixed_detector_parity():
    saved=[frame(i,[100+i,300-i]) for i in range(20)]
    def run(rows):
        tracker=LearnedAssociation(4096,.2)
        return [tracker.step(row,[[1.,0.],[0.,1.]]) for row in rows]
    full=run(saved)
    assert run(saved[:9])==full[:9]
    labelled=copy.deepcopy(saved)
    for f in labelled:
        f["reference"]="DO_NOT_USE"
        for d in f["detections"]:
            d["player_id"]="UNTRUSTED"
    result=run(labelled)
    assert [[d["track"] for d in f["detections"]] for f in result]==[[d["track"] for d in f["detections"]] for f in full]
    same_detections(saved,full)


def test_disposable_tracker_process_restart_outputs_exactly(tmp_path):
    directory=tmp_path/"interruption"
    directory.mkdir()
    fixture=[frame(i,[] if i==11 else [100+i,400-i],cut=i==25) for i in range(35)]
    write_json(directory/"fixture.json",{"frames":fixture})
    for mode in ("full","first","resume"):
        fixture_process(tmp_path,mode)
    assert (directory/"full.jsonl").read_text()==(directory/"resumed.jsonl").read_text()


def test_exclusive_run_lock_rejects_second_owner(tmp_path):
    path=tmp_path/"run.lock"
    with ExclusiveLock(path):
        with pytest.raises(RuntimeError,match="RUN_ALREADY_ACTIVE"):
            with ExclusiveLock(path):
                pytest.fail("Second owner acquired lock")
    with ExclusiveLock(path):
        pass


def test_encoder_initialization_and_image_shape_and_protected_weights():
    from src.backbone.identity_program import BODY, file_digest
    before=file_digest(BODY)
    torch.manual_seed(42)
    model=AppearanceEncoder(4).eval()
    x=image_tensor(np.zeros((40,20,3),np.uint8))[None].repeat(2,1,1,1)
    with torch.no_grad():
        embedding,logits=model(x)
    assert embedding.shape==(2,256) and logits.shape==(2,4)
    assert torch.isfinite(embedding).all()
    assert before==file_digest(BODY)
