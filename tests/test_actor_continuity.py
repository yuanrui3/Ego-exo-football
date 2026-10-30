from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from src.backbone.actor_continuity import (
    Cohort, SessionAssociation, associate_sessions, due_queries, remap_external_row,
)
from src.backbone.contracts import Measurement, Observation, TrackingFrame
from src.backbone.contracts import ActionOutput
from src.backbone.memory import CausalMemory
from src.backbone.perception_continuation import same_detections
from src.backbone.pipeline import ReplayEngine


FEATURE = [1.]+[0.]*31


def frame(i, points, *, timestamp=None, cut=False):
    return {"frame_index":i*5,"timestamp_s":i*.2 if timestamp is None else timestamp,
        "camera_cut":cut,"detections":[
            {"track":f"old-{j}","footpoint_xy_px":[float(x),100.],
             "bbox_xyxy_px":[x-3,90.,x+3,100.],"detector_confidence":.9}
            for j,x in enumerate(points)]}


def features(points):
    return [FEATURE for _ in points]


def test_brief_occlusion_requires_motion_and_appearance_and_preserves_session():
    tracker=SessionAssociation(4096)
    ids=[]
    for i in range(6):
        ids.append(tracker.step(frame(i,[100+i*2]),features([1]))["detections"][0]["track"])
    tracker.step(frame(6,[]),[])
    recovered=tracker.step(frame(7,[114]),features([1]))["detections"][0]
    assert recovered["track"]==ids[0]
    assert recovered["association_quality"]["status"]=="JUSTIFIED_GAP_RELINK"
    assert recovered["tracklet_id"]=="old-0"


def test_same_kit_alone_cannot_authorize_long_or_motion_inconsistent_relink():
    tracker=SessionAssociation(4096)
    for i in range(6):
        prior=tracker.step(frame(i,[100]),features([1]))["detections"][0]["track"]
    distant=tracker.step(frame(8,[170]),features([1]))["detections"][0]
    assert distant["track"]!=prior
    expired=tracker.step(frame(20,[170]),features([1]))["detections"][0]
    assert expired["track"]!=distant["track"]


def test_similar_kit_ambiguous_crossing_abstains_instead_of_splicing():
    tracker=SessionAssociation(4096)
    original=tracker.step(frame(0,[100,120]),features([1,1]))
    returned=tracker.step(frame(1,[110,111]),features([1,1]))
    old={d["track"] for d in original["detections"]}
    assert all(d["track"] not in old for d in returned["detections"])
    assert all(d["association_quality"]["status"]=="AMBIGUOUS_ABSTAIN_NEW_SESSION" for d in returned["detections"])


def test_ambiguous_gap_return_abstains():
    tracker=SessionAssociation(4096)
    for i in range(6):
        original=tracker.step(frame(i,[100,120]),features([1,1]))
    tracker.step(frame(6,[]),[])
    returned=tracker.step(frame(7,[110]),features([1]))
    assert returned["detections"][0]["track"] not in {d["track"] for d in original["detections"]}


def test_camera_cut_and_irregular_clock_use_elapsed_time_not_sample_count():
    tracker=SessionAssociation(4096)
    first=tracker.step(frame(0,[100]),features([1]))["detections"][0]["track"]
    cut=tracker.step(frame(1,[100],cut=True),features([1]))["detections"][0]["track"]
    assert first!=cut
    late=tracker.step(frame(2,[100],timestamp=10.),features([1]))["detections"][0]["track"]
    assert late!=cut
    with pytest.raises(ValueError,match="increasing"):
        tracker.step(frame(3,[100],timestamp=10.),features([1]))


def test_irregular_timestamp_motion_gate_does_not_use_frame_index_age():
    tracker=SessionAssociation(4096)
    for i in range(6):
        original=tracker.step(frame(i,[100+i]),features([1]))
    near=tracker.step(frame(100,[106],timestamp=1.2),features([1]))
    assert near["detections"][0]["track"]==original["detections"][0]["track"]


def test_prefix_consistency_no_label_input_and_detection_parity():
    saved=[frame(i,[100+i*2,300-i]) for i in range(20)]
    copied=copy.deepcopy(saved)
    for f in copied:
        f["reference_identity"]="FORBIDDEN_UNREAD_SENTINEL"
        for d in f["detections"]:
            d["team"]="DO_NOT_USE"
    feature=[features(f["detections"]) for f in saved]
    full=associate_sessions(saved,feature,4096)
    prefix=associate_sessions(saved[:8],feature[:8],4096)
    assert full[:8]==prefix
    leaked=associate_sessions(copied,feature,4096)
    assert [[d["track"] for d in f["detections"]] for f in leaked]==[[d["track"] for d in f["detections"]] for f in full]
    same_detections(saved,full)
    assert saved[0]["detections"][0]["track"]=="old-0"


def test_projection_loss_does_not_retire_image_session_or_invent_position():
    tracker=SessionAssociation(4096)
    session=None
    for i in range(15):
        predicted=tracker.step(frame(i,[100+i]),features([1]))
        session=session or predicted["detections"][0]["track"]
        row={"frame_index":i*5,"measurements":[{"track":"short","position_status":"missing" if 4<=i<10 else "observed",
            "xy":None if 4<=i<10 else [20.,20.],"team":"UNKNOWN","goalkeeper":None}]}
        changed=remap_external_row(predicted,row)
        assert changed["measurements"][0]["track"]==session
        assert changed["measurements"][0]["xy"]==row["measurements"][0]["xy"]
        assert changed["measurements"][0]["team"]=="UNKNOWN"
        assert row["measurements"][0]["track"]=="short"


def observation(index, *, missing=False):
    measurements=() if missing else (Measurement("actor",0,(1.,2.),1,False),)
    return Observation("fixture","1st",index*5,index*.2,"actor",measurements)


def test_genuine_actor_gap_requires_eleven_fresh_samples_not_relinked_history():
    models=SimpleNamespace(world=None,backend="last_seen")
    memory=CausalMemory(models)
    for i in range(11):
        memory.step(observation(i))
    assert memory.current_input is not None
    memory.step(observation(11,missing=True))
    assert memory.current_input is None
    for i in range(12,22):
        memory.step(observation(i))
        assert memory.current_input is None
    memory.step(observation(22))
    assert memory.current_input is not None


def tracking(i, tracks, *, discontinuity=False):
    return TrackingFrame("fixture","1st",i*5,i*.2,
        tuple(Measurement(t,0,(1.,2.),1,False) for t in tracks),0,discontinuity)


def test_cohort_is_causal_each_admission_separate_and_warms_up():
    cohort=Cohort()
    prefix=[cohort.step(tracking(i,["a"])) for i in range(15)]
    assert all(row==[None,None] for row in prefix[:10])
    assert prefix[10]==["a",None]
    for i in range(15,35):
        last=cohort.step(tracking(i,["b"]))
    assert "b" in last and "a" not in last
    assert {r["session"] for r in cohort.admissions}=={"a","b"}
    assert cohort.step(tracking(35,["a"],discontinuity=True))==[None,None]


def test_cohort_prefix_invariance_and_unknown_attributes_never_admitted():
    def run(count):
        cohort=Cohort()
        return [cohort.step(tracking(i,["a","b"] if i>30 else ["a"])) for i in range(count)]
    assert run(20)==run(45)[:20]
    cohort=Cohort()
    for i in range(20):
        row=TrackingFrame("fixture","1st",i*5,i*.2,(Measurement("u",0,(1.,2.),0,None),),0)
        assert cohort.step(row)==[None,None]


def test_full_duration_schedule_keeps_unsupported_attempts_and_missing_clock_ticks():
    attempted=[]
    previous=-1e-6
    for i in range(3000):
        attempted.extend(due_queries(previous,i*.2,20))
        previous=i*.2
    assert attempted==list(range(20,600,20))
    assert due_queries(19.,61.,20)==[20.,40.,60.]
    for period in (0,-1,float("nan")):
        with pytest.raises(ValueError):
            due_queries(0,20,period)
    engine=ReplayEngine(SimpleNamespace(world=None,backend="last_seen"))
    row=engine.step(observation(0,missing=True),include_passing_assessment=False,action_query=True)
    assert row["action"]["support_status"]=="unsupported"
    assert not row["belief"]["actor_available"]


def test_remapping_rejects_misaligned_external_frames():
    with pytest.raises(ValueError,match="aligned"):
        remap_external_row(frame(1,[100]),{"frame_index":0,"measurements":[]})


def test_real_engine_retains_each_full_duration_unsupported_query():
    class UnsupportedAction:
        def __init__(self):
            self.attempts=0

        def predict(self,inputs):
            assert inputs is None
            self.attempts+=1
            return SimpleNamespace(
                output=ActionOutput(reason="Missing genuine actor history"),
                to_dict=lambda:{"status":"unsupported","reason":"Missing genuine actor history","scores":None})
    action=UnsupportedAction()
    engine=ReplayEngine(SimpleNamespace(world=None,backend="last_seen"),action_model=action)
    previous=-1e-6
    results=[]
    for i in range(3000):
        due=due_queries(previous,i*.2,20)
        previous=i*.2
        if due:
            row=engine.step(observation(i,missing=True),include_passing_assessment=False,action_query=True)
            results.append(row["action_inference"]["status"])
    assert action.attempts==29
    assert results==["unsupported"]*29
