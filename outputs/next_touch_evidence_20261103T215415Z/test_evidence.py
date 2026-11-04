import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import common
import experiments
from evidence import contact_signals
from possession_features import BallTrack, load_ball


def fixture():
    xy=np.zeros((220,3,2),np.float32)
    xy[:,0]=[0,0];xy[:,1]=[10,0];xy[:,2]=[20,0]
    b=np.zeros((220,2),np.float32)
    b[:,0]=np.minimum(np.maximum(np.arange(220)-60,0)/25*10,20)
    raw={"xy":xy,"observed":np.ones((220,3),bool),"segments":np.ones((220,3),np.int32),"teams":np.tile([1,1,2],(220,1)),"roles":np.zeros((220,3),np.int8)}
    track=BallTrack(np.arange(220),b,np.ones(220,bool),Path("synthetic"),"","SYNTHETIC","")
    d={"target":np.zeros(3,int),"match":np.repeat("M",3),"half_code":np.ones(3,int),"query_frame":np.repeat(60,3),"actor_slot_local_only":np.arange(3),"segment_local_only":np.ones(3,int),"x":np.zeros((3,196),np.float32),"ball":np.zeros((3,12),np.float32)}
    return raw,track,d


def test_query_clock_and_no_future_ball():
    raw,track,d=fixture()
    _,_,age,valid=track.asof(np.array([60]))
    assert valid[0] and age[0]==0
    before=common.prior.aligned_state(raw,track,np.repeat(60,3),np.arange(3),np.ones(3,int))
    track.xy[61:]=9999;raw["xy"][61:]=-9999
    after=common.prior.aligned_state(raw,track,np.repeat(60,3),np.arange(3),np.ones(3,int))
    np.testing.assert_array_equal(before,after)


def test_prefix_equivalence():
    raw,track,d=fixture()
    full=common.prior.aligned_state(raw,track,np.repeat(60,3),np.arange(3),np.ones(3,int))
    prefix={k:v[:61] for k,v in raw.items()}
    shorter=BallTrack(track.frames[:61],track.xy[:61],track.valid[:61],track.path,"","SYNTHETIC","")
    np.testing.assert_array_equal(full,common.prior.aligned_state(prefix,shorter,np.repeat(60,3),np.arange(3),np.ones(3,int)))


def test_future_shift_rejected():
    raw,t,d=fixture()
    with pytest.raises(ValueError,match="Future"):
        common.prior.aligned_state(raw,t,np.repeat(60,3),np.arange(3),np.ones(3,int),lag=-1)


def test_segment_break_zeroes_state():
    raw,t,d=fixture();raw["segments"][60,1]=2
    s=common.prior.aligned_state(raw,t,np.repeat(60,3),np.arange(3),np.ones(3,int))
    assert not s[1].any()


def test_stale_is_joint_actor_ball():
    raw,t,d=fixture()
    q=np.repeat(60,3);slots=np.arange(3);seg=np.ones(3,int)
    a=common.prior.aligned_state(raw,t,q,slots,seg,lag=12)
    b=common.prior.aligned_state(raw,t,q-12,slots,seg)
    np.testing.assert_array_equal(a,b)


def test_missing_ball_not_origin_contact():
    raw,t,d=fixture();t.valid[:]=False
    s=common.prior.aligned_state(raw,t,np.repeat(60,3),np.arange(3),np.ones(3,int))
    assert not s.any()
    c=contact_signals(raw,t)
    assert not c["proximity"].any() and not c["intercept"].any() and not c["changepoint"].any()


def test_actor_permutation_equivariance():
    raw,t,d=fixture()
    q=np.repeat(60,3);slots=np.arange(3);seg=np.ones(3,int)
    base=common.prior.aligned_state(raw,t,q,slots,seg)
    perm=np.array([2,0,1])
    permuted=common.prior.aligned_state(raw,t,q,slots[perm],seg)
    np.testing.assert_array_equal(base[perm],permuted)


def test_duplicate_frame_blocks_ball(tmp_path):
    p=tmp_path/"duplicate.npz"
    np.savez(p,frame=np.array([1,1]),x=np.array([0,1]),y=np.array([0,0]))
    t=load_ball(p)
    assert t.status=="BAD_FRAME_INDEX" and not t.asof(np.array([1]))[-1][0]


def test_ball_pitch_bounds_not_clamped(tmp_path):
    p=tmp_path/"bounds.npz"
    np.savez(p,frame=np.arange(3),x=np.array([0,100,1]),y=np.array([0,0,0]))
    t=load_ball(p)
    assert not t.asof(np.array([1]))[-1][0]
    assert t.xy[1,0]==100


def test_raw_negative_ranking_not_probability():
    result=common.metrics(np.array([1,0,0]),np.array([-1.,-5.,-20.]))
    assert result["AP"]==1 and result["NLL"] is None
    with pytest.raises(ValueError,match="probabilities"):
        common.metrics(np.array([1,0,0]),np.array([-1.,-5.,-20.]),probability=np.array([-1.,-5.,-20.]))


def install_sources(monkeypatch,tmp_path,events=None):
    raw,t,d=fixture()
    monkeypatch.setattr(experiments,"RUN",tmp_path)
    monkeypatch.setattr(experiments,"FIT",("M",))
    monkeypatch.setattr(experiments,"source",lambda m,h:(raw,t))
    monkeypatch.setattr(experiments,"event_rows",lambda m,h,r:events or [])
    return raw,t,d


def test_future_geometry_is_label_only(monkeypatch,tmp_path):
    raw,t,d=install_sources(monkeypatch,tmp_path)
    y=experiments.targets(d)
    np.testing.assert_array_equal(y["geometry_1s"],[0,1,0])
    np.testing.assert_array_equal(y["geometry_2s"],[0,0,1])
    np.testing.assert_array_equal(d["target"],0)


def test_window_edges_and_same_actor(monkeypatch,tmp_path):
    events=[{"frame":60,"label":1,"known_actor":True,"slot":2,"segment":1},{"frame":85,"label":1,"known_actor":True,"slot":1,"segment":1},{"frame":86,"label":1,"known_actor":True,"slot":2,"segment":1}]
    raw,t,d=install_sources(monkeypatch,tmp_path,events)
    y=experiments.targets(d)
    np.testing.assert_array_equal(y["next_BAS_0_1s"],[0,1,0])
    np.testing.assert_array_equal(y["next_BAS_1_2s"],[0,0,1])


def test_censor_disappeared_actor(monkeypatch,tmp_path):
    raw,t,d=install_sources(monkeypatch,tmp_path)
    raw["observed"][85,2]=False
    y=experiments.targets(d)
    assert np.all(y["geometry_1s"]==-1)


def test_tied_annotated_actors_censored(monkeypatch,tmp_path):
    events=[{"frame":85,"label":1,"known_actor":True,"slot":s,"segment":1} for s in (1,2)]
    raw,t,d=install_sources(monkeypatch,tmp_path,events)
    assert np.all(experiments.targets(d)["next_BAS_0_1s"]==-1)


def test_proximity_signal_is_not_physical_truth():
    raw,t,d=fixture()
    c=contact_signals(raw,t)
    assert c["proximity"][30]
    assert c["nearest"][30]==0
    assert "physical_truth" not in c


def test_split_excludes_exposed_matches():
    assert set(common.FIT)=={"117092","117093","118576","118578","128057","132831"}
    assert not {"118577","128058","132877"} & set(common.FIT)


def test_distinct_event_deduplication():
    raw,t,d=fixture()
    d={k:np.concatenate([v,v],axis=0) for k,v in d.items()}
    d["query_frame"][3:]=61
    frames=np.array([85,-1,-1,85,-1,-1])
    groups=common.groups(d)
    result=common.prior.review_metrics([0,1],groups,d,np.array([1,0,0,1,0,0]),frames,1)
    assert result["unique_events_recalled"]==1 and result["reviewed_scenes"]==2


def test_geometry_tie_censored(monkeypatch,tmp_path):
    raw,t,d=install_sources(monkeypatch,tmp_path)
    t.xy[85]=[5,0]
    assert np.all(experiments.targets(d)["geometry_1s"]==-1)


def test_missing_future_ball_censored(monkeypatch,tmp_path):
    raw,t,d=install_sources(monkeypatch,tmp_path)
    t.valid[85]=False
    assert np.all(experiments.targets(d)["geometry_1s"]==-1)


def test_truncated_long_window_censored(monkeypatch,tmp_path):
    raw,t,d=install_sources(monkeypatch,tmp_path)
    y=experiments.targets(d,geometry_horizons=(10,),bas_windows=(("long",200,300),),cache_name="long_test")
    assert np.all(y["geometry_10s"]==-1) and np.all(y["long"]==-1)


def test_target_cache_rejects_roster_reorder(monkeypatch,tmp_path):
    raw,t,d=install_sources(monkeypatch,tmp_path)
    experiments.targets(d)
    d["actor_slot_local_only"]=d["actor_slot_local_only"][::-1]
    with pytest.raises(ValueError,match="cache identity"):
        experiments.targets(d)


def test_ambiguous_player_slot_excluded(monkeypatch):
    import build_cohort
    monkeypatch.setattr(common,"read",lambda p:{"slots":{"A":0,"B":0,"C":1}} if "associations_" in str(p) else {"gsr_info":{}})
    monkeypatch.setattr(build_cohort,"_event_rows",lambda *args:args[5])
    assert common.event_rows("M","1st",{"xy":np.zeros((2,2,2))})=={1}


def test_package_crc_member_and_archive_hash(tmp_path):
    import hashlib,json,zipfile
    from verify_package import verify
    path=tmp_path/"test.zip"
    payload=b"synthetic reproducibility test"
    manifest={"run/test.txt":hashlib.sha256(payload).hexdigest()}
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr("run/test.txt",payload)
        archive.writestr("run/MANIFEST.sha256.json",json.dumps(manifest))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path)+".sha256").write_text(digest+"  test.zip\n",encoding="ascii")
    result=verify(path)
    assert result["member_SHA256"]=="PASS" and result["CRC"]=="PASS"
    with pytest.raises(ValueError,match="Archive SHA-256"):
        verify(path,"0"*64)


def test_package_member_tamper_detected(tmp_path):
    import hashlib,json,zipfile
    from verify_package import verify
    path=tmp_path/"bad.zip"
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr("run/test.txt",b"modified")
        archive.writestr("run/MANIFEST.sha256.json",json.dumps({"run/test.txt":hashlib.sha256(b"original").hexdigest()}))
    with pytest.raises(ValueError,match="Member SHA-256"):
        verify(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_scheduler_scores_queries_with_unknown_future(monkeypatch,tmp_path):
    class Model:
        def __init__(self,**kwargs):pass
        def fit(self,x,y,**kwargs):
            assert set(y.tolist())=={False,True}
            return self
        def predict_proba(self,x):
            return np.tile([.8,.2],(len(x),1))
    monkeypatch.setattr(common,"HistGradientBoostingClassifier",Model)
    monkeypatch.setattr(common,"RUN",tmp_path)
    monkeypatch.setattr(common,"FIT",("A","B"))
    d={"match":np.array(["A"]*3+["B"]*3)}
    y=np.array([0,1,-1,0,1,-1])
    p,_=common.fit("all_queries",np.zeros((6,2)),d,y,all_queries=True)
    assert np.isfinite(p).all()
    p,_=common.fit("supported_only",np.zeros((6,2)),d,y)
    assert np.isnan(p[y<0]).all()


def test_unknown_future_does_not_change_finite_scheduler_scores(monkeypatch,tmp_path):
    class Model:
        def __init__(self,**kwargs):pass
        def fit(self,x,y,**kwargs):return self
        def predict_proba(self,x):
            p=.1+.1*x[:,0]
            return np.column_stack((1-p,p))
    monkeypatch.setattr(common,"HistGradientBoostingClassifier",Model)
    monkeypatch.setattr(common,"RUN",tmp_path)
    monkeypatch.setattr(common,"FIT",("A","B"))
    d={"match":np.array(["A"]*3+["B"]*3)}
    x=np.arange(6,dtype=float)[:,None]
    y=np.array([0,1,-1,0,1,-1])
    p,_=common.fit("unknown",x,d,y,all_queries=True)
    assert np.argmax(p)==5 and y[5]<0
