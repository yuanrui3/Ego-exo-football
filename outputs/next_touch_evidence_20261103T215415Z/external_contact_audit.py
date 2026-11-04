import csv
from collections import Counter

import numpy as np

from common import RUN, write
from evidence import contact_signals, quantiles
from external import download, read_tracking
from possession_features import BallTrack


def run():
    folder=download()
    a=read_tracking(folder/"Sample_Game_1_RawTrackingData_Home_Team.csv")
    b=read_tracking(folder/"Sample_Game_1_RawTrackingData_Away_Team.csv")
    xy=np.concatenate([a["xy"],b["xy"]],axis=1);ids=a["ids"]+b["ids"]
    observed=np.isfinite(xy).all(2);ball=a["ball"]
    valid=np.isfinite(ball).all(1)&(np.abs(ball[:,0])<=52.5)&(np.abs(ball[:,1])<=34)
    raw={"xy":np.nan_to_num(xy),"observed":observed,"segments":np.ones(observed.shape,np.int32)}
    track=BallTrack(np.arange(len(ball)),np.nan_to_num(ball),valid,folder/"Sample_Game_1_RawTrackingData_Home_Team.csv","","PUBLIC_SAMPLE","")
    signals=contact_signals(raw,track)
    with (folder/"Sample_Game_1_RawEventsData.csv").open(newline="",encoding="utf-8-sig") as stream:
        events=list(csv.DictReader(stream))
    mapping={int(frame):i for i,frame in enumerate(a["frame"])}
    result={}
    for kind in ("PASS","SHOT"):
        counts=Counter();distances=[];lags={k:[] for k in ("proximity","intercept","changepoint")}
        for event in events:
            if event["Type"]!=kind:continue
            counts["source_events"]+=1
            q=mapping.get(int(event["Start Frame"]));actor=event["From"]
            if q is None or actor not in ids:counts["unmapped"]+=1;continue
            slot=ids.index(actor)
            if not valid[q] or not observed[q,slot]:counts["missing_ball_or_actor"]+=1;continue
            counts["eligible"]+=1;distances.append(float(np.linalg.norm(xy[q,slot]-ball[q])))
            span=np.arange(max(0,q-12),min(len(ball),q+13))
            span=span[a["period"][span]==a["period"][q]]
            for name in lags:
                hits=span[signals[name][span]&(signals["nearest"][span]==slot)]
                if len(hits):
                    best=hits[np.argmin(np.abs(hits-q))]
                    counts[name+"_same_actor_within_0p48s"]+=1
                    lags[name].append(float((best-q)/25))
        result[kind]={"counts":dict(counts),"actor_ball_distance_at_source_start_m":quantiles(distances),"same_actor_detection_rates":{name:counts[name+"_same_actor_within_0p48s"]/max(counts["eligible"],1) for name in lags},"candidate_time_lag_s":{name:quantiles(v) for name,v in lags.items()}}
    n=int(signals["valid"].sum())
    report={"status":"COMPLETE","claim":"INDEPENDENT_SOURCE_ANNOTATION_ASSOCIATION_NOT_VIDEO_TOUCH_CERTIFICATION","frames":len(ball),"valid_ball_frames":n,"event_types":dict(Counter(e["Type"] for e in events)),"candidate_frames":{name:int(signals[name].sum()) for name in ("proximity","intercept","changepoint")},"fraction_of_valid_frames":{name:float(signals[name].sum()/max(n,1)) for name in ("proximity","intercept","changepoint")},"associations":result,"limits":["Exact same three frozen algorithms as SoccerTrack; no parameter tuning on Metrica.","25Hz synchronization supplied by source, not a new independent video check.","Candidate frames are states and repeated bursts, not counts of distinct physical touches.","Sparse source actions do not certify every other frame as negative."]}
    write(RUN/"external_contact_association.json",report)
    return report


if __name__=="__main__":
    run()
