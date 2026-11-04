import csv
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from common import RUN, FIT, PARAMS, prior, fit, write, digest

COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
FILES = {
    "Sample_Game_1_RawEventsData.csv": 139446,
    "Sample_Game_1_RawTrackingData_Away_Team.csv": 32782073,
    "Sample_Game_1_RawTrackingData_Home_Team.csv": 32782961,
}
GAME2_FILES = {
    "Sample_Game_2_RawEventsData.csv": (153897,"decccd47c1d53960d290cb6679337eec83a8a97f"),
    "Sample_Game_2_RawTrackingData_Away_Team.csv": (29627711,"e7334d8d762954a215c49e55f211855000243fdc"),
    "Sample_Game_2_RawTrackingData_Home_Team.csv": (31890338,"8b4d3648888cc8b033e7142919ebaea04f6328e0"),
}


def download(game=1):
    folder = RUN / ("external_local" if game==1 else f"external_game{game}_local")
    folder.mkdir(exist_ok=True)
    records = []
    files=FILES if game==1 else {name:value[0] for name,value in GAME2_FILES.items()}
    for name, size in files.items():
        path = folder / name
        url = f"https://raw.githubusercontent.com/metrica-sports/sample-data/{COMMIT}/data/Sample_Game_{game}/{name}"
        if not path.exists() or path.stat().st_size != size:
            part = path.with_suffix(".part")
            for attempt in range(3):
                offset = part.stat().st_size if part.exists() else 0
                req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-", "User-Agent": "football-local-research"})
                try:
                    with urllib.request.urlopen(req, timeout=90) as response:
                        append = response.status == 206 and offset > 0
                        with part.open("ab" if append else "wb") as stream:
                            while block := response.read(1024 * 1024):
                                stream.write(block)
                    if part.stat().st_size != size:
                        raise ValueError(f"{name}: expected {size} bytes, got {part.stat().st_size}")
                    part.replace(path)
                    break
                except (OSError, ValueError) as exc:
                    write(RUN / "external_download_status.json", {"file":name,"attempt":attempt+1,"error":str(exc)})
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        blob=None
        if game==2:
            h=hashlib.sha1(f"blob {size}\0".encode())
            with path.open("rb") as stream:
                while block:=stream.read(8*1024*1024):h.update(block)
            blob=h.hexdigest()
            if blob!=GAME2_FILES[name][1]:raise ValueError("Official Game2 Git blob SHA mismatch")
        records.append({"name":name,"bytes":size,"sha256":digest(path),"official_git_blob_sha1":blob,"url":url})
    write(RUN / ("external_sources.json" if game==1 else f"external_sources_game{game}.json"), {"commit":COMMIT,"files":records,"scope":"Anonymous public synchronized sample; README community sample permission, no general sublicensing assertion; private raw data excluded from package."})
    return folder


def read_tracking(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header0, header1, header2 = next(reader), next(reader), next(reader)
        rows = list(reader)
    values = np.asarray([[float(x) if x else np.nan for x in row] for row in rows], dtype=np.float64)
    player_cols = [i for i in range(3, len(header2)-2, 2) if header1[i]]
    ids = ["Player" + header1[i] for i in player_cols]
    xy = np.stack([values[:, i:i+2] for i in player_cols], axis=1)
    ball = values[:, -2:]
    xy = (xy - .5) * np.asarray([105.,68.])
    ball = (ball - .5) * np.asarray([105.,68.])
    return {"period":values[:,0].astype(int),"frame":values[:,1].astype(int),"seconds":values[:,2],"xy":xy.astype(np.float32),"ball":ball.astype(np.float32),"ids":ids,"headers":[header0,header1,header2]}


def compact(state, seq, scenes):
    n = len(state)
    x = np.zeros((n,196),np.float32)
    dummy = {"x":x, "ball":state, "match":np.repeat("Metrica1",n), "half_code":np.ones(n,int), "query_frame":scenes}
    kin, geo = prior.dynamics(dummy, seq)
    return np.concatenate([state,kin,geo],axis=1).astype(np.float32)


def external_arrays(folder, quantize_players=False, quantize_ball=False, game=1):
    a = read_tracking(folder / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv")
    b = read_tracking(folder / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv")
    if not np.array_equal(a["frame"], b["frame"]) or not np.array_equal(a["period"], b["period"]):
        raise ValueError("Metrica team clocks differ")
    both_ball = np.isfinite(a["ball"]).all(1) & np.isfinite(b["ball"]).all(1)
    if np.max(np.abs(a["ball"][both_ball]-b["ball"][both_ball]),initial=0) > .001:
        raise ValueError("Team ball streams disagree")
    xy = np.concatenate([a["xy"],b["xy"]],axis=1)
    ids = a["ids"]+b["ids"]
    if len(set(ids)) != len(ids):
        raise ValueError("Metrica player names are not globally unique")
    ball = a["ball"]
    ball_valid = np.isfinite(ball).all(1) & (np.abs(ball[:,0])<=52.5) & (np.abs(ball[:,1])<=34)
    observed = np.isfinite(xy).all(2)
    # Sample each second with a complete two-second target window in the same period.
    queries = np.arange(60,len(xy)-51,25)
    queries = queries[a["period"][queries-50] == a["period"][queries+50]]
    rows=[]; scenes=[]; slots=[]; states=[]; sequences=[]
    labels={h:[] for h in (1,2)}
    now=[]
    annotation=[]; event_negative=[]; label_rows=[]
    with (folder / f"Sample_Game_{game}_RawEventsData.csv").open(newline="",encoding="utf-8-sig") as stream:
        events=list(csv.DictReader(stream))
    passes=[e for e in events if e["Type"]=="PASS" and e["From"] in ids]
    frame_lookup={int(f):i for i,f in enumerate(a["frame"])}
    event_q=[(frame_lookup.get(int(e["Start Frame"])), ids.index(e["From"]),e) for e in passes]
    event_q=[e for e in event_q if e[0] is not None]
    from possession_features import BallTrack
    feature_xy=(np.round(xy/[1.05,.68])*[1.05,.68]).astype(np.float32) if quantize_players else xy
    feature_ball=(np.round(ball/[1.05,.68])*[1.05,.68]).astype(np.float32) if quantize_ball else ball
    tracking={"xy":feature_xy,"observed":observed,"segments":np.ones(observed.shape,np.int32)}
    track=BallTrack(np.arange(len(ball)),np.nan_to_num(feature_ball),ball_valid,folder/f"Sample_Game_{game}_RawTrackingData_Home_Team.csv","","PUBLIC_SAMPLE","")
    def state_at(q, roster):
        return prior.aligned_state(tracking,track,np.repeat(q,len(roster)),roster,np.ones(len(roster),int))
    rejected={"missing_ball":0,"missing_future":0,"target_not_current_roster":0,"tie":0}
    for sid,q in enumerate(queries):
        roster=np.flatnonzero(observed[q])
        if len(roster)<2 or not ball_valid[q]:
            rejected["missing_ball"]+=1
            continue
        ds=np.linalg.norm(xy[q,roster]-ball[q],axis=1)
        near=int(roster[np.argmin(ds)])
        future={}
        for horizon in (1,2):
            f=q+25*horizon
            avail=np.flatnonzero(observed[f])
            if not ball_valid[f] or len(avail)<2:
                future[horizon]=None;rejected["missing_future"]+=1;continue
            distances=np.linalg.norm(xy[f,avail]-ball[f],axis=1)
            order=np.argsort(distances)
            win=int(avail[order[0]])
            if win not in roster:
                future[horizon]=None;rejected["target_not_current_roster"]+=1;continue
            if distances[order[1]]-distances[order[0]]<=.2:
                future[horizon]=None;rejected["tie"]+=1;continue
            future[horizon]=win
        pass_future=[(f,s) for f,s,e in event_q if q<f<=q+25 and a["period"][f]==a["period"][q]]
        first=min(pass_future,default=(None,None))
        anno=first[1] if first[1] in roster else None
        current=state_at(q,roster)
        seq=np.stack([state_at(q+lag,roster)[:,:8] for lag in (-50,-40,-30,-20,-10,0)],axis=1)
        states.append(current);sequences.append(seq)
        scenes.extend([sid]*len(roster));slots.extend(roster.tolist());rows.extend([int(q)]*len(roster))
        now.extend((roster==near).astype(int).tolist())
        for horizon in (1,2):
            labels[horizon].extend([-1]*len(roster) if future[horizon] is None else (roster==future[horizon]).astype(int).tolist())
        annotation.extend([-1]*len(roster) if pass_future and anno is None else (roster==anno).astype(int).tolist())
        event_negative.extend([not pass_future]*len(roster))
    state=np.concatenate(states);seq=np.concatenate(sequences)
    return {"state":state,"seq":seq,"kinematic":compact(state,seq,np.array(scenes)),"scene":np.array(scenes),"actor":np.array(slots),"frame":np.array(rows),"now":np.array(now),"geometry_1s":np.array(labels[1]),"geometry_2s":np.array(labels[2]),"next_pass_actor_0_1s":np.array(annotation),"ids":ids,"rejected":rejected,"events":events,"raw":a}


def external_study(game=1):
    from experiments import compact_features, targets, rules
    folder=download(game)
    ext=external_arrays(folder,game=game)
    d=prior.data("fit")
    c=compact_features(d); tar=targets(d)
    models={}
    report={}
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    for target in ("geometry_1s","geometry_2s","next_BAS_0_1s"):
        external_target=target if target.startswith("geometry") else "next_pass_actor_0_1s"
        y=ext[external_target]
        valid=y>=0
        train_y=tar[target];train_valid=train_y>=0
        scores={}
        for family in ("state","kinematic"):
            model=HistGradientBoostingClassifier(**PARAMS)
            with threadpool_limits(limits=8):
                model.fit(c[family][train_valid],train_y[train_valid])
                score=model.predict_proba(ext[family])[:,1]
            scores[family]=score
        scores["nearest"]=-ext["state"][:,4]*105
        horizon=1 if target!="geometry_2s" else 2
        rel=ext["state"][:,2:4]*[105,68]
        # Kinematics are reconstructed by the identical training feature producer.
        kin=ext["kinematic"][:,12:]
        scores["closest_approach"]=-kin[:,8]*105
        scores["constant_velocity"]=-np.linalg.norm(rel+kin[:,12:14]*40*horizon,axis=1)
        report[target]={"external_target":external_target,"rows":int(valid.sum()),"scenes":int(len(np.unique(ext["scene"][valid]))),"positive_rows":int(y[valid].sum()),"models":{}}
        for name,score in scores.items():
            scene_groups=prior.groups({"match":np.repeat("M",int(valid.sum())),"half_code":np.ones(int(valid.sum()),int),"query_frame":ext["scene"][valid]})
            m=prior.metrics(y[valid],score[valid],scene_groups,probability=score[valid] if name in ("state","kinematic") else None)
            switch=valid.copy()
            if target.startswith("geometry"):
                switch_scenes=np.unique(ext["scene"][valid & (y==1) & (ext["now"]==0)])
                switch&=np.isin(ext["scene"],switch_scenes)
            switch_groups=prior.groups({"match":np.repeat("M",int(switch.sum())),"half_code":np.ones(int(switch.sum()),int),"query_frame":ext["scene"][switch]})
            report[target]["models"][name]={"all":m,"switch_only":prior.metrics(y[switch],score[switch],switch_groups,probability=score[switch] if name in ("state","kinematic") else None) if switch.any() else {}}
        np.savez_compressed(folder/f"{target}_predictions_mapping2.npz",y=y,scene=ext["scene"],**scores)
    result={"status":"COMPLETE","mapping_revision":2,"claim":"INDEPENDENT_DOMAIN_PILOT_NOT_SEALED","sample":f"Metrica Game{game}, frozen pretrained SoccerTrack six-match models, no Metrica fitting/selection","source_rows":len(ext["raw"]["frame"]),"player_slots":len(ext["ids"]),"event_count":len(ext["events"]),"censoring":ext["rejected"],"results":report,"limits":["Anonymous domain sample, not a population generalization guarantee.","Source PASS differs from SoccerTrack contact-like BAS taxonomy; task names separate.","Provided events synchronized by source; no independent visual physical-contact adjudication.","Unknown velocity channels are explicitly zeroed, no future imputation.","Mapping revision2 repairs Game1 source-denominator and pitch-mask mismatches after Game1 revision1 outcomes were seen; Game2 mapping was frozen before download.","External cohort conditions on available current ball and current roster; not a full-game missing-ball deployment score."]}
    old=RUN/("external_results.json" if game==1 else f"external_results_game{game}.json")
    if old.exists() and "mapping_revision" not in json.loads(old.read_text(encoding="utf-8")):
        old.replace(RUN/"external_results_mapping1_invalid.json")
    write(old,result)
    return result


if __name__=="__main__":
    download()
