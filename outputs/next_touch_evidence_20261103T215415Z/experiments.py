from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import average_precision_score

from common import ROOT, RUN, PRIOR, FIT, PARAMS, prior, source, event_rows, fit, evaluate, groups, subset, metrics, contrast, bas_targets, read, write, save, log


def row_signature(d):
    import hashlib
    h=hashlib.sha256()
    for key in ("match","half_code","query_frame","actor_slot_local_only","segment_local_only"):
        h.update(np.asarray(d[key]).tobytes())
    return h.hexdigest()


def compact_features(d, lag=0):
    """Fixed causal query roster; no future sampling or target-dependent filtering."""
    suffix="" if set(np.unique(d["match"]))<=set(FIT) else "_development"
    cache=RUN/"features_local"/f"compact_lag{lag}{suffix}.npz"
    signature=row_signature(d)+f":lag={lag}:compact-v1"
    meta=cache.with_suffix(".json")
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:loaded={k:z[k] for k in z.files}
        if meta.exists():
            saved=read(meta)
            if saved["signature"]!=signature or saved["sha256"]!=prior.digest(cache):
                raise ValueError("Compact feature cache identity/checksum mismatch")
        else:
            if len(loaded["state"])!=len(d["target"]):
                raise ValueError("Compact cache row count mismatch")
            for index in np.linspace(0,len(d["target"])-1,min(48,len(d["target"]))).astype(int):
                match=str(d["match"][index]);half="1st" if d["half_code"][index]==1 else "2nd"
                raw,track=source(match,half)
                expected=prior.aligned_state(raw,track,d["query_frame"][index:index+1],d["actor_slot_local_only"][index:index+1],d["segment_local_only"][index:index+1],lag=lag)
                np.testing.assert_array_equal(expected[0],loaded["state"][index])
            write(meta,{"signature":signature,"sha256":prior.digest(cache),"legacy_cache_native_checks":min(48,len(d["target"]))})
        return loaded
    n=len(d["target"])
    seq=np.zeros((n,6,8),np.float32)
    state=np.zeros((n,12),np.float32)
    actor=np.zeros((n,196),np.float32)
    for match in np.unique(d["match"]):
        for halfcode,half in ((1,"1st"),(2,"2nd")):
            mask=(d["match"]==match)&(d["half_code"]==halfcode)
            if not mask.any():continue
            raw,track=source(str(match),half)
            q=d["query_frame"][mask].astype(int)
            slot=d["actor_slot_local_only"][mask].astype(int)
            seg=d["segment_local_only"][mask].astype(int)
            state[mask]=prior.aligned_state(raw,track,q,slot,seg,lag=lag)
            for j,back in enumerate((50,40,30,20,10,0)):
                seq[mask,j]=prior.aligned_state(raw,track,q,slot,seg,lag=lag+back)[:,:8]
    actor[:,:2]=state[:,8:10]
    actor[:,2:4]=state[:,10:12]
    dd={**d,"x":actor,"ball":np.column_stack((seq[:,-1],np.zeros((n,4),np.float32)))}
    kin,geo=prior.dynamics(dd,seq)
    compact=np.column_stack((state,kin,geo)).astype(np.float32)
    history=np.column_stack((state,seq[:,:-1].reshape(n,-1))).astype(np.float32)
    save(cache,state=state,sequence=seq,kinematic=compact,history=history,proximity=state[:,[0,4]])
    write(meta,{"signature":signature,"sha256":prior.digest(cache)})
    return {"state":state,"sequence":seq,"kinematic":compact,"history":history,"proximity":state[:,[0,4]]}


def pressure_features(d):
    cache=RUN/"features_local"/"pressure.npz"
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:return z["pressure"],z["teams"]
    teams=np.zeros(len(d["target"]),np.int8)
    for match in np.unique(d["match"]):
        for hc,half in ((1,"1st"),(2,"2nd")):
            ix=np.flatnonzero((d["match"]==match)&(d["half_code"]==hc))
            if not len(ix):continue
            raw,_=source(str(match),half)
            q=d["query_frame"][ix].astype(int);sl=d["actor_slot_local_only"][ix].astype(int)
            if not np.all(raw["observed"][q,sl]) or not np.array_equal(raw["segments"][q,sl],d["segment_local_only"][ix]):
                raise ValueError("Current candidate identity join failed")
            teams[ix]=raw["teams"][q,sl]
    xy=d["x"][:,:2]*[52.5,34];v=d["x"][:,2:4]*12
    result=np.zeros((len(teams),18),np.float32)
    for rows in groups(d):
        p=xy[rows];vel=v[rows];t=teams[rows];known=np.isin(t,[1,2])
        delta=p[None]-p[:,None]
        distance=np.linalg.norm(delta,axis=2)
        same=(t[:,None]==t[None])&known[:,None]&known[None]&~np.eye(len(rows),dtype=bool)
        opp=(t[:,None]!=t[None])&known[:,None]&known[None]
        vals=[]
        for mask in (same,opp):
            nearest=np.where(mask,distance,1000).min(axis=1)
            counts=np.stack([np.sum(mask&(distance<r),axis=1)/11 for r in (3,5,10)],axis=1)
            closing=-np.sum((vel[None]-vel[:,None])*delta,axis=2)/np.maximum(distance,1e-4)
            near_index=np.argmin(np.where(mask,distance,1000),axis=1)
            approaching=closing[np.arange(len(rows)),near_index]/12
            vals.extend([nearest[:,None]/105,counts,approaching[:,None]])
        ballvalid=d["ball"][rows,0]>.5
        ball=xy[rows]+d["ball"][rows,2:4]*[105,68]
        vector=p-ball
        # Corridor between as-of ball and candidate; opponents/teammates are source team labels, not inferred possession.
        for mask in (same,opp):
            rel=p[None]-ball[:,None]
            fraction=np.sum(rel*vector[:,None],axis=2)/np.maximum(np.sum(vector*vector,axis=1)[:,None],1e-6)
            perpendicular=np.linalg.norm(rel-fraction[:,:,None]*vector[:,None],axis=2)
            corridor=mask&(fraction>0)&(fraction<1)
            cmin=np.where(corridor,perpendicular,1000).min(axis=1)
            ccount=np.sum(corridor&(perpendicular<2),axis=1)/11
            vals.extend([np.where(ballvalid,cmin,0)[:,None]/68,np.where(ballvalid,ccount,0)[:,None]])
        vals.extend([known[:,None].astype(float),np.abs(p)/[52.5,34],np.full((len(rows),1),np.mean(known))])
        result[rows]=np.column_stack(vals)
    result=np.clip(result,-20,20)
    save(cache,pressure=result,teams=teams)
    write(RUN/"context_feature_dictionary.json",{"columns":["same_nearest","same_count3m","same_count5m","same_count10m","same_nearest_closing","opponent_nearest","opponent_count3m","opponent_count5m","opponent_count10m","opponent_nearest_closing","same_corridor_min","same_corridor_count2m","opponent_corridor_min","opponent_corridor_count2m","team_known","abs_field_x","abs_field_y","scene_team_known_fraction"],"scope":"Full causal eligible exocentric roster, not simulated human-visible subset. No possession team or attacking-direction assumption; geometry normalized to metric pitch. Same/opponent are actor-relative source metadata."})
    return result,teams


def targets(d, geometry_horizons=(1,2), bas_windows=(("next_BAS_0_1s",0,25),("next_BAS_1_2s",25,50)), cache_name="targets_local"):
    p=RUN/(cache_name+".npz")
    meta=p.with_suffix(".json")
    signature=row_signature(d)+repr((tuple(geometry_horizons),tuple(bas_windows),.2))
    cached=None
    if p.exists():
        with np.load(p,allow_pickle=False) as z:cached={k:z[k] for k in z.files}
        if meta.exists():
            saved=read(meta)
            if saved["signature"]!=signature or saved["sha256"]!=prior.digest(p):
                raise ValueError("Target cache identity/checksum mismatch")
            return cached
    n=len(d["target"]);out={k:np.full(n,-1,np.int16) for k in tuple(f"geometry_{h}s" for h in geometry_horizons)+tuple(w[0] for w in bas_windows)}
    out["now_nearest"]=np.zeros(n,bool)
    audits={k:Counter() for k in out if k!="now_nearest"}
    for match in map(str,np.unique(d["match"])):
        if match not in (*FIT,"132877"):
            raise ValueError("Target reconstruction is limited to fitting/development matches")
        for hc,half in ((1,"1st"),(2,"2nd")):
            raw,track=source(match,half);events=event_rows(match,half,raw)
            ev=[e for e in events if e["label"] in (1,2,4,5,6)]
            evf=np.array([e["frame"] for e in ev])
            dd=subset(d,(d["match"]==match)&(d["half_code"]==hc))
            indices=np.flatnonzero((d["match"]==match)&(d["half_code"]==hc))
            if not len(indices):
                continue
            for local in groups(dd):
                rows=indices[local];q=int(d["query_frame"][rows[0]])
                slots=d["actor_slot_local_only"][rows].astype(int);segments=d["segment_local_only"][rows]
                b,_,_,valid=track.asof(np.array([q]))
                if valid[0]:
                    distances=np.linalg.norm(raw["xy"][q,slots]-b[0],axis=1)
                    if len(distances)>1 and np.sort(distances)[1]-distances.min()>.2:
                        out["now_nearest"][rows[np.argmin(distances)]]=True
                for sec in geometry_horizons:
                    name=f"geometry_{sec}s";f=q+sec*25
                    if f>=len(raw["xy"]):audits[name]["truncated"]+=1;continue
                    b,_,_,valid=track.asof(np.array([f]))
                    if not valid[0]:audits[name]["missing_ball"]+=1;continue
                    active=np.flatnonzero(raw["observed"][f])
                    if not len(active):audits[name]["empty_roster"]+=1;continue
                    dist=np.linalg.norm(raw["xy"][f,active]-b[0],axis=1)
                    if len(dist)<2 or np.sort(dist)[1]-dist.min()<=.2:audits[name]["tie_le_0p2m"]+=1;continue
                    winner=active[np.argmin(dist)]
                    present=(slots==winner)&(segments==raw["segments"][f,winner])
                    if not present.any():audits[name]["winner_not_current_candidate"]+=1;continue
                    if not np.all(raw["observed"][f,slots]&(raw["segments"][f,slots]==segments)):audits[name]["future_label_continuity"]+=1;continue
                    out[name][rows]=present.astype(int);audits[name]["eligible_scenes"]+=1
                for name,lo,hi in bas_windows:
                    if q+hi>=len(raw["xy"]):audits[name]["truncated"]+=1;continue
                    idx=np.flatnonzero((evf>q+lo)&(evf<=q+hi))
                    if not len(idx):
                        out[name][rows]=0;audits[name]["no_supported_annotation"]+=1;continue
                    first=ev[idx[0]]
                    tied=[ev[j] for j in idx if ev[j]["frame"]==first["frame"]]
                    audits[name]["annotated_scene_opportunities"]+=1
                    if len({(e["slot"],e["segment"]) for e in tied})>1:audits[name]["simultaneous_actor_ambiguity"]+=1;continue
                    if not first["known_actor"]:audits[name]["unknown_event_actor"]+=1;continue
                    present=(slots==first["slot"])&(segments==first["segment"])
                    if not present.any():audits[name]["actor_not_current_candidate"]+=1;continue
                    out[name][rows]=present.astype(int);audits[name]["event_candidate_recalled"]+=1
    if cached is not None:
        for key in out:
            np.testing.assert_array_equal(cached[key],out[key])
    else:
        save(p,**out)
    write(meta,{"signature":signature,"sha256":prior.digest(p),"legacy_cache_regenerated_and_compared":cached is not None})
    write(RUN/(cache_name+"_validity.json"),{"claim":{"geometry":"PROXY_ONLY","next_BAS":"INTERNAL_OOF_ANNOTATION; not physical touch"},"audits":{k:dict(v) for k,v in audits.items()},"tie_threshold_m":.2,"future_labels":"Future trajectory validity is target-only censoring, candidate roster and features remain fixed at current query; no future availability input.","BAS_semantics":{"included":["PASS","HIGH_PASS","CROSS","SHOT","HEADER"],"excluded":{"DRIVE":"interval of motion/control, not unique touch","DEFENSIVE_INTERVENTION":"mixed block/tackle may not uniquely indicate touch"},"physical_contact_certified":False}})
    return out


def rules(d,features,horizon):
    s=features["sequence"];rel=s[:,-1,2:4]*[105,68];v=(rel-s[:,-2,2:4]*[105,68])/.4
    valid=s[:,-1,0]>.5;pair=valid&(s[:,-2,0]>.5)
    current=np.linalg.norm(rel,axis=1)
    projected=np.linalg.norm(rel+v*horizon,axis=1)
    vv=np.sum(v*v,axis=1)
    tca=np.clip(-np.sum(rel*v,axis=1)/np.maximum(vv,.04),0,horizon)
    cpa=np.linalg.norm(rel+v*tca[:,None],axis=1)
    return {"nearest":np.where(valid,-current,-10000),"intercept":np.where(pair,-projected,-10000),"reachable_nearest":np.where(valid&(current<=8*horizon+2),-current,-10000),"closest_approach":np.where(pair,-cpa-.5*tca,-10000)}


def projection(features,horizon):
    seq=features["sequence"];rel=seq[:,-1,2:4]*[105,68];v=(rel-seq[:,-2,2:4]*[105,68])/.4
    pair=(seq[:,-1,0]>.5)&(seq[:,-2,0]>.5)
    speed2=np.sum(v*v,axis=1)
    tca=np.clip(-np.sum(rel*v,axis=1)/np.maximum(speed2,.04),0,horizon)
    proj=rel+v*horizon
    values=np.column_stack((pair,proj/[105,68],np.linalg.norm(proj,axis=1)/105,tca/max(horizon,.1),np.linalg.norm(rel+v*tca[:,None],axis=1)/105))
    values[~pair]=0
    return values.astype(np.float32)


def targets_experiment():
    d=prior.data();f=compact_features(d);fs=prior.feature_sets(d,prior.sequence())
    target=targets(d);result={}
    for name in ("geometry_1s","geometry_2s","next_BAS_0_1s","next_BAS_1_2s"):
        horizon=1 if name in ("geometry_1s","next_BAS_0_1s") else 2
        y=target[name];pred=rules(d,f,horizon);probs=[]
        for model,x in (("compact_current",f["state"]),("raw_history",f["history"]),("interpretable",fs["interpretable"]),("time_to_contact",np.column_stack((f["kinematic"],projection(f,horizon))))):
            pred[model],_=fit(name+"_"+model,x,d,y);probs.append(model)
        r=evaluate(d,y,pred,probs)
        switched=np.zeros(len(y),bool);eventscene=np.zeros(len(y),bool)
        for rows in groups(d):
            if np.all(y[rows]>=0) and np.any(y[rows]>0):
                eventscene[rows]=True
                switched[rows]=target["now_nearest"][rows].any() and not target["now_nearest"][rows[y[rows]>0]].any()
        r["conditional_event_scenes"]=evaluate(subset(d,eventscene),y[eventscene],{k:v[eventscene] for k,v in pred.items()},probs)
        r["switch_only"]=evaluate(subset(d,switched),y[switched],{k:v[switched] for k,v in pred.items()},probs) if switched.any() else {"status":"BLOCKED"}
        r["support"]={"total_rows":len(y),"eligible_rows":int(np.sum(y>=0)),"event_scene_rows":int(eventscene.sum()),"switch_rows":int(switched.sum()),"switch_fraction_of_all_rows":float(switched.mean())}
        r["contrasts"]={"ttc_minus_interpretable":contrast(r,"time_to_contact","interpretable"),"interpretable_minus_current":contrast(r,"interpretable","compact_current")}
        save(RUN/"predictions_local"/(name+".npz"),**pred)
        result[name]=r;write(RUN/"next_interactor_results.json",result)
        print("target",name,r["macro_match"],flush=True)
    log(4,"Four prospective tasks were fitted on current fixed candidates: future geometry at1/2s and earliest supported BAS contact-like actor in0-1/1-2s. Exact physical contact remains unverified; separate conditional-event and switched-nearest metrics prevent proxy promotion.",{k:{"macro":v["macro_match"],"support":v["support"]} for k,v in result.items()})
    return {"status":"COMPLETE","claim":"INTERNAL_OOF_AND_PROXY_ONLY","targets":result}


def challenge_masks(d,features,pressure):
    s=features["sequence"];valid=s[:,-1,0]>.5;dist=s[:,-1,4]*105;speed=s[:,-1,6]*40
    return {
        "near_le2m":valid&(dist<=2),"far_gt5m":valid&(dist>5),"stationary_ball":valid&(speed<=1),
        "fast_long_ball":valid&(speed>=15),"crowded_midfield":(pressure[:,7]*11>=3)&(np.abs(d["x"][:,0])*52.5<20),
        "far_to_near_history":valid&(s[:,0,0]>.5)&(s[:,0,4]*105>5)&(dist<3),
        "approaching_intercept_proxy":valid&(features["kinematic"][:,19]>.5)&(features["kinematic"][:,20]*105<2),
        "missing_or_stale":~valid|(s[:,-1,1]>0),
    }


def action_experiment():
    d=prior.data();f=compact_features(d);stale=compact_features(d,12);pressure,_=pressure_features(d)
    target,_=bas_targets(d);results={}
    with np.load(PRIOR/"anticipation_predictions_local.npz",allow_pickle=False) as z:
        baseline={key:z[key] for key in z.files}
    challenges=challenge_masks(d,f,pressure)
    for name in ("lead_0_1s","lead_1_2s","lead_2_3s"):
        y=target[name];pred={model:baseline[name+"__"+model] for model in ("current","raw_history","interpretable")}
        pred["nearest"]=rules(d,f,1)["nearest"]
        pred["compact_kinematic"],_=fit(name+"_compact",f["kinematic"],d,y)
        pred["joint_stale_0p48s"],_=fit(name+"_stale",stale["kinematic"],d,y)
        probs=tuple(k for k in pred if k!="nearest")
        r=evaluate(d,y,pred,probs)
        r["challenges"]={}
        for label,mask in challenges.items():
            if np.sum((y>0)&mask)>=5:
                r["challenges"][label]=evaluate(subset(d,mask),y[mask],{k:v[mask] for k,v in pred.items()},probs)
        r["raw_class_support"]={prior.old.CLASS_NAMES[k]:int(np.sum(y==k)) for k in range(1,8)}
        r["class_specific_ranking"]={}
        for cls in range(1,8):
            if np.sum(y==cls)<5:continue
            # A common event score is not a class classifier. Positives of other classes remain negatives for this diagnostic.
            mask=y>=0
            r["class_specific_ranking"][prior.old.CLASS_NAMES[cls]]={k:metrics((y[mask]==cls).astype(int),p[mask]) for k,p in pred.items()}
        r["class_scope"]="Shared any-action score used for per-class ranking diagnostic; not a new seven-way class-probability head. Raw class labels preserved."
        r["contrasts"]={"compact_minus_stale":contrast(r,"compact_kinematic","joint_stale_0p48s")}
        results[name]=r
        save(RUN/"predictions_local"/(name+".npz"),**pred)
    write(RUN/"anticipation_challenges.json",results)
    log(5,"Reused exact immutable strong OOF baselines and fitted matched compact causal versus fully joint-stale kinematics. Both ball and actor streams end12frames earlier in the intervention; no current context channels remain in that stale arm. Report class support and causal challenge masks separately.",{k:v["macro_match"] for k,v in results.items()})
    return {"status":"COMPLETE","horizons":results,"counterfactual":"All compact player/ball histories end at q-12; fixed query candidate identities; missing past samples masked. Inherited full-context baselines are additional references, not equal-input interventions."}


def context_experiment():
    d=prior.data();f=compact_features(d);p,teams=pressure_features(d);tt=targets(d);action,_=bas_targets(d)
    families={"distance":f["proximity"],"kinematic":f["kinematic"],"same_team":np.column_stack((f["kinematic"],p[:,[0,1,2,3,4,10,11,14,17]])),"opponent":np.column_stack((f["kinematic"],p[:,[5,6,7,8,9,12,13,14,17]])),"full_context":np.column_stack((f["kinematic"],p))}
    result={}
    for targetname,y in (("early_action",action["lead_1_2s"]),("geometric_switch_risk",tt["geometry_2s"]),("next_BAS_actor",tt["next_BAS_1_2s"])):
        pred={}
        for family,x in families.items():pred[family],_=fit("context_"+targetname+"_"+family,x,d,y)
        r=evaluate(d,y,pred,tuple(pred))
        r["context_minus_kinematic"]=contrast(r,"full_context","kinematic")
        if targetname!="early_action":
            switch=np.zeros(len(y),bool)
            for rows in groups(d):
                if np.all(y[rows]>=0) and np.any(y[rows]>0) and tt["now_nearest"][rows].any() and not tt["now_nearest"][rows[y[rows]>0]].any():
                    switch[rows]=True
            r["switch_only"]=evaluate(subset(d,switch),y[switch],{k:v[switch] for k,v in pred.items()},tuple(pred)) if switch.any() else {"status":"BLOCKED"}
        r["challenge"]={}
        for name,mask in challenge_masks(d,f,p).items():
            if np.sum((y>0)&mask)>=5:r["challenge"][name]=evaluate(subset(d,mask),y[mask],{k:v[mask] for k,v in pred.items()},tuple(pred))
        result[targetname]=r;save(RUN/"predictions_local"/("context_"+targetname+".npz"),**pred)
        write(RUN/"tactical_context_results.json",result)
    log(6,"Distance, kinematics, same-team, opponent and full causal context were independently fitted with matched capacity. Source team labels are verified; possession and attacking direction are intentionally not guessed. Treat any lift as spatial-information value, not optimal tactics.",{k:{"macro":v["macro_match"],"delta":v["context_minus_kinematic"]} for k,v in result.items()})
    return {"status":"COMPLETE","tasks":result,"scope":"Full eligible exocentric current roster; source teams; no human perception claim"}


def retrieval():
    import shutil
    old=RUN/"scene_prioritization.json"
    if old.exists() and not read(old).get("lead_0_1s",{}).get("scheduler_revision"):
        shutil.copyfile(old,RUN/"scene_prioritization_invalid_future_mask.json")
        write(RUN/"scheduling_integrity_correction.json",{"utc":prior.utc(),"reason":"Initial offline quota filtered out future-label-unsupported scenes (2 imminent,82 early), contrary to causal scheduling. This is repaired by scoring ALL held-match current queries and applying quotas without future target masks. Unknown selected outcomes are separately unverifiable, not false reviews.","old_result_sha256":prior.digest(old),"preserved_as":"scene_prioritization_invalid_future_mask.json","effect_on_other_results":"None: target-defined evaluation and all model-training comparisons remain unchanged; only scheduling eligibility was invalid.","adaptive_lock_resolution":"Earlier adaptive evidence hash for scene_prioritization.json resolves to the preserved invalid version; do not rewrite lock timestamps."})
    d=prior.data();g=groups(d);target,frames=bas_targets(d);result={}
    f=compact_features(d);pressure,_=pressure_features(d)
    inherited=prior.feature_sets(d,prior.sequence())
    for horizon,contextname in (("lead_0_1s",None),("lead_1_2s","early_action")):
        y=target[horizon]
        if contextname:
            matrices={"kinematic":f["kinematic"],"full_context":np.column_stack((f["kinematic"],pressure)),"distance":f["proximity"]}
            pred={name:fit("scheduler_"+horizon+"_"+name,x,d,y,all_queries=True)[0] for name,x in matrices.items()}
        else:
            pred={name:fit("scheduler_"+horizon+"_"+name,x,d,y,all_queries=True)[0] for name,x in (("interpretable",inherited["interpretable"]),("compact_kinematic",f["kinematic"]))}
            pred["nearest"]=rules(d,f,1)["nearest"]
        scene_y=np.array([int(np.any(y[r]>0)) for r in g])
        eligible=np.array([np.all(y[r]>=0) for r in g])
        sm=np.array([d["match"][r[0]] for r in g])
        scene={};work={}
        for model,prediction in pred.items():
            scores=np.array([prediction[r].max() for r in g])
            if not np.isfinite(scores).all():raise ValueError("Every causal current scene must have a finite scheduling score")
            scene[model]={"pooled":metrics(scene_y[eligible],scores[eligible]),"per_match":{m:metrics(scene_y[eligible&(sm==m)],scores[eligible&(sm==m)]) for m in FIT}}
            work[model]={}
            for budget in (5,10,20):
                selected,suppressed=prior.select_reviews(scores,g,d,budget)
                for k in (1,3):
                    scored=[i for i in selected if eligible[i]]
                    v=prior.review_metrics(scored,g,d,prediction,frames[horizon],k)
                    v["supported_review_scenes"]=len(scored)
                    v["unverifiable_review_scenes"]=len(selected)-len(scored)
                    v["reviewed_scenes"]=len(selected)
                    v["reviewed_actor_candidates"]=len(selected)*k
                    v["actor_hit_at_k_per_supported_review"]=v.pop("actor_hit_at_k_per_review")
                    v["unique_events_per_reviewed_scene"]=v["unique_events_recalled"]/max(len(selected),1)
                    v["suppressed"]=suppressed
                    work[model][f"{budget}_top{k}"]=v
        result[horizon]={"scheduler_revision":2,"scheduling_contract":"All current causal queries scored before target-validity filtering; unknown selected outcomes separate. Target masks affect evaluation only.","scenes":len(g),"eligible_scenes":int(eligible.sum()),"positive_scenes":int(scene_y[eligible].sum()),"prevalence":float(scene_y[eligible].mean()),"scene_metrics":scene,"workload":work}
    write(RUN/"scene_prioritization.json",result)
    log(7,"Scene occurrence and conditional actor selection are measured at imminent and early horizons without a redundant hierarchy. Offline quotas use only causal OOF scores, five-second suppression and distinct-event counts; annotation-relative false reviews remain unverified physical negatives.",{k:{"prevalence":v["prevalence"],"workload10":{m:r["10_top3"] for m,r in v["workload"].items()}} for k,v in result.items()})
    return {"status":"COMPLETE","horizons":result,"scope":"Offline label-relative retrieval, not operational alerts"}
