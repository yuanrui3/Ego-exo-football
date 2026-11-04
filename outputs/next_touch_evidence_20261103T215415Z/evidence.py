from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter

import numpy as np

from common import ROOT, RUN, PRIOR, FIT, prior, source, event_rows, read, write, save, digest, log


def quantiles(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return dict(zip(("p0", "p50", "p90", "p99", "max"), np.quantile(a, [0, .5, .9, .99, 1]).tolist())) if len(a) else None


def contact_signals(raw, track):
    """Retrospective geometric evidence only, not a physical touch label."""
    n = len(raw["xy"])
    q = np.arange(n)
    b, _, age, good = track.asof(q)
    oldb, _, _, oldgood = track.asof(q-5)
    older, _, _, older_good = track.asof(q-10)
    v = (b-oldb)/.2
    pv = (oldb-older)/.2
    stable = good & oldgood & older_good
    delta_v = np.linalg.norm(v-pv, axis=1)
    speed = np.linalg.norm(v, axis=1)
    cosine = np.sum(v*pv, axis=1) / np.maximum(np.linalg.norm(v,axis=1)*np.linalg.norm(pv,axis=1), 1e-6)
    distance = np.linalg.norm(raw["xy"]-b[:, None], axis=2)
    distance = np.where(raw["observed"] & good[:, None], distance, 10000)
    nearest = np.argmin(distance, axis=1)
    minimum = distance[q, nearest]
    previous_nearest = nearest[np.maximum(q-5, 0)]
    persistence = (minimum <= 2) & good & (previous_nearest == nearest)
    aold = raw["xy"][np.maximum(q-5, 0), nearest]
    continuous = raw["observed"][np.maximum(q-5, 0), nearest] & (raw["segments"][q, nearest] == raw["segments"][np.maximum(q-5, 0), nearest])
    av = (raw["xy"][q, nearest] - aold)/.2
    rel = b - raw["xy"][q, nearest]
    rv = v-av
    tca = np.clip(-np.sum(rel*rv, axis=1)/np.maximum(np.sum(rv*rv, axis=1), .04), 0, .4)
    intercept_dist = np.linalg.norm(rel+rv*tca[:, None], axis=1)
    intercept = stable & continuous & (minimum < 5) & (intercept_dist < 1.5)
    change = stable & continuous & (minimum < 3) & ((delta_v/.2 > 15) | ((cosine < .5) & (speed > 2)))
    # Quantized positions can generate spurious jerk; forbid >45m/s and isolated unsupported samples.
    change &= speed <= 45
    return {"ball_xy": b, "valid": good, "age": age, "speed": speed, "accel": delta_v/.2, "nearest": nearest.astype(np.int16), "distance": minimum, "proximity": persistence, "intercept": intercept, "changepoint": change, "tca": tca}


def clocks():
    records, clips = {}, []
    bins = ROOT / "data" / "video_review_tools" / "portable"
    probes = sorted(bins.rglob("ffprobe.exe"))
    encoders = sorted(bins.rglob("ffmpeg.exe"))
    if not probes or not encoders:
        raise FileNotFoundError("Existing portable ffmpeg/ffprobe is required.")
    private = RUN / "evidence_local"
    private.mkdir(exist_ok=True)
    for match in (*FIT, "132877"):
        for half in ("1st", "2nd"):
            raw, track = source(match, half)
            events = event_rows(match, half, raw)
            sig = contact_signals(raw, track)
            meta = read(ROOT / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{match}_{half}.json")
            video = ROOT / "data" / "video_review_v1_2" / "videos" / match / f"{match}_panorama_{half}_half.mp4"
            command = [str(probes[0]), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,start_time,time_base", "-of", "json", str(video)]
            probe = json.loads(subprocess.run(command,capture_output=True,text=True,check=True,timeout=60).stdout)["streams"][0]
            offsets = {}
            for offset in (-5,-1,0,1,5):
                distances = []
                for e in events:
                    q = e["frame"]+offset
                    if not e["known_actor"] or not 0 <= q < len(raw["xy"]):
                        continue
                    slot=e["slot"]
                    b,_,_,valid=track.asof(np.array([q]))
                    if valid[0] and raw["observed"][q,slot] and raw["segments"][q,slot]==e["segment"]:
                        distances.append(float(np.linalg.norm(b[0]-raw["xy"][q,slot])))
                offsets[str(offset)]={"n":len(distances),"actor_ball_distance_m":quantiles(distances)}
            validxy=raw["xy"][raw["observed"]]
            observed=raw["observed"]
            both=observed[1:]&observed[:-1]
            repeated=np.all(raw["xy"][1:]==raw["xy"][:-1],axis=2)&both
            records[f"{match}_{half}"]={
                "frames":len(raw["xy"]),"gsr":meta["gsr_info"],"video_probe":probe,
                "ball_first_frame":int(track.frames[0]),"ball_last_frame":int(track.frames[-1]),
                "ball_nonunit_gaps":int(np.sum(np.diff(track.frames)!=1)),
                "ball_missing_fraction":float(np.mean(~sig["valid"])),"ball_speed_mps":quantiles(sig["speed"][sig["valid"]]),
                "ball_acceleration_mps2":quantiles(sig["accel"][sig["valid"]]),
                "ball_speed_gt45_fraction":float(np.mean(sig["speed"][sig["valid"]]>45)),
                "actor_repeated_position_fraction":float(repeated.sum()/max(both.sum(),1)),
                "actor_native_grid":meta.get("quality",{}),
                "out_of_pitch_actor_fraction":float(np.mean((np.abs(validxy[:,0])>52.5)|(np.abs(validxy[:,1])>34))),
                "source_segment_changes":int(np.sum((raw["segments"][1:]!=raw["segments"][:-1])&both)),
                "supported_event_rows":len(events),"known_actor_event_rows":sum(bool(e["known_actor"]) for e in events),
                "offset_diagnostics":offsets,"offset_choice":"NONE: diagnostic distances are not physical synchronization evidence.",
                "physical_sync":"PHYSICAL_SYNC_UNVERIFIED",
            }
            # One fit-independent, midpoint-nearest annotated sample per half, not selected by model score or optimal offset.
            eligible=[e for e in events if e["known_actor"] and e["label"] in (1,2,4,5,6)]
            if eligible:
                e=min(eligible,key=lambda x:abs(x["frame"]-len(raw["xy"])/2))
                case=f"S{len(clips)+1:03d}"
                destination=private/(case+".mp4")
                if not destination.exists():
                    subprocess.run([str(encoders[0]),"-v","error","-nostdin","-ss",str(max(0,e["frame"]/25-2)),"-i",str(video),"-t","4","-an","-vf","scale=1920:-2","-c:v","libx264","-preset","fast","-crf","20",str(destination)],check=True,capture_output=True,timeout=120)
                clips.append({"case_id":case,"match":match,"half":half,"query_frame":e["frame"],"native_seconds":e["frame"]/25,"event_label":e["label"],"video":str(video),"clip":str(destination.relative_to(RUN)),"sample_rule":"nearest half midpoint among known-actor contact-like annotations","timestamp_confidence":"source metadata only","physical_contact":"UNVERIFIABLE_WITHOUT_INDEPENDENT_REVIEW","source_video_bytes":video.stat().st_size,"clip_sha256":digest(destination)})
            print("clock",match,half,"done",flush=True)
    write(RUN/"clip_evidence_manifest_local.json",clips)
    result={"status":"PARTIAL","claim":"PHYSICAL_SYNC_UNVERIFIED","halves":records,"private_sample_clips":len(clips),"conventions":{"GSR":"zero-based source frame","ball":"one-based file labels preserved in asof; no silent -1 correction","BAS":"floor(position_ms*25/1000); stable frame separately checked by existing resolver","video":"ordinal zero-based, PTS and native FPS probed; metadata agreement is not physical contact evidence"}}
    write(RUN/"clock_and_sync_audit.json",result)
    (RUN/"clock_and_sync_audit.md").write_text("# Clock and physical evidence audit\n\nFourteen native halves audited. Per-half FPS, duration, ball gaps, speeds, quantization, segment changes and off-by-one diagnostics are in the JSON. Fourteen private four-second midpoint-selected clips were extracted. No offset was selected by event proximity. No independently corroborated physical touch time is declared. PHYSICAL_SYNC_UNVERIFIED remains the gate.\n",encoding="utf-8")
    log(1,"Native source timing and offset diagnostics are executed across fourteen halves; physical calibration cannot be inferred from annotation-distance optimization. Extract independent midpoint-selected private clips and continue with explicit proxy labels.",{"halves":len(records),"clips":len(clips),"physical_sync":"UNVERIFIED","median_repeat_fraction":float(np.median([v["actor_repeated_position_fraction"] for v in records.values()]))})
    return result


def reviews():
    pack=ROOT/"analysis"/"pass_video_review_20261015_204039_494955"
    hp=pack/"completed_batch_20261015"/"annotation_yuanrui_simple.csv"
    mp=pack/"internal_manifest.csv"
    with hp.open(encoding="utf-8-sig",newline="") as f: human=list(csv.DictReader(f))
    with mp.open(encoding="utf-8-sig",newline="") as f: original=list(csv.DictReader(f))
    lookup={r["review_id"]:r for r in original}
    if len(lookup)!=len(original) or len({r["review_id"] for r in human})!=len(human):
        raise ValueError("Duplicate historical human review IDs")
    counts=Counter();joins=[];sources={}
    for row in human:
        oldrow=lookup[row["review_id"]]
        match=oldrow["match"]
        if match not in sources:sources[match]=read(ROOT/"data"/f"{match}_12_class_events.json")["actions"]
        e=sources[match][int(oldrow["BAS_annotation_index"])]
        actual=f"{match}_p{int(e['gameTime'].split('-')[0].strip())}_f{e['frame']}_{oldrow['BAS_annotation_index']}"
        if actual!=oldrow["source_id"] or row["clip_reference"]!=oldrow["clip_reference"]:
            raise ValueError("Historical human source/clip join mismatch")
        counts["visible_"+row["target_pass_visible"].strip().upper()]+=1
        counts["touch_team_"+row["first_touch_team"].strip().upper()]+=1
        counts["control_team_"+row["first_control_team"].strip().upper()]+=1
        clip=pack/"completed_batch_20261015"/row["clip_reference"]
        joins.append({"review_id":row["review_id"],"match":match,"source_id":actual,"clip_exists":clip.exists(),"clip":str(clip),"physical_exact_touch_actor_available":False})
    write(RUN/"historical_review_provenance_local.json",joins)
    queue=read(PRIOR/"review_queue_local"/"review_queue_local.json")
    private=RUN/"review_local";private.mkdir(exist_ok=True)
    cases=[]
    for row in queue:
        case={"case_id":"N_"+row["case_id"],"evidence_kind":"SCHEMATIC_ONLY","sync_verified":False,"schematic":str((PRIOR/"review_queue_local"/row["schematic"]).resolve()),"clip":None,"hidden_predictions":row["hidden_predictions"],"match":row["match"],"half":row["half"],"query_frame":row["query_frame"],"stratum":row["stratum_private"]}
        cases.append(case)
    for row in read(RUN/"clip_evidence_manifest_local.json"):
        cases.append({"case_id":row["case_id"],"evidence_kind":"VIDEO_SOURCE_CLOCK_ONLY","sync_verified":False,"clip":str((RUN/row["clip"]).resolve()),"schematic":None,"hidden_predictions":{},"match":row["match"],"half":row["half"],"query_frame":row["query_frame"]})
    write(private/"queue.json",cases)
    # Revision mismatch counts are evidence to retain, not a reason to overwrite either source.
    lineage=read(ROOT/"outputs"/"hf_soccertrack_audit_20261031_212547"/"audit_summary.json")
    revisions={}
    for match in (*FIT,"132877"):
        local=ROOT/"data"/f"{match}_12_class_events.json"
        official=ROOT/"data"/"video_review_v1_2"/"bas"/match/f"{match}_12_class_events.json"
        alternate=ROOT/"data"/"video_review_v1_2"/"bas_comparison"/"bas"/match/f"{match}_12_class_events.json"
        revisions[match]={"local_sha256":digest(local),"local_actions":len(read(local)["actions"]),"official_candidate_exists":official.exists(),"official_sha256":digest(official) if official.exists() else None,"official_actions":len(read(official)["actions"]) if official.exists() else None,"same_bytes":digest(local)==digest(official) if official.exists() else None,"alternate_sha256":digest(alternate) if alternate.exists() else None,"alternate_actions":len(read(alternate)["actions"]) if alternate.exists() else None}
    result={"status":"PARTIAL","historical_human_reviews":len(human),"historical_counts":dict(counts),"historical_clip_paths_found":sum(r["clip_exists"] for r in joins),"historical_scope":"One reviewer, enriched pass clips; team-of-first-touch/control and pass visibility, not actor/time contact labels or verified event-negative windows. No inter-annotator kappa can be computed.","new_120_case_labels":0,"new_cases":len(cases),"new_review_status":"PAUSED_BY_USER; preparation only; zero new labels","source_revision":revisions,"upstream_audit_summary_keys":list(lineage),"exact_touch_ground_truth":"BLOCKED","claim_correction":"Prior zero-human statement applies only to the 120-case queue, not the pre-existing 60-case pass review."}
    write(RUN/"event_label_coverage.json",result)
    (RUN/"reviewer_instructions.md").write_text("# Two-reviewer workflow (paused, not requested now)\n\nUse review.html to load review_local/queue.json plus local evidence files. Source-clock-only clips may be viewed but are not independently synchronized. NO_EVENT_VERIFIED is disabled without independently verified synchronization. Record EVENT_PRESENT, UNCERTAIN, NO_PLAYER_VISIBLE, BAD_SYNCHRONIZATION or OTHER as appropriate; never infer negatives from the schematic. Predictions remain hidden until an immutable first judgment. Independent reviewers export separate JSON/CSV; arbitration imports both without overriding originals. Report agreement only if real paired labels exist. Historical 60-case pass judgments remain a separate enriched single-reviewer dataset, not new exact-contact ground truth.\n",encoding="utf-8")
    log(2,"Recovered sixty historical human judgments with exact source-event and clip-reference joins. They address pass visibility and first-touch team, not precise touch actor/time or reliable negatives. New review remains paused; prepare a working evidence-aware interface without fabricating labels.",{"historical":len(human),"counts":dict(counts),"new_labels":0,"cases":len(cases)})
    return result


def contacts():
    results={};payload=[];eventassociations=[]
    for match in FIT:
        for half in ("1st","2nd"):
            raw,track=source(match,half)
            sig=contact_signals(raw,track)
            events=event_rows(match,half,raw)
            n=len(sig["valid"])
            key=match+"_"+half
            save(RUN/"contacts_local"/(key+".npz"),**sig)
            byclass={}
            for cls in range(1,8):
                rows=[e for e in events if e["label"]==cls and e["known_actor"]]
                byclass[prior.old.CLASS_NAMES[cls]]={}
                for algorithm in ("proximity","intercept","changepoint"):
                    hits=0;lags=[];distances=[];unsupported=0
                    for e in rows:
                        f=e["frame"]
                        qs=np.arange(max(0,f-12),min(n,f+13))
                        if not sig["valid"][f]:
                            unsupported+=1;continue
                        same=sig["nearest"][qs]==e["slot"]
                        supported=sig[algorithm][qs]&same&(raw["segments"][qs,e["slot"]]==e["segment"])
                        distances.append(float(np.linalg.norm(raw["xy"][f,e["slot"]]-sig["ball_xy"][f])))
                        if supported.any():
                            hits+=1
                            lag=qs[supported]-f
                            lags.append(float(lag[np.argmin(np.abs(lag))]/25))
                    byclass[prior.old.CLASS_NAMES[cls]][algorithm]={"annotated_events":len(rows),"ball_missing_at_event":unsupported,"candidate_association_count":hits,"association_fraction":hits/max(1,len(rows)),"nearest_candidate_lag_seconds":quantiles(lags),"actor_ball_distance_m":quantiles(distances)}
            counts={name:int(sig[name].sum()) for name in ("proximity","intercept","changepoint")}
            overlap={f"{a}__{b}":float(np.sum(sig[a]&sig[b])/max(1,np.sum(sig[a]|sig[b]))) for a,b in (("proximity","intercept"),("proximity","changepoint"),("intercept","changepoint"))}
            results[key]={"frames":n,"valid_ball_frames":int(sig["valid"].sum()),"candidate_frames":counts,"algorithm_jaccard":overlap,"event_association":byclass,"insufficient_ball_height_frames":n,"near_goalkeeper_candidate_frames":int(np.sum(sig["changepoint"]&(raw["roles"][np.arange(n),sig["nearest"]]==1)))}
            # Nonmaximum suppression yields disputed candidates for later physical review, not labels.
            last=-1000
            for frame in np.flatnonzero(sig["changepoint"]&~sig["proximity"]):
                if frame-last<50:continue
                last=int(frame)
                eventassociations.append({"case_id":f"D{len(eventassociations)+1:05d}","match":match,"half":half,"frame":int(frame),"distance_m":float(sig["distance"][frame]),"accel_mps2":float(sig["accel"][frame]),"status":"TOUCH_CANDIDATE_PROXY"})
                if len([r for r in eventassociations if r["match"]==match and r["half"]==half])>=10:break
            print("contact",key,counts,flush=True)
    write(RUN/"disputed_contacts_local.json",eventassociations)
    result={"status":"COMPLETE","claim":"PROXY_ONLY","halves":results,"algorithms":{"proximity":"nearest <=2m, same nearest 0.2s ago","intercept":"nearest <=5m, relative constant-velocity closest approach <=1.5m within0.4s; continuous actor and ball","changepoint":"nearest<=3m and smoothed backward ball acceleration>15m/s2 or direction change>60deg at speed>2; suppress >45m/s and invalid samples"},"limitations":"2D ball only; no verified height, airborne/rolling regimes are not identifiable. Quantization can induce apparent acceleration. Association with BAS is not independent physical truth.","disputed_cases":len(eventassociations)}
    write(RUN/"contact_candidate_audit.json",result)
    log(3,"Three independent geometric algorithms were executed at native frame cadence on twelve fit halves. BAS actor/time association and disagreements are reported without treating agreement as a contact certificate. Proceed to separate annotated-actor and geometric prediction targets.",{"halves":len(results),"frames":sum(v["frames"] for v in results.values()),"disputed":len(eventassociations),"validity":"PROXY_ONLY"})
    return result
