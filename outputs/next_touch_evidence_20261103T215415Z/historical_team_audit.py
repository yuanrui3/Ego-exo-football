import csv
from collections import Counter

import numpy as np

from common import ROOT, RUN, source, read, write, digest


def run():
    pack=ROOT/"analysis"/"pass_video_review_20261015_204039_494955"
    with (pack/"completed_batch_20261015"/"annotation_yuanrui_simple.csv").open(newline="",encoding="utf-8-sig") as stream:
        human=list(csv.DictReader(stream))
    with (pack/"internal_manifest.csv").open(newline="",encoding="utf-8-sig") as stream:
        manifest={r["review_id"]:r for r in csv.DictReader(stream)}
    records=[];excluded=Counter();cache={}
    for row in human:
        target=row["first_touch_team"].strip().upper()
        if row["target_pass_visible"].strip().upper()!="YES":
            excluded["target_pass_not_visible_yes"]+=1;continue
        if target not in ("TEAMMATE","OPPONENT"):
            excluded["human_touch_team_uncertain"]+=1;continue
        event=manifest[row["review_id"]];match=event["match"];half="1st" if event["half"]=="1" else "2nd"
        if (match,half) not in cache:
            path=ROOT/"outputs"/"trajectory_forecasting_20261016"/"cache"/f"{match}_{half}.npz"
            if not path.exists():
                excluded["no_verified_tracking"]+=1;continue
            raw,track=source(match,half)
            association=read(ROOT/"outputs"/"trajectory_audit_20261016"/f"associations_{match}_{half}.json")
            cache[match,half]=(raw,track,association)
        raw,track,association=cache[match,half]
        actor=event["source_actor"]
        if actor not in association["slots"]:
            excluded["unmapped_actor"]+=1;continue
        slot=int(association["slots"][actor])
        q=int(float(event["BAS_position_ms"])*25/1000)
        if q<10 or q+75>=len(raw["xy"]) or not raw["observed"][q,slot]:
            excluded["invalid_query"]+=1;continue
        own=int(raw["teams"][q,slot])
        if own not in (1,2):
            excluded["unknown_team"]+=1;continue
        roster=np.flatnonzero(raw["observed"][q])
        b,bv,_,valid=track.asof(np.array([q]))
        if not valid[0]:
            excluded["missing_current_ball"]+=1;continue
        def relation(team):
            return "TEAMMATE" if team==own else ("OPPONENT" if team in (1,2) else "UNKNOWN")
        distance=np.linalg.norm(raw["xy"][q,roster]-b[0],axis=1)
        nearest=int(roster[np.argmin(distance)])
        future_ball=b[0]+bv[0]*2
        past=np.maximum(q-5,0)
        continuity=raw["observed"][past,roster] & (raw["segments"][past,roster]==raw["segments"][q,roster])
        av=np.where(continuity[:,None],(raw["xy"][q,roster]-raw["xy"][past,roster])/.2,0)
        intercept=int(roster[np.argmin(np.linalg.norm(raw["xy"][q,roster]+av*2-future_ball,axis=1))])
        predictions={"always_passer_team": "TEAMMATE","causal_current_nearest":relation(int(raw["teams"][q,nearest])),"causal_2s_projection":relation(int(raw["teams"][q,intercept]))}
        future=np.arange(q+5,q+76)
        fb,_,_,ok=track.asof(future)
        observed=raw["observed"][future]
        dist=np.linalg.norm(raw["xy"][future]-fb[:,None],axis=2)
        dist[~observed]=np.inf
        near=np.argmin(dist,axis=1);mind=dist[np.arange(len(future)),near]
        candidates=ok&(mind<=2)&(near!=slot)
        stable=np.r_[False,(near[1:]==near[:-1])&candidates[:-1]]&candidates
        j=np.flatnonzero(stable)
        predictions["retrospective_first_stable_nonpasser_near_ball"]=relation(int(raw["teams"][future[j[0]],near[j[0]]])) if len(j) else "UNKNOWN"
        records.append({"review_id":row["review_id"],"match":match,"target_first_touch_team":target,"predictions":predictions,"note":"Last method uses future tracking as a label-association diagnostic, not prospective input"})
    methods={}
    for name in ("always_passer_team","causal_current_nearest","causal_2s_projection","retrospective_first_stable_nonpasser_near_ball"):
        known=[r for r in records if r["predictions"][name]!="UNKNOWN"]
        methods[name]={"n":len(records),"covered":len(known),"coverage":len(known)/max(len(records),1),"correct":sum(r["predictions"][name]==r["target_first_touch_team"] for r in known),"accuracy_on_covered":sum(r["predictions"][name]==r["target_first_touch_team"] for r in known)/max(len(known),1),"matched_always_passer_team_correct":sum(r["target_first_touch_team"]=="TEAMMATE" for r in known),"per_match":{m:{"n":sum(r["match"]==m for r in known),"correct":sum(r["match"]==m and r["predictions"][name]==r["target_first_touch_team"] for r in known)} for m in sorted(set(r["match"] for r in known))}}
    write(RUN/"historical_team_records_local.json",records)
    result={"status":"COMPLETE","claim":"HISTORICAL_HUMAN_TEAM_ASSOCIATION_PILOT_NOT_NEW_TOUCH_GROUND_TRUTH","n":len(records),"human_target_counts":dict(Counter(r["target_first_touch_team"] for r in records)),"excluded":dict(excluded),"methods":methods,"limits":["One reviewer, enriched historical pass sample already exposed; no representative test or inter-reviewer agreement.","No precise first-touch timestamp/player; only reported touch team.","Retrospective geometric association is not a forecast and does not verify physical contact.","Source clocks remain independently unverified; unresolved disagreement must not overwrite human judgments."]}
    write(RUN/"historical_team_association.json",result)
    return result


if __name__=="__main__":
    run()
