import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from common import RUN, PARAMS, prior, read, write
from experiments import compact_features, targets
from external import download, external_arrays
from followups import lock


def scene_outcomes(ext,y,p):
    correct=[];blocks=[];rank=[]
    for sid in np.unique(ext["scene"][y>=0]):
        rows=np.flatnonzero((ext["scene"]==sid)&(y>=0))
        order=rows[np.argsort(-p[rows],kind="stable")]
        r=int(np.flatnonzero(y[order]>0)[0])+1
        correct.append(r==1);rank.append(1/r)
        blocks.append(int(ext["frame"][rows[0]])//750)
    return np.array(correct,float),np.array(rank,float),np.array(blocks)


def block_interval(delta,block):
    unique=np.unique(block)
    sums=np.array([delta[block==b].sum() for b in unique])
    counts=np.array([(block==b).sum() for b in unique])
    rng=np.random.default_rng(20261103)
    indices=rng.integers(len(unique),size=(2000,len(unique)))
    draws=sums[indices].sum(1)/counts[indices].sum(1)
    return {"delta":float(delta.mean()),"ci95":np.quantile(draws,[.025,.975]).tolist(),"time_blocks":len(unique),"scope":"30s temporal block bootstrap within one match; not across-domain population uncertainty."}


def run():
    a=read(RUN/"adaptive_quantization.json")
    lock("D_domain_resolution","A's1s causal smoothing improved geometry AP only0.0048 and next-BAS AP0.0040, while SoccerTrack positions lie on a metre-scale grid and Metrica pass-start actor-ball median distance is0.088m. Resolution-domain mismatch may alter frozen model reliability independently of useful football information.","Fixed Metrica native geometric1/2s labels, candidates and eligibility; compare native/player-rounded/ball-rounded/both-rounded causal inputs. AP and paired switch/top1 differences.","Two frozen all-six-match35-feature HGB models; no external fitting or threshold changes; four input-resolution arms,30s temporal-block intervals. Future target coordinates never rounded.")
    folder=download()
    arms={"native":external_arrays(folder)}
    for name,p,b in (("players_only",True,False),("ball_only",False,True),("both",True,True)):
        arms[name]=external_arrays(folder,quantize_players=p,quantize_ball=b)
        for key in ("scene","actor","frame","geometry_1s","geometry_2s"):
            np.testing.assert_array_equal(arms[name][key],arms["native"][key])
    d=prior.data();f=compact_features(d);labels=targets(d);result={}
    for target in ("geometry_1s","geometry_2s"):
        valid=labels[target]>=0
        model=HistGradientBoostingClassifier(**PARAMS).fit(f["kinematic"][valid],labels[target][valid])
        report={};outcomes={}
        for name,ext in arms.items():
            y=ext[target];eligible=y>=0
            p=model.predict_proba(ext["kinematic"])[:,1]
            dd={"match":np.repeat("M",int(eligible.sum())),"half_code":np.ones(int(eligible.sum()),int),"query_frame":ext["scene"][eligible]}
            report[name]=prior.metrics(y[eligible],p[eligible],prior.groups(dd),probability=p[eligible])
            outcomes[name]=scene_outcomes(ext,y,p)
        baseline=outcomes["native"]
        differences={name:{"top1":block_interval(v[0]-baseline[0],baseline[2]),"MRR":block_interval(v[1]-baseline[1],baseline[2])} for name,v in outcomes.items() if name!="native"}
        result[target]={"models":report,"paired_input_interventions_minus_native":differences,"invariant_target_and_roster_checks":"PASS"}
        write(RUN/"adaptive_domain_resolution.json",result)
        print("resolution",target,{k:v["AP"] for k,v in report.items()},flush=True)
    return result


if __name__=="__main__":
    run()
