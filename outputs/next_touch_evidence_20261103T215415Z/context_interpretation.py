import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from common import RUN, FIT, PARAMS, prior, bas_targets, read, write, digest
from experiments import compact_features, pressure_features


def run():
    d=prior.data();f=compact_features(d);pressure,_=pressure_features(d)
    target,_=bas_targets(d);y=target["lead_1_2s"];valid=y>=0
    x=np.column_stack((f["kinematic"],pressure))
    families={"state":list(range(12)),"relative_dynamics":list(range(12,26)),"current_competition":list(range(26,35)),"same_team_pressure":list(range(35,40)),"opponent_pressure":list(range(40,45)),"corridors":list(range(45,49)),"field_and_team_quality":list(range(49,53))}
    result={"per_match":{},"scope":"Grouped held-match permutation importance and partial dependence, descriptive only. Correlated feature groups and impossible synthetic combinations prevent causal interpretation."}
    for held in FIT:
        cache=RUN/"fits_local"/f"context_interpretation_{held}.json"
        if cache.exists():
            result["per_match"][held]=read(cache);continue
        start=time.perf_counter();tr=valid&(d["match"]!=held);te=valid&(d["match"]==held)
        model=HistGradientBoostingClassifier(**PARAMS).fit(x[tr],y[tr]>0)
        xx=x[te];yy=y[te]>0;p=model.predict_proba(xx)[:,1]
        baseline=float(average_precision_score(yy,p));rng=np.random.default_rng(20261103)
        values={}
        for name,cols in families.items():
            counterfactual=xx.copy()
            counterfactual[:,cols]=xx[rng.permutation(len(xx))][:,cols]
            values[name]=baseline-float(average_precision_score(yy,model.predict_proba(counterfactual)[:,1]))
        subset=rng.choice(len(xx),min(5000,len(xx)),replace=False)
        pdp={}
        for metres in (1,2,4,8,16):
            counterfactual=xx[subset].copy();counterfactual[:,40]=metres/105
            pdp[str(metres)]=float(model.predict_proba(counterfactual)[:,1].mean())
        r={"baseline_AP":baseline,"permutation_AP_drop":values,"opponent_distance_partial_dependence":pdp,"held":held,"features":x.shape[1],"train_rows":int(tr.sum()),"test_rows":int(te.sum()),"seconds":time.perf_counter()-start,"seed":42}
        write(cache,r);result["per_match"][held]=r
        write(RUN/"context_interpretation.json",result)
        print("context interpretation",held,"done",flush=True)
    result["mean_group_AP_drop"]={name:float(np.mean([r["permutation_AP_drop"][name] for r in result["per_match"].values()])) for name in families}
    write(RUN/"context_interpretation.json",result)
    return result


if __name__=="__main__":
    run()
