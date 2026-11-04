import numpy as np
from common import RUN, prior, fit, evaluate, groups, subset, write, save, log
from experiments import targets, compact_features, projection, rules


def run():
    d=prior.data()
    f=compact_features(d)
    labels=targets(d,geometry_horizons=(3,5),bas_windows=(("next_BAS_3_5s",75,125),),cache_name="long_targets_local")
    fs=prior.feature_sets(d,prior.sequence())
    results={}
    for name in ("geometry_3s","geometry_5s","next_BAS_3_5s"):
        horizon=3 if name=="geometry_3s" else 5
        y=labels[name]
        pred=rules(d,f,horizon);probs=[]
        for model,x in (("compact_current",f["state"]),("kinematic",np.column_stack((f["kinematic"],projection(f,horizon)))),("interpretable",fs["interpretable"])):
            pred[model],_=fit("long_"+name+"_"+model,x,d,y)
            probs.append(model)
        r=evaluate(d,y,pred,probs)
        switch=np.zeros(len(y),bool)
        for rows in groups(d):
            if np.all(y[rows]>=0) and np.any(y[rows]>0) and labels["now_nearest"][rows].any() and not labels["now_nearest"][rows[y[rows]>0]].any():
                switch[rows]=True
        r["switch_only"]=evaluate(subset(d,switch),y[switch],{k:v[switch] for k,v in pred.items()},probs) if switch.any() else {}
        results[name]=r
        save(RUN/"predictions_local"/("long_"+name+".npz"),**pred)
        write(RUN/"longer_horizon_results.json",results)
        print("longer",name,r["macro_match"],flush=True)
    log("5-longer","Executed additional3/5-second geometric forecasts and3-5s earliest contact-like BAS actor. These extend beyond previously cached0-3s targets; geometric proximity and incomplete BAS labels are not contact truth.",{k:v["macro_match"] for k,v in results.items()})
    return results


if __name__=="__main__":
    run()
