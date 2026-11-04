import numpy as np

from common import RUN, prior, fit, evaluate, groups, write, save
from experiments import targets, compact_features


def run():
    d=prior.data();f=compact_features(d);labels=targets(d);y=labels["geometry_2s"]
    gg=groups(d);pred={}
    for seed in (11,29,47):
        rng=np.random.default_rng(seed);shuffled=y.copy()
        for rows in gg:
            if np.all(y[rows]>=0):
                shuffled[rows]=rng.permutation(y[rows])
        p,_=fit(f"negative_label_shuffle_{seed}",f["kinematic"],d,shuffled)
        pred[f"shuffled_seed{seed}"]=p
    with np.load(RUN/"predictions_local"/"context_geometric_switch_risk.npz") as z:
        pred["true_label_model"]=z["kinematic"]
    result=evaluate(d,y,pred,tuple(pred))
    result["scope"]="Train labels independently permuted within each eligible current scene, preserving one positive and target-only censoring; evaluate against real held-match geometric labels. This is an adversarial null, not an additional scientific improvement."
    save(RUN/"predictions_local"/"negative_controls.npz",**pred)
    write(RUN/"adversarial_negative_controls.json",result)
    return result


if __name__=="__main__":
    run()
