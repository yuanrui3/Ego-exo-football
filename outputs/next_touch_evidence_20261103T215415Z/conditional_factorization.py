import numpy as np

from common import RUN, FIT, prior, groups, subset, read, write, save
from experiments import compact_features, targets
from followups import lock


def forward_scenes(d,x,held):
    import torch
    from torch import nn
    gg=groups(d);width=max(map(len,gg))
    packed=np.zeros((len(gg),width,x.shape[1]),np.float32)
    mask=np.zeros((len(gg),width),bool)
    for i,rows in enumerate(gg):
        packed[i,:len(rows)]=x[rows];mask[i,:len(rows)]=True
    probs=[]
    for seed in (11,29,47):
        path=RUN/"models_local"/f"listwise_next_BAS_1_2s_{seed}_{held}.pt"
        # These are this run's own locally created checkpoints, never downloaded weights.
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        model=nn.Sequential(nn.Linear(x.shape[1],64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1)).cuda()
        model.load_state_dict(checkpoint["state_dict"]);model.eval()
        tx=torch.tensor(np.clip((packed-checkpoint["mean"])/checkpoint["std"],-10,10),device="cuda")
        tm=torch.tensor(mask,device="cuda")
        with torch.no_grad():
            chunks=[]
            for start in range(0,len(tx),1024):
                chunks.append(torch.softmax(model(tx[start:start+1024]).squeeze(-1).masked_fill(~tm[start:start+1024],-1e4),dim=1).cpu().numpy())
        probs.append(np.concatenate(chunks))
    probabilities=np.mean(probs,axis=0);out=np.zeros(len(x),np.float32)
    for i,rows in enumerate(gg):out[rows]=probabilities[i,:len(rows)]
    return out,gg


def run():
    from sklearn.linear_model import LogisticRegression
    from common import evaluate
    lock("E_unconditional_factorization","B improves next-BAS switch top1 from0.2158 to0.2466 but worsens conditional NLL from1.4042 to1.4293. Determine whether the better conditional choice survives an unconditional upcoming-action test, rather than assuming occurrence is given.","NextBAS1-2s all eligible actor rows: AP/NLL and paired six-match differences against equal scene-probability HGB allocation; positive scenes never supplied to scoring.","Reuse18 existing conditional CUDA checkpoints (3seeds x6folds), no neural retraining. Calibrate HGB maximum to scene occurrence using132877 only; allocate that SAME scene probability with normalized HGB versus learned scene-softmax. Infer all current scenes before target filtering.")
    d=prior.data();dev=prior.data("development")
    f=compact_features(d);df=compact_features(dev);tt=targets(d);dt=targets(dev,cache_name="development_targets_local")
    y=tt["next_BAS_1_2s"];dy=dt["next_BAS_1_2s"]
    with np.load(RUN/"predictions_local"/"calibrated_nextBAS.npz") as z:base=z["raw"]
    allocated=np.full(len(y),np.nan,np.float32);joint=allocated.copy()
    scene_metrics={};calibration={}
    for held in FIT:
        test=d["match"]==held;dd=subset(d,test);xx=f["kinematic"][test]
        q,gg=forward_scenes(dd,xx,held)
        with np.load(RUN/"fits_local"/f"calibration_nextBAS_{held}.npz") as z:devbase=z["dev"]
        dg=groups(dev)
        valid=np.array([np.all(dy[r]>=0) for r in dg])
        dev_scene_y=np.array([np.any(dy[r]>0) for r in dg])
        dev_score=np.array([devbase[r].max() for r in dg])
        logit=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)).reshape(-1,1)
        calibrator=LogisticRegression(C=1000,max_iter=200).fit(logit(dev_score[valid]),dev_scene_y[valid])
        testbase=base[test];test_y=y[test]
        eligible=np.array([np.all(test_y[r]>=0) for r in gg])
        score=np.array([testbase[r].max() for r in gg])
        pscene=np.full(len(gg),np.nan)
        pscene[eligible]=calibrator.predict_proba(logit(score[eligible]))[:,1]
        one=np.full(len(test_y),np.nan,np.float32);two=one.copy()
        for i,rows in enumerate(gg):
            if not eligible[i]:continue
            allocation=testbase[rows]/max(float(testbase[rows].sum()),1e-12)
            one[rows]=allocation*pscene[i];two[rows]=q[rows]*pscene[i]
            np.testing.assert_allclose(two[rows].sum(),pscene[i],rtol=1e-5,atol=1e-7)
        allocated[test]=one;joint[test]=two
        scene_metrics[held]=prior.metrics(np.array([np.any(test_y[r]>0) for r in gg])[eligible],pscene[eligible],probability=pscene[eligible])
        calibration[held]={"slope":float(calibrator.coef_[0,0]),"intercept":float(calibrator.intercept_[0]),"development_scenes":int(valid.sum()),"test_scenes":int(eligible.sum()),"scene_score_identical_between_allocations":True}
        print("unconditional factorization",held,"done",flush=True)
    pred={"original_HGB":base,"scene_calibrated_HGB_allocation":allocated,"scene_calibrated_listwise_allocation":joint}
    r=evaluate(d,y,pred,tuple(pred))
    r["paired_AP"]=prior.contrast(r,"scene_calibrated_listwise_allocation","scene_calibrated_HGB_allocation")
    r["paired_NLL"]=prior.contrast(r,"scene_calibrated_listwise_allocation","scene_calibrated_HGB_allocation","NLL")
    r["scene_occurrence_metrics"]=scene_metrics;r["calibration"]=calibration
    r["scope"]="Unconditional annotation-relative forecast: no future occurrence supplied at scoring. Scene occurrence is identical by construction; any gain is actor-probability allocation. This is the single fresh factorization justified by B, not another scene detector."
    save(RUN/"predictions_local"/"unconditional_factorization.npz",**pred)
    write(RUN/"adaptive_unconditional_factorization.json",r)
    return r


if __name__=="__main__":
    run()
