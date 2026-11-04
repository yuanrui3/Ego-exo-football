import copy
import hashlib
import time
from pathlib import Path

import numpy as np

from common import RUN, FIT, prior, source, fit, evaluate, groups, subset, read, write, save, digest, log
from experiments import compact_features, targets


def lock(name,hypothesis,endpoint,plan):
    p=RUN/"adaptive_decisions.json"
    records=read(p) if p.exists() else {}
    if name not in records:
        evidence={f:digest(RUN/f) for f in ("next_interactor_results.json","tactical_context_results.json","scene_prioritization.json","external_results.json","clock_and_sync_audit.json")}
        records[name]={"locked_utc":prior.utc(),"hypothesis":hypothesis,"endpoint":endpoint,"compute_plan":plan,"evidence_sha256":evidence,"data":"six known fitting matches,132877 development only; no exposed retrospective data","expected_failure_modes":["Noisy/incomplete annotation negatives","Six-match uncertainty","Proxy predictability is not physical-touch validity"]}
        write(p,records)


def smoothed_features(d):
    cache=RUN/"features_local"/"smoothed_causal1s.npz"
    if cache.exists():
        with np.load(cache) as z:return z["x"]
    f=compact_features(d);x=f["kinematic"].copy()
    for match in FIT:
        for hc,half in ((1,"1st"),(2,"2nd")):
            mask=(d["match"]==match)&(d["half_code"]==hc)
            rows=np.flatnonzero(mask)
            raw,track=source(match,half)
            q=d["query_frame"][rows].astype(int);slots=d["actor_slot_local_only"][rows].astype(int);seg=d["segment_local_only"][rows]
            now=prior.aligned_state(raw,track,q,slots,seg)
            past=prior.aligned_state(raw,track,q,slots,seg,lag=25)
            old=prior.aligned_state(raw,track,q,slots,seg,lag=20)
            pair=(now[:,0]>.5)&(past[:,0]>.5)
            actor_velocity=(now[:,8:10]-past[:,8:10])*[52.5,34]
            ball_now=now[:,2:4]*[105,68]+now[:,8:10]*[52.5,34]
            ball_past=past[:,2:4]*[105,68]+past[:,8:10]*[52.5,34]
            ball_velocity=ball_now-ball_past
            actor_velocity[~pair]=0;ball_velocity[~pair]=0
            rel=now[:,2:4]*[105,68]
            unit=rel/np.maximum(np.linalg.norm(rel,axis=1)[:,None],1e-6)
            x[rows,5]=(actor_velocity*unit).sum(1)/12
            x[rows,6]=np.linalg.norm(ball_velocity,axis=1)/40
            x[rows,7]=((actor_velocity-ball_velocity)*unit).sum(1)/40
            x[rows,10:12]=actor_velocity/12
            dv=(rel-old[:,2:4]*[105,68])/.8
            good=(now[:,0]>.5)&(old[:,0]>.5)
            vv=(dv*dv).sum(1);defined=good&(vv>.04)
            tca=np.clip(-(rel*dv).sum(1)/np.maximum(vv,.04),0,3)
            x[rows,14]=np.where(good,(now[:,4]-old[:,4])*105/.8/12,0)
            x[rows,18]=np.where(defined,tca/3,0)
            x[rows,19]=defined
            x[rows,20]=np.where(defined,np.linalg.norm(rel+dv*tca[:,None],axis=1)/105,0)
            x[rows,24:26]=np.where(good[:,None],dv/40,0)
    x=np.clip(x,-20,20).astype(np.float32)
    save(cache,x=x)
    return x


def quantization():
    lock("A_quantization","Coarse1.05m/0.68m coordinate steps and>85% repeated native player positions make0.2s velocity unstable; longer strictly backward velocity may improve transferable anticipation without more capacity.","Macro six-match AP and switch-only conditional top1 on geometry2s and nextBAS1-2s; same35-channel capacity","12 HGB fits;1s actor/ball finite differences and0.8s relative approach, fixed endpoint and no threshold tuning")
    d=prior.data();tt=targets(d);x=smoothed_features(d);result={}
    for name in ("geometry_2s","next_BAS_1_2s"):
        p,_=fit("adaptive_smoothed_"+name,x,d,tt[name])
        context="geometric_switch_risk" if name=="geometry_2s" else "next_BAS_actor"
        with np.load(RUN/"predictions_local"/("context_"+context+".npz")) as z:base=z["kinematic"]
        pred={"native_velocity":base,"smoothed_velocity":p}
        r=evaluate(d,tt[name],pred,tuple(pred))
        r["contrast"]=prior.contrast(r,"smoothed_velocity","native_velocity")
        switch=np.zeros(len(p),bool)
        for rows in groups(d):
            if np.all(tt[name][rows]>=0) and np.any(tt[name][rows]>0) and tt["now_nearest"][rows].any() and not tt["now_nearest"][rows[tt[name][rows]>0]].any():
                switch[rows]=True
        r["switch_only"]=evaluate(subset(d,switch),tt[name][switch],{k:v[switch] for k,v in pred.items()},tuple(pred))
        result[name]=r;save(RUN/"predictions_local"/("smooth_"+name+".npz"),**pred)
        write(RUN/"adaptive_quantization.json",result)
    return result


def pack_scenes(d,x,y):
    gg=[rows for rows in groups(d) if np.all(y[rows]>=0) and np.sum(y[rows]>0)==1]
    width=max(map(len,gg));features=np.zeros((len(gg),width,x.shape[1]),np.float32)
    mask=np.zeros((len(gg),width),bool);label=np.zeros(len(gg),np.int64);matches=[]
    for i,rows in enumerate(gg):
        features[i,:len(rows)]=x[rows];mask[i,:len(rows)]=True
        label[i]=np.flatnonzero(y[rows]>0)[0];matches.append(d["match"][rows[0]])
    return gg,features,mask,label,np.array(matches)


def listwise(objective="softmax"):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required by locked listwise experiment")
    if objective not in ("softmax","binary"):
        raise ValueError("Unknown conditional training objective")
    if objective=="softmax":
        lock("B_conditional_ranking","External constant-velocity switch top1 outperforms its AP ranking, and HGB classification is not optimized for conditional actor choice; a shared permutation-equivariant candidate scorer with scene softmax may improve choice.","Conditional occurrence-known scene top1/MRR/NLL on geometry2s and nextBAS1-2s, including current-nearest switches; never an unconditional event detector.","Two targets, six held-match folds, matched seeds11/29/47,25 epochs,512 scenes/batch, AdamW0.001; epoch selection only on132877 positive scenes")
    else:
        lock("G_matched_objective_control","B versus HGB confounds architecture, ensemble and training objective, and E found essentially no unconditional AP gain. Isolate softmax versus independent binary loss with the EXACT same candidate MLP, train scenes, seeds, initialization, batch schedule and conditional-development-NLL selection.","Paired conditional top1/MRR/NLL and switch ranking for geometry2s and nextBAS1-2s; compare against B with three matched seeds. This is not a new unconditional forecast.","36 CUDA fits,25 epochs,512 scenes/batch, same AdamW; binary logits trained with unweighted candidate BCE, normalized sigmoid for categorical development/test NLL. No external data or threshold tuning.")
    tag="listwise" if objective=="softmax" else "binary_objective"
    ensemble_name="listwise_ensemble" if objective=="softmax" else "binary_ensemble"
    output_file="adaptive_listwise.json" if objective=="softmax" else "adaptive_objective_control.json"
    d=prior.data();dev=prior.data("development")
    f=compact_features(d);df=compact_features(dev)
    tt=targets(d);dt=targets(dev,cache_name="development_targets_local")
    result={}
    checkpoint_dir=RUN/"models_local";checkpoint_dir.mkdir(exist_ok=True)
    for name in ("geometry_2s","next_BAS_1_2s"):
        gg,x,mask,y,match=pack_scenes(d,f["kinematic"],tt[name])
        dg,dx,dm,dy,_=pack_scenes(dev,df["kinematic"],dt[name])
        seed_predictions=[]
        metadata=[]
        for seed in (11,29,47):
            out=np.full(len(d["target"]),np.nan,np.float32)
            for held in FIT:
                path=RUN/"fits_local"/f"{tag}_{name}_{seed}_{held}.npz"
                meta_path=path.with_suffix(".json")
                tr=match!=held;te=match==held
                signature=hashlib.sha256(x.tobytes()+y.tobytes()+dx.tobytes()+dy.tobytes()+str((seed,held,25)).encode()+(b"" if objective=="softmax" else b":binary-objective-v1")).hexdigest()
                if path.exists() and meta_path.exists() and read(meta_path)["fingerprint"]==signature and read(meta_path).get("sha256")==digest(path):
                    with np.load(path) as z:out[z["rows"]]=z["p"]
                    metadata.append(read(meta_path));continue
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                mean=x[tr][mask[tr]].mean(0);std=x[tr][mask[tr]].std(0).clip(.05)
                tx=torch.tensor(np.clip((x-mean)/std,-10,10),device="cuda")
                td=torch.tensor(np.clip((dx-mean)/std,-10,10),device="cuda")
                tm=torch.tensor(mask,device="cuda");tdm=torch.tensor(dm,device="cuda")
                ty=torch.tensor(y,device="cuda");tdy=torch.tensor(dy,device="cuda")
                model=nn.Sequential(nn.Linear(x.shape[2],64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1)).cuda()
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                train=np.flatnonzero(tr);rng=np.random.default_rng(seed)
                best=float("inf");best_epoch=-1;weights=None;steps=0;start=time.perf_counter()
                for epoch in range(25):
                    model.train()
                    for batch in np.array_split(rng.permutation(train),max(1,int(np.ceil(len(train)/512)))):
                        index=torch.tensor(batch,device="cuda")
                        logits=model(tx[index]).squeeze(-1).masked_fill(~tm[index],-1e4)
                        if objective=="softmax":
                            loss=nn.functional.cross_entropy(logits,ty[index])
                        else:
                            binary_target=nn.functional.one_hot(ty[index],num_classes=logits.shape[1]).float()
                            loss=nn.functional.binary_cross_entropy_with_logits(logits[tm[index]],binary_target[tm[index]])
                        optimizer.zero_grad();loss.backward();optimizer.step();steps+=1
                    model.eval()
                    with torch.no_grad():
                        dev_logits=model(td).squeeze(-1).masked_fill(~tdm,-1e4)
                        if objective=="softmax":
                            dev_loss=nn.functional.cross_entropy(dev_logits,tdy).item()
                        else:
                            dp=torch.sigmoid(dev_logits)*tdm
                            dp=dp/dp.sum(1,keepdim=True).clamp_min(1e-12)
                            dev_loss=-torch.log(dp[torch.arange(len(tdy),device="cuda"),tdy].clamp_min(1e-12)).mean().item()
                    if dev_loss<best:
                        best=dev_loss;best_epoch=epoch+1;weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(weights)
                model.eval()
                with torch.no_grad():
                    logits=model(tx[te]).squeeze(-1).masked_fill(~tm[te],-1e4)
                    if objective=="softmax":
                        prob=torch.softmax(logits,dim=1).cpu().numpy()
                    else:
                        pp=torch.sigmoid(logits)*tm[te]
                        prob=(pp/pp.sum(1,keepdim=True).clamp_min(1e-12)).cpu().numpy()
                indices=[];values=[]
                for j,scene in enumerate(np.flatnonzero(te)):
                    indices.extend(gg[scene]);values.extend(prob[j,:len(gg[scene])])
                indices=np.array(indices);values=np.array(values,np.float32);out[indices]=values
                save(path,rows=indices,p=values)
                torch.save({"state_dict":weights,"mean":mean,"std":std},checkpoint_dir/f"{tag}_{name}_{seed}_{held}.pt")
                meta={"fingerprint":signature,"sha256":digest(path),"held":held,"seed":seed,"steps":steps,"epochs":25,"selected_epoch":best_epoch,"development_NLL":best,"seconds":time.perf_counter()-start,"train_scenes":int(tr.sum()),"test_scenes":int(te.sum()),"development_scenes":len(dg),"gpu":torch.cuda.get_device_name(0),"peak_GPU_allocated_MB":torch.cuda.max_memory_allocated()/1e6,"parameters":sum(p.numel() for p in model.parameters())}
                write(meta_path,meta);metadata.append(meta)
                print("listwise",name,seed,held,"done",flush=True)
            seed_predictions.append(out)
        with np.load(RUN/"predictions_local"/("context_"+("geometric_switch_risk" if name=="geometry_2s" else "next_BAS_actor")+".npz")) as z:base=z["kinematic"]
        chosen=np.concatenate(gg);eligible=np.zeros(len(d["target"]),bool);eligible[chosen]=True
        averaged=np.nanmean(np.stack(seed_predictions)[:,eligible],axis=0)
        pred={"hgb":base[eligible],ensemble_name:averaged}
        dd=subset(d,eligible);yy=tt[name][eligible]
        r=evaluate(dd,yy,pred)
        r["per_seed"]={str(seed):evaluate(dd,yy,{"listwise":p[eligible]}) for seed,p in zip((11,29,47),seed_predictions)}
        r["conditional_NLL"]={}
        for model,p in pred.items():
            values=[]
            for rows in groups(dd):
                norm=p[rows]/max(float(p[rows].sum()),1e-12)
                values.append(-np.log(max(float(norm[yy[rows]>0].sum()),1e-12)))
            r["conditional_NLL"][model]=float(np.mean(values))
        switch_scene=np.zeros(len(yy),bool)
        for rows in groups(dd):
            oldnear=tt["now_nearest"][eligible][rows]
            switch_scene[rows]=oldnear.any() and not oldnear[yy[rows]>0].any()
        r["switch_only"]=evaluate(subset(dd,switch_scene),yy[switch_scene],{k:p[switch_scene] for k,p in pred.items()})
        r["training"]=metadata;r["scope"]="Conditional on supported future occurrence for this diagnostic; not unconditional anticipation."
        r["training_objective"]=objective
        result[name]=r;write(RUN/output_file,result)
        save(RUN/"predictions_local"/(tag+"_"+name+".npz"),rows=np.flatnonzero(eligible),**{f"seed{s}":p[eligible] for s,p in zip((11,29,47),seed_predictions)})
    return result


def reliability(y,p):
    y=np.asarray(y)>0
    bins=[]
    for lo,hi in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
        mask=(p>=lo)&(p<hi if hi<1 else p<=1)
        if mask.any():bins.append({"n":int(mask.sum()),"predicted":float(p[mask].mean()),"observed":float(y[mask].mean())})
    return {"bins":bins,"ECE":sum(b["n"]*abs(b["predicted"]-b["observed"]) for b in bins)/len(y)}


def calibration():
    from sklearn.linear_model import LogisticRegression
    lock("C_calibrated_abstention","One external match does not establish operational reliability; incomplete labels and current sparse positives make raw probability calibration and evidence-conditioned coverage essential.","NextBAS1-2s six-match NLL/Brier/ECE before/after development-only Platt calibration; coverage and retained annotated positives under evidence-only abstention","Six35-feature HGB folds with independent132877 predictions; one monotone scalar calibration per fold, no test thresholds or external tuning")
    d=prior.data();dev=prior.data("development");f=compact_features(d);df=compact_features(dev)
    tt=targets(d);dt=targets(dev,cache_name="development_targets_local")
    y=tt["next_BAS_1_2s"];dy=dt["next_BAS_1_2s"];valid=dy>=0
    raw,dp=fit("calibration_nextBAS",f["kinematic"],d,y,dev=df["kinematic"])
    calibrated=raw.copy();coefficients={}
    logit=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)).reshape(-1,1)
    for i,held in enumerate(FIT):
        model=LogisticRegression(C=1e3,max_iter=200)
        model.fit(logit(dp[i,valid]),dy[valid]>0)
        mask=(d["match"]==held)&(y>=0)
        calibrated[mask]=model.predict_proba(logit(raw[mask]))[:,1]
        coefficients[held]={"slope":float(model.coef_[0,0]),"intercept":float(model.intercept_[0]),"dev_rows":int(valid.sum()),"dev_positives":int((dy[valid]>0).sum())}
    pred={"raw":raw,"development_calibrated":calibrated}
    result=evaluate(d,y,pred,tuple(pred));eligible=y>=0
    result["reliability"]={k:reliability(y[eligible],p[eligible]) for k,p in pred.items()}
    quality=(f["sequence"][:,:,0]>.5).mean(1)
    result["evidence_abstention"]={}
    for threshold in (.5,.8,1.):
        mask=eligible&(quality>=threshold)
        result["evidence_abstention"][str(threshold)]={"row_coverage":float(mask.sum()/eligible.sum()),"annotated_positive_retention":float(np.sum((y>0)&mask)/np.sum(y>0)),"metrics":evaluate(subset(d,mask),y[mask],{k:p[mask] for k,p in pred.items()},tuple(pred))}
    result["calibration_coefficients"]=coefficients
    result["scope"]="Development-domain calibration can worsen known-fit transfer; descriptive reliability against annotations, not physically verified event probabilities."
    save(RUN/"predictions_local"/"calibrated_nextBAS.npz",**pred)
    write(RUN/"adaptive_calibration.json",result)
    return result


def followups():
    statuses=read(RUN/"stage_status.json")
    if any(statuses.get(str(i),{}).get("status")!="COMPLETE" for i in (0,3,4,5,6,7,8)):
        raise RuntimeError("Main evidence gates must complete before adaptive experiments")
    results={}
    for name,fn in (("A",quantization),("B",listwise),("C",calibration)):
        results[name]=fn()
        write(RUN/"adaptive_completion.json",{"status":"RUNNING","completed":list(results)})
    log(9,"Executed three distinct result-motivated hypotheses: quantization-aware velocities, conditional GPU listwise choice, and development-calibrated evidence abstention. Each was frozen after main results, not a random-seed relabeling.",{"completed":list(results)})
    write(RUN/"adaptive_completion.json",{"status":"COMPLETE","completed":list(results)})
    return {"status":"COMPLETE","completed":list(results)}


if __name__=="__main__":
    followups()
