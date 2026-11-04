import hashlib
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss

from common import RUN, FIT, PARAMS, prior, bas_targets, read, write, save, digest, log
from experiments import compact_features, pressure_features


def metrics(y,p):
    predicted=np.argmax(p,axis=1)+1
    labels=list(range(1,8))
    return {"n":len(y),"top1":float(accuracy_score(y,predicted)),"macro_F1":float(f1_score(y,predicted,labels=labels,zero_division=0,average="macro")),"balanced_accuracy":float(balanced_accuracy_score(y,predicted)),"NLL":float(log_loss(y,p,labels=labels)),"confusion":confusion_matrix(y,predicted,labels=labels).tolist(),"class_support":{prior.old.CLASS_NAMES[k]:int(np.sum(y==k)) for k in labels}}


def run():
    d=prior.data();f=compact_features(d);pressure,_=pressure_features(d);targets,_=bas_targets(d)
    result={}
    for horizon in ("lead_0_1s","lead_1_2s","lead_2_3s"):
        y=targets[horizon];valid=y>0;models={}
        for name,x in (("kinematic",f["kinematic"]),("full_context",np.column_stack((f["kinematic"],pressure)))):
            prob=np.zeros((len(y),7),np.float32)
            for held in FIT:
                tr=valid&(d["match"]!=held);te=valid&(d["match"]==held)
                path=RUN/"fits_local"/f"semantic_{horizon}_{name}_{held}.npz";mp=path.with_suffix(".json")
                signature=hashlib.sha256(x.tobytes()+y.tobytes()+held.encode()).hexdigest()
                if path.exists() and mp.exists() and read(mp)["fingerprint"]==signature:
                    with np.load(path) as z:prob[te]=z["p"]
                    continue
                start=time.perf_counter();model=HistGradientBoostingClassifier(**PARAMS)
                model.fit(x[tr],y[tr])
                pp=np.zeros((int(te.sum()),7),np.float32);pp[:,model.classes_.astype(int)-1]=model.predict_proba(x[te])
                pp=np.clip(pp,1e-8,1);pp/=pp.sum(1,keepdims=True)
                prob[te]=pp;save(path,p=pp,rows=np.flatnonzero(te))
                write(mp,{"fingerprint":signature,"seconds":time.perf_counter()-start,"held":held,"features":x.shape[1],"train_rows":int(tr.sum()),"test_rows":int(te.sum()),"classes":model.classes_.tolist(),"seed":42})
                print("semantics",horizon,name,held,"done",flush=True)
            models[name]=prob
        majority=np.zeros((len(y),7),np.float32)
        for held in FIT:
            tr=valid&(d["match"]!=held);te=valid&(d["match"]==held)
            counts=np.bincount(y[tr],minlength=8)[1:]+1
            majority[te]=counts/counts.sum()
        models["train_class_prior"]=majority
        r={"scope":"Action type conditional on a supported annotated action in the future window; not an unconditional type detector.","pooled":{name:metrics(y[valid],p[valid]) for name,p in models.items()},"per_match":{}}
        for held in FIT:
            te=valid&(d["match"]==held)
            r["per_match"][held]={name:metrics(y[te],p[te]) for name,p in models.items()}
        r["macro_match"]={name:{metric:float(np.mean([r["per_match"][held][name][metric] for held in FIT])) for metric in ("top1","macro_F1","balanced_accuracy","NLL")} for name in models}
        result[horizon]=r
        save(RUN/"predictions_local"/("semantics_"+horizon+".npz"),**models)
        write(RUN/"conditional_action_type.json",result)
    log("7-type","Executed a separate seven-class conditional action-type challenge, rather than relabeling a binary any-action score as a type classifier. Matched kinematics and team context with per-match class priors.",{h:r["macro_match"] for h,r in result.items()})
    return result


if __name__=="__main__":
    run()
