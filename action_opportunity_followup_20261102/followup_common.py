"""Read-only adapters to the immutable November 2 actor-time experiment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

PRIOR_REL = 'outputs/action_opportunity_research_20261102_144500'
OUT_REL = 'outputs/action_opportunity_followup_20261102'
SEEDS = (42,137,901)
CLASS_NAMES = ('NO_ANNOTATED_EVENT_WITHIN_HORIZON','PASS','HIGH_PASS','DRIVE','CROSS','SHOT','HEADER','DEFENSIVE_INTERVENTION')
FEATURE = 'x_causal_uncertainty'

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path: Path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def local_path(repo: Path, relative: str) -> Path:
    return repo/Path(relative.replace('\\','/'))

def validate_assets(repo: Path, prior: Path, *, verify_shards=True):
    m=read_json(prior/'cohort_manifest.json')
    if m.get('status')!='COMPLETE' or len(m.get('shards',[]))!=18:
        raise ValueError('Expected the complete immutable 18-shard cohort manifest')
    protected=read_json(prior/'protected_asset_verification.json')
    pins=[]
    for item in protected['protected_checkpoints']:
        p=local_path(repo,item['path'])
        if not p.is_file() or sha256(p)!=item['expected_sha256']:
            raise ValueError(f'Protected checkpoint absent or changed: {p}')
        pins.append({'path':item['path'],'sha256':item['expected_sha256']})
    rows=[]
    if verify_shards:
        for shard in m['shards']:
            p=local_path(repo,shard['shard_path'])
            if not p.is_file() or sha256(p)!=shard['shard_sha256']:
                raise ValueError(f'Cohort shard absent or changed: {p}')
            rows.append({'path':shard['shard_path'],'sha256':shard['shard_sha256']})
    return m,{'protected':pins,'shards':rows,'manifest_sha256':sha256(prior/'cohort_manifest.json')}

def read_cohort(repo:Path, manifest:dict, *, splits=('fit','development','previously_exposed_retrospective'), features=True, private_keys=False):
    keys=['target','match','half_code','query_frame','scene_id_local_only']
    if private_keys: keys.extend(['actor_slot_local_only','segment_local_only'])
    results={}
    for split in splits:
        parts={k:[] for k in keys};features_parts=[]
        for shard in manifest['shards']:
            if shard['split']!=split:continue
            with np.load(local_path(repo,shard['shard_path']),allow_pickle=False) as z:
                for k in keys:parts[k].append(z[k])
                if features:features_parts.append(z[FEATURE].astype(np.float32))
        if not parts['target']:raise ValueError(f'No cohort rows for split {split}')
        d={k:np.concatenate(v) for k,v in parts.items()}
        d['target']=d['target'].astype(np.int64)
        if features:d['x']=np.concatenate(features_parts)
        results[split]=d
    return results

def hazard_scores(p):
    p=np.asarray(p,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=len(CLASS_NAMES) or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('Invalid N x 8 probability matrix')
    if not np.allclose(p.sum(axis=1),1,rtol=0,atol=2e-4):
        raise ValueError('Probabilities not normalized')
    return p[:,1:].sum(axis=1)

def scene_ranking(y,hazard,d):
    """Any positive actor top-k among players at same match,half,query scene."""
    groups={}
    for i,key in enumerate(zip(d['match'].astype(str),d['half_code'],d['scene_id_local_only'])):
        groups.setdefault(key,[]).append(i)
    ranks=[]
    for idx in groups.values():
        positive=[i for i in idx if y[i]>0]
        if not positive:continue
        ordered=sorted(idx,key=lambda i:(-hazard[i],i))
        rank={index:i+1 for i,index in enumerate(ordered)}
        ranks.append(min(rank[i] for i in positive))
    if not ranks:return {'positive_scenes':0,'top1':None,'top3':None,'mrr':None}
    ranks=np.asarray(ranks)
    return {'positive_scenes':len(ranks),'top1':float(np.mean(ranks<=1)),
            'top3':float(np.mean(ranks<=3)),'mrr':float(np.mean(1/ranks))}

def binary_metrics(y, hazard, data=None):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    y=np.asarray(y)>0;h=np.clip(np.asarray(hazard,dtype=np.float64),1e-12,1-1e-12)
    result={'n':len(y),'positives':int(y.sum()),'prevalence':float(np.mean(y)),
            'AP':float(average_precision_score(y,h)) if y.any() else None,
            'brier':float(brier_score_loss(y,h)),
            'hazard_NLL':float(log_loss(y.astype(int),np.column_stack((1-h,h)),labels=[0,1]))}
    if data is not None:result['scene_ranking']=scene_ranking(y,h,data)
    return result

def paired_match_bootstrap(per_match_a,per_match_b,metric='AP',lower_is_better=False,seed=42):
    matches=sorted(set(per_match_a)&set(per_match_b))
    deltas=np.array([per_match_a[m][metric]-per_match_b[m][metric] for m in matches],float)
    if len(matches)<2:return None
    rng=np.random.default_rng(seed)
    draws=deltas[rng.integers(0,len(deltas),size=(10000,len(deltas)))].mean(axis=1)
    return {'n_matches':len(matches),'mean_delta':float(deltas.mean()),
            'ci95':np.quantile(draws,[0.025,0.975]).tolist(),
            'per_match_delta':dict(zip(matches,deltas.tolist())),
            'lower_is_better':lower_is_better,
            'caveat':'Match-bootstrap with few source matches; descriptive only.'}
