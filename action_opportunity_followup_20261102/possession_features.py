"""As-of ball-proximity features for prospective actor-time forecasting.

Ball tracks are EXOCENTRIC reference data. Proximity is NOT proof of possession,
player awareness, or a measured player-view ball observation.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from followup_common import (PRIOR_REL,OUT_REL,local_path,read_json,sha256,
                              validate_assets,write_json)

BALL_FEATURES=('ball_available','ball_age_0p2s','ball_dx_105m','ball_dy_68m',
               'actor_ball_distance_105m','actor_approach_12mps','ball_speed_40mps',
               'relative_closing_speed_40mps','within_2m','within_5m',
               'proximity_rank_score','nearest_actor_distance_105m')

@dataclass
class BallTrack:
    frames: np.ndarray
    xy: np.ndarray
    valid: np.ndarray
    path: Path
    digest: str
    status: str
    reason: str
    def asof(self,queries,max_age_frames=5):
        q=np.asarray(queries,dtype=np.int64)
        xy=np.zeros((len(q),2),np.float32); velocity=np.zeros_like(xy)
        ages=np.ones(len(q),np.float32)
        valid=np.zeros(len(q),bool)
        if not len(self.frames):return xy,velocity,ages,valid
        idx=np.searchsorted(self.frames,q,side='right')-1
        inrange=idx>=0
        good_idx=np.maximum(idx,0)
        gaps=q-self.frames[good_idx]
        valid=inrange&(gaps>=0)&(gaps<=max_age_frames)&self.valid[good_idx]
        ages=np.where(valid,gaps/25.,1.).astype(np.float32)
        xy[valid]=self.xy[good_idx[valid]]
        prev=np.maximum(good_idx-1,0)
        dt=(self.frames[good_idx]-self.frames[prev])/25.
        previous_valid=(good_idx>0)&self.valid[prev]&(dt>0)&(dt<=.24)
        both=valid&previous_valid
        velocity[both]=(self.xy[good_idx[both]]-self.xy[prev[both]])/dt[both,None]
        return xy,velocity,ages,valid

def load_ball(path:Path)->BallTrack:
    if not path.is_file():
        return BallTrack(np.zeros(0,np.int64),np.zeros((0,2),np.float32),
                         np.zeros(0,bool),path,'','MISSING','No ball archive')
    digest=sha256(path)
    with np.load(path,allow_pickle=False) as blob:
        keys=set(blob.files)
        if not {'frame','x','y'}<=keys:
            return BallTrack(np.zeros(0,np.int64),np.zeros((0,2),np.float32),
                             np.zeros(0,bool),path,digest,'UNSUPPORTED_SCHEMA',f'keys={sorted(keys)}')
        rawframe=np.asarray(blob['frame']);x=np.asarray(blob['x'],dtype=np.float64)
        y=np.asarray(blob['y'],dtype=np.float64)
        observed=None
        for name in ('observed','valid','visible'):
            if name in keys and np.asarray(blob[name]).shape==x.shape:
                observed=np.asarray(blob[name]).astype(bool);break
    if not (rawframe.ndim==1 and x.ndim==1 and y.ndim==1 and len(rawframe)==len(x)==len(y)):
        raise ValueError(f'Invalid ball track shapes: {path}')
    if not len(x):return BallTrack(np.zeros(0,np.int64),np.zeros((0,2),np.float32),np.zeros(0,bool),path,digest,'EMPTY','No rows')
    if np.issubdtype(rawframe.dtype,np.integer):frames=rawframe.astype(np.int64)
    elif np.isfinite(rawframe).all() and np.all(rawframe==np.floor(rawframe)):
        frames=rawframe.astype(np.int64)
    else:raise ValueError(f'Invalid ball frame IDs: {path}')
    order=np.argsort(frames,kind='stable');frames=frames[order]
    xy=np.column_stack((x[order],y[order])).astype(np.float32)
    if observed is None: observed=np.ones(len(frames),bool)
    else:observed=observed[order]
    if np.any(frames[1:]<=frames[:-1]):
        return BallTrack(frames,xy,np.zeros(len(frames),bool),path,digest,'BAD_FRAME_INDEX','Duplicated or unsorted frame indices')
    valid=observed&np.isfinite(xy).all(axis=1)&(np.abs(xy[:,0])<=52.5)&(np.abs(xy[:,1])<=34)
    # No automatic coordinate conversion: unknown pixel/normalized coordinates block claims.
    if valid.mean()<.20:
        status='COORDINATES_UNVERIFIED'
        reason='Fewer than 20% finite points lie in pitch-centred metre bounds; do not guess coordinate conversion'
    else:status='PITCH_BOUNDS_PLAUSIBLE';reason='Pitch-coordinate bounds only; event-time calibration still required'
    return BallTrack(frames,xy,valid,path,digest,status,reason)

def _events_actor_ball_distances(repo:Path,match:str,half:str,track:BallTrack,raw,assoc)->list[float]:
    """Training/development-only ball alignment diagnostic; NEVER a model input."""
    import sys
    if str(repo) not in sys.path:sys.path.insert(0,str(repo))
    from src.backbone.event_time import resolve_event_time
    doc=read_json(repo/'data'/f'{match}_12_class_events.json')
    md=read_json(repo/'outputs'/'trajectory_forecasting_20261016'/'cache'/f'{match}_{half}.json')
    distances=[]; slots=assoc['slots']
    for event in doc.get('actions',[]):
        period=str(event.get('gameTime','')).split(' - ')[0].strip()
        if period!=('1' if half=='1st' else '2'):continue
        if str(event.get('label','')).upper() not in ('PASS','DRIVE','SHOT','CROSS','HIGH PASS'):continue
        pid=str(event.get('player_id'))
        if pid not in slots:continue
        r=resolve_event_time(event.get('position'),event.get('frame'),clock=doc.get('clock'),
                             gsr_info=md['gsr_info'],source_frame_count=len(raw['observed']))
        if r.status!='supported':continue
        fr=int(r.native_zero_based_frame);slot=int(slots[pid])
        if not raw['observed'][fr,slot]:continue
        ball,_,_,good=track.asof(np.asarray([fr],dtype=np.int64))
        if good[0]:
            distances.append(float(np.linalg.norm(raw['xy'][fr,slot]-ball[0])))
    return distances

def run(repo:Path,prior:Path,out:Path):
    manifest,prov=validate_assets(repo,prior)
    featdir=out/'ball_feature_shards_local';featdir.mkdir(parents=True,exist_ok=True)
    reports=[];all_fit_dists=[];per_split_valid={'fit':[0,0],'development':[0,0],'previously_exposed_retrospective':[0,0]}
    for shard in manifest['shards']:
        match=str(shard['match']);half=str(shard['half']);split=shard['split']
        key=f'{match}_{half}'
        ballpath=repo/'data'/f'{key}_ball.npz'
        track=load_ball(ballpath)
        shardpath=local_path(repo,shard['shard_path'])
        cache_path=repo/'outputs'/'trajectory_forecasting_20261016'/'cache'/f'{key}.npz'
        association_path=repo/'outputs'/'trajectory_audit_20261016'/f'associations_{key}.json'
        if sha256(cache_path)!=shard['source_cache_sha256'] or sha256(association_path)!=shard['association_sha256']:
            raise ValueError(f'Upstream tracking or association hash mismatch for {key}')
        with np.load(shardpath,allow_pickle=False) as src:
            queries=src['query_frame'].astype(np.int64)
            slots=src['actor_slot_local_only'].astype(np.int64)
            segment=src['segment_local_only'].astype(np.int64)
            x=src['x_causal_uncertainty'].astype(np.float32)
            scenes=src['scene_id_local_only'].astype(np.int64)
        # x[:, 0:2] stores the CURRENT actor's position, not the future target.
        actor_xy=x[:,:2]*np.array([52.5,34.0],np.float32)
        actor_v=x[:,2:4]*12.
        # Causal row join: never trust row ordering without a source actor check.
        with np.load(cache_path,allow_pickle=False) as source:
            raw_observed=source['observed'][queries,slots]
            raw_segments=source['segments'][queries,slots]
            raw_actor_xy=source['xy'][queries,slots].astype(np.float32)
        max_delta=float(np.max(np.linalg.norm(raw_actor_xy-actor_xy,axis=1)))
        if (not raw_observed.all() or not np.array_equal(raw_segments,segment)
                or max_delta>.2):
            raise ValueError(f'Actor-time source row mismatch {key}: max coordinate error {max_delta:.3f}m')
        ball_xy,ball_v,ages,good=track.asof(queries)
        feat=np.zeros((len(x),len(BALL_FEATURES)),np.float32)
        if track.status=='PITCH_BOUNDS_PLAUSIBLE':
            dxy=ball_xy-actor_xy
            dist=np.linalg.norm(dxy,axis=1)
            unit=dxy/np.maximum(dist[:,None],1e-6)
            approach=np.sum(actor_v*unit,axis=1)
            bspeed=np.linalg.norm(ball_v,axis=1)
            close=np.sum((actor_v-ball_v)*unit,axis=1)
            feat[:,0]=good.astype(float)
            feat[:,1]=np.where(good,ages/.2,0)
            feat[:,2]=np.where(good,dxy[:,0]/105,0)
            feat[:,3]=np.where(good,dxy[:,1]/68,0)
            feat[:,4]=np.where(good,dist/105,0)
            feat[:,5]=np.where(good,np.clip(approach/12,-2,2),0)
            feat[:,6]=np.where(good,np.minimum(bspeed/40,2),0)
            feat[:,7]=np.where(good,np.clip(close/40,-2,2),0)
            feat[:,8]=good&(dist<=2)
            feat[:,9]=good&(dist<=5)
            # Within each causal scene, actor proximity rank is derived only from
            # the currently observed actors, never future labels or hidden actor truth.
            starts=np.r_[0,np.flatnonzero(scenes[1:]!=scenes[:-1])+1,len(scenes)]
            for a,b in zip(starts[:-1],starts[1:]):
                if good[a] and np.all(good[a:b]):
                    ranks=np.argsort(np.argsort(dist[a:b],kind='stable'),kind='stable')
                    feat[a:b,10]=1-ranks/max(1,b-a)
                    feat[a:b,11]=dist[a:b].min()/105
            feat[~good]=0
        else:
            good[:]=False
        if not np.isfinite(feat).all():raise FloatingPointError(f'Invalid ball features: {key}')
        per_split_valid[split][0]+=int(good.sum());per_split_valid[split][1]+=len(good)
        fitdist=[]
        if split in ('fit','development') and track.status=='PITCH_BOUNDS_PLAUSIBLE':
            with np.load(cache_path,allow_pickle=False) as src:
                raw={k:src[k] for k in ('xy','observed')}
            assoc=read_json(association_path)
            fitdist=_events_actor_ball_distances(repo,match,half,track,raw,assoc)
            if split=='fit':all_fit_dists.extend(fitdist)
        path=featdir/f'{key}.npz'
        np.savez_compressed(path,features=feat,match=np.full(len(feat),match),
            half_code=np.full(len(feat),1 if half=='1st' else 2,dtype=np.int8),
            query_frame=queries.astype(np.int32),actor_slot_local_only=slots.astype(np.int16),
            segment_local_only=segment.astype(np.int16))
        reports.append({'match':match,'half':half,'split':split,'ball_status':track.status,
                        'ball_reason':track.reason,'ball_file_sha256':track.digest,
                        'ball_file':str(ballpath.relative_to(repo)),
                        'ball_frame_count':len(track.frames),
                        'ball_frame_valid_fraction':float(np.mean(track.valid)) if len(track.valid) else 0,
                        'query_rows':len(good),'query_actor_ball_available':int(good.sum()),
                        'actor_source_row_max_position_delta_m':max_delta,
                        'fit_or_dev_action_ball_distances_n':len(fitdist),
                        'fit_or_dev_action_ball_distance_median_m':float(np.median(fitdist)) if fitdist else None,
                        'feature_shard':str(path.relative_to(out)),
                        'feature_shard_sha256':sha256(path),'source_shard_sha256':shard['shard_sha256']})
    median=float(np.median(all_fit_dists)) if all_fit_dists else None
    p90=float(np.percentile(all_fit_dists,90)) if all_fit_dists else None
    fit_fraction=per_split_valid['fit'][0]/max(1,per_split_valid['fit'][1])
    # Conservative exocentric alignment gate: not a perception/gaze claim.
    accept=bool(len(all_fit_dists)>=100 and median is not None and median<=8 and p90<=18 and fit_fraction>=.2)
    summary={'status':'BALL_ALIGNMENT_PASS' if accept else 'BALL_ALIGNMENT_BLOCKED',
             'feature_names':BALL_FEATURES,'source_checks':prov,'halves':reports,
             'fit_event_ball_distance_count':len(all_fit_dists),'fit_event_ball_distance_median_m':median,
             'fit_event_ball_distance_p90_m':p90,
             'fit_actor_time_ball_valid_fraction':fit_fraction,
             'coverage_by_split':{k:{'valid_rows':v[0],'rows':v[1],'fraction':v[0]/max(1,v[1])}
                                 for k,v in per_split_valid.items()},
             'claim_boundary':'As-of EXOCENTRIC ball location and proximity, not measured possession or player perception. Train-time action labels only audit alignment; they are never input features.',
             'block_reason':None if accept else 'Missing/unsupported/unverified ball coordinates or insufficient training-only event alignment. Do not train ball-augmented claims.'}
    write_json(out/'ball_alignment_audit.json',summary)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--prior',type=Path,default=Path(PRIOR_REL));p.add_argument('--out',type=Path,default=Path(OUT_REL))
    a=p.parse_args();repo=a.repo.resolve();prior=(repo/a.prior).resolve();out=(repo/a.out).resolve()
    result=run(repo,prior,out)
    print(result['status'],'fit event alignment n=',result['fit_event_ball_distance_count'])
    print(out/'ball_alignment_audit.json')
if __name__=='__main__':main()
