import argparse, json
from pathlib import Path
import ijson

VALID_LABELS={"Pass","Drive","Header","High Pass","Out","Cross","Throw In","Shot","Ball Player Block","Player Successful Tackle","Free Kick","Goal"}

def load_bas(path,period):
    data=json.loads(Path(path).read_text(encoding='utf-8')); rows=[]
    events=data.get('annotations') or data.get('actions') or []
    for i,e in enumerate(events):
        try: half=int(str(e.get('gameTime','')).split(' - ')[0].strip())
        except: continue
        if half!=period: continue
        label=str(e.get('label','')).strip().title()
        if label not in VALID_LABELS: raise ValueError(f'Unknown BAS label: {label}')
        frame=e.get('frame')
        if frame is None: frame=round(int(e.get('position',0))/40.0)
        rows.append({'idx':i,'frame':int(frame),'timestamp_ms':int(e.get('position',0)),'label':label,'team':e.get('team'),'player_id':e.get('player_id'),'visibility':e.get('visibility')})
    return rows

def decode_frame(image_id):
    s=str(image_id)
    if s.isdigit() and len(s)>=7: return int(s[-6:])-1
    try: return int(s)
    except: return None

def collect_ids(gsr,target_frames):
    targets=set()
    for f in target_frames: targets.update({f-1,f,f+1})
    mapping={}
    with open(gsr,'rb') as fh:
        for im in ijson.items(fh,'images.item'):
            iid=im.get('image_id'); cand=[]
            fr=decode_frame(iid)
            if fr is not None: cand.append(fr)
            stem=Path(str(im.get('file_name',''))).stem
            if stem.isdigit(): cand += [int(stem),int(stem)-1]
            for c in cand:
                if c in targets: mapping.setdefault(c,iid)
    return mapping

def stream_states(gsr,image_ids):
    needed={str(x) for x in image_ids if x is not None}; states={x:[] for x in needed}
    with open(gsr,'rb') as fh:
        for a in ijson.items(fh,'annotations.item'):
            iid=str(a.get('image_id'))
            if iid not in needed or a.get('supercategory')!='object': continue
            attrs=a.get('attributes') or {}; role=attrs.get('role')
            if role not in ('player','goalkeeper'): continue
            bp=a.get('bbox_pitch') or {}; x=bp.get('x_bottom_middle'); y=bp.get('y_bottom_middle')
            if x is None or y is None: continue
            states[iid].append({'player_id':attrs.get('player_id',a.get('track_id')),'track_id':a.get('track_id'),'team':attrs.get('team'),'role':role,'jersey':attrs.get('jersey'),'x':float(x),'y':float(y),'vx':None,'vy':None})
    return states

def choose_state(frame,mapping,states):
    for fr in (frame,frame-1,frame+1):
        iid=mapping.get(fr); players=states.get(str(iid),[]) if iid is not None else []
        if players: return fr,iid,players
    return None,None,[]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bas',required=True); ap.add_argument('--gsr',required=True); ap.add_argument('--match-id',required=True); ap.add_argument('--period',required=True,type=int,choices=[1,2]); ap.add_argument('--output',required=True); ap.add_argument('--include-actor-missing',action='store_true',help='include actor-missing records for diagnostics (excluded by default)'); a=ap.parse_args()
    events=load_bas(a.bas,a.period); mapping=collect_ids(a.gsr,[e['frame'] for e in events]); states=stream_states(a.gsr,mapping.values())
    written=missing=actor_missing=state_matches=skipped_actor_missing=0
    offset_counts={-1:0,0:0,1:0}
    with open(a.output,'w',encoding='utf-8') as out:
        for e in events:
            matched,iid,players=choose_state(e['frame'],mapping,states)
            if not players: missing+=1; continue
            state_matches+=1
            frame_offset=matched-e['frame']
            offset_counts[frame_offset]+=1
            actor_ids={str(p['player_id']) for p in players}
            actor_present=e['player_id'] is not None and str(e['player_id']) in actor_ids
            if not actor_present:
                actor_missing+=1
                if not a.include_actor_missing:
                    skipped_actor_missing+=1
                    continue
            row={'decision_id':f"{a.match_id}_p{a.period}_f{e['frame']}_{e['idx']}",'match_id':str(a.match_id),'period':a.period,'timestamp_ms':e['timestamp_ms'],'frame':e['frame'],'matched_gsr_frame':matched,'gsr_frame_offset':frame_offset,'gsr_image_id':iid,'possession_team':None,'actor_id':e['player_id'],'actor_team':e['team'],'event_visibility':e['visibility'],'players':players,'ball':{'x':None,'y':None,'z':None,'vx':None,'vy':None,'vz':None},'action':{'label':e['label'],'target_player_id':None,'target_x':None,'target_y':None},'outcome':{'success':None,'future_value':None,'shot_within_5s':None,'xg_within_5s':None}}
            out.write(json.dumps(row,ensure_ascii=False)+'\n'); written+=1
    state_match_rate=state_matches/len(events) if events else 0.0
    actor_presence_rate=(state_matches-actor_missing)/state_matches if state_matches else 0.0
    print(f'BAS event count:                         {len(events)}')
    print(f'records written:                         {written}')
    print(f'missing GSR state count:                 {missing}')
    print(f'actor-missing count:                     {actor_missing}')
    print(f'state-match rate:                        {state_match_rate:.2%}')
    print(f'actor-presence rate:                     {actor_presence_rate:.2%}')
    print(f'skipped because actor missing:           {skipped_actor_missing}')
    print(f'GSR frame offsets over state-matched events (including actor-missing) (-1, 0, +1): -1={offset_counts[-1]}, 0={offset_counts[0]}, +1={offset_counts[1]}')
if __name__=='__main__': main()
