import argparse,json,random,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from serialize_state import has_valid_anonymous_actor,serialize_state
SYSTEM="You are a football tactical decision model. Given the structured global pitch state, predict the actor's next ball action. Return exactly one action label and no explanation."
def read(p):
    with open(p,encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def write(p,rows):
    with open(p,'w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--train',required=True); ap.add_argument('--val',required=True); ap.add_argument('--val-frac',type=float,default=.2); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--serializer-mode',choices=['identity','anonymous'],default='anonymous'); a=ap.parse_args(); by=defaultdict(list)
    for r in read(a.input):by[str(r['match_id'])].append(r)
    ms=sorted(by); random.Random(a.seed).shuffle(ms); nv=max(1,round(len(ms)*a.val_frac)) if len(ms)>1 else 0; vm=set(ms[:nv]); tr=[];va=[]
    skipped_invalid_actor_state=0
    for m in ms:
        target=va if m in vm else tr
        for r in by[m]:
            if a.serializer_mode=='anonymous' and not has_valid_anonymous_actor(r):
                skipped_invalid_actor_state+=1
                continue
            target.append({'id':r['decision_id'],'match_id':r['match_id'],'prompt':f"<|system|>\n{SYSTEM}\n<|user|>\n{serialize_state(r,mode=a.serializer_mode)}\n<|assistant|>\n",'response':str(r['action']['label']).strip()})
    write(a.train,tr);write(a.val,va);print('train',len(tr),'val',len(va),'val_matches',sorted(vm));print('skipped invalid actor state',skipped_invalid_actor_state)
if __name__=='__main__':main()
