import argparse,json
from collections import Counter
ap=argparse.ArgumentParser();ap.add_argument('--decisions',required=True);a=ap.parse_args();c=Counter();m=set();n=[]
with open(a.decisions,encoding='utf-8') as f:
    for l in f:
        if l.strip():
            r=json.loads(l);c[r['action']['label']]+=1;m.add(str(r['match_id']));n.append(len(r.get('players') or []))
print('matches',len(m));print('decisions',sum(c.values()));print('mean_players',sum(n)/len(n) if n else 0)
for k,v in c.most_common():print(f'{k:28s} {v}')
