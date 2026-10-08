import argparse,json
from collections import Counter
from sklearn.metrics import accuracy_score,f1_score
def read(p):
    with open(p,encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--val',required=True);a=ap.parse_args();tr=read(a.train);va=read(a.val);c=Counter(r['response'] for r in tr);maj=c.most_common(1)[0][0];yt=[r['response'] for r in va];yp=[maj]*len(va)
print('train label counts:',dict(c));print('majority class:',maj);print('val top-1 accuracy:',accuracy_score(yt,yp));print('val macro F1:',f1_score(yt,yp,average='macro',zero_division=0))
