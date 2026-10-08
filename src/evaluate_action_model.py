import argparse,json,torch
from sklearn.metrics import accuracy_score,f1_score,classification_report
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
LABELS=['Pass','Drive','Header','High Pass','Out','Cross','Throw In','Shot','Ball Player Block','Player Successful Tackle','Free Kick','Goal']
def read(p):
    with open(p,encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def norm(t):
    z=t.strip().lower()
    for l in sorted(LABELS,key=len,reverse=True):
        if l.lower() in z:return l
    return '<invalid>'
@torch.inference_mode()
def pred(m,t,p):
    b=t(p,return_tensors='pt',truncation=True,max_length=2048);b={k:v.to(m.device) for k,v in b.items()};o=m.generate(**b,max_new_tokens=8,do_sample=False,pad_token_id=t.pad_token_id,eos_token_id=t.eos_token_id);return t.decode(o[0,b['input_ids'].shape[1]:],skip_special_tokens=True).strip()
ap=argparse.ArgumentParser();ap.add_argument('--val',required=True);ap.add_argument('--adapter',required=True);ap.add_argument('--base-model',default='Qwen/Qwen3-4B-Base');ap.add_argument('--predictions',default='predictions.jsonl');a=ap.parse_args();tok=AutoTokenizer.from_pretrained(a.adapter);tok.pad_token=tok.pad_token or tok.eos_token;dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32);base=AutoModelForCausalLM.from_pretrained(a.base_model,torch_dtype=dtype,device_map='auto' if torch.cuda.is_available() else None);m=PeftModel.from_pretrained(base,a.adapter);m.eval();rows=read(a.val);yt=[];yp=[]
with open(a.predictions,'w',encoding='utf-8') as f:
    for i,r in enumerate(rows,1):
        raw=pred(m,tok,r['prompt']);p=norm(raw);yt.append(r['response']);yp.append(p);f.write(json.dumps({'id':r['id'],'true':r['response'],'pred':p,'raw':raw})+'\n')
        if i%50==0:print(i,'/',len(rows))
print('Top-1 accuracy:',accuracy_score(yt,yp));print('Macro F1:',f1_score(yt,yp,labels=LABELS,average='macro',zero_division=0));print(classification_report(yt,yp,labels=LABELS,zero_division=0));print('Invalid:',sum(x=='<invalid>' for x in yp),'/',len(yp))
