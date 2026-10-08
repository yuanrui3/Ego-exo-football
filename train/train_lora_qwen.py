import argparse,json,torch
from datasets import Dataset
from peft import LoraConfig,get_peft_model
from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments

def read(path):
    with open(path,encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--val',required=True);ap.add_argument('--output',required=True);ap.add_argument('--model',default='Qwen/Qwen3-4B-Base');ap.add_argument('--max-length',type=int,default=2048);ap.add_argument('--epochs',type=float,default=3);ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--batch-size',type=int,default=1);ap.add_argument('--grad-accum',type=int,default=16);a=ap.parse_args()
tok=AutoTokenizer.from_pretrained(a.model,use_fast=True)
if tok.pad_token is None:tok.pad_token=tok.eos_token
dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype,device_map='auto' if torch.cuda.is_available() else None);model.config.use_cache=False
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
def enc(r):
    p=tok(r['prompt'],add_special_tokens=True,truncation=True,max_length=a.max_length)['input_ids'];q=tok(r['response']+tok.eos_token,add_special_tokens=False,truncation=True,max_length=max(1,a.max_length-len(p)))['input_ids'];ids=(p+q)[:a.max_length];return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':([-100]*len(p)+q)[:a.max_length]}
def ds(path):
    rows=read(path);return Dataset.from_list(rows).map(enc,remove_columns=['id','match_id','prompt','response']) if rows else None
class C:
    def __call__(self,fs):
        m=max(len(x['input_ids']) for x in fs);pad=lambda x,v:x+[v]*(m-len(x));return {'input_ids':torch.tensor([pad(x['input_ids'],tok.pad_token_id) for x in fs]),'attention_mask':torch.tensor([pad(x['attention_mask'],0) for x in fs]),'labels':torch.tensor([pad(x['labels'],-100) for x in fs])}
tr,va=ds(a.train),ds(a.val);args=TrainingArguments(output_dir=a.output,num_train_epochs=a.epochs,learning_rate=a.lr,per_device_train_batch_size=a.batch_size,per_device_eval_batch_size=a.batch_size,gradient_accumulation_steps=a.grad_accum,logging_steps=10,save_strategy='epoch',eval_strategy='epoch' if va is not None else 'no',report_to='none',bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),gradient_checkpointing=True,remove_unused_columns=False)
Trainer(model=model,args=args,train_dataset=tr,eval_dataset=va,data_collator=C()).train();model.save_pretrained(a.output);tok.save_pretrained(a.output)
