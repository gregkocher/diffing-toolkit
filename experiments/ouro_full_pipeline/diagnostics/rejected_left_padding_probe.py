"""Actual Ouro greedy generation: left-padded batches versus individual prompts."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

p=argparse.ArgumentParser();p.add_argument('--model-paths',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
paths=json.loads(a.model_paths.read_text())
tok=AutoTokenizer.from_pretrained(paths['base'],trust_remote_code=True)
original_padding_side=tok.padding_side
tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(paths['base'],trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
mask_layout_copies = []
def contiguous_mask(module, args, kwargs):
    mask = kwargs.get('attention_mask')
    if mask is not None and (not mask.is_contiguous() or mask.stride(-1) != 1):
        if len(mask_layout_copies)<5: print('MASK_LAYOUT',list(mask.shape),list(mask.stride()),flush=True)
        mask_layout_copies.append({'shape': list(mask.shape), 'stride': list(mask.stride())})
        kwargs = {**kwargs, 'attention_mask': mask.clone(memory_format=torch.contiguous_format)}
    return args, kwargs
for module in model.modules():
    if type(module).__name__ == 'OuroAttention':
        module.register_forward_pre_hook(contiguous_mask, with_kwargs=True)
prompts=['The capital of France is','Answer briefly: what is the capital city of France?']
def generate(items,side):
    enc=tok(items,return_tensors='pt',padding=True,padding_side=side,add_special_tokens=True).to('cuda')
    with torch.inference_mode():
        out=model.generate(**enc,disable_compile=True,max_new_tokens=32,do_sample=False,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    rows=[]
    for row in out[:,enc.input_ids.shape[1]:].tolist():
        if tok.eos_token_id in row:row=row[:row.index(tok.eos_token_id)+1]
        rows.append(row)
    return rows
result={'base_id':paths['base_id'],'base_revision':paths['base_revision'],'prompts':prompts,'dtype':'bfloat16','max_new_tokens':32,'greedy':True,'records':{}}
for variant in ['base','finetuned']:
    if variant=='finetuned':model=PeftModel.from_pretrained(model,paths['adapter']).eval()
    single=[generate([prompt],'left')[0] for prompt in prompts]
    left=generate(prompts,'left');right=generate(prompts,'right')
    result['records'][variant]={'single_token_ids':single,'left_token_ids':left,'right_token_ids':right,'left_matches_individual':[x==y for x,y in zip(left,single)],'right_matches_individual':[x==y for x,y in zip(right,single)],'single_text':[tok.decode(x) for x in single],'left_text':[tok.decode(x) for x in left],'right_text':[tok.decode(x) for x in right]}
assert tok.padding_side==original_padding_side
result['mask_layout_copies']=mask_layout_copies
result['all_left_exact']=all(all(r['left_matches_individual']) for r in result['records'].values())
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['all_left_exact']
