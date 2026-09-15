"""Verify toolkit single-prompt execution against ordinary native generation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace, MethodType
import torch
import anyio
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nnterp import StandardizedTransformer
from diffing.methods.diffing_method import DiffingMethod


def validate(model, tokenizer, paths, method_class=DiffingMethod):
    anyio.run(anyio.sleep,0)
    prompts=['The capital of France is', 'Answer briefly: what is the capital city of France?']
    records={}
    for variant in ['base','finetuned']:
        if variant=='finetuned':
            model=AutoModelForCausalLM.from_pretrained(paths['base'],trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
            model=PeftModel.from_pretrained(model,paths['adapter']).eval()
        native_ids=[]
        for prompt in prompts:
            enc=tokenizer([prompt],return_tensors='pt',padding=True,add_special_tokens=True).to('cuda')
            with torch.inference_mode():
                out=model.generate(**enc,max_new_tokens=32,do_sample=False,
                    disable_compile=True,pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            native_ids.append(out[0,enc.input_ids.shape[1]:].tolist())
        native_text=[tokenizer.decode(ids,skip_special_tokens=True) for ids in native_ids]
        wrapper=StandardizedTransformer(model,tokenizer=tokenizer)
        context=SimpleNamespace(tokenizer=tokenizer,base_model=wrapper,finetuned_model=wrapper,
            base_model_cfg=SimpleNamespace(disable_compile=True),finetuned_model_cfg=SimpleNamespace(disable_compile=True))
        context.generate_texts=MethodType(method_class.generate_texts,context)
        source_module=sys.modules[method_class.__module__]
        original_decoder=source_module.decode_generated_continuations
        toolkit_ids=[]
        def capture_ids(outputs,padded_input_width,*args,**kwargs):
            toolkit_ids.extend(row[padded_input_width:].tolist() for row in outputs)
            return original_decoder(outputs,padded_input_width,*args,**kwargs)
        source_module.decode_generated_continuations=capture_ids
        try:
            actual=context.generate_texts(prompts,model_type=variant,max_new_tokens=32,
                temperature=1.,do_sample=False,return_only_generation=True,native_batch_size=1)
        finally:
            source_module.decode_generated_continuations=original_decoder
        records[variant]={'native_token_ids':native_ids,'native_text':native_text,
            'toolkit_text':actual,'toolkit_token_ids':toolkit_ids,'exact_equal':actual==native_text and toolkit_ids==native_ids}
    result={'generation_protocol':'native_single_prompt_v1','native_batch_size':1,
        'max_new_tokens':32,'do_sample':False,'dtype':'bfloat16','attn_implementation':'sdpa',
        'disable_compile':True,'prompts':prompts,'records':records,
        'base_id':paths['base_id'],'base_revision':paths['base_revision'],
        'target_adapter_path':paths['adapter'],'target_revision':paths.get('adapter_event',{}).get('commit'),
        'target_repo':paths.get('adapter_event',{}).get('repo_id'),'target_subfolder':paths.get('adapter_event',{}).get('prefix'),
        'generation_source_sha256':hashlib.sha256(Path(sys.modules[method_class.__module__].__file__).read_bytes()).hexdigest(),
        'all_exact':all(x['exact_equal'] for x in records.values())}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-paths',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paths=json.loads(a.model_paths.read_text())
    tok=AutoTokenizer.from_pretrained(paths['base'],trust_remote_code=True);tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(paths['base'],trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
    result=validate(model,tok,paths)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    assert result['all_exact']
