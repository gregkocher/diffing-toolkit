"""Verify recurrent J-lens extraction against separately loaded native models."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from diffing.utils.model import load_model, gc_collect_cuda_cache, _MODEL_CACHE
from diffing.methods.diff_mining.logit_extraction import JLensExtractor

p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
paths=json.loads(Path('/workspace/standard_v1/model_paths.json').read_text())
n=json.loads((a.directory/'COMPLETE.json').read_text())['n_prompts']
records=[]
for arm in ('base','target'):
    wrapped=load_model(paths['base'],torch.bfloat16,'sdpa',adapter_ids='ouro_target_adapter' if arm=='target' else None,
        trust_remote_code=True,ignore_cache=True,subfolder='',adapter_backend='peft')
    wrapped.eval()
    batch=wrapped.tokenizer(['The old city contained many streets and small shops.',
        'A short story.'],padding=True,return_tensors='pt')
    inputs={k:v.cuda() for k,v in batch.items() if k in ('input_ids','attention_mask')}
    native=AutoModelForCausalLM.from_pretrained(paths['base'],trust_remote_code=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    if arm=='target': native=PeftModel.from_pretrained(native,paths['adapter']).eval()
    underlying=native.get_base_model() if arm=='target' else native
    activations=[];visits=[]
    def record(module,args,kwargs,output):
        visits.append(int(kwargs['current_ut']));activations.append(output.detach())
    handle=underlying.model.layers[-1].register_forward_hook(record,with_kwargs=True)
    with torch.no_grad(): native_logits=native(**inputs).logits
    handle.remove()
    for r in (0,1,2):
        lens_path=a.directory/f'loop{r+1}_n{n}.pt'
        extractor=JLensExtractor(layer_idx=23,lens_path=str(lens_path),recurrence_idx=r)
        J=extractor.J.to(device='cuda',dtype=torch.bfloat16)
        with torch.no_grad():
            expected=underlying.lm_head(underlying.model.norm(activations[r]@J.T))
            actual=extractor.extract_logits(wrapped,**inputs)
        print({'arm':arm,'r':r,'hidden_max_abs':float((extractor.last_hidden-activations[r]).abs().max()),'readout_max_abs':float((expected-actual).abs().max()),'readout_mean_abs':float((expected-actual).abs().float().mean()),'terminal_max_abs':float((extractor.last_native_logits-native_logits).abs().max()),'actual_dtype':str(actual.dtype),'expected_dtype':str(expected.dtype)},flush=True)
        assert extractor._J_dev.device.type != 'meta'
        assert torch.equal(extractor._J_dev,J), 'Materialized lens matrix mismatch'
        assert torch.equal(extractor.last_hidden,activations[r]), 'Recurrent hidden mismatch'
        assert torch.equal(extractor.last_native_logits, native_logits), 'Terminal native output mismatch'
        assert torch.equal(expected,actual),f'{arm} recurrence {r}: toolkit/native mismatch'
        records.append({'arm':arm,'recurrence_idx':r,'exact':True,'max_abs_error':float((actual-expected).abs().max()),'native_visits':visits.copy(),'terminal_native_logits_exact':True})
    with torch.no_grad():
        unchanged=native(**inputs).logits
    assert torch.equal(unchanged,native_logits)
    del wrapped,native,underlying,activations,actual,expected,unchanged,native_logits
    _MODEL_CACHE.clear();gc_collect_cuda_cache()
(a.directory/'extractor_parity.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
