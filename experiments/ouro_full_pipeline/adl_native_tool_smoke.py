from pathlib import Path
from types import SimpleNamespace
import json
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from diffing.methods.activation_difference_lens.agents import ADLAgent
from diffing.methods.activation_difference_lens.agent_tools import get_overview
root=Path('/workspace/full_audit_20260915')
cfg=OmegaConf.create({'diffing':{'method':{'logit_lens':{'cache':True},'auto_patch_scope':{'enabled':False},'steering':{'enabled':False},'agent':{'drilldown':{'max_sample_chars':100},'generate_steered':{'max_new_tokens':30,'temperature':1.,'do_sample':True}}}}})
tokenizer=AutoTokenizer.from_pretrained('/workspace/hf/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1',trust_remote_code=True)
method=SimpleNamespace(cfg=cfg,results_dir=root/'cache/adl/diffing_results/ouro_1_4B/ouro_cake_eos1221/activation_difference_lens/recurrence_4',tokenizer=tokenizer)
overview,mapping=get_overview(method,OmegaConf.create({'datasets':['corpus.jsonl'],'layers':[23],'positions':[0,1,2,3,4],'top_k_tokens':20}))
a=object.__new__(ADLAgent);a.cfg=cfg;a._dataset_mapping=mapping
tools=a.get_method_tools(method)
assert set(tools)=={'get_logitlens_details'}
r=tools['get_logitlens_details']('ds1',23,[0,1,2,3,4],20)
assert r['dataset']=='ds1' and 'corpus.jsonl' not in json.dumps(r)
for p in r['positions'].values(): assert len(p['tokens'])==20
cert={'passed':True,'tool':'get_logitlens_details','dataset_alias':'ds1','positions':[0,1,2,3,4],'k':20,'enabled_tools':list(tools),'result':r,'overview':overview}
(root/'ADL_FIXED_NATIVE_TOOL_CERTIFICATE.json').write_text(json.dumps(cert,indent=2));print('ADL_FIXED_NATIVE_TOOL_SMOKE_PASSED')
