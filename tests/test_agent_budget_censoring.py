"""Exercise native agent budget control with deterministic provider responses."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, List, Callable
from omegaconf import OmegaConf, SCMode

ROOT = Path(__file__).parents[1]

def run_native(responses, budget=10):
    tree=ast.parse((ROOT/'src/diffing/utils/agents/base_agent.py').read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseAgent')
    run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    limits=[]; snapshots=[]
    class LLM:
        def __init__(self,**kwargs):pass
        def chat(self,messages,max_completion_tokens=None):
            limits.append(max_completion_tokens)
            return responses.pop(0)
    ns=dict(Any=Any,Dict=Dict,List=List,Callable=Callable,AgentLLM=LLM,
            logger=NS(info=lambda *x:None,debug=lambda *x:None),POST_TOOL_RESULT_PROMPT='')
    exec(compile(ast.Module(body=functions+[run],type_ignores=[]),'<native agent>','exec'),ns)
    cfg=OmegaConf.create({'diffing':{'evaluation':{'agent':{'hints':'','llm':{'model_id':'gpt-5','base_url':'https://api.openai.com/v1','api_key_path':'unused','temperature':1.,'max_tokens_per_call':50},'budgets':{'agent_llm_calls':3,'token_budget_generated':budget}}}}})
    obj=NS(cfg=cfg,get_system_prompt=lambda n:'system',build_first_user_message=lambda x:'signals',get_tools=lambda x:{})
    result=ns['run'](obj,None,10,True,lambda s:snapshots.append(json.loads(json.dumps(s))))
    return result,limits,snapshots

def response(text,tokens):return {'content':text,'usage':{'completion_tokens':tokens,'prompt_tokens':2,'total_tokens':tokens+2},'finish_reason':'length'}

def test_remaining_cap_and_censored_raw_transcript_preserved():
    (desc,stats),limits,snapshots=run_native([response('malformed',6),response('',4)])
    assert limits==[10,4]
    assert desc is None and stats['status']=='budget_exhausted'
    assert stats['agent_completion_tokens']==10
    assert len(stats['responses'])==2
    assert snapshots[-1]['messages'][-1]['content']==''
    assert snapshots[-1]['status']=='budget_exhausted'

def test_valid_final_at_exact_budget_is_completed():
    (desc,stats),limits,_=run_native([response('FINAL(description: "observed behavior")',10)])
    assert desc=='observed behavior' and stats['status']=='completed_final'
    assert limits==[10]

def test_provider_overshoot_is_preserved_without_fabricating_description():
    (desc,stats),_,snapshots=run_native([response('raw provider text',11)])
    assert desc is None and stats['status']=='provider_budget_violation'
    assert snapshots[-1]['responses'][0]['content']=='raw provider text'

def test_pipeline_skips_grading_censored_run(tmp_path):
    tree=ast.parse((ROOT/'src/diffing/pipeline/evaluation_pipeline.py').read_text())
    save=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_description')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EvaluationPipeline')
    run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_agent')
    import hashlib
    from typing import Tuple
    ns=dict(OmegaConf=OmegaConf,SCMode=SCMode,Path=Path,Tuple=Tuple,BaseAgent=Any,hashlib=hashlib,json=json,logger=NS(info=lambda *x:None))
    exec(compile(ast.Module(body=[save,run],type_ignores=[]),'<native pipeline>','exec'),ns)
    called=[]
    class Agent:
        def get_dataset_mapping(self): return {'ds1': 'corpus.jsonl'}
        def run(self,*args,**kwargs):
            called.append(1)
            stats={'messages':[{'role':'assistant','content':'partial'}],'status':'budget_exhausted','agent_completion_tokens':10}
            kwargs['progress_callback'](stats)
            return None,stats
    obj=NS(diffing_method=NS(cfg=OmegaConf.create({'budget':10}),agent_cfg_hash='hash',get_or_create_results_dir=lambda:tmp_path))
    result=ns['run_agent'](obj,Agent(),10,0,False,'test',hints='',grader_num_repeat=3)
    assert result==(None,'budget_exhausted',None)
    assert not list(tmp_path.rglob('description.txt'))
    assert json.loads(next(tmp_path.rglob('config.json')).read_text())['budget']==10
    assert json.loads(next(tmp_path.rglob('dataset_mapping.json')).read_text())=={'ds1':'corpus.jsonl'}
    assert json.loads(next(tmp_path.rglob('messages.json')).read_text())[0]['content']=='partial'
    ns['run_agent'](obj,Agent(),10,0,False,'test',hints='',grader_num_repeat=3)
    assert len(called)==1
