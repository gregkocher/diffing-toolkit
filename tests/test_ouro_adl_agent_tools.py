"""ADL auditor routing and blinding regressions without loading an LLM."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Callable, Dict, List

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
ADL = ROOT / 'src/diffing/methods/activation_difference_lens'


def agent_namespace():
    class Blackbox:
        def get_tool_descriptions(self):
            return 'ask_model'
        def get_interaction_examples(self):
            return 'CALL(ask_model: {"prompts":["Hello"]})'
    class Diffing(Blackbox):
        def get_tool_descriptions(self):
            return super().get_tool_descriptions() + self.tool_descriptions
    tree = ast.parse((ADL/'agents.py').read_text())
    body = [x for x in tree.body if isinstance(x, ast.Assign) or
            (isinstance(x, ast.ClassDef) and x.name == 'ADLAgent')]
    ns = dict(Any=Any, Callable=Callable, Dict=Dict, List=List,
              BlackboxAgent=Blackbox, DiffingMethodAgent=Diffing)
    exec(compile(ast.Module(body=body, type_ignores=[]), '<ADLAgent>', 'exec'), ns)
    return ns


def config(patch=False, steering=False):
    return OmegaConf.create({'diffing': {'method': {
        'logit_lens': {'cache': True}, 'auto_patch_scope': {'enabled': patch},
        'steering': {'enabled': steering}, 'agent': {
            'drilldown': {'max_sample_chars': 100},
            'generate_steered': {'max_new_tokens': 30, 'temperature': 1., 'do_sample': True}}}}})


def test_aliases_are_resolved_for_all_tools_and_returned_payloads_stay_blind():
    ns = agent_namespace()
    seen = []
    def cache_reader(method, **kwargs):
        seen.append(kwargs['dataset'])
        return {'dataset': kwargs['dataset'], 'tokens': ['visible token']}
    for name in ('get_logitlens_details', 'get_patchscope_details', 'get_steering_samples'):
        ns[name] = cache_reader
    def steering_reader(method, **kwargs):
        seen.append(kwargs['dataset'])
        return ['generated text']
    ns['generate_steered'] = steering_reader
    agent = ns['ADLAgent'](); agent.cfg = config(True, True)
    hidden = '/private/organism_secret_training_data'
    agent._dataset_mapping = {'ds1': hidden}
    tools = agent.get_method_tools(NS())
    outputs = [tools['get_logitlens_details']('ds1', 1., [0], 10),
               tools['get_patchscope_details']('ds1', 1., [0], 10),
               tools['get_steering_samples']('ds1', 1., 0, None, 1),
               tools['generate_steered']('ds1', 1., 0, ['hello'], 1)]
    assert seen == [hidden]*4
    assert all(x['dataset'] == 'ds1' for x in outputs[:3])
    assert hidden not in json.dumps(outputs)
    with pytest.raises(ValueError, match='Unknown anonymous'):
        tools['get_logitlens_details'](hidden, 1., [0], 10)


def test_disabled_tools_are_neither_callable_nor_advertised():
    ns = agent_namespace(); agent = ns['ADLAgent'](); agent.cfg = config()
    assert set(agent.get_method_tools(NS())) == {'get_logitlens_details'}
    text = agent.get_tool_descriptions()
    assert 'get_logitlens_details' in text and 'ask_model' in text
    for name in ('get_patchscope_details', 'get_steering_samples', 'generate_steered'):
        assert name not in text
    overview = agent.get_first_user_message_description()
    assert 'available_positions' in overview
    assert '- Layers:' in overview and '- Positions:' in overview
    assert 'Downweight hubs and artifacts' in text
    assert 'Consider both frequency and effect size' in text
    assert 'YOU SHOULD PRIORITIZE THE ONE THAT IS MOST CONSISTENT WITH THE OVERVIEW' in agent.get_additional_conduct()


def test_disabled_signal_caches_are_not_exposed_in_overview(tmp_path):
    tree = ast.parse((ADL/'agent_tools.py').read_text())
    funcs = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'get_overview']
    layer = tmp_path/'layer_0'/'secret_dataset'; layer.mkdir(parents=True)
    (layer/'mean_pos_0.pt').touch()
    torch.save((torch.tensor([.5]), torch.tensor([1]), None, None), layer/'logit_lens_pos_0.pt')
    (layer/'auto_patch_scope_pos_0.pt').write_text('This disabled cache must never be read')
    steer = layer/'steering/position_0'; steer.mkdir(parents=True)
    (steer/'generations.jsonl').write_text('This disabled cache must never be read')
    ns = dict(Any=Any, Dict=Dict, List=List, torch=torch,
              logger=NS(info=lambda *x: None), _abs_layers_from_rel=lambda *x: [0],
              _load_ll_topk=lambda *x: (['token'], [.5]))
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<ADL overview>', 'exec'), ns)
    method = NS(cfg=config(), results_dir=tmp_path, tokenizer=None)
    overview, mapping = ns['get_overview'](method, OmegaConf.create({
        'datasets':['secret_dataset'], 'layers':[0.0], 'positions':[0]}))
    record = overview['datasets']['ds1']['layers'][0]
    assert record['available_positions'] == {'logit_lens':[0], 'patch_scope':[], 'steering':[]}
    assert 'secret_dataset' not in json.dumps(overview)
    assert mapping == {'ds1':'secret_dataset'}


def test_interface_version_changes_standard_agent_cache_hash():
    tree = ast.parse((ROOT/'src/diffing/methods/diffing_method.py').read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'DiffingMethod')
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'agent_cfg_hash')
    fn.decorator_list = []
    ns = dict(Any=Any, Dict=Dict, OmegaConf=OmegaConf, hashlib=hashlib)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<agent hash>', 'exec'), ns)
    obj = NS(method_cfg=OmegaConf.create({'agent':{'overview':{'positions':[0,1]}}}),
             extra_agent_relevant_cfg=lambda: {})
    old = ns['agent_cfg_hash'](obj)
    obj.method_cfg.agent.tool_interface_version = 'dataset_aliases_enabled_tools_v1'
    assert ns['agent_cfg_hash'](obj) != old
