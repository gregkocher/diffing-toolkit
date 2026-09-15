"""Regression: a lazily loaded model must not erase lens values on meta."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

# Load the actual device-resolution method without importing the full optional
# GPU-serving stack. No implementation is duplicated in this CPU regression.
source=Path(__file__).parents[2]/'src/diffing/methods/diff_mining/logit_extraction.py'
tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JLensExtractor')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_J_for')
namespace={'torch':torch,'StandardizedTransformer':object}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
resolve=namespace['_J_for']

class LazyModel:
    dtype=torch.float32
    def __init__(self,remain_meta=False):
        self.parameter=torch.nn.Parameter(torch.empty(2,device='meta'))
        self.remain_meta=remain_meta
        self.calls=0
    def dispatch(self):
        self.calls+=1
        if not self.remain_meta:self.parameter=torch.nn.Parameter(torch.ones(2))
    def named_parameters(self):return [('model.layers.23.weight',self.parameter)]
    def parameters(self):return iter([self.parameter])


def test_dispatch_preserves_nontrivial_lens_values():
    model=LazyModel()
    lens=SimpleNamespace(layer_idx=23,J=torch.tensor([[1.,2.],[3.,5.]]),_J_dev=None)
    result=resolve(lens,model)
    assert model.calls==1
    assert result.device.type=='cpu'
    torch.testing.assert_close(result,torch.tensor([[1.,2.],[3.,5.]]))


def test_reject_unmaterialized_model_instead_of_silent_meta_copy():
    lens=SimpleNamespace(layer_idx=23,J=torch.eye(2),_J_dev=None)
    with pytest.raises(ValueError,match='meta device'):
        resolve(lens,LazyModel(remain_meta=True))
    assert lens._J_dev is None
