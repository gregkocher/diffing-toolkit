"""Analytic check that reference fitting differentiates through recurrence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from jlens.fitting import jacobian_for_prompt

spec = importlib.util.spec_from_file_location("ouro_lens_model", Path(__file__).with_name("model.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Block(nn.Module):
    def forward(self, x, *, current_ut):
        return x * (current_ut + 2)

class Text(nn.Module):
    total_ut_steps = 4
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(2, 2)
        self.layers = nn.ModuleList([Block()])
        self.norm = nn.Identity()
    def forward(self, input_ids, use_cache=False):
        x = self.embed_tokens(input_ids)
        for r in range(4):
            x = self.layers[0](x, current_ut=r)
            x = x * 0.5  # Represents a nontrivial inter-recurrence transform.
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Text()
        self.config = SimpleNamespace(hidden_size=2)
        self.lm_head = nn.Identity()

class Tokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.zeros((1, 5), dtype=torch.long))


def test_native_and_full_downstream_jacobian():
    hf = Model()
    ids = torch.zeros((1, 5), dtype=torch.long)
    expected = hf.model(ids)
    adapter = module.OuroLensModel(hf, Tokenizer())
    assert torch.equal(expected, adapter.forward(ids))
    matrices, _, _ = jacobian_for_prompt(adapter, "test", [0, 1, 2], dim_batch=2, skip_first=0)
    # Pre-norm boundary -> pre-norm final boundary: .5*3*.5*4*.5*5 etc.
    for r, factor in enumerate((7.5, 5.0, 2.5)):
        torch.testing.assert_close(matrices[r], torch.eye(2) * factor)
    assert adapter.visits == [0, 1, 2, 3]
    adapter.close()
    assert not hf.model.layers[-1]._forward_hooks


def test_reference_merge_weights_counts_not_half_average():
    from jlens import JacobianLens
    first=JacobianLens({0:torch.eye(2)},n_prompts=31,d_model=2)
    last=JacobianLens({0:torch.eye(2)*33},n_prompts=1,d_model=2)
    merged=JacobianLens.merge([first,last])
    assert merged.n_prompts==32
    torch.testing.assert_close(merged.jacobians[0],torch.eye(2)*2)
