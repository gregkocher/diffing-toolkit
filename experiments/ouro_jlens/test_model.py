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


def test_early_window_matches_explicit_mask_with_causal_cross_token_terms(monkeypatch):
    """The reference mask controls both cotangents and source averaging.

    A causal cumsum block has real cross-token Jacobian terms, so this checks
    more than a tokenwise identity model. Explicit masks are test-only; the
    production fit uses the reference estimator's existing public arguments.
    """
    import jlens.fitting as fitting

    class CausalBlock(nn.Module):
        def forward(self, x, *, current_ut):
            return x + x.cumsum(dim=1) * 0.01 * (current_ut + 1)

    class LengthTokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.zeros(
                (1, kwargs.get('max_length', 64)), dtype=torch.long))

    hf = Model()
    hf.model.layers = nn.ModuleList([CausalBlock()])
    adapter = module.OuroLensModel(hf, LengthTokenizer())
    early, seq_len, count = fitting.jacobian_for_prompt(
        adapter, 'test', [0, 1, 2], dim_batch=2, max_seq_len=48, skip_first=0)
    assert (seq_len, count) == (48, 47)
    unrestricted, _, full_count = fitting.jacobian_for_prompt(
        adapter, 'test', [0, 1, 2], dim_batch=2, max_seq_len=64, skip_first=0)
    assert full_count == 63

    def explicit_early_mask(seq_len, *, skip_first):
        assert seq_len == 64
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[:47] = True
        return mask

    monkeypatch.setattr(fitting, 'valid_position_mask', explicit_early_mask)
    masked, seq_len, count = fitting.jacobian_for_prompt(
        adapter, 'test', [0, 1, 2], dim_batch=2, max_seq_len=64, skip_first=0)
    assert (seq_len, count) == (64, 47)
    for r in (0, 1, 2):
        torch.testing.assert_close(early[r], masked[r])
        assert not torch.allclose(early[r], unrestricted[r])
    adapter.close()
