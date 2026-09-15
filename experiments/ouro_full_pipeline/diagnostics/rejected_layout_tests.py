import importlib.util
from pathlib import Path
import torch

p=Path(__file__).parents[1]/'src/diffing/utils/ouro_attention_layout.py'
spec=importlib.util.spec_from_file_location('ouro_layout',p);layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)

def test_mask_layout_preserves_values_shape_dtype_and_gradients():
    original=torch.arange(6.,requires_grad=True)
    mask=original.reshape(2,3).T
    assert not mask.is_contiguous()
    args,kwargs=layout.contiguous_attention_mask(None,(),{'attention_mask':mask})
    fixed=kwargs['attention_mask']
    assert fixed.is_contiguous() and torch.equal(fixed,mask)
    assert fixed.shape==mask.shape and fixed.dtype==mask.dtype
    fixed.sum().backward();assert torch.equal(original.grad,torch.ones_like(original))
    args,_=layout.contiguous_attention_mask(None,(None,None,mask),{})
    assert torch.equal(args[2],mask) and args[2].is_contiguous()

def test_contiguous_and_absent_masks_are_unchanged():
    mask=torch.ones(2,3);kwargs={'attention_mask':mask}
    assert layout.contiguous_attention_mask(None,(),kwargs)[1] is kwargs
    assert layout.contiguous_attention_mask(None,(),{'attention_mask':None})[1]['attention_mask'] is None

def test_shared_attention_hook_installed_once():
    class OuroAttention(torch.nn.Module):
        def forward(self,hidden_states=None,position_embeddings=None,attention_mask=None):
            return attention_mask.is_contiguous()
    model=torch.nn.Sequential(OuroAttention())
    assert layout.install_ouro_attention_layout(model)==1
    assert layout.install_ouro_attention_layout(model)==0
    assert model[0](attention_mask=torch.ones(2,3).T)
