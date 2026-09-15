"""Value-preserving mask layout compatibility for Ouro's native SDPA calls."""

def contiguous_attention_mask(module, args, kwargs):
    """Materialize expanded masks; preserve values, shape, dtype, and autograd."""
    if 'attention_mask' in kwargs:
        mask = kwargs['attention_mask']
        if mask is not None and not mask.is_contiguous():
            kwargs = {**kwargs, 'attention_mask': mask.contiguous()}
    elif len(args) > 2:
        mask = args[2]
        if mask is not None and not mask.is_contiguous():
            args = (*args[:2], mask.contiguous(), *args[3:])
    return args, kwargs


def install_ouro_attention_layout(model):
    """Install once on each shared attention module, covering all recurrence passes."""
    count = 0
    for module in model.modules():
        if type(module).__name__ != 'OuroAttention':
            continue
        if getattr(module, '_ouro_contiguous_attention_mask_v1', False):
            continue
        module.register_forward_pre_hook(contiguous_attention_mask, with_kwargs=True)
        module._ouro_contiguous_attention_mask_v1 = True
        count += 1
    return count
