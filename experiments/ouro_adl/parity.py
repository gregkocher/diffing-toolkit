"""Identify the recurrent invocation read by the unchanged ADL extractor.

This validation uses independent native hooks solely as a reference. Production
ADL inference and statistics remain in ActivationDifferenceLensMethod.
"""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from diffing.utils.model import load_model, gc_collect_cuda_cache, _MODEL_CACHE, logit_lens
from diffing.methods.activation_difference_lens.method import extract_first_n_tokens_activations

out = Path('/workspace/methods_v2/adl_v2')
out.mkdir(exist_ok=True)
paths = json.loads(Path('/workspace/standard_v1/model_paths.json').read_text())
records = []
for arm in ('base', 'target'):
    model = load_model(paths['base'], torch.bfloat16, 'sdpa',
        adapter_ids='ouro_target_adapter' if arm == 'target' else None,
        trust_remote_code=True, ignore_cache=True, subfolder='', adapter_backend='peft')
    model.eval()
    seq = model.tokenizer.encode('The history of the town began with a small settlement. Its residents built roads.', add_special_tokens=True)[:16]
    layers = [0, 11, 23]
    actual = extract_first_n_tokens_activations(model, [seq], layers, batch_size=1)
    native = AutoModelForCausalLM.from_pretrained(paths['base'], trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    if arm == 'target':
        native = PeftModel.from_pretrained(native, paths['adapter']).eval()
    underlying = native.get_base_model() if arm == 'target' else native
    captured = {layer: [] for layer in layers}
    handles = []
    for layer in layers:
        def capture(module, args, result, layer=layer):
            hidden = result[0] if isinstance(result, tuple) else result
            captured[layer].append(hidden.detach().cpu())
        handles.append(underlying.model.layers[layer].register_forward_hook(capture))
    with torch.no_grad():
        native(torch.tensor([seq], device='cuda'))
    for handle in handles:
        handle.remove()
    for layer in layers:
        errors = [(actual[layer] - expected).abs().max().item() for expected in captured[layer]]
        matches = [r + 1 for r, error in enumerate(errors) if error == 0]
        record = {'arm': arm, 'layer': layer, 'native_pass_count': len(errors),
                  'max_errors_by_recurrence': errors, 'exact_matching_recurrences': matches}
        records.append(record)
        print(record, flush=True)
        assert len(errors) == 4 and len(matches) == 1, record
    for recurrence_index in range(4):
        selected = extract_first_n_tokens_activations(model, [seq], layers, batch_size=1, recurrence_index=recurrence_index)
        for layer in layers:
            error = (selected[layer] - captured[layer][recurrence_index]).abs().max().item()
            record = {"arm": arm, "layer": layer, "selected_recurrence": recurrence_index + 1, "max_abs_error": error}
            records.append(record)
            print(record, flush=True)
            assert error == 0, record
    latent = actual[23][0, 0]
    with torch.no_grad():
        positive, negative = logit_lens(latent, model)
        normed = underlying.model.norm(latent.cuda().to(torch.bfloat16))
        expected = underlying.lm_head(normed).softmax(-1).cpu()
    assert torch.equal(positive, expected), 'Standard ADL lens differs from native projection'
    del model, native, underlying, actual, captured
    _MODEL_CACHE.clear()
    gc_collect_cuda_cache()
(out / 'recurrence_parity.json').write_text(json.dumps(records, indent=2) + '\n')
