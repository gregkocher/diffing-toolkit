"""Regression for native generation with unequal-length padded prompts."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace


def load_decoder(logs):
    source = Path(__file__).parents[1] / "src/diffing/methods/diffing_method.py"
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "decode_generated_continuations")
    scope = {"json": json, "logger": SimpleNamespace(info=lambda fmt, value: logs.append(json.loads(value)))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[node.name]


class Row(list):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Row(value) if isinstance(key, slice) else value
    def tolist(self):
        return list(self)


def test_padding_is_not_returned_as_prompt_text():
    logs = []
    tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda ids, **kw: " ".join(str(i) for i in ids if i != 99))
    # Both padded input widths are 4, but first true input length is only 2.
    outputs = [Row([99, 99, 7, 8, 21, 99]), Row([3, 4, 5, 6, 22, 23])]
    assert load_decoder(logs)(outputs, 4, tokenizer, 2, "base") == ["21", "22 23"]
    assert logs[0]["observed_stop"] == "eos_token"
    assert logs[1]["observed_stop"] == "max_new_tokens"
    assert logs[0]["generated_tokens_including_eos"] == 2


def test_single_unpadded_prompt_keeps_same_decode():
    logs = []
    tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda ids, **kw: str(ids))
    assert load_decoder(logs)([Row([1, 2, 3, 4])], 2, tokenizer, 10, "finetuned") == ["[3, 4]"]
    assert logs[0]["observed_stop"] == "other"


def test_native_generation_left_padding_is_per_call_not_tokenizer_mutation():
    source = Path(__file__).parents[1]/"src/diffing/methods/diffing_method.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DiffingMethod")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate_texts")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "tokenizer"]
    assert len(calls) == 1
    assert next(k.value.value for k in calls[0].keywords if k.arg == "padding_side") == "left"
    assert not any(isinstance(n, ast.Attribute) and n.attr == "padding_side" for n in ast.walk(fn))


def test_native_microbatches_preserve_prompt_order_and_generation_options():
    from typing import List
    source=Path(__file__).parents[1]/'src/diffing/methods/diffing_method.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DiffingMethod')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate_texts')
    fn.decorator_list=[]
    scope={'List':List}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
    calls=[]
    def leaf(prompts,**kwargs):
        calls.append((prompts,kwargs)); return [f'answer:{p}' for p in prompts]
    method=SimpleNamespace(generate_texts=leaf)
    result=scope['generate_texts'](method,['short','a much longer prompt','third'],model_type='finetuned',max_new_tokens=1024,temperature=.8,do_sample=True,return_only_generation=True,native_batch_size=1)
    assert result==['answer:short','answer:a much longer prompt','answer:third']
    assert [x[0] for x in calls]==[['short'],['a much longer prompt'],['third']]
    assert all(x[1]==dict(model_type='finetuned',max_new_tokens=1024,temperature=.8,do_sample=True,return_only_generation=True,use_vllm=False,native_batch_size=1) for x in calls)
