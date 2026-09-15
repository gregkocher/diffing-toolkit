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
