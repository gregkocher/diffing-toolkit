"""Transport compatibility without paid API calls or model imports."""
import importlib.util
from pathlib import Path

p = Path(__file__).parents[1] / "src/diffing/utils/chat_completion_params.py"
spec = importlib.util.spec_from_file_location("chat_params_test", p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_direct_gpt5_accepts_reasoning_parameters_without_mutating_messages():
    messages = [{"role": "system", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}]
    params = module.chat_completion_params(model="gpt-5-mini", base_url="https://api.openai.com/v1", messages=messages, max_tokens=10000, temperature=.7)
    assert params["max_completion_tokens"] == 10000
    assert "temperature" not in params and "max_tokens" not in params
    assert "cache_control" not in params["messages"][0]["content"][0]
    assert "cache_control" in messages[0]["content"][0]


def test_openrouter_wire_format_is_unchanged():
    params = module.chat_completion_params(model="openai/gpt-5", base_url="https://openrouter.ai/api/v1", messages=[], max_tokens=100, temperature=.7)
    assert params == {"model": "openai/gpt-5", "messages": [], "max_tokens": 100, "temperature": .7}


def test_other_openai_models_keep_temperature():
    params = module.chat_completion_params(model="gpt-4.1", base_url="https://api.openai.com/v1", messages=[], max_tokens=100, temperature=.7)
    assert params["temperature"] == .7 and params["max_tokens"] == 100
