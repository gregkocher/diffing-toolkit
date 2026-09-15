"""Provider-specific transport parameters; prompts and grading are unchanged."""
from copy import deepcopy
from urllib.parse import urlparse


def chat_completion_params(*, model, base_url, messages, max_tokens, temperature=None):
    params = {"model": model, "messages": deepcopy(messages)}
    direct_openai = urlparse(base_url).hostname == "api.openai.com"
    # The original GPT-5 reasoning endpoints do not accept custom temperature
    # or legacy max_tokens. Other providers retain their existing wire format.
    reasoning = model in {"gpt-5", "gpt-5-mini", "gpt-5-nano"} or any(
        model.startswith(prefix) for prefix in ("gpt-5-2025-", "gpt-5-mini-2025-", "gpt-5-nano-2025-")
    )
    params["max_completion_tokens" if direct_openai and reasoning else "max_tokens"] = max_tokens
    if temperature is not None and not (direct_openai and reasoning):
        params["temperature"] = temperature
    if direct_openai:
        for message in params["messages"]:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    part.pop("cache_control", None)
    return params
