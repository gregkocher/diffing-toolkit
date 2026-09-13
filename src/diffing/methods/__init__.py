"""Diffing methods, imported lazily to permit lightweight analysis runtimes."""
from importlib import import_module

_EXPORTS = {
    "KLDivergenceDiffingMethod": "kl",
    "ActivationAnalysisDiffingMethod": "activation_analysis",
    "CrosscoderDiffingMethod": "crosscoder",
    "SAEDifferenceMethod": "sae_difference",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
