"""Diffing module; heavy method dependencies are loaded on demand."""
from importlib import import_module

__all__ = ["methods", "evaluators"]


def __getattr__(name):
    if name in __all__:
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(name)
