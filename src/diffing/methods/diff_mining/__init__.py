"""Diff mining analysis; the pipeline class is loaded only when requested."""
__all__ = ["DiffMiningMethod"]


def __getattr__(name):
    if name == "DiffMiningMethod":
        from .diff_mining import DiffMiningMethod
        return DiffMiningMethod
    raise AttributeError(name)
