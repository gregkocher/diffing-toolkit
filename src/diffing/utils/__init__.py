"""Shared utilities, with optional model/tracing dependencies loaded on demand."""
from importlib import import_module

_EXPORTS = {
    "get_layer_indices": "activations",
    "ModelConfig": "configs",
    "DatasetConfig": "configs",
    "get_model_configurations": "configs",
    "get_dataset_configurations": "configs",
    "load_model": "model",
    "load_model_from_config": "model",
    "get_ft_model_id": "model",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
