from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    "DeepCQ": ("deepcq.core", "DeepCQ"),
    "MemorySchema": ("deepcq.schema", "MemorySchema"),
    "QueryResult": ("deepcq.result", "QueryResult"),
    "OptimizationMetrics": ("deepcq.result", "OptimizationMetrics"),
    "AppState": ("deepcq.memory.state", "AppState"),
    "SessionCache": ("deepcq.memory.session", "SessionCache"),
    "History": ("deepcq.memory.session", "History"),
    "LongTermStore": ("deepcq.memory.store", "LongTermStore"),
    "ReferenceData": ("deepcq.memory.store", "ReferenceData"),
}

__all__ = [
    "DeepCQ", "MemorySchema", "QueryResult", "OptimizationMetrics",
    "AppState", "SessionCache", "History", "LongTermStore", "ReferenceData",
]


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from None
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
