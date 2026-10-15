from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AppState": ("deepcq.memory.state", "AppState"),
    "SessionCache": ("deepcq.memory.session", "SessionCache"),
    "History": ("deepcq.memory.session", "History"),
    "LongTermStore": ("deepcq.memory.store", "LongTermStore"),
    "ReferenceData": ("deepcq.memory.store", "ReferenceData"),
}

__all__ = ["AppState", "SessionCache", "History", "LongTermStore", "ReferenceData"]


def __getattr__(name: str):
	try:
		module_name, attribute_name = _EXPORTS[name]
	except KeyError:
		raise AttributeError(name) from None
	module = import_module(module_name)
	value = getattr(module, attribute_name)
	globals()[name] = value
	return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
