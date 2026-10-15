from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "PromptLibrary": ("deepcq.integrations.ternary.prompt_library", "PromptLibrary"),
    "ToolContextProvider": ("deepcq.integrations.ternary.adapters", "ToolContextProvider"),
    "LangChainMemoryAdapter": ("deepcq.integrations.ternary.adapters", "LangChainMemoryAdapter"),
}

__all__ = ["PromptLibrary", "ToolContextProvider", "LangChainMemoryAdapter"]


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
