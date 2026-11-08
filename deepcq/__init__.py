from __future__ import annotations

from importlib import import_module

__version__ = "0.4.0"

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
    "locate": ("deepcq.locate", "locate"),
    "Candidate": ("deepcq.locate", "Candidate"),
    "ScoreExplanation": ("deepcq.result", "ScoreExplanation"),
    "Recency": ("deepcq.decay", "Recency"),
    "no_decay": ("deepcq.decay", "no_decay"),
    "exponential_decay": ("deepcq.decay", "exponential_decay"),
    "linear_decay": ("deepcq.decay", "linear_decay"),
    "SynonymMap": ("deepcq.vocabulary", "SynonymMap"),
    "load_pack": ("deepcq.vocabulary", "load_pack"),
    "LabeledExample": ("deepcq.calibration", "LabeledExample"),
    "ThresholdStats": ("deepcq.calibration", "ThresholdStats"),
    "CalibrationReport": ("deepcq.calibration", "CalibrationReport"),
    "calibrate": ("deepcq.calibration", "calibrate"),
    "TIER_NAMES": ("deepcq.schema", "TIER_NAMES"),
    "Section": ("deepcq.sections", "Section"),
    "SectionSplit": ("deepcq.sections", "SectionSplit"),
    "split_sections": ("deepcq.sections", "split_sections"),
    "fallback_counts": ("deepcq.core", "fallback_counts"),
    "reset_fallback_counts": ("deepcq.core", "reset_fallback_counts"),
    "gap_counts": ("deepcq.core", "gap_counts"),
    "reset_gap_counts": ("deepcq.core", "reset_gap_counts"),
    "SectionInfo": ("deepcq.result", "SectionInfo"),
    "MemoryGap": ("deepcq.result", "MemoryGap"),
    "SectionUsage": ("deepcq.usage", "SectionUsage"),
    "UsageStats": ("deepcq.usage", "UsageStats"),
    "HealthReport": ("deepcq.usage", "HealthReport"),
}

__all__ = [
    "DeepCQ", "MemorySchema", "QueryResult", "OptimizationMetrics",
    "AppState", "SessionCache", "History", "LongTermStore", "ReferenceData",
    "locate", "Candidate", "ScoreExplanation",
    "Recency", "no_decay", "exponential_decay", "linear_decay",
    "SynonymMap", "load_pack",
    "LabeledExample", "ThresholdStats", "CalibrationReport", "calibrate",
    "TIER_NAMES",
    "Section", "SectionSplit", "split_sections",
    "fallback_counts", "reset_fallback_counts", "gap_counts", "reset_gap_counts",
    "SectionInfo", "MemoryGap", "SectionUsage", "UsageStats", "HealthReport",
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


# `locate` is both a submodule and an exported function. Importing the
# submodule (as `import deepcq.locate` or `from deepcq.locate import ...`
# does) binds the package attribute to the module, so the lazy export above
# would then return the module. Bind the function eagerly, after that import.
from .locate import locate  # noqa: E402
