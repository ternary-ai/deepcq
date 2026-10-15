from __future__ import annotations
from dataclasses import dataclass


@dataclass
class OptimizationMetrics:
    original_chars: int
    filtered_chars: int
    reduction_pct: float


@dataclass
class QueryResult:
    content: str
    intent: dict[str, float]
    sections_included: list[str]
    sections_excluded: list[str]
    content_pattern: str
    metrics: OptimizationMetrics
