from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ScoreExplanation:
    """Why a section scored the way it did — for debug/trust UIs."""
    score: float
    matched_tokens: list[str]
    matched_anchors: list[str]
    literal_hit: bool
    matched_phrases: list[str] = field(default_factory=list)
    # Source -> share of the token score it supplied (each matched token is
    # credited to the highest-weighted place it was found). Sources are tier
    # names plus "schema" for the section's own name/keywords.
    tier_contributions: dict[str, float] = field(default_factory=dict)


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
    explanations: dict[str, ScoreExplanation] = field(default_factory=dict)
