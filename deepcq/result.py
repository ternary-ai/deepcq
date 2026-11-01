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
    # False when no tier contains the section at all (score is then 0.0).
    present: bool = True


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
    # Schema sections no memory tier contains (not scored, not in excluded).
    sections_absent: list[str] = field(default_factory=list)
    # True when fallback_include_all returned everything unfiltered; the
    # reason says why. Check this before trusting reduction numbers.
    fallback_used: bool = False
    fallback_reason: str | None = None
