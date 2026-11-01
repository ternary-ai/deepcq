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
    # For an absent section: the score its name/keywords alone gave the query
    # (what it would have scored on the schema). 0.0 for present sections.
    absent_score: float = 0.0


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
    # Included below threshold because MemorySchema.pinned lists them.
    sections_pinned: list[str] = field(default_factory=list)
    # Schema sections no memory tier contains (not scored, not in excluded).
    sections_absent: list[str] = field(default_factory=list)
    # Section headers found in the state tier whose name isn't in the schema —
    # query() can never return their content.
    unknown_sections: list[str] = field(default_factory=list)
    # True when fallback_include_all returned everything unfiltered; the
    # reason says why. Check this before trusting reduction numbers.
    fallback_used: bool = False
    fallback_reason: str | None = None

    def omitted_note(self, reload_hint: str | None = None) -> str:
        """
        One line naming the sections that were present but left out, so an
        LLM reading `content` doesn't conclude that data doesn't exist.
        `reload_hint` is appended, e.g. 'Load one with get_section(name).'
        Empty string when nothing was omitted.
        """
        if not self.sections_excluded:
            return ""
        note = f"Omitted as not relevant to this query: {', '.join(self.sections_excluded)}."
        return f"{note} {reload_hint}" if reload_hint else note
