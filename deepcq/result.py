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
    # Sectioned tiers (state / notes / reference) holding more than one
    # line-start copy of this section; only the first copy is used.
    duplicate_tiers: list[str] = field(default_factory=list)


@dataclass
class SectionInfo:
    """One schema section's outcome for a query, enough to render the
    section map without the schema."""
    name: str
    status: str              # "included" | "excluded" | "over_budget" | "absent"
    score: float
    chars: int               # characters it adds to content when included
                             # (its pieces across sectioned tiers, joined); 0 when absent
    description: str | None = None


@dataclass
class MemoryGap:
    """Something the question asked for that memory doesn't hold."""
    kind: str                                          # "absent_section" | "unmatched_anchor" | "no_match"
    sections: list[str] = field(default_factory=list)  # absent_section: the sections asked for
    anchors: list[str] = field(default_factory=list)   # unmatched_anchor: figures/identifiers found nowhere


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
    # Per-section outcome, in schema order (see section_map()).
    sections: dict[str, SectionInfo] = field(default_factory=dict)
    # Passed the threshold but cut by the context budget (max_chars).
    sections_over_budget: list[str] = field(default_factory=list)
    # Free-text tiers ("cache", "history") that matched but were cut by the budget.
    tiers_over_budget: list[str] = field(default_factory=list)
    # The budget this result was built under; None = no budget.
    max_chars: int | None = None
    # Section -> sectioned tiers holding more than one copy of it (only the
    # first copy is used), whether or not the section was included.
    duplicate_sections: dict[str, list[str]] = field(default_factory=dict)
    # What the question asked for that memory doesn't hold (see gap_note()).
    gaps: list[MemoryGap] = field(default_factory=list)
    # Included below threshold because MemorySchema.pinned lists them.
    sections_pinned: list[str] = field(default_factory=list)
    # Section headers found in the state tier whose name isn't in the schema —
    # query() can never return their content.
    unknown_sections: list[str] = field(default_factory=list)

    def omitted_note(self, reload_hint: str | None = None) -> str:
        """
        The sections left out of `content` (below threshold or over budget),
        as section_map(only_omitted=True) lists them, so an LLM reading
        `content` doesn't conclude that data doesn't exist. `reload_hint` is
        appended on its own line, e.g. 'Load one with get_section(name).'
        "" when nothing was omitted.
        """
        note = self.section_map(only_omitted=True)
        if not note:
            return ""
        return f"{note}\n{reload_hint}" if reload_hint else note

    def section_map(self, only_omitted: bool = False) -> str:
        """
        A compact index of the present sections — name, status, size, and
        description when declared — so a model can see what exists without
        it being loaded. With `only_omitted`, only the sections left out
        (below threshold or over budget). "" when there is nothing to list.
        Not part of `content`, and not counted against `max_chars`.
        """
        lines: list[str] = []
        for info in self.sections.values():
            if info.status == "absent":
                continue
            if only_omitted and info.status == "included":
                continue
            label = "included" if info.status == "included" else f"{info.chars:,} chars, not loaded"
            line = f"- {info.name} ({label})"
            if info.description:
                line += f": {info.description}"
            lines.append(line)
        if not lines:
            return ""
        header = "Not loaded (ask for any of these by name):" if only_omitted else "Memory sections:"
        return "\n".join([header, *lines])

    def gap_note(self) -> str:
        """
        One short paragraph stating what memory doesn't hold — sections the
        question asked for, figures/identifiers found nowhere, or no match
        at all — so the agent doesn't mistake missing data for filtered
        data. Facts only. "" when there are no gaps.
        """
        sentences: list[str] = []
        for gap in self.gaps:
            if gap.kind == "absent_section":
                names = []
                for name in gap.sections:
                    info = self.sections.get(name)
                    desc = info.description if info else None
                    names.append(f"{name} ({desc})" if desc else name)
                sentences.append(f"Not in memory: {', '.join(names)}.")
            elif gap.kind == "unmatched_anchor":
                quoted = ", ".join(f'"{a}"' for a in gap.anchors)
                sentences.append(f"No stored memory mentions {quoted}.")
            elif gap.kind == "no_match":
                sentences.append(
                    "Nothing in memory matched this question; the memory shown is unfiltered."
                    if self.fallback_used else "Nothing in memory matched this question."
                )
        return " ".join(sentences)
