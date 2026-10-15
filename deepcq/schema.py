"""MemorySchema — defines section names and optional keyword overrides."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .vocabulary import SynonymMap

# The five memory tiers, in the order DeepCQ.query() reads them.
TIER_NAMES: tuple[str, ...] = ("state", "cache", "history", "notes", "reference")


@dataclass
class MemorySchema:
    """
    Declares the logical sections that structure stored memory text.

    sections:
        List of section names, e.g. ["errors", "metrics", "status"].
        No keyword lists required — deepcq derives patterns from the
        query itself (query-term-first approach).

    keywords:
        Optional per-section extra terms that boost relevance scoring.
        Useful when section names are abbreviations or jargon.

    section_header_format:
        How sections are delimited in stored text.
        Default "[{name}]" means sections look like "[errors]", "[metrics]".

    confidence_threshold:
        Minimum score (0.0–1.0) for a section to be included.

    fallback_include_all:
        When True (default), return all content unfiltered if no section
        scores above the threshold. Prevents silent data loss.

    anchor_patterns:
        Optional override for the regex patterns used to extract literal
        anchors (see anchors.py) from queries against this schema. None
        (default) uses anchors.DEFAULT_ANCHOR_PATTERNS.

    vocabulary:
        Optional SynonymMap (see vocabulary.py). Each section's vocabulary
        is expanded with every synonym cluster whose term occurs in it, so
        "eps" in a query matches "earnings per share" in stored text and
        vice versa. None (default) = no expansion.

    phrase_sizes / phrase_boost:
        N-gram sizes for phrase-aware matching (see phrases.py). Each
        stemmed query phrase also found in a section adds `phrase_boost`
        to its score (bonus only, total capped at 1.0). () (default) = off.
        Turning this on shifts scores — recalibrate thresholds after.

    tier_weights / schema_weight:
        Per-tier weight for a token match, e.g. {"history": 0.5}. A query
        token counts at the highest weight among the places it was found
        (never summed across tiers); tiers missing from the dict weigh 1.0.
        `schema_weight` is the weight of a match on the section's own
        name/keywords. None / 1.0 (defaults) = current unweighted scoring.
    """

    sections: Sequence[str]
    keywords: dict[str, list[str]] = field(default_factory=dict)
    section_header_format: str = "[{name}]"
    confidence_threshold: float = 0.3
    fallback_include_all: bool = True
    anchor_patterns: Sequence[re.Pattern[str]] | None = None
    vocabulary: SynonymMap | None = None
    phrase_sizes: tuple[int, ...] = ()
    phrase_boost: float = 0.25
    tier_weights: dict[str, float] | None = None
    schema_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.tier_weights is not None:
            unknown = set(self.tier_weights) - set(TIER_NAMES)
            if unknown:
                raise ValueError(f"unknown tier(s) in tier_weights: {sorted(unknown)}; expected {TIER_NAMES}")
            if any(w < 0 for w in self.tier_weights.values()):
                raise ValueError("tier_weights must be non-negative")
        if self.schema_weight < 0:
            raise ValueError("schema_weight must be non-negative")
        if any(n < 2 for n in self.phrase_sizes):
            raise ValueError(f"phrase_sizes must all be >= 2, got {self.phrase_sizes}")
        if self.phrase_boost < 0:
            raise ValueError("phrase_boost must be non-negative")

        name_idx = self.section_header_format.index("{name}")
        self._prefix = self.section_header_format[:name_idx]

        escaped_prefix = re.escape(self._prefix)
        lookahead = rf"(?=\n{escaped_prefix}|\Z)"

        self._extraction_patterns: dict[str, re.Pattern[str]] = {}
        for section in self.sections:
            header = self.section_header_format.format(name=section)
            pat = re.compile(
                rf"({re.escape(header)}.*?){lookahead}",
                re.DOTALL | re.IGNORECASE,
            )
            self._extraction_patterns[section] = pat

        self._id = hash((tuple(self.sections), self.section_header_format))

    def extraction_pattern(self, section: str) -> re.Pattern[str]:
        return self._extraction_patterns[section]

    def schema_id(self) -> int:
        return self._id
