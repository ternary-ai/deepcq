"""MemorySchema — defines section names and optional keyword overrides."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .vocabulary import SynonymMap

# The five memory tiers, in the order DeepCQ.query() reads them.
TIER_NAMES: tuple[str, ...] = ("state", "cache", "history", "notes", "reference")


# Longest allowed section description (one line, see MemorySchema).
MAX_DESCRIPTION_CHARS = 200
# Default context budget for query()/locate(), in characters: the same order
# as the 25 KB of MEMORY.md that Claude Code loads, about 6,000 English tokens.
DEFAULT_MAX_CHARS = 25_000


# Name of a header line that isn't a declared section: starts with a letter,
# no commas, quotes or brackets — "[deploys]" is a header, "[AAPL, MSFT]" and
# '["a", "b"]' are data.
_HEADER_NAME = r"""[^\W\d][^\n,"'\[\]{}<>]*?"""


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
        A header must start a line. A section runs up to the next header
        line: a declared section's header, or — for formats with text after
        {name}, like the default — a header-format line whose name starts
        with a letter and has no commas, quotes or brackets, followed by a
        space or the end of the line (so data lines like "[AAPL, MSFT]" or
        "[link](url)" don't end the section). For
        formats with nothing after {name} ("## {name}"), any line starting
        with the prefix is a header.

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

    descriptions / score_descriptions:
        Optional one-line description per section, saying what it holds
        ({"risk": "position limits and drawdown history"}). Shown in
        QueryResult.section_map() and gap_note(). With `score_descriptions`
        (default True) its words also join the section's scoring vocabulary
        at `schema_weight`, like the name and keywords — fixed schema text,
        so scoring stays deterministic. Empty (default) changes nothing.

    max_chars:
        Context budget for query(), in characters of returned content
        (separators included). Highest-scoring sections are kept first; a
        section that doesn't fit is skipped whole, never truncated, and
        reported in QueryResult.sections_over_budget. Default 25,000; None
        turns the budget off. query(max_chars=...) overrides it per call.
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
    descriptions: dict[str, str] = field(default_factory=dict)
    score_descriptions: bool = True
    max_chars: int | None = DEFAULT_MAX_CHARS

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
        unknown_desc = [name for name in self.descriptions if name not in self.sections]
        if unknown_desc:
            raise ValueError(f"descriptions for section(s) not in sections: {unknown_desc}")
        for name, text in self.descriptions.items():
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"description for {name!r} must be a non-empty string")
            if "\n" in text or "\r" in text:
                raise ValueError(f"description for {name!r} must be one line")
            if len(text) > MAX_DESCRIPTION_CHARS:
                raise ValueError(
                    f"description for {name!r} is {len(text)} characters; at most "
                    f"{MAX_DESCRIPTION_CHARS} allowed"
                )
        validate_max_chars(self.max_chars)

        name_idx = self.section_header_format.index("{name}")
        self._prefix = self.section_header_format[:name_idx]
        self._suffix = self.section_header_format[name_idx + len("{name}"):]

        def header_re(section: str, line_start: bool = False) -> str:
            header = self.section_header_format.format(name=section)
            escaped = re.escape(header)
            # A header ending in a letter/digit must end there: "## SEC" must
            # not match the header "## Sector".
            boundary = r"(?![A-Za-z0-9_])" if header[-1:].isalnum() else ""
            # Line start checked by a fixed-width lookbehind *after* the
            # literal, so the regex engine can still jump to the literal
            # (a leading "^" made locate() several times slower).
            start = rf"(?<![^\n]{escaped})" if line_start else ""
            return escaped + start + boundary

        # What ends a section: the next line that is a header (see the class
        # docstring) — not any line that merely starts with the prefix.
        # Declared names the generic rule can't recognise are listed
        # explicitly; listing every header would make each pattern O(sections)
        # and locate() quadratic in the number of chunks.
        if self._suffix:
            generic = (
                rf"{re.escape(self._prefix)}{_HEADER_NAME}"
                rf"{re.escape(self._suffix)}(?=[ \t\r]|$)"
            )
            name_ok = re.compile(_HEADER_NAME + r"\Z")
            known = "|".join(header_re(s) for s in self.sections if not name_ok.match(s))
        else:
            generic = re.escape(self._prefix)
            known = ""
        ends = rf"(?=\n(?:{known}|{generic})|\Z)" if known else rf"(?=\n{generic}|\Z)"

        flags = re.DOTALL | re.IGNORECASE | re.MULTILINE
        self._extraction_patterns: dict[str, re.Pattern[str]] = {}
        # Compiled on first use: only the free-text tiers need them, and
        # compiling is the main cost of a schema with many sections.
        self._mention_sources: dict[str, str] = {}
        self._mention_patterns: dict[str, re.Pattern[str]] = {}
        self._flags = flags
        for section in self.sections:
            # In sectioned text the header must start a line: a mid-line
            # mention ("see [status]") does not start the section.
            self._extraction_patterns[section] = re.compile(
                rf"({header_re(section, line_start=True)}.*?){ends}", flags
            )
            # Free-text tiers ("USER: [deploy] failed") mention a section
            # anywhere in a line.
            self._mention_sources[section] = rf"({header_re(section)}.*?){ends}"

        self._id = hash((tuple(self.sections), self.section_header_format))

    def extraction_pattern(self, section: str) -> re.Pattern[str]:
        """The section's block in sectioned text (state / notes / reference):
        from a line-start header up to the next header line."""
        return self._extraction_patterns[section]

    def mention_pattern(self, section: str) -> re.Pattern[str]:
        """Like extraction_pattern(), but the header may appear mid-line —
        used for the free-text tiers (cache / history) when scoring."""
        pat = self._mention_patterns.get(section)
        if pat is None:
            pat = re.compile(self._mention_sources[section], self._flags)
            self._mention_patterns[section] = pat
        return pat

    def schema_id(self) -> int:
        return self._id


def validate_max_chars(max_chars: int | None) -> None:
    """A context budget is None (off) or a positive int of characters."""
    if max_chars is None:
        return
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise ValueError(f"max_chars must be a positive int or None, got {max_chars!r}")
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1 or None, got {max_chars}")
