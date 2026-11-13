"""Literal substrings a tokenizer would drop but exact-matching still needs.

Anchor extraction is a pluggable list of regex patterns rather than a
hardcoded pair — callers can pass their own `patterns` sequence to
`extract_literal_anchors()`, or extend the defaults:

    from deepcq.anchors import DEFAULT_ANCHOR_PATTERNS, extract_literal_anchors
    import re

    ticket_id = re.compile(r"\\bTICKET-\\d+\\b")
    anchors = extract_literal_anchors(query, [*DEFAULT_ANCHOR_PATTERNS, ticket_id])
"""
from __future__ import annotations
import re
from functools import lru_cache
from typing import Sequence

EMOJI_SYMBOL_PATTERN = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\U00002190-\U000021FF"
    "\U00002600-\U000027BF\U00002B00-\U00002BFF‍️]+"
)
# Single quotes only count when no letter/digit touches them on the outside,
# so apostrophes ("what's the user's plan") never form a quoted phrase.
QUOTED_PATTERN = re.compile(
    r'"([^"\n]{2,80})"|(?<!\w)\'([^\'\n]{2,80})\'(?!\w)|“([^”\n]{2,80})”'
)
# Optional leading minus sign on a number: kept, since "-15%" and "15%" are
# different figures. A hyphen after a letter/digit is a range or compound
# ("10-15%", "Q3-2024"), not a sign, and is not captured.
_SIGN = r"(?:(?<![A-Za-z0-9.,])[-\u2212])?"
# One figure written with commas, kept whole: thousands groups ("1,250,000",
# "1,250.50") or a single decimal comma ("3,5%"). Split, their parts ("250",
# "000") match nothing. Anything else with commas is a list ("10,20,30") and
# is left to the number patterns, one anchor per item.
GROUPED_NUMBER_PATTERN = re.compile(
    _SIGN + r"(?<!\d,)\b(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+,\d+)(?![,.]?\d)%?"
)
# Dotted version-like strings: "v2.3.1", "1.0.0".
VERSION_PATTERN = re.compile(_SIGN + r"\bv?\d+(?:\.\d+){1,3}\b")
# "42%", "3.5%" — checked before NUMBER_PATTERN so the '%' is kept.
PERCENTAGE_PATTERN = re.compile(_SIGN + r"\b\d+(?:\.\d+)?%")
# Hyphen/underscore identifier-like tokens: "circuit-breaker-v2", "user_id_9".
IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+\b")
# Identifiers that start with a digit: filing forms and the like ("10-K",
# "8-K", "10-Q"). A hyphenated part must start with a letter, so ranges
# ("10-15%") and dates ("2024-10-05") are left to the number patterns.
DIGIT_IDENTIFIER_PATTERN = re.compile(r"\b\d+(?:-[A-Za-z][A-Za-z0-9]*)+\b")
# Standalone numbers worth anchoring on: decimals, or 2+ digit integers.
# Single digits are excluded — too common to be a useful literal anchor.
NUMBER_PATTERN = re.compile(_SIGN + r"(?:\b\d+\.\d+\b|\b\d{2,}\b)")

DEFAULT_ANCHOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    EMOJI_SYMBOL_PATTERN,
    QUOTED_PATTERN,
    GROUPED_NUMBER_PATTERN,
    VERSION_PATTERN,
    PERCENTAGE_PATTERN,
    IDENTIFIER_PATTERN,
    DIGIT_IDENTIFIER_PATTERN,
    NUMBER_PATTERN,
)


def _match_text_and_span(m: re.Match[str]) -> tuple[str, tuple[int, int]]:
    if m.lastindex:
        for i, g in enumerate(m.groups(), start=1):
            if g is not None:
                return g, m.span(i)
    return m.group(0), m.span()


def extract_literal_anchors(
    text: str,
    patterns: Sequence[re.Pattern[str]] | None = None,
) -> list[str]:
    """Verbatim, unstemmed, unstripped literal substrings from `text` —
    emoji/symbol runs, quoted phrases, version strings, percentages,
    identifier-like tokens, and standalone numbers, for exact substring
    matching. Pass `patterns` to use a different or extended set instead
    of DEFAULT_ANCHOR_PATTERNS.

    Patterns are applied in order and claim the character spans they
    match; a later pattern's match is skipped if it overlaps a span an
    earlier pattern already claimed. This is what keeps e.g. "15%"
    (PERCENTAGE_PATTERN) from also yielding a bare "15" (NUMBER_PATTERN)."""
    anchors: list[str] = []
    seen: set[str] = set()
    claimed: list[tuple[int, int]] = []

    def overlaps_claimed(span: tuple[int, int]) -> bool:
        start, end = span
        return any(start < c_end and c_start < end for c_start, c_end in claimed)

    for pattern in (DEFAULT_ANCHOR_PATTERNS if patterns is None else patterns):
        for m in pattern.finditer(text):
            anchor, span = _match_text_and_span(m)
            if overlaps_claimed(span):
                continue
            claimed.append(span)
            if anchor not in seen:
                seen.add(anchor)
                anchors.append(anchor)

    return anchors


_MINUS = "-\u2212"


@lru_cache(maxsize=4096)
def _anchor_regex(anchor: str) -> re.Pattern[str]:
    # An anchor matches only the exact figure / word, never part of a longer
    # one: "10" must not match "100", "2010", "2.10" or "10-K", "15%" must not
    # match "115%" or "-15%", and "-15%" must not match the range "10-15%".
    # Anchors bounded by symbols (emoji, quotes' inner punctuation) match
    # anywhere, as before.
    first, last = anchor[:1], anchor[-1:]
    left = r"(?<![A-Za-z0-9])" if first.isalnum() else ""
    if first.isdigit():
        # Not a decimal/thousands tail ("2.10"), not after a minus sign. A
        # hyphen after a letter/digit is a range or compound and still matches.
        left += r"(?<!\d[.,])(?<!(?<![A-Za-z0-9])-)(?<!\u2212)"
    elif first in _MINUS and anchor[1:2].isdigit():
        left = r"(?<![A-Za-z0-9.,])"
    right = r"(?![A-Za-z0-9])" if last.isalnum() else ""
    if last.isdigit():
        # Not followed by more decimals ("10.5") or a word ("10-K", "10-year");
        # a range ("10-15%") still matches.
        right += r"(?![.,]\d)(?!-[A-Za-z])"
    # Case-insensitive: "10-k" asks about "10-K", "form 10-k" about "Form 10-K".
    return re.compile(left + re.escape(anchor) + right, re.IGNORECASE)


def anchor_in(anchor: str, text: str) -> bool:
    """True if `anchor` occurs in `text` (ignoring case) and not as part of a
    longer word or number (see _anchor_regex)."""
    return _anchor_regex(anchor).search(text) is not None
