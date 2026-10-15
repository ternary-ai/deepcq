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
from typing import Sequence

EMOJI_SYMBOL_PATTERN = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\U00002190-\U000021FF"
    "\U00002600-\U000027BF\U00002B00-\U00002BFF‍️]+"
)
QUOTED_PATTERN = re.compile(r'"([^"\n]{2,80})"|\'([^\'\n]{2,80})\'|“([^”\n]{2,80})”')
# Dotted version-like strings: "v2.3.1", "1.0.0".
VERSION_PATTERN = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b")
# "42%", "3.5%" — checked before NUMBER_PATTERN so the '%' is kept.
PERCENTAGE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%")
# Hyphen/underscore identifier-like tokens: "circuit-breaker-v2", "user_id_9".
IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+\b")
# Standalone numbers worth anchoring on: decimals, or 2+ digit integers.
# Single digits are excluded — too common to be a useful literal anchor.
NUMBER_PATTERN = re.compile(r"\b\d+\.\d+\b|\b\d{2,}\b")

DEFAULT_ANCHOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    EMOJI_SYMBOL_PATTERN,
    QUOTED_PATTERN,
    VERSION_PATTERN,
    PERCENTAGE_PATTERN,
    IDENTIFIER_PATTERN,
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
