"""Literal substrings a tokenizer would drop but exact-matching still needs."""
from __future__ import annotations
import re

_EMOJI_SYMBOL_RE = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\U00002190-\U000021FF"
    "\U00002600-\U000027BF\U00002B00-\U00002BFF‍️]+"
)
_QUOTED_RE = re.compile(r'"([^"\n]{2,80})"|\'([^\'\n]{2,80})\'|“([^”\n]{2,80})”')


def extract_literal_anchors(text: str) -> list[str]:
    """Emoji/symbol runs and quoted phrases from raw text — verbatim,
    unstemmed, unstripped, for exact substring matching."""
    anchors: list[str] = []
    seen: set[str] = set()

    for m in _EMOJI_SYMBOL_RE.finditer(text):
        anchor = m.group(0)
        if anchor not in seen:
            seen.add(anchor)
            anchors.append(anchor)

    for m in _QUOTED_RE.finditer(text):
        anchor = next(g for g in m.groups() if g is not None)
        if anchor not in seen:
            seen.add(anchor)
            anchors.append(anchor)

    return anchors
