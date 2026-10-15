"""Dynamic regex pattern builder from query tokens."""
from __future__ import annotations
import re
import functools
from .tokens import tokenize


@functools.lru_cache(maxsize=512)
def _compile_content_pattern(token_tuple: tuple[str, ...]) -> re.Pattern[str]:
    if not token_tuple:
        return re.compile(r"(?!)")
    escaped = [re.escape(t) for t in sorted(token_tuple, key=len, reverse=True)]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


def build_content_pattern(query: str) -> re.Pattern[str]:
    tokens = tuple(sorted(set(tokenize(query, include_stems=True))))
    return _compile_content_pattern(tokens)


def content_pattern_string(query: str) -> str:
    return build_content_pattern(query).pattern
