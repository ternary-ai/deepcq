"""Per-tier cache of section-extraction + tokenization results."""
from __future__ import annotations
from typing import Callable

VocabEntry = tuple[str, list[str]]   # (matched_text, vocab_tokens)


class VocabCache:
    """
    Mixin for memory tiers: a monotonic `_version` bumped by every mutating
    method, plus a cache of `(matched_text, vocab_tokens)` keyed by
    `(schema_id, section)` that is recomputed only when the version moved.

    The key space is schema × section — small and stable for a given
    integration — so no eviction is needed.
    """

    def __init__(self) -> None:
        self._version = 0
        self._vocab_cache: dict[tuple[int, str], VocabEntry] = {}
        self._vocab_version_seen: dict[tuple[int, str], int] = {}

    def _bump(self) -> None:
        self._version += 1

    @property
    def version(self) -> int:
        return self._version

    def _vocab_entry(
        self, schema_id: int, section: str, compute: Callable[[], VocabEntry]
    ) -> VocabEntry:
        key = (schema_id, section)
        if self._vocab_version_seen.get(key) != self._version:
            self._vocab_cache[key] = compute()
            self._vocab_version_seen[key] = self._version
        return self._vocab_cache[key]
