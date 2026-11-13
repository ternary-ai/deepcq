"""Per-tier cache of section-extraction + tokenization results."""
from __future__ import annotations
from typing import Callable

# (matched_text, vocab_tokens, copies): copies is 1, or 2 when the tier holds
# more than one line-start copy of the section (only the first is used).
VocabEntry = tuple[str, list[str], int]


class VocabCache:
    """
    Mixin for memory tiers: a monotonic `_version` bumped by every mutating
    method, plus a cache of `(matched_text, vocab_tokens, copies)` keyed by
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

    def _fresh_entry(self, schema_id: int, section: str) -> VocabEntry | None:
        """The cached entry if it is current for this tier's version, else None."""
        key = (schema_id, section)
        if self._vocab_version_seen.get(key) == self._version:
            return self._vocab_cache[key]
        return None

    def _vocab_entry(
        self, schema_id: int, section: str, compute: Callable[[], VocabEntry]
    ) -> VocabEntry:
        key = (schema_id, section)
        if self._vocab_version_seen.get(key) != self._version:
            self._vocab_cache[key] = compute()
            self._vocab_version_seen[key] = self._version
        return self._vocab_cache[key]
