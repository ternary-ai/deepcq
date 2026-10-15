"""T2 — SessionCache and T3 — History."""
from __future__ import annotations
from collections import deque
from typing import NamedTuple
from ._vocab_cache import VocabCache


class SessionCache(VocabCache):
    """T2 — Ephemeral per-session cache. Deduplicates repeated (key, value) puts."""

    def __init__(self, max_entries: int = 30) -> None:
        super().__init__()
        self._max = max_entries
        self._entries: dict[str, str] = {}
        self._order: list[str] = []

    def put(self, key: str, value: str) -> None:
        if self._entries.get(key) == value:
            return
        if key in self._order:
            self._order.remove(key)
        self._entries[key] = value
        self._order.append(key)
        while len(self._order) > self._max:
            evicted = self._order.pop(0)
            self._entries.pop(evicted, None)
        self._bump()

    def get(self, key: str) -> str | None:
        return self._entries.get(key)

    def as_text(self) -> str:
        if not self._entries:
            return ""
        return "\n".join(f"[cache:{k}]\n{v}" for k, v in self._entries.items())

    def clear(self) -> None:
        self._entries.clear()
        self._order.clear()
        self._bump()


class Turn(NamedTuple):
    role: str
    text: str


class History(VocabCache):
    """T3 — Sliding conversation window. Default: 6 turns."""

    def __init__(self, max_turns: int = 6) -> None:
        super().__init__()
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, role: str, text: str) -> None:
        self._turns.append(Turn(role=role, text=text))
        self._bump()

    def turns(self) -> list[Turn]:
        return list(self._turns)

    def as_text(self) -> str:
        if not self._turns:
            return ""
        return "\n\n".join(f"{t.role.upper()}: {t.text}" for t in self._turns)

    def clear(self) -> None:
        self._turns.clear()
        self._bump()
