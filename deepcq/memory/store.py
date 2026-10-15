"""T4 — LongTermStore and T5 — ReferenceData."""
from __future__ import annotations
import json
from pathlib import Path
from ._vocab_cache import VocabCache


class LongTermStore(VocabCache):
    """T4 — Persistent, append-only notes store. File-backed when path given."""

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__()
        self._path = Path(path) if path else None
        self._data: dict[str, list[str]] = {}
        if self._path and self._path.exists():
            self._load()

    def append(self, key: str, text: str) -> None:
        self._data.setdefault(key, []).append(text)
        self._bump()
        if self._path:
            self._save()

    def get(self, key: str) -> list[str]:
        return list(self._data.get(key, []))

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def as_text(self) -> str:
        if not self._data:
            return ""
        parts: list[str] = []
        for key, entries in self._data.items():
            parts.append(f"[notes:{key}]")
            parts.extend(entries)
        return "\n".join(parts)

    def _load(self) -> None:
        try:
            self._data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            self._data = {}

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2))


class ReferenceData(VocabCache):
    """T5 — Lazy-loaded reference / configuration data. File-backed when path given.

    Mutate values through set(): the scoring vocabulary cache is invalidated
    by set() only, so editing a value returned by get() in place (e.g. a
    nested dict) leaves stale cached scoring data behind.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__()
        self._path = Path(path) if path else None
        self._data: dict | None = None

    def _ensure_loaded(self) -> None:
        if self._data is None:
            if self._path and self._path.exists():
                try:
                    self._data = json.loads(self._path.read_text())
                except (json.JSONDecodeError, OSError):
                    self._data = {}
            else:
                self._data = {}

    def get(self, key: str, default=None):
        self._ensure_loaded()
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._ensure_loaded()
        self._data[key] = value
        self._bump()
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2))

    def as_text(self) -> str:
        self._ensure_loaded()
        if not self._data:
            return ""
        lines = ["[reference]"]
        lines.extend(f"{k}: {v}" for k, v in self._data.items())
        return "\n".join(lines)
