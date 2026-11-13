"""T4 — LongTermStore and T5 — ReferenceData."""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from ._vocab_cache import VocabCache


def _read_json_dict(path: Path) -> dict:
    """Load a store file. A file that can't be read or isn't a JSON object
    raises: loading it as empty memory would hide the data, and the next
    write would overwrite the file with nothing in it."""
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"deepcq: {path} is not valid JSON ({e}); refusing to load it as empty "
            "memory, which the next write would save over the file"
        ) from e
    if not isinstance(data, dict):
        raise ValueError(f"deepcq: {path} holds a JSON {type(data).__name__}, expected an object")
    return data


def _file_mode(path: Path) -> int:
    try:
        return path.stat().st_mode & 0o777
    except FileNotFoundError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write `data` to a temporary file next to `path`, flush it to disk,
    then swap it into place with os.replace(). The swap is atomic, so a
    crash, kill or full disk mid-write leaves the previous file intact —
    a reader sees the old file or the new one, never a half-written one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        # mkstemp creates the file 0600; keep the store's existing mode, or
        # the umask default a plain write would give a new file.
        os.chmod(tmp, _file_mode(path))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


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
        self._data = _read_json_dict(self._path)

    def _save(self) -> None:
        if self._path:
            _write_json_atomic(self._path, self._data)


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
                self._data = _read_json_dict(self._path)
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
            _write_json_atomic(self._path, self._data)

    def as_text(self) -> str:
        self._ensure_loaded()
        if not self._data:
            return ""
        lines = ["[reference]"]
        lines.extend(f"{k}: {v}" for k, v in self._data.items())
        return "\n".join(lines)
