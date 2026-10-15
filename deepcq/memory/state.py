"""T1 — Live application state."""
from __future__ import annotations
from ._vocab_cache import VocabCache


class AppState(VocabCache):
    """
    Primary memory tier: the current state of the running application.

    Each section is stored as raw text that includes its own header, e.g.::

        state.set("errors", "[errors]\\n- OOM at 02:15\\n")
        state.set("metrics", "[metrics]\\nCPU: 78%\\n")
    """

    def __init__(self) -> None:
        super().__init__()
        self._sections: dict[str, str] = {}

    def set(self, section: str, text: str) -> None:
        self._sections[section] = text
        self._bump()

    def get(self, section: str) -> str:
        return self._sections.get(section, "")

    def remove(self, section: str) -> None:
        self._sections.pop(section, None)
        self._bump()

    def sections(self) -> list[str]:
        return list(self._sections.keys())

    def as_text(self) -> str:
        return "\n".join(self._sections.values())

    def as_dict(self) -> dict[str, str]:
        return dict(self._sections)

    def clear(self) -> None:
        self._sections.clear()
        self._bump()
