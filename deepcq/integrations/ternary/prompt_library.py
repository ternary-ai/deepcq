"""
PromptLibrary — loads ternary-ai/prompts CSVs into queryable deepcq memory.

Each CSV file becomes one schema section. Each non-empty line in the CSV
becomes one prompt entry in that section. Prompt templates may contain
{variable} placeholders — these are stored and returned verbatim.

Usage::

    from deepcq.integrations.ternary import PromptLibrary

    # From a local clone of ternary-ai/prompts:
    lib = PromptLibrary.from_local("/path/to/ternary-ai/prompts/")

    # From raw GitHub URLs (uses urllib, no extra deps):
    lib = PromptLibrary.from_urls(PromptLibrary.DEFAULT_URLS)

    result = lib.query("portfolio concentration risk")
    print(result.sections_included)   # e.g. ["risk-management", "portfolio"]
    print(result.content)             # matching prompt templates
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable

from deepcq import DeepCQ, MemorySchema, QueryResult


_BASE_URL = "https://raw.githubusercontent.com/ternary-ai/prompts/main/"

SECTIONS = [
    "agents",
    "analysis",
    "concepts",
    "content",
    "journal",
    "knowledgegraphs",
    "named-entities",
    "portfolio",
    "research",
    "risk-management",
    "scenario-analysis",
    "summarization",
    "trading",
]

DEFAULT_URLS: dict[str, str] = {s: f"{_BASE_URL}{s}.csv" for s in SECTIONS}


class PromptLibrary:
    """
    Queryable library of ternary-ai financial prompt templates.

    Wraps DeepCQ with a schema whose sections map 1-to-1 to the CSV files
    in the ternary-ai/prompts repository. Querying returns only the
    sections whose stored prompts are most relevant to the user's query.
    """

    SECTIONS = SECTIONS
    DEFAULT_URLS = DEFAULT_URLS

    def __init__(self) -> None:
        schema = MemorySchema(
            sections=self.SECTIONS,
            section_header_format="[{name}]",
            confidence_threshold=0.25,  # broader recall for prompt retrieval
            fallback_include_all=False,  # don't dump everything on vague queries
        )
        self._cq = DeepCQ(schema=schema)
        self._loaded: set[str] = set()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def load_section(self, section: str, rows: Iterable[str]) -> None:
        """Store a list of prompt strings into the named section."""
        lines = [r.strip() for r in rows if r.strip()]
        if not lines:
            return
        body = "\n".join(lines)
        self._cq.state.set(section, f"[{section}]\n{body}\n")
        self._loaded.add(section)

    def load_text(self, section: str, raw: str) -> None:
        """Parse raw CSV text (one prompt per line) and load into section."""
        reader = csv.reader(io.StringIO(raw))
        prompts: list[str] = []
        for row in reader:
            # Each row may be a single cell or bare line
            text = " ".join(row).strip()
            if text:
                prompts.append(text)
        self.load_section(section, prompts)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_local(cls, directory: str | Path) -> "PromptLibrary":
        """
        Load all CSV files from a local clone of ternary-ai/prompts.

        Only files whose stem matches a known section name are loaded.
        """
        lib = cls()
        base = Path(directory)
        for section in cls.SECTIONS:
            path = base / f"{section}.csv"
            if path.exists():
                lib.load_text(section, path.read_text(encoding="utf-8"))
        return lib

    @classmethod
    def from_urls(
        cls,
        url_map: dict[str, str] | None = None,
        timeout: int = 10,
    ) -> "PromptLibrary":
        """
        Fetch CSV files from raw GitHub URLs using urllib (stdlib only).

        Defaults to ternary-ai/prompts on GitHub. Pass a custom url_map to
        override individual sections or point to a fork.

        Args:
            url_map: section_name → raw URL. Defaults to DEFAULT_URLS.
            timeout: HTTP request timeout in seconds.
        """
        import urllib.request
        from urllib.parse import urlparse

        lib = cls()
        urls = url_map if url_map is not None else cls.DEFAULT_URLS

        for section, url in urls.items():
            if urlparse(url).scheme not in ("http", "https"):
                print(
                    f"[deepcq] Warning: refusing to load '{section}' from {url}: "
                    "only http/https URLs are allowed"
                )
                continue
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                lib.load_text(section, raw)
            except Exception as exc:
                # Non-fatal: skip sections that fail to fetch
                print(f"[deepcq] Warning: could not load '{section}' from {url}: {exc}")

        return lib

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def query(self, text: str) -> QueryResult:
        """Return the prompt templates most relevant to the query."""
        return self._cq.query(text)

    def loaded_sections(self) -> list[str]:
        """Sections that have been successfully loaded."""
        return sorted(self._loaded)

    def __repr__(self) -> str:
        return f"PromptLibrary(loaded={self.loaded_sections()})"
