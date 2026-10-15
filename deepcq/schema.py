"""MemorySchema — defines section names and optional keyword overrides."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class MemorySchema:
    """
    Declares the logical sections that structure stored memory text.

    sections:
        List of section names, e.g. ["errors", "metrics", "status"].
        No keyword lists required — deepcq derives patterns from the
        query itself (query-term-first approach).

    keywords:
        Optional per-section extra terms that boost relevance scoring.
        Useful when section names are abbreviations or jargon.

    section_header_format:
        How sections are delimited in stored text.
        Default "[{name}]" means sections look like "[errors]", "[metrics]".

    confidence_threshold:
        Minimum score (0.0–1.0) for a section to be included.

    fallback_include_all:
        When True (default), return all content unfiltered if no section
        scores above the threshold. Prevents silent data loss.
    """

    sections: Sequence[str]
    keywords: dict[str, list[str]] = field(default_factory=dict)
    section_header_format: str = "[{name}]"
    confidence_threshold: float = 0.3
    fallback_include_all: bool = True

    def __post_init__(self) -> None:
        name_idx = self.section_header_format.index("{name}")
        self._prefix = self.section_header_format[:name_idx]

        escaped_prefix = re.escape(self._prefix)
        lookahead = rf"(?=\n{escaped_prefix}|\Z)"

        self._extraction_patterns: dict[str, re.Pattern[str]] = {}
        for section in self.sections:
            header = self.section_header_format.format(name=section)
            pat = re.compile(
                rf"({re.escape(header)}.*?){lookahead}",
                re.DOTALL | re.IGNORECASE,
            )
            self._extraction_patterns[section] = pat

        self._id = hash((tuple(self.sections), self.section_header_format))

    def extraction_pattern(self, section: str) -> re.Pattern[str]:
        return self._extraction_patterns[section]

    def schema_id(self) -> int:
        return self._id
