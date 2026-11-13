"""Split stored text into its sections — the parser behind
DeepCQ.get_section() and QueryResult.unknown_sections, exposed for callers
that rebuild filtered text themselves."""
from __future__ import annotations
from dataclasses import dataclass, field
from .schema import MemorySchema


@dataclass
class Section:
    name: str
    text: str        # the block as stored, header line included
    known: bool      # True if `name` is one of the schema's sections


@dataclass
class SectionSplit:
    preamble: str                                   # text before the first header
    sections: list[Section] = field(default_factory=list)

    @property
    def unknown(self) -> list[str]:
        """Names of headers in the text that the schema doesn't declare."""
        return [s.name for s in self.sections if not s.known]

    def find(self, name: str, prefix: bool = True) -> Section | None:
        """Case-insensitive lookup by full name, then (if `prefix`) by name prefix."""
        wanted = name.strip().lower()
        for s in self.sections:
            if s.name.lower() == wanted:
                return s
        if prefix and wanted:
            for s in self.sections:
                if s.name.lower().startswith(wanted):
                    return s
        return None


def split_sections(text: str, schema: MemorySchema) -> SectionSplit:
    """
    Split `text` at every line-start header in `schema.section_header_format`
    (known to the schema or not). Each Section's text runs up to the next
    header, trailing newlines stripped; joining preamble and section texts
    with newlines reproduces the original modulo blank lines between blocks.
    """
    headers = list(schema.header_pattern().finditer(text or ""))
    if not headers:
        return SectionSplit(preamble=text or "")
    known = {s.lower() for s in schema.sections}
    split = SectionSplit(preamble=text[: headers[0].start()])
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        name = m.group("name").strip()
        split.sections.append(Section(name=name, text=text[m.start():end].rstrip("\n"), known=name.lower() in known))
    return split
