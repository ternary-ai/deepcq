"""Locate the relevant chunk(s) of a text blob for a query — the same
question a code-search tool answers for a query against a codebase."""
from __future__ import annotations
from dataclasses import dataclass
from .schema import MemorySchema
from .core import DeepCQ

_CHUNK_HEADER_FORMAT = "<<CHUNK:{name}>>"   # avoids markdown '[' collision


@dataclass
class Candidate:
    index: int
    text: str        # exact, unmodified original chunk text
    score: float


def locate(
    text: str,
    query: str,
    chunk_on: str = "\n\n",
    confidence_threshold: float = 0.3,
    max_candidates: int | None = None,
) -> list[Candidate]:
    chunks = [c.strip() for c in text.split(chunk_on) if c.strip()]
    if not chunks:
        return []

    names = [f"chunk_{i}" for i in range(len(chunks))]
    schema = MemorySchema(
        sections=names,
        section_header_format=_CHUNK_HEADER_FORMAT,
        confidence_threshold=confidence_threshold,
        fallback_include_all=False,
    )
    cq = DeepCQ(schema=schema)
    for i, c in enumerate(chunks):
        cq.state.set(f"chunk_{i}", f"{_CHUNK_HEADER_FORMAT.format(name=f'chunk_{i}')}\n{c}")

    result = cq.query(query)
    candidates = [
        Candidate(index=i, text=chunks[i], score=result.intent.get(f"chunk_{i}", 0.0))
        for i in (int(s.split("_")[1]) for s in result.sections_included)
    ]
    candidates.sort(key=lambda c: c.index)
    if max_candidates is not None and len(candidates) > max_candidates:
        candidates = sorted(candidates, key=lambda c: -c.score)[:max_candidates]
        candidates.sort(key=lambda c: c.index)
    return candidates
