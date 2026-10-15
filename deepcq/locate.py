"""Locate the relevant chunk(s) of a text blob for a query — the same
question a code-search tool answers for a query against a codebase."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Sequence
from .schema import MemorySchema
from .core import DeepCQ
from .decay import Recency

_CHUNK_HEADER_FORMAT = "<<CHUNK:{name}>>"   # avoids markdown '[' collision


@dataclass
class Candidate:
    index: int
    text: str        # exact, unmodified original chunk text
    score: float
    matched_tokens: list[str] = field(default_factory=list)
    matched_anchors: list[str] = field(default_factory=list)
    literal_hit: bool = False


def locate(
    text: str,
    query: str,
    chunk_on: str = "\n\n",
    confidence_threshold: float = 0.3,
    max_candidates: int | None = None,
    anchor_patterns: Sequence[re.Pattern[str]] | None = None,
    recency: dict[int, Recency] | None = None,
    now: float | None = None,
) -> list[Candidate]:
    """
    `recency` optionally maps a chunk *index* (not the internal section
    name) to a Recency that time-weights that chunk's score — see
    decay.py. `anchor_patterns` overrides the literal-anchor patterns used
    for this call (see anchors.py); None uses the library defaults.
    """
    chunks = [c.strip() for c in text.split(chunk_on) if c.strip()]
    if not chunks:
        return []

    names = [f"chunk_{i}" for i in range(len(chunks))]
    schema = MemorySchema(
        sections=names,
        section_header_format=_CHUNK_HEADER_FORMAT,
        confidence_threshold=confidence_threshold,
        fallback_include_all=False,
        anchor_patterns=anchor_patterns,
    )
    cq = DeepCQ(schema=schema)
    for i, c in enumerate(chunks):
        cq.state.set(f"chunk_{i}", f"{_CHUNK_HEADER_FORMAT.format(name=f'chunk_{i}')}\n{c}")

    _recency_by_name = (
        {f"chunk_{i}": r for i, r in recency.items()} if recency else None
    )
    result = cq.query(query, recency=_recency_by_name, now=now)

    candidates = []
    for i in (int(s.split("_")[1]) for s in result.sections_included):
        name = f"chunk_{i}"
        exp = result.explanations.get(name)
        candidates.append(Candidate(
            index=i,
            text=chunks[i],
            score=result.intent.get(name, 0.0),
            matched_tokens=exp.matched_tokens if exp else [],
            matched_anchors=exp.matched_anchors if exp else [],
            literal_hit=exp.literal_hit if exp else False,
        ))

    candidates.sort(key=lambda c: c.index)
    if max_candidates is not None and len(candidates) > max_candidates:
        candidates = sorted(candidates, key=lambda c: -c.score)[:max_candidates]
        candidates.sort(key=lambda c: c.index)
    return candidates
