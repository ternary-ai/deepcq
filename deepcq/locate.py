"""Locate the relevant chunk(s) of a text blob for a query — the same
question a code-search tool answers for a query against a codebase."""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import Sequence
from .schema import DEFAULT_MAX_CHARS, MemorySchema, validate_max_chars
from .core import DeepCQ, _FALLBACK_COUNTS
from .decay import Recency

logger = logging.getLogger("deepcq")

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
    max_chars: int | None = DEFAULT_MAX_CHARS,
) -> list[Candidate]:
    """
    `recency` optionally maps a chunk *index* (not the internal section
    name) to a Recency that time-weights that chunk's score — see
    decay.py. `anchor_patterns` overrides the literal-anchor patterns used
    for this call (see anchors.py); None uses the library defaults.

    `max_chars` (default 25,000; None = off) caps the total length of the
    returned chunk texts: candidates are kept highest score first (ties by
    index), a chunk that doesn't fit is skipped, never truncated, and the
    cut is logged and counted as "budget_cut". `max_candidates` applies
    first, then the budget. The result stays sorted by index.
    """
    validate_max_chars(max_chars)
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
        max_chars=None,
    )
    # An internal, throwaway instance: no usage counters, and no memory-gap
    # reports — the text being searched isn't stored memory.
    cq = DeepCQ(schema=schema, track_usage=False)
    cq._report_gaps = False
    for i, c in enumerate(chunks):
        cq.state.set(f"chunk_{i}", f"{_CHUNK_HEADER_FORMAT.format(name=f'chunk_{i}')}\n{c}")

    _recency_by_name = (
        {f"chunk_{i}": r for i, r in recency.items()} if recency else None
    )
    result = cq.query(query, recency=_recency_by_name, now=now, max_chars=None)

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
    if max_chars is not None:
        used = 0
        kept: list[Candidate] = []
        cut: list[int] = []
        for cand in sorted(candidates, key=lambda c: (-c.score, c.index)):
            if used + len(cand.text) <= max_chars:
                kept.append(cand)
                used += len(cand.text)
            else:
                cut.append(cand.index)
        if cut:
            _FALLBACK_COUNTS["budget_cut"] += 1
            logger.warning(
                "deepcq: locate() context budget max_chars=%d left out %d chunk(s) at indices %s "
                "(process total budget_cut: %d)",
                max_chars, len(cut), sorted(cut), _FALLBACK_COUNTS["budget_cut"],
            )
        candidates = sorted(kept, key=lambda c: c.index)
    return candidates
