"""Two-phase section relevance scorer."""
from __future__ import annotations
from .schema import MemorySchema
from .tokens import tokenize


def score_sections(
    query: str,
    schema: MemorySchema,
    tier_texts: dict[str, str],
) -> dict[str, float]:
    """
    Score each schema section for relevance to the query.

    For each section we build a vocabulary from three sources:
      1. The section name tokens (and optional schema keywords).
      2. The tokenized content of that section's stored text.

    Score = proportion of unique query tokens found in the section vocabulary.

    This is query-term-first: the patterns come from the query itself —
    no keyword lists are required. Optional schema keywords extend the
    vocabulary for sections whose names are abbreviations or jargon.
    """
    query_tokens = set(tokenize(query, include_stems=True))
    if not query_tokens:
        return {s: 0.0 for s in schema.sections}

    scores: dict[str, float] = {}

    for section in schema.sections:
        vocab: set[str] = set()

        vocab.update(tokenize(section, include_stems=True))

        for kw in schema.keywords.get(section, []):
            vocab.update(tokenize(kw, include_stems=True))

        pat = schema.extraction_pattern(section)
        for text in tier_texts.values():
            if not text:
                continue
            m = pat.search(text)
            if m:
                vocab.update(tokenize(m.group(0), include_stems=True))

        matched = query_tokens & vocab
        score = round(len(matched) / len(query_tokens), 4)
        scores[section] = score

    return scores
