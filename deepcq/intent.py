"""Two-phase section relevance scorer."""
from __future__ import annotations
from .schema import MemorySchema
from .tokens import tokenize
from .anchors import extract_literal_anchors


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

    A query also carries "literal anchors" (emoji/symbol runs, quoted
    phrases — see anchors.py) that a tokenizer would otherwise drop. If
    a section's raw matched text contains one of those anchors verbatim,
    the section scores 1.0 regardless of token overlap — this only ever
    raises a score, so it never affects queries without such anchors.
    """
    query_tokens = set(tokenize(query, include_stems=True))
    literal_anchors = extract_literal_anchors(query)
    if not query_tokens and not literal_anchors:
        return {s: 0.0 for s in schema.sections}

    scores: dict[str, float] = {}

    for section in schema.sections:
        vocab: set[str] = set()

        vocab.update(tokenize(section, include_stems=True))

        for kw in schema.keywords.get(section, []):
            vocab.update(tokenize(kw, include_stems=True))

        section_text = ""
        pat = schema.extraction_pattern(section)
        for text in tier_texts.values():
            if not text:
                continue
            m = pat.search(text)
            if m:
                section_text += m.group(0)
                vocab.update(tokenize(m.group(0), include_stems=True))

        token_score = (
            round(len(query_tokens & vocab) / len(query_tokens), 4)
            if query_tokens else 0.0
        )
        literal_hit = any(anchor in section_text for anchor in literal_anchors)
        scores[section] = 1.0 if literal_hit else token_score

    return scores
