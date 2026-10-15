"""Two-phase section relevance scorer."""
from __future__ import annotations
from .schema import MemorySchema
from .tokens import tokenize
from .anchors import extract_literal_anchors
from .result import ScoreExplanation
from .decay import Recency


def score_sections_detailed(
    query: str,
    schema: MemorySchema,
    tier_texts: dict[str, str],
    recency: dict[str, Recency] | None = None,
    now: float | None = None,
) -> dict[str, ScoreExplanation]:
    """
    Score each schema section for relevance to the query, with an
    explanation of what drove the score.

    For each section we build a vocabulary from three sources:
      1. The section name tokens (and optional schema keywords).
      2. The tokenized content of that section's stored text.

    Score = proportion of unique query tokens found in the section vocabulary.

    This is query-term-first: the patterns come from the query itself —
    no keyword lists are required. Optional schema keywords extend the
    vocabulary for sections whose names are abbreviations or jargon.

    A query also carries "literal anchors" (emoji/symbol runs, quoted
    phrases, numbers, identifiers — see anchors.py) that a tokenizer would
    otherwise drop. If a section's raw matched text contains one of those
    anchors verbatim, the section scores 1.0 regardless of token overlap —
    this only ever raises a score, so it never affects queries without
    such anchors.

    `recency` optionally maps a section name to a Recency (timestamp +
    decay_fn) that multiplies the section's score down based on age,
    evaluated against `now` (defaults to the current time).
    """
    query_tokens = set(tokenize(query, include_stems=True))
    literal_anchors = extract_literal_anchors(query, schema.anchor_patterns)

    explanations: dict[str, ScoreExplanation] = {}

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

        matched_tokens = sorted(query_tokens & vocab)
        token_score = (
            round(len(matched_tokens) / len(query_tokens), 4)
            if query_tokens else 0.0
        )
        matched_anchors = [a for a in literal_anchors if a in section_text]
        literal_hit = bool(matched_anchors)
        score = 1.0 if literal_hit else token_score

        if recency and section in recency:
            score = round(score * recency[section].multiplier(now), 4)

        explanations[section] = ScoreExplanation(
            score=score,
            matched_tokens=matched_tokens,
            matched_anchors=matched_anchors,
            literal_hit=literal_hit,
        )

    return explanations


def score_sections(
    query: str,
    schema: MemorySchema,
    tier_texts: dict[str, str],
    recency: dict[str, Recency] | None = None,
    now: float | None = None,
) -> dict[str, float]:
    """Score-only view of score_sections_detailed() — see that function
    for which tokens/anchors drove each score."""
    detailed = score_sections_detailed(query, schema, tier_texts, recency=recency, now=now)
    return {section: explanation.score for section, explanation in detailed.items()}
