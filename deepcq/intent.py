"""Two-phase section relevance scorer."""
from __future__ import annotations
from typing import Mapping
from .schema import MemorySchema
from .tokens import tokenize
from .anchors import anchor_in, extract_literal_anchors
from .phrases import extract_phrases
from .result import ScoreExplanation
from .decay import Recency
from .memory._vocab_cache import VocabCache, VocabEntry

# Tiers without section structure: a section header may appear mid-line.
FREE_TEXT_TIERS = frozenset({"cache", "history"})
SCHEMA_SOURCE = "schema"   # tier_contributions key for name/keyword matches


def score_sections_detailed(
    query: str,
    schema: MemorySchema,
    tier_texts: dict[str, str],
    recency: dict[str, Recency] | None = None,
    now: float | None = None,
    tiers: Mapping[str, VocabCache] | None = None,
) -> dict[str, ScoreExplanation]:
    """
    Score each schema section for relevance to the query, with an
    explanation of what drove the score.

    Each section's vocabulary comes from several sources, kept separate so
    they can be weighted:
      - "schema": the section name tokens (and optional schema keywords),
        weighted by `schema.schema_weight`.
      - one per tier: the tokenized text of that section as matched in the
        tier, weighted by `schema.tier_weights[tier]` (default 1.0).
    With `schema.vocabulary` set, each source is also expanded with every
    synonym cluster whose term occurs in it.

    Token score = mean over unique query tokens of the highest weight among
    the sources containing that token (0 if none), each capped at 1.0. At
    default weights this is exactly the proportion of query tokens found.

    With `schema.phrase_sizes` set, each query phrase (stemmed n-gram, see
    phrases.py) also found in a source adds `schema.phrase_boost` times that
    source's weight; the bonus only ever raises the score, capped at 1.0.

    A query also carries "literal anchors" (emoji/symbol runs, quoted
    phrases, numbers, identifiers — see anchors.py) that a tokenizer would
    otherwise drop. If a section's matched text in some tier contains one of
    those anchors verbatim — as a whole word/number, see anchors.anchor_in —
    the score is raised to that tier's weight (1.0
    by default) — this only ever raises a score, so it never affects
    queries without such anchors.

    `recency` optionally maps a section name to a Recency (timestamp +
    decay_fn) that multiplies the section's score down based on age,
    evaluated against `now` (defaults to the current time).

    A section found in no tier scores 0.0 with `present=False`; the score
    its name/keywords alone would have given is kept as `absent_score`.

    `tiers` optionally maps tier names to the tier objects that produced
    `tier_texts`; their per-version cache then skips re-extracting and
    re-tokenizing sections whose tier has not changed since the last call.
    """
    query_tokens = sorted(set(tokenize(query, include_stems=True)))
    literal_anchors = extract_literal_anchors(query, schema.anchor_patterns)
    query_phrases = extract_phrases(query, schema.phrase_sizes)
    vocabulary = schema.vocabulary
    tier_weights = schema.tier_weights or {}
    schema_id = schema.schema_id()

    explanations: dict[str, ScoreExplanation] = {}

    for section in schema.sections:
        # (source name, weight, vocab, texts the source's phrases come from)
        schema_pieces = [section, *schema.keywords.get(section, [])]
        schema_vocab: set[str] = set()
        for piece in schema_pieces:
            schema_vocab.update(tokenize(piece, include_stems=True))
            if vocabulary is not None:
                schema_vocab |= vocabulary.expand_text(piece)
        sources: list[tuple[str, float, set[str], list[str]]] = [
            (SCHEMA_SOURCE, schema.schema_weight, schema_vocab, schema_pieces)
        ]

        for tier_name, text in tier_texts.items():
            if not text:
                continue

            pat = (schema.mention_pattern(section) if tier_name in FREE_TEXT_TIERS
                   else schema.extraction_pattern(section))

            def compute(text: str = text, pat=pat) -> VocabEntry:
                m = pat.search(text)
                if not m:
                    return "", []
                return m.group(0), tokenize(m.group(0), include_stems=True)

            tier = tiers.get(tier_name) if tiers else None
            matched_text, tier_tokens = (
                tier._vocab_entry(schema_id, section, compute) if tier is not None else compute()
            )
            if not matched_text:
                continue
            tier_vocab = set(tier_tokens)
            if vocabulary is not None:
                tier_vocab |= vocabulary.expand_text(matched_text)
            sources.append((tier_name, tier_weights.get(tier_name, 1.0), tier_vocab, [matched_text]))

        matched_tokens: list[str] = []
        contributions: dict[str, float] = {}
        total = 0.0
        for token in query_tokens:
            best: tuple[str, float] | None = None
            for name, weight, vocab, _ in sources:
                if token in vocab and (best is None or weight > best[1]):
                    best = (name, weight)
            if best is None:
                continue
            matched_tokens.append(token)
            contribution = min(1.0, best[1])
            total += contribution
            contributions[best[0]] = contributions.get(best[0], 0.0) + contribution

        token_score = round(total / len(query_tokens), 4) if query_tokens else 0.0
        tier_contributions = {
            name: round(c / len(query_tokens), 4) for name, c in contributions.items() if c > 0
        }
        score = token_score

        matched_phrases: list[str] = []
        if query_phrases:
            phrase_weights: dict[str, float] = {}
            for _, weight, _, texts in sources:
                for text in texts:
                    for phrase in extract_phrases(text, schema.phrase_sizes) & query_phrases:
                        phrase_weights[phrase] = max(phrase_weights.get(phrase, 0.0), min(1.0, weight))
            matched_phrases = sorted(phrase_weights)
            phrase_bonus = min(1.0, schema.phrase_boost * sum(phrase_weights.values()))
            if phrase_bonus > 0:
                score = round(min(1.0, score + phrase_bonus), 4)

        matched_anchors: list[str] = []
        anchor_weight = 0.0
        for anchor in literal_anchors:
            hit_weights = [
                min(1.0, weight) for name, weight, _, texts in sources
                if name != SCHEMA_SOURCE and anchor_in(anchor, texts[0])
            ]
            if hit_weights:
                matched_anchors.append(anchor)
                anchor_weight = max(anchor_weight, *hit_weights)
        literal_hit = bool(matched_anchors)
        if literal_hit:
            score = max(score, anchor_weight)

        if recency and section in recency:
            score = round(score * recency[section].multiplier(now), 4)

        # A section no tier contains has nothing to return; matches on its
        # name/keywords alone must not give it a passing score.
        present = len(sources) > 1
        absent_score = 0.0
        if not present:
            absent_score, score = score, 0.0
            matched_tokens, matched_anchors, matched_phrases = [], [], []
            literal_hit, tier_contributions = False, {}

        explanations[section] = ScoreExplanation(
            score=score,
            matched_tokens=matched_tokens,
            matched_anchors=matched_anchors,
            literal_hit=literal_hit,
            matched_phrases=matched_phrases,
            tier_contributions=tier_contributions,
            present=present,
            absent_score=absent_score,
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
