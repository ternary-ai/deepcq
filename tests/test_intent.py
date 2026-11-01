import re

import pytest

from deepcq.schema import MemorySchema
from deepcq.intent import score_sections, score_sections_detailed
from deepcq.decay import Recency, exponential_decay


def _make_schema():
    return MemorySchema(sections=["status", "errors", "metrics"], confidence_threshold=0.3)


def _make_texts():
    return {
        "state": (
            "[status]\nAll services running\n"
            "[errors]\n- OOM at 02:15\n- Timeout at 08:44\n"
            "[metrics]\nCPU: 78%\nMemory: 4.2GB\nLatency: 120ms\n"
        )
    }


def test_metrics_scores_high_for_cpu_query():
    schema = _make_schema()
    scores = score_sections("any cpu or memory issues?", schema, _make_texts())
    assert scores["metrics"] > scores["status"]
    assert scores["metrics"] > scores["errors"]


def test_errors_scores_high_for_error_query():
    schema = _make_schema()
    scores = score_sections("show me the latest errors", schema, _make_texts())
    assert scores["errors"] > scores["status"]
    assert scores["errors"] >= schema.confidence_threshold


def test_status_scores_high_for_status_query():
    schema = _make_schema()
    scores = score_sections("is the service running?", schema, _make_texts())
    assert scores["status"] >= schema.confidence_threshold


def test_scores_bounded():
    schema = _make_schema()
    scores = score_sections("random query with no hits", schema, _make_texts())
    for v in scores.values():
        assert 0.0 <= v <= 1.0


def test_keyword_boost():
    schema = MemorySchema(
        sections=["db"],
        keywords={"db": ["database", "postgres", "query"]},
    )
    texts = {"state": "[db]\npostgres latency: 50ms\n"}
    scores = score_sections("how fast are database queries?", schema, texts)
    assert scores["db"] >= schema.confidence_threshold


def test_literal_emoji_anchor_forces_inclusion():
    # "status" holds only an emoji + a short label — the tokenizer alone
    # gives near-zero overlap with a query about it, so without literal
    # anchor matching this section would fall below the threshold.
    schema = MemorySchema(sections=["status", "errors"], confidence_threshold=0.3)
    texts = {
        "state": (
            "[status]\n🚀 launched\n"
            "[errors]\n- OOM at 02:15\n"
        )
    }
    scores = score_sections("what does 🚀 mean?", schema, texts)
    assert scores["status"] == 1.0
    assert scores["status"] > scores["errors"]


def test_literal_quoted_anchor_forces_inclusion():
    # Query is diluted with several unrelated non-stopword tokens so that
    # bag-of-tokens overlap alone stays well below 1.0 — only the exact
    # quoted phrase should be able to push the score to 1.0.
    schema = MemorySchema(sections=["config", "errors"], confidence_threshold=0.3)
    texts = {
        "state": (
            "[config]\nfeature toggle: circuit-breaker-v2 enabled\n"
            "[errors]\n- OOM at 02:15\n"
        )
    }
    query = (
        'please explain the deployment rollback throttle safeguard '
        '"circuit-breaker-v2" policy now'
    )
    scores = score_sections(query, schema, texts)
    assert scores["config"] == 1.0


def test_detailed_explanation_reports_matched_tokens_and_anchors():
    schema = _make_schema()
    explanations = score_sections_detailed(
        "show me the latest errors", schema, _make_texts()
    )
    errors_exp = explanations["errors"]
    assert errors_exp.score == score_sections(
        "show me the latest errors", schema, _make_texts()
    )["errors"]
    assert "error" in errors_exp.matched_tokens
    assert errors_exp.literal_hit is False
    assert errors_exp.matched_anchors == []


def test_detailed_explanation_reports_literal_hit():
    schema = MemorySchema(sections=["status", "errors"], confidence_threshold=0.3)
    texts = {
        "state": (
            "[status]\n🚀 launched\n"
            "[errors]\n- OOM at 02:15\n"
        )
    }
    explanations = score_sections_detailed("what does 🚀 mean?", schema, texts)
    assert explanations["status"].literal_hit is True
    assert explanations["status"].matched_anchors == ["🚀"]
    assert explanations["status"].score == 1.0


def test_recency_multiplies_down_score():
    schema = _make_schema()
    texts = _make_texts()
    now = 10_000.0
    recency = {"errors": Recency(timestamp=now - 100.0, decay_fn=exponential_decay(100.0))}

    undamped = score_sections("show me the latest errors", schema, texts)
    damped = score_sections(
        "show me the latest errors", schema, texts, recency=recency, now=now
    )
    assert damped["errors"] == round(undamped["errors"] * 0.5, 4)
    assert damped["status"] == undamped["status"]


def test_recency_does_not_affect_unmapped_sections():
    schema = _make_schema()
    texts = _make_texts()
    recency = {"errors": Recency(timestamp=0.0, decay_fn=exponential_decay(1.0))}
    scores = score_sections(
        "show me the latest errors", schema, texts, recency=recency, now=1_000_000.0
    )
    assert scores["errors"] == 0.0


def test_schema_custom_anchor_patterns_used_instead_of_defaults():
    ticket_id = re.compile(r"\bTICKET-\d+\b")
    schema = MemorySchema(
        sections=["tickets", "errors"],
        anchor_patterns=[ticket_id],
    )
    texts = {
        "state": (
            "[tickets]\nTICKET-4471 open\n"
            "[errors]\n- OOM at 02:15\n"
        )
    }
    # Query has no ticket-id but does have a quoted phrase; since the
    # schema overrides with only the ticket pattern, quotes shouldn't
    # trigger a literal hit.
    scores = score_sections('any "OOM" issues?', schema, texts)
    assert scores["errors"] < 1.0

    scores = score_sections("what about TICKET-4471?", schema, texts)
    assert scores["tickets"] == 1.0


# --- 0.3.0: parity net -------------------------------------------------------
# Reference copy of the pre-0.3.0 scorer: tier texts concatenated into one
# vocabulary, score = matched / query tokens, literal anchor forces 1.0.
# Updated for the two intended 0.3.1 changes: anchors match whole words /
# numbers only (anchor_in), and a section found in no tier scores 0.0.
# Headers start a line in sectioned tiers; free-text tiers (cache, history)
# still match a mid-line mention.

from deepcq.tokens import tokenize
from deepcq.anchors import anchor_in, extract_literal_anchors
from deepcq.schema import TIER_NAMES
from deepcq.vocabulary import SynonymMap


def _legacy_scores(query, schema, tier_texts):
    query_tokens = set(tokenize(query, include_stems=True))
    anchors = extract_literal_anchors(query, schema.anchor_patterns)
    scores = {}
    for section in schema.sections:
        vocab = set(tokenize(section, include_stems=True))
        for kw in schema.keywords.get(section, []):
            vocab.update(tokenize(kw, include_stems=True))
        section_texts = []
        for tier_name, text in tier_texts.items():
            pat = (schema.mention_pattern(section) if tier_name in ("cache", "history")
                   else schema.extraction_pattern(section))
            if text and (m := pat.search(text)):
                section_texts.append(m.group(0))
                vocab.update(tokenize(m.group(0), include_stems=True))
        if not section_texts:
            scores[section] = 0.0
            continue
        token_score = (
            round(len(query_tokens & vocab) / len(query_tokens), 4) if query_tokens else 0.0
        )
        hit = any(anchor_in(a, t) for a in anchors for t in section_texts)
        scores[section] = 1.0 if hit else token_score
    return scores


_PARITY_QUERIES = [
    "any cpu or memory issues?", "show me the latest errors", "is the service running?",
    'what about "OOM" at 02:15?', "what does 🚀 mean?", "timeout latency 120ms",
    "random words nothing", "", "postgres database queries", "78% cpu moving average",
    "is 10 too high?", "8% cpu", "any audit findings?",
]

_PARITY_FIXTURES = [
    (_make_schema(), _make_texts()),
    (
        MemorySchema(sections=["db", "status"], keywords={"db": ["database", "postgres"]}),
        {
            "state": "[status]\nAll services running 🚀\n",
            "cache": "[db]\npostgres latency 50ms",
            "history": "USER: [db] what about database queries?",
            "notes": "[db]\nmigrated to postgres 16, moving average latency down",
            "reference": "[status] OOM at 02:15",
        },
    ),
    (
        # "audit" is absent from every tier but has keywords; numbers that
        # only occur inside longer numbers ("100", "2.10", "78%").
        MemorySchema(sections=["metrics", "audit"], keywords={"audit": ["audit", "findings"]}),
        {"state": "[metrics]\nlatency 100ms p99 2.10s CPU: 78%\n"},
    ),
]


@pytest.mark.parametrize("schema,texts", _PARITY_FIXTURES)
@pytest.mark.parametrize("query", _PARITY_QUERIES)
def test_defaults_match_legacy_scorer(schema, texts, query):
    assert score_sections(query, schema, texts) == _legacy_scores(query, schema, texts)


@pytest.mark.parametrize("schema,texts", _PARITY_FIXTURES)
@pytest.mark.parametrize("query", _PARITY_QUERIES)
def test_opt_in_fields_at_neutral_values_match_legacy(schema, texts, query):
    neutral = MemorySchema(
        sections=schema.sections,
        keywords=schema.keywords,
        tier_weights={t: 1.0 for t in TIER_NAMES},
        schema_weight=1.0,
        vocabulary=SynonymMap(),
        phrase_sizes=(),
    )
    assert score_sections(query, neutral, texts) == _legacy_scores(query, schema, texts)


# --- 0.3.0: synonym vocabulary ----------------------------------------------

def _fundamentals_texts():
    return {
        "state": (
            "[fundamentals]\nearnings per share 4.12, capital expenditure 1.2B\n"
            "[news]\nCEO interview scheduled\n"
        )
    }


def test_synonym_resolves_section_without_keywords():
    plain = MemorySchema(sections=["fundamentals", "news"])
    vocab = MemorySchema(
        sections=["fundamentals", "news"],
        vocabulary=SynonymMap.from_dict({"eps": ["earnings per share"], "capex": ["capital expenditure"]}),
    )
    query = "eps and capex?"
    assert score_sections(query, plain, _fundamentals_texts())["fundamentals"] == 0.0
    scores = score_sections(query, vocab, _fundamentals_texts())
    assert scores["fundamentals"] == 1.0
    assert scores["news"] == 0.0


def test_synonym_expansion_works_from_stored_jargon_side():
    schema = MemorySchema(
        sections=["fundamentals"],
        vocabulary=SynonymMap.from_dict({"eps": ["earnings per share"]}),
    )
    texts = {"state": "[fundamentals]\nEPS 4.12\n"}
    exp = score_sections_detailed("earnings per share", schema, texts)["fundamentals"]
    assert exp.score == 1.0
    assert "share" in exp.matched_tokens


def test_synonym_does_not_dilute_unrelated_query_tokens():
    # Expansion is vocab-side only: a query mentioning "eps" keeps the same
    # number of query tokens, so a section matching only "revenue" scores
    # the same with or without a vocabulary.
    texts = {"state": "[sales]\nrevenue up 4%\n"}
    plain = MemorySchema(sections=["sales"])
    vocab = MemorySchema(sections=["sales"], vocabulary=SynonymMap.from_dict({"eps": ["earnings per share"]}))
    assert score_sections("eps revenue", plain, texts) == score_sections("eps revenue", vocab, texts)


# --- 0.3.0: phrase-aware matching -------------------------------------------

def _indicator_texts():
    return {
        "state": (
            "[indicators]\nmoving average 50d crossed above 200d; RSI 61; MACD flat; volume light\n"
            "[news]\nmoving to a new office, average commute shorter\n"
        )
    }


def test_phrase_bonus_pushes_section_over_threshold():
    query = "moving average signal trend strength momentum"
    plain = MemorySchema(sections=["indicators", "news"], confidence_threshold=0.5)
    phrased = MemorySchema(
        sections=["indicators", "news"], confidence_threshold=0.5, phrase_sizes=(2,)
    )
    before = score_sections(query, plain, _indicator_texts())
    after = score_sections_detailed(query, phrased, _indicator_texts())
    assert before["indicators"] < 0.5
    assert after["indicators"].score >= 0.5
    assert after["indicators"].matched_phrases == ["mov average"]
    assert after["indicators"].score == round(before["indicators"] + 0.25, 4)
    # "news" has both words, but not adjacent — no bonus.
    assert after["news"].score == before["news"]
    assert after["news"].matched_phrases == []


def test_phrase_bonus_capped_at_one():
    schema = MemorySchema(sections=["indicators"], phrase_sizes=(2, 3), phrase_boost=1.0)
    exp = score_sections_detailed("moving average", schema, _indicator_texts())["indicators"]
    assert exp.score == 1.0


def test_phrase_sizes_empty_is_parity():
    query = "moving average signal"
    assert score_sections(query, MemorySchema(sections=["indicators", "news"]), _indicator_texts()) == \
        _legacy_scores(query, MemorySchema(sections=["indicators", "news"]), _indicator_texts())


# --- 0.3.0: per-tier weighting ----------------------------------------------

def _two_tier_texts():
    return {
        "state": "[deploy]\nrollout healthy\n",
        "history": "USER: [deploy] the canary rollback happened yesterday",
    }


def test_token_counts_at_max_tier_weight_not_sum():
    schema = MemorySchema(sections=["deploy"], tier_weights={"history": 0.5})
    # "rollout" only in state (1.0); "rollback" only in history (0.5);
    # "deploy" is in the section name (schema_weight 1.0) and both tiers.
    exp = score_sections_detailed("deploy rollout rollback", schema, _two_tier_texts())["deploy"]
    assert exp.score == round((1.0 + 1.0 + 0.5) / 3, 4)
    assert exp.tier_contributions == {"schema": round(1 / 3, 4), "state": round(1 / 3, 4), "history": round(0.5 / 3, 4)}


def test_token_found_in_two_tiers_counts_once_at_best_weight():
    schema = MemorySchema(sections=["deploy"], tier_weights={"state": 0.8, "history": 0.4}, schema_weight=0.0)
    texts = {"state": "[deploy]\ncanary ok\n", "history": "USER: [deploy] canary failing"}
    exp = score_sections_detailed("canary", schema, texts)["deploy"]
    assert exp.score == 0.8
    assert exp.tier_contributions == {"state": 0.8}


def test_weights_above_one_are_capped_per_token():
    schema = MemorySchema(sections=["deploy"], tier_weights={"state": 3.0})
    exp = score_sections_detailed("rollout", schema, _two_tier_texts())["deploy"]
    assert exp.score == 1.0


def test_literal_anchor_weighted_by_tier_it_was_found_in():
    schema = MemorySchema(sections=["deploy"], tier_weights={"history": 0.4})
    texts = {"state": "[deploy]\nrollout healthy\n", "history": "USER: [deploy] build v2.3.1 failed"}
    exp = score_sections_detailed("anything on v2.3.1?", schema, texts)["deploy"]
    assert exp.literal_hit is True
    assert exp.score == 0.4


def test_tier_contributions_sum_to_token_score():
    schema = _make_schema()
    for exp in score_sections_detailed("any cpu or memory issues?", schema, _make_texts()).values():
        assert abs(sum(exp.tier_contributions.values()) - exp.score) < 1e-3


def test_schema_rejects_bad_opt_in_values():
    with pytest.raises(ValueError, match="unknown tier"):
        MemorySchema(sections=["a"], tier_weights={"stat": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        MemorySchema(sections=["a"], tier_weights={"state": -1.0})
    with pytest.raises(ValueError, match="phrase_sizes"):
        MemorySchema(sections=["a"], phrase_sizes=(1,))
