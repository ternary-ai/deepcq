import re

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
