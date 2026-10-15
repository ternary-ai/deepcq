from deepcq.schema import MemorySchema
from deepcq.intent import score_sections


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
