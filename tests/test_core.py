import pytest
from deepcq import DeepCQ, MemorySchema


@pytest.fixture
def cq():
    schema = MemorySchema(sections=["status", "errors", "metrics"])
    instance = DeepCQ(schema=schema)
    instance.state.set("status",  "[status]\nAll services running\nUptime: 12h\n")
    instance.state.set("errors",  "[errors]\n- OOM at 02:15\n- Timeout at 08:44\n")
    instance.state.set("metrics", "[metrics]\nCPU: 78%\nMemory: 4.2GB\nLatency: 120ms\n")
    return instance


def test_metrics_section_returned_for_cpu_query(cq):
    result = cq.query("any memory or cpu issues?")
    assert "metrics" in result.sections_included
    assert "[metrics]" in result.content
    assert "CPU" in result.content


def test_status_excluded_for_cpu_query(cq):
    result = cq.query("any memory or cpu issues?")
    assert "[status]" not in result.content


def test_errors_section_returned_for_error_query(cq):
    result = cq.query("show me the latest errors")
    assert "errors" in result.sections_included
    assert "OOM" in result.content


def test_reduction_positive(cq):
    result = cq.query("any memory or cpu issues?")
    assert result.metrics.reduction_pct > 0


def test_content_pattern_populated(cq):
    result = cq.query("cpu spikes?")
    assert "cpu" in result.content_pattern.lower()


def test_fallback_returns_all_when_no_match(cq):
    result = cq.query("xyzzy flurble")
    assert result.sections_included == list(cq.schema.sections)
    assert result.metrics.reduction_pct == 0.0


def test_history_included_when_relevant(cq):
    cq.history.add("user", "CPU was at 90% yesterday")
    cq.history.add("assistant", "That spike was from the batch job.")
    result = cq.query("what caused the cpu spike?")
    assert "batch job" in result.content or "CPU" in result.content


def test_cache_included_when_relevant(cq):
    cq.cache.put("perf_report", "CPU usage peaked at 95% during deploy")
    result = cq.query("cpu performance during deploy?")
    assert "95%" in result.content or "deploy" in result.content


def test_session_cache_deduplication(cq):
    cq.cache.put("report", "CPU: 78%")
    cq.cache.put("report", "CPU: 78%")
    assert len(cq.cache._entries) == 1


def test_long_term_store(tmp_path):
    schema = MemorySchema(sections=["notes"])
    store_file = tmp_path / "store.json"
    cq = DeepCQ(schema=schema, store_path=str(store_file))
    cq.notes.append("notes", "[notes]\nSystem restarted at midnight.")
    assert store_file.exists()
    cq2 = DeepCQ(schema=schema, store_path=str(store_file))
    assert "System restarted" in cq2.notes.as_text()


def test_intent_scores_returned(cq):
    result = cq.query("any cpu issues?")
    assert set(result.intent.keys()) == {"status", "errors", "metrics"}
    for v in result.intent.values():
        assert 0.0 <= v <= 1.0


def test_sections_included_excluded_are_complete(cq):
    result = cq.query("any memory or cpu issues?")
    all_sections = set(result.sections_included) | set(result.sections_excluded)
    assert all_sections == set(cq.schema.sections)


def test_minimal_state_only():
    schema = MemorySchema(sections=["health"])
    cq = DeepCQ(schema=schema)
    cq.state.set("health", "[health]\nStatus: degraded\nLatency: 5000ms\n")
    result = cq.query("is anything degraded?")
    assert "degraded" in result.content
