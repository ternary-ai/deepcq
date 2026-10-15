import pytest

import deepcq.intent as intent_mod
from deepcq import DeepCQ, MemorySchema
from deepcq.memory.session import SessionCache


@pytest.fixture
def cq():
    instance = DeepCQ(schema=MemorySchema(sections=["status", "errors", "metrics"]))
    instance.state.set("status", "[status]\nAll services running\n")
    instance.state.set("errors", "[errors]\n- OOM at 02:15\n")
    instance.state.set("metrics", "[metrics]\nCPU: 78%\nMemory: 4.2GB\n")
    instance.notes.append("errors", "[errors]\nrecurring OOM on worker-3")
    return instance


@pytest.fixture
def tokenize_calls(monkeypatch):
    calls: list[str] = []
    original = intent_mod.tokenize

    def counting(text, include_stems=True):
        calls.append(text)
        return original(text, include_stems=include_stems)

    monkeypatch.setattr(intent_mod, "tokenize", counting)
    return calls


def _stored_text_calls(calls):
    # Tokenize calls on extracted section text (starts with a header);
    # excludes the per-query query/name/keyword tokenization.
    return [c for c in calls if c.startswith("[")]


def test_unchanged_memory_is_not_retokenized(cq, tokenize_calls):
    cq.query("any memory issues?")
    first = len(_stored_text_calls(tokenize_calls))
    assert first > 0
    for _ in range(10):
        cq.query("any memory issues?")
        cq.query("what errors happened?")
    assert len(_stored_text_calls(tokenize_calls)) == first


def test_cached_results_identical_to_uncached(cq):
    from deepcq.intent import score_sections_detailed
    for q in ["any memory issues?", "recurring OOM", "status?"]:
        cached = cq.query(q)
        cached_again = cq.query(q)
        texts = {"state": cq.state.as_text(), "notes": cq.notes.as_text()}
        uncached = score_sections_detailed(q, cq.schema, texts)
        assert cached.explanations == cached_again.explanations == uncached


def test_invalidated_after_state_set(cq, tokenize_calls):
    assert cq.query("disk quota exceeded").intent["metrics"] == 0.0
    cq.state.set("metrics", "[metrics]\ndisk quota exceeded on /var\n")
    result = cq.query("disk quota exceeded")
    assert "metrics" in result.sections_included
    assert result.intent["metrics"] == 1.0


@pytest.mark.parametrize("mutate", [
    lambda cq: cq.state.remove("errors"),
    lambda cq: cq.state.clear(),
    lambda cq: cq.notes.append("errors", "x"),
    lambda cq: cq.cache.put("k", "v"),
    lambda cq: cq.cache.clear(),
    lambda cq: cq.history.add("user", "hi"),
    lambda cq: cq.history.clear(),
    lambda cq: cq.reference.set("k", "v"),
])
def test_every_mutator_bumps_version(cq, mutate):
    before = {t: getattr(cq, t).version for t in ("state", "cache", "history", "notes", "reference")}
    mutate(cq)
    after = {t: getattr(cq, t).version for t in before}
    assert sum(after[t] - before[t] for t in before) == 1


def test_removed_section_not_served_from_cache(cq):
    assert cq.query("OOM").intent["errors"] == 1.0
    cq.state.remove("errors")
    cq.notes = type(cq.notes)()
    assert cq.query("OOM").intent["errors"] < 1.0


def test_session_cache_identical_put_does_not_bump_or_invalidate(tokenize_calls):
    cq = DeepCQ(schema=MemorySchema(sections=["db"]))
    cq.cache.put("q1", "[db]\npostgres latency 50ms")
    cq.query("postgres latency")
    version = cq.cache.version
    calls = len(_stored_text_calls(tokenize_calls))
    cq.cache.put("q1", "[db]\npostgres latency 50ms")
    assert cq.cache.version == version
    cq.query("postgres latency")
    assert len(_stored_text_calls(tokenize_calls)) == calls


def test_session_cache_changed_put_bumps():
    cache = SessionCache()
    cache.put("k", "a")
    v = cache.version
    cache.put("k", "b")
    assert cache.version == v + 1
