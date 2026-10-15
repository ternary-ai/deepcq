import re
import time

from deepcq.locate import locate, Candidate
from deepcq.decay import Recency, exponential_decay


_DOC = (
    "The authentication service validates tokens on every request.\n\n"
    "The billing pipeline reconciles invoices nightly and emails a report.\n\n"
    "Rate limiting is enforced per API key using a sliding window counter.\n\n"
    "The database migration tool applies schema changes in order."
)


def test_locates_relevant_chunk():
    candidates = locate(_DOC, "how does rate limiting work?")
    assert candidates
    assert any("sliding window" in c.text for c in candidates)


def test_candidate_text_is_exact_original_chunk():
    candidates = locate(_DOC, "how does rate limiting work?")
    for c in candidates:
        assert c.text in _DOC
        assert "<<CHUNK:" not in c.text


def test_candidates_ordered_by_index():
    candidates = locate(_DOC, "tell me about tokens and invoices and schema changes")
    indices = [c.index for c in candidates]
    assert indices == sorted(indices)


def test_max_candidates_truncates_to_highest_scoring():
    candidates = locate(
        _DOC,
        "tell me about tokens and invoices and schema changes",
        max_candidates=1,
    )
    assert len(candidates) <= 1


def test_empty_text_returns_no_candidates():
    assert locate("", "anything?") == []


def test_unrelated_query_returns_no_or_low_confidence_candidates():
    candidates = locate(_DOC, "xyzzy flurble nonsense", confidence_threshold=0.9)
    assert candidates == []


def test_custom_chunk_delimiter():
    text = "alpha section about caching---beta section about logging"
    candidates = locate(text, "tell me about logging", chunk_on="---")
    assert any("logging" in c.text for c in candidates)


def test_candidate_is_dataclass_with_expected_fields():
    candidates = locate(_DOC, "how does rate limiting work?")
    c = candidates[0]
    assert isinstance(c, Candidate)
    assert isinstance(c.index, int)
    assert isinstance(c.text, str)
    assert isinstance(c.score, float)
    assert isinstance(c.matched_tokens, list)
    assert isinstance(c.matched_anchors, list)
    assert isinstance(c.literal_hit, bool)


def test_candidate_reports_matched_tokens():
    candidates = locate(_DOC, "how does rate limiting work?")
    c = next(c for c in candidates if "sliding window" in c.text)
    assert "rate" in c.matched_tokens
    assert c.literal_hit is False


def test_candidate_reports_literal_anchor_hit():
    text = "The 🚀 launch service handles deploys.\n\nThe cache warms nightly."
    candidates = locate(text, "what does 🚀 do?")
    c = next(c for c in candidates if "launch" in c.text)
    assert c.literal_hit is True
    assert "🚀" in c.matched_anchors


def test_custom_anchor_patterns_override_defaults():
    ticket_id = re.compile(r"\bTICKET-\d+\b")
    text = "Fix applied per TICKET-4471.\n\nUnrelated maintenance notes here."
    candidates = locate(text, "what about TICKET-4471?", anchor_patterns=[ticket_id])
    assert any("TICKET-4471" in c.text for c in candidates)


def test_recency_maps_by_chunk_index_and_decays_score():
    text = "Alpha topic discussion here.\n\nBeta topic discussion here."
    now = 10_000.0
    recency = {0: Recency(timestamp=now - 100.0, decay_fn=exponential_decay(100.0))}

    baseline = locate(text, "tell me about alpha and beta topics", confidence_threshold=0.01)
    damped = locate(
        text, "tell me about alpha and beta topics",
        confidence_threshold=0.01, recency=recency, now=now,
    )
    base_score = next(c.score for c in baseline if c.index == 0)
    damped_score = next(c.score for c in damped if c.index == 0)
    assert damped_score < base_score
