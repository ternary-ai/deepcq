import re

from deepcq.anchors import (
    DEFAULT_ANCHOR_PATTERNS,
    extract_literal_anchors,
)


def test_extracts_emoji_run():
    anchors = extract_literal_anchors("what does 🚀 mean here?")
    assert "🚀" in anchors


def test_extracts_multiple_emoji_as_one_run():
    anchors = extract_literal_anchors("status is 🟢✅ today")
    assert "🟢✅" in anchors


def test_extracts_double_quoted_phrase():
    anchors = extract_literal_anchors('find the section about "rate limiting"')
    assert "rate limiting" in anchors


def test_extracts_single_quoted_phrase():
    anchors = extract_literal_anchors("look for 'circuit breaker' config")
    assert "circuit breaker" in anchors


def test_extracts_curly_quoted_phrase():
    anchors = extract_literal_anchors("what about “graceful shutdown”?")
    assert "graceful shutdown" in anchors


def test_no_anchors_in_plain_query():
    assert extract_literal_anchors("any cpu or memory issues?") == []


def test_no_duplicate_anchors():
    anchors = extract_literal_anchors('"retry" logic and "retry" budget 🚀 twice 🚀')
    assert anchors.count("retry") == 1
    assert anchors.count("🚀") == 1


def test_extracts_version_string():
    anchors = extract_literal_anchors("upgrade to v2.3.1 before rollout")
    assert "v2.3.1" in anchors


def test_extracts_percentage():
    anchors = extract_literal_anchors("error rate hit 42% overnight")
    assert "42%" in anchors


def test_extracts_identifier_token():
    anchors = extract_literal_anchors("enable circuit-breaker-v2 for this service")
    assert "circuit-breaker-v2" in anchors


def test_extracts_multidigit_number():
    anchors = extract_literal_anchors("saw 42 retries in the last hour")
    assert "42" in anchors


def test_single_digit_number_not_anchored():
    anchors = extract_literal_anchors("only 1 retry so far")
    assert "1" not in anchors


def test_percentage_not_double_counted_as_number():
    anchors = extract_literal_anchors("latency improved by 15%")
    assert anchors.count("15%") == 1
    assert "15" not in anchors


def test_custom_patterns_override_defaults():
    ticket_id = re.compile(r"\bTICKET-\d+\b")
    anchors = extract_literal_anchors("see TICKET-4471 for details", [ticket_id])
    assert anchors == ["TICKET-4471"]


def test_custom_patterns_can_extend_defaults():
    ticket_id = re.compile(r"\bTICKET-\d+\b")
    anchors = extract_literal_anchors(
        'follow up on "rate limiting" in TICKET-4471',
        [*DEFAULT_ANCHOR_PATTERNS, ticket_id],
    )
    assert "rate limiting" in anchors
    assert "TICKET-4471" in anchors
