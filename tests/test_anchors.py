from deepcq.anchors import extract_literal_anchors


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
