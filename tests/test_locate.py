from deepcq.locate import locate, Candidate


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
