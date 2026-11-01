"""0.3.1: exact-figure anchors, header boundaries, absent sections and loud
fallbacks."""
import logging

import pytest

from deepcq import DeepCQ, MemorySchema, fallback_counts, reset_fallback_counts
from deepcq.anchors import anchor_in, extract_literal_anchors


# --- anchors match the exact figure / whole word -------------------------------

@pytest.mark.parametrize("anchor,text,expected", [
    ("10", "latency 100ms", False),
    ("10", "eps 2.10", False),
    ("10", "fiscal 2010", False),
    ("10", "1,050.10 total", False),
    ("10", "up 10.5 points", False),
    ("10", "the 10th row", False),
    ("10", "the 10-K filing", False),
    ("10", "a 10-year bond", False),
    ("10", "top 10 holders", True),
    ("10", "costs $10.", True),
    ("10", "News (latest 10)", True),
    ("10", "a 10-15% band", True),
    ("15%", "up 115% yoy", False),
    ("15%", "up 15.5% yoy", False),
    ("15%", "down -15% yoy", False),
    ("15%", "down −15% yoy", False),
    ("15%", "up 15% yoy", True),
    ("15%", "a 10-15% band", True),
    ("-15%", "down -15% yoy", True),
    ("-15%", "a 10-15% band", False),
    ("2.5", "fell -2.5 points", False),
    ("2.5", "rose 12.5 points", False),
    ("2.5", "rose 2.55 points", False),
    ("1,000", "paid 11,000", False),
    ("2024", "in Q3-2024", True),
    ("15", "costs $15", True),
    ("10-K", "the 10-K filing", True),
    ("10-K", "the 10-KA filing", False),
    ("Vision Pro", "Vision Pro flop", True),
    ("Vision Pro", "Vision Prof", False),
    ("\U0001F680", "launch\U0001F680now", True),
])
def test_anchor_in_matches_exact_figure_only(anchor, text, expected):
    assert anchor_in(anchor, text) is expected


@pytest.mark.parametrize("query,anchors", [
    ("was it -15% or 15%?", ["-15%", "15%"]),
    ("a 10-15% band", ["15%", "10"]),
    ("fell -2.5 points", ["-2.5"]),
])
def test_minus_sign_is_kept_on_extracted_numbers(query, anchors):
    assert extract_literal_anchors(query) == anchors


def test_number_inside_longer_number_does_not_force_inclusion():
    cq = DeepCQ(schema=MemorySchema(sections=["metrics", "status"]))
    cq.state.set("metrics", "[metrics]\nlatency 100ms\n")
    cq.state.set("status", "[status]\nok\n")
    exp = cq.query("is 10 too slow?").explanations["metrics"]
    assert exp.matched_anchors == []
    assert not exp.literal_hit


# --- header boundary -----------------------------------------------------------

def test_header_ending_in_word_does_not_match_longer_header():
    schema = MemorySchema(sections=["SEC"], section_header_format="## {name}")
    assert schema.extraction_pattern("SEC").search("## Sector\nTech\n") is None
    assert schema.extraction_pattern("SEC").search("## SEC\n10-K\n") is not None


# --- absent sections -----------------------------------------------------------

def _cq():
    schema = MemorySchema(sections=["status", "errors", "audit"],
                          keywords={"audit": ["audit", "findings"]})
    cq = DeepCQ(schema=schema)
    cq.state.set("status", "[status]\nAll services running\n")
    cq.state.set("errors", "[errors]\n- OOM at 02:15\n")
    return cq


def test_absent_section_scores_zero_even_with_matching_keywords():
    result = _cq().query("any audit findings or errors?")
    assert result.intent["audit"] == 0.0
    assert result.explanations["audit"].present is False
    assert result.sections_absent == ["audit"]
    assert "audit" not in result.sections_included
    assert "audit" not in result.sections_excluded


# --- loud fallback -------------------------------------------------------------

def test_fallback_is_flagged_logged_and_counted(caplog):
    before = fallback_counts().get("no_section_scored", 0)
    with caplog.at_level(logging.WARNING, logger="deepcq"):
        result = _cq().query("xyzzy flurble")
    assert result.fallback_used
    assert "no section scored" in result.fallback_reason
    assert result.sections_included == ["status", "errors"]  # present ones only
    assert fallback_counts()["no_section_scored"] == before + 1
    assert any("fallback_include_all" in r.message for r in caplog.records)


def test_fallback_with_empty_memory_has_its_own_kind():
    before = fallback_counts().get("no_section_present", 0)
    result = DeepCQ(schema=MemorySchema(sections=["a"])).query("anything")
    assert result.fallback_used
    assert fallback_counts()["no_section_present"] == before + 1


def test_fallback_warning_truncates_long_query(caplog):
    with caplog.at_level(logging.WARNING, logger="deepcq"):
        _cq().query("xyzzy " * 100)
    message = next(r.getMessage() for r in caplog.records if "fallback_include_all" in r.message)
    assert "xyzzy " * 20 not in message
    assert "..." in message


def test_reset_fallback_counts():
    _cq().query("xyzzy flurble")
    reset_fallback_counts()
    assert fallback_counts() == {}


def test_normal_result_is_not_flagged():
    result = _cq().query("show me the errors")
    assert not result.fallback_used and result.fallback_reason is None
