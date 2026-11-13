"""Regression tests for the 0.4.1 bug fixes."""
import json
import logging

import pytest

from deepcq import DeepCQ, MemorySchema, fallback_counts, reset_fallback_counts
from deepcq.anchors import anchor_in, extract_literal_anchors
from deepcq.memory.store import LongTermStore, ReferenceData
from deepcq.tokens import content_words, tokenize
from deepcq.integrations.ternary import LangChainMemoryAdapter


def _cq(**sections):
    cq = DeepCQ(MemorySchema(sections=list(sections)))
    for name, body in sections.items():
        cq.state.set(name, f"[{name}]\n{body}")
    return cq


# --- apostrophes are not quotes -------------------------------------------------

def test_apostrophes_do_not_form_a_quoted_anchor():
    assert extract_literal_anchors("what's the user's plan and what isn't done") == []


def test_real_single_quotes_still_anchor():
    assert extract_literal_anchors("status of 'blue green' deploy") == ["blue green"]
    assert extract_literal_anchors("it's the 'blue green' deploy") == ["blue green"]


def test_apostrophes_report_no_false_gap():
    cq = _cq(plan="Ship v2 next week", profile="Name: Ada")
    assert cq.query("what's the user's plan?").gap_note() == ""


# --- anchors match regardless of case -------------------------------------------

def test_anchor_matching_ignores_case():
    assert anchor_in("10-k", "Form 10-K filed")
    assert anchor_in("form 10-k", "Form 10-K filed")
    assert not anchor_in("10-k", "Form 10-Q filed")


def test_lowercase_identifier_finds_section_without_false_gap():
    cq = _cq(filings="Form 10-K filed 2026-02-01", ops="nothing")
    result = cq.query('status of "form 10-k"')
    assert result.intent["filings"] == 1.0
    assert result.gaps == []
    assert cq.query("when was the 10-k filed?").gap_note() == ""


# --- grouped numbers stay whole -------------------------------------------------

@pytest.mark.parametrize("query, anchor", [
    ("is 1,250,000 right?", "1,250,000"),
    ("revenue 1,250.50 USD", "1,250.50"),
    ("Marge 3,5%", "3,5%"),
    ("change of -1,200", "-1,200"),
])
def test_grouped_numbers_are_one_anchor(query, anchor):
    assert extract_literal_anchors(query) == [anchor]


@pytest.mark.parametrize("query, anchors", [
    ("scores 10,20,30", ["10", "20", "30"]),
    ("ids 10,20,30,40", ["10", "20", "30", "40"]),
    ("1,250,000 and 3,5%", ["1,250,000", "3,5%"]),
])
def test_comma_lists_split_into_items(query, anchors):
    assert extract_literal_anchors(query) == anchors


def test_grouped_number_matches_only_the_same_figure():
    assert anchor_in("1,250,000", "revenue: 1,250,000 USD")
    assert not anchor_in("1,250,000", "revenue: 11,250,000 USD")
    assert not anchor_in("250,000", "revenue: 1,250,000 USD")
    assert anchor_in("3,5%", "Marge: 3,5%")


def test_grouped_number_finds_section_without_fallback():
    cq = _cq(fin="revenue: 1,250,000 USD", ops="all good")
    result = cq.query("is 1,250,000 right?")
    assert result.sections_included == ["fin"]
    assert not result.fallback_used
    assert result.gaps == []


# --- non-ASCII words stay whole -------------------------------------------------

def test_tokenizer_keeps_non_ascii_words_whole():
    assert tokenize("Größe der Übersicht", include_stems=False) == ["größe", "der", "übersicht"]
    assert content_words("Kündigungen offen") == ["kündigungen", "offen"]
    assert tokenize("user_id v2", include_stems=False) == ["user", "id", "v2"]


def test_non_ascii_section_scores_on_the_whole_word():
    cq = _cq(**{"kündigungen": "3 offen", "status": "ok"})
    explanation = cq.query("Kündigungen?").explanations["kündigungen"]
    assert "kündigungen" in explanation.matched_tokens
    assert explanation.score == 1.0


# --- corrupt store files raise instead of being wiped ----------------------------

@pytest.mark.parametrize("content", ['{"a": ["important"],', '["not", "an", "object"]'])
def test_corrupt_notes_file_raises_and_is_kept(tmp_path, content):
    path = tmp_path / "notes.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="notes.json"):
        LongTermStore(path)
    assert path.read_text() == content


def test_corrupt_reference_file_raises_and_is_kept(tmp_path):
    path = tmp_path / "ref.json"
    path.write_text("{broken")
    ref = ReferenceData(path)
    with pytest.raises(ValueError, match="ref.json"):
        ref.set("k", "v")
    assert path.read_text() == "{broken"


def test_valid_store_files_still_load(tmp_path):
    path = tmp_path / "notes.json"
    path.write_text(json.dumps({"a": ["x"]}))
    assert LongTermStore(path).get("a") == ["x"]


# --- pinned sections in usage and health ----------------------------------------

def test_pinned_sections_counted_as_pinned_not_never_included():
    schema = MemorySchema(sections=["rules", "errors"], pinned=["rules"])
    cq = DeepCQ(schema)
    cq.state.set("rules", "[rules]\nbe polite")
    cq.state.set("errors", "[errors]\nOOM crash")
    for _ in range(20):
        assert "rules" in cq.query("any OOM crash errors?").sections_included
    usage = cq.usage().sections["rules"]
    assert (usage.pinned, usage.included, usage.excluded) == (20, 0, 0)
    assert "rules" not in cq.health_report().never_included


def test_pinned_section_cut_by_budget_counts_as_over_budget():
    schema = MemorySchema(sections=["rules", "errors"], pinned=["rules"], max_chars=20)
    cq = DeepCQ(schema)
    cq.state.set("rules", "[rules]\n" + "be polite " * 10)
    cq.state.set("errors", "[errors]\nOOM")
    cq.query("OOM errors")
    usage = cq.usage().sections["rules"]
    assert (usage.pinned, usage.over_budget) == (0, 1)


# --- LangChain adapter with no input is loud ------------------------------------

def test_langchain_adapter_without_input_logs_and_counts(caplog):
    reset_fallback_counts()
    cq = DeepCQ(MemorySchema(sections=["a"]))
    memory = LangChainMemoryAdapter(cq)
    memory.save_context({"input": "hi"}, {"output": "hello"})
    with caplog.at_level(logging.WARNING, logger="deepcq"):
        out = memory.load_memory_variables({})
    assert out["history"] == cq.history.as_text()
    assert "no input" in caplog.text
    assert fallback_counts()["no_query"] == 1
    assert cq.usage().fallbacks["no_query"] == 1
