"""0.3.2: pinning, unknown sections, section parsing / lookup, omitted
notes, pre-chunked locate() and the agent reload tool."""
import logging

import pytest

from deepcq import DeepCQ, MemorySchema, locate, split_sections
from deepcq.integrations.ternary import ToolContextProvider


def _cq(**schema_kwargs):
    schema = MemorySchema(sections=["status", "errors", "audit"],
                          keywords={"audit": ["audit", "findings"]}, **schema_kwargs)
    cq = DeepCQ(schema=schema)
    cq.state.set("status", "[status]\nAll services running\n")
    cq.state.set("errors", "[errors]\n- OOM at 02:15\n")
    return cq


# --- pinning -------------------------------------------------------------------

def test_pinned_section_included_below_threshold():
    result = _cq(pinned=["status"]).query("show me the errors")
    assert result.sections_included == ["status", "errors"]
    assert result.sections_pinned == ["status"]
    assert "All services running" in result.content
    assert "status" not in result.sections_excluded


def test_pins_do_not_suppress_fallback():
    result = _cq(pinned=["status"]).query("xyzzy flurble")
    assert result.fallback_used


def test_pinned_must_be_a_declared_section():
    with pytest.raises(ValueError, match="pinned"):
        MemorySchema(sections=["a"], pinned=["b"])


def test_absent_pinned_section_is_not_reported_pinned():
    result = _cq(pinned=["audit"]).query("show me the errors")
    assert result.sections_pinned == []


# --- unknown sections ------------------------------------------------------------

def test_unknown_header_reported_and_warned_once(caplog):
    cq = _cq()
    cq.state.set("extra", "[deploys]\nv2 shipped\n")
    with caplog.at_level(logging.WARNING, logger="deepcq"):
        first = cq.query("show me the errors")
        cq.query("show me the errors again")
    assert first.unknown_sections == ["deploys"]
    warnings = [r.getMessage() for r in caplog.records if "not in the schema" in r.getMessage()]
    assert len(warnings) == 1
    # Header names come from memory: logged only with log_query_text (0.4.0).
    assert "deploys" not in warnings[0]


def test_unknown_header_named_in_log_with_log_query_text(caplog):
    cq = _cq()
    cq.log_query_text = True
    cq.state.set("extra", "[releases]\nv3 shipped\n")
    with caplog.at_level(logging.WARNING, logger="deepcq"):
        cq.query("show me the errors")
    assert any("'releases'" in r.getMessage() for r in caplog.records)


# --- split_sections / get_section ---------------------------------------------------

MD_SCHEMA = MemorySchema(sections=["News (latest 10)", "SEC"], section_header_format="## {name}")
MD_TEXT = "# Stock\nticker: X\n## News (latest 10)\nbeat\n## Sector Notes\ntech\n## SEC\n10-K filed\n"


def test_split_sections_preamble_known_and_unknown():
    split = split_sections(MD_TEXT, MD_SCHEMA)
    assert split.preamble == "# Stock\nticker: X\n"
    assert [(s.name, s.known) for s in split.sections] == [
        ("News (latest 10)", True), ("Sector Notes", False), ("SEC", True),
    ]
    assert split.unknown == ["Sector Notes"]
    assert split.sections[2].text == "## SEC\n10-K filed"


def test_split_sections_does_not_split_on_data_lines():
    schema = MemorySchema(sections=["positions"])
    split = split_sections("[positions]\nholdings:\n[AAPL, MSFT]\n[deploys]\nv2\n", schema)
    assert [s.name for s in split.sections] == ["positions", "deploys"]
    assert "[AAPL, MSFT]" in split.sections[0].text
    assert split.unknown == ["deploys"]


def test_split_sections_without_headers_is_all_preamble():
    split = split_sections("just text", MD_SCHEMA)
    assert split.preamble == "just text" and split.sections == []


def test_get_section_exact_prefix_unknown_and_miss():
    cq = DeepCQ(schema=MD_SCHEMA)
    cq.state.set("body", MD_TEXT)
    assert cq.get_section("news") == "## News (latest 10)\nbeat"
    assert cq.get_section("sec") == "## SEC\n10-K filed"  # exact beats prefix "Sector Notes"
    assert cq.get_section("sector") == "## Sector Notes\ntech"
    assert cq.get_section("news", prefix=False) is None
    assert cq.get_section("lobbying") is None


# --- omitted note ------------------------------------------------------------------

def test_omitted_note_names_excluded_sections():
    # 0.4.0: the note is section_map(only_omitted=True), plus the hint.
    result = _cq().query("show me the errors")
    assert result.omitted_note() == (
        "Not loaded (ask for any of these by name):\n- status (29 chars, not loaded)"
    )
    assert result.omitted_note("Use get_section.").endswith("not loaded)\nUse get_section.")


def test_omitted_note_empty_when_nothing_omitted():
    assert _cq().query("xyzzy flurble").omitted_note() == ""


# --- locate() with pre-chunked input ---------------------------------------------------

def test_locate_list_keeps_indices_and_exact_text():
    chunks = ["rate limiting\n\nuses a sliding window", "", "  ", "database migrations"]
    found = locate(chunks, "how does rate limiting work?")
    assert [c.index for c in found] == [0]
    assert found[0].text == chunks[0]
    found = locate(chunks, "database migrations")
    assert [c.index for c in found] == [3]


def test_locate_list_recency_uses_list_indices():
    from deepcq import Recency, exponential_decay
    chunks = ["", "cpu spike at noon"]
    fresh = locate(chunks, "cpu spike")
    old = locate(chunks, "cpu spike", recency={1: Recency(0.0, exponential_decay(60))}, now=10_000.0)
    assert [c.index for c in fresh] == [1] and old == []


# --- agent reload tool ---------------------------------------------------------------

def test_tool_context_provider_note_spec_and_handler():
    cq = _cq()
    provider = ToolContextProvider(cq)
    context = provider.get_context_with_note("show me the errors")
    assert "OOM" in context
    assert '- status (29 chars, not loaded)\nLoad one with get_section(name="<section>").' in context
    spec = provider.section_tool_spec()
    assert spec["name"] == "get_section"
    assert spec["parameters"]["required"] == ["name"]
    assert provider.get_section("stat") == "[status]\nAll services running"
    assert provider.get_section("nope").startswith("No section named 'nope'")
