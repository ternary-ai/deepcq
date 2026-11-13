"""0.4.0: section descriptions, section map, context budget, duplicate-section
warning, usage statistics + health report (with privacy guarantees), memory
gap signal, and lexical-miss measurement."""
import dataclasses
import json
import logging
import os
import subprocess
import sys
import textwrap

import pytest

from deepcq import (
    DeepCQ, MemorySchema, LabeledExample, Recency, calibrate, fallback_counts,
    gap_counts, locate, reset_gap_counts,
)


def _cq(**schema_kwargs):
    schema_kwargs.setdefault("sections", ["status", "errors", "audit"])
    schema_kwargs.setdefault("keywords", {"audit": ["audit", "findings"]})
    cq = DeepCQ(schema=MemorySchema(**schema_kwargs))
    cq.state.set("status", "[status]\nAll services running\n")
    cq.state.set("errors", "[errors]\n- OOM at 02:15\n")
    return cq


def _sized(sizes: dict[str, int], **schema_kwargs) -> DeepCQ:
    """Sections whose body is `word` repeated to roughly the given size."""
    cq = DeepCQ(schema=MemorySchema(sections=list(sizes), **schema_kwargs))
    for name, size in sizes.items():
        cq.state.set(name, f"[{name}]\n" + ("revenue " * (size // 8)).strip() + "\n")
    return cq


# --- §1 section descriptions -----------------------------------------------------

@pytest.mark.parametrize("descriptions,match", [
    ({"nope": "x"}, "not in sections"),
    ({"status": "two\nlines"}, "one line"),
    ({"status": "   "}, "non-empty"),
    ({"status": "x" * 201}, "at most 200"),
])
def test_description_validation(descriptions, match):
    with pytest.raises(ValueError, match=match):
        MemorySchema(sections=["status"], descriptions=descriptions)


def test_no_descriptions_scores_unchanged():
    base = _cq().query("any errors or audit findings?").intent
    same = _cq(descriptions={}).query("any errors or audit findings?").intent
    assert base == same


def test_description_words_lift_score_and_can_be_turned_off():
    def build(score_descriptions):
        cq = DeepCQ(schema=MemorySchema(
            sections=["positions", "news"],
            descriptions={"positions": "current holdings and sector exposure"},
            score_descriptions=score_descriptions,
        ))
        cq.state.set("positions", "[positions]\nAAPL 40%\nMSFT 30%\n")
        cq.state.set("news", "[news]\nFed holds rates\n")
        return cq
    assert build(True).query("what is our sector exposure?").sections_included == ["positions"]
    off = build(False).query("what is our sector exposure?")
    assert off.intent["positions"] == 0.0 and off.fallback_used


# --- §2 section map -------------------------------------------------------------------

def test_section_map_lists_present_sections_with_status_size_and_description():
    cq = _cq(descriptions={"errors": "recent failures"})
    result = cq.query("show me the errors")
    assert result.section_map() == (
        "Memory sections:\n"
        "- status (29 chars, not loaded)\n"
        "- errors (included): recent failures"
    )
    assert result.section_map(only_omitted=True) == (
        "Not loaded (ask for any of these by name):\n- status (29 chars, not loaded)"
    )
    assert result.sections["status"].chars == len("[status]\nAll services running")
    assert result.sections["audit"].status == "absent"


def test_section_map_size_matches_content_contribution():
    result = _cq().query("show me the errors")
    assert result.sections["errors"].chars == len(result.content)


def test_section_map_over_budget_and_empty_cases():
    cq = _sized({"big": 400, "small": 40})
    result = cq.query("revenue", max_chars=100)
    assert result.sections["big"].status == "over_budget"
    assert "- big (" in result.section_map(only_omitted=True)
    empty = DeepCQ(schema=MemorySchema(sections=["a"])).query("anything")
    assert empty.section_map() == "" and empty.section_map(only_omitted=True) == ""


def test_section_map_on_fallback_has_nothing_omitted():
    result = _cq().query("xyzzy flurble")
    assert result.fallback_used
    assert result.section_map(only_omitted=True) == ""


# --- §3 context budget ------------------------------------------------------------------

def test_default_budget_is_25000_and_overridable():
    cq = _sized({"a": 30_000})
    assert cq.schema.max_chars == 25_000
    assert cq.query("revenue").sections_over_budget == ["a"]
    assert cq.query("revenue", max_chars=None).sections_included == ["a"]
    assert cq.query("revenue", max_chars=40_000).sections_included == ["a"]
    assert _sized({"a": 30_000}, max_chars=None).query("revenue").sections_included == ["a"]


@pytest.mark.parametrize("bad", [0, -5, 1.5, True])
def test_invalid_budget_raises(bad):
    with pytest.raises(ValueError):
        MemorySchema(sections=["a"], max_chars=bad)
    with pytest.raises(ValueError):
        _cq().query("errors", max_chars=bad)
    with pytest.raises(ValueError):
        locate("some text", "text", max_chars=bad)


def test_budget_larger_than_content_is_identical_to_no_budget():
    for question in ("show me the errors", "xyzzy flurble", "any audit findings?"):
        with_budget = _cq().query(question, max_chars=10_000)
        without = _cq().query(question, max_chars=None)
        assert with_budget.content == without.content
        assert with_budget.sections_included == without.sections_included
        assert with_budget.metrics == without.metrics


def test_large_high_score_section_is_skipped_and_smaller_one_kept():
    cq = _sized({"big": 400, "small": 40})
    cq.state.set("big", "[big]\n" + "revenue margin " * 30 + "\n")   # scores 1.0
    cq.state.set("small", "[small]\nrevenue notes\n")                 # scores 0.5
    result = cq.query("revenue margin", max_chars=100)
    assert result.intent["big"] > result.intent["small"]
    assert result.sections_included == ["small"]
    assert result.sections_over_budget == ["big"]
    assert "big" not in result.sections_excluded


def test_ties_keep_schema_order_and_content_order_is_unchanged():
    cq = _sized({"c": 40, "a": 40, "b": 40})
    full = cq.query("revenue", max_chars=None)
    cut = cq.query("revenue", max_chars=87)            # two 43-char sections + separator
    assert cut.sections_included == ["c", "a"]          # equal scores: schema order wins
    assert cut.content == "\n".join(full.content.split("\n")[:4])


@pytest.mark.parametrize("budget", [1, 30, 37, 38, 39, 75, 76, 77, 120])
def test_content_never_exceeds_budget(budget):
    cq = _sized({"a": 30, "b": 30, "c": 30})
    cq.history.add("user", "revenue question")
    result = cq.query("revenue", max_chars=budget)
    assert result.metrics.filtered_chars <= budget


def test_free_text_tiers_cut_after_sections():
    cq = _sized({"a": 40})
    cq.history.add("user", "revenue " * 20)
    full = cq.query("revenue", max_chars=None)
    cut = cq.query("revenue", max_chars=len(full.content) - 1)
    assert cut.sections_included == ["a"]
    assert cut.tiers_over_budget == ["history"]


def test_fallback_with_budget_takes_highest_scores_and_says_so():
    cq = DeepCQ(schema=MemorySchema(sections=["alpha", "beta"], confidence_threshold=0.9))
    cq.state.set("alpha", "[alpha]\n" + "zeta " * 40 + "\n")
    cq.state.set("beta", "[beta]\nother words\n")
    result = cq.query("zeta unknown words here", max_chars=60)
    assert result.fallback_used
    assert result.fallback_reason.endswith("; limited to max_chars=60")
    assert result.metrics.filtered_chars <= 60
    assert result.sections_included == ["beta"]
    assert result.sections_over_budget == ["alpha"]


def test_budget_cut_is_logged_and_counted(caplog):
    before = fallback_counts().get("budget_cut", 0)
    with caplog.at_level(logging.WARNING, logger="deepcq"):
        _sized({"big": 400}).query("revenue", max_chars=100)
    assert fallback_counts()["budget_cut"] == before + 1
    message = next(r.getMessage() for r in caplog.records if "context budget" in r.getMessage())
    assert "max_chars=100" in message and "'big'" in message


def test_locate_budget_alone_and_with_max_candidates():
    text = "\n\n".join([
        "revenue revenue revenue " * 5,     # 0: best score, long
        "revenue note",                     # 1
        "revenue memo",                     # 2
        "unrelated text",                   # 3
    ])
    assert [c.index for c in locate(text, "revenue", max_chars=None)] == [0, 1, 2]
    assert [c.index for c in locate(text, "revenue", max_chars=30)] == [1, 2]
    assert [c.index for c in locate(text, "revenue", max_chars=30, max_candidates=2)] == [1]
    assert locate(text, "revenue") == locate(text, "revenue", max_chars=None)   # default fits


# --- §4 duplicate-section warning ----------------------------------------------------

def test_duplicate_section_reported_warned_once_counted_per_query(caplog):
    cq = DeepCQ(schema=MemorySchema(sections=["news"]))
    cq.state.set("a", "[news]\nfirst item\n")
    cq.state.set("b", "[news]\nsecond item\n")
    before = fallback_counts().get("duplicate_section", 0)
    with caplog.at_level(logging.WARNING, logger="deepcq"):
        first = cq.query("news")
        cq.query("news again")
    assert first.duplicate_sections == {"news": ["state"]}
    assert first.content == "[news]\nfirst item"          # still the first copy
    assert fallback_counts()["duplicate_section"] == before + 2
    assert sum("appears 2 times" in r.getMessage() for r in caplog.records) <= 1


def test_same_section_in_two_tiers_or_mid_line_is_not_a_duplicate():
    cq = DeepCQ(schema=MemorySchema(sections=["news"]))
    cq.state.set("a", "[news]\nfrom state\n")
    cq.notes.append("n", "[news]\nfrom notes")
    cq.history.add("user", "about [news] and [news] again")
    assert cq.query("news").duplicate_sections == {}


def test_unchanged_tier_is_not_rescanned():
    cq = DeepCQ(schema=MemorySchema(sections=["news"]))
    cq.state.set("a", "[news]\nfirst\n")
    cq.state.set("b", "[news]\nsecond\n")
    cq.query("news")
    key = (cq.schema.schema_id(), "news")
    entry = cq.state._vocab_cache[key]
    cq.query("other news")
    assert cq.state._vocab_cache[key] is entry


# --- §6 memory gap signal ------------------------------------------------------------

def test_absent_only_question_gives_gap_and_counts_both(caplog):
    before_fb = fallback_counts().get("absent_only", 0)
    before_gap = gap_counts().get("absent_section", 0)
    with caplog.at_level(logging.INFO, logger="deepcq"):
        result = _cq().query("any audit findings?")
    assert result.content == ""
    assert [g.kind for g in result.gaps] == ["absent_section"]
    assert result.gaps[0].sections == ["audit"]
    assert fallback_counts()["absent_only"] == before_fb + 1
    assert gap_counts()["absent_section"] == before_gap + 1
    assert any(r.levelno == logging.INFO and "memory gap" in r.getMessage() for r in caplog.records)


def test_present_and_absent_sections_together():
    result = _cq().query("audit findings and errors?")
    assert "errors" in result.sections_included
    assert result.gaps[0].kind == "absent_section" and result.gaps[0].sections == ["audit"]


def test_unmatched_anchor_uses_exact_figure_matching():
    cq = DeepCQ(schema=MemorySchema(sections=["metrics"]))
    cq.state.set("metrics", "[metrics]\nlatency 100ms\n")
    result = cq.query("was latency 10 ms, per the 10-K?")
    gap = next(g for g in result.gaps if g.kind == "unmatched_anchor")
    assert gap.anchors == ["10-K", "10"]          # "10" isn't found in "100ms"
    cq.history.add("user", "the 10-K was filed")
    result = cq.query("what did the 10-K say about latency?")
    assert all(g.kind != "unmatched_anchor" for g in result.gaps)


def test_no_match_note_wording_with_and_without_fallback():
    on = _cq().query("xyzzy flurble")
    off = _cq(fallback_include_all=False).query("xyzzy flurble")
    assert [g.kind for g in on.gaps] == ["no_match"]
    assert on.gap_note() == "Nothing in memory matched this question; the memory shown is unfiltered."
    assert off.gap_note() == "Nothing in memory matched this question."


def test_gap_note_orders_kinds_and_uses_descriptions():
    result = _cq(descriptions={"audit": "internal audit findings"}).query('audit findings on the "10-K"?')
    assert [g.kind for g in result.gaps] == ["absent_section", "unmatched_anchor"]
    assert result.gap_note() == (
        'Not in memory: audit (internal audit findings). No stored memory mentions "10-K".'
    )
    plain = _cq().query("any audit findings?")
    assert plain.gap_note() == "Not in memory: audit."
    assert _cq().query("show me the errors").gap_note() == ""


def test_reset_gap_counts_and_locate_reports_no_gaps():
    _cq().query("any audit findings?")
    reset_gap_counts()
    assert gap_counts() == {}
    locate("nothing relevant here", "xyzzy 10-K")
    assert gap_counts() == {}


# --- §5 usage statistics and health report ------------------------------------------

def test_usage_counters_per_status():
    cq = _sized({"big": 400, "small": 40, "other": 40})
    cq.state.set("other", "[other]\nunrelated words\n")
    cq.query("revenue", max_chars=100)         # big over budget, small included, other excluded
    cq.query("revenue 2024")                   # all fit; "2024" unmatched anchor
    cq.query("other words")
    usage = cq.usage()
    assert usage.queries == 3
    assert dataclasses.asdict(usage.sections["big"]) == dict(
        queries=3, included=1, excluded=1, over_budget=1, absent=0, requested_absent=0, literal_hits=0)
    assert usage.sections["other"].included == 1
    assert usage.fallbacks["budget_cut"] == 1
    assert usage.gaps["unmatched_anchor"] == 1


def test_literal_hits_and_requested_absent_counted():
    cq = _cq()
    cq.query("what happened at 02:15?")
    cq.query("any audit findings?")
    usage = cq.usage()
    assert usage.sections["errors"].literal_hits == 1
    assert usage.sections["audit"].requested_absent == 1
    assert usage.sections["audit"].absent == 2


def test_reset_usage_and_tracking_off():
    cq = _cq()
    cq.query("errors")
    cq.reset_usage()
    assert cq.usage().queries == 0
    off = DeepCQ(schema=MemorySchema(sections=["a"]), track_usage=False)
    off.query("anything")
    with pytest.raises(RuntimeError):
        off.usage()
    report = off.health_report()
    assert any("usage tracking is off" in reason for reason in report.skipped)


def test_health_report_checks():
    cq = DeepCQ(schema=MemorySchema(sections=["hot", "cold", "ghost", "big"], max_chars=500,
                                    keywords={"ghost": ["ghost"]}))
    cq.state.set("hot", "[hot]\nrevenue revenue\n")
    cq.state.set("cold", "[cold]\nunrelated archive\n")
    cq.state.set("big", "[big]\n" + "x " * 400 + "\n")
    cq.state.set("dup", "[hot]\nsecond copy\n")
    for _ in range(3):
        cq.query("revenue")
    cq.query("ghost stories")
    report = cq.health_report(min_queries=3, recency={"cold": Recency(timestamp=0.0)},
                              max_age=86400, now=10 * 86400)
    assert report.never_included == ["cold", "big"]
    assert report.always_included == []           # hot missed the ghost query
    assert report.oversized == {"big": len("[big]\n" + ("x " * 400).strip())}
    assert report.duplicates == {"hot": ["state"]}
    assert report.never_present == ["ghost"]      # absent in all 4 queries
    assert report.requested_but_absent == {"ghost": 1}
    assert report.stale == {"cold": 864000.0}
    assert "stale" not in " ".join(report.skipped)
    assert "never present: ghost" in report.summary()
    assert "requested but absent: ghost (1x)" in report.summary()


def test_health_report_never_present_and_skips():
    cq = DeepCQ(schema=MemorySchema(sections=["a", "b"], max_chars=None))
    cq.state.set("a", "[a]\ntext\n")
    for _ in range(2):
        cq.query("text")
    report = cq.health_report(min_queries=2)
    assert report.never_present == ["b"]
    assert any(reason.startswith("oversized") for reason in report.skipped)
    assert any(reason.startswith("stale") for reason in report.skipped)
    few = cq.health_report(min_queries=50)
    assert any("need min_queries=50" in reason for reason in few.skipped)


def test_health_report_is_read_only():
    cq = _cq()
    cq.query("errors")
    before = cq.usage()
    versions = cq.state.version
    cq.health_report()
    assert cq.usage() == before and cq.state.version == versions


def test_requested_but_absent_ordering_and_min_count():
    cq = DeepCQ(schema=MemorySchema(sections=["a", "audit", "medical"],
                                    keywords={"audit": ["audit"], "medical": ["medical"]}))
    cq.state.set("a", "[a]\ntext\n")
    cq.query("medical")
    cq.query("audit")
    cq.query("audit")
    report = cq.health_report()
    assert list(report.requested_but_absent.items()) == [("audit", 2), ("medical", 1)]
    private = cq.health_report(min_count=2)
    assert private.requested_but_absent == {"audit": 2}
    assert any("withheld" in reason for reason in private.skipped)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)
    elif dataclasses.is_dataclass(value):
        yield from _strings(dataclasses.asdict(value))


PRIVATE_QUESTIONS = [
    'what about "Jane Doe" and account 4417-1234?',
    "is ACCT_99812 over 15% or -2.5?",
    "audit findings for patient 778899",
]


def test_privacy_usage_and_report_hold_no_query_or_memory_strings():
    cq = _cq(descriptions={"errors": "recent failures"})
    cq.state.set("secret", "[status]\nJane Doe holds 4417-1234\n")
    for q in PRIVATE_QUESTIONS:
        cq.query(q)
    allowed = set(cq.schema.sections)
    found = set(_strings(cq.usage())) | set(_strings(cq.health_report(min_queries=1)))
    summary = cq.health_report(min_queries=1).summary()
    secrets = ["Jane", "4417", "ACCT", "15%", "2.5", "778899", "patient", "holds"]
    for secret in secrets:
        assert not any(secret in s for s in found), secret
        assert secret not in summary, secret
    keys = {k for s in found for k in [s]} - allowed
    assert all(not any(ch.isdigit() for ch in k) for k in keys)


def test_privacy_instances_share_nothing():
    a, b = _cq(), _cq()
    a.query("errors")
    assert b.usage().queries == 0


def test_privacy_logs_carry_no_query_content(caplog):
    def run(cq):
        cq.state.set("dup", "[errors]\nsecond copy\n")             # duplicate path
        cq.query('any audit findings for "Jane Doe"?')             # absent_only + gaps
        cq.query("xyzzy 4417-1234")                                # fallback + gaps
        cq.query("errors", max_chars=5)                            # budget cut
        cq.state.set("x", "[patient-778899]\nnotes\n")
        cq.query("errors")                                         # unknown header
    with caplog.at_level(logging.DEBUG, logger="deepcq"):
        run(_cq())
    text = "\n".join(r.getMessage() for r in caplog.records)
    for fragment in ("Jane", "4417", "xyzzy", "findings for", "patient"):
        assert fragment not in text, fragment
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="deepcq"):
        run(DeepCQ(schema=_cq().schema, log_query_text=True))
    assert "xyzzy 4417-1234" in "\n".join(r.getMessage() for r in caplog.records)


# --- §7 recall beyond shared words ---------------------------------------------------

def test_calibrate_reports_lexical_misses():
    def build():
        cq = DeepCQ(schema=MemorySchema(sections=["positions", "news"]))
        cq.state.set("positions", "[positions]\nAAPL 40%\nMSFT 30%\n")
        cq.state.set("news", "[news]\nFed holds rates\n")
        return cq
    report = calibrate([
        LabeledExample(build, "are we overexposed to tech?", {"positions"}),   # no shared word
        LabeledExample(build, "latest news on rates", {"news"}),
    ])
    assert report.lexical_misses == 1
    assert report.lexical_miss_examples == [(0, "positions")]
    assert "lexical misses: 1 of 2 expected sections (50%)" in report.summary()


def test_results_are_deterministic_across_processes():
    script = textwrap.dedent('''
        import json, logging
        logging.disable(logging.CRITICAL)
        from dataclasses import asdict
        from deepcq import DeepCQ, MemorySchema
        cq = DeepCQ(schema=MemorySchema(
            sections=["status", "errors", "metrics", "audit"],
            keywords={"audit": ["audit"]},
            descriptions={"metrics": "cpu and memory readings"}, max_chars=60))
        cq.state.set("errors", "[errors]\\n- OOM at 02:15\\n- Timeout 08:44\\n")
        cq.state.set("metrics", "[metrics]\\nCPU 78%\\nMemory 4.2GB\\n")
        cq.state.set("status", "[status]\\nAll services running\\n")
        cq.history.add("user", "cpu spike at 02:15?")
        out = []
        for q in ["cpu or memory spikes at 02:15?", "audit on the 10-K?", "xyzzy", "errors"]:
            r = cq.query(q)
            d = asdict(r)
            d["map"], d["note"] = r.section_map(), r.gap_note()
            out.append(d)
        print(json.dumps(out, sort_keys=True))
    ''')
    outputs = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                              env=env, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        outputs.add(proc.stdout)
    assert len(outputs) == 1
    assert json.loads(outputs.pop())


def test_package_locate_is_the_function_even_after_submodule_import():
    script = (
        "import deepcq.locate\n"
        "from deepcq import locate\n"
        "import deepcq\n"
        "assert callable(locate) and callable(deepcq.locate)\n"
        "print(locate('revenue grew\\n\\nweather is fine', 'revenue')[0].index)\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          cwd=os.path.dirname(os.path.dirname(__file__)))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0"


# --- merged 0.3.2 primitives with the 0.4.0 budget ------------------------------------

def test_pinned_sections_go_first_within_the_budget():
    cq = _cq(pinned=["status"])
    full = cq.query("show me the errors", max_chars=None)
    assert full.sections_included == ["status", "errors"] and full.sections_pinned == ["status"]
    tight = cq.query("show me the errors", max_chars=len("[status]\nAll services running"))
    assert tight.sections_included == ["status"]
    assert tight.sections_over_budget == ["errors"]


def test_pinned_section_larger_than_budget_is_cut_and_reported():
    cq = _cq(pinned=["status"])
    result = cq.query("show me the errors", max_chars=len("[errors]\n- OOM at 02:15"))
    assert result.sections_included == ["errors"]
    assert result.sections_pinned == []
    assert result.sections_over_budget == ["status"]
    assert result.sections["status"].status == "over_budget"


def test_locate_list_input_with_budget_keeps_positions():
    chunks = ["revenue " * 10, "", "revenue memo", "weather"]
    assert [c.index for c in locate(chunks, "revenue", max_chars=None)] == [0, 2]
    found = locate(chunks, "revenue", max_chars=20)
    assert [c.index for c in found] == [2]
    assert found[0].text == chunks[2]
