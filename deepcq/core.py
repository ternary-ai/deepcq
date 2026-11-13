"""DeepCQ — main orchestrator."""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import replace

from .schema import MemorySchema, validate_max_chars
from .sections import split_sections
from .intent import score_sections_detailed
from .anchors import anchor_in, extract_literal_anchors
from .regex_gen import build_content_pattern, content_pattern_string
from .result import QueryResult, OptimizationMetrics, SectionInfo, MemoryGap
from .usage import SectionUsage, UsageStats, HealthReport
from .decay import Recency
from .memory.state import AppState
from .memory.session import SessionCache, History
from .memory.store import LongTermStore, ReferenceData

logger = logging.getLogger("deepcq")

# Process-lifetime count of query() fallbacks and degraded results, by kind.
_FALLBACK_COUNTS: Counter[str] = Counter()
# Process-lifetime count of memory gaps reported by query(), by kind.
_GAP_COUNTS: Counter[str] = Counter()
# (schema id, section, tier) duplicates already warned about.
_WARNED_DUPLICATES: set[tuple[int, str, str]] = set()
# (schema id, header name) pairs already warned about as unknown.
_WARNED_UNKNOWN: set[tuple[int, str]] = set()
# Longest query excerpt written to a log record (only with log_query_text=True).
_LOG_QUERY_CHARS = 80

_SECTIONED_TIERS = ("state", "notes", "reference")
_FREE_TEXT_TIERS = ("cache", "history")

# Distinguishes "max_chars not given" (use the schema's) from max_chars=None (off).
_UNSET = object()


def fallback_counts() -> dict[str, int]:
    """How often query() took a fallback or degraded path, by kind, since
    process start or the last reset_fallback_counts():

    - "no_section_scored", "no_section_present": fallback_include_all
      returned memory unfiltered by relevance.
    - "absent_only": the query matched only sections that are not in memory,
      so no section content was returned (and the include-all fallback was
      deliberately not used).
    - "budget_cut": the context budget (max_chars) left out content that
      passed the threshold (query() or locate()).
    - "duplicate_section": a section had more than one copy in one tier; only
      the first was used.
    """
    return dict(_FALLBACK_COUNTS)


def reset_fallback_counts() -> None:
    """Zero the fallback_counts() totals (e.g. per evaluation run or test)."""
    _FALLBACK_COUNTS.clear()


def gap_counts() -> dict[str, int]:
    """How often query() reported each kind of memory gap ("absent_section",
    "unmatched_anchor", "no_match") since process start or the last
    reset_gap_counts(). See QueryResult.gaps."""
    return dict(_GAP_COUNTS)


def reset_gap_counts() -> None:
    """Zero the gap_counts() totals."""
    _GAP_COUNTS.clear()


def _excerpt(query: str) -> str:
    return query if len(query) <= _LOG_QUERY_CHARS else query[:_LOG_QUERY_CHARS] + "..."


class DeepCQ:
    """
    Deep Context Query.

    Manages five memory tiers and queries them dynamically using regex
    patterns generated from the user's query. No domain knowledge is
    baked in — the schema declares section names only.

    `track_usage` (default True) keeps per-section counters on this instance
    for usage() and health_report(): counts only, keyed by declared section
    names — never query text or memory content. `log_query_text` (default
    False) adds an 80-character query excerpt (and unmatched anchors) to
    deepcq's log records, for debugging in a trusted environment; by default
    log records carry no query content.

    Example::

        from deepcq import DeepCQ, MemorySchema

        schema = MemorySchema(sections=["status", "errors", "metrics"])
        cq = DeepCQ(schema=schema)

        cq.state.set("errors",  "[errors]\\n- OOM at 02:15\\n")
        cq.state.set("metrics", "[metrics]\\nCPU: 78%\\nMemory: 4.2GB\\n")
        cq.state.set("status",  "[status]\\nAll services running\\n")

        result = cq.query("any memory or cpu spikes?")
        print(result.content)
        print(f"Reduction: {result.metrics.reduction_pct:.1f}%")
    """

    def __init__(
        self,
        schema: MemorySchema,
        store_path: str | None = None,
        reference_path: str | None = None,
        track_usage: bool = True,
        log_query_text: bool = False,
    ) -> None:
        self.schema = schema
        self.state = AppState()
        self.cache = SessionCache()
        self.history = History()
        self.notes = LongTermStore(store_path)
        self.reference = ReferenceData(reference_path)
        self.track_usage = track_usage
        self.log_query_text = log_query_text
        # locate() turns this off: the text it searches isn't stored memory.
        self._report_gaps = True
        self.reset_usage()

    # -- helpers -------------------------------------------------------------

    def _tier_texts(self) -> dict[str, str]:
        return {
            "state":     self.state.as_text(),
            "cache":     self.cache.as_text(),
            "history":   self.history.as_text(),
            "notes":     self.notes.as_text(),
            "reference": self.reference.as_text(),
        }

    def _section_pieces(self, section: str, tier_texts: dict[str, str]) -> list[str]:
        """The section's blocks as query() returns them: the first line-start
        copy in each sectioned tier, stripped. Reuses the text scoring just
        extracted (the tier's cache) when it is current."""
        pat = self.schema.extraction_pattern(section)
        schema_id = self.schema.schema_id()
        pieces = []
        for tier_name in _SECTIONED_TIERS:
            text = tier_texts[tier_name]
            if not text:
                continue
            entry = getattr(self, tier_name)._fresh_entry(schema_id, section)
            if entry is not None:
                matched = entry[0]
            else:
                m = pat.search(text)
                matched = m.group(0) if m else ""
            piece = matched.strip()
            if piece:
                pieces.append(piece)
        return pieces

    def _for_query(self, query: str) -> str:
        """The query part of a log record: empty unless log_query_text."""
        return f" for query {_excerpt(query)!r}" if self.log_query_text else ""

    def _count_fallback(self, kind: str) -> int:
        _FALLBACK_COUNTS[kind] += 1
        if self.track_usage:
            self._fallbacks[kind] += 1
        return _FALLBACK_COUNTS[kind]

    @staticmethod
    def _select(
        order: list[str],
        pieces: dict[str, list[str]],
        free: list[str],
        tier_texts: dict[str, str],
        budget: int | None,
    ) -> tuple[set[str], list[str], list[str], list[str]]:
        """Walk sections (in `order`) then free-text tiers, keeping each one
        that fits in `budget` and skipping — never truncating — those that
        don't. The cost counts the newline separators of the final join, so
        the returned content is never longer than `budget`."""
        if budget is None:
            return set(order), [], list(free), []
        used = 0
        parts = 0

        def fits(lengths: list[int]) -> bool:
            nonlocal used, parts
            if not lengths:
                return True
            cost = sum(lengths) + len(lengths) - (0 if parts else 1)
            if used + cost > budget:
                return False
            used += cost
            parts += len(lengths)
            return True

        kept: set[str] = set()
        over: list[str] = []
        for section in order:
            if fits([len(p) for p in pieces[section]]):
                kept.add(section)
            else:
                over.append(section)
        kept_tiers: list[str] = []
        over_tiers: list[str] = []
        for tier_name in free:
            if fits([len(tier_texts[tier_name].strip())]):
                kept_tiers.append(tier_name)
            else:
                over_tiers.append(tier_name)
        return kept, over, kept_tiers, over_tiers

    def _detect_gaps(
        self, query: str, tier_texts: dict[str, str], scored: list[str], absent_matched: list[str]
    ) -> list[MemoryGap]:
        gaps: list[MemoryGap] = []
        if absent_matched:
            gaps.append(MemoryGap(kind="absent_section", sections=list(absent_matched)))
        unmatched: list[str] = []
        texts = [t for t in tier_texts.values() if t]
        for anchor in extract_literal_anchors(query, self.schema.anchor_patterns):
            # Emoji/symbol anchors rarely name the information asked for.
            if anchor in unmatched or not any(c.isalnum() for c in anchor):
                continue
            if not any(anchor_in(anchor, t) for t in texts):
                unmatched.append(anchor)
        if unmatched:
            gaps.append(MemoryGap(kind="unmatched_anchor", anchors=unmatched))
        if not scored and not absent_matched:
            gaps.append(MemoryGap(kind="no_match"))
        return gaps

    # -- query -----------------------------------------------------------------

    def query(
        self,
        query: str,
        recency: dict[str, Recency] | None = None,
        now: float | None = None,
        max_chars: int | None = _UNSET,  # type: ignore[assignment]
    ) -> QueryResult:
        """
        Query all memory tiers and return only content relevant to the query.

        `recency` optionally maps a section name to a Recency (timestamp +
        decay_fn) that time-weights that section's score; `now` pins the
        reference time for that decay (defaults to the current time).

        `max_chars` caps the returned content (default: the schema's
        max_chars, None by default = no budget). Sections that pass the
        threshold are kept highest score first; one that doesn't fit is
        skipped whole and listed in `sections_over_budget`.
        """
        schema = self.schema
        budget = schema.max_chars if max_chars is _UNSET else max_chars
        validate_max_chars(budget)

        tier_texts = self._tier_texts()
        full_text = "\n".join(v for v in tier_texts.values() if v)
        original_chars = len(full_text)

        tiers = {
            "state": self.state, "cache": self.cache, "history": self.history,
            "notes": self.notes, "reference": self.reference,
        }
        explanations = score_sections_detailed(
            query, schema, tier_texts, recency=recency, now=now, tiers=tiers
        )
        intent = {section: exp.score for section, exp in explanations.items()}
        position = {section: i for i, section in enumerate(schema.sections)}

        absent = [sec for sec in schema.sections if not explanations[sec].present]
        present = [sec for sec in schema.sections if explanations[sec].present]
        scored = [sec for sec in present if intent[sec] >= schema.confidence_threshold]
        # Pinned sections are kept when present even below threshold, but
        # don't count as a match (the fallback still applies without one).
        pinned = [sec for sec in present if sec in schema.pinned and sec not in scored]

        # Headers in state that the schema doesn't declare: query() can never
        # return their content. Header names come from memory, so the log
        # names them only with log_query_text; the result always lists them.
        unknown = split_sections(tier_texts["state"], schema).unknown
        new_unknown = [n for n in unknown if (schema.schema_id(), n.lower()) not in _WARNED_UNKNOWN]
        if new_unknown:
            _WARNED_UNKNOWN.update((schema.schema_id(), n.lower()) for n in new_unknown)
            logger.warning(
                "deepcq: %d section header(s) in state are not in the schema%s; query() never "
                "returns their content (see QueryResult.unknown_sections)",
                len(new_unknown), f" {new_unknown}" if self.log_query_text else "",
            )

        # Duplicates: only the first copy of a section in a tier is used.
        duplicates = {
            sec: list(exp.duplicate_tiers) for sec, exp in explanations.items() if exp.duplicate_tiers
        }
        if duplicates:
            total = self._count_fallback("duplicate_section")
            for sec, dup_tiers in duplicates.items():
                for tier_name in dup_tiers:
                    key = (schema.schema_id(), sec, tier_name)
                    if key in _WARNED_DUPLICATES:
                        continue
                    _WARNED_DUPLICATES.add(key)
                    copies = sum(1 for _ in schema.extraction_pattern(sec).finditer(tier_texts[tier_name]))
                    logger.warning(
                        "deepcq: section %r appears %d times in tier %r; only the first copy "
                        "is used (process total duplicate_section: %d)", sec, copies, tier_name, total,
                    )

        # The query asked about sections memory doesn't hold: the right answer
        # is no section content, not everything — skip the include-all fallback.
        absent_matched = [
            sec for sec in absent
            if explanations[sec].absent_score >= schema.confidence_threshold
        ]
        if not scored and absent_matched:
            total = self._count_fallback("absent_only")
            logger.warning(
                "deepcq: query matched only section(s) not in memory %s%s; returning "
                "no section content (process total absent_only: %d)",
                absent_matched, self._for_query(query), total,
            )

        pieces = {sec: self._section_pieces(sec, tier_texts) for sec in present}
        by_score = lambda secs: sorted(secs, key=lambda s: (-intent[s], position[s]))  # noqa: E731

        fallback_used = False
        fallback_reason: str | None = None
        if not scored and not absent_matched and schema.fallback_include_all:
            fallback_used = True
            kind = "no_section_scored" if present else "no_section_present"
            fallback_reason = (
                f"no section scored >= {schema.confidence_threshold}"
                if present else "no schema section is present in memory"
            )
            if budget is not None and original_chars > budget:
                fallback_reason += f"; limited to max_chars={budget}"
            total = self._count_fallback(kind)
            logger.warning(
                "deepcq: fallback_include_all — %s%s; returning memory unfiltered by "
                "relevance (process total %s: %d)", fallback_reason, self._for_query(query), kind, total,
            )

        if fallback_used and (budget is None or original_chars <= budget):
            # 0.3.1 behaviour: everything, exactly as stored (it fits).
            content = full_text
            kept, over, over_tiers = set(present), [], []
        else:
            if fallback_used:
                order = by_score(present)
                free = [t for t in _FREE_TEXT_TIERS if tier_texts[t]]
            else:
                order = pinned + by_score(scored)
                content_pat = build_content_pattern(query)
                free = [t for t in _FREE_TEXT_TIERS if tier_texts[t] and content_pat.search(tier_texts[t])]
            kept, over, kept_tiers, over_tiers = self._select(order, pieces, free, tier_texts, budget)
            parts = [p for sec in schema.sections if sec in kept for p in pieces[sec]]
            parts += [tier_texts[t].strip() for t in _FREE_TEXT_TIERS if t in kept_tiers]
            # Joined like full_text (one newline), so the filtered content can
            # never be longer than the original: reduction_pct stays >= 0.
            content = "\n".join(p for p in parts if p)

        over = [sec for sec in schema.sections if sec in over]  # schema order
        if over or over_tiers:
            total = self._count_fallback("budget_cut")
            sizes = {sec: len("\n".join(pieces[sec])) for sec in over}
            sizes.update({t: len(tier_texts[t].strip()) for t in over_tiers})
            logger.warning(
                "deepcq: context budget max_chars=%d left out section(s) %s and tier(s) %s "
                "(sizes in chars: %s) (process total budget_cut: %d)",
                budget, over, over_tiers, sizes, total,
            )

        included = [sec for sec in schema.sections if sec in kept]
        excluded = [] if fallback_used else [
            sec for sec in present if sec not in scored and sec not in pinned
        ]

        filtered_chars = len(content)
        reduction_pct = (
            round((1.0 - filtered_chars / original_chars) * 100, 1)
            if original_chars > 0
            else 0.0
        )

        gaps = self._detect_gaps(query, tier_texts, scored, absent_matched) if self._report_gaps else []
        if gaps:
            for gap in gaps:
                _GAP_COUNTS[gap.kind] += 1
                if self.track_usage:
                    self._gaps[gap.kind] += 1
            unmatched = next((g.anchors for g in gaps if g.kind == "unmatched_anchor"), [])
            absent_names = next((g.sections for g in gaps if g.kind == "absent_section"), [])
            logger.info(
                "deepcq: memory gap(s) %s — absent sections %s, %d unmatched anchor(s)%s%s",
                [g.kind for g in gaps], absent_names, len(unmatched),
                f" {unmatched}" if self.log_query_text and unmatched else "",
                self._for_query(query),
            )

        sections: dict[str, SectionInfo] = {}
        for sec in schema.sections:
            if sec in kept:
                status = "included"
            elif sec in over:
                status = "over_budget"
            elif sec in present:
                status = "excluded"
            else:
                status = "absent"
            sections[sec] = SectionInfo(
                name=sec,
                status=status,
                score=intent[sec],
                chars=len("\n".join(pieces.get(sec, []))),
                description=schema.descriptions.get(sec),
            )

        if self.track_usage:
            self._queries += 1
            for sec in schema.sections:
                usage = self._usage[sec]
                if sec in present:
                    usage.queries += 1
                    # Relevance-oriented: "included" = passed the threshold
                    # and returned. A fallback returns sections that didn't
                    # pass, so they count as excluded.
                    if sec in scored and sec in kept:
                        usage.included += 1
                        if explanations[sec].literal_hit:
                            usage.literal_hits += 1
                    elif sec in scored:
                        usage.over_budget += 1
                    else:
                        usage.excluded += 1
                else:
                    usage.absent += 1
                    if sec in absent_matched:
                        usage.requested_absent += 1

        return QueryResult(
            content=content,
            intent=intent,
            sections_included=included,
            sections_excluded=excluded,
            content_pattern=content_pattern_string(query),
            metrics=OptimizationMetrics(
                original_chars=original_chars,
                filtered_chars=filtered_chars,
                reduction_pct=reduction_pct,
            ),
            explanations=explanations,
            sections_absent=absent,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            sections=sections,
            sections_over_budget=over,
            tiers_over_budget=over_tiers,
            max_chars=budget,
            duplicate_sections=duplicates,
            gaps=gaps,
            sections_pinned=[] if fallback_used else [sec for sec in pinned if sec in kept],
            unknown_sections=unknown,
        )

    def get_section(self, name: str, prefix: bool = True) -> str | None:
        """
        Return one stored section's text (header included), looked up
        case-insensitively by full name, then by prefix ("news" → "News
        (latest 10)"), in the state, notes and reference tiers. Finds
        sections the schema doesn't declare too. None if there's no match.

        Backs a "load an omitted section" tool for agents — see
        QueryResult.omitted_note() and section_map().
        """
        for tier in (self.state, self.notes, self.reference):
            found = split_sections(tier.as_text(), self.schema).find(name, prefix=prefix)
            if found:
                return found.text
        return None

    # -- usage and health ----------------------------------------------------------

    def usage(self) -> UsageStats:
        """A copy of this instance's usage counters. Raises RuntimeError when
        track_usage is off, rather than returning zeros that look real."""
        if not self.track_usage:
            raise RuntimeError("usage tracking is off for this DeepCQ (track_usage=False)")
        return UsageStats(
            queries=self._queries,
            sections={sec: replace(u) for sec, u in self._usage.items()},
            fallbacks=dict(self._fallbacks),
            gaps=dict(self._gaps),
        )

    def reset_usage(self) -> None:
        """Zero this instance's usage counters."""
        self._queries = 0
        self._usage = {sec: SectionUsage() for sec in self.schema.sections}
        self._fallbacks: Counter[str] = Counter()
        self._gaps: Counter[str] = Counter()

    def health_report(
        self,
        min_queries: int = 20,
        min_count: int = 1,
        max_section_chars: int | None = _UNSET,  # type: ignore[assignment]
        recency: dict[str, Recency] | None = None,
        max_age: float | None = None,
        now: float | None = None,
    ) -> HealthReport:
        """
        Read-only findings for an upkeep pass: sections never or always
        included, oversized, stale, duplicated, never present, or requested
        while absent, plus this instance's fallback and gap counts. Checks
        that couldn't run are listed in `skipped` — a check that didn't run
        never looks like a clean result.

        `min_queries`: usage checks only report sections observed at least
        this often. `min_count`: privacy — counts below it are withheld (use
        >= 5 for an instance shared by several users). `max_section_chars`
        defaults to the schema's max_chars: a section larger than the budget
        can never be returned. `stale` needs `recency` and `max_age`
        (seconds), since deepcq stores no timestamps of its own.
        """
        if min_queries < 1:
            raise ValueError("min_queries must be >= 1")
        if min_count < 1:
            raise ValueError("min_count must be >= 1")
        schema = self.schema
        report = HealthReport(queries=self._queries if self.track_usage else 0)

        if self.track_usage:
            for sec in schema.sections:
                usage = self._usage[sec]
                if usage.queries >= min_queries and usage.included == 0:
                    report.never_included.append(sec)
                if usage.queries >= min_queries and usage.included == usage.queries:
                    report.always_included.append(sec)
                if self._queries >= min_queries and usage.absent == self._queries:
                    report.never_present.append(sec)
            position = {sec: i for i, sec in enumerate(schema.sections)}
            requested = [(sec, u.requested_absent) for sec, u in self._usage.items() if u.requested_absent]
            requested.sort(key=lambda item: (-item[1], position[item[0]]))
            withheld = 0
            for sec, count in requested:
                if count >= min_count:
                    report.requested_but_absent[sec] = count
                else:
                    withheld += 1
            for source, target in ((self._fallbacks, report.fallbacks), (self._gaps, report.gaps)):
                for kind in sorted(source):
                    if source[kind] >= min_count:
                        target[kind] = source[kind]
                    else:
                        withheld += 1
            if self._queries < min_queries:
                report.skipped.append(
                    f"never/always included, never present: {self._queries} queries observed, "
                    f"need min_queries={min_queries}"
                )
            if withheld:
                report.skipped.append(f"{withheld} count(s) below min_count={min_count} withheld")
        else:
            report.skipped.append(
                "never/always included, never present, requested but absent, fallbacks, gaps: "
                "usage tracking is off"
            )

        tier_texts = self._tier_texts()
        limit = schema.max_chars if max_section_chars is _UNSET else max_section_chars
        if limit is None:
            report.skipped.append("oversized: no size limit (max_section_chars and schema max_chars are None)")
        else:
            for sec in schema.sections:
                size = len("\n".join(self._section_pieces(sec, tier_texts)))
                if size > limit:
                    report.oversized[sec] = size

        for sec in schema.sections:
            pat = schema.extraction_pattern(sec)
            dup_tiers = [
                tier_name for tier_name in _SECTIONED_TIERS
                if tier_texts[tier_name] and sum(1 for _ in pat.finditer(tier_texts[tier_name])) > 1
            ]
            if dup_tiers:
                report.duplicates[sec] = dup_tiers

        if recency is None or max_age is None:
            report.skipped.append("stale: needs recency and max_age")
        else:
            current = time.time() if now is None else now
            for sec in schema.sections:
                if sec in recency:
                    age = max(0.0, current - recency[sec].timestamp)
                    if age > max_age:
                        report.stale[sec] = round(age, 1)

        return report
