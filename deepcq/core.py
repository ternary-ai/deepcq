"""DeepCQ — main orchestrator."""
from __future__ import annotations

import logging
from collections import Counter

from .schema import MemorySchema
from .intent import score_sections_detailed
from .regex_gen import build_content_pattern, content_pattern_string
from .result import QueryResult, OptimizationMetrics
from .decay import Recency
from .memory.state import AppState
from .memory.session import SessionCache, History
from .memory.store import LongTermStore, ReferenceData

logger = logging.getLogger("deepcq")

# Process-lifetime count of query() fallbacks, by reason kind.
_FALLBACK_COUNTS: Counter[str] = Counter()
# Longest query excerpt written to the fallback warning.
_LOG_QUERY_CHARS = 80


def fallback_counts() -> dict[str, int]:
    """How often query() fell back to returning everything, by kind
    ("no_section_scored", "no_section_present"), since process start or the
    last reset_fallback_counts()."""
    return dict(_FALLBACK_COUNTS)


def reset_fallback_counts() -> None:
    """Zero the fallback_counts() totals (e.g. per evaluation run or test)."""
    _FALLBACK_COUNTS.clear()


class DeepCQ:
    """
    Deep Context Query.

    Manages five memory tiers and queries them dynamically using regex
    patterns generated from the user's query. No domain knowledge is
    baked in — the schema declares section names only.

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
    ) -> None:
        self.schema = schema
        self.state = AppState()
        self.cache = SessionCache()
        self.history = History()
        self.notes = LongTermStore(store_path)
        self.reference = ReferenceData(reference_path)

    def query(
        self,
        query: str,
        recency: dict[str, Recency] | None = None,
        now: float | None = None,
    ) -> QueryResult:
        """
        Query all memory tiers and return only content relevant to the query.

        `recency` optionally maps a section name to a Recency (timestamp +
        decay_fn) that time-weights that section's score; `now` pins the
        reference time for that decay (defaults to the current time).
        """
        tier_texts: dict[str, str] = {
            "state":     self.state.as_text(),
            "cache":     self.cache.as_text(),
            "history":   self.history.as_text(),
            "notes":     self.notes.as_text(),
            "reference": self.reference.as_text(),
        }

        full_text = "\n".join(v for v in tier_texts.values() if v)
        original_chars = len(full_text)

        tiers = {
            "state": self.state, "cache": self.cache, "history": self.history,
            "notes": self.notes, "reference": self.reference,
        }
        explanations = score_sections_detailed(
            query, self.schema, tier_texts, recency=recency, now=now, tiers=tiers
        )
        intent = {section: exp.score for section, exp in explanations.items()}

        absent = [sec for sec in self.schema.sections if not explanations[sec].present]
        present = [sec for sec in self.schema.sections if explanations[sec].present]
        included = [sec for sec in present if intent[sec] >= self.schema.confidence_threshold]
        excluded = [sec for sec in present if sec not in included]

        if not included and self.schema.fallback_include_all:
            kind = "no_section_scored" if present else "no_section_present"
            reason = (
                f"no section scored >= {self.schema.confidence_threshold}"
                if present else "no schema section is present in memory"
            )
            _FALLBACK_COUNTS[kind] += 1
            excerpt = query if len(query) <= _LOG_QUERY_CHARS else query[:_LOG_QUERY_CHARS] + "..."
            logger.warning(
                "deepcq: fallback_include_all — %s for query %r; returning all content "
                "unfiltered (process total %s: %d)", reason, excerpt, kind, _FALLBACK_COUNTS[kind],
            )
            return QueryResult(
                content=full_text,
                intent=intent,
                sections_included=present,
                sections_excluded=[],
                content_pattern=content_pattern_string(query),
                metrics=OptimizationMetrics(
                    original_chars=original_chars,
                    filtered_chars=original_chars,
                    reduction_pct=0.0,
                ),
                explanations=explanations,
                sections_absent=absent,
                fallback_used=True,
                fallback_reason=reason,
            )

        parts: list[str] = []

        for section in included:
            pat = self.schema.extraction_pattern(section)
            for tier_name in ("state", "notes", "reference"):
                text = tier_texts[tier_name]
                if not text:
                    continue
                m = pat.search(text)
                if m:
                    parts.append(m.group(0).strip())

        content_pat = build_content_pattern(query)
        for tier_name in ("cache", "history"):
            text = tier_texts[tier_name]
            if text and content_pat.search(text):
                parts.append(text.strip())

        content = "\n\n".join(p for p in parts if p)
        filtered_chars = len(content)
        reduction_pct = (
            round((1.0 - filtered_chars / original_chars) * 100, 1)
            if original_chars > 0
            else 0.0
        )

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
        )
