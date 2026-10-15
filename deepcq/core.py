"""DeepCQ — main orchestrator."""
from __future__ import annotations

from .schema import MemorySchema
from .intent import score_sections
from .regex_gen import build_content_pattern, content_pattern_string
from .result import QueryResult, OptimizationMetrics
from .memory.state import AppState
from .memory.session import SessionCache, History
from .memory.store import LongTermStore, ReferenceData


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

    def query(self, query: str) -> QueryResult:
        """
        Query all memory tiers and return only content relevant to the query.
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

        intent = score_sections(query, self.schema, tier_texts)

        included: list[str] = []
        excluded: list[str] = []
        for section in self.schema.sections:
            if intent.get(section, 0.0) >= self.schema.confidence_threshold:
                included.append(section)
            else:
                excluded.append(section)

        if not included and self.schema.fallback_include_all:
            return QueryResult(
                content=full_text,
                intent=intent,
                sections_included=list(self.schema.sections),
                sections_excluded=[],
                content_pattern=content_pattern_string(query),
                metrics=OptimizationMetrics(
                    original_chars=original_chars,
                    filtered_chars=original_chars,
                    reduction_pct=0.0,
                ),
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
        )
