"""Usage statistics and the memory health report.

Privacy contract (tested): these structures hold only integers and sizes,
keyed by section names declared in the schema and by fixed kind names. They
never hold query text, tokens, phrases, anchors, memory content, header
names found in memory, timestamps or caller identity. Nothing is kept per
query, nothing leaves the DeepCQ instance, and nothing here is logged.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SectionUsage:
    """How one declared section fared across this instance's queries."""
    queries: int = 0           # queries run while the section was present
    included: int = 0
    excluded: int = 0          # present, below threshold
    over_budget: int = 0       # passed the threshold (or pinned), cut by max_chars
    pinned: int = 0            # below threshold, included because it is pinned
    absent: int = 0            # queries run while the section was absent
    requested_absent: int = 0  # absent, but the query matched its name/keywords/description
    literal_hits: int = 0      # included, with an exact-figure/identifier anchor hit


@dataclass
class UsageStats:
    """A copy of one DeepCQ instance's counters (see DeepCQ.usage())."""
    queries: int
    sections: dict[str, SectionUsage]
    fallbacks: dict[str, int]  # this instance's fallback kinds (same keys as fallback_counts())
    gaps: dict[str, int]       # this instance's gap kinds (same keys as gap_counts())


@dataclass
class HealthReport:
    """Read-only findings for an upkeep pass (see DeepCQ.health_report())."""
    queries: int
    never_included: list[str] = field(default_factory=list)
    always_included: list[str] = field(default_factory=list)
    oversized: dict[str, int] = field(default_factory=dict)       # section -> chars
    stale: dict[str, float] = field(default_factory=dict)         # section -> age in seconds
    duplicates: dict[str, list[str]] = field(default_factory=dict)
    never_present: list[str] = field(default_factory=list)
    requested_but_absent: dict[str, int] = field(default_factory=dict)  # most-requested first
    fallbacks: dict[str, int] = field(default_factory=dict)
    gaps: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)              # checks not run, and why

    def summary(self) -> str:
        """Plain-text report, one finding per line: section names, kind
        names and numbers only."""
        lines = [f"queries observed: {self.queries}"]
        if self.never_included:
            lines.append(f"never included: {', '.join(self.never_included)}")
        if self.always_included:
            lines.append(f"always included (too broad?): {', '.join(self.always_included)}")
        for name, chars in self.oversized.items():
            lines.append(f"oversized: {name} ({chars:,} chars)")
        for name, age in self.stale.items():
            lines.append(f"stale: {name} ({age / 86400:.1f} days old)")
        for name, tiers in self.duplicates.items():
            lines.append(f"duplicate: {name} in {', '.join(tiers)}")
        if self.never_present:
            lines.append(f"never present: {', '.join(self.never_present)}")
        for name, count in self.requested_but_absent.items():
            lines.append(f"requested but absent: {name} ({count}x)")
        for kind, count in self.fallbacks.items():
            lines.append(f"fallback {kind}: {count}")
        for kind, count in self.gaps.items():
            lines.append(f"gap {kind}: {count}")
        for reason in self.skipped:
            lines.append(f"skipped: {reason}")
        if len(lines) == 1:
            lines.append("no findings")
        return "\n".join(lines)
