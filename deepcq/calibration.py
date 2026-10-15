"""Threshold calibration — pick `confidence_threshold` from labeled examples
instead of sweeping and eyeballing by hand."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
from .core import DeepCQ

DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(0.02 * i, 2) for i in range(1, 26))  # 0.02..0.50


@dataclass
class LabeledExample:
    """One (memory-state, query, expected-sections) calibration case."""
    build: Callable[[], DeepCQ]   # returns a fully populated DeepCQ for this case
    query: str
    expected_sections: set[str]   # section names a correct filter should include


@dataclass
class ThresholdStats:
    threshold: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class CalibrationReport:
    stats: list[ThresholdStats]   # one per swept threshold, ascending
    best: ThresholdStats
    # False when min_precision was set but no threshold reached it, so
    # `best` fell back to the unconstrained F1-best threshold.
    constraint_met: bool = True
    min_precision: float | None = None

    def summary(self) -> str:
        """Pretty-printed table: threshold | precision | recall | f1."""
        lines = [
            "threshold | precision | recall |   f1   |  tp |  fp |  fn",
            "----------+-----------+--------+--------+-----+-----+-----",
        ]
        for s in self.stats:
            marker = "  <- best" if s is self.best else ""
            lines.append(
                f"{s.threshold:9.2f} | {s.precision:9.3f} | {s.recall:6.3f} | {s.f1:6.3f} "
                f"| {s.true_positives:3d} | {s.false_positives:3d} | {s.false_negatives:3d}{marker}"
            )
        if self.min_precision is not None and not self.constraint_met:
            lines.append(
                f"WARNING: no threshold reached min_precision={self.min_precision}; "
                f"best is the unconstrained F1-best threshold ({self.best.threshold:.2f})."
            )
        return "\n".join(lines)


def _stats(threshold: float, tp: int, fp: int, fn: int) -> ThresholdStats:
    # Undefined ratios (nothing included / nothing expected) count as 0.0.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ThresholdStats(threshold, round(precision, 4), round(recall, 4), round(f1, 4), tp, fp, fn)


def calibrate(
    examples: Sequence[LabeledExample],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    min_precision: float | None = None,
) -> CalibrationReport:
    """
    Sweep `thresholds` over labeled examples and report precision/recall/F1
    per threshold.

    Each example is built and queried exactly once; the sweep re-thresholds
    that query's raw `result.intent` scores (`score >= t`), so N examples
    cost N query() calls regardless of how many thresholds are swept. The
    schema's own confidence_threshold and fallback_include_all are ignored —
    a bad threshold must not hide behind the fallback in these numbers.

    TP/FP/FN are summed across all examples per threshold (micro-averaged).
    `best` is the highest-F1 threshold, ties broken toward the higher
    (more conservative) threshold. With `min_precision`, `best` is instead
    the highest-recall threshold whose precision >= min_precision (ties →
    higher threshold); if none qualifies, it falls back to the F1-best with
    `constraint_met=False`, which summary() flags.
    """
    if not examples:
        raise ValueError("calibrate() needs at least one LabeledExample")
    if not thresholds:
        raise ValueError("calibrate() needs at least one threshold")

    scored: list[tuple[dict[str, float], set[str]]] = []
    for i, example in enumerate(examples):
        cq = example.build()
        unknown = set(example.expected_sections) - set(cq.schema.sections)
        if unknown:
            raise ValueError(
                f"example {i} ({example.query!r}) expects sections not in its schema: {sorted(unknown)}"
            )
        result = cq.query(example.query)
        scored.append((result.intent, set(example.expected_sections)))

    stats: list[ThresholdStats] = []
    for t in sorted(thresholds):
        tp = fp = fn = 0
        for intent, expected in scored:
            included = {s for s, score in intent.items() if score >= t}
            tp += len(included & expected)
            fp += len(included - expected)
            fn += len(expected - included)
        stats.append(_stats(t, tp, fp, fn))

    f1_best = max(stats, key=lambda s: (s.f1, s.threshold))
    if min_precision is None:
        return CalibrationReport(stats=stats, best=f1_best)

    qualifying = [s for s in stats if s.true_positives and s.precision >= min_precision]
    if not qualifying:
        return CalibrationReport(
            stats=stats, best=f1_best, constraint_met=False, min_precision=min_precision
        )
    best = max(qualifying, key=lambda s: (s.recall, s.threshold))
    return CalibrationReport(stats=stats, best=best, min_precision=min_precision)
