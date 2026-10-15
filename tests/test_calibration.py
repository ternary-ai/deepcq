import pytest

from deepcq import DeepCQ, MemorySchema
from deepcq.calibration import LabeledExample, calibrate


def _builder(texts):
    def build():
        cq = DeepCQ(MemorySchema(sections=["a", "b"]))
        for section, text in texts.items():
            cq.state.set(section, f"[{section}]\n{text}\n")
        return cq
    return build


def _fixed_scores_example(intent, expected, monkeypatch=None):
    """Example whose query() returns a hand-picked intent dict."""
    class Fixed(DeepCQ):
        def query(self, query, recency=None, now=None):
            result = super().query(query)
            result.intent = dict(intent)
            return result
    return LabeledExample(build=lambda: Fixed(MemorySchema(sections=["a", "b"])), query="q",
                          expected_sections=set(expected))


def test_precision_recall_arithmetic():
    examples = [
        _fixed_scores_example({"a": 0.4, "b": 0.1}, {"a"}),
        _fixed_scores_example({"a": 0.2, "b": 0.3}, {"a", "b"}),
    ]
    report = calibrate(examples, thresholds=[0.1, 0.25, 0.35])
    by_t = {s.threshold: s for s in report.stats}
    # t=0.10: ex1 {a,b} vs {a} -> tp1 fp1; ex2 {a,b} vs {a,b} -> tp2. tp3 fp1 fn0
    assert (by_t[0.1].true_positives, by_t[0.1].false_positives, by_t[0.1].false_negatives) == (3, 1, 0)
    assert by_t[0.1].precision == 0.75 and by_t[0.1].recall == 1.0
    assert by_t[0.1].f1 == round(2 * 0.75 / 1.75, 4)
    # t=0.25: ex1 {a} -> tp1; ex2 {b} -> tp1 fn1. tp2 fp0 fn1
    assert (by_t[0.25].true_positives, by_t[0.25].false_positives, by_t[0.25].false_negatives) == (2, 0, 1)
    # t=0.35: ex1 {a} -> tp1; ex2 {} -> fn2. tp1 fp0 fn2
    assert by_t[0.35].recall == round(1 / 3, 4)
    assert [s.threshold for s in report.stats] == [0.1, 0.25, 0.35]


def test_finds_known_correct_threshold():
    # Relevant sections score >= 0.5, irrelevant ones <= 0.25, so any
    # threshold in (0.25, 0.5] is perfect; ties go to the highest one.
    examples = [
        LabeledExample(_builder({"a": "cpu memory", "b": "disk"}), "cpu memory load", {"a"}),
        LabeledExample(_builder({"a": "cpu", "b": "disk usage full"}), "disk usage", {"b"}),
        LabeledExample(_builder({"a": "cpu load", "b": "disk load"}), "cpu load", {"a", "b"}),
    ]
    report = calibrate(examples)
    assert report.best.f1 == 1.0
    assert report.best.threshold == 0.5
    assert "<- best" in report.summary()


def test_ties_broken_toward_higher_threshold():
    examples = [_fixed_scores_example({"a": 0.9, "b": 0.0}, {"a"})]
    report = calibrate(examples, thresholds=[0.2, 0.4, 0.6])
    assert report.best.threshold == 0.6


def test_min_precision_picks_highest_recall_above_floor():
    examples = [
        _fixed_scores_example({"a": 0.4, "b": 0.1}, {"a"}),
        _fixed_scores_example({"a": 0.2, "b": 0.3}, {"a", "b"}),
    ]
    report = calibrate(examples, thresholds=[0.1, 0.25, 0.35], min_precision=0.9)
    assert report.constraint_met is True
    assert report.best.threshold == 0.25  # recall 2/3 at precision 1.0


def test_min_precision_unsatisfiable_falls_back_loudly():
    examples = [_fixed_scores_example({"a": 0.5, "b": 0.5}, {"a"})]
    report = calibrate(examples, thresholds=[0.1, 0.3], min_precision=0.9)
    assert report.constraint_met is False
    assert report.best.threshold == 0.3   # F1-best, higher on tie
    assert "WARNING" in report.summary()


def test_query_called_once_per_example(monkeypatch):
    calls = []
    original = DeepCQ.query

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DeepCQ, "query", counting)
    examples = [LabeledExample(_builder({"a": "cpu", "b": "disk"}), "cpu", {"a"})] * 3
    report = calibrate(examples)
    assert len(report.stats) == 25
    assert len(calls) == 3


def test_fallback_not_simulated():
    # Nothing scores above 0.3, so query() would fall back to including
    # everything — calibrate() must still count those sections as excluded.
    examples = [LabeledExample(_builder({"a": "cpu", "b": "disk"}), "unrelated words here", {"a"})]
    stats = {s.threshold: s for s in calibrate(examples, thresholds=[0.3]).stats}
    assert stats[0.3].true_positives == 0 and stats[0.3].false_negatives == 1


def test_validation_errors():
    with pytest.raises(ValueError, match="at least one"):
        calibrate([])
    with pytest.raises(ValueError, match="not in its schema"):
        calibrate([LabeledExample(_builder({"a": "cpu"}), "cpu", {"zzz"})])
