import pytest
from deepcq.decay import Recency, no_decay, exponential_decay, linear_decay


def test_no_decay_always_one():
    assert no_decay(0.0) == 1.0
    assert no_decay(1_000_000.0) == 1.0


def test_exponential_decay_halves_at_half_life():
    decay_fn = exponential_decay(100.0)
    assert decay_fn(0.0) == pytest.approx(1.0)
    assert decay_fn(100.0) == pytest.approx(0.5)
    assert decay_fn(200.0) == pytest.approx(0.25)


def test_exponential_decay_rejects_non_positive_half_life():
    with pytest.raises(ValueError):
        exponential_decay(0.0)
    with pytest.raises(ValueError):
        exponential_decay(-5.0)


def test_linear_decay_reaches_zero_at_max_age():
    decay_fn = linear_decay(100.0)
    assert decay_fn(0.0) == pytest.approx(1.0)
    assert decay_fn(50.0) == pytest.approx(0.5)
    assert decay_fn(100.0) == pytest.approx(0.0)


def test_linear_decay_floors_at_zero_past_max_age():
    decay_fn = linear_decay(100.0)
    assert decay_fn(1000.0) == 0.0


def test_linear_decay_rejects_non_positive_max_age():
    with pytest.raises(ValueError):
        linear_decay(0.0)


def test_recency_multiplier_uses_decay_fn():
    recency = Recency(timestamp=1000.0, decay_fn=exponential_decay(100.0))
    assert recency.multiplier(now=1100.0) == pytest.approx(0.5)


def test_recency_multiplier_defaults_to_no_decay():
    recency = Recency(timestamp=1000.0)
    assert recency.multiplier(now=50_000.0) == 1.0


def test_recency_multiplier_clamps_negative_age_to_zero():
    # `now` before `timestamp` (e.g. clock skew) shouldn't produce a
    # multiplier greater than the fn's value at age 0.
    recency = Recency(timestamp=1000.0, decay_fn=exponential_decay(100.0))
    assert recency.multiplier(now=500.0) == pytest.approx(1.0)


def test_recency_multiplier_uses_current_time_when_now_omitted():
    recency = Recency(timestamp=0.0, decay_fn=exponential_decay(1.0))
    assert recency.multiplier() == pytest.approx(0.0, abs=1e-6)
