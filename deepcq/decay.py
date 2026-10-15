"""Recency decay — optional per-chunk time-weighting as a scoring primitive."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable

DecayFn = Callable[[float], float]  # age_seconds -> multiplier in [0, 1]


def no_decay(age_seconds: float) -> float:
    return 1.0


def exponential_decay(half_life_seconds: float) -> DecayFn:
    """Multiplier halves every `half_life_seconds` of age."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")

    def _decay(age_seconds: float) -> float:
        return 0.5 ** (age_seconds / half_life_seconds)

    return _decay


def linear_decay(max_age_seconds: float) -> DecayFn:
    """Multiplier falls linearly to 0 at `max_age_seconds`, floored at 0."""
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be positive")

    def _decay(age_seconds: float) -> float:
        return max(0.0, 1.0 - age_seconds / max_age_seconds)

    return _decay


@dataclass
class Recency:
    """A section/chunk's timestamp plus how its score should decay with age."""
    timestamp: float
    decay_fn: DecayFn = no_decay

    def multiplier(self, now: float | None = None) -> float:
        current = time.time() if now is None else now
        age = max(0.0, current - self.timestamp)
        return self.decay_fn(age)
