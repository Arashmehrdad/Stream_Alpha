"""Simple inspectable concept-drift helpers for the Stream Alpha M19 layer."""

from __future__ import annotations

from collections.abc import Sequence
import math


def population_stability_index(
    reference_values: Sequence[float],
    live_values: Sequence[float],
    *,
    bucket_count: int = 10,
) -> float:
    """Return a simple PSI score using equal-width buckets across both windows."""
    if not reference_values or not live_values:
        return 0.0
    min_value = min(*reference_values, *live_values)
    max_value = max(*reference_values, *live_values)
    if math.isclose(min_value, max_value):
        return 0.0
    width = (max_value - min_value) / float(bucket_count)
    reference_counts = [0 for _ in range(bucket_count)]
    live_counts = [0 for _ in range(bucket_count)]
    for value in reference_values:
        reference_counts[_bucket_index(value, min_value, width, bucket_count)] += 1
    for value in live_values:
        live_counts[_bucket_index(value, min_value, width, bucket_count)] += 1
    return _psi_from_counts(reference_counts, live_counts)


def classify_drift(
    score: float,
    *,
    warning_threshold: float,
    breach_threshold: float,
) -> tuple[str, str]:
    """Return the bounded drift status and explicit reason code for one score."""
    if score >= breach_threshold:
        return "BREACHED", "DRIFT_BREACH"
    if score >= warning_threshold:
        return "WATCH", "DRIFT_WATCH"
    return "HEALTHY", "DRIFT_HEALTHY"


def _bucket_index(value: float, minimum: float, width: float, bucket_count: int) -> int:
    if width <= 0.0:
        return 0
    index = int((value - minimum) / width)
    return max(0, min(bucket_count - 1, index))


def _psi_from_counts(reference_counts: list[int], live_counts: list[int]) -> float:
    reference_total = max(1, sum(reference_counts))
    live_total = max(1, sum(live_counts))
    score = 0.0
    for reference_count, live_count in zip(reference_counts, live_counts, strict=True):
        reference_ratio = max(reference_count / reference_total, 1e-6)
        live_ratio = max(live_count / live_total, 1e-6)
        score += (live_ratio - reference_ratio) * math.log(live_ratio / reference_ratio)
    return score
