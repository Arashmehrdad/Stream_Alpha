"""Bounded adaptive sizing helpers for the Stream Alpha M19 layer."""

from __future__ import annotations

from app.adaptation.config import SizingBoundsConfig
from app.adaptation.schemas import AdaptivePerformanceWindow


def bounded_size_multiplier(
    *,
    configured_multiplier: float,
    calibrated_confidence: float | None,
    performance: AdaptivePerformanceWindow | None,
    bounds: SizingBoundsConfig,
) -> float:
    """Return a bounded size multiplier that still flows through M10 clamps."""
    multiplier = configured_multiplier
    if calibrated_confidence is not None:
        multiplier += (calibrated_confidence - 0.5) * bounds.calibration_weight
    if performance is not None:
        multiplier += performance.net_pnl_after_costs * bounds.performance_weight
        multiplier -= performance.max_drawdown * bounds.drawdown_penalty_weight
    return min(bounds.max_multiplier, max(bounds.min_multiplier, multiplier))
