"""Bounded threshold-tuning helpers for the Stream Alpha M19 layer."""

# pylint: disable=too-many-arguments

from __future__ import annotations

from app.adaptation.config import ThresholdBoundsConfig
from app.adaptation.schemas import AdaptivePerformanceWindow, EffectiveThresholds


def bounded_effective_thresholds(
    *,
    base_buy_prob_up: float,
    base_sell_prob_up: float,
    calibrated_confidence: float | None,
    performance: AdaptivePerformanceWindow | None,
    configured_delta: float,
    bounds: ThresholdBoundsConfig,
) -> EffectiveThresholds:
    """Return bounded additive thresholds without changing M4 authority."""
    delta = max(-bounds.max_absolute_delta, min(bounds.max_absolute_delta, configured_delta))
    if performance is not None:
        improvement_signal = performance.net_pnl_after_costs - performance.max_drawdown
        scaled = improvement_signal * bounds.improvement_sensitivity
        delta += max(-bounds.max_absolute_delta, min(bounds.max_absolute_delta, scaled))
    if calibrated_confidence is not None and calibrated_confidence < 0.5:
        delta -= min(bounds.max_absolute_delta, 0.5 - calibrated_confidence)
    buy_prob_up = min(
        bounds.max_buy_prob_up,
        max(bounds.min_buy_prob_up, base_buy_prob_up - delta),
    )
    sell_prob_up = min(
        bounds.max_sell_prob_up,
        max(bounds.min_sell_prob_up, base_sell_prob_up + delta),
    )
    return EffectiveThresholds(
        buy_prob_up=buy_prob_up,
        sell_prob_up=sell_prob_up,
    )
