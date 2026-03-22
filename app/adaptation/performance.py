"""Rolling performance-window helpers for the Stream Alpha M19 layer."""

# pylint: disable=too-many-arguments,too-many-locals

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from app.adaptation.schemas import AdaptivePerformanceWindow


def build_rolling_performance_windows(
    *,
    execution_mode: str,
    symbol: str,
    regime_label: str,
    rows: Sequence[dict[str, Any]],
    trade_counts: Sequence[int],
    day_windows: Sequence[int],
    now: datetime,
) -> list[AdaptivePerformanceWindow]:
    """Build fixed rolling performance windows from deterministic trade rows."""
    items: list[AdaptivePerformanceWindow] = []
    ordered_rows = sorted(rows, key=lambda row: row["event_time"])
    for trade_count in trade_counts:
        window_rows = ordered_rows[-trade_count:]
        if not window_rows:
            continue
        items.append(
            _build_window(
                execution_mode=execution_mode,
                symbol=symbol,
                regime_label=regime_label,
                window_id=f"last_{trade_count}_trades",
                window_type="trade_count",
                rows=window_rows,
            )
        )
    for day_count in day_windows:
        cutoff = now - timedelta(days=day_count)
        window_rows = [row for row in ordered_rows if row["event_time"] >= cutoff]
        if not window_rows:
            continue
        items.append(
            _build_window(
                execution_mode=execution_mode,
                symbol=symbol,
                regime_label=regime_label,
                window_id=f"last_{day_count}d",
                window_type="time",
                rows=window_rows,
            )
        )
    return items


def _build_window(
    *,
    execution_mode: str,
    symbol: str,
    regime_label: str,
    window_id: str,
    window_type: str,
    rows: Sequence[dict[str, Any]],
) -> AdaptivePerformanceWindow:
    pnl_values = [float(row.get("realized_pnl", 0.0)) for row in rows]
    trade_count = len(rows)
    wins = [value for value in pnl_values if value > 0.0]
    losses = [value for value in pnl_values if value < 0.0]
    equity = 0.0
    high_watermark = 0.0
    max_drawdown = 0.0
    for pnl in pnl_values:
        equity += pnl
        high_watermark = max(high_watermark, equity)
        max_drawdown = min(max_drawdown, equity - high_watermark)
    blocked_count = sum(1 for row in rows if bool(row.get("blocked", False)))
    shadow_divergence_count = sum(
        1 for row in rows if bool(row.get("shadow_diverged", False))
    )
    slippage_values = [float(row.get("slippage_bps", 0.0)) for row in rows]
    predicted_positive = sum(1 for row in rows if bool(row.get("predicted_positive", False)))
    true_positive = sum(1 for row in rows if bool(row.get("true_positive", False)))
    precision = 0.0 if predicted_positive == 0 else true_positive / predicted_positive
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return AdaptivePerformanceWindow(
        execution_mode=execution_mode,
        symbol=symbol,
        regime_label=regime_label,
        window_id=window_id,
        window_type=window_type,
        window_start=rows[0]["event_time"],
        window_end=rows[-1]["event_time"],
        trade_count=trade_count,
        net_pnl_after_costs=sum(pnl_values),
        max_drawdown=abs(max_drawdown),
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf"),
        expectancy=(sum(pnl_values) / trade_count) if trade_count else 0.0,
        win_rate=(len(wins) / trade_count) if trade_count else 0.0,
        precision=precision,
        avg_slippage_bps=(sum(slippage_values) / trade_count) if trade_count else 0.0,
        blocked_trade_rate=(blocked_count / trade_count) if trade_count else 0.0,
        shadow_divergence_rate=(shadow_divergence_count / trade_count) if trade_count else 0.0,
        health_context=dict(rows[-1].get("health_context", {})),
    )
