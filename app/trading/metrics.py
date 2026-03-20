"""Performance metrics and summary helpers for Stream Alpha M5."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

from app.regime.service import REGIME_LABELS
from app.trading.config import PaperTradingConfig
from app.trading.risk import calculate_fee
from app.trading.schemas import PaperPosition


def build_summary(
    *,
    config: PaperTradingConfig,
    positions: list[PaperPosition],
    latest_prices: dict[str, float],
    cash_balance: float,
) -> dict[str, Any]:
    """Build overall and by-asset trading summaries."""
    closed_positions = [position for position in positions if position.status == "CLOSED"]
    open_positions = [position for position in positions if position.status == "OPEN"]
    realized_pnl = sum((position.realized_pnl or 0.0) for position in closed_positions)
    unrealized_pnl = sum(
        _unrealized_pnl(position, latest_prices.get(position.symbol), config.risk.fee_bps)
        for position in open_positions
    )
    total_pnl = realized_pnl + unrealized_pnl
    trade_returns = [position.realized_return for position in closed_positions]
    realized_returns = [value for value in trade_returns if value is not None]
    winning_trades = [
        position for position in closed_positions if (position.realized_pnl or 0.0) > 0
    ]
    overall = {
        "service_name": config.service_name,
        "symbols": list(config.symbols),
        "cash_balance": cash_balance,
        "cumulative_pnl_realized": realized_pnl,
        "cumulative_pnl_total": total_pnl,
        "win_rate": 0.0 if not closed_positions else len(winning_trades) / len(closed_positions),
        "max_drawdown": _max_drawdown(config.risk.initial_cash, closed_positions, total_pnl),
        "average_trade_return": 0.0 if not realized_returns else mean(realized_returns),
        "turnover": _turnover(config.risk.initial_cash, positions),
        "hit_rate_by_asset": _hit_rate_by_asset(closed_positions),
        "hit_rate_by_regime": _hit_rate_by_regime(closed_positions),
        "realized_pnl_by_regime": _realized_pnl_by_regime(closed_positions),
        "closed_position_count_by_regime": _closed_position_count_by_regime(closed_positions),
        "sharpe_like": _sharpe_like(realized_returns),
        "open_position_count": len(open_positions),
        "closed_position_count": len(closed_positions),
    }
    return {
        "overall": overall,
        "by_asset": _by_asset_summary(
            symbols=config.symbols,
            positions=positions,
            latest_prices=latest_prices,
            fee_bps=config.risk.fee_bps,
        ),
        "by_regime": _by_regime_summary(
            positions=positions,
            latest_prices=latest_prices,
            fee_bps=config.risk.fee_bps,
        ),
    }


def _unrealized_pnl(position: PaperPosition, latest_price: float | None, fee_bps: float) -> float:
    if latest_price is None:
        return 0.0
    exit_notional = latest_price * position.quantity
    exit_fee = calculate_fee(exit_notional, fee_bps)
    return (exit_notional - exit_fee) - (position.entry_notional + position.entry_fee)


def _max_drawdown(
    initial_cash: float,
    closed_positions: list[PaperPosition],
    total_pnl: float,
) -> float:
    equity = initial_cash
    peak = equity
    max_drawdown = 0.0
    for position in sorted(
        closed_positions,
        key=lambda row: row.exit_fill_time or row.entry_fill_time,
    ):
        equity += position.realized_pnl or 0.0
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    final_equity = initial_cash + total_pnl
    peak = max(peak, final_equity)
    if peak > 0:
        max_drawdown = max(max_drawdown, (peak - final_equity) / peak)
    return max_drawdown


def _turnover(initial_cash: float, positions: list[PaperPosition]) -> float:
    if initial_cash <= 0:
        return 0.0
    traded_notional = 0.0
    for position in positions:
        traded_notional += position.entry_notional
        if position.exit_notional is not None:
            traded_notional += position.exit_notional
    return traded_notional / initial_cash


def _hit_rate_by_asset(closed_positions: list[PaperPosition]) -> dict[str, float]:
    by_asset: dict[str, list[PaperPosition]] = {}
    for position in closed_positions:
        by_asset.setdefault(position.symbol, []).append(position)
    return {
        symbol: (
            0.0
            if not rows
            else sum(int((row.realized_pnl or 0.0) > 0.0) for row in rows) / len(rows)
        )
        for symbol, rows in by_asset.items()
    }


def _hit_rate_by_regime(closed_positions: list[PaperPosition]) -> dict[str, float]:
    by_regime: dict[str, list[PaperPosition]] = {}
    for position in closed_positions:
        by_regime.setdefault(_regime_key(position.entry_regime_label), []).append(position)
    return {
        regime_label: (
            0.0
            if not rows
            else sum(int((row.realized_pnl or 0.0) > 0.0) for row in rows) / len(rows)
        )
        for regime_label, rows in by_regime.items()
    }


def _realized_pnl_by_regime(closed_positions: list[PaperPosition]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for position in closed_positions:
        regime_label = _regime_key(position.entry_regime_label)
        totals[regime_label] = totals.get(regime_label, 0.0) + (position.realized_pnl or 0.0)
    return totals


def _closed_position_count_by_regime(closed_positions: list[PaperPosition]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for position in closed_positions:
        regime_label = _regime_key(position.entry_regime_label)
        counts[regime_label] = counts.get(regime_label, 0) + 1
    return counts


def _sharpe_like(realized_returns: list[float]) -> float:
    if len(realized_returns) < 2:
        return 0.0
    standard_deviation = pstdev(realized_returns)
    if standard_deviation == 0.0:
        return 0.0
    return (mean(realized_returns) / standard_deviation) * math.sqrt(len(realized_returns))


def _by_asset_summary(
    *,
    symbols: tuple[str, ...],
    positions: list[PaperPosition],
    latest_prices: dict[str, float],
    fee_bps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        symbol_positions = [position for position in positions if position.symbol == symbol]
        open_positions = [
            position for position in symbol_positions if position.status == "OPEN"
        ]
        closed_positions = [
            position for position in symbol_positions if position.status == "CLOSED"
        ]
        realized_pnl = sum((position.realized_pnl or 0.0) for position in closed_positions)
        unrealized_pnl = sum(
            _unrealized_pnl(position, latest_prices.get(symbol), fee_bps)
            for position in open_positions
        )
        rows.append(
            {
                "symbol": symbol,
                "open_positions": len(open_positions),
                "closed_positions": len(closed_positions),
                "realized_pnl": realized_pnl,
                "total_pnl": realized_pnl + unrealized_pnl,
                "win_rate": (
                    0.0
                    if not closed_positions
                    else (
                        sum(
                            int((position.realized_pnl or 0.0) > 0.0)
                            for position in closed_positions
                        )
                        / len(closed_positions)
                    )
                ),
            }
        )
    return rows


def _by_regime_summary(
    *,
    positions: list[PaperPosition],
    latest_prices: dict[str, float],
    fee_bps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime_label in _ordered_regime_labels(positions):
        regime_positions = [
            position
            for position in positions
            if _regime_key(position.entry_regime_label) == regime_label
        ]
        open_positions = [
            position for position in regime_positions if position.status == "OPEN"
        ]
        closed_positions = [
            position for position in regime_positions if position.status == "CLOSED"
        ]
        realized_pnl = sum((position.realized_pnl or 0.0) for position in closed_positions)
        unrealized_pnl = sum(
            _unrealized_pnl(position, latest_prices.get(position.symbol), fee_bps)
            for position in open_positions
        )
        rows.append(
            {
                "regime_label": regime_label,
                "open_positions": len(open_positions),
                "closed_positions": len(closed_positions),
                "realized_pnl": realized_pnl,
                "total_pnl": realized_pnl + unrealized_pnl,
                "win_rate": (
                    0.0
                    if not closed_positions
                    else (
                        sum(
                            int((position.realized_pnl or 0.0) > 0.0)
                            for position in closed_positions
                        )
                        / len(closed_positions)
                    )
                ),
            }
        )
    return rows


def _ordered_regime_labels(positions: list[PaperPosition]) -> list[str]:
    encountered = {
        _regime_key(position.entry_regime_label)
        for position in positions
    }
    ordered = list(REGIME_LABELS)
    if "UNKNOWN" in encountered or not encountered:
        ordered.append("UNKNOWN")
    else:
        extras = sorted(encountered - set(REGIME_LABELS))
        ordered.extend(extras)
    return ordered


def _regime_key(value: str | None) -> str:
    if value is None or not value.strip():
        return "UNKNOWN"
    return value
