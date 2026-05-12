"""Tests for M20 trading-aware label artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_trading_aware_labels import build_m20_trading_aware_labels

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixtures(base: Path) -> Path:
    research = base / "research_labels" / "vol_scaled"
    _write_csv(
        research / "economic_outcome_artifacts" / "economic_outcomes.csv",
        [
            {
                "fold_index": 1,
                "symbol": "BTC",
                "interval_begin": "2024-01-01T00:00:00Z",
                "future_return": 0.04,
                "net_value_proxy": 0.02,
                "triple_barrier_label": 1,
            },
            {
                "fold_index": 1,
                "symbol": "ETH",
                "interval_begin": "2024-01-01T00:00:00Z",
                "future_return": -0.01,
                "net_value_proxy": -0.02,
                "triple_barrier_label": -1,
            },
        ],
    )
    _write_csv(
        research / "research_feature_enrichment" / "research_features.csv",
        [
            {
                "fold_index": 1,
                "symbol": "BTC",
                "interval_begin": "2024-01-01T00:00:00Z",
                "realized_vol_12": 0.02,
            },
            {
                "fold_index": 1,
                "symbol": "ETH",
                "interval_begin": "2024-01-01T00:00:00Z",
                "realized_vol_12": 0.02,
            },
        ],
    )
    return base


def test_trading_aware_labels_are_deterministic_and_research_only(tmp_path: Path) -> None:
    source = _fixtures(tmp_path)

    result = build_m20_trading_aware_labels(source_run_dir=source)
    labels_path = Path(result["output_files"]["trading_aware_labels_csv"])
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["fee_plus_slippage_exceedance_label"] == "1"
    assert rows[0]["volatility_adjusted_return_bucket"] == "POSITIVE_STRONG"
    assert rows[1]["fee_plus_slippage_exceedance_label"] == "0"
    assert "NO_RUNTIME_EFFECT" in result["overall_status"]
    assert "NOT_PROMOTABLE" in result["overall_status"]
    assert "NO_PROFIT_CLAIM" in result["overall_status"]


def test_trading_aware_labels_block_missing_multi_horizon_sources(tmp_path: Path) -> None:
    source = _fixtures(tmp_path)

    result = build_m20_trading_aware_labels(source_run_dir=source)
    blocked = {row["label_name"] for row in result["blocked_labels"]}

    assert "forward_return_6_candles" in blocked
    assert result["recommendation"] == "RE_RUN_DECISION_POLICY_EVALUATOR_WITH_TRADING_AWARE_LABELS"
