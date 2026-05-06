"""Focused tests for M20 rank-gate tail-risk filter simulation."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_rank_gate_tail_filter import simulate_m20_rank_gate_tail_filter

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_selected_rows(base: Path) -> None:
    rows = []
    for run_label in ("original", "confirmation"):
        for index in range(20):
            label = 1 if index < 8 else 0
            net = 0.01 if label else -0.003
            if index in (18, 19):
                net = -0.06 if run_label == "original" else -0.01
            rows.append(
                {
                    "run_label": run_label,
                    "symbol": "BTC/USD" if index % 2 else "ETH/USD",
                    "interval_begin": f"2025-01-{index + 1:02d}T00:00:00Z",
                    "month": "2025-01",
                    "quarter": "2025Q1",
                    "label": label,
                    "probability": 0.99 - index * 0.001,
                    "net_return_proxy": net,
                    "range_bucket": "high" if index >= 15 else "low",
                    "volatility_bucket": "high",
                    "volume_bucket": "high",
                    "momentum_bucket": "negative" if index % 3 else "positive",
                    "macd_bucket": "positive",
                    "probability_bin": "p99_plus",
                }
            )
    _write_csv(
        base
        / "research_labels"
        / "vol_scaled"
        / "rank_gate_net_diagnostics"
        / "selected_row_diagnostics.csv",
        rows,
    )


def test_tail_filter_writes_metrics(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_selected_rows(base)

    result = simulate_m20_rank_gate_tail_filter(base_run_dir=base)

    assert "FILTER_SIM_ONLY" in result["honesty_flags"]
    assert result["filter_count"] >= 5
    assert Path(result["output_files"]["filter_metrics_csv"]).exists()
    assert Path(result["output_files"]["stability_csv"]).exists()


def test_tail_filter_blocks_when_rows_missing(tmp_path: Path) -> None:
    result = simulate_m20_rank_gate_tail_filter(base_run_dir=tmp_path / "base")

    assert result["recommendation"] == "BLOCKED_ROW_LEVEL_NET_DIAGNOSTICS_MISSING"
    assert "ROW_LEVEL_ECONOMICS_MISSING" in result["blockers"]
