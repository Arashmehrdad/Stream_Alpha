"""Focused tests for M20 rank-gate tail concentration analysis."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_rank_gate_tail_analysis import analyze_m20_rank_gate_tail

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
    for run_label in ("negative_window", "positive_window"):
        for index in range(20):
            label = 1 if index < 8 else 0
            net = 0.01 if label else -0.003
            if run_label == "negative_window" and index in (18, 19):
                net = -0.08
            rows.append(
                {
                    "run_label": run_label,
                    "selected_rank": index,
                    "symbol": "BTC/USD" if index % 2 else "ETH/USD",
                    "interval_begin": f"2025-01-{index + 1:02d}T00:00:00Z",
                    "month": "2025-01",
                    "quarter": "2025Q1",
                    "label": label,
                    "probability": 0.99 - index * 0.001,
                    "future_return": net + 0.002,
                    "cost_per_trade": 0.002,
                    "net_return_proxy": net,
                    "probability_bin": "p99_plus",
                    "volatility_bucket": "high",
                    "range_bucket": "high" if index % 2 else "low",
                    "volume_bucket": "high",
                    "momentum_bucket": "positive" if index % 3 else "negative",
                    "macd_bucket": "positive",
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


def test_tail_analysis_writes_outputs(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_selected_rows(base)

    result = analyze_m20_rank_gate_tail(base_run_dir=base)

    assert "RESEARCH_ONLY" in result["honesty_flags"]
    assert result["selected_rows"] == 40
    assert Path(result["output_files"]["tail_contribution_csv"]).exists()
    assert Path(result["output_files"]["worst_rows_csv"]).exists()
    assert result["tail_contribution"]


def test_tail_analysis_blocks_when_selected_rows_missing(tmp_path: Path) -> None:
    result = analyze_m20_rank_gate_tail(base_run_dir=tmp_path / "base")

    assert result["recommendation"] == "BLOCKED_ROW_LEVEL_NET_DIAGNOSTICS_MISSING"
    assert "ROW_LEVEL_ECONOMICS_MISSING" in result["blockers"]
