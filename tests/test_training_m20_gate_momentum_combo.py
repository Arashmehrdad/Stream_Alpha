"""Focused tests for M20 gate + momentum combo diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_gate_momentum_combo import analyze_m20_gate_momentum_combo

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_run(run_dir: Path) -> None:
    features = []
    labels = []
    predictions = []
    for index in range(100):
        symbol = "BTC/USD" if index % 2 else "ETH/USD"
        timestamp = f"2025-01-{(index % 28) + 1:02d}T00:{index % 60:02d}:00Z"
        positive = 1 if index < 25 else 0
        probability = 0.99 - index * 0.006
        features.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "high_price": 103,
                "low_price": 100,
                "close_price": 102,
                "vwap": 101,
                "volume": 1000 + index,
                "realized_vol_12": 0.001 + index * 0.0001,
                "log_return_1": 0.001 if index < 25 else -0.001,
                "momentum_3": 0.002 if index < 25 else -0.001,
                "macd_line_12_26": 0.1 if index < 25 else -0.1,
            }
        )
        labels.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "label": positive,
                "future_return": 0.01 if positive else -0.004,
                "cost_per_trade": 0.002,
                "scenario_name": "current_fee",
            }
        )
        predictions.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "y_true": positive,
                "probability": probability,
                "scenario_name": "current_fee",
            }
        )
    _write_csv(run_dir / "training_frame" / "m20_training_frame_features.csv", features)
    _write_csv(
        run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_labels_vol_scaled.csv",
        labels,
    )
    _write_csv(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_baselines"
        / "predictions_logistic_regression_tiny_test_full.csv",
        predictions,
    )
    _write_csv(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "strategy_selector_design"
        / "strategy_selector_condition_weights.csv",
        [
            {
                "slice_family": "symbol",
                "slice_value": "BTC/USD",
                "evidence_weight": 1.0,
                "proposed_selector_action": "ENABLE_RESEARCH_CANDIDATE",
            }
        ],
    )
    _write_json(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "rank_gate_evidence_packet"
        / "rank_gate_evidence_packet.json",
        {"window_metrics": [{"window_label": "base", "run_dir": str(run_dir)}]},
    )


def test_gate_momentum_combo_writes_outputs(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_gate_momentum_combo(base_run_dir=base)

    assert "RESEARCH_ONLY" in result["honesty_flags"]
    assert result["policy_count"] == 8
    assert Path(result["output_files"]["policy_metrics_csv"]).exists()
    assert Path(result["output_files"]["tail_summary_csv"]).exists()


def test_gate_momentum_combo_preserves_no_runtime_flag(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_gate_momentum_combo(base_run_dir=base)

    assert "NO_RUNTIME" in result["honesty_flags"]
    assert "NOT_BACKTEST" in result["blockers"]
