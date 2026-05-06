"""Focused tests for M20 rank-gate economics diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_rank_gate_economics import simulate_m20_rank_gate_economics

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


def _write_run(run_dir: Path, split: str = "test") -> None:
    features = []
    predictions = []
    labels = []
    for index in range(100):
        symbol = "BTC/USD" if index % 2 == 0 else "ETH/USD"
        timestamp = f"2025-01-{(index % 28) + 1:02d}T00:{index % 60:02d}:00Z"
        row_id = f"{symbol}|{timestamp}"
        positive = 1 if index < 20 else 0
        probability = 0.99 - index * 0.007
        features.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "row_id": row_id,
                "high_price": 101,
                "low_price": 100,
                "close_price": 100,
                "log_return_1": 0.002,
                "macd_line_12_26": 0.1,
                "volume_zscore_12": -1.0,
                "realized_vol_12": 0.001,
                "rsi_14": 50,
            }
        )
        predictions.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "row_id": row_id,
                "split": split,
                "y_true": positive,
                "y_pred": int(probability >= 0.5),
                "probability": probability,
                "model_name": "logistic_regression_tiny",
                "scenario_name": "current_fee",
            }
        )
        labels.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "row_id": row_id,
                "label": positive,
                "future_return": 0.01 if positive else -0.004,
                "cost_per_trade": 0.002,
                "scenario_name": "current_fee",
            }
        )
    _write_csv(run_dir / "training_frame" / "m20_training_frame_features.csv", features)
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
        / "fee_exceedance_labels_vol_scaled.csv",
        labels,
    )


def _write_packet(base: Path, runs: list[Path]) -> None:
    packet_dir = base / "research_labels" / "vol_scaled" / "rank_gate_evidence_packet"
    packet_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "evidence_status": "RESEARCH_CONFIRMED_RANK_GATE",
        "window_metrics": [
            {"window_label": f"run_{index}", "run_dir": str(run_dir)}
            for index, run_dir in enumerate(runs)
        ],
    }
    (packet_dir / "rank_gate_evidence_packet.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


def _write_design(base: Path) -> None:
    _write_csv(
        base
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


def test_rank_gate_economics_writes_outputs(tmp_path: Path) -> None:
    base = tmp_path / "base"
    confirmation = tmp_path / "confirmation"
    _write_run(base)
    _write_run(confirmation)
    _write_design(base)
    _write_packet(base, [base, confirmation])

    result = simulate_m20_rank_gate_economics(base_run_dir=base)

    assert "RESEARCH_ONLY" in result["honesty_flags"]
    assert result["policy_count"] == 5
    assert Path(result["output_files"]["policy_metrics_csv"]).exists()
    locked = [
        row for row in result["policy_metrics"]
        if row["policy_name"] == "CONDITION_THEN_TOP_0.25"
    ]
    assert locked
    assert locked[0]["selected_rows"] >= 1
    assert locked[0]["net_value_proxy"] != 0


def test_rank_gate_economics_records_not_profit_evidence(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_run(base)
    _write_design(base)
    _write_packet(base, [base])

    result = simulate_m20_rank_gate_economics(base_run_dir=base)

    assert "NOT_PROFIT_EVIDENCE" in result["honesty_flags"]
    assert "NOT_BACKTEST" in result["blockers"]
