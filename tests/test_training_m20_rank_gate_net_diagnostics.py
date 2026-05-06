"""Focused tests for M20 rank-gate net-proxy diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_rank_gate_net_diagnostics import diagnose_m20_rank_gate_net

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


def _write_run(run_dir: Path, positive_return: float, negative_return: float) -> None:
    features = []
    predictions = []
    labels = []
    for index in range(100):
        symbol = "BTC/USD" if index % 2 == 0 else "ETH/USD"
        timestamp = f"2025-01-{(index % 28) + 1:02d}T00:{index % 60:02d}:00Z"
        row_id = f"{symbol}|{timestamp}"
        positive = 1 if index < 30 else 0
        probability = 0.99 - index * 0.006
        features.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "row_id": row_id,
                "high_price": 102,
                "low_price": 100,
                "close_price": 101,
                "volume": 1000 + index,
                "log_return_1": 0.001 if index % 3 else -0.001,
                "macd_line_12_26": 0.1 if index % 2 else -0.1,
                "realized_vol_12": 0.002,
            }
        )
        predictions.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "row_id": row_id,
                "split": "test",
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
                "future_return": positive_return if positive else negative_return,
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


def test_net_diagnostics_write_tail_and_bucket_outputs(tmp_path: Path) -> None:
    base = tmp_path / "base"
    confirmation = tmp_path / "confirmation"
    _write_run(base, positive_return=0.01, negative_return=-0.004)
    _write_run(confirmation, positive_return=0.004, negative_return=-0.02)
    _write_design(base)
    _write_packet(base, [base, confirmation])

    result = diagnose_m20_rank_gate_net(base_run_dir=base)

    assert "RESEARCH_ONLY" in result["honesty_flags"]
    assert "NET_PROXY_MIXED" in result["honesty_flags"]
    assert Path(result["output_files"]["tail_events_csv"]).exists()
    assert Path(result["output_files"]["by_feature_bucket_csv"]).exists()


def test_net_diagnostics_records_not_pnl_blocker(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_run(base, positive_return=0.01, negative_return=-0.004)
    _write_design(base)
    _write_packet(base, [base])

    result = diagnose_m20_rank_gate_net(base_run_dir=base)

    assert "NOT_PNL" in result["honesty_flags"]
    assert "NOT_PROFIT_EVIDENCE" in result["blockers"]
