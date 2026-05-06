"""Focused tests for M20 volatility combo economics diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_volatility_combo_economics import (
    analyze_m20_volatility_combo_economics,
)

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


def _write_run(run_dir: Path, label_prefix: str = "") -> None:
    features = []
    labels = []
    predictions = []
    for index in range(120):
        symbol = "BTC/USD" if index % 2 else "ETH/USD"
        timestamp = f"2025-01-{(index % 28) + 1:02d}T00:{index % 60:02d}:00Z"
        positive = 1 if index < 30 or index >= 100 else 0
        probability = 0.99 - index * 0.005
        high_vol = index >= 80
        features.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "high_price": 106 if high_vol else 102,
                "low_price": 94 if high_vol else 100,
                "close_price": 100,
                "vwap": 100,
                "volume": 3000 if high_vol else 1000 + index,
                "realized_vol_12": 0.02 + index * 0.0001 if high_vol else 0.001,
                "log_return_1": 0.01 if positive else -0.001,
                "momentum_3": 0.01 if positive else -0.001,
                "macd_line_12_26": 0.1 if positive else -0.1,
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
        {
            "window_metrics": [
                {
                    "window_label": f"{label_prefix}base",
                    "run_dir": str(run_dir),
                }
            ]
        },
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_volatility_combo_writes_outputs(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_volatility_combo_economics(base_run_dir=base)

    assert "RESEARCH_ONLY" in result["honesty_flags"]
    assert result["policy_count"] == 9
    assert Path(result["output_files"]["policy_metrics_csv"]).exists()
    assert Path(result["output_files"]["stability_csv"]).exists()


def test_volatility_combo_metrics_are_deterministic(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_volatility_combo_economics(base_run_dir=base)
    rows = _read_csv(Path(result["output_files"]["policy_metrics_csv"]))
    gate_rows = [
        row for row in rows
        if row["policy_name"] == "RANK_GATE_ONLY_CONDITION_THEN_TOP_0.25"
    ]

    assert gate_rows
    assert int(gate_rows[0]["selected_rows"]) == 1
    assert float(gate_rows[0]["lift"]) >= 1.0
    assert "NO_RUNTIME" in result["honesty_flags"]


def test_volatility_combo_stability_classifies_policies(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_volatility_combo_economics(base_run_dir=base)
    rows = _read_csv(Path(result["output_files"]["stability_csv"]))
    classifications = {row["policy_name"]: row["classification"] for row in rows}

    assert "RANK_GATE_AND_VOLATILITY" in classifications
    assert result["recommendation"] in {
        "TEST_VOLATILITY_COMBO_CONFIRMATION_WINDOW",
        "TRY_VOLATILITY_AS_OPTIONAL_GATE_FILTER",
        "KEEP_VOLATILITY_LABEL_LIFT_ONLY",
        "REJECT_VOLATILITY_COMBO_FOR_NOW",
    }
