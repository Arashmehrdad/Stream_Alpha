"""Focused tests for M20 abstention/HOLD research diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_abstention_hold_research import analyze_m20_abstention_hold

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
    for index in range(120):
        symbol = "BTC/USD" if index % 2 else "ETH/USD"
        timestamp = f"2025-01-{(index % 28) + 1:02d}T00:{index % 60:02d}:00Z"
        positive = 1 if index < 20 else 0
        probability = 0.99 - index * 0.005
        features.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "high_price": 104,
                "low_price": 96 if index >= 80 else 100,
                "close_price": 100,
                "vwap": 100,
                "volume": 3000 if index >= 80 else 1000,
                "realized_vol_12": 0.03 if index >= 80 else 0.001,
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
        {"window_metrics": [{"window_label": "base", "run_dir": str(run_dir)}]},
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_abstention_hold_writes_outputs(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_abstention_hold(base_run_dir=base)

    assert "RESEARCH_ONLY" in result["honesty_flags"]
    assert result["rule_count"] == 6
    assert Path(result["output_files"]["hold_rule_metrics_csv"]).exists()
    assert Path(result["output_files"]["avoided_loss_proxy_csv"]).exists()


def test_abstention_hold_metrics_are_deterministic(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_abstention_hold(base_run_dir=base)
    rows = _read_csv(Path(result["output_files"]["hold_rule_metrics_csv"]))
    negative_net = [row for row in rows if row["rule_name"] == "HOLD_SELECTED_NEGATIVE_NET_PROXY"]

    assert negative_net
    assert float(negative_net[0]["avoided_negative_net_proxy"]) >= 0.0
    assert int(negative_net[0]["selected_rows_before_hold"]) == 1


def test_abstention_hold_preserves_research_only_flags(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_abstention_hold(base_run_dir=base)

    assert "NO_RUNTIME" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]
    assert result["recommendation"] in {
        "TEST_ABSTENTION_WITH_STRATEGY_FAMILIES_NEXT",
        "KEEP_ABSTENTION_AS_RESEARCH_FILTER",
        "NEED_RICHER_BAD_REGIME_LABELS",
        "REJECT_CURRENT_ABSTENTION_RULES",
    }
