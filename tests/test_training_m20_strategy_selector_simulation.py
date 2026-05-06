"""Focused tests for M20 research strategy selector simulation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.m20_strategy_selector_simulation import simulate_m20_strategy_selector

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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_design(original: Path) -> None:
    design = original / "research_labels" / "vol_scaled" / "strategy_selector_design"
    _write_json(design / "strategy_selector_candidate_spec.json", {"selector_id": "x"})
    _write_csv(
        design / "strategy_selector_condition_weights.csv",
        [
            {
                "slice_family": "symbol",
                "slice_value": "BTC/USD",
                "evidence_weight": 1.2,
                "proposed_selector_action": "ENABLE_RESEARCH_CANDIDATE",
            },
            {
                "slice_family": "momentum",
                "slice_value": "positive",
                "evidence_weight": 0.8,
                "proposed_selector_action": "ENABLE_RESEARCH_CANDIDATE",
            },
        ],
    )


def _write_run(run_dir: Path, month: str = "2025-01") -> None:
    rows = []
    preds = []
    labels = []
    for index in range(20):
        symbol = "BTC/USD" if index % 2 == 0 else "ETH/USD"
        timestamp = f"{month}-{index + 1:02d}T00:00:00Z"
        row_id = f"{symbol}|{timestamp}"
        positive = 1 if index in {0, 2, 4, 6, 8, 10} else 0
        probability = 0.9 - index * 0.02
        rows.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 1,
                "row_id": row_id,
                "high_price": 101,
                "low_price": 100,
                "close_price": 100,
                "log_return_1": 0.001 if index % 3 == 0 else -0.001,
                "macd_line_12_26": 0.1,
                "volume_zscore_12": -1.0,
                "realized_vol_12": 0.001,
                "rsi_14": 50,
            }
        )
        preds.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 1,
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
                "fold_index": 1,
                "row_id": row_id,
                "label": positive,
                "scenario_name": "current_fee",
            }
        )
    _write_csv(run_dir / "training_frame" / "m20_training_frame_features.csv", rows)
    _write_csv(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_baselines"
        / "predictions_logistic_regression_tiny_test_full.csv",
        preds,
    )
    _write_csv(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_labels_vol_scaled.csv",
        labels,
    )


def test_selector_simulation_writes_outputs_and_metrics(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    _write_design(original)
    _write_run(original)
    _write_run(confirmation)

    result = simulate_m20_strategy_selector(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    assert result["selector_id"] == "fee_exceedance_gate_v0_research"
    assert "RESEARCH_ONLY_SELECTOR_SIMULATION" in result["honesty_flags"]
    assert Path(result["output_files"]["selector_policy_metrics_by_run_csv"]).exists()
    policies = {row["policy_name"] for row in result["policy_metrics"]}
    assert "GLOBAL_LOGISTIC_TOP5" in policies
    assert "SELECTOR_WEIGHTED_CONFIRMED" in policies
    assert "SELECTOR_ANY_CONFIRMED" in policies
    assert "SELECTOR_STRICT_MULTI_CONDITION" in policies


def test_disable_gap_conditions_are_tracked(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    _write_design(original)
    _write_run(original, month="2026-04")
    _write_run(confirmation)

    result = simulate_m20_strategy_selector(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    original_weighted = [
        row for row in result["policy_metrics"]
        if row["run_label"] == "original"
        and row["policy_name"] == "SELECTOR_WEIGHTED_CONFIRMED"
    ][0]
    assert original_weighted["disable_gap_unknown_count"] == 0
    with open(
        result["output_files"]["selector_disable_gap_exposure_csv"],
        newline="",
        encoding="utf-8",
    ) as input_file:
        gap_rows = list(csv.DictReader(input_file))
    assert any(row["disable_gap_condition"] == "month=2026-04" for row in gap_rows)


def test_missing_selector_spec_fails_cleanly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MISSING_SELECTOR_SPEC"):
        simulate_m20_strategy_selector(
            original_run_dir=tmp_path / "original",
            confirmation_run_dir=tmp_path / "confirmation",
        )
