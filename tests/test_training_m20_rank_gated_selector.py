"""Focused tests for M20 research rank-gated selector evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_rank_gated_selector import tune_m20_rank_gated_selector

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


def _write_design(original: Path) -> None:
    design = original / "research_labels" / "vol_scaled" / "strategy_selector_design"
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
    features = []
    predictions = []
    labels = []
    for index in range(100):
        symbol = "BTC/USD" if index % 2 == 0 else "ETH/USD"
        timestamp = f"{month}-{(index % 28) + 1:02d}T00:{index % 60:02d}:00Z"
        row_id = f"{symbol}|{timestamp}"
        positive = 1 if index < 20 else 0
        probability = 0.99 - index * 0.007
        features.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "row_id": row_id,
                "open_price": 100,
                "high_price": 101,
                "low_price": 100,
                "close_price": 100,
                "volume": 1000 + index,
                "log_return_1": 0.002 if index % 3 == 0 else -0.001,
                "macd_line_12_26": 0.1,
                "volume_zscore_12": -1.0 if index % 4 == 0 else 0.5,
                "realized_vol_12": 0.001 + index * 0.000001,
                "rsi_14": 50,
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


def test_rank_gated_selector_writes_outputs_and_policy_metrics(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    _write_design(original)
    _write_run(original)
    _write_run(confirmation)

    result = tune_m20_rank_gated_selector(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    assert "RESEARCH_ONLY_RANK_GATED_SELECTOR" in result["honesty_flags"]
    assert Path(result["output_files"]["manifest_json"]).exists()
    policies = {row["policy_name"] for row in result["metrics"]}
    assert "GLOBAL_TOP_5" in policies
    assert "CONDITION_THEN_TOP_5" in policies
    assert "TOP_5_WITH_2_CONDITIONS" in policies
    assert "PER_CONDITION_TOP_5" in policies
    assert "DISABLE_GAP_FILTERED_TOP_5" in policies


def test_rank_gated_selector_tracks_disable_gap_exposure(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    _write_design(original)
    _write_run(original, month="2026-04")
    _write_run(confirmation)

    result = tune_m20_rank_gated_selector(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    gap_rows = result["gaps"]
    assert any(
        row["run_label"] == "original"
        and row["disable_gap_condition"] == "month=2026-04"
        and row["encountered_rows"] == 100
        for row in gap_rows
    )
    filtered = [
        row for row in result["metrics"]
        if row["run_label"] == "original"
        and row["policy_name"] == "DISABLE_GAP_FILTERED_TOP_5"
    ][0]
    assert filtered["disable_gap_exposure"] == 0


def test_rank_gated_selector_comparison_is_deterministic(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    _write_design(original)
    _write_run(original)
    _write_run(confirmation)

    first = tune_m20_rank_gated_selector(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )
    second = tune_m20_rank_gated_selector(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    assert first["comparison"] == second["comparison"]
    assert first["recommendation"] == second["recommendation"]
