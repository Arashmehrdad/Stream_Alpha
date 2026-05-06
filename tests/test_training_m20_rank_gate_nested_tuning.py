"""Focused tests for M20 nested rank-gate tuning."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_rank_gate_nested_tuning import tune_m20_rank_gate_nested

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
    _write_csv(
        original
        / "research_labels"
        / "vol_scaled"
        / "strategy_selector_design"
        / "strategy_selector_condition_weights.csv",
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
    _write_csv(
        original
        / "research_labels"
        / "vol_scaled"
        / "rank_gated_selector"
        / "comparison.csv",
        [
            {
                "policy_name": "CONDITION_THEN_TOP_1",
                "original_coverage": 0.01,
                "confirmation_coverage": 0.01,
                "original_precision": 0.3,
                "confirmation_precision": 0.4,
                "original_lift": 1.5,
                "confirmation_lift": 1.6,
            }
        ],
    )


def _write_run(run_dir: Path, month: str = "2025-01") -> None:
    # pylint: disable=too-many-locals
    features = []
    labels = []
    split_rows = {"train": [], "validation": [], "test": []}
    for split, offset in (("train", 0), ("validation", 100), ("test", 200)):
        for index in range(120):
            absolute = offset + index
            symbol = "BTC/USD" if index % 2 == 0 else "ETH/USD"
            timestamp = f"{month}-{(index % 28) + 1:02d}T00:{index % 60:02d}:00Z"
            row_id = f"{symbol}|{timestamp}|{split}"
            positive = 1 if index < 24 else 0
            probability = 0.99 - index * 0.006
            features.append(
                {
                    "symbol": symbol,
                    "interval_begin": timestamp,
                    "fold_index": 4,
                    "row_id": row_id,
                    "high_price": 101,
                    "low_price": 100,
                    "close_price": 100,
                    "volume": 1000 + absolute,
                    "log_return_1": 0.002 if index % 3 == 0 else -0.001,
                    "macd_line_12_26": 0.1,
                    "volume_zscore_12": -1.0,
                    "realized_vol_12": 0.001,
                    "rsi_14": 50,
                }
            )
            split_rows[split].append(
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
                    "scenario_name": "current_fee",
                }
            )
    _write_csv(run_dir / "training_frame" / "m20_training_frame_features.csv", features)
    base = run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_baselines"
    for split, rows in split_rows.items():
        _write_csv(base / f"predictions_logistic_regression_tiny_{split}_full.csv", rows)
    _write_csv(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_labels_vol_scaled.csv",
        labels,
    )


def test_nested_tuning_writes_locked_outputs(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    _write_design(original)
    _write_run(original)
    _write_run(confirmation)

    result = tune_m20_rank_gate_nested(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    assert "NESTED_HELDOUT_TUNING" in result["honesty_flags"]
    assert result["selected_params"]["validation_policy_name"]
    assert result["locked_test_metrics"][0]["run_label"] == "original_test"
    assert result["confirmation_metrics"][0]["run_label"] == "confirmation_test"
    assert Path(result["output_files"]["validation_grid_csv"]).exists()
    assert Path(result["output_files"]["recommendation_json"]).exists()


def test_nested_tuning_tracks_disable_gaps(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    _write_design(original)
    _write_run(original, month="2026-04")
    _write_run(confirmation)

    result = tune_m20_rank_gate_nested(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    assert any(
        row["run_label"] == "original_test"
        and row["disable_gap_condition"] == "month=2026-04"
        and row["encountered_rows"] == 120
        for row in result["disable_gap_exposure"]
    )


def test_nested_tuning_is_deterministic(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    _write_design(original)
    _write_run(original)
    _write_run(confirmation)

    first = tune_m20_rank_gate_nested(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )
    second = tune_m20_rank_gate_nested(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    assert first["selected_params"] == second["selected_params"]
    assert first["stability"] == second["stability"]
