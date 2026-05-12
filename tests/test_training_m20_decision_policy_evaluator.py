"""Tests for M20 decision-policy evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_decision_policy_evaluator import evaluate_m20_decision_policies

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixtures(base: Path) -> tuple[Path, Path]:
    source = base / "source"
    prediction = base / "prediction"
    research = source / "research_labels" / "vol_scaled"
    oof_rows = []
    outcome_rows = []
    candidate_rows = []
    for index in range(1200):
        symbol = "BTC" if index % 2 == 0 else "ETH"
        timestamp = f"2024-01-{(index % 28) + 1:02d}T00:00:00Z"
        net = 0.01 if index < 700 else -0.02
        oof_rows.append(
            {
                "model_name": "model_a",
                "fold_index": 1,
                "symbol": symbol,
                "interval_begin": timestamp,
                "prob_up": 0.8 if index < 700 else 0.4,
                "confidence": 0.8 if index < 600 else 0.3,
                "regime_label": "TREND_UP" if index < 700 else "RANGE",
                "long_trade_taken": 1,
                "y_true": 1 if index < 700 else 0,
            }
        )
        outcome_rows.append(
            {
                "fold_index": 1,
                "symbol": symbol,
                "interval_begin": timestamp,
                "net_value_proxy": net,
                "gross_value_proxy": net + 0.002,
                "fee_exceedance_label": 1 if net > 0 else 0,
                "triple_barrier_label": 1 if net > 0 else -1,
            }
        )
        if index < 800:
            candidate_rows.append(
                {
                    "fold_index": 1,
                    "symbol": symbol,
                    "interval_begin": timestamp,
                    "candidate_name": "candidate_a",
                }
            )
    _write_csv(prediction / "oof_predictions.csv", oof_rows)
    _write_csv(
        research / "economic_outcome_artifacts" / "economic_outcomes.csv",
        outcome_rows,
    )
    _write_csv(
        research / "strategy_candidate_v2_refined_factory" / "strategy_candidates_v2.csv",
        candidate_rows,
    )
    return source, prediction


def test_decision_policy_evaluator_builds_generic_policy_metrics(tmp_path: Path) -> None:
    source, prediction = _fixtures(tmp_path)

    result = evaluate_m20_decision_policies(
        source_run_dir=source,
        prediction_run_dir=prediction,
    )
    families = {row["policy_family"] for row in result["policy_metrics"]}

    assert "PROBABILITY_THRESHOLD" in families
    assert "CANDIDATE_EVENT_POLICY" in families
    assert result["policy_count"] > 1
    assert result["best_policy_candidate"]


def test_decision_policy_evaluator_outputs_baseline_and_calibration(tmp_path: Path) -> None:
    source, prediction = _fixtures(tmp_path)

    result = evaluate_m20_decision_policies(
        source_run_dir=source,
        prediction_run_dir=prediction,
    )

    assert result["baseline_comparison"]
    assert result["calibration_metrics"][0]["label_column"] == "y_true"
    assert "NO_RUNTIME_EFFECT" in result["overall_status"]
    assert "NOT_PROMOTABLE" in result["overall_status"]
    assert "NO_PROFIT_CLAIM" in result["overall_status"]


def test_decision_policy_evaluator_uses_trading_aware_labels_for_calibration(
    tmp_path: Path,
) -> None:
    source, prediction = _fixtures(tmp_path)
    label_dir = source / "research_labels" / "vol_scaled" / "trading_aware_labels"
    _write_csv(
        label_dir / "trading_aware_labels.csv",
        [
            {
                "fold_index": 1,
                "symbol": "BTC" if index % 2 == 0 else "ETH",
                "interval_begin": f"2024-01-{(index % 28) + 1:02d}T00:00:00Z",
                "fee_plus_slippage_exceedance_label": 1 if index < 700 else 0,
            }
            for index in range(1200)
        ],
    )

    result = evaluate_m20_decision_policies(
        source_run_dir=source,
        prediction_run_dir=prediction,
        trading_aware_label_dir=label_dir,
        output_name="trading_aware_policy_eval",
    )
    labels = {row["label_column"] for row in result["calibration_metrics"]}

    assert "trading_aware_label" in labels
