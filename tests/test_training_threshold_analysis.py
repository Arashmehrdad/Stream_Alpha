"""Focused tests for post-training M7 threshold and regime analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.threshold_analysis import (
    analyze_completed_run,
    select_best_policy_result,
)


def _write_completed_run(
    run_dir: Path,
    *,
    rows: list[dict[str, object]],
    winner_model_name: str = "autogluon_tabular",
    fee_rate: float = 0.002,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "economics_contract": {"fee_rate": fee_rate},
        "winner": {"model_name": winner_model_name},
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload),
        encoding="utf-8",
    )
    with (run_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "model_name",
                "fold_index",
                "row_id",
                "symbol",
                "interval_begin",
                "as_of_time",
                "y_true",
                "y_pred",
                "prob_up",
                "confidence",
                "regime_label",
                "long_trade_taken",
                "future_return_3",
                "long_only_gross_value_proxy",
                "long_only_net_value_proxy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_oof_rows() -> list[dict[str, object]]:
    return [
        {
            "model_name": "autogluon_tabular",
            "fold_index": 0,
            "row_id": "row-0",
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-01T00:00:00Z",
            "as_of_time": "2026-04-01T00:05:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.55,
            "confidence": 0.55,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": 0.004,
            "long_only_gross_value_proxy": 0.004,
            "long_only_net_value_proxy": 0.002,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 0,
            "row_id": "row-1",
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-01T00:05:00Z",
            "as_of_time": "2026-04-01T00:10:00Z",
            "y_true": 0,
            "y_pred": 1,
            "prob_up": 0.52,
            "confidence": 0.52,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": -0.001,
            "long_only_gross_value_proxy": -0.001,
            "long_only_net_value_proxy": -0.003,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": "row-2",
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-01T00:10:00Z",
            "as_of_time": "2026-04-01T00:15:00Z",
            "y_true": 0,
            "y_pred": 1,
            "prob_up": 0.80,
            "confidence": 0.80,
            "regime_label": "TREND_DOWN",
            "long_trade_taken": 1,
            "future_return_3": -0.005,
            "long_only_gross_value_proxy": -0.005,
            "long_only_net_value_proxy": -0.007,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": "row-3",
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-01T00:15:00Z",
            "as_of_time": "2026-04-01T00:20:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.75,
            "confidence": 0.75,
            "regime_label": "TREND_UP",
            "long_trade_taken": 1,
            "future_return_3": 0.006,
            "long_only_gross_value_proxy": 0.006,
            "long_only_net_value_proxy": 0.004,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 2,
            "row_id": "row-4",
            "symbol": "SOL/USD",
            "interval_begin": "2026-04-01T00:20:00Z",
            "as_of_time": "2026-04-01T00:25:00Z",
            "y_true": 1,
            "y_pred": 0,
            "prob_up": 0.48,
            "confidence": 0.52,
            "regime_label": "HIGH_VOL",
            "long_trade_taken": 0,
            "future_return_3": 0.003,
            "long_only_gross_value_proxy": 0.0,
            "long_only_net_value_proxy": 0.0,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 2,
            "row_id": "row-5",
            "symbol": "SOL/USD",
            "interval_begin": "2026-04-01T00:25:00Z",
            "as_of_time": "2026-04-01T00:30:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.72,
            "confidence": 0.72,
            "regime_label": "HIGH_VOL",
            "long_trade_taken": 1,
            "future_return_3": 0.005,
            "long_only_gross_value_proxy": 0.005,
            "long_only_net_value_proxy": 0.003,
        },
    ]


def test_threshold_sweep_calculations_identify_better_stricter_threshold(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, rows=_synthetic_oof_rows())

    analysis = analyze_completed_run(
        run_dir=run_dir,
        thresholds=(0.50, 0.70, 0.90),
    )

    best_global = analysis["best_global_threshold_policy"]
    assert best_global["threshold"] == 0.70
    assert best_global["trade_count"] == 3
    assert best_global["mean_long_only_net_value_proxy"] == 0.0
    assert best_global["after_cost_positive"] is False


def test_trend_down_block_policy_can_rescue_after_cost_net_value(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, rows=_synthetic_oof_rows())

    analysis = analyze_completed_run(
        run_dir=run_dir,
        thresholds=(0.50, 0.70),
    )

    best_trend_down_block = analysis["best_no_long_in_trend_down_policy"]
    assert best_trend_down_block["threshold"] == 0.70
    assert best_trend_down_block["trade_count"] == 2
    assert best_trend_down_block["after_cost_positive"] is True
    assert best_trend_down_block["mean_long_only_net_value_proxy"] == 0.0011666666666666668
    regime_breakdown = {
        row["regime_label"]: row for row in best_trend_down_block["per_regime_breakdown"]
    }
    assert regime_breakdown["TREND_DOWN"]["trade_count"] == 0


def test_fold_breakdown_is_present_for_each_analyzed_policy_threshold(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, rows=_synthetic_oof_rows())

    analysis = analyze_completed_run(
        run_dir=run_dir,
        thresholds=(0.70,),
    )

    best_overall = analysis["best_overall_policy"]
    fold_breakdown = best_overall["per_fold_breakdown"]
    assert [row["fold_index"] for row in fold_breakdown] == [0, 1, 2]
    assert all("mean_long_only_net_value_proxy" in row for row in fold_breakdown)
    assert analysis["worst_fold_for_best_overall"]["fold_index"] == 0


def test_analysis_writes_expected_artifacts_into_run_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, rows=_synthetic_oof_rows())

    analysis = analyze_completed_run(
        run_dir=run_dir,
        thresholds=(0.50, 0.70),
    )

    output_files = analysis["output_files"]
    expected_files = {
        "threshold_sweep_json",
        "threshold_sweep_csv",
        "regime_policy_comparison_json",
        "regime_policy_comparison_csv",
        "fold_policy_breakdown_csv",
        "summary_md",
    }
    assert expected_files == set(output_files)
    for path_value in output_files.values():
        assert Path(path_value).exists()


def test_best_policy_selection_ignores_zero_trade_candidate_when_trade_candidate_exists() -> None:
    zero_trade_result = {
        "policy_name": "baseline_threshold_only",
        "threshold": 0.90,
        "trade_count": 0,
        "mean_long_only_net_value_proxy": 0.0,
        "cumulative_long_only_net_value_proxy": 0.0,
        "blocked_regimes": [],
        "per_regime_thresholds": None,
    }
    traded_result = {
        "policy_name": "baseline_threshold_only",
        "threshold": 0.70,
        "trade_count": 3,
        "mean_long_only_net_value_proxy": -0.0003,
        "cumulative_long_only_net_value_proxy": -0.001,
        "blocked_regimes": [],
        "per_regime_thresholds": None,
    }

    selected = select_best_policy_result([zero_trade_result, traded_result])

    assert selected["threshold"] == 0.70
