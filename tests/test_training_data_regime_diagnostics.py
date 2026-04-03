"""Focused tests for completed-run M7 data and regime diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.data_regime_diagnostics import analyze_completed_run


def _write_completed_run(
    run_dir: Path,
    *,
    rows: list[dict[str, object]],
    fold_metric_rows: list[dict[str, object]],
    winner_model_name: str = "autogluon_tabular",
    fee_rate: float = 0.002,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "economics_contract": {"fee_rate": fee_rate},
                "winner": {"model_name": winner_model_name},
            }
        ),
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
    with (run_dir / "fold_metrics.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "model_name",
                "fold_index",
                "train_rows",
                "test_rows",
                "directional_accuracy",
                "precision_class_0",
                "precision_class_1",
                "recall_class_0",
                "recall_class_1",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
                "brier_score",
                "trade_count",
                "trade_rate",
                "mean_long_only_gross_value_proxy",
                "mean_long_only_net_value_proxy",
            ],
        )
        writer.writeheader()
        writer.writerows(fold_metric_rows)
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "eligible_rows": len(rows),
                "unique_timestamps": 8,
                "feature_columns": [
                    "open_price",
                    "close_price",
                    "return_mean_12",
                    "realized_vol_12",
                ],
                "label_counts": {"0": 4, "1": 4},
                "symbols": {"BTC/USD": 3, "ETH/USD": 3, "SOL/USD": 2},
            }
        ),
        encoding="utf-8",
    )


def _synthetic_oof_rows() -> list[dict[str, object]]:
    return [
        {
            "model_name": "autogluon_tabular",
            "fold_index": 0,
            "row_id": "row-0",
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-02T00:00:00Z",
            "as_of_time": "2026-04-02T00:05:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.82,
            "confidence": 0.82,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": 0.0050,
            "long_only_gross_value_proxy": 0.0050,
            "long_only_net_value_proxy": 0.0030,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 0,
            "row_id": "row-1",
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-02T00:05:00Z",
            "as_of_time": "2026-04-02T00:10:00Z",
            "y_true": 0,
            "y_pred": 1,
            "prob_up": 0.72,
            "confidence": 0.72,
            "regime_label": "TREND_DOWN",
            "long_trade_taken": 1,
            "future_return_3": -0.0030,
            "long_only_gross_value_proxy": -0.0030,
            "long_only_net_value_proxy": -0.0050,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 0,
            "row_id": "row-2",
            "symbol": "SOL/USD",
            "interval_begin": "2026-04-02T00:10:00Z",
            "as_of_time": "2026-04-02T00:15:00Z",
            "y_true": 0,
            "y_pred": 1,
            "prob_up": 0.68,
            "confidence": 0.68,
            "regime_label": "TREND_DOWN",
            "long_trade_taken": 1,
            "future_return_3": -0.0040,
            "long_only_gross_value_proxy": -0.0040,
            "long_only_net_value_proxy": -0.0060,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": "row-3",
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-02T00:15:00Z",
            "as_of_time": "2026-04-02T00:20:00Z",
            "y_true": 0,
            "y_pred": 1,
            "prob_up": 0.81,
            "confidence": 0.81,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": -0.0010,
            "long_only_gross_value_proxy": -0.0010,
            "long_only_net_value_proxy": -0.0030,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": "row-4",
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-02T00:20:00Z",
            "as_of_time": "2026-04-02T00:25:00Z",
            "y_true": 1,
            "y_pred": 0,
            "prob_up": 0.48,
            "confidence": 0.52,
            "regime_label": "HIGH_VOL",
            "long_trade_taken": 0,
            "future_return_3": 0.0030,
            "long_only_gross_value_proxy": 0.0000,
            "long_only_net_value_proxy": 0.0000,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": "row-5",
            "symbol": "SOL/USD",
            "interval_begin": "2026-04-02T00:25:00Z",
            "as_of_time": "2026-04-02T00:30:00Z",
            "y_true": 1,
            "y_pred": 0,
            "prob_up": 0.44,
            "confidence": 0.56,
            "regime_label": "TREND_UP",
            "long_trade_taken": 0,
            "future_return_3": 0.0010,
            "long_only_gross_value_proxy": 0.0000,
            "long_only_net_value_proxy": 0.0000,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 2,
            "row_id": "row-6",
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-02T00:30:00Z",
            "as_of_time": "2026-04-02T00:35:00Z",
            "y_true": 0,
            "y_pred": 1,
            "prob_up": 0.77,
            "confidence": 0.77,
            "regime_label": "TREND_DOWN",
            "long_trade_taken": 1,
            "future_return_3": -0.0020,
            "long_only_gross_value_proxy": -0.0020,
            "long_only_net_value_proxy": -0.0040,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 2,
            "row_id": "row-7",
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-02T00:35:00Z",
            "as_of_time": "2026-04-02T00:40:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.88,
            "confidence": 0.88,
            "regime_label": "TREND_UP",
            "long_trade_taken": 1,
            "future_return_3": 0.0045,
            "long_only_gross_value_proxy": 0.0045,
            "long_only_net_value_proxy": 0.0025,
        },
    ]


def _synthetic_fold_metrics() -> list[dict[str, object]]:
    return [
        {
            "model_name": "autogluon_tabular",
            "fold_index": 0,
            "train_rows": 100,
            "test_rows": 3,
            "directional_accuracy": 0.60,
            "precision_class_0": 0.50,
            "precision_class_1": 0.67,
            "recall_class_0": 0.50,
            "recall_class_1": 0.67,
            "true_negative": 1,
            "false_positive": 1,
            "false_negative": 0,
            "true_positive": 1,
            "brier_score": 0.24,
            "trade_count": 3,
            "trade_rate": 1.0,
            "mean_long_only_gross_value_proxy": -0.0007,
            "mean_long_only_net_value_proxy": -0.0027,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "train_rows": 103,
            "test_rows": 3,
            "directional_accuracy": 0.33,
            "precision_class_0": 0.33,
            "precision_class_1": 0.33,
            "recall_class_0": 0.33,
            "recall_class_1": 0.33,
            "true_negative": 1,
            "false_positive": 1,
            "false_negative": 1,
            "true_positive": 0,
            "brier_score": 0.31,
            "trade_count": 1,
            "trade_rate": 0.33,
            "mean_long_only_gross_value_proxy": -0.0003,
            "mean_long_only_net_value_proxy": -0.0010,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 2,
            "train_rows": 106,
            "test_rows": 2,
            "directional_accuracy": 0.50,
            "precision_class_0": 0.50,
            "precision_class_1": 0.50,
            "recall_class_0": 0.50,
            "recall_class_1": 0.50,
            "true_negative": 1,
            "false_positive": 0,
            "false_negative": 1,
            "true_positive": 0,
            "brier_score": 0.28,
            "trade_count": 2,
            "trade_rate": 1.0,
            "mean_long_only_gross_value_proxy": 0.0013,
            "mean_long_only_net_value_proxy": -0.0007,
        },
    ]


def test_opportunity_density_counts_overall_and_by_regime(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=_synthetic_oof_rows(),
        fold_metric_rows=_synthetic_fold_metrics(),
    )

    diagnostics = analyze_completed_run(run_dir=run_dir)

    overall_by_threshold = {
        row["threshold_bps"]: row
        for row in diagnostics["opportunity_density"]["overall"]
    }
    regime_rows = [
        row
        for row in diagnostics["opportunity_density"]["by_regime"]
        if row["threshold_bps"] == 20
    ]
    regime_by_name = {row["regime_label"]: row for row in regime_rows}

    assert overall_by_threshold[0]["opportunity_count"] == 4
    assert overall_by_threshold[20]["opportunity_count"] == 3
    assert diagnostics["opportunity_density"]["twenty_bps_sparse"] is True
    assert regime_by_name["TREND_UP"]["opportunity_count"] == 1
    assert regime_by_name["TREND_DOWN"]["opportunity_count"] == 0


def test_regime_routing_highlights_trend_down_dominance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=_synthetic_oof_rows(),
        fold_metric_rows=_synthetic_fold_metrics(),
    )

    diagnostics = analyze_completed_run(run_dir=run_dir)

    assert (
        "TREND_DOWN accounts for at least half of default long-only trades."
        in diagnostics["regime_routing"]["suspicious_findings"]
    )
    policy_rows = {
        row["policy_name"]: row for row in diagnostics["regime_routing"]["policy_results"]
    }
    research_regimes = {
        row["regime_label"]: row
        for row in policy_rows["m7_research_long_only_v1"]["per_regime_breakdown"]
    }

    assert research_regimes["TREND_DOWN"]["trade_count"] == 0
    assert research_regimes["TREND_UP"]["trade_count"] == 1


def test_weakest_fold_selection_uses_best_named_candidate_economics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=_synthetic_oof_rows(),
        fold_metric_rows=_synthetic_fold_metrics(),
    )

    diagnostics = analyze_completed_run(run_dir=run_dir)

    assert diagnostics["best_named_candidate"]["policy_name"] == "m7_research_long_only_v1"
    assert diagnostics["fold_diagnostics"]["weakest_fold"]["fold_index"] == 1
    assert (
        diagnostics["fold_diagnostics"]["weakest_fold"][
            "best_named_mean_long_only_net_value_proxy"
        ]
        < 0.0
    )


def test_missing_feature_shift_support_is_warned_honestly(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=_synthetic_oof_rows(),
        fold_metric_rows=_synthetic_fold_metrics(),
    )

    diagnostics = analyze_completed_run(run_dir=run_dir)

    assert diagnostics["feature_shift_support"]["available"] is False
    assert "Completed artifacts do not support feature-shift diagnostics." in diagnostics["warnings"]
    for output_path in diagnostics["output_files"].values():
        assert Path(output_path).exists()
