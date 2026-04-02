"""Focused tests for bounded M7 AutoGluon research experiments."""

from __future__ import annotations

import json
from pathlib import Path

from app.training.research_experiments import (
    ResearchExperimentSpec,
    discover_research_configs,
    summarize_experiment_runs,
)


def _write_research_config(config_path: Path, *, presets: str) -> None:
    config_path.write_text(
        json.dumps(
            {
                "artifact_root": "artifacts/training/m7",
                "source_table": "feature_ohlc",
                "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
                "time_column": "as_of_time",
                "interval_column": "interval_begin",
                "close_column": "close_price",
                "categorical_feature_columns": ["symbol"],
                "numeric_feature_columns": [
                    "open_price",
                    "high_price",
                    "low_price",
                    "close_price",
                    "vwap",
                    "trade_count",
                    "volume",
                    "log_return_1",
                    "log_return_3",
                    "momentum_3",
                    "return_mean_12",
                    "return_std_12",
                    "realized_vol_12",
                    "rsi_14",
                    "macd_line_12_26",
                    "volume_mean_12",
                    "volume_std_12",
                    "volume_zscore_12",
                    "close_zscore_12",
                    "lag_log_return_1",
                    "lag_log_return_2",
                    "lag_log_return_3",
                ],
                "label_horizon_candles": 3,
                "purge_gap_candles": 3,
                "test_folds": 5,
                "first_train_fraction": 0.5,
                "test_fraction": 0.1,
                "round_trip_fee_bps": 20,
                "comparison_policy": {
                    "primary_metric": "mean_long_only_net_value_proxy",
                    "max_directional_accuracy_regression": 0.01,
                    "max_brier_score_worsening": 0.01,
                },
                "models": {
                    "autogluon_tabular": {
                        "presets": presets,
                        "time_limit": 900,
                        "eval_metric": "log_loss",
                        "hyperparameters": None,
                        "fit_weighted_ensemble": True,
                        "num_bag_folds": 5,
                        "num_stack_levels": 1,
                        "num_bag_sets": 1,
                        "fold_fitting_strategy": "sequential_local",
                        "dynamic_stacking": False,
                        "calibrate_decision_threshold": False,
                        "verbosity": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_experiment_run(
    run_dir: Path,
    *,
    winner_after_cost_positive: bool,
    meets_acceptance_target: bool,
    best_policy_name: str,
    best_policy_net: float,
    best_policy_trade_count: int,
    best_policy_positive: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "winner": {"model_name": "autogluon_tabular"},
                "acceptance": {
                    "winner_after_cost_positive": winner_after_cost_positive,
                    "meets_acceptance_target": meets_acceptance_target,
                },
            }
        ),
        encoding="utf-8",
    )
    policy_dir = run_dir / "policy_candidate_analysis"
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "policy_candidate_summary.json").write_text(
        json.dumps(
            {
                "best_candidate": {
                    "policy_name": best_policy_name,
                    "mean_long_only_net_value_proxy": best_policy_net,
                    "trade_count": best_policy_trade_count,
                    "after_cost_positive": best_policy_positive,
                    "caution_text": "",
                }
            }
        ),
        encoding="utf-8",
    )


def test_discover_research_configs_returns_bounded_sorted_set(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_research_config(
        config_dir / "training.m7.research.best_quality_v150.json",
        presets="best_quality_v150",
    )
    _write_research_config(
        config_dir / "training.m7.research.high_quality.json",
        presets="high_quality",
    )
    _write_research_config(
        config_dir / "training.m7.research.best_quality.json",
        presets="best_quality",
    )

    configs = discover_research_configs(config_dir)

    assert [config.config_name for config in configs] == [
        "best_quality",
        "best_quality_v150",
        "high_quality",
    ]


def test_summarize_experiment_runs_ranks_deterministically_and_writes_outputs(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    best_quality_config = config_dir / "training.m7.research.best_quality.json"
    best_quality_v150_config = config_dir / "training.m7.research.best_quality_v150.json"
    high_quality_config = config_dir / "training.m7.research.high_quality.json"
    _write_research_config(best_quality_config, presets="best_quality")
    _write_research_config(best_quality_v150_config, presets="best_quality_v150")
    _write_research_config(high_quality_config, presets="high_quality")

    runs_root = tmp_path / "runs"
    _write_experiment_run(
        runs_root / "20260402T000001Z",
        winner_after_cost_positive=False,
        meets_acceptance_target=False,
        best_policy_name="m7_research_long_only_v1",
        best_policy_net=0.000010,
        best_policy_trade_count=18,
        best_policy_positive=True,
    )
    _write_experiment_run(
        runs_root / "20260402T000002Z",
        winner_after_cost_positive=True,
        meets_acceptance_target=False,
        best_policy_name="m7_research_long_only_v1",
        best_policy_net=0.000010,
        best_policy_trade_count=18,
        best_policy_positive=True,
    )
    _write_experiment_run(
        runs_root / "20260402T000003Z",
        winner_after_cost_positive=False,
        meets_acceptance_target=False,
        best_policy_name="default_long_only_050",
        best_policy_net=-0.000100,
        best_policy_trade_count=150,
        best_policy_positive=False,
    )

    analysis_dir = tmp_path / "analysis"
    summary = summarize_experiment_runs(
        [
            ResearchExperimentSpec(
                config_name="best_quality",
                config_path=best_quality_config,
                run_dir=runs_root / "20260402T000001Z",
            ),
            ResearchExperimentSpec(
                config_name="best_quality_v150",
                config_path=best_quality_v150_config,
                run_dir=runs_root / "20260402T000002Z",
            ),
            ResearchExperimentSpec(
                config_name="high_quality",
                config_path=high_quality_config,
                run_dir=runs_root / "20260402T000003Z",
            ),
        ],
        analysis_dir=analysis_dir,
    )

    assert summary["best_experiment"]["config_name"] == "best_quality_v150"
    assert [row["config_name"] for row in summary["experiments"]] == [
        "best_quality_v150",
        "best_quality",
        "high_quality",
    ]
    assert (analysis_dir / "experiment_summary.json").exists()
    assert (analysis_dir / "experiment_summary.csv").exists()
    assert (analysis_dir / "summary.md").exists()
