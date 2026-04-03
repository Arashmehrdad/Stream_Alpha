"""Focused tests for named research-only M7 policy-candidate evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.policy_candidates import build_default_policy_candidates
from app.training.policy_candidate_analysis import evaluate_policy_candidates


def _write_completed_run(
    run_dir: Path,
    *,
    rows: list[dict[str, object]],
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
            "prob_up": 0.81,
            "confidence": 0.81,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": 0.010,
            "long_only_gross_value_proxy": 0.010,
            "long_only_net_value_proxy": 0.008,
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
            "prob_up": 0.82,
            "confidence": 0.82,
            "regime_label": "TREND_DOWN",
            "long_trade_taken": 1,
            "future_return_3": -0.007,
            "long_only_gross_value_proxy": -0.007,
            "long_only_net_value_proxy": -0.009,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": "row-2",
            "symbol": "SOL/USD",
            "interval_begin": "2026-04-01T00:10:00Z",
            "as_of_time": "2026-04-01T00:15:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.83,
            "confidence": 0.83,
            "regime_label": "HIGH_VOL",
            "long_trade_taken": 1,
            "future_return_3": 0.003,
            "long_only_gross_value_proxy": 0.003,
            "long_only_net_value_proxy": 0.001,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": "row-3",
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-01T00:15:00Z",
            "as_of_time": "2026-04-01T00:20:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.86,
            "confidence": 0.86,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": 0.004,
            "long_only_gross_value_proxy": 0.004,
            "long_only_net_value_proxy": 0.002,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 2,
            "row_id": "row-4",
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-01T00:20:00Z",
            "as_of_time": "2026-04-01T00:25:00Z",
            "y_true": 0,
            "y_pred": 1,
            "prob_up": 0.60,
            "confidence": 0.60,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": -0.002,
            "long_only_gross_value_proxy": -0.002,
            "long_only_net_value_proxy": -0.004,
        },
    ]


def test_named_candidate_evaluation_prefers_research_candidate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, rows=_synthetic_oof_rows())

    analysis = evaluate_policy_candidates(
        run_dir=run_dir,
        candidate_names=["default_long_only_050", "m7_research_long_only_v1"],
    )

    assert analysis["best_candidate"]["policy_name"] == "m7_research_long_only_v1"
    assert analysis["best_candidate"]["after_cost_positive"] is True


def test_blocked_regime_behavior_excludes_trend_down_and_high_vol_for_research_candidate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, rows=_synthetic_oof_rows())

    analysis = evaluate_policy_candidates(
        run_dir=run_dir,
        candidate_names=["m7_research_long_only_v1"],
    )
    candidate_results = {
        result["policy_name"]: result
        for result in analysis["candidate_results"]
    }
    research_candidate = candidate_results["m7_research_long_only_v1"]
    regime_breakdown = {
        row["regime_label"]: row for row in research_candidate["per_regime_breakdown"]
    }

    assert regime_breakdown["TREND_DOWN"]["trade_count"] == 0
    assert regime_breakdown["HIGH_VOL"]["trade_count"] == 0


def test_candidate_summary_writing_creates_expected_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, rows=_synthetic_oof_rows())

    analysis = evaluate_policy_candidates(run_dir=run_dir)

    output_files = analysis["output_files"]
    assert set(output_files) == {
        "policy_candidate_summary_json",
        "policy_candidate_summary_csv",
        "policy_candidate_fold_breakdown_csv",
        "summary_md",
    }
    for output_path in output_files.values():
        assert Path(output_path).exists()


def test_low_trade_caution_is_recorded_for_sparse_best_candidate(
    tmp_path: Path,
) -> None:
    sparse_rows = _synthetic_oof_rows()[:4]
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, rows=sparse_rows)

    analysis = evaluate_policy_candidates(run_dir=run_dir)

    assert analysis["best_candidate"]["trade_count"] < 20
    assert (
        analysis["best_candidate"]["caution_text"]
        == "Positive but too sparse to count as robust promotion evidence."
    )
    assert analysis["best_candidate"]["positive_but_sparse"] is True


def test_expanded_candidate_family_contains_bounded_regime_ablation_set() -> None:
    candidate_names = {
        candidate.name: candidate
        for candidate in build_default_policy_candidates()
    }

    assert {
        "default_long_only_050",
        "m7_research_long_only_v1",
        "range_only_080",
        "trend_up_only_070",
        "trend_up_only_080",
        "range_or_trend_up_080",
        "no_long_in_trend_down_080",
        "no_long_in_trend_down_high_vol_075",
        "no_long_in_trend_down_high_vol_080",
        "no_long_in_trend_down_high_vol_085",
        "per_regime_thresholds_v1",
    }.issubset(candidate_names)
    assert candidate_names["range_only_080"].allowed_regimes == frozenset({"RANGE"})
    assert candidate_names["trend_up_only_070"].allowed_regimes == frozenset({"TREND_UP"})
    assert candidate_names["per_regime_thresholds_v1"].per_regime_thresholds == {
        "RANGE": 0.80,
        "TREND_UP": 0.70,
    }


def test_range_only_and_trend_up_only_candidates_trade_only_their_allowed_regimes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            {
                "model_name": "autogluon_tabular",
                "fold_index": 0,
                "row_id": "range-row",
                "symbol": "BTC/USD",
                "interval_begin": "2026-04-01T00:00:00Z",
                "as_of_time": "2026-04-01T00:05:00Z",
                "y_true": 1,
                "y_pred": 1,
                "prob_up": 0.82,
                "confidence": 0.82,
                "regime_label": "RANGE",
                "long_trade_taken": 1,
                "future_return_3": 0.010,
                "long_only_gross_value_proxy": 0.010,
                "long_only_net_value_proxy": 0.008,
            },
            {
                "model_name": "autogluon_tabular",
                "fold_index": 0,
                "row_id": "trend-up-row",
                "symbol": "ETH/USD",
                "interval_begin": "2026-04-01T00:05:00Z",
                "as_of_time": "2026-04-01T00:10:00Z",
                "y_true": 1,
                "y_pred": 1,
                "prob_up": 0.75,
                "confidence": 0.75,
                "regime_label": "TREND_UP",
                "long_trade_taken": 1,
                "future_return_3": 0.009,
                "long_only_gross_value_proxy": 0.009,
                "long_only_net_value_proxy": 0.007,
            },
            {
                "model_name": "autogluon_tabular",
                "fold_index": 1,
                "row_id": "trend-down-row",
                "symbol": "SOL/USD",
                "interval_begin": "2026-04-01T00:10:00Z",
                "as_of_time": "2026-04-01T00:15:00Z",
                "y_true": 0,
                "y_pred": 1,
                "prob_up": 0.84,
                "confidence": 0.84,
                "regime_label": "TREND_DOWN",
                "long_trade_taken": 1,
                "future_return_3": -0.010,
                "long_only_gross_value_proxy": -0.010,
                "long_only_net_value_proxy": -0.012,
            },
        ],
    )

    analysis = evaluate_policy_candidates(
        run_dir=run_dir,
        candidate_names=["range_only_080", "trend_up_only_070"],
    )
    candidate_results = {
        result["policy_name"]: result
        for result in analysis["candidate_results"]
    }

    assert candidate_results["range_only_080"]["trades_in_range"] == 1
    assert candidate_results["range_only_080"]["trades_in_trend_up"] == 0
    assert candidate_results["trend_up_only_070"]["trades_in_trend_up"] == 1
    assert candidate_results["trend_up_only_070"]["trades_in_range"] == 0


def test_family_summary_flags_when_positive_candidates_depend_on_range_only_blocking(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            {
                "model_name": "autogluon_tabular",
                "fold_index": 0,
                "row_id": "range-positive-1",
                "symbol": "BTC/USD",
                "interval_begin": "2026-04-01T00:00:00Z",
                "as_of_time": "2026-04-01T00:05:00Z",
                "y_true": 1,
                "y_pred": 1,
                "prob_up": 0.82,
                "confidence": 0.82,
                "regime_label": "RANGE",
                "long_trade_taken": 1,
                "future_return_3": 0.010,
                "long_only_gross_value_proxy": 0.010,
                "long_only_net_value_proxy": 0.008,
            },
            {
                "model_name": "autogluon_tabular",
                "fold_index": 0,
                "row_id": "range-positive-2",
                "symbol": "ETH/USD",
                "interval_begin": "2026-04-01T00:05:00Z",
                "as_of_time": "2026-04-01T00:10:00Z",
                "y_true": 1,
                "y_pred": 1,
                "prob_up": 0.86,
                "confidence": 0.86,
                "regime_label": "RANGE",
                "long_trade_taken": 1,
                "future_return_3": 0.004,
                "long_only_gross_value_proxy": 0.004,
                "long_only_net_value_proxy": 0.002,
            },
            {
                "model_name": "autogluon_tabular",
                "fold_index": 1,
                "row_id": "trend-up-available",
                "symbol": "SOL/USD",
                "interval_begin": "2026-04-01T00:10:00Z",
                "as_of_time": "2026-04-01T00:15:00Z",
                "y_true": 1,
                "y_pred": 1,
                "prob_up": 0.75,
                "confidence": 0.75,
                "regime_label": "TREND_UP",
                "long_trade_taken": 1,
                "future_return_3": 0.009,
                "long_only_gross_value_proxy": 0.009,
                "long_only_net_value_proxy": 0.007,
            },
            {
                "model_name": "autogluon_tabular",
                "fold_index": 1,
                "row_id": "trend-down-negative",
                "symbol": "BTC/USD",
                "interval_begin": "2026-04-01T00:15:00Z",
                "as_of_time": "2026-04-01T00:20:00Z",
                "y_true": 0,
                "y_pred": 1,
                "prob_up": 0.84,
                "confidence": 0.84,
                "regime_label": "TREND_DOWN",
                "long_trade_taken": 1,
                "future_return_3": -0.010,
                "long_only_gross_value_proxy": -0.010,
                "long_only_net_value_proxy": -0.012,
            },
            {
                "model_name": "autogluon_tabular",
                "fold_index": 2,
                "row_id": "high-vol-negative",
                "symbol": "ETH/USD",
                "interval_begin": "2026-04-01T00:20:00Z",
                "as_of_time": "2026-04-01T00:25:00Z",
                "y_true": 0,
                "y_pred": 1,
                "prob_up": 0.82,
                "confidence": 0.82,
                "regime_label": "HIGH_VOL",
                "long_trade_taken": 1,
                "future_return_3": -0.006,
                "long_only_gross_value_proxy": -0.006,
                "long_only_net_value_proxy": -0.008,
            },
        ],
    )

    analysis = evaluate_policy_candidates(
        run_dir=run_dir,
        candidate_names=["range_only_080", "m7_research_long_only_v1"],
    )
    family_summary = analysis["candidate_family_summary"]

    assert family_summary["positivity_depends_on_trend_up_blocking"] is True
    assert family_summary["range_only_behavior_dominates_positive_candidates"] is True
    assert analysis["best_candidate"]["trend_up_blocked_entirely"] is True
