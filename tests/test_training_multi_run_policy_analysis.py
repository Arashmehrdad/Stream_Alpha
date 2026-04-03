"""Focused tests for multi-run named M7 policy-candidate analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.multi_run_policy_analysis import analyze_policy_candidates_across_runs


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


def _write_legacy_completed_run(
    run_dir: Path,
    *,
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
                "future_return_3",
                "gross_value_proxy",
                "net_value_proxy",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_name": winner_model_name,
                "fold_index": 0,
                "row_id": "legacy-row",
                "symbol": "BTC/USD",
                "interval_begin": "2026-04-01T00:00:00Z",
                "as_of_time": "2026-04-01T00:05:00Z",
                "y_true": 1,
                "y_pred": 1,
                "prob_up": 0.81,
                "confidence": 0.81,
                "future_return_3": 0.01,
                "gross_value_proxy": 0.01,
                "net_value_proxy": 0.008,
            }
        )


def _run_rows(run_name: str, futures: tuple[float, ...]) -> list[dict[str, object]]:
    return [
        {
            "model_name": "autogluon_tabular",
            "fold_index": 0,
            "row_id": f"{run_name}-range-positive",
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-01T00:00:00Z",
            "as_of_time": "2026-04-01T00:05:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.81,
            "confidence": 0.81,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": futures[0],
            "long_only_gross_value_proxy": futures[0],
            "long_only_net_value_proxy": futures[0] - 0.002,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 0,
            "row_id": f"{run_name}-trend-down",
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-01T00:05:00Z",
            "as_of_time": "2026-04-01T00:10:00Z",
            "y_true": 0,
            "y_pred": 1,
            "prob_up": 0.82,
            "confidence": 0.82,
            "regime_label": "TREND_DOWN",
            "long_trade_taken": 1,
            "future_return_3": futures[1],
            "long_only_gross_value_proxy": futures[1],
            "long_only_net_value_proxy": futures[1] - 0.002,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": f"{run_name}-high-vol",
            "symbol": "SOL/USD",
            "interval_begin": "2026-04-01T00:10:00Z",
            "as_of_time": "2026-04-01T00:15:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.83,
            "confidence": 0.83,
            "regime_label": "HIGH_VOL",
            "long_trade_taken": 1,
            "future_return_3": futures[2],
            "long_only_gross_value_proxy": futures[2],
            "long_only_net_value_proxy": futures[2] - 0.002,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": f"{run_name}-range-secondary",
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-01T00:15:00Z",
            "as_of_time": "2026-04-01T00:20:00Z",
            "y_true": 1 if futures[3] > 0 else 0,
            "y_pred": 1,
            "prob_up": 0.86,
            "confidence": 0.86,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": futures[3],
            "long_only_gross_value_proxy": futures[3],
            "long_only_net_value_proxy": futures[3] - 0.002,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 2,
            "row_id": f"{run_name}-range-low-threshold",
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-01T00:20:00Z",
            "as_of_time": "2026-04-01T00:25:00Z",
            "y_true": 0,
            "y_pred": 1,
            "prob_up": 0.60,
            "confidence": 0.60,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": futures[4],
            "long_only_gross_value_proxy": futures[4],
            "long_only_net_value_proxy": futures[4] - 0.002,
        },
    ]


def test_multi_run_aggregation_prefers_research_candidate_by_median_net(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "m7"
    _write_completed_run(
        artifact_root / "20260401T000001Z",
        rows=_run_rows("run1", (0.010, -0.008, 0.003, 0.004, -0.003)),
    )
    _write_completed_run(
        artifact_root / "20260401T000002Z",
        rows=_run_rows("run2", (0.006, -0.006, 0.001, 0.005, -0.002)),
    )
    _write_completed_run(
        artifact_root / "20260401T000003Z",
        rows=_run_rows("run3", (-0.004, -0.007, 0.002, 0.003, -0.001)),
    )

    analysis = analyze_policy_candidates_across_runs(
        artifact_root=artifact_root,
        candidate_names=["default_long_only_050", "m7_research_long_only_v1"],
    )

    assert analysis["best_candidate"]["policy_name"] == "m7_research_long_only_v1"
    assert analysis["best_candidate"]["positive_run_count"] == 2
    assert analysis["best_candidate"]["positive_run_rate"] == 2 / 3
    assert analysis["best_candidate"]["total_trade_count"] == 6


def test_incomplete_runs_are_excluded_from_multi_run_analysis(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "m7"
    _write_completed_run(
        artifact_root / "20260401T000001Z",
        rows=_run_rows("run1", (0.010, -0.008, 0.003, 0.004, -0.003)),
    )
    incomplete_run = artifact_root / "20260401T000004Z"
    incomplete_run.mkdir(parents=True, exist_ok=True)
    (incomplete_run / "summary.json").write_text(
        json.dumps({"winner": {"model_name": "autogluon_tabular"}}),
        encoding="utf-8",
    )

    analysis = analyze_policy_candidates_across_runs(
        artifact_root=artifact_root,
        candidate_names=["default_long_only_050", "m7_research_long_only_v1"],
    )

    assert analysis["scanned_run_count"] == 2
    assert analysis["complete_run_count"] == 1
    assert analysis["skipped_runs"] == [
        {
            "run_id": "20260401T000004Z",
            "reason": "missing required files: oof_predictions.csv",
        }
    ]


def test_sparse_evidence_warnings_are_recorded(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "m7"
    _write_completed_run(
        artifact_root / "20260401T000001Z",
        rows=_run_rows("run1", (0.010, -0.008, 0.003, 0.004, -0.003)),
    )
    _write_completed_run(
        artifact_root / "20260401T000002Z",
        rows=_run_rows("run3", (-0.004, -0.007, 0.002, 0.003, -0.001)),
    )

    analysis = analyze_policy_candidates_across_runs(
        artifact_root=artifact_root,
        candidate_names=["default_long_only_050", "m7_research_long_only_v1"],
    )
    best_candidate = analysis["best_candidate"]

    assert best_candidate["evidence_too_sparse"] is True
    assert "Total trade count remains below 50 across runs." in best_candidate["warnings"]
    assert "Positive run rate remains below 0.60 across runs." in best_candidate["warnings"]
    assert "Fewer than 3 analyzable runs were available." in best_candidate["warnings"]


def test_positive_run_rate_is_calculated_from_complete_runs(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "m7"
    _write_completed_run(
        artifact_root / "20260401T000001Z",
        rows=_run_rows("run1", (0.010, -0.008, 0.003, 0.004, -0.003)),
    )
    _write_completed_run(
        artifact_root / "20260401T000002Z",
        rows=_run_rows("run2", (0.006, -0.006, 0.001, 0.005, -0.002)),
    )
    _write_completed_run(
        artifact_root / "20260401T000003Z",
        rows=_run_rows("run3", (-0.004, -0.007, 0.002, 0.003, -0.001)),
    )

    analysis = analyze_policy_candidates_across_runs(
        artifact_root=artifact_root,
        candidate_names=["default_long_only_050", "m7_research_long_only_v1"],
    )
    candidate_rows = {
        row["policy_name"]: row for row in analysis["candidate_summaries"]
    }

    assert candidate_rows["m7_research_long_only_v1"]["positive_run_rate"] == 2 / 3


def test_legacy_complete_runs_are_skipped_when_oof_schema_is_incompatible(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "m7"
    _write_completed_run(
        artifact_root / "20260401T000001Z",
        rows=_run_rows("run1", (0.010, -0.008, 0.003, 0.004, -0.003)),
    )
    _write_legacy_completed_run(artifact_root / "20260320T134537Z")

    analysis = analyze_policy_candidates_across_runs(
        artifact_root=artifact_root,
        candidate_names=["default_long_only_050", "m7_research_long_only_v1"],
    )
    candidate_rows = {
        row["policy_name"]: row for row in analysis["candidate_summaries"]
    }

    assert analysis["scanned_run_count"] == 2
    assert analysis["complete_run_count"] == 2
    assert analysis["analyzable_run_count"] == 1
    assert analysis["skipped_runs"] == [
        {
            "run_id": "20260320T134537Z",
            "reason": (
                "incompatible for policy-candidate analysis: Out-of-fold predictions are "
                "missing required columns for threshold analysis: ['regime_label', "
                "'long_only_gross_value_proxy', 'long_only_net_value_proxy']"
            ),
        }
    ]
    assert candidate_rows["default_long_only_050"]["run_count"] == 1
    assert candidate_rows["default_long_only_050"]["complete_run_count"] == 2


def test_expanded_candidate_family_can_rank_regime_aware_candidate_across_runs(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "m7"
    run_rows = [
        {
            "model_name": "autogluon_tabular",
            "fold_index": 0,
            "row_id": "range-positive",
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-01T00:00:00Z",
            "as_of_time": "2026-04-01T00:05:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.83,
            "confidence": 0.83,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": 0.010,
            "long_only_gross_value_proxy": 0.010,
            "long_only_net_value_proxy": 0.008,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 0,
            "row_id": "trend-up-mid-1",
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-01T00:05:00Z",
            "as_of_time": "2026-04-01T00:10:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.72,
            "confidence": 0.72,
            "regime_label": "TREND_UP",
            "long_trade_taken": 1,
            "future_return_3": 0.009,
            "long_only_gross_value_proxy": 0.009,
            "long_only_net_value_proxy": 0.007,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": "trend-up-mid-2",
            "symbol": "SOL/USD",
            "interval_begin": "2026-04-01T00:10:00Z",
            "as_of_time": "2026-04-01T00:15:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.76,
            "confidence": 0.76,
            "regime_label": "TREND_UP",
            "long_trade_taken": 1,
            "future_return_3": 0.008,
            "long_only_gross_value_proxy": 0.008,
            "long_only_net_value_proxy": 0.006,
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
        {
            "model_name": "autogluon_tabular",
            "fold_index": 2,
            "row_id": "range-low-negative",
            "symbol": "SOL/USD",
            "interval_begin": "2026-04-01T00:25:00Z",
            "as_of_time": "2026-04-01T00:30:00Z",
            "y_true": 0,
            "y_pred": 1,
            "prob_up": 0.60,
            "confidence": 0.60,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": -0.003,
            "long_only_gross_value_proxy": -0.003,
            "long_only_net_value_proxy": -0.005,
        },
    ]
    for run_name in ("20260401T000001Z", "20260401T000002Z", "20260401T000003Z"):
        _write_completed_run(artifact_root / run_name, rows=run_rows)

    analysis = analyze_policy_candidates_across_runs(artifact_root=artifact_root)
    candidate_rows = {
        row["policy_name"]: row for row in analysis["candidate_summaries"]
    }

    assert analysis["best_candidate"]["policy_name"] == "per_regime_thresholds_v1"
    assert candidate_rows["range_only_080"]["never_trades_trend_up_across_runs"] is True
    assert (
        "Candidate never trades TREND_UP across analyzable runs."
        in candidate_rows["range_only_080"]["warnings"]
    )
