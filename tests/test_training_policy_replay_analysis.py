"""Focused tests for research-only M7 policy replay analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.policy_replay_analysis import (
    analyze_policy_replay,
    analyze_policy_replay_across_runs,
)


def _write_completed_run(
    run_dir: Path,
    *,
    rows: list[dict[str, object]],
    winner_model_name: str = "autogluon_tabular",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"winner": {"model_name": winner_model_name}}),
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


def _replay_rows() -> list[dict[str, object]]:
    return [
        {
            "model_name": "autogluon_tabular",
            "fold_index": 0,
            "row_id": "row-late",
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-01T00:10:00Z",
            "as_of_time": "2026-04-01T00:15:00Z",
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
            "row_id": "row-early-loss",
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-01T00:00:00Z",
            "as_of_time": "2026-04-01T00:05:00Z",
            "y_true": 0,
            "y_pred": 1,
            "prob_up": 0.86,
            "confidence": 0.86,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": -0.006,
            "long_only_gross_value_proxy": -0.006,
            "long_only_net_value_proxy": -0.008,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": "row-mid-win",
            "symbol": "SOL/USD",
            "interval_begin": "2026-04-01T00:05:00Z",
            "as_of_time": "2026-04-01T00:10:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.81,
            "confidence": 0.81,
            "regime_label": "RANGE",
            "long_trade_taken": 1,
            "future_return_3": 0.009,
            "long_only_gross_value_proxy": 0.009,
            "long_only_net_value_proxy": 0.007,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 1,
            "row_id": "row-trend-up",
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-01T00:15:00Z",
            "as_of_time": "2026-04-01T00:20:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.72,
            "confidence": 0.72,
            "regime_label": "TREND_UP",
            "long_trade_taken": 1,
            "future_return_3": 0.006,
            "long_only_gross_value_proxy": 0.006,
            "long_only_net_value_proxy": 0.004,
        },
        {
            "model_name": "autogluon_tabular",
            "fold_index": 2,
            "row_id": "row-trend-down",
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-01T00:20:00Z",
            "as_of_time": "2026-04-01T00:25:00Z",
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
    ]


def test_replay_ledger_is_ordered_chronologically(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, rows=_replay_rows())

    analysis = analyze_policy_replay(
        run_dir=run_dir,
        candidate_names=["default_long_only_050"],
    )
    ledger_path = Path(analysis["output_files"]["replay_trade_ledger_csv"])
    with ledger_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        ledger_rows = list(reader)

    assert [row["row_id"] for row in ledger_rows] == [
        "row-early-loss",
        "row-mid-win",
        "row-late",
        "row-trend-up",
        "row-trend-down",
    ]
    assert [row["trade_index"] for row in ledger_rows] == ["1", "2", "3", "4", "5"]


def test_replay_summary_computes_cumulative_net_and_drawdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, rows=_replay_rows())

    analysis = analyze_policy_replay(
        run_dir=run_dir,
        candidate_names=["default_long_only_050"],
    )
    result = analysis["candidate_results"][0]

    assert result["trade_count"] == 5
    assert result["cumulative_net_proxy"] == pytest.approx(-0.001)
    assert result["mean_net_proxy"] == pytest.approx(-0.0002)
    assert result["max_drawdown_proxy"] == pytest.approx(-0.012)


def test_replay_summary_tracks_longest_loss_streak(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            {
                "model_name": "autogluon_tabular",
                "fold_index": 0,
                "row_id": "loss-1",
                "symbol": "BTC/USD",
                "interval_begin": "2026-04-01T00:00:00Z",
                "as_of_time": "2026-04-01T00:05:00Z",
                "y_true": 0,
                "y_pred": 1,
                "prob_up": 0.81,
                "confidence": 0.81,
                "regime_label": "RANGE",
                "long_trade_taken": 1,
                "future_return_3": -0.004,
                "long_only_gross_value_proxy": -0.004,
                "long_only_net_value_proxy": -0.006,
            },
            {
                "model_name": "autogluon_tabular",
                "fold_index": 0,
                "row_id": "loss-2",
                "symbol": "ETH/USD",
                "interval_begin": "2026-04-01T00:05:00Z",
                "as_of_time": "2026-04-01T00:10:00Z",
                "y_true": 0,
                "y_pred": 1,
                "prob_up": 0.82,
                "confidence": 0.82,
                "regime_label": "RANGE",
                "long_trade_taken": 1,
                "future_return_3": -0.003,
                "long_only_gross_value_proxy": -0.003,
                "long_only_net_value_proxy": -0.005,
            },
            {
                "model_name": "autogluon_tabular",
                "fold_index": 1,
                "row_id": "win",
                "symbol": "SOL/USD",
                "interval_begin": "2026-04-01T00:10:00Z",
                "as_of_time": "2026-04-01T00:15:00Z",
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
        ],
    )

    analysis = analyze_policy_replay(
        run_dir=run_dir,
        candidate_names=["default_long_only_050"],
    )

    assert analysis["candidate_results"][0]["longest_loss_streak"] == 2


def test_multi_run_replay_aggregation_prefers_range_candidate_by_cumulative_net(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "m7"
    run1_rows = _replay_rows()
    run2_rows = _replay_rows()
    run3_rows = _replay_rows()
    for rows in (run1_rows, run2_rows, run3_rows):
        for row in rows:
            if row["row_id"] == "row-trend-up":
                row["future_return_3"] = -0.002
                row["long_only_gross_value_proxy"] = -0.002
                row["long_only_net_value_proxy"] = -0.004
    _write_completed_run(artifact_root / "20260401T000001Z", rows=run1_rows)
    _write_completed_run(artifact_root / "20260401T000002Z", rows=run2_rows)
    _write_completed_run(artifact_root / "20260401T000003Z", rows=run3_rows)

    analysis = analyze_policy_replay_across_runs(
        artifact_root=artifact_root,
        candidate_names=["default_long_only_050", "range_only_080"],
    )

    assert analysis["best_candidate"]["policy_name"] == "range_only_080"
    assert analysis["best_candidate"]["positive_cumulative_run_rate"] == 1.0
    assert analysis["best_candidate"]["total_trade_count"] == 9


def test_replay_marks_candidates_that_never_trade_trend_up(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, rows=_replay_rows())

    analysis = analyze_policy_replay(
        run_dir=run_dir,
        candidate_names=["range_only_080"],
    )
    result = analysis["candidate_results"][0]

    assert result["never_trades_trend_up"] is True
    assert "Candidate never trades TREND_UP in this run." in result["warnings"]
