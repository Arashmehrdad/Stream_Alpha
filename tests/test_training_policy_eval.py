"""Focused tests for research-only offline M20 threshold policy evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.policy_eval import analyze_completed_run
from app.training.oof_signal_diagnostics import diagnose_completed_run
from app.training.research_labels import (
    build_fee_exceedance_labels,
    build_incumbent_meta_labels,
    build_label_diagnostics,
    build_triple_barrier_labels,
    write_label_diagnostics,
)

# pylint: disable=missing-function-docstring,too-many-arguments


def _write_completed_run(
    run_dir: Path,
    *,
    rows: list[dict[str, object]],
    winner_model_name: str = "neuralforecast_patchtst",
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
    field_names = [
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
        "long_only_gross_value_proxy",
        "long_only_net_value_proxy",
    ]
    if any("regime_label" in row for row in rows):
        field_names.insert(10, "regime_label")
    for row in rows:
        for column in row:
            if column not in field_names:
                field_names.append(column)
    with (run_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    row_id: str,
    *,
    fold_index: int,
    prob_up: float,
    net_value: float,
    y_true: int = 1,
    regime_label: str | None = "RANGE",
    interval_begin: str | None = None,
) -> dict[str, object]:
    row = {
        "model_name": "neuralforecast_patchtst",
        "fold_index": fold_index,
        "row_id": row_id,
        "symbol": "BTC/USD",
        "interval_begin": interval_begin or f"2026-04-01T00:0{fold_index}:00Z",
        "as_of_time": interval_begin or f"2026-04-01T00:0{fold_index}:05Z",
        "y_true": y_true,
        "y_pred": int(prob_up >= 0.5),
        "prob_up": prob_up,
        "confidence": prob_up,
        "future_return_3": net_value + 0.002,
        "long_only_gross_value_proxy": net_value + 0.002,
        "long_only_net_value_proxy": net_value,
    }
    if regime_label is not None:
        row["regime_label"] = regime_label
    return row


def test_threshold_sweep_prefers_stricter_threshold_when_baseline_is_dragged_by_bad_trades(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            _row(
                "row-0",
                fold_index=0,
                prob_up=0.92,
                net_value=0.030,
                regime_label="RANGE",
            ),
            _row(
                "row-1",
                fold_index=0,
                prob_up=0.85,
                net_value=-0.005,
                y_true=0,
                regime_label="RANGE",
            ),
            _row(
                "row-2",
                fold_index=1,
                prob_up=0.72,
                net_value=0.008,
                regime_label="TREND_UP",
            ),
            _row(
                "row-3",
                fold_index=1,
                prob_up=0.60,
                net_value=-0.020,
                y_true=0,
                regime_label="RANGE",
            ),
            _row(
                "row-4",
                fold_index=2,
                prob_up=0.52,
                net_value=-0.010,
                y_true=0,
                regime_label="HIGH_VOL",
            ),
        ],
    )

    analysis = analyze_completed_run(
        run_dir=run_dir,
        thresholds=(0.50, 0.70, 0.80),
    )

    baseline = analysis["baseline_result"]
    best_candidate = analysis["best_candidate"]
    assert baseline["trade_count"] == 5
    assert best_candidate["threshold"] == 0.70
    assert best_candidate["trade_count"] == 3
    assert best_candidate["mean_long_only_net_value_proxy"] == pytest.approx(0.0066)
    assert best_candidate["beats_baseline_mean_net"] is True
    assert (
        best_candidate["delta_vs_baseline_mean_long_only_net_value_proxy"]
        == pytest.approx(0.006)
    )


def test_abstention_and_coverage_are_recorded_for_high_thresholds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            _row("row-0", fold_index=0, prob_up=0.81, net_value=0.006),
            _row("row-1", fold_index=0, prob_up=0.79, net_value=0.004),
            _row("row-2", fold_index=1, prob_up=0.45, net_value=0.000, y_true=0),
            _row("row-3", fold_index=1, prob_up=0.20, net_value=0.000, y_true=0),
        ],
    )

    analysis = analyze_completed_run(run_dir=run_dir, thresholds=(0.80,))

    result = analysis["best_candidate"]
    assert result["trade_count"] == 1
    assert result["coverage"] == pytest.approx(0.25)
    assert result["abstention_count"] == 3
    assert result["abstention_rate"] == pytest.approx(0.75)


def test_drawdown_uses_chronological_trade_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            _row(
                "row-0",
                fold_index=0,
                prob_up=0.80,
                net_value=0.030,
                interval_begin="2026-04-01T00:00:00Z",
            ),
            _row(
                "row-1",
                fold_index=0,
                prob_up=0.82,
                net_value=-0.010,
                y_true=0,
                interval_begin="2026-04-01T00:05:00Z",
            ),
            _row(
                "row-2",
                fold_index=1,
                prob_up=0.84,
                net_value=-0.040,
                y_true=0,
                interval_begin="2026-04-01T00:10:00Z",
            ),
            _row(
                "row-3",
                fold_index=1,
                prob_up=0.86,
                net_value=0.020,
                interval_begin="2026-04-01T00:15:00Z",
            ),
        ],
    )

    analysis = analyze_completed_run(run_dir=run_dir, thresholds=(0.80,))

    assert analysis["best_candidate"]["max_drawdown_proxy"] == pytest.approx(0.05)


def test_regime_breakdown_only_exists_when_regime_labels_are_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            _row("row-0", fold_index=0, prob_up=0.80, net_value=0.010, regime_label=None),
            _row("row-1", fold_index=1, prob_up=0.30, net_value=0.000, y_true=0, regime_label=None),
        ],
    )

    analysis = analyze_completed_run(run_dir=run_dir, thresholds=(0.50,))

    assert analysis["regime_summary_available"] is False
    assert analysis["best_candidate"]["per_regime_breakdown"] is None


def test_row_local_threshold_decisions_do_not_use_future_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    base_rows = [
        _row(
            "row-0",
            fold_index=0,
            prob_up=0.81,
            net_value=0.012,
            interval_begin="2026-04-01T00:00:00Z",
        ),
        _row(
            "row-1",
            fold_index=0,
            prob_up=0.79,
            net_value=-0.003,
            y_true=0,
            interval_begin="2026-04-01T00:05:00Z",
        ),
        _row(
            "row-2",
            fold_index=1,
            prob_up=0.20,
            net_value=-0.050,
            y_true=0,
            interval_begin="2026-04-01T00:10:00Z",
        ),
    ]
    _write_completed_run(run_dir, rows=base_rows)
    original = analyze_completed_run(run_dir=run_dir, thresholds=(0.80,))

    mutated_rows = list(base_rows)
    mutated_rows[-1] = _row(
        "row-2",
        fold_index=1,
        prob_up=0.99,
        net_value=0.500,
        interval_begin="2026-04-01T00:10:00Z",
    )
    _write_completed_run(run_dir, rows=mutated_rows)
    mutated = analyze_completed_run(run_dir=run_dir, thresholds=(0.80,))

    assert original["best_candidate"]["trade_row_ids"][0] == "row-0"
    assert mutated["candidate_results"][0]["trade_row_ids"][:1] == ["row-0"]
    assert original["candidate_results"][0]["trade_row_ids"][:1] == ["row-0"]


def test_missing_completed_run_artifacts_fail_clearly(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="missing summary.json"):
        analyze_completed_run(run_dir=run_dir, thresholds=(0.50,))


def test_partial_summary_and_empty_winner_rows_fail_clearly(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({"winner": {}}), encoding="utf-8")
    (run_dir / "oof_predictions.csv").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="winner.model_name"):
        analyze_completed_run(run_dir=run_dir, thresholds=(0.50,))

    _write_completed_run(
        run_dir,
        rows=[_row("row-0", fold_index=0, prob_up=0.70, net_value=0.01)],
        winner_model_name="missing_model",
    )

    with pytest.raises(ValueError, match="No out-of-fold predictions"):
        analyze_completed_run(run_dir=run_dir, thresholds=(0.50,))


def test_fold_stability_and_low_trade_flags_are_reported(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            _row("row-0", fold_index=0, prob_up=0.90, net_value=0.020),
            _row("row-1", fold_index=0, prob_up=0.60, net_value=-0.010, y_true=0),
            _row("row-2", fold_index=1, prob_up=0.91, net_value=-0.005, y_true=0),
            _row("row-3", fold_index=1, prob_up=0.55, net_value=0.020),
        ],
    )

    analysis = analyze_completed_run(
        run_dir=run_dir,
        thresholds=(0.90,),
        min_honest_trades=3,
    )
    candidate = analysis["best_candidate"]

    assert candidate["fold_count"] == 2
    assert candidate["fold_win_count_vs_baseline"] == 1
    assert candidate["weakest_fold"]["fold_index"] == 1
    assert "LOW_TRADE_COUNT" in candidate["honesty_flags"]
    assert "Fold stability" in (run_dir / "policy_eval" / "policy_report.md").read_text(
        encoding="utf-8"
    )


def test_cost_scenarios_are_reported_without_changing_base_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            _row("row-0", fold_index=0, prob_up=0.90, net_value=0.008),
            _row("row-1", fold_index=0, prob_up=0.20, net_value=0.000, y_true=0),
        ],
        fee_rate=0.002,
    )

    analysis = analyze_completed_run(
        run_dir=run_dir,
        thresholds=(0.80,),
        scenario_slippage_rates=(0.0, 0.001),
    )

    candidate = analysis["best_candidate"]
    scenarios = {
        scenario["scenario_name"]: scenario
        for scenario in candidate["cost_scenario_results"]
    }
    assert candidate["mean_long_only_net_value_proxy"] == pytest.approx(0.004)
    assert scenarios["current_fee"]["mean_long_only_net_value_proxy"] == pytest.approx(0.004)
    assert scenarios["double_fee"]["mean_long_only_net_value_proxy"] == pytest.approx(0.003)
    assert scenarios["current_fee_plus_10bps_slippage"]["mean_long_only_net_value_proxy"] == (
        pytest.approx(0.0035)
    )


def test_triple_barrier_labels_use_declared_forward_horizon_only() -> None:
    rows = [
        {
            "row_id": "a",
            "close_price": 100.0,
            "high_price": 100.0,
            "low_price": 100.0,
            "realized_vol_12": 0.01,
        },
        {
            "row_id": "b",
            "close_price": 100.5,
            "high_price": 101.2,
            "low_price": 100.2,
            "realized_vol_12": 0.01,
        },
        {
            "row_id": "c",
            "close_price": 99.0,
            "high_price": 99.5,
            "low_price": 98.5,
            "realized_vol_12": 0.01,
        },
        {
            "row_id": "d",
            "close_price": 105.0,
            "high_price": 106.0,
            "low_price": 104.0,
            "realized_vol_12": 0.01,
        },
    ]

    labels = build_triple_barrier_labels(rows, horizon=2)

    assert labels[0]["label"] == 1
    assert labels[0]["barrier_hit"] == "upper"
    assert labels[0]["event_end_row_id"] == "b"
    assert labels[1]["label"] == -1
    assert labels[1]["event_end_row_id"] == "c"
    assert len(labels) == 2


def test_fee_exceedance_and_meta_labels_are_deterministic() -> None:
    rows = [
        {"row_id": "a", "close_price": 100.0, "high_price": 100.0},
        {"row_id": "b", "close_price": 100.0, "high_price": 100.4},
        {"row_id": "c", "close_price": 100.0, "high_price": 101.0},
        {"row_id": "d", "close_price": 100.0, "high_price": 100.8},
    ]
    fee_labels = build_fee_exceedance_labels(rows, horizon=2, fee_rate=0.005)

    assert fee_labels[0]["label"] == 1
    assert fee_labels[1]["label"] == 1

    meta_labels = build_incumbent_meta_labels(
        [
            {"row_id": "a", "prob_up": 0.60, "long_only_net_value_proxy": 0.010},
            {"row_id": "b", "prob_up": 0.40, "long_only_net_value_proxy": 0.020},
            {"row_id": "c", "prob_up": 0.70, "long_only_net_value_proxy": -0.005},
        ]
    )

    assert [row["meta_label"] for row in meta_labels] == [1, 0, 0]
    with pytest.raises(ValueError, match="missing prob_up"):
        build_incumbent_meta_labels([{"row_id": "bad", "long_only_net_value_proxy": 0.0}])


def test_label_diagnostics_are_persisted(tmp_path: Path) -> None:
    diagnostics = build_label_diagnostics(
        [{"label": 1}, {"label": -1}, {"label": 1}],
        skipped_count=2,
    )
    output_path = tmp_path / "label_diagnostics.json"

    write_label_diagnostics(output_path, diagnostics)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["total_count"] == 3
    assert payload["skipped_count"] == 2
    assert payload["label_distribution"] == {"-1": 1, "1": 2}


def test_oof_signal_diagnostics_write_deterministic_reports(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            _row("row-0", fold_index=0, prob_up=0.10, net_value=-0.002, y_true=0),
            _row("row-1", fold_index=0, prob_up=0.40, net_value=-0.001, y_true=0),
            _row("row-2", fold_index=1, prob_up=0.80, net_value=0.006, y_true=1),
            _row("row-3", fold_index=1, prob_up=0.90, net_value=0.008, y_true=1),
        ],
    )

    diagnostics = diagnose_completed_run(run_dir=run_dir, thresholds=(0.50, 0.85))

    output_files = diagnostics["output_files"]
    assert Path(output_files["oof_signal_diagnostics_json"]).exists()
    assert Path(output_files["oof_score_quantiles_csv"]).exists()
    assert Path(output_files["oof_threshold_crossing_counts_csv"]).exists()
    assert Path(output_files["oof_filter_funnel_csv"]).exists()
    assert Path(output_files["oof_signal_diagnostics_md"]).exists()
    assert diagnostics["score_quantiles"][0]["p50"] == pytest.approx(0.60)
    assert diagnostics["honesty_flags"] == ["LOW_TRADE_COUNT"]


def test_oof_threshold_crossing_counts_and_filter_funnel(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            _row("row-0", fold_index=0, prob_up=0.20, net_value=0.001, regime_label="RANGE"),
            _row("row-1", fold_index=0, prob_up=0.60, net_value=0.002, regime_label="RANGE"),
            _row("row-2", fold_index=1, prob_up=0.90, net_value=0.003, regime_label="TREND_UP"),
        ],
    )

    diagnostics = diagnose_completed_run(
        run_dir=run_dir,
        thresholds=(0.50,),
        regime_label="RANGE",
    )

    funnel = {row["stage"]: row["row_count"] for row in diagnostics["filter_funnel"]}
    assert funnel["raw_oof_rows_loaded"] == 3
    assert funnel["after_candidate_model_filter"] == 3
    assert funnel["after_fold_symbol_regime_filters"] == 2
    overall_crossing = [
        row for row in diagnostics["threshold_crossing_rows"]
        if row["group_type"] == "overall"
    ] if "threshold_crossing_rows" in diagnostics else []
    assert overall_crossing == []
    crossing_csv = Path(
        diagnostics["output_files"]["oof_threshold_crossing_counts_csv"]
    ).read_text(encoding="utf-8")
    assert "RANGE" in crossing_csv


def test_oof_zero_trade_before_cost_is_diagnosed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            _row("row-0", fold_index=0, prob_up=0.10, net_value=0.000, y_true=0),
            _row("row-1", fold_index=0, prob_up=0.20, net_value=0.000, y_true=0),
        ],
    )

    diagnostics = diagnose_completed_run(run_dir=run_dir, thresholds=(0.50,))

    assert "NO_THRESHOLD_CROSSINGS" in diagnostics["honesty_flags"]
    assert diagnostics["cost_gating_diagnostics"]["thresholds"][0]["zero_before_cost"] is True


def test_oof_zero_trade_after_cost_is_diagnosed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            _row("row-0", fold_index=0, prob_up=0.80, net_value=-0.001, y_true=1),
            _row("row-1", fold_index=0, prob_up=0.90, net_value=-0.001, y_true=1),
        ],
        fee_rate=0.002,
    )

    diagnostics = diagnose_completed_run(run_dir=run_dir, thresholds=(0.50,))
    threshold_row = diagnostics["cost_gating_diagnostics"]["thresholds"][0]

    assert threshold_row["count_before_cost"] == 2
    assert threshold_row["zero_before_cost"] is False
    assert threshold_row["zero_after_all_cost_scenarios"] is True
    assert "LOW_TRADE_COUNT" in diagnostics["honesty_flags"]


def test_oof_out_of_range_and_missing_label_warnings(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"winner": {"model_name": "m"}}),
        encoding="utf-8",
    )
    with (run_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["model_name", "fold_index", "row_id", "prob_up"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"model_name": "m", "fold_index": 0, "row_id": "a", "prob_up": 1.2},
                {"model_name": "m", "fold_index": 0, "row_id": "b", "prob_up": -0.1},
            ]
        )

    diagnostics = diagnose_completed_run(run_dir=run_dir, thresholds=(0.50,))

    assert "SCORE_OUT_OF_RANGE" in diagnostics["honesty_flags"]
    assert "UNCALIBRATED_OR_REGRESSION_SCORE" in diagnostics["honesty_flags"]
    assert "MISSING_LABELS" in diagnostics["honesty_flags"]


def test_oof_possible_inverted_score_warning(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        rows=[
            _row("row-0", fold_index=0, prob_up=0.90, net_value=-0.010, y_true=0),
            _row("row-1", fold_index=0, prob_up=0.80, net_value=-0.005, y_true=0),
            _row("row-2", fold_index=1, prob_up=0.20, net_value=0.005, y_true=1),
            _row("row-3", fold_index=1, prob_up=0.10, net_value=0.010, y_true=1),
        ],
    )

    diagnostics = diagnose_completed_run(run_dir=run_dir, thresholds=(0.50,))

    assert diagnostics["label_diagnostics"]["possible_inverted_score"] is True
    assert "POSSIBLE_INVERTED_SCORE" in diagnostics["honesty_flags"]
