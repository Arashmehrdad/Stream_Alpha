"""Focused tests for M20 confirmation-window planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_confirmation_compare import (
    classify_confirmation,
    compare_m20_confirmation_slices,
)
from app.training.m20_confirmation_planner import plan_m20_confirmation_window

# pylint: disable=missing-function-docstring,too-many-arguments,too-many-positional-arguments


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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_current_run(run_dir: Path) -> None:
    conditional = run_dir / "research_labels" / "vol_scaled" / "conditional_usefulness_full_test"
    baseline = run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_baselines"
    audit = run_dir / "research_labels" / "vol_scaled" / "model_member_audit"
    _write_csv(
        conditional / "conditional_usefulness_by_slice.csv",
        [
            _slice("momentum", "flat", "ENABLE_CANDIDATE", 1500, 120, 2.0, 0.25, 0.10),
            _slice("range", "low", "ENABLE_CANDIDATE", 1800, 140, 1.8, 0.22, 0.09),
            _slice("symbol", "BTC/USD", "ENABLE_CANDIDATE", 2000, 200, 1.7, 0.30, 0.15),
            _slice("macd", "positive", "ENABLE_CANDIDATE", 2100, 180, 1.6, 0.26, 0.12),
            _slice("volume", "low", "WATCHLIST_CANDIDATE", 1700, 100, 1.3, 0.18, 0.08),
            _slice("month", "2026-04", "DISABLE_CANDIDATE", 1200, 90, 0.8, 0.07, 0.09),
            _slice("quarter", "2026Q2", "DISABLE_CANDIDATE", 1200, 90, 0.8, 0.07, 0.09),
        ],
    )
    _write_csv(
        conditional / "conditional_enable_disable_summary.csv",
        [{"classification": "ENABLE_CANDIDATE", "slice_count": 4}],
    )
    _write_json(
        conditional / "conditional_usefulness_report.json",
        {
            "prediction_rows_analyzed": 47229,
            "search_breadth": {
                "enable_candidate_count": 4,
                "watchlist_candidate_count": 1,
                "disable_candidate_count": 2,
            },
        },
    )
    _write_json(
        baseline / "fee_baseline_manifest.json",
        {"best_baseline_name": "logistic_regression_tiny"},
    )
    _write_json(
        baseline / "fee_baseline_metrics.json",
        {"baselines": [{"baseline_name": "logistic_regression_tiny", "average_precision": 0.3}]},
    )
    _write_json(run_dir / "training_frame" / "m20_training_frame_export_manifest.json", {})
    _write_csv(
        audit / "strategy_ensemble_candidate_ledger.csv",
        [
            {
                "candidate_id": "run:logistic_regression_tiny",
                "model_name": "logistic_regression_tiny",
            }
        ],
    )


def _slice(
    family: str,
    value: str,
    classification: str,
    rows: int,
    positives: int,
    top_5_lift: float,
    average_precision: float,
    positive_rate: float,
) -> dict[str, object]:
    return {
        "slice_family": family,
        "slice_value": value,
        "classification": classification,
        "row_count": rows,
        "positive_count": positives,
        "positive_rate": positive_rate,
        "average_precision": average_precision,
        "top_5_precision": positive_rate * top_5_lift,
        "top_5_lift": top_5_lift,
    }


def test_planner_writes_deterministic_plan_and_manual_commands(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = tmp_path / "training.m20.json"
    _write_current_run(run_dir)
    _write_json(config, {"recent_scoring_window_days": 365, "test_folds": 5})

    first = plan_m20_confirmation_window(run_dir=run_dir, config_path=config)
    second = plan_m20_confirmation_window(run_dir=run_dir, config_path=config)

    assert first["target_slice_count"] == second["target_slice_count"]
    assert first["window_override_support"]["supported"] is True
    assert "CONFIRMATION_WINDOW_OVERRIDE_SUPPORTED" in first["honesty_flags"]
    targets_path = Path(first["output_files"]["confirmation_slice_targets_csv"])
    commands_path = Path(first["output_files"]["confirmation_manual_commands_md"])
    rules_path = Path(first["output_files"]["confirmation_success_rules_json"])
    assert targets_path.read_text(encoding="utf-8") == Path(
        second["output_files"]["confirmation_slice_targets_csv"]
    ).read_text(encoding="utf-8")
    commands = commands_path.read_text(encoding="utf-8")
    assert "--export-training-frame-only" in commands
    assert "--confirmation-window-start <START>" in commands
    assert "--confirmation-window-end <END>" in commands
    assert "STRONGLY_CONFIRMED" in rules_path.read_text(encoding="utf-8")
    assert "momentum=flat" in targets_path.read_text(encoding="utf-8")


def test_compare_handles_missing_confirmation_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_current_run(run_dir)
    plan_m20_confirmation_window(run_dir=run_dir, config_path=tmp_path / "missing.json")

    result = compare_m20_confirmation_slices(
        original_run_dir=run_dir,
        confirmation_run_dir=tmp_path / "does-not-exist",
    )

    assert result["status"] == "CONFIRMATION_RUN_NOT_AVAILABLE"
    assert result["recommendation"] == "INCONCLUSIVE_LOW_SAMPLE"


def test_compare_classifies_confirmation_slices(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    _write_current_run(original)
    _write_current_run(confirmation)
    plan_m20_confirmation_window(run_dir=original, config_path=tmp_path / "missing.json")
    _write_csv(
        confirmation
        / "research_labels"
        / "vol_scaled"
        / "conditional_usefulness_full_test"
        / "conditional_usefulness_by_slice.csv",
        [
            _slice("momentum", "flat", "ENABLE_CANDIDATE", 1500, 120, 1.6, 0.25, 0.10),
            _slice("range", "low", "ENABLE_CANDIDATE", 1500, 120, 1.3, 0.13, 0.10),
            _slice("symbol", "BTC/USD", "DISABLE_CANDIDATE", 1500, 120, 0.9, 0.09, 0.10),
            _slice("macd", "positive", "ENABLE_CANDIDATE", 500, 40, 2.0, 0.30, 0.10),
        ],
    )

    report = compare_m20_confirmation_slices(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    statuses = {row["slice_id"]: row["confirmation_status"] for row in report["comparison_rows"]}
    assert statuses["momentum=flat"] == "STRONGLY_CONFIRMED"
    assert statuses["range=low"] == "CONFIRMED"
    assert statuses["symbol=BTC/USD"] == "NOT_CONFIRMED"
    assert statuses["macd=positive"] == "INCONCLUSIVE"


def test_classify_confirmation_rules_are_stable() -> None:
    assert classify_confirmation(_slice("x", "y", "", 1500, 100, 1.6, 0.2, 0.1)) == (
        "STRONGLY_CONFIRMED"
    )
    assert classify_confirmation(_slice("x", "y", "", 1500, 100, 1.25, 0.12, 0.1)) == (
        "CONFIRMED"
    )
    assert classify_confirmation(_slice("x", "y", "", 1500, 100, 0.9, 0.09, 0.1)) == (
        "NOT_CONFIRMED"
    )
    assert classify_confirmation(_slice("x", "y", "", 500, 20, 2.0, 0.3, 0.1)) == (
        "INCONCLUSIVE"
    )
