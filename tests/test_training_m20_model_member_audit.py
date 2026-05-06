"""Focused tests for M20 model/member audit ledger."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_model_member_audit import audit_m20_model_members

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


def _write_current_run(run_dir: Path) -> None:
    baseline_dir = run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_baselines"
    conditional_dir = (
        run_dir / "research_labels" / "vol_scaled" / "conditional_usefulness_full_test"
    )
    baseline_dir.mkdir(parents=True, exist_ok=True)
    conditional_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "fee_baseline_metrics.json").write_text(
        json.dumps(
            {
                "baselines": [
                    {"baseline_name": "always_negative", "average_precision": 0.2},
                    {"baseline_name": "logistic_regression_tiny", "average_precision": 0.3},
                ]
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "predictions_logistic_regression_tiny.csv").write_text(
        "row_id,label,probability\n1,1,0.8\n",
        encoding="utf-8",
    )
    (conditional_dir / "conditional_usefulness_report.json").write_text(
        json.dumps(
            {
                "prediction_rows_analyzed": 1000,
                "search_breadth": {"enable_candidate_count": 1},
                "honesty_flags": ["SINGLE_RECENT_FOLD_ONLY"],
                "recommendation": "A. confirm promising slices on another run/fold/window",
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        conditional_dir / "conditional_usefulness_by_slice.csv",
        [
            {
                "slice_family": "symbol",
                "slice_value": "BTC/USD",
                "classification": "ENABLE_CANDIDATE",
            },
            {
                "slice_family": "month",
                "slice_value": "2026-04",
                "classification": "DISABLE_CANDIDATE",
            },
        ],
    )


def test_audit_discovers_and_classifies_candidates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    previous = tmp_path / "previous"
    fitted = tmp_path / "fitted_models"
    _write_current_run(run_dir)
    _write_csv(
        previous / "fold_metrics.csv",
        [
            {"model_name": "neuralforecast_nhits", "mean_long_only_net_value_proxy": -0.1},
            {"model_name": "neuralforecast_patchtst", "mean_long_only_net_value_proxy": -0.2},
        ],
    )
    (previous / "oof_predictions.csv").write_text(
        "model_name,probability\nx,0.1\n",
        encoding="utf-8",
    )
    (fitted / "fold0").mkdir(parents=True)
    (fitted / "fold0" / "neuralforecast_nhits.joblib").write_text("x", encoding="utf-8")

    report = audit_m20_model_members(
        run_dir=run_dir,
        previous_run_dir=previous,
        fitted_models_dir=fitted,
    )

    ledger = report["ledger"]
    assert any(row["candidate_taxonomy"] == "NEURALFORECAST_SPECIALIST" for row in ledger)
    assert any(row["candidate_taxonomy"] == "RESEARCH_LOGISTIC_BASELINE" for row in ledger)
    assert any(row["evidence_state"] == "HAS_FULL_TEST_CONDITIONAL_EVIDENCE" for row in ledger)
    assert "LONG_RUNS_MANUAL_ONLY" in report["honesty_flags"]


def test_autogluon_metadata_is_handled_gracefully(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    fitted = tmp_path / "fitted_models"
    _write_current_run(run_dir)
    (fitted / "autogluon" / "leaderboard.csv").parent.mkdir(parents=True)
    (fitted / "autogluon" / "leaderboard.csv").write_text(
        "model,score\nWeightedEnsemble,1\n",
        encoding="utf-8",
    )

    report = audit_m20_model_members(run_dir=run_dir, fitted_models_dir=fitted)

    inventory_path = Path(report["output_files"]["autogluon_member_inventory_csv"])
    assert inventory_path.exists()
    assert any(row["candidate_taxonomy"].startswith("AUTOGLUON") for row in report["ledger"])


def test_outputs_are_deterministic_and_preserve_weak_candidates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    previous = tmp_path / "previous"
    _write_current_run(run_dir)
    _write_csv(previous / "fold_metrics.csv", [{"model_name": "dummy_most_frequent"}])

    first = audit_m20_model_members(run_dir=run_dir, previous_run_dir=previous)
    second = audit_m20_model_members(run_dir=run_dir, previous_run_dir=previous)

    first_csv = Path(first["output_files"]["strategy_ensemble_candidate_ledger_csv"]).read_text(
        encoding="utf-8"
    )
    second_csv = Path(second["output_files"]["strategy_ensemble_candidate_ledger_csv"]).read_text(
        encoding="utf-8"
    )
    assert first_csv == second_csv
    assert "dummy_most_frequent" in first_csv
    assert Path(first["output_files"]["manual_confirmation_commands_md"]).exists()
