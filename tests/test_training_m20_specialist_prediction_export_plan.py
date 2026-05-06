"""Focused tests for M20 specialist prediction export planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_specialist_prediction_export_plan import (
    write_m20_specialist_prediction_export_plan,
)

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_sources(base: Path, fitted: Path, previous: Path) -> None:
    vol = base / "research_labels" / "vol_scaled"
    _write_json(
        vol
        / "m20_research_path_adjudication"
        / "research_path_adjudication.json",
        {"recommended_next_action": "PLAN_ROW_LEVEL_SPECIALIST_PREDICTION_EXPORT"},
    )
    ledger_rows = [
        {
            "candidate_id": "fit:fold0:neuralforecast_nhits",
            "model_name": "neuralforecast_nhits",
            "source_run": "fit",
            "candidate_taxonomy": "NEURALFORECAST_SPECIALIST",
            "evidence_state": "HAS_MODEL_ARTIFACT_ONLY",
            "target_type": "direction",
            "prediction_availability": "False",
            "probability_availability": "False",
            "next_required_action": "export row-level probabilities before conditional claims",
        },
        {
            "candidate_id": "fit:fold0:neuralforecast_patchtst",
            "model_name": "neuralforecast_patchtst",
            "source_run": "fit",
            "candidate_taxonomy": "NEURALFORECAST_SPECIALIST",
            "evidence_state": "HAS_MODEL_ARTIFACT_ONLY",
            "target_type": "direction",
            "prediction_availability": "False",
            "probability_availability": "False",
            "next_required_action": "export row-level probabilities before conditional claims",
        },
        {
            "candidate_id": "autogluon:member",
            "model_name": "autogluon_member",
            "source_run": "fit",
            "candidate_taxonomy": "AUTOGLUON_MEMBER_MODEL",
            "evidence_state": "NOT_INSPECTABLE_CURRENTLY",
            "target_type": "unknown",
            "prediction_availability": "False",
            "probability_availability": "False",
            "next_required_action": "member prediction export required",
        },
    ]
    _write_csv(
        vol / "model_member_audit" / "strategy_ensemble_candidate_ledger.csv",
        ledger_rows,
    )
    _write_csv(
        vol / "model_member_audit" / "candidate_next_actions.csv",
        [
            {
                "candidate_id": "20260427T112021Z:neuralforecast_nhits",
                "model_name": "neuralforecast_nhits",
                "next_required_action": (
                    "run conditional analysis on existing row-level predictions"
                ),
            },
            {
                "candidate_id": "20260427T112021Z:neuralforecast_patchtst",
                "model_name": "neuralforecast_patchtst",
                "next_required_action": (
                    "run conditional analysis on existing row-level predictions"
                ),
            },
        ],
    )
    fitted.mkdir(parents=True)
    _write_csv(
        previous / "oof_predictions.csv",
        [
            {
                "model_name": "neuralforecast_nhits",
                "fold_index": 4,
                "symbol": "BTC/USD",
                "interval_begin": "2025-01-01T00:00:00Z",
                "y_true": 1,
                "prob_up": 0.8,
            }
        ],
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_plan_selects_nhits_and_patchtst_targets(tmp_path: Path) -> None:
    base = tmp_path / "base"
    fitted = tmp_path / "fitted"
    previous = tmp_path / "previous"
    _write_sources(base, fitted, previous)

    result = write_m20_specialist_prediction_export_plan(
        base_run_dir=base,
        fitted_models_dir=fitted,
        previous_run_dir=previous,
    )

    targets = _read_csv(Path(result["output_files"]["candidate_export_targets_csv"]))
    model_names = {row["model_name"] for row in targets}
    assert {"neuralforecast_nhits", "neuralforecast_patchtst"} <= model_names
    assert "autogluon_member" not in model_names


def test_plan_preserves_manual_only_blockers(tmp_path: Path) -> None:
    base = tmp_path / "base"
    fitted = tmp_path / "fitted"
    previous = tmp_path / "previous"
    _write_sources(base, fitted, previous)

    result = write_m20_specialist_prediction_export_plan(
        base_run_dir=base,
        fitted_models_dir=fitted,
        previous_run_dir=previous,
    )

    blockers = _read_csv(Path(result["output_files"]["blockers_csv"]))
    blocker_names = {row["blocker"] for row in blockers}
    assert "LONG_RUNS_MANUAL_ONLY" in blocker_names
    assert "AUTOGLUON_MEMBER_PREDICTIONS_MISSING" in blocker_names
    assert result["manual_only"] is True
    assert result["promotion_status"] == "NOT_PROMOTABLE"


def test_plan_writes_schema_and_commands_without_execution(tmp_path: Path) -> None:
    base = tmp_path / "base"
    fitted = tmp_path / "fitted"
    previous = tmp_path / "previous"
    _write_sources(base, fitted, previous)

    result = write_m20_specialist_prediction_export_plan(
        base_run_dir=base,
        fitted_models_dir=fitted,
        previous_run_dir=previous,
    )

    schema_path = Path(result["output_files"]["required_prediction_schema_json"])
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    columns = {row["column"] for row in schema["required_columns"]}
    commands = Path(result["output_files"]["manual_export_commands_md"]).read_text(
        encoding="utf-8"
    )

    assert {"symbol", "interval_begin", "model_name", "candidate_id"} <= columns
    assert "Codex must stop here" in commands
    assert "python -m app.training" in commands
    assert "NO_EXPORT_EXECUTED" in result["honesty_flags"]
