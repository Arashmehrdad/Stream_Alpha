"""Focused tests for M20 specialist confirmation planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_specialist_confirmation_plan import (
    write_m20_specialist_confirmation_plan,
)

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_sources(base: Path) -> None:
    output_dir = base / "research_labels" / "vol_scaled" / "specialist_conditional_usefulness"
    _write_csv(
        output_dir / "comparison.csv",
        [
            {
                "model_name": "neuralforecast_nhits",
                "top5_lift": "1.35",
                "pr_auc": "0.12",
                "enable_slice_count": "2",
            },
            {
                "model_name": "neuralforecast_patchtst",
                "top5_lift": "1.74",
                "pr_auc": "0.14",
                "enable_slice_count": "15",
            },
        ],
    )
    _write_csv(
        output_dir / "by_slice.csv",
        [
            {
                "model_name": "neuralforecast_patchtst",
                "slice_family": "symbol",
                "slice_value": "BTC/USD",
                "row_count": "1000",
                "positive_count": "100",
                "positive_rate": "0.1",
                "top5_lift": "1.8",
                "classification": "KEEP_CONDITIONAL_RESEARCH_CANDIDATE",
            },
            {
                "model_name": "neuralforecast_nhits",
                "slice_family": "month",
                "slice_value": "2025-05",
                "row_count": "1000",
                "positive_count": "100",
                "positive_rate": "0.1",
                "top5_lift": "1.4",
                "classification": "KEEP_CONDITIONAL_RESEARCH_CANDIDATE",
            },
            {
                "model_name": "neuralforecast_patchtst",
                "slice_family": "month",
                "slice_value": "2026-04",
                "row_count": "1000",
                "positive_count": "100",
                "positive_rate": "0.1",
                "top5_lift": "0.7",
                "classification": "WEAK_OR_UNSTABLE",
            },
        ],
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_confirmation_plan_writes_target_slices(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    fitted = tmp_path / "fitted"
    fitted.mkdir(parents=True)
    _write_sources(base)

    result = write_m20_specialist_confirmation_plan(
        base_run_dir=base,
        previous_run_dir=previous,
        fitted_models_dir=fitted,
    )

    targets = _read_csv(Path(result["output_files"]["target_slices_csv"]))
    assert result["primary_candidate"] == "neuralforecast_patchtst"
    assert result["recommendation"] == "ADD_SPECIALIST_CONFIRMATION_EXPORT_HOOK_FIRST"
    assert targets[0]["model_name"] == "neuralforecast_patchtst"
    assert {row["classification"] for row in targets} == {
        "KEEP_CONDITIONAL_RESEARCH_CANDIDATE"
    }


def test_confirmation_plan_preserves_manual_only_blockers(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    fitted = tmp_path / "fitted"
    fitted.mkdir(parents=True)
    _write_sources(base)

    result = write_m20_specialist_confirmation_plan(
        base_run_dir=base,
        previous_run_dir=previous,
        fitted_models_dir=fitted,
    )

    blockers = _read_csv(Path(result["output_files"]["blockers_csv"]))
    blocker_names = {row["blocker"] for row in blockers}
    assert "LONG_RUNS_MANUAL_ONLY" in blocker_names
    assert "PATCHTST_CONFIRMATION_RUN_NOT_AVAILABLE" in blocker_names
    assert result["manual_only"] is True
    assert result["promotion_status"] == "NOT_PROMOTABLE"


def test_confirmation_plan_writes_schema_and_commands(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    fitted = tmp_path / "fitted"
    fitted.mkdir(parents=True)
    _write_sources(base)

    result = write_m20_specialist_confirmation_plan(
        base_run_dir=base,
        previous_run_dir=previous,
        fitted_models_dir=fitted,
    )
    schema = json.loads(
        Path(result["output_files"]["required_export_schema_json"]).read_text(
            encoding="utf-8"
        )
    )
    commands = Path(result["output_files"]["manual_commands_md"]).read_text(
        encoding="utf-8"
    )

    assert "prob_up" in schema["required_columns"]
    assert "Codex must stop here" in commands
    assert "--export-specialist-predictions-only" in commands
    assert "NO_EXPORT_EXECUTED" in result["honesty_flags"]
