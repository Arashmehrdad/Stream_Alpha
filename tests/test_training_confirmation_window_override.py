"""Focused tests for M20 confirmation-window override plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import subprocess
from pathlib import Path

import pytest

from app.training.confirmation_window import (
    build_confirmation_window_override,
    confirmation_window_validation,
    filter_rows_for_confirmation_window,
    parse_utc_timestamp,
)

# pylint: disable=missing-function-docstring


@dataclass(frozen=True)
class _Row:
    as_of_time: datetime
    symbol: str = "BTC/USD"


def test_parse_confirmation_window_normalizes_to_utc() -> None:
    parsed = parse_utc_timestamp("2024-04-02T12:30:00+01:00")

    assert parsed == datetime(2024, 4, 2, 11, 30, tzinfo=timezone.utc)


def test_confirmation_window_rejects_start_after_end() -> None:
    with pytest.raises(ValueError, match="start must be before end"):
        build_confirmation_window_override(
            start="2025-04-02T11:30:00Z",
            end="2024-04-02T11:30:00Z",
            tag="confirm_prev_year",
        )


def test_confirmation_window_absent_preserves_default_behavior() -> None:
    assert build_confirmation_window_override() is None


def test_confirmation_window_filters_synthetic_rows() -> None:
    override = build_confirmation_window_override(
        start="2024-01-01T00:00:00Z",
        end="2024-02-01T00:00:00Z",
        tag="confirm_prev_year",
    )
    rows = [
        _Row(parse_utc_timestamp("2023-12-31T23:00:00Z")),
        _Row(parse_utc_timestamp("2024-01-15T00:00:00Z")),
        _Row(parse_utc_timestamp("2024-02-01T00:00:00Z")),
    ]

    selected = filter_rows_for_confirmation_window(rows, override)

    assert len(selected) == 1
    assert selected[0].as_of_time == parse_utc_timestamp("2024-01-15T00:00:00Z")


def test_confirmation_window_empty_fails_cleanly() -> None:
    override = build_confirmation_window_override(
        start="2024-01-01T00:00:00Z",
        end="2024-02-01T00:00:00Z",
        tag="confirm_prev_year",
    )

    with pytest.raises(ValueError, match="CONFIRMATION_WINDOW_EMPTY"):
        filter_rows_for_confirmation_window(
            [_Row(parse_utc_timestamp("2024-03-01T00:00:00Z"))],
            override,
        )


def test_confirmation_window_detects_overlap_and_not_distinct() -> None:
    override = build_confirmation_window_override(
        start="2024-01-01T00:00:00Z",
        end="2024-02-01T00:00:00Z",
        tag="confirm_prev_year",
    )
    validation = confirmation_window_validation(
        selected_rows=[_Row(parse_utc_timestamp("2024-01-15T00:00:00Z"))],
        all_symbols=["BTC/USD"],
        override=override,
        original_window={
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-02-01T00:00:00Z",
        },
    )

    assert validation["distinct_from_original"] is False
    assert validation["overlap_with_original"] is True
    assert "CONFIRMATION_WINDOW_NOT_DISTINCT" in validation["honesty_flags"]
    assert "CONFIRMATION_WINDOW_OVERLAPS_ORIGINAL" in validation["honesty_flags"]


def test_confirmation_tag_must_be_safe() -> None:
    with pytest.raises(ValueError, match="confirmation tag"):
        build_confirmation_window_override(
            start="2024-01-01T00:00:00Z",
            end="2024-02-01T00:00:00Z",
            tag="../bad",
        )


def test_powershell_dry_run_includes_confirmation_flags() -> None:
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts/start_m20_training.ps1",
        "-DryRun",
        "-ExportTrainingFrameOnly",
        "-ConfirmationWindowStart",
        "2024-04-02T11:30:00Z",
        "-ConfirmationWindowEnd",
        "2025-04-02T11:30:00Z",
        "-ConfirmationTag",
        "confirm_prev_year",
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--export-training-frame-only" in result.stdout
    assert "--confirmation-window-start 2024-04-02T11:30:00Z" in result.stdout
    assert "--confirmation-window-end 2025-04-02T11:30:00Z" in result.stdout
    assert "--confirmation-tag confirm_prev_year" in result.stdout
    assert "model scoring: skipped by export-only mode" in result.stdout
    assert "confirmation run: manual-only" in result.stdout


def test_powershell_dry_run_includes_specialist_prediction_export_flag() -> None:
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts/start_m20_training.ps1",
        "-DryRun",
        "-ScoreOnly",
        "artifacts/training/m20/20260405T023104Z/fitted_models",
        "-ParquetDir",
        "exports/feature_ohlc_for_colab",
        "-ExportSpecialistPredictionsOnly",
        "-ConfirmationWindowStart",
        "2024-04-02T11:30:00Z",
        "-ConfirmationWindowEnd",
        "2025-04-02T11:30:00Z",
        "-ConfirmationTag",
        "confirm_prev_year",
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--export-specialist-predictions-only" in result.stdout
    assert "--score-only artifacts/training/m20/20260405T023104Z/fitted_models" in result.stdout
    assert "--parquet-dir exports/feature_ohlc_for_colab" in result.stdout
    assert "export specialist predictions only: True" in result.stdout
