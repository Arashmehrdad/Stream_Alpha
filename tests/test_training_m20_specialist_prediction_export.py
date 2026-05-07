"""Focused tests for M20 specialist prediction export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_specialist_prediction_export import (
    export_existing_m20_specialist_predictions,
)

# pylint: disable=missing-function-docstring


def _write_oof(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model_name": "neuralforecast_nhits",
            "fold_index": "4",
            "row_id": "BTC/USD|2025-01-01T00:00:00Z",
            "symbol": "BTC/USD",
            "interval_begin": "2025-01-01T00:00:00Z",
            "as_of_time": "2025-01-01T00:05:00Z",
            "y_true": "1",
            "y_pred": "1",
            "prob_up": "0.8",
            "confidence": "0.6",
            "regime_label": "TREND_UP",
            "future_return_3": "0.1",
            "long_only_net_value_proxy": "0.09",
        },
        {
            "model_name": "neuralforecast_patchtst",
            "fold_index": "4",
            "row_id": "ETH/USD|2025-01-01T00:00:00Z",
            "symbol": "ETH/USD",
            "interval_begin": "2025-01-01T00:00:00Z",
            "as_of_time": "2025-01-01T00:05:00Z",
            "y_true": "0",
            "y_pred": "1",
            "prob_up": "0.7",
            "confidence": "0.4",
            "regime_label": "RANGE",
            "future_return_3": "-0.1",
            "long_only_net_value_proxy": "-0.12",
        },
        {
            "model_name": "persistence_3",
            "fold_index": "4",
            "row_id": "SOL/USD|2025-01-01T00:00:00Z",
            "symbol": "SOL/USD",
            "interval_begin": "2025-01-01T00:00:00Z",
            "as_of_time": "2025-01-01T00:05:00Z",
            "y_true": "0",
            "y_pred": "0",
            "prob_up": "0.2",
            "confidence": "0.8",
            "regime_label": "RANGE",
            "future_return_3": "0.0",
            "long_only_net_value_proxy": "0.0",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_export_writes_per_specialist_files(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    _write_oof(previous / "oof_predictions.csv")

    result = export_existing_m20_specialist_predictions(
        base_run_dir=base,
        previous_run_dir=previous,
    )

    assert result["exported_row_count"] == 2
    assert Path(
        result["output_files"]["predictions_neuralforecast_nhits_oof_csv"]
    ).exists()
    assert Path(
        result["output_files"]["predictions_neuralforecast_patchtst_oof_csv"]
    ).exists()


def test_export_quarantines_future_and_net_columns(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    _write_oof(previous / "oof_predictions.csv")

    result = export_existing_m20_specialist_predictions(
        base_run_dir=base,
        previous_run_dir=previous,
    )
    nhits_rows = _read_csv(
        Path(result["output_files"]["predictions_neuralforecast_nhits_oof_csv"])
    )
    audit_rows = _read_csv(Path(result["output_files"]["schema_audit_csv"]))
    audit_by_column = {row["column"]: row for row in audit_rows}

    assert "future_return_3" not in nhits_rows[0]
    assert "long_only_net_value_proxy" not in nhits_rows[0]
    assert audit_by_column["future_return_3"]["exported"] == "False"
    assert "quarantined" in audit_by_column["long_only_net_value_proxy"]["reason"]


def test_export_is_research_only_and_deterministic(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    _write_oof(previous / "oof_predictions.csv")

    first = export_existing_m20_specialist_predictions(
        base_run_dir=base,
        previous_run_dir=previous,
    )
    second = export_existing_m20_specialist_predictions(
        base_run_dir=base,
        previous_run_dir=previous,
    )
    manifest = json.loads(
        Path(second["output_files"]["manifest_json"]).read_text(encoding="utf-8")
    )

    assert first["output_files"] == second["output_files"]
    assert "NO_RUNTIME_EFFECT" in manifest["honesty_flags"]
    assert manifest["promotion_status"] == "NOT_PROMOTABLE"
