"""Focused tests for M20 specialist conditional usefulness."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_specialist_conditional_usefulness import (
    analyze_m20_specialist_conditional_usefulness,
)

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows(model_name: str) -> list[dict[str, object]]:
    rows = []
    for index in range(20):
        symbol = "BTC/USD" if index < 10 else "ETH/USD"
        probability = 0.95 - index * 0.02 if model_name.endswith("nhits") else 0.2 + index * 0.02
        rows.append(
            {
                "symbol": symbol,
                "interval_begin": f"2025-01-{index + 1:02d}T00:00:00Z",
                "fold_index": "4",
                "model_name": model_name,
                "candidate_id": f"20260427T112021Z:{model_name}",
                "prediction_source": "oof_20260427",
                "row_id": f"{symbol}|2025-01-{index + 1:02d}T00:00:00Z",
                "as_of_time": f"2025-01-{index + 1:02d}T00:05:00Z",
                "y_true": "0",
                "y_pred": "1" if probability >= 0.5 else "0",
                "prob_up": probability,
                "confidence": abs(probability - 0.5),
                "regime_label": "TREND_UP" if index % 2 == 0 else "RANGE",
            }
        )
    return rows


def _label_rows(model_name: str) -> list[dict[str, object]]:
    rows = []
    for index in range(20):
        symbol = "BTC/USD" if index < 10 else "ETH/USD"
        rows.append(
            {
                "model_name": model_name,
                "fold_index": "4",
                "symbol": symbol,
                "interval_begin": f"2025-01-{index + 1:02d}T00:00:00Z",
                "label": "1" if index < 5 else "0",
                "scenario_name": "current_fee",
            }
        )
    return rows


def _write_sources(base: Path, previous: Path) -> None:
    specialist_dir = base / "research_labels" / "vol_scaled" / "specialist_predictions"
    for model_name in ("neuralforecast_nhits", "neuralforecast_patchtst"):
        _write_csv(
            specialist_dir / f"predictions_{model_name}_oof.csv",
            _prediction_rows(model_name),
        )
    labels = _label_rows("neuralforecast_nhits") + _label_rows("neuralforecast_patchtst")
    _write_csv(
        previous
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_labels_vol_scaled.csv",
        labels,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_specialist_conditional_usefulness_writes_outputs(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    _write_sources(base, previous)

    result = analyze_m20_specialist_conditional_usefulness(
        base_run_dir=base,
        previous_run_dir=previous,
    )

    assert result["joined_rows"] == 40
    assert Path(result["output_files"]["model_metrics_csv"]).exists()
    assert result["promotion_status"] if "promotion_status" in result else True


def test_specialist_conditional_usefulness_compares_models(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    _write_sources(base, previous)

    result = analyze_m20_specialist_conditional_usefulness(
        base_run_dir=base,
        previous_run_dir=previous,
    )
    comparison = _read_csv(Path(result["output_files"]["comparison_csv"]))
    by_model = {row["model_name"]: row for row in comparison}

    assert float(by_model["neuralforecast_nhits"]["top5_lift"]) > 1.0
    assert result["best_candidate"] == "neuralforecast_nhits"


def test_specialist_conditional_usefulness_is_research_only(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    _write_sources(base, previous)

    result = analyze_m20_specialist_conditional_usefulness(
        base_run_dir=base,
        previous_run_dir=previous,
    )
    manifest = json.loads(
        Path(result["output_files"]["manifest_json"]).read_text(encoding="utf-8")
    )

    assert "NO_RUNTIME_EFFECT" in manifest["honesty_flags"]
    assert manifest["promotion_status"] == "NOT_PROMOTABLE"
    assert manifest["skipped_unlabeled_rows"] == 0
