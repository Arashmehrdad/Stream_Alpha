"""Focused tests for M20 momentum-breakout research diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_momentum_breakout_research import analyze_m20_momentum_breakout

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_run(run_dir: Path) -> None:
    features = []
    labels = []
    for index in range(30):
        symbol = "BTC/USD" if index % 2 else "ETH/USD"
        timestamp = f"2025-01-{index + 1:02d}T00:00:00Z"
        label = 1 if index % 3 == 0 else 0
        features.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "open_price": 100,
                "high_price": 103,
                "low_price": 99,
                "close_price": 102 + index * 0.1,
                "vwap": 101,
                "volume": 1000 + index,
                "log_return_1": 0.001 if index % 3 == 0 else -0.001,
                "momentum_3": 0.002 if index % 3 == 0 else -0.001,
                "realized_vol_12": 0.002 + index * 0.0001,
                "macd_line_12_26": 0.1 if index % 3 == 0 else -0.1,
                "close_zscore_12": 1.2 if index % 3 == 0 else 0.0,
            }
        )
        labels.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "label": label,
                "scenario_name": "current_fee",
            }
        )
    _write_csv(run_dir / "training_frame" / "m20_training_frame_features.csv", features)
    _write_csv(
        run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_labels_vol_scaled.csv",
        labels,
    )
    _write_json(
        run_dir / "training_frame" / "m20_training_frame_feature_columns.json",
        {
            "feature_columns": [
                "macd_line_12_26",
                "log_return_1",
                "close_price",
                "momentum_3",
                "realized_vol_12",
                "volume",
            ]
        },
    )


def test_momentum_breakout_writes_outputs(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_momentum_breakout(base_run_dir=base)

    assert "RESEARCH_ONLY" in result["honesty_flags"]
    assert result["setup_count"] == 10
    assert Path(result["output_files"]["setup_metrics_csv"]).exists()
    assert Path(result["output_files"]["rank_gate_overlap_csv"]).exists()


def test_momentum_breakout_blocks_missing_required_features(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_json(
        base / "training_frame" / "m20_training_frame_feature_columns.json",
        {"feature_columns": ["close_price"]},
    )

    result = analyze_m20_momentum_breakout(base_run_dir=base)

    assert result["recommendation"] == "BLOCKED_REQUIRED_FEATURES_MISSING"
    assert "macd_line_12_26" in result["missing_required_features"]
