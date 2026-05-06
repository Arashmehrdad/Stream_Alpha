"""Focused tests for M20 volatility-expansion research diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_volatility_expansion_research import (
    analyze_m20_volatility_expansion,
)

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
        timestamp = f"2025-01-{index + 1:02d}T00:00:00Z"
        high_vol = index >= 20
        label = 1 if high_vol else 0
        features.append(
            {
                "symbol": "BTC/USD" if index % 2 else "ETH/USD",
                "interval_begin": timestamp,
                "fold_index": 4,
                "open_price": 100,
                "high_price": 105 if high_vol else 101,
                "low_price": 95 if high_vol else 99,
                "close_price": 100,
                "volume": 2000 if high_vol else 1000,
                "realized_vol_12": 0.02 + index * 0.0001 if high_vol else 0.001 + index * 0.00001,
                "log_return_1": 0.01 if high_vol else 0.0001,
                "lag_log_return_1": 0.005 if high_vol else -0.0001,
                "macd_line_12_26": 0.2 if high_vol else 0.01,
            }
        )
        labels.append(
            {
                "symbol": "BTC/USD" if index % 2 else "ETH/USD",
                "interval_begin": timestamp,
                "fold_index": 4,
                "label": label,
                "scenario_name": "current_fee",
                "future_return": 0.01 if label else -0.002,
                "cost_per_trade": 0.001,
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
                "realized_vol_12",
                "high_price",
                "low_price",
                "close_price",
                "volume",
                "log_return_1",
                "lag_log_return_1",
                "macd_line_12_26",
            ]
        },
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_volatility_expansion_writes_outputs(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_volatility_expansion(base_run_dir=base)

    assert "RESEARCH_ONLY" in result["honesty_flags"]
    assert result["setup_count"] == 10
    assert Path(result["output_files"]["setup_metrics_csv"]).exists()
    assert Path(result["output_files"]["rank_gate_overlap_csv"]).exists()


def test_volatility_expansion_stable_high_vol_candidate(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_volatility_expansion(base_run_dir=base)
    rows = _read_csv(Path(result["output_files"]["setup_metrics_csv"]))
    high_vol = [row for row in rows if row["setup_name"] == "realized_vol_high"][0]

    assert result["recommendation"] == "KEEP_VOLATILITY_EXPANSION_AS_RESEARCH_DIAGNOSTIC_CANDIDATE"
    assert float(high_vol["lift_vs_base"]) > 1.0
    assert float(high_vol["net_proxy_mean"]) > 0.0


def test_volatility_expansion_blocks_missing_required_features(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_json(
        base / "training_frame" / "m20_training_frame_feature_columns.json",
        {"feature_columns": ["close_price"]},
    )

    result = analyze_m20_volatility_expansion(base_run_dir=base)

    assert result["recommendation"] == "BLOCKED_REQUIRED_FEATURES_MISSING"
    assert "realized_vol_12" in result["missing_required_features"]
