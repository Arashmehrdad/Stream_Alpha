"""Focused tests for M20 volatility-expansion deep-dive diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_volatility_expansion_deep_dive import (
    analyze_m20_volatility_expansion_deep_dive,
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
    selected = []
    for index in range(30):
        timestamp = f"2025-01-{index + 1:02d}T00:00:00Z"
        high_vol = index >= 20
        label = 1 if index >= 19 else 0
        symbol = "BTC/USD" if index % 2 else "ETH/USD"
        features.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "high_price": 105 if high_vol else 101,
                "low_price": 95 if high_vol else 99,
                "close_price": 100,
                "volume": 2000 if high_vol else 1000,
                "realized_vol_12": (
                    0.02 + index * 0.0001 if high_vol else 0.001 + index * 0.00001
                ),
                "log_return_1": 0.01 if high_vol else 0.0001,
                "lag_log_return_1": 0.005 if high_vol else -0.0001,
                "macd_line_12_26": 0.2 if high_vol else 0.01,
            }
        )
        labels.append(
            {
                "symbol": symbol,
                "interval_begin": timestamp,
                "fold_index": 4,
                "label": label,
                "scenario_name": "current_fee",
                "future_return": 0.01 if label else -0.002,
                "cost_per_trade": 0.001,
            }
        )
        if index in (20, 21):
            selected.append(
                {
                    "run_label": "base",
                    "symbol": symbol,
                    "interval_begin": timestamp,
                    "fold_index": 4,
                }
            )
    _write_csv(run_dir / "training_frame" / "m20_training_frame_features.csv", features)
    _write_csv(
        run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_labels_vol_scaled.csv",
        labels,
    )
    _write_csv(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "rank_gate_net_diagnostics"
        / "selected_row_diagnostics.csv",
        selected,
    )
    _write_json(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "rank_gate_evidence_packet"
        / "rank_gate_evidence_packet.json",
        {"window_metrics": [{"window_label": "base", "run_dir": str(run_dir)}]},
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_deep_dive_writes_required_outputs(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_volatility_expansion_deep_dive(base_run_dir=base)

    assert "RESEARCH_ONLY_VOLATILITY_EXPANSION_DEEP_DIVE" in result["honesty_flags"]
    assert Path(result["output_files"]["setup_deep_metrics_csv"]).exists()
    assert Path(result["output_files"]["condition_intersection_matrix_csv"]).exists()


def test_deep_dive_metrics_and_overlap_are_deterministic(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)

    result = analyze_m20_volatility_expansion_deep_dive(base_run_dir=base)
    rows = _read_csv(Path(result["output_files"]["setup_deep_metrics_csv"]))
    vol_rows = [row for row in rows if row["setup_name"] == "realized_vol_high"]

    assert vol_rows
    assert float(vol_rows[0]["lift_vs_base"]) > 1.0
    assert int(vol_rows[0]["rank_gate_overlap_rows"]) == 2
    assert int(vol_rows[0]["false_positive_count"]) == 0


def test_deep_dive_handles_missing_rank_gate_overlap(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_run(base)
    (
        base
        / "research_labels"
        / "vol_scaled"
        / "rank_gate_net_diagnostics"
        / "selected_row_diagnostics.csv"
    ).unlink()

    result = analyze_m20_volatility_expansion_deep_dive(base_run_dir=base)
    rows = _read_csv(Path(result["output_files"]["rank_gate_overlap_deep_csv"]))

    assert rows
    assert all(int(row["rank_gate_overlap_rows"]) == 0 for row in rows)
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
