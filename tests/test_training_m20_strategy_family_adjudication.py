"""Focused tests for M20 strategy-family adjudication."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_strategy_family_adjudication import (
    adjudicate_m20_strategy_families,
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


def _write_family(
    base: Path,
    directory: str,
    recommendation: str,
    setups: dict[str, list[float]],
    write_overlap: bool = True,
) -> None:
    rows = []
    overlap_rows = []
    for setup_name, lifts in setups.items():
        for index, lift in enumerate(lifts):
            run_label = f"run_{index}"
            rows.append(
                {
                    "run_label": run_label,
                    "setup_name": setup_name,
                    "setup_frequency": 0.2 + index * 0.01,
                    "setup_positive_rate": 0.1 * lift,
                    "lift_vs_base": lift,
                    "net_proxy_mean": 0.0,
                }
            )
            overlap_rows.append(
                {
                    "run_label": run_label,
                    "setup_name": setup_name,
                    "rank_gate_overlap_rate": 0.01,
                    "overlap_positive_rate": 0.5,
                }
            )
    family_dir = base / "research_labels" / "vol_scaled" / directory
    _write_csv(family_dir / "setup_metrics.csv", rows)
    if write_overlap:
        _write_csv(family_dir / "rank_gate_overlap.csv", overlap_rows)
    _write_json(family_dir / "recommendation.json", {"recommendation": recommendation})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_strategy_family_adjudication_ranks_by_min_lift(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_family(
        base,
        "volatility_expansion_research",
        "KEEP_VOLATILITY_EXPANSION_AS_RESEARCH_DIAGNOSTIC_CANDIDATE",
        {"vol_plus_range_high": [1.7, 1.6, 1.8]},
    )
    _write_family(
        base,
        "momentum_breakout_research",
        "KEEP_MOMENTUM_BREAKOUT_AS_RESEARCH_DIAGNOSTIC_CANDIDATE",
        {"realized_vol_high": [1.4, 1.35, 1.5]},
    )
    _write_family(
        base,
        "range_mean_reversion_research",
        "KEEP_RANGE_MEAN_REVERSION_AS_RESEARCH_DIAGNOSTIC_CANDIDATE",
        {"macd_near_zero": [1.12, 1.08, 1.2]},
        write_overlap=False,
    )

    result = adjudicate_m20_strategy_families(base_run_dir=base)
    family_rows = _read_csv(Path(result["output_files"]["family_comparison_csv"]))

    assert result["recommendation"] == "TEST_VOLATILITY_EXPANSION_NEXT"
    assert family_rows[0]["family_id"] == "volatility_expansion"
    assert family_rows[0]["family_classification"] == "PRIMARY_FAMILY"


def test_strategy_family_adjudication_preserves_weaker_families(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_family(base, "volatility_expansion_research", "KEEP", {"vol": [1.6, 1.6]})
    _write_family(base, "momentum_breakout_research", "KEEP", {"mom": [1.35, 1.36]})
    _write_family(base, "range_mean_reversion_research", "KEEP", {"range": [1.07, 1.06]})

    result = adjudicate_m20_strategy_families(base_run_dir=base)
    family_rows = _read_csv(Path(result["output_files"]["family_comparison_csv"]))
    classifications = {row["family_id"]: row["family_classification"] for row in family_rows}

    assert classifications["momentum_breakout"] == "SECONDARY_FAMILY"
    assert classifications["range_mean_reversion"] == "WATCHLIST_FAMILY"
    assert Path(result["output_files"]["rank_gate_overlap_summary_csv"]).exists()


def test_strategy_family_adjudication_flags_non_promotable(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_family(base, "volatility_expansion_research", "KEEP", {"vol": [1.0]})
    _write_family(base, "momentum_breakout_research", "KEEP", {"mom": [1.0]})
    _write_family(base, "range_mean_reversion_research", "KEEP", {"range": [1.0]})

    result = adjudicate_m20_strategy_families(base_run_dir=base)

    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]
    assert Path(result["output_files"]["report_md"]).exists()
