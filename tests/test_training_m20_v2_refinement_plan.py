"""Tests for M20 v2 refinement planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_v2_refinement_plan import plan_m20_v2_refinement

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _factory(base: Path) -> Path:
    factory = base / "research_labels" / "vol_scaled" / "strategy_candidate_v2_factory"
    _write_csv(
        factory / "candidate_metrics.csv",
        [
            {
                "strategy_family": "x",
                "candidate_name": "broad",
                "selected_rows": 1000,
                "coverage": 0.50,
                "mean_net_proxy": -0.01,
                "classification": "V2_STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE",
            }
        ],
    )
    _write_csv(
        factory / "candidate_decisions.csv",
        [
            {
                "candidate_name": "broad",
                "candidate_decision": "V2_STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE",
            }
        ],
    )
    (factory / "manifest.json").write_text(json.dumps({}), encoding="utf-8")
    return factory


def test_refinement_plan_detects_negative_economics_and_broad_coverage(tmp_path: Path) -> None:
    _factory(tmp_path)

    result = plan_m20_v2_refinement(source_run_dir=tmp_path)
    modes = {row["failure_mode"] for row in result["failure_mode_diagnostics"]}

    assert "NEGATIVE_NET_PROXY" in modes
    assert "BROAD_OR_HIGH_TURNOVER_COVERAGE" in modes
    assert result["recommendation"] == "BUILD_REFINED_V2_STRATEGY_DEFINITIONS"


def test_refinement_plan_preserves_research_only_statuses(tmp_path: Path) -> None:
    _factory(tmp_path)

    result = plan_m20_v2_refinement(source_run_dir=tmp_path)

    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]
    assert (Path(result["output_files"]["v2_refinement_plan_md"])).exists()
