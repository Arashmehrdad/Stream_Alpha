"""Tests for M20 shadow observer planning."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_shadow_observer_plan import plan_m20_shadow_observer

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _policy_eval(base: Path, positive: bool) -> Path:
    eval_dir = base / "research_labels" / "vol_scaled" / "trading_aware_policy_eval"
    _write_csv(
        eval_dir / "policy_metrics.csv",
        [
            {
                "policy_name": "policy",
                "selected_rows": 1200,
                "mean_net_value_proxy": 0.01 if positive else -0.01,
            }
        ],
    )
    _write_csv(
        eval_dir / "candidate_decisions.csv",
        [
            {
                "policy_name": "policy",
                "policy_decision": "POLICY_RESEARCH_WATCHLIST_POSITIVE_PROXY"
                if positive
                else "POLICY_ECONOMICS_NEGATIVE",
            }
        ],
    )
    return eval_dir


def test_shadow_observer_plan_blocks_when_no_policy_is_ready(tmp_path: Path) -> None:
    _policy_eval(tmp_path, positive=False)

    result = plan_m20_shadow_observer(source_run_dir=tmp_path)
    blockers = {row["blocker"] for row in result["blockers"]}

    assert "NO_POLICY_READY_FOR_SHADOW_OBSERVATION" in blockers
    assert result["recommendation"] == "PAUSE_M20_POLICY_ROUTE_AND_REDESIGN_INPUTS"


def test_shadow_observer_plan_remains_research_only_for_plausible_policy(
    tmp_path: Path,
) -> None:
    _policy_eval(tmp_path, positive=True)

    result = plan_m20_shadow_observer(source_run_dir=tmp_path)

    assert result["recommendation"] == "DESIGN_SHADOW_ONLY_POLICY_OBSERVATION_ARTIFACTS"
    assert "NO_RUNTIME_EFFECT" in result["overall_status"]
    assert "NOT_PROMOTABLE" in result["overall_status"]
    assert "NO_PROFIT_CLAIM" in result["overall_status"]
