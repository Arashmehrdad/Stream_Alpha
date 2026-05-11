"""Focused tests for M20 strategy-conditioned model factory planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_strategy_model_factory_plan import (
    plan_m20_strategy_model_factory,
)

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_sources(base: Path, *, policy_recommendation: str) -> None:
    vol = base / "research_labels" / "vol_scaled"
    refinement = vol / "strategy_candidate_refinement"
    policy = vol / "strategy_slice_policy_evaluator"
    _write_csv(
        refinement / "refined_candidate_metrics.csv",
        [
            {
                "strategy_family": "return_reversal",
                "candidate_name": "return_positive",
                "source_mean_net_proxy": "-0.002",
                "best_slice_family": "symbol",
                "best_slice_value": "BTC/USD",
                "best_slice_mean_net_proxy": "0.001",
                "refinement_decision": "REFINED_SLICE_POLICY_WATCHLIST",
            }
        ],
    )
    _write_csv(
        policy / "candidate_decisions.csv",
        [
            {
                "strategy_family": "return_reversal",
                "candidate_name": "return_positive",
                "candidate_decision": (
                    "SLICE_POLICY_RESEARCH_WATCHLIST_POSITIVE_NET_PROXY"
                ),
            }
        ],
    )
    (policy / "recommendation.json").parent.mkdir(parents=True, exist_ok=True)
    (policy / "recommendation.json").write_text(
        json.dumps({"recommendation": policy_recommendation}),
        encoding="utf-8",
    )


def test_plan_writes_design_contract_without_training(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        policy_recommendation="DESIGN_GENERIC_STRATEGY_MODEL_FACTORY",
    )

    result = plan_m20_strategy_model_factory(source_run_dir=tmp_path)
    contract = json.loads(
        Path(result["output_files"]["factory_contract_json"]).read_text(
            encoding="utf-8"
        )
    )
    commands = Path(result["output_files"]["manual_commands_md"]).read_text(
        encoding="utf-8"
    )

    assert contract["factory_kind"] == "strategy_conditioned_model_factory"
    assert "tournament factory" in contract["model_factory_role"]
    assert "No training command is approved" in commands
    assert "NO_TRAINING_EXECUTED" in result["honesty_flags"]


def test_candidate_inputs_are_generic_and_not_promotable(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        policy_recommendation="DESIGN_GENERIC_STRATEGY_MODEL_FACTORY",
    )

    result = plan_m20_strategy_model_factory(source_run_dir=tmp_path)
    inputs = _read_csv(Path(result["output_files"]["candidate_inputs_csv"]))

    assert inputs[0]["strategy_family"] == "return_reversal"
    assert inputs[0]["model_factory_status"] == (
        "ELIGIBLE_FOR_FUTURE_RESEARCH_MODEL_FACTORY"
    )
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"


def test_current_negative_policy_result_blocks_execution(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        policy_recommendation="REFINE_STRATEGY_CANDIDATE_DEFINITIONS",
    )

    result = plan_m20_strategy_model_factory(source_run_dir=tmp_path)
    blockers = _read_csv(Path(result["output_files"]["blockers_csv"]))
    blocker_names = {row["blocker"] for row in blockers}

    assert "NO_APPROVED_STRATEGY_POLICY_CANDIDATE" in blocker_names
    assert result["next_required_action"] == (
        "REFINE_STRATEGY_CANDIDATES_BEFORE_MODEL_FACTORY_EXECUTION"
    )


def test_missing_policy_artifacts_fail_as_blocker_not_training(tmp_path: Path) -> None:
    result = plan_m20_strategy_model_factory(source_run_dir=tmp_path)
    blockers = _read_csv(Path(result["output_files"]["blockers_csv"]))

    assert blockers[0]["blocker"] == "STRATEGY_SLICE_POLICY_ARTIFACTS_MISSING"
    assert result["candidate_input_count"] == 0


def test_outputs_include_research_only_statuses(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        policy_recommendation="REFINE_STRATEGY_CANDIDATE_DEFINITIONS",
    )

    result = plan_m20_strategy_model_factory(source_run_dir=tmp_path)
    plan = json.loads(
        Path(result["output_files"]["strategy_model_factory_plan_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "NO_RUNTIME_EFFECT" in plan["honesty_flags"]
    assert "NOT_PROMOTABLE" in plan["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in plan["honesty_flags"]


def test_no_runtime_imports_or_registry_writes() -> None:
    source = Path("app/training/m20_strategy_model_factory_plan.py").read_text(
        encoding="utf-8"
    )

    assert "app.inference" not in source
    assert "app.trading" not in source
    assert "app.training.registry" not in source
