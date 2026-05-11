"""Research-only plan for a generic M20 strategy-conditioned model factory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_OUTPUT_NAME = "strategy_model_factory_plan"
DEFAULT_SLICE_POLICY_NAME = "strategy_slice_policy_evaluator"
DEFAULT_REFINEMENT_NAME = "strategy_candidate_refinement"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "DESIGN_ONLY",
    "NO_TRAINING_EXECUTED",
    "NO_SCORING_EXECUTED",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
RECOMMENDATION = "DESIGN_REUSABLE_STRATEGY_CONDITIONED_MODEL_FACTORY_CONTRACT"
NEXT_ACTION = "REFINE_STRATEGY_CANDIDATES_BEFORE_MODEL_FACTORY_EXECUTION"


def plan_m20_strategy_model_factory(
    *,
    source_run_dir: Path,
    slice_policy_dir: Path | None = None,
    refinement_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Write a design-only generic model-factory plan from existing artifacts."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    vol_scaled_dir = source_dir / "research_labels" / "vol_scaled"
    policy_dir = (
        Path(slice_policy_dir).resolve()
        if slice_policy_dir is not None
        else vol_scaled_dir / DEFAULT_SLICE_POLICY_NAME
    )
    candidate_refinement_dir = (
        Path(refinement_dir).resolve()
        if refinement_dir is not None
        else vol_scaled_dir / DEFAULT_REFINEMENT_NAME
    )
    output_dir = vol_scaled_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_rows = _optional_csv(policy_dir / "candidate_decisions.csv")
    refined_rows = _optional_csv(candidate_refinement_dir / "refined_candidate_metrics.csv")
    policy_recommendation = _optional_json(policy_dir / "recommendation.json")
    candidate_inputs = _candidate_inputs(policy_rows, refined_rows)
    blockers = _blockers(policy_rows, policy_recommendation)
    required_artifacts = _required_artifacts()
    factory_contract = _factory_contract()
    recommendation = _recommendation(blockers)
    output_files = _output_files(output_dir)
    plan = {
        "source_run_dir": str(source_dir),
        "slice_policy_dir": str(policy_dir),
        "refinement_dir": str(candidate_refinement_dir),
        "candidate_input_count": len(candidate_inputs),
        "blocker_count": len(blockers),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "source_run_dir": str(source_dir),
        "source_artifacts": {
            "slice_policy_candidate_decisions": str(
                policy_dir / "candidate_decisions.csv"
            ),
            "slice_policy_recommendation": str(policy_dir / "recommendation.json"),
            "refined_candidate_metrics": str(
                candidate_refinement_dir / "refined_candidate_metrics.csv"
            ),
        },
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["strategy_model_factory_plan_json"]), plan)
    Path(output_files["strategy_model_factory_plan_md"]).write_text(
        _markdown(plan, blockers),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["candidate_inputs_csv"]), candidate_inputs)
    write_json_artifact(Path(output_files["required_artifacts_json"]), required_artifacts)
    write_json_artifact(Path(output_files["factory_contract_json"]), factory_contract)
    Path(output_files["manual_commands_md"]).write_text(
        _manual_commands(),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["blockers_csv"]), blockers)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **plan,
            "manifest": manifest,
            "candidate_inputs": candidate_inputs,
            "required_artifacts": required_artifacts,
            "factory_contract": factory_contract,
            "blockers": blockers,
            "recommendation_payload": recommendation,
        }
    )


def _candidate_inputs(
    policy_rows: Sequence[Mapping[str, str]],
    refined_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    policy_by_candidate = {
        (row.get("strategy_family", ""), row.get("candidate_name", "")): row
        for row in policy_rows
    }
    output = []
    for row in refined_rows:
        key = (row.get("strategy_family", ""), row.get("candidate_name", ""))
        policy = policy_by_candidate.get(key, {})
        output.append(
            {
                "strategy_family": key[0],
                "candidate_name": key[1],
                "candidate_source": "strategy_candidate_refinement",
                "best_slice": _best_slice(row),
                "refinement_decision": row.get("refinement_decision", ""),
                "source_mean_net_proxy": row.get("source_mean_net_proxy", ""),
                "best_slice_mean_net_proxy": row.get("best_slice_mean_net_proxy", ""),
                "slice_policy_decision": policy.get("candidate_decision", ""),
                "model_factory_status": _candidate_status(row, policy),
            }
        )
    return output


def _candidate_status(
    refined_row: Mapping[str, str],
    policy_row: Mapping[str, str],
) -> str:
    if policy_row.get("candidate_decision") == (
        "SLICE_POLICY_RESEARCH_WATCHLIST_POSITIVE_NET_PROXY"
    ):
        return "ELIGIBLE_FOR_FUTURE_RESEARCH_MODEL_FACTORY"
    if refined_row.get("refinement_decision") == "REFINED_SLICE_POLICY_WATCHLIST":
        return "NEEDS_GENERIC_SLICE_POLICY_CONFIRMATION"
    return "BLOCKED_PENDING_STRATEGY_REFINEMENT"


def _blockers(
    policy_rows: Sequence[Mapping[str, str]],
    recommendation: Mapping[str, Any],
) -> list[dict[str, str]]:
    rows = []
    if not policy_rows:
        rows.append(
            {
                "blocker": "STRATEGY_SLICE_POLICY_ARTIFACTS_MISSING",
                "detail": "Run generic strategy slice policy evaluator first.",
            }
        )
    if recommendation.get("recommendation") == "REFINE_STRATEGY_CANDIDATE_DEFINITIONS":
        rows.append(
            {
                "blocker": "NO_APPROVED_STRATEGY_POLICY_CANDIDATE",
                "detail": "Current generic slice policies remain negative or low-sample.",
            }
        )
    rows.append(
        {
            "blocker": "MODEL_FACTORY_EXECUTION_NOT_APPROVED",
            "detail": "This batch writes a contract only; no training or scoring is run.",
        }
    )
    return rows


def _required_artifacts() -> dict[str, Any]:
    return {
        "candidate_event_source": "strategy_candidate_factory/strategy_candidates.csv",
        "refinement_source": "strategy_candidate_refinement/refined_candidate_metrics.csv",
        "policy_source": "strategy_slice_policy_evaluator/policy_metrics.csv",
        "label_source": "research_labels/vol_scaled/fee_exceedance_labels_vol_scaled.csv",
        "economic_source": (
            "research_labels/vol_scaled/economic_outcome_artifacts/"
            "economic_outcomes.csv"
        ),
        "required_keys": ["symbol", "interval_begin", "fold_index"],
        "forbidden_inputs": [
            "future_return_as_feature",
            "net_value_proxy_as_feature",
            "label_columns_as_features",
            "runtime_registry_fields",
        ],
    }


def _factory_contract() -> dict[str, Any]:
    return {
        "factory_kind": "strategy_conditioned_model_factory",
        "model_factory_role": (
            "AutoGluon or any future model family acts as a tournament factory, "
            "not as a final universal model."
        ),
        "input_contract": {
            "candidate_id": "stable strategy_family:candidate_name:slice identifier",
            "training_rows": "rows selected by setup/filter only",
            "target_columns": "evaluation labels only, never candidate features",
            "economic_columns": "evaluation-only net/gross proxy outputs",
        },
        "output_contract": [
            "manifest.json",
            "report.json",
            "report.md",
            "model_candidate_metrics.csv",
            "policy_metrics.csv",
            "candidate_decisions.csv",
            "next_actions.csv",
            "recommendation.json",
        ],
        "required_statuses": list(HONESTY_FLAGS),
    }


def _recommendation(blockers: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "recommendation": RECOMMENDATION,
        "next_required_action": NEXT_ACTION,
        "execution_blockers": [row["blocker"] for row in blockers],
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _next_actions(recommendation: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action": str(recommendation["next_required_action"]),
            "rationale": "Current strategy candidates are not ready for model-factory execution.",
        },
        {
            "priority": "2",
            "action": "KEEP_MODEL_FACTORY_DESIGN_RESEARCH_ONLY",
            "rationale": "No training, scoring, runtime, promotion, backtest, or profit claim.",
        },
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "strategy_model_factory_plan_json": str(
            output_dir / "strategy_model_factory_plan.json"
        ),
        "strategy_model_factory_plan_md": str(
            output_dir / "strategy_model_factory_plan.md"
        ),
        "candidate_inputs_csv": str(output_dir / "candidate_inputs.csv"),
        "required_artifacts_json": str(output_dir / "required_artifacts.json"),
        "factory_contract_json": str(output_dir / "factory_contract.json"),
        "manual_commands_md": str(output_dir / "manual_commands.md"),
        "blockers_csv": str(output_dir / "blockers.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(plan: Mapping[str, Any], blockers: Sequence[Mapping[str, str]]) -> str:
    lines = [
        "# M20 Strategy Model Factory Plan",
        "",
        f"- Recommendation: `{plan['recommendation']}`",
        f"- Next required action: `{plan['next_required_action']}`",
        f"- Candidate inputs: `{plan['candidate_input_count']}`",
        "- Status: `RESEARCH_ONLY`, `DESIGN_ONLY`, `NO_TRAINING_EXECUTED`, "
        "`NO_SCORING_EXECUTED`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Blockers",
    ]
    for row in blockers:
        lines.append(f"- `{row['blocker']}`: {row['detail']}")
    lines.append("")
    return "\n".join(lines)


def _manual_commands() -> str:
    return "\n".join(
        [
            "# M20 Strategy Model Factory Manual Commands",
            "",
            "No training command is approved by this plan.",
            "Codex must stop here unless Arash explicitly approves a later execution batch.",
            "",
        ]
    )


def _best_slice(row: Mapping[str, str]) -> str:
    family = row.get("best_slice_family", "")
    value = row.get("best_slice_value", "")
    if not family:
        return ""
    return f"{family}={value}"


def _optional_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["plan_m20_strategy_model_factory"]
