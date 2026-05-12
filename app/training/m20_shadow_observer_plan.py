"""Research-only M20 shadow adaptation observer plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_policy_research_common import (
    HONESTY_FLAGS,
    present,
    read_csv_rows,
    to_float,
    vol_scaled_dir,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_OUTPUT_NAME = "shadow_adaptation_observer_plan"
DEFAULT_POLICY_EVAL_NAME = "trading_aware_policy_eval"
FALLBACK_POLICY_EVAL_NAME = "decision_policy_eval"


def plan_m20_shadow_observer(
    *,
    source_run_dir: Path,
    policy_eval_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Write a research-only shadow-observer plan for M20 policies."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    research_dir = vol_scaled_dir(source_dir)
    eval_dir = _resolve_policy_eval_dir(research_dir, policy_eval_dir)
    output_dir = research_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_csv_rows(eval_dir / "policy_metrics.csv")
    decisions = read_csv_rows(eval_dir / "candidate_decisions.csv")
    plausible = _plausible_policies(metrics)
    blockers = _blockers(plausible)
    recommendation = _recommendation(plausible)
    monitored = _monitored_metrics(plausible)
    contract = _observer_contract(plausible)
    output_files = _output_files(output_dir)
    manifest = {
        "source_run_dir": str(source_dir),
        "policy_eval_dir": str(eval_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    report = {
        "summary": "Research-only M20 shadow adaptation observer plan.",
        "policy_eval_dir": str(eval_dir),
        "plausible_policy_count": len(plausible),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["shadow_observer_plan_json"]), report)
    Path(output_files["shadow_observer_plan_md"]).write_text(
        _markdown(report, blockers),
        encoding="utf-8",
    )
    write_json_artifact(Path(output_files["observer_contract_json"]), contract)
    write_csv_artifact(Path(output_files["monitored_metrics_csv"]), monitored)
    write_csv_artifact(Path(output_files["blockers_csv"]), blockers)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "observer_contract": contract,
            "monitored_metrics": monitored,
            "blockers": blockers,
            "candidate_decisions": decisions,
            "recommendation_payload": recommendation,
        }
    )


def _resolve_policy_eval_dir(research_dir: Path, policy_eval_dir: Path | None) -> Path:
    if policy_eval_dir:
        return Path(policy_eval_dir).resolve()
    preferred = research_dir / DEFAULT_POLICY_EVAL_NAME
    if preferred.exists():
        return preferred
    return research_dir / FALLBACK_POLICY_EVAL_NAME


def _plausible_policies(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if present(row.get("mean_net_value_proxy"))
        and to_float(row["mean_net_value_proxy"]) > 0.0
        and to_float(row.get("selected_rows")) >= 1000
    ]


def _blockers(plausible: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    base = [
        {"blocker": "NOT_BACKTEST", "rationale": "No backtest claim is made."},
        {"blocker": "NOT_RUNTIME_READY", "rationale": "No runtime path is changed."},
        {"blocker": "NOT_PROMOTABLE", "rationale": "No promotion gate is satisfied."},
        {"blocker": "NO_PROFIT_CLAIM", "rationale": "Proxy metrics are not profit evidence."},
    ]
    if not plausible:
        base.insert(
            0,
            {
                "blocker": "NO_POLICY_READY_FOR_SHADOW_OBSERVATION",
                "rationale": "No adequate-sample positive proxy policy was found.",
            },
        )
    return base


def _monitored_metrics(plausible: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    policies = [str(row["policy_name"]) for row in plausible] or ["NO_POLICY_SELECTED"]
    metrics = (
        "calibration_drift",
        "threshold_drift",
        "regime_acceptance_rate",
        "rolling_net_proxy_by_symbol",
        "rolling_net_proxy_by_regime",
        "coverage_and_abstention_rate",
    )
    return [
        {
            "policy_name": policy_name,
            "metric_name": metric,
            "observer_mode": "SHADOW_ONLY_PLAN_NOT_RUNTIME",
        }
        for policy_name in policies
        for metric in metrics
    ]


def _observer_contract(plausible: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "m20_shadow_observer_plan_v1",
        "eligible_policy_count": len(plausible),
        "allowed_mode": "SHADOW_ONLY",
        "forbidden_actions": [
            "runtime_adaptation",
            "registry_promotion",
            "paper_or_live_execution_change",
            "profitability_claim",
        ],
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _recommendation(plausible: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recommendation = (
        "DESIGN_SHADOW_ONLY_POLICY_OBSERVATION_ARTIFACTS"
        if plausible
        else "PAUSE_M20_POLICY_ROUTE_AND_REDESIGN_INPUTS"
    )
    return {
        "recommendation": recommendation,
        "next_required_action": recommendation,
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
            "rationale": "Keep policy adaptation outside runtime until future approval.",
        }
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "shadow_observer_plan_json": str(output_dir / "shadow_observer_plan.json"),
        "shadow_observer_plan_md": str(output_dir / "shadow_observer_plan.md"),
        "observer_contract_json": str(output_dir / "observer_contract.json"),
        "monitored_metrics_csv": str(output_dir / "monitored_metrics.csv"),
        "blockers_csv": str(output_dir / "blockers.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    blockers: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Shadow Adaptation Observer Plan",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Plausible policy count: `{report['plausible_policy_count']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Blockers",
    ]
    lines.extend(f"- `{row['blocker']}`: {row['rationale']}" for row in blockers)
    return "\n".join(lines) + "\n"


__all__ = ["plan_m20_shadow_observer"]
