"""Final research-only M20 input redesign decision artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_policy_research_common import (
    HONESTY_FLAGS,
    present,
    read_csv_rows,
    read_json_payload,
    to_float,
    vol_scaled_dir,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_OUTPUT_NAME = "m20_input_redesign_decision"


def write_m20_input_redesign_decision(
    *,
    source_run_dir: Path,
    policy_eval_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Write the final input-redesign decision from existing artifacts."""
    source_dir = Path(source_run_dir).resolve()
    research_dir = vol_scaled_dir(source_dir)
    eval_dir = (
        Path(policy_eval_dir).resolve()
        if policy_eval_dir
        else research_dir / "m20_redesigned_policy_eval"
    )
    output_dir = research_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_csv_rows(eval_dir / "policy_metrics.csv")
    redesigned = read_json_payload(
        research_dir / "m20_redesigned_research_inputs" / "recommendation.json"
    )
    decision = _decision(metrics, redesigned)
    evidence = _evidence(metrics, redesigned)
    recommendation = _recommendation(decision)
    output_files = _output_files(output_dir)
    report = {
        "summary": "M20 input-redesign final decision.",
        "final_decision": decision,
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "source_run_dir": str(source_dir),
        "policy_eval_dir": str(eval_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["decision_json"]), report)
    Path(output_files["decision_md"]).write_text(_markdown(report, evidence), "utf-8")
    write_csv_artifact(Path(output_files["decision_evidence_csv"]), evidence)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "decision_evidence": evidence,
            "recommendation_payload": recommendation,
        }
    )


def _decision(
    metrics: Sequence[Mapping[str, str]],
    redesigned: Mapping[str, Any],
) -> str:
    if redesigned.get("recommendation") == "M20_BLOCKED_MISSING_SAFE_INPUTS":
        return "M20_BLOCKED_MISSING_SAFE_INPUTS"
    positive = [
        row for row in metrics
        if present(row.get("mean_net_value_proxy"))
        and to_float(row["mean_net_value_proxy"]) > 0.0
        and to_float(row.get("selected_rows")) >= 1000
    ]
    if positive:
        return "M20_READY_FOR_CONSERVATIVE_VALIDATION_AUDIT"
    if metrics:
        return "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
    return "M20_INPUTS_INSUFFICIENT_NEED_DATA_UPGRADE"


def _evidence(
    metrics: Sequence[Mapping[str, str]],
    redesigned: Mapping[str, Any],
) -> list[dict[str, Any]]:
    best = _best_metric(metrics)
    return [
        {
            "evidence_name": "redesigned_input_recommendation",
            "value": redesigned.get("recommendation", ""),
        },
        {
            "evidence_name": "policy_count",
            "value": len(metrics),
        },
        {
            "evidence_name": "best_policy",
            "value": best.get("policy_name", ""),
        },
        {
            "evidence_name": "best_mean_net_value_proxy",
            "value": best.get("mean_net_value_proxy", ""),
        },
    ]


def _best_metric(metrics: Sequence[Mapping[str, str]]) -> Mapping[str, str]:
    usable = [row for row in metrics if present(row.get("mean_net_value_proxy"))]
    if not usable:
        return {}
    return max(usable, key=lambda row: to_float(row["mean_net_value_proxy"]))


def _recommendation(decision: str) -> dict[str, Any]:
    if decision == "M20_READY_FOR_CONSERVATIVE_VALIDATION_AUDIT":
        recommendation = "RUN_CONSERVATIVE_VALIDATION_AUDIT"
    elif decision == "M20_BLOCKED_MISSING_SAFE_INPUTS":
        recommendation = "ADD_HIGHER_QUALITY_SAFE_INPUT_SOURCE"
    else:
        recommendation = "PAUSE_M20_POLICY_ROUTE_AND_REDESIGN_INPUTS"
    return {
        "recommendation": recommendation,
        "next_required_action": recommendation,
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _next_actions(recommendation: Mapping[str, Any]) -> list[dict[str, str]]:
    return [{"priority": "1", "action": str(recommendation["next_required_action"])}]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "decision_json": str(output_dir / "m20_input_redesign_decision.json"),
        "decision_md": str(output_dir / "m20_input_redesign_decision.md"),
        "decision_evidence_csv": str(output_dir / "decision_evidence.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# M20 Input Redesign Decision",
        "",
        f"- Final decision: `{report['final_decision']}`",
        f"- Recommendation: `{report['recommendation']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Evidence",
    ]
    lines.extend(f"- `{row['evidence_name']}`: `{row['value']}`" for row in evidence)
    return "\n".join(lines) + "\n"


__all__ = ["write_m20_input_redesign_decision"]
