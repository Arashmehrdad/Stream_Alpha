"""Research-only validation and search-breadth audit for M20 policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_policy_research_common import (
    HONESTY_FLAGS,
    MIN_POLICY_ROWS,
    present,
    read_csv_rows,
    to_float,
    vol_scaled_dir,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_POLICY_EVAL_NAME = "decision_policy_eval"
DEFAULT_OUTPUT_NAME = "policy_validation_audit"


def audit_m20_policy_validation(
    *,
    source_run_dir: Path,
    policy_eval_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Audit search breadth and validation risk for policy-evaluation outputs."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    research_dir = vol_scaled_dir(source_dir)
    eval_dir = (
        Path(policy_eval_dir).resolve()
        if policy_eval_dir
        else research_dir / DEFAULT_POLICY_EVAL_NAME
    )
    output_dir = research_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_csv_rows(eval_dir / "policy_metrics.csv")
    decisions = read_csv_rows(eval_dir / "candidate_decisions.csv")
    baseline = read_csv_rows(eval_dir / "baseline_comparison.csv")
    search_breadth = read_csv_rows(eval_dir / "search_breadth.csv")
    if not metrics:
        raise ValueError(f"Missing policy metrics: {eval_dir / 'policy_metrics.csv'}")
    breadth_audit = _search_breadth_audit(metrics, search_breadth)
    stability = _stability_audit(metrics)
    paired_readiness = _paired_comparison_readiness(baseline)
    low_sample = _low_sample_warnings(metrics)
    recommendation = _recommendation(decisions, low_sample)
    output_files = _output_files(output_dir)
    manifest = {
        "source_run_dir": str(source_dir),
        "policy_eval_dir": str(eval_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    report = {
        "summary": "M20 policy validation and search-breadth audit.",
        "policy_eval_dir": str(eval_dir),
        "policy_count": len(metrics),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["policy_validation_report_json"]), report)
    Path(output_files["policy_validation_report_md"]).write_text(
        _markdown(report, breadth_audit, recommendation),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["search_breadth_audit_csv"]), breadth_audit)
    write_csv_artifact(Path(output_files["stability_audit_csv"]), stability)
    write_csv_artifact(
        Path(output_files["paired_comparison_readiness_csv"]),
        paired_readiness,
    )
    write_csv_artifact(Path(output_files["low_sample_warnings_csv"]), low_sample)
    write_csv_artifact(Path(output_files["candidate_decisions_csv"]), decisions)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "search_breadth_audit": breadth_audit,
            "stability_audit": stability,
            "paired_comparison_readiness": paired_readiness,
            "low_sample_warnings": low_sample,
            "candidate_decisions": decisions,
            "recommendation_payload": recommendation,
        }
    )


def _search_breadth_audit(
    metrics: Sequence[Mapping[str, Any]],
    breadth_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    payload = breadth_rows[0] if breadth_rows else {}
    return [
        {
            "audit_name": "POLICY_SEARCH_BREADTH",
            "policy_configurations_tried": payload.get(
                "policy_configurations_tried",
                len(metrics),
            ),
            "candidate_definitions_referenced": payload.get(
                "candidate_definitions_referenced",
                "",
            ),
            "warning": "MULTIPLE_COMPARISON_RESEARCH_ONLY",
            "promotion_status": "NOT_PROMOTABLE",
        }
    ]


def _stability_audit(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    positive = [
        row for row in metrics
        if present(row.get("mean_net_value_proxy"))
        and to_float(row["mean_net_value_proxy"]) > 0.0
    ]
    return [
        {
            "audit_name": "SLICE_STABILITY_PROXY",
            "positive_policy_count": len(positive),
            "stability_status": (
                "REQUIRES_SEPARATE_OUT_OF_SAMPLE_CONFIRMATION"
                if positive
                else "NO_POSITIVE_PROXY_POLICY"
            ),
            "final_segment_status": "NOT_CLAIMED_UNTOUCHED",
        }
    ]


def _paired_comparison_readiness(
    baseline_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "audit_name": "PAIRED_BASELINE_COMPARISON",
            "comparison_rows": len(baseline_rows),
            "readiness_status": (
                "PAIRED_PROXY_COMPARISON_AVAILABLE"
                if baseline_rows
                else "BASELINE_COMPARISON_NOT_AVAILABLE"
            ),
            "claim_status": "PROXY_COMPARISON_ONLY_NOT_BACKTEST",
        }
    ]


def _low_sample_warnings(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy_name": str(row["policy_name"]),
            "selected_rows": row.get("selected_rows", ""),
            "warning": "LOW_SAMPLE_POLICY_DIAGNOSTIC_ONLY",
        }
        for row in metrics
        if 0 < to_float(row.get("selected_rows")) < MIN_POLICY_ROWS
    ]


def _recommendation(
    decisions: Sequence[Mapping[str, Any]],
    low_sample_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    positive = any(
        row.get("policy_decision") == "POLICY_RESEARCH_WATCHLIST_POSITIVE_PROXY"
        for row in decisions
    )
    recommendation = (
        "PLAN_SHADOW_ONLY_ADAPTATION_OBSERVER"
        if positive and not low_sample_rows
        else "DESIGN_TRADING_AWARE_RESEARCH_LABELS"
    )
    return {
        "recommendation": recommendation,
        "next_required_action": recommendation,
        "evidence_blockers": [
            "NOT_BACKTEST",
            "NOT_RUNTIME_READY",
            "NOT_PROMOTABLE",
            "NO_PROFIT_CLAIM",
        ],
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
            "rationale": "Policy evidence remains research-only and not promotable.",
        }
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "policy_validation_report_json": str(output_dir / "policy_validation_report.json"),
        "policy_validation_report_md": str(output_dir / "policy_validation_report.md"),
        "search_breadth_audit_csv": str(output_dir / "search_breadth_audit.csv"),
        "stability_audit_csv": str(output_dir / "stability_audit.csv"),
        "paired_comparison_readiness_csv": str(
            output_dir / "paired_comparison_readiness.csv"
        ),
        "low_sample_warnings_csv": str(output_dir / "low_sample_warnings.csv"),
        "candidate_decisions_csv": str(output_dir / "candidate_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    breadth_audit: Sequence[Mapping[str, Any]],
    recommendation: Mapping[str, Any],
) -> str:
    lines = [
        "# M20 Policy Validation Audit",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Policy count: `{report['policy_count']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Search Breadth",
    ]
    lines.extend(
        f"- `{row['audit_name']}`: `{row['warning']}`"
        for row in breadth_audit
    )
    lines.extend(["", "## Evidence Blockers"])
    lines.extend(f"- `{blocker}`" for blocker in recommendation["evidence_blockers"])
    return "\n".join(lines) + "\n"


__all__ = ["audit_m20_policy_validation"]
