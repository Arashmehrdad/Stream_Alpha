"""Research-only M20 input failure analyzer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_policy_research_common import (
    HONESTY_FLAGS,
    read_csv_rows,
    read_json_payload,
    vol_scaled_dir,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_OUTPUT_NAME = "m20_input_failure_analysis"


def analyze_m20_input_failures(
    *,
    source_run_dir: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Summarize why the current M20 candidate and policy inputs failed."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    research_dir = vol_scaled_dir(source_dir)
    output_dir = research_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    refined_metrics = read_csv_rows(
        research_dir / "strategy_candidate_v2_refined_factory" / "candidate_metrics.csv"
    )
    policy_metrics = read_csv_rows(research_dir / "decision_policy_eval" / "policy_metrics.csv")
    label_policy_metrics = read_csv_rows(
        research_dir / "trading_aware_policy_eval" / "policy_metrics.csv"
    )
    blocked_labels = read_csv_rows(research_dir / "trading_aware_labels" / "blocked_labels.csv")
    validation = read_json_payload(
        research_dir / "trading_aware_policy_validation_audit" / "recommendation.json"
    )
    shadow = read_json_payload(
        research_dir / "shadow_adaptation_observer_plan" / "recommendation.json"
    )
    attribution = _failure_attribution(refined_metrics, policy_metrics, label_policy_metrics)
    routes = _route_decisions(refined_metrics, policy_metrics, label_policy_metrics, shadow)
    gaps = _input_gap_hypotheses(blocked_labels, validation, shadow)
    recommendation = _recommendation(gaps)
    output_files = _output_files(output_dir)
    report = {
        "summary": "M20 input failure analysis from existing research artifacts.",
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "candidate_metric_count": len(refined_metrics),
        "policy_metric_count": len(policy_metrics),
        "label_policy_metric_count": len(label_policy_metrics),
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "source_run_dir": str(source_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["analysis_json"]), report)
    Path(output_files["analysis_md"]).write_text(
        _markdown(report, attribution, gaps),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["failure_attribution_csv"]), attribution)
    write_csv_artifact(Path(output_files["route_decisions_csv"]), routes)
    write_csv_artifact(Path(output_files["input_gap_hypotheses_csv"]), gaps)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "failure_attribution": attribution,
            "route_decisions": routes,
            "input_gap_hypotheses": gaps,
            "recommendation_payload": recommendation,
        }
    )


def _failure_attribution(
    refined_metrics: Sequence[Mapping[str, str]],
    policy_metrics: Sequence[Mapping[str, str]],
    label_policy_metrics: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    return [
        _row(
            "CANDIDATE_DEFINITIONS",
            _count_negative(refined_metrics, "classification"),
            "Refined v2 candidates remain net-proxy negative.",
            "HIGH",
        ),
        _row(
            "POLICY_RULES",
            _count_negative(policy_metrics, "classification"),
            "Generic TAKE/HOLD policies did not find an adequate positive proxy path.",
            "HIGH",
        ),
        _row(
            "LABEL_HORIZON",
            _count_negative(label_policy_metrics, "classification"),
            "Label-aware policy rerun still used the same short-horizon economics.",
            "MEDIUM",
        ),
        _row(
            "SEARCH_BREADTH",
            len(policy_metrics) + len(label_policy_metrics),
            "Policy search is broad enough to require conservative interpretation.",
            "MEDIUM",
        ),
    ]


def _route_decisions(
    refined_metrics: Sequence[Mapping[str, str]],
    policy_metrics: Sequence[Mapping[str, str]],
    label_policy_metrics: Sequence[Mapping[str, str]],
    shadow: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "route": "REFINED_V2_CANDIDATES",
            "rows": len(refined_metrics),
            "decision": "FAILED_ECONOMICALLY",
        },
        {"route": "DECISION_POLICIES", "rows": len(policy_metrics), "decision": "FAILED"},
        {
            "route": "LABEL_AWARE_POLICIES",
            "rows": len(label_policy_metrics),
            "decision": "FAILED",
        },
        {
            "route": "SHADOW_OBSERVER",
            "rows": 1,
            "decision": shadow.get("recommendation", "UNKNOWN"),
        },
    ]


def _input_gap_hypotheses(
    blocked_labels: Sequence[Mapping[str, str]],
    validation: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> list[dict[str, Any]]:
    blocked = "|".join(sorted(row.get("label_name", "") for row in blocked_labels))
    return [
        {
            "gap_family": "MULTI_HORIZON_LABELS",
            "evidence": blocked,
            "hypothesis": "Current label horizon may be too narrow for policy discovery.",
            "recommended_action": "AUDIT_AND_BUILD_SAFE_MULTI_HORIZON_RESEARCH_LABELS",
        },
        {
            "gap_family": "INPUT_QUALITY",
            "evidence": str(validation.get("recommendation", "")),
            "hypothesis": "Candidate and policy rules likely need richer inputs before retesting.",
            "recommended_action": "BUILD_RESEARCH_INPUT_CATALOGUE",
        },
        {
            "gap_family": "ROUTE_STATUS",
            "evidence": str(shadow.get("recommendation", "")),
            "hypothesis": "Current route should pause unless input redesign creates new evidence.",
            "recommended_action": "PLAN_SAFE_INPUT_REDESIGN",
        },
    ]


def _count_negative(rows: Sequence[Mapping[str, str]], column: str) -> int:
    return sum(1 for row in rows if "NEGATIVE" in str(row.get(column, "")))


def _row(family: str, count: int, rationale: str, severity: str) -> dict[str, Any]:
    return {
        "failure_family": family,
        "affected_count": count,
        "severity": severity,
        "rationale": rationale,
    }


def _recommendation(gaps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    del gaps
    return {
        "recommendation": "BUILD_RESEARCH_INPUT_CATALOGUE_AND_REDESIGN_PLAN",
        "next_required_action": "BUILD_RESEARCH_INPUT_CATALOGUE_AND_REDESIGN_PLAN",
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
        "analysis_json": str(output_dir / "m20_input_failure_analysis.json"),
        "analysis_md": str(output_dir / "m20_input_failure_analysis.md"),
        "failure_attribution_csv": str(output_dir / "failure_attribution.csv"),
        "route_decisions_csv": str(output_dir / "route_decisions.csv"),
        "input_gap_hypotheses_csv": str(output_dir / "input_gap_hypotheses.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    attribution: Sequence[Mapping[str, Any]],
    gaps: Sequence[Mapping[str, Any]],
) -> str:
    counts = Counter(row["severity"] for row in attribution)
    lines = [
        "# M20 Input Failure Analysis",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Severity counts: `{dict(counts)}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Input Gaps",
    ]
    lines.extend(f"- `{row['gap_family']}`: {row['hypothesis']}" for row in gaps)
    return "\n".join(lines) + "\n"


__all__ = ["analyze_m20_input_failures"]
