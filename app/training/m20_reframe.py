"""Research-only M20 reframe artifact writer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_REFINED_FACTORY_NAME = "strategy_candidate_v2_refined_factory"
DEFAULT_OUTPUT_NAME = "m20_reframe"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)


def write_m20_reframe(
    *,
    source_run_dir: Path,
    refined_factory_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Write the M20 research reframe and policy candidate schema."""
    source_dir = Path(source_run_dir).resolve()
    vol_scaled_dir = source_dir / "research_labels" / "vol_scaled"
    factory_dir = (
        Path(refined_factory_dir).resolve()
        if refined_factory_dir
        else vol_scaled_dir / DEFAULT_REFINED_FACTORY_NAME
    )
    output_dir = vol_scaled_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = _optional_csv(factory_dir / "candidate_decisions.csv")
    summary = _decision_summary(decisions)
    recommendation = _recommendation(summary)
    schema = _policy_candidate_schema()
    output_files = _output_files(output_dir)
    report = {
        "source_run_dir": str(source_dir),
        "refined_factory_dir": str(factory_dir),
        "candidate_count": len(decisions),
        "overall_decision": "REFRAME_M20_AS_CONTEXT_AWARE_DECISION_SELECTION",
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
        "refined_factory_dir": str(factory_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["m20_reframe_json"]), report)
    Path(output_files["m20_reframe_md"]).write_text(
        _markdown(report, summary),
        encoding="utf-8",
    )
    write_json_artifact(Path(output_files["policy_candidate_schema_json"]), schema)
    write_csv_artifact(Path(output_files["decision_summary_csv"]), summary)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "policy_candidate_schema": schema,
            "decision_summary": summary,
            "recommendation_payload": recommendation,
        }
    )


def _decision_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, int] = {}
    for row in rows:
        decision = row.get("candidate_decision", "")
        grouped[decision] = grouped.get(decision, 0) + 1
    return [
        {"candidate_decision": decision, "count": count}
        for decision, count in sorted(grouped.items())
    ]


def _policy_candidate_schema() -> dict[str, Any]:
    return {
        "schema_version": "m20_policy_candidate_v1",
        "purpose": "Research-only context-aware decision-policy candidates.",
        "required_fields": [
            "policy_name",
            "candidate_source",
            "context_filters",
            "decision_action",
            "coverage",
            "mean_net_proxy",
            "max_drawdown_proxy",
            "honesty_flags",
        ],
        "forbidden_claims": [
            "runtime_ready",
            "promotable",
            "profitability_claim",
            "backtest_claim",
        ],
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _recommendation(summary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    negative = any(
        row["candidate_decision"] == "V2_STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE"
        for row in summary
    )
    recommendation = (
        "DESIGN_RESEARCH_ONLY_DECISION_POLICY_EVALUATOR"
        if negative
        else "REVIEW_M20_REFRAME_MANUALLY"
    )
    return {
        "recommendation": recommendation,
        "next_required_action": recommendation,
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
    }


def _next_actions(recommendation: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action": str(recommendation["next_required_action"]),
            "rationale": (
                "Move from raw candidate definitions to reusable decision-policy research."
            ),
        }
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "m20_reframe_json": str(output_dir / "m20_reframe.json"),
        "m20_reframe_md": str(output_dir / "m20_reframe.md"),
        "policy_candidate_schema_json": str(output_dir / "policy_candidate_schema.json"),
        "decision_summary_csv": str(output_dir / "decision_summary.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], summary: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# M20 Research Reframe",
        "",
        "M20 is reframed as a context-aware decision-selection milestone, not a "
        "specialist-forecaster-beats-incumbent-on-raw-direction milestone.",
        "",
        f"- Overall decision: `{report['overall_decision']}`",
        f"- Recommendation: `{report['recommendation']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "## Refined Candidate Decisions",
    ]
    lines.extend(f"- `{row['candidate_decision']}`: `{row['count']}`" for row in summary)
    return "\n".join(lines) + "\n"


def _optional_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
