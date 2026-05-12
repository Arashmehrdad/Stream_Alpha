"""Final research-only M20 negative-result summary artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_OUTPUT_NAME = "m20_final_research_summary"
PROJECT_ROUTE_RECOMMENDATION = (
    "KEEP_M20_PAUSED_AS_NEGATIVE_RESULT_AND_MOVE_TO_PLATFORM_MATURITY"
)
FINAL_DECISION = "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "EXISTING_ARTIFACTS_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)


def write_m20_final_research_summary(
    *,
    source_run_dir: Path,
    prediction_run_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Write the final M20 negative research-result artifact."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    prediction_dir = Path(prediction_run_dir).resolve() if prediction_run_dir else None
    research_dir = source_dir / "research_labels" / "vol_scaled"
    output_dir = research_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rollup = _research_sequence_rollup(research_dir, prediction_dir)
    negative_evidence = _negative_result_evidence(research_dir, rollup)
    data_plan = _data_upgrade_plan()
    terminal = _terminal_decision()
    project_route = _project_route_recommendation()
    recommendation = _recommendation()
    output_files = _output_files(output_dir)
    report = {
        "summary": "Final M20 negative research-result summary.",
        "source_run_dir": str(source_dir),
        "prediction_run_dir": str(prediction_dir) if prediction_dir else "",
        "final_m20_decision": FINAL_DECISION,
        "status": "RESEARCH_ONLY_NEGATIVE_RESULT",
        "project_route_recommendation": project_route["recommendation"],
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "sequence_count": len(rollup),
        "negative_evidence_count": len(negative_evidence),
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "source_run_dir": str(source_dir),
        "prediction_run_dir": str(prediction_dir) if prediction_dir else "",
        "source_artifacts": [row["artifact_path"] for row in rollup],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["summary_json"]), report)
    Path(output_files["summary_md"]).write_text(
        _markdown(report, rollup, negative_evidence, data_plan),
        encoding="utf-8",
    )
    write_json_artifact(Path(output_files["terminal_decision_json"]), terminal)
    write_csv_artifact(Path(output_files["research_sequence_rollup_csv"]), rollup)
    write_csv_artifact(Path(output_files["negative_result_evidence_csv"]), negative_evidence)
    write_csv_artifact(Path(output_files["data_upgrade_plan_csv"]), data_plan)
    write_json_artifact(Path(output_files["project_route_recommendation_json"]), project_route)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "terminal_decision": terminal,
            "research_sequence_rollup": rollup,
            "negative_result_evidence": negative_evidence,
            "data_upgrade_plan": data_plan,
            "project_route": project_route,
            "recommendation_payload": recommendation,
        }
    )


def _research_sequence_rollup(
    research_dir: Path,
    prediction_dir: Path | None,
) -> list[dict[str, Any]]:
    rows = [
        _artifact_row(
            "specialist_route",
            _prediction_artifact(prediction_dir, "cost_aware_policy_adjudication"),
            "Specialist statistical signal failed safe economic policy adjudication.",
        ),
        _artifact_row(
            "enriched_v2_candidate_route",
            research_dir / "strategy_candidate_v2_factory" / "recommendation.json",
            "Enriched v2 candidates remained economics-negative.",
        ),
        _artifact_row(
            "refined_v2_candidate_route",
            research_dir / "strategy_candidate_v2_refined_factory" / "recommendation.json",
            "Refined v2 candidates remained economics-negative.",
        ),
        _artifact_row(
            "policy_route",
            research_dir / "decision_policy_eval" / "recommendation.json",
            "Generic decision policies found no adequate positive proxy policy.",
        ),
        _artifact_row(
            "trading_aware_label_route",
            research_dir / "trading_aware_policy_eval" / "recommendation.json",
            "Trading-aware label diagnostics did not recover policy economics.",
        ),
        _artifact_row(
            "input_redesign_route",
            research_dir / "m20_input_redesign_decision" / "recommendation.json",
            "Redesigned 6/12-candle labels still produced no positive policy route.",
        ),
        _artifact_row(
            "research_dashboard",
            research_dir / "m20_research_dashboard" / "recommendation.json",
            "Dashboard records no runtime or promotion decision.",
        ),
        _artifact_row(
            "research_candidate_comparator",
            research_dir / "research_candidate_comparator" / "recommendation.json",
            "Comparator found no positive proxy research candidate.",
        ),
    ]
    return rows


def _prediction_artifact(prediction_dir: Path | None, name: str) -> Path:
    if prediction_dir is None:
        return Path("__missing_optional_prediction_run__") / name / "recommendation.json"
    return prediction_dir / "research_labels" / "vol_scaled" / name / "recommendation.json"


def _artifact_row(route: str, recommendation_path: Path, interpretation: str) -> dict[str, Any]:
    payload = _optional_json(recommendation_path)
    present = recommendation_path.exists()
    return {
        "route": route,
        "artifact_path": str(recommendation_path.parent),
        "artifact_present": present,
        "artifact_status": "PRESENT" if present else "MISSING_OPTIONAL_ARTIFACT",
        "recommendation": payload.get("recommendation", "MISSING_OPTIONAL_ARTIFACT"),
        "next_required_action": payload.get("next_required_action", ""),
        "interpretation": interpretation if present else "Optional artifact missing.",
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
    }


def _negative_result_evidence(
    research_dir: Path,
    rollup: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    decision_payload = _optional_json(
        research_dir / "m20_input_redesign_decision" / "m20_input_redesign_decision.json"
    )
    blockers = [
        "NO_POSITIVE_ADEQUATE_SAMPLE_PROXY_POLICY",
        "NOT_BACKTEST",
        "NOT_RUNTIME_READY",
        "NOT_PROMOTABLE",
        "NO_PROFIT_CLAIM",
    ]
    rows = [
        {
            "evidence": "terminal_input_redesign_decision",
            "value": decision_payload.get("final_decision", FINAL_DECISION),
            "supports_pause": True,
        },
        {
            "evidence": "completed_route_count",
            "value": sum(1 for row in rollup if row["artifact_present"]),
            "supports_pause": True,
        },
    ]
    rows.extend(
        {"evidence": "blocker", "value": blocker, "supports_pause": True}
        for blocker in blockers
    )
    return rows


def _data_upgrade_plan() -> list[dict[str, str]]:
    families = (
        ("market_microstructure_order_book", "Order book depth and imbalance features."),
        ("spread_liquidity", "Same-venue spread, liquidity, and slippage context."),
        ("trade_flow_imbalance", "Aggressor-side or trade-flow imbalance features."),
        (
            "execution_quality_measurements",
            "Same-venue fill quality and execution-cost observations.",
        ),
        (
            "stricter_untouched_segments",
            "Explicit untouched evaluation segments before reopening M20.",
        ),
        (
            "lower_turnover_event_labels",
            "Research-only event-sampled labels with lower turnover requirements.",
        ),
    )
    return [
        {
            "upgrade_family": family,
            "description": description,
            "plan_status": "PLANNING_ONLY_NOT_IMPLEMENTED",
            "runtime_effect": "NO_RUNTIME_EFFECT",
        }
        for family, description in families
    ]


def _terminal_decision() -> dict[str, Any]:
    return {
        "decision": FINAL_DECISION,
        "status": "RESEARCH_ONLY_NEGATIVE_RESULT",
        "runtime_effect": "NO_RUNTIME_EFFECT",
        "promotable": False,
        "backtest_claim": False,
        "profit_claim": False,
        "next_route_required": True,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _project_route_recommendation() -> dict[str, Any]:
    return {
        "recommendation": PROJECT_ROUTE_RECOMMENDATION,
        "rationale": (
            "M20 has preserved a complete negative research result; avoid looping on "
            "one-off candidate, policy, threshold, label, or input tweaks."
        ),
        "allowed_next_routes": [
            "platform_maturity",
            "data_upgrade_planning",
            "future_input_feasibility_only",
        ],
        "forbidden_next_routes": [
            "one_off_model_retry",
            "one_off_strategy_retry",
            "runtime_promotion",
            "profit_claim",
        ],
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": PROJECT_ROUTE_RECOMMENDATION,
        "next_required_action": PROJECT_ROUTE_RECOMMENDATION,
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
            "rationale": "Stop current M20 route and move to platform maturity or data planning.",
        }
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "summary_json": str(output_dir / "m20_final_research_summary.json"),
        "summary_md": str(output_dir / "m20_final_research_summary.md"),
        "terminal_decision_json": str(output_dir / "terminal_decision.json"),
        "research_sequence_rollup_csv": str(output_dir / "research_sequence_rollup.csv"),
        "negative_result_evidence_csv": str(output_dir / "negative_result_evidence.csv"),
        "data_upgrade_plan_csv": str(output_dir / "data_upgrade_plan.csv"),
        "project_route_recommendation_json": str(
            output_dir / "project_route_recommendation.json"
        ),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    rollup: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    data_plan: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Final Research Summary",
        "",
        "M20 is consolidated as an honest negative research result.",
        "",
        f"- Final decision: `{report['final_m20_decision']}`",
        f"- Status: `{report['status']}`",
        f"- Project route: `{report['project_route_recommendation']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "## Research Sequence",
    ]
    lines.extend(
        f"- `{row['route']}`: `{row['artifact_status']}`; {row['interpretation']}"
        for row in rollup
    )
    lines.extend(["", "## Pause Evidence"])
    lines.extend(f"- `{row['evidence']}`: `{row['value']}`" for row in evidence)
    lines.extend(["", "## Future Data Upgrade Plan"])
    lines.extend(
        f"- `{row['upgrade_family']}`: {row['description']}"
        for row in data_plan
    )
    return "\n".join(lines) + "\n"


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["write_m20_final_research_summary"]
