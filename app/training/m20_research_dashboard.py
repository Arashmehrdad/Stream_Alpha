"""Static research-only M20 dashboard artifact from existing evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_OUTPUT_NAME = "m20_research_dashboard"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "EXISTING_ARTIFACTS_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
RECOMMENDATION = "PAUSE_CURRENT_M20_RESEARCH_PATHS_AND_REFINE_STRATEGY_DEFINITIONS"
NEXT_ACTION = "REFINE_STRATEGY_CANDIDATE_DEFINITIONS"


def write_m20_research_dashboard(
    *,
    source_run_dir: Path,
    prediction_run_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Write a static M20 research dashboard from existing artifacts only."""
    source_dir = Path(source_run_dir).resolve()
    prediction_dir = Path(prediction_run_dir).resolve() if prediction_run_dir else None
    vol_scaled_dir = source_dir / "research_labels" / "vol_scaled"
    output_dir = vol_scaled_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = _evidence_index(vol_scaled_dir, prediction_dir)
    timeline = _decision_timeline(evidence)
    blockers = _open_blockers(vol_scaled_dir)
    recommendation = _recommendation(blockers)
    output_files = _output_files(output_dir)
    dashboard = {
        "source_run_dir": str(source_dir),
        "prediction_run_dir": str(prediction_dir) if prediction_dir else "",
        "overall_decision": "PAUSE_CURRENT_M20_RESEARCH_PATHS",
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "evidence_count": len(evidence),
        "open_blocker_count": len(blockers),
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "source_run_dir": str(source_dir),
        "prediction_run_dir": str(prediction_dir) if prediction_dir else "",
        "honesty_flags": list(HONESTY_FLAGS),
        "source_artifacts": [row["artifact_path"] for row in evidence],
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["m20_research_dashboard_json"]), dashboard)
    Path(output_files["m20_research_dashboard_md"]).write_text(
        _markdown(dashboard, evidence, blockers),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["evidence_index_csv"]), evidence)
    write_csv_artifact(Path(output_files["decision_timeline_csv"]), timeline)
    write_csv_artifact(Path(output_files["open_blockers_csv"]), blockers)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **dashboard,
            "manifest": manifest,
            "evidence_index": evidence,
            "decision_timeline": timeline,
            "open_blockers": blockers,
            "recommendation_payload": recommendation,
        }
    )


def _evidence_index(
    vol_scaled_dir: Path,
    prediction_dir: Path | None,
) -> list[dict[str, Any]]:
    rows = [
        _artifact_row(
            "safe_economic_outcomes",
            vol_scaled_dir / "economic_outcome_artifacts" / "recommendation.json",
            "economic_outcome_artifacts",
        ),
        _artifact_row(
            "strategy_candidate_factory",
            vol_scaled_dir / "strategy_candidate_factory" / "recommendation.json",
            "strategy_candidate_factory",
        ),
        _artifact_row(
            "strategy_candidate_refinement",
            vol_scaled_dir / "strategy_candidate_refinement" / "recommendation.json",
            "strategy_candidate_refinement",
        ),
        _artifact_row(
            "strategy_slice_policy_evaluator",
            vol_scaled_dir / "strategy_slice_policy_evaluator" / "recommendation.json",
            "strategy_slice_policy_evaluator",
        ),
        _artifact_row(
            "strategy_model_factory_plan",
            vol_scaled_dir / "strategy_model_factory_plan" / "recommendation.json",
            "strategy_model_factory_plan",
        ),
        _artifact_row(
            "research_candidate_comparator",
            vol_scaled_dir / "research_candidate_comparator" / "recommendation.json",
            "research_candidate_comparator",
        ),
    ]
    if prediction_dir is not None:
        rows.append(
            _artifact_row(
                "cost_aware_policy_adjudication",
                prediction_dir
                / "research_labels"
                / "vol_scaled"
                / "cost_aware_policy_adjudication"
                / "recommendation.json",
                "cost_aware_policy_adjudication",
            )
        )
    return rows


def _artifact_row(source: str, path: Path, artifact_dir_name: str) -> dict[str, Any]:
    payload = _optional_json(path)
    recommendation = payload.get("recommendation", "MISSING")
    next_action = payload.get("next_required_action", "")
    return {
        "source": source,
        "artifact_path": str(path.parent),
        "artifact_present": path.exists(),
        "recommendation": recommendation,
        "next_required_action": next_action,
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "artifact_dir_name": artifact_dir_name,
    }


def _decision_timeline(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(evidence, start=1):
        output.append(
            {
                "step": index,
                "source": row["source"],
                "recommendation": row["recommendation"],
                "next_required_action": row["next_required_action"],
                "decision_summary": _decision_summary(row),
            }
        )
    return output


def _decision_summary(row: Mapping[str, Any]) -> str:
    if not row["artifact_present"]:
        return "artifact missing"
    recommendation = str(row["recommendation"])
    if "PAUSE" in recommendation:
        return "current path paused"
    if "REFINE" in recommendation:
        return "needs generic strategy refinement"
    if "DESIGN" in recommendation:
        return "design-only contract"
    return "research evidence recorded"


def _open_blockers(vol_scaled_dir: Path) -> list[dict[str, str]]:
    recommendation = _optional_json(
        vol_scaled_dir / "research_candidate_comparator" / "recommendation.json"
    )
    blockers = [
        {
            "blocker": blocker,
            "source": "research_candidate_comparator",
            "status": "OPEN",
        }
        for blocker in recommendation.get("evidence_blockers", [])
    ]
    if not blockers:
        blockers.append(
            {
                "blocker": "NO_POSITIVE_PROXY_RESEARCH_CANDIDATE",
                "source": "m20_research_dashboard",
                "status": "OPEN",
            }
        )
    blockers.append(
        {
            "blocker": "NO_RUNTIME_OR_PROMOTION_DECISION",
            "source": "m20_research_dashboard",
            "status": "OPEN",
        }
    )
    return _dedupe_blockers(blockers)


def _dedupe_blockers(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    output = []
    seen = set()
    for row in rows:
        blocker = row["blocker"]
        if blocker in seen:
            continue
        seen.add(blocker)
        output.append(dict(row))
    return output


def _recommendation(blockers: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "recommendation": RECOMMENDATION,
        "next_required_action": NEXT_ACTION,
        "evidence_blockers": [row["blocker"] for row in blockers],
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
            "rationale": "Current M20 evidence does not support model execution or promotion.",
        },
        {
            "priority": "2",
            "action": "KEEP_M20_RESEARCH_DASHBOARD_STATIC",
            "rationale": "No runtime, registry, promotion, backtest, trading, or profit claim.",
        },
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "m20_research_dashboard_json": str(output_dir / "m20_research_dashboard.json"),
        "m20_research_dashboard_md": str(output_dir / "m20_research_dashboard.md"),
        "evidence_index_csv": str(output_dir / "evidence_index.csv"),
        "decision_timeline_csv": str(output_dir / "decision_timeline.csv"),
        "open_blockers_csv": str(output_dir / "open_blockers.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    dashboard: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    blockers: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Research Dashboard",
        "",
        f"- Overall decision: `{dashboard['overall_decision']}`",
        f"- Recommendation: `{dashboard['recommendation']}`",
        f"- Next required action: `{dashboard['next_required_action']}`",
        "- Status: `RESEARCH_ONLY`, `EXISTING_ARTIFACTS_ONLY`, "
        "`NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, `NOT_RUNTIME_READY`, "
        "`NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Evidence",
    ]
    for row in evidence:
        lines.append(
            f"- `{row['source']}`: `{row['recommendation']}` "
            f"present=`{row['artifact_present']}`"
        )
    lines.append("")
    lines.append("## Open Blockers")
    for row in blockers:
        lines.append(f"- `{row['blocker']}` from `{row['source']}`")
    lines.append("")
    return "\n".join(lines)


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["write_m20_research_dashboard"]
