"""Generic research-only comparator for M20 model/strategy/policy candidates."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_OUTPUT_NAME = "research_candidate_comparator"
DEFAULT_STRATEGY_FACTORY_NAME = "strategy_candidate_factory"
DEFAULT_STRATEGY_REFINEMENT_NAME = "strategy_candidate_refinement"
DEFAULT_STRATEGY_POLICY_NAME = "strategy_slice_policy_evaluator"
DEFAULT_MODEL_FACTORY_PLAN_NAME = "strategy_model_factory_plan"
DEFAULT_SPECIALIST_ADJUDICATION_NAME = "cost_aware_policy_adjudication"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "EXISTING_ARTIFACTS_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
RECOMMEND_PAUSE = "PAUSE_CURRENT_M20_CANDIDATE_PATHS"
NEXT_REFINE = "REFINE_STRATEGY_CANDIDATE_DEFINITIONS"


def compare_m20_research_candidates(
    *,
    source_run_dir: Path,
    prediction_run_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Compare current known M20 research candidates from existing artifacts."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    prediction_dir = Path(prediction_run_dir).resolve() if prediction_run_dir else None
    vol_scaled_dir = source_dir / "research_labels" / "vol_scaled"
    output_dir = vol_scaled_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_rows = _strategy_scorecard(vol_scaled_dir)
    policy_rows = _strategy_policy_scorecard(vol_scaled_dir)
    model_plan_rows = _model_plan_scorecard(vol_scaled_dir)
    specialist_rows = _specialist_scorecard(prediction_dir)
    scorecard = strategy_rows + policy_rows + model_plan_rows + specialist_rows
    economics = _economics_comparison(scorecard)
    stability = _stability_comparison(scorecard)
    decisions = _candidate_decisions(scorecard)
    blockers = _blockers(scorecard, prediction_dir)
    recommendation = _recommendation(blockers)
    output_files = _output_files(output_dir)
    report = {
        "summary": "Generic M20 research candidate comparator.",
        "source_run_dir": str(source_dir),
        "prediction_run_dir": str(prediction_dir) if prediction_dir else "",
        "candidate_count": len(scorecard),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "source_run_dir": str(source_dir),
        "prediction_run_dir": str(prediction_dir) if prediction_dir else "",
        "supported_artifacts": [
            DEFAULT_STRATEGY_FACTORY_NAME,
            DEFAULT_STRATEGY_REFINEMENT_NAME,
            DEFAULT_STRATEGY_POLICY_NAME,
            DEFAULT_MODEL_FACTORY_PLAN_NAME,
            DEFAULT_SPECIALIST_ADJUDICATION_NAME,
        ],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(
        Path(output_files["research_candidate_comparison_report_json"]),
        report,
    )
    Path(output_files["research_candidate_comparison_report_md"]).write_text(
        _markdown(report, decisions, blockers),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["candidate_scorecard_csv"]), scorecard)
    write_csv_artifact(Path(output_files["economics_comparison_csv"]), economics)
    write_csv_artifact(Path(output_files["stability_comparison_csv"]), stability)
    write_csv_artifact(Path(output_files["candidate_decisions_csv"]), decisions)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "candidate_scorecard": scorecard,
            "economics_comparison": economics,
            "stability_comparison": stability,
            "candidate_decisions": decisions,
            "blockers": blockers,
            "recommendation_payload": recommendation,
        }
    )


def _strategy_scorecard(vol_scaled_dir: Path) -> list[dict[str, Any]]:
    rows = _optional_csv(
        vol_scaled_dir / DEFAULT_STRATEGY_FACTORY_NAME / "candidate_decisions.csv"
    )
    return [
        _scorecard_row(
            candidate_type="strategy_candidate",
            candidate_id=f"{row.get('strategy_family', '')}:{row.get('candidate_name', '')}",
            source_decision=row.get("candidate_decision", ""),
            mean_net_proxy=row.get("mean_net_proxy", ""),
            cumulative_net_proxy=row.get("cumulative_net_proxy", ""),
            selected_rows=row.get("selected_rows", ""),
            signal_metric=row.get("coverage", ""),
            stability_metric="",
        )
        for row in rows
    ]


def _strategy_policy_scorecard(vol_scaled_dir: Path) -> list[dict[str, Any]]:
    rows = _optional_csv(
        vol_scaled_dir / DEFAULT_STRATEGY_POLICY_NAME / "candidate_decisions.csv"
    )
    return [
        _scorecard_row(
            candidate_type="strategy_slice_policy",
            candidate_id=row.get("policy_name", ""),
            source_decision=row.get("candidate_decision", ""),
            mean_net_proxy=row.get("mean_net_proxy", ""),
            cumulative_net_proxy="",
            selected_rows=row.get("selected_rows", ""),
            signal_metric=row.get("tail_loss_rate", ""),
            stability_metric=row.get("tail_loss_rate", ""),
        )
        for row in rows
    ]


def _model_plan_scorecard(vol_scaled_dir: Path) -> list[dict[str, Any]]:
    rows = _optional_csv(
        vol_scaled_dir / DEFAULT_MODEL_FACTORY_PLAN_NAME / "candidate_inputs.csv"
    )
    return [
        _scorecard_row(
            candidate_type="strategy_model_factory_input",
            candidate_id=f"{row.get('strategy_family', '')}:{row.get('candidate_name', '')}",
            source_decision=row.get("model_factory_status", ""),
            mean_net_proxy=row.get("best_slice_mean_net_proxy", ""),
            cumulative_net_proxy="",
            selected_rows="",
            signal_metric=row.get("best_slice", ""),
            stability_metric=row.get("slice_policy_decision", ""),
        )
        for row in rows
    ]


def _specialist_scorecard(prediction_dir: Path | None) -> list[dict[str, Any]]:
    if prediction_dir is None:
        return []
    rows = _optional_csv(
        prediction_dir
        / "research_labels"
        / "vol_scaled"
        / DEFAULT_SPECIALIST_ADJUDICATION_NAME
        / "model_decisions.csv"
    )
    return [
        _scorecard_row(
            candidate_type="specialist_policy",
            candidate_id=row.get("model_name", ""),
            source_decision=row.get("final_decision", ""),
            mean_net_proxy=row.get("mean_net_proxy", ""),
            cumulative_net_proxy=row.get("cumulative_net_proxy", ""),
            selected_rows=row.get("selected_rows", ""),
            signal_metric=row.get("lift_vs_base", ""),
            stability_metric=row.get("best_policy", ""),
        )
        for row in rows
    ]


def _scorecard_row(
    *,
    candidate_type: str,
    candidate_id: str,
    source_decision: str,
    mean_net_proxy: Any,
    cumulative_net_proxy: Any,
    selected_rows: Any,
    signal_metric: Any,
    stability_metric: Any,
) -> dict[str, Any]:
    # pylint: disable=too-many-arguments
    final_decision = _final_decision(source_decision, mean_net_proxy)
    return {
        "candidate_type": candidate_type,
        "candidate_id": candidate_id,
        "source_decision": source_decision,
        "final_decision": final_decision,
        "selected_rows": selected_rows,
        "signal_metric": signal_metric,
        "mean_net_proxy": _empty_or_float(mean_net_proxy),
        "cumulative_net_proxy": _empty_or_float(cumulative_net_proxy),
        "stability_metric": stability_metric,
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
    }


def _final_decision(source_decision: str, mean_net_proxy: Any) -> str:
    if "LOW_SAMPLE" in source_decision:
        return "WATCHLIST_LOW_SAMPLE_RESEARCH_ONLY"
    if "BLOCKED" in source_decision:
        return "BLOCKED_RESEARCH_ONLY"
    if "NEGATIVE" in source_decision or _is_negative(mean_net_proxy):
        return "ECONOMICS_NEGATIVE_RESEARCH_ONLY"
    if _is_positive(mean_net_proxy):
        return "WATCHLIST_POSITIVE_PROXY_RESEARCH_ONLY"
    return "INSUFFICIENT_EVIDENCE_RESEARCH_ONLY"


def _economics_comparison(
    scorecard: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for candidate_type in sorted({row["candidate_type"] for row in scorecard}):
        rows = [row for row in scorecard if row["candidate_type"] == candidate_type]
        numeric_net = [
            float(row["mean_net_proxy"]) for row in rows
            if row.get("mean_net_proxy") not in ("", None)
        ]
        output.append(
            {
                "candidate_type": candidate_type,
                "candidate_count": len(rows),
                "negative_count": sum(
                    1 for row in rows
                    if row["final_decision"] == "ECONOMICS_NEGATIVE_RESEARCH_ONLY"
                ),
                "positive_proxy_count": sum(
                    1 for row in rows
                    if row["final_decision"] == "WATCHLIST_POSITIVE_PROXY_RESEARCH_ONLY"
                ),
                "mean_net_proxy": _mean(numeric_net) if numeric_net else "",
            }
        )
    return output


def _stability_comparison(
    scorecard: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_type": row["candidate_type"],
            "candidate_id": row["candidate_id"],
            "stability_metric": row["stability_metric"],
            "final_decision": row["final_decision"],
        }
        for row in scorecard
    ]


def _candidate_decisions(
    scorecard: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_type": row["candidate_type"],
            "candidate_id": row["candidate_id"],
            "candidate_decision": row["final_decision"],
            "mean_net_proxy": row["mean_net_proxy"],
            "runtime_status": "NO_RUNTIME_EFFECT",
            "promotion_status": "NOT_PROMOTABLE",
            "profitability_status": "NO_PROFIT_CLAIM",
        }
        for row in scorecard
    ]


def _blockers(
    scorecard: Sequence[Mapping[str, Any]],
    prediction_dir: Path | None,
) -> list[dict[str, str]]:
    blockers = []
    if not scorecard:
        blockers.append(
            {
                "blocker": "NO_SUPPORTED_RESEARCH_CANDIDATE_ARTIFACTS",
                "detail": "No known generic M20 candidate artifacts were found.",
            }
        )
    if prediction_dir is None:
        blockers.append(
            {
                "blocker": "SPECIALIST_POLICY_ADJUDICATION_NOT_LINKED",
                "detail": "Prediction run dir was not supplied.",
            }
        )
    if all(row["final_decision"] != "WATCHLIST_POSITIVE_PROXY_RESEARCH_ONLY" for row in scorecard):
        blockers.append(
            {
                "blocker": "NO_POSITIVE_PROXY_RESEARCH_CANDIDATE",
                "detail": "Compared candidates are negative, blocked, low-sample, or insufficient.",
            }
        )
    blockers.append(
        {
            "blocker": "NO_RUNTIME_OR_PROMOTION_DECISION",
            "detail": "Comparator is research-only and cannot promote candidates.",
        }
    )
    return blockers


def _recommendation(blockers: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "recommendation": RECOMMEND_PAUSE,
        "next_required_action": NEXT_REFINE,
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
            "rationale": "Current compared candidate paths do not justify model execution.",
        },
        {
            "priority": "2",
            "action": "KEEP_COMPARATOR_RESEARCH_ONLY",
            "rationale": "No runtime, registry, promotion, backtest, trading, or profit claim.",
        },
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "research_candidate_comparison_report_json": str(
            output_dir / "research_candidate_comparison_report.json"
        ),
        "research_candidate_comparison_report_md": str(
            output_dir / "research_candidate_comparison_report.md"
        ),
        "candidate_scorecard_csv": str(output_dir / "candidate_scorecard.csv"),
        "economics_comparison_csv": str(output_dir / "economics_comparison.csv"),
        "stability_comparison_csv": str(output_dir / "stability_comparison.csv"),
        "candidate_decisions_csv": str(output_dir / "candidate_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    blockers: Sequence[Mapping[str, str]],
) -> str:
    counts: dict[str, int] = {}
    for row in decisions:
        decision = str(row["candidate_decision"])
        counts[decision] = counts.get(decision, 0) + 1
    lines = [
        "# M20 Research Candidate Comparator",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Candidate count: `{report['candidate_count']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Decision Counts",
    ]
    for decision, count in sorted(counts.items()):
        lines.append(f"- `{decision}`: `{count}`")
    lines.append("")
    lines.append("## Blockers")
    for row in blockers:
        lines.append(f"- `{row['blocker']}`: {row['detail']}")
    lines.append("")
    return "\n".join(lines)


def _optional_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _empty_or_float(value: Any) -> float | str:
    if value in ("", None):
        return ""
    return _to_float(value)


def _is_negative(value: Any) -> bool:
    return value not in ("", None) and _to_float(value) < 0.0


def _is_positive(value: Any) -> bool:
    return value not in ("", None) and _to_float(value) > 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["compare_m20_research_candidates"]
