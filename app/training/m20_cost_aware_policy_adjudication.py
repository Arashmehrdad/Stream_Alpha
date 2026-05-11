"""Research-only adjudication for M20 cost-aware specialist policy outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_POLICY_EVALUATOR_NAME = "cost_aware_specialist_policy_evaluator"
DEFAULT_EDGE_EVALUATOR_NAME = "specialist_edge_evaluator"
DEFAULT_OUTPUT_NAME = "cost_aware_policy_adjudication"
NEXT_ACTION = "MOVE_TO_GENERIC_STRATEGY_CONDITIONED_CANDIDATE_FACTORY"
OVERALL_DECISION = "PAUSE_NEURALFORECAST_SPECIALIST_POLICY_PATH"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "EXISTING_ARTIFACTS_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
REQUIRED_POLICY_FILES = (
    "cost_aware_policy_report.json",
    "recommendation.json",
    "candidate_decisions.csv",
    "model_policy_metrics.csv",
    "topk_policy_metrics.csv",
    "economics_availability.json",
)


def write_m20_cost_aware_policy_adjudication(
    *,
    prediction_run_dir: Path,
    policy_evaluator_dir: Path | None = None,
    edge_evaluator_dir: Path | None = None,
    adjudication_output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Write cost-aware policy adjudication from existing artifacts only."""
    # pylint: disable=too-many-locals
    prediction_dir = Path(prediction_run_dir).resolve()
    vol_scaled_dir = prediction_dir / "research_labels" / "vol_scaled"
    policy_dir = (
        Path(policy_evaluator_dir).resolve()
        if policy_evaluator_dir is not None
        else vol_scaled_dir / DEFAULT_POLICY_EVALUATOR_NAME
    )
    edge_dir = (
        Path(edge_evaluator_dir).resolve()
        if edge_evaluator_dir is not None
        else vol_scaled_dir / DEFAULT_EDGE_EVALUATOR_NAME
    )
    _assert_required_policy_artifacts(policy_dir)
    output_dir = vol_scaled_dir / adjudication_output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_report = _read_json(policy_dir / "cost_aware_policy_report.json")
    policy_recommendation = _read_json(policy_dir / "recommendation.json")
    economics_availability = _read_json(policy_dir / "economics_availability.json")
    candidate_rows = _read_csv(policy_dir / "candidate_decisions.csv")
    model_policy_rows = _read_csv(policy_dir / "model_policy_metrics.csv")
    topk_rows = _read_csv(policy_dir / "topk_policy_metrics.csv")
    edge_report = _optional_json(edge_dir / "specialist_edge_report.json")
    confirmation_adjudication = _optional_json(
        vol_scaled_dir
        / "specialist_confirmation_adjudication"
        / "specialist_confirmation_adjudication.json"
    )

    model_decisions = _model_decisions(candidate_rows)
    evidence_rollup = _evidence_rollup(
        model_decisions=model_decisions,
        model_policy_rows=model_policy_rows,
        topk_rows=topk_rows,
    )
    rejected_or_watchlist = [
        row for row in model_decisions
        if row["final_decision"] != "ECONOMICALLY_PROMISING_RESEARCH_CANDIDATE"
    ]
    recommendation = _recommendation()
    output_files = _output_files(output_dir)
    adjudication = {
        "prediction_run_dir": str(prediction_dir),
        "policy_evaluator_dir": str(policy_dir),
        "edge_evaluator_dir": str(edge_dir),
        "overall_decision": OVERALL_DECISION,
        "recommendation": recommendation["recommendation"],
        "next_required_action": NEXT_ACTION,
        "economics_available": bool(economics_availability.get("economics_available")),
        "best_policy_candidate": policy_recommendation.get("best_policy_candidate", ""),
        "model_decisions": {
            row["model_name"]: row["final_decision"] for row in model_decisions
        },
        "evidence_blockers": list(policy_recommendation.get("evidence_blockers", [])),
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "prediction_run_dir": str(prediction_dir),
        "policy_evaluator_dir": str(policy_dir),
        "edge_evaluator_dir": str(edge_dir),
        "edge_report_present": bool(edge_report),
        "confirmation_adjudication_present": bool(confirmation_adjudication),
        "policy_report": policy_report,
        "policy_recommendation": policy_recommendation,
        "economics_availability": economics_availability,
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(
        Path(output_files["cost_aware_policy_adjudication_json"]),
        adjudication,
    )
    Path(output_files["cost_aware_policy_adjudication_md"]).write_text(
        _markdown(adjudication, evidence_rollup),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["model_decisions_csv"]), model_decisions)
    write_csv_artifact(Path(output_files["evidence_rollup_csv"]), evidence_rollup)
    write_csv_artifact(
        Path(output_files["rejected_or_watchlist_policies_csv"]),
        rejected_or_watchlist,
    )
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions())
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **adjudication,
            "manifest": manifest,
            "evidence_rollup": evidence_rollup,
            "rejected_or_watchlist_policies": rejected_or_watchlist,
            "recommendation_payload": recommendation,
        }
    )


def _model_decisions(candidate_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in candidate_rows:
        final_decision = _final_decision(row)
        output.append(
            {
                "model_name": row.get("model_name", ""),
                "best_policy": row.get("best_policy", ""),
                "source_policy_decision": row.get("candidate_decision", ""),
                "final_decision": final_decision,
                "selected_rows": _to_int(row.get("selected_rows")),
                "coverage": _to_float(row.get("coverage")),
                "precision": _to_float(row.get("precision")),
                "lift_vs_base": _to_float(row.get("lift_vs_base")),
                "mean_net_proxy": _to_float(row.get("mean_net_proxy")),
                "cumulative_net_proxy": _to_float(row.get("cumulative_net_proxy")),
                "runtime_status": "NO_RUNTIME_EFFECT",
                "promotion_status": "NOT_PROMOTABLE",
                "profitability_status": "NO_PROFIT_CLAIM",
                "decision_rationale": _decision_rationale(row, final_decision),
            }
        )
    return output


def _final_decision(row: Mapping[str, str]) -> str:
    source_decision = row.get("candidate_decision", "")
    mean_net = _to_float(row.get("mean_net_proxy"))
    lift = _to_float(row.get("lift_vs_base"))
    if source_decision == "ECONOMICALLY_PROMISING_RESEARCH_CANDIDATE":
        return "ECONOMICALLY_PROMISING_RESEARCH_CANDIDATE"
    if source_decision == "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE":
        return "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE"
    if mean_net < 0.0 and lift >= 1.1:
        return "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE"
    if mean_net < 0.0:
        return "WATCHLIST_ONLY_ECONOMICS_NEGATIVE"
    return "INSUFFICIENT_ECONOMIC_EVIDENCE"


def _evidence_rollup(
    *,
    model_decisions: Sequence[Mapping[str, Any]],
    model_policy_rows: Sequence[Mapping[str, str]],
    topk_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    policy_by_model = {row.get("model_name", ""): row for row in model_policy_rows}
    topk_by_model = {}
    for row in topk_rows:
        if row.get("policy_name") == "TOP_1_PERCENT":
            topk_by_model[row.get("model_name", "")] = row
    output = []
    for decision in model_decisions:
        model_name = str(decision["model_name"])
        policy_row = policy_by_model.get(model_name, {})
        topk_row = topk_by_model.get(model_name, {})
        output.append(
            {
                "model_name": model_name,
                "statistical_signal": _signal_summary(decision, topk_row),
                "economic_evidence": _economic_summary(decision),
                "final_decision": decision["final_decision"],
                "best_policy": decision["best_policy"],
                "best_lift_vs_base": decision["lift_vs_base"],
                "best_precision": decision["precision"],
                "mean_net_proxy": decision["mean_net_proxy"],
                "cumulative_net_proxy": decision["cumulative_net_proxy"],
                "best_policy_classification": policy_row.get(
                    "best_policy_classification",
                    "",
                ),
            }
        )
    return output


def _signal_summary(
    decision: Mapping[str, Any],
    topk_row: Mapping[str, str],
) -> str:
    return (
        f"best_policy_lift={float(decision['lift_vs_base']):.6f}; "
        f"best_policy_precision={float(decision['precision']):.6f}; "
        f"top1_lift={_to_float(topk_row.get('lift_vs_base')):.6f}"
    )


def _economic_summary(decision: Mapping[str, Any]) -> str:
    return (
        f"mean_net_proxy={float(decision['mean_net_proxy']):.10f}; "
        f"cumulative_net_proxy={float(decision['cumulative_net_proxy']):.10f}; "
        "not backtested; no profit claim"
    )


def _decision_rationale(row: Mapping[str, str], final_decision: str) -> str:
    return (
        f"{final_decision}: statistical lift={_to_float(row.get('lift_vs_base')):.6f}, "
        f"mean_net_proxy={_to_float(row.get('mean_net_proxy')):.10f}"
    )


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": "WATCHLIST_NEURALFORECAST_SPECIALISTS_DO_NOT_PROMOTE",
        "overall_decision": OVERALL_DECISION,
        "next_required_action": NEXT_ACTION,
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _next_actions() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action": NEXT_ACTION,
            "rationale": (
                "Stop rerunning the same specialist policy path; move to a "
                "generic strategy-conditioned candidate factory."
            ),
        },
        {
            "priority": "2",
            "action": "KEEP_NEURALFORECAST_SPECIALIST_POLICIES_RESEARCH_WATCHLIST_ONLY",
            "rationale": "Statistical signal did not survive current safe net-proxy economics.",
        },
    ]


def _assert_required_policy_artifacts(policy_dir: Path) -> None:
    missing = [
        str(policy_dir / name)
        for name in REQUIRED_POLICY_FILES
        if not (policy_dir / name).exists()
    ]
    if missing:
        raise ValueError(
            "Missing cost-aware policy evaluator artifact(s): " + "; ".join(missing)
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "cost_aware_policy_adjudication_json": str(
            output_dir / "cost_aware_policy_adjudication.json"
        ),
        "cost_aware_policy_adjudication_md": str(
            output_dir / "cost_aware_policy_adjudication.md"
        ),
        "model_decisions_csv": str(output_dir / "model_decisions.csv"),
        "evidence_rollup_csv": str(output_dir / "evidence_rollup.csv"),
        "rejected_or_watchlist_policies_csv": str(
            output_dir / "rejected_or_watchlist_policies.csv"
        ),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    adjudication: Mapping[str, Any],
    evidence_rollup: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# M20 Cost-Aware Policy Adjudication",
        "",
        f"- Overall decision: `{adjudication['overall_decision']}`",
        f"- Recommendation: `{adjudication['recommendation']}`",
        f"- Next required action: `{adjudication['next_required_action']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Evidence Rollup",
    ]
    for row in evidence_rollup:
        lines.append(
            f"- `{row['model_name']}`: `{row['final_decision']}`; "
            f"{row['statistical_signal']}; {row['economic_evidence']}"
        )
    lines.extend(
        [
            "",
            "Existing artifacts only. No training, scoring, runtime, registry, "
            "promotion, backtest, trading, or profit claim was added.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["write_m20_cost_aware_policy_adjudication"]
