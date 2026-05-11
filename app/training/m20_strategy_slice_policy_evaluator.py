"""Generic research-only policy evaluator for M20 strategy candidate slices."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_REFINEMENT_NAME = "strategy_candidate_refinement"
DEFAULT_OUTPUT_NAME = "strategy_slice_policy_evaluator"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
RECOMMEND_MODEL_FACTORY = "DESIGN_GENERIC_STRATEGY_MODEL_FACTORY"
RECOMMEND_REFINE = "REFINE_STRATEGY_CANDIDATE_DEFINITIONS"
NEXT_MODEL_FACTORY = "PLAN_GENERIC_STRATEGY_CONDITIONED_MODEL_FACTORY"
NEXT_REFINE = "REFINE_STRATEGY_CANDIDATE_DEFINITIONS"


def analyze_m20_strategy_slice_policy(
    *,
    source_run_dir: Path,
    refinement_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
    min_policy_rows: int = 100,
    max_tail_loss_rate: float = 0.75,
) -> dict[str, Any]:
    """Evaluate generic strategy slice policies from refinement artifacts."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    vol_scaled_dir = source_dir / "research_labels" / "vol_scaled"
    input_dir = (
        Path(refinement_dir).resolve()
        if refinement_dir is not None
        else vol_scaled_dir / DEFAULT_REFINEMENT_NAME
    )
    _assert_refinement_artifacts(input_dir)
    output_dir = vol_scaled_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_rows = _read_csv(input_dir / "slice_diagnostics.csv")
    policy_candidates = _policy_candidates(slice_rows, min_policy_rows)
    policy_metrics = [
        _policy_metric(row, max_tail_loss_rate) for row in policy_candidates
    ]
    decisions = _candidate_decisions(policy_metrics)
    recommendation = _recommendation(decisions)
    output_files = _output_files(output_dir)
    report = {
        "summary": "Generic M20 strategy slice policy evaluator.",
        "source_run_dir": str(source_dir),
        "refinement_dir": str(input_dir),
        "policy_count": len(policy_metrics),
        "min_policy_rows": min_policy_rows,
        "max_tail_loss_rate": max_tail_loss_rate,
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
        "refinement_dir": str(input_dir),
        "input_slice_rows": len(slice_rows),
        "min_policy_rows": min_policy_rows,
        "max_tail_loss_rate": max_tail_loss_rate,
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["strategy_slice_policy_report_json"]), report)
    Path(output_files["strategy_slice_policy_report_md"]).write_text(
        _markdown(report, decisions),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["policy_candidates_csv"]), policy_candidates)
    write_csv_artifact(Path(output_files["policy_metrics_csv"]), policy_metrics)
    write_csv_artifact(
        Path(output_files["by_symbol_csv"]),
        _filter_family(policy_metrics, "symbol"),
    )
    write_csv_artifact(Path(output_files["by_time_csv"]), _time_rows(policy_metrics))
    write_csv_artifact(Path(output_files["tail_risk_csv"]), _tail_risk(policy_metrics))
    write_csv_artifact(Path(output_files["candidate_decisions_csv"]), decisions)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "policy_candidates": policy_candidates,
            "policy_metrics": policy_metrics,
            "candidate_decisions": decisions,
            "recommendation_payload": recommendation,
        }
    )


def _assert_refinement_artifacts(refinement_dir: Path) -> None:
    missing = [
        path.name for path in (
            refinement_dir / "slice_diagnostics.csv",
            refinement_dir / "refined_candidate_metrics.csv",
            refinement_dir / "manifest.json",
        )
        if not path.exists()
    ]
    if missing:
        raise ValueError(
            "Missing strategy candidate refinement artifacts: "
            + ", ".join(missing)
        )


def _policy_candidates(
    slice_rows: Sequence[Mapping[str, str]],
    min_policy_rows: int,
) -> list[dict[str, Any]]:
    output = []
    for row in slice_rows:
        selected_rows = _to_int(row.get("selected_rows"))
        output.append(
            {
                "policy_name": _policy_name(row),
                "policy_family": _policy_family(row.get("slice_family", "")),
                "strategy_family": row.get("strategy_family", ""),
                "candidate_name": row.get("candidate_name", ""),
                "slice_family": row.get("slice_family", ""),
                "slice_value": row.get("slice_value", ""),
                "selected_rows": selected_rows,
                "min_policy_rows": min_policy_rows,
                "sample_gate_passed": selected_rows >= min_policy_rows,
                "source_slice_decision": row.get("slice_decision", ""),
                "mean_net_proxy": row.get("mean_net_proxy", ""),
                "cumulative_net_proxy": row.get("cumulative_net_proxy", ""),
                "max_drawdown_proxy": row.get("max_drawdown_proxy", ""),
                "tail_loss_rate": row.get("tail_loss_rate", ""),
                "win_rate_proxy": row.get("win_rate_proxy", ""),
                "selected_positive_rate": row.get("selected_positive_rate", ""),
            }
        )
    return sorted(output, key=lambda row: str(row["policy_name"]))


def _policy_metric(
    row: Mapping[str, Any],
    max_tail_loss_rate: float,
) -> dict[str, Any]:
    mean_net = _empty_or_float(row.get("mean_net_proxy"))
    tail_loss = _empty_or_float(row.get("tail_loss_rate"))
    classification = _policy_classification(
        sample_gate_passed=bool(row["sample_gate_passed"]),
        mean_net=mean_net,
        tail_loss_rate=tail_loss,
        max_tail_loss_rate=max_tail_loss_rate,
    )
    return {
        **row,
        "mean_net_proxy": mean_net,
        "cumulative_net_proxy": _empty_or_float(row.get("cumulative_net_proxy")),
        "max_drawdown_proxy": _empty_or_float(row.get("max_drawdown_proxy")),
        "tail_loss_rate": tail_loss,
        "win_rate_proxy": _empty_or_float(row.get("win_rate_proxy")),
        "selected_positive_rate": _empty_or_float(row.get("selected_positive_rate")),
        "policy_classification": classification,
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
    }


def _policy_classification(
    *,
    sample_gate_passed: bool,
    mean_net: float | str,
    tail_loss_rate: float | str,
    max_tail_loss_rate: float,
) -> str:
    if not sample_gate_passed:
        return "SLICE_POLICY_LOW_SAMPLE"
    if mean_net == "":
        return "SLICE_POLICY_ECONOMICS_UNKNOWN"
    if float(mean_net) > 0.0 and (
        tail_loss_rate == "" or float(tail_loss_rate) <= max_tail_loss_rate
    ):
        return "SLICE_POLICY_RESEARCH_WATCHLIST_POSITIVE_NET_PROXY"
    if float(mean_net) > 0.0:
        return "SLICE_POLICY_POSITIVE_NET_WITH_TAIL_RISK"
    return "SLICE_POLICY_ECONOMICS_NEGATIVE"


def _candidate_decisions(
    policy_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in policy_metrics:
        output.append(
            {
                "policy_name": row["policy_name"],
                "strategy_family": row["strategy_family"],
                "candidate_name": row["candidate_name"],
                "slice_family": row["slice_family"],
                "slice_value": row["slice_value"],
                "candidate_decision": row["policy_classification"],
                "selected_rows": row["selected_rows"],
                "mean_net_proxy": row["mean_net_proxy"],
                "tail_loss_rate": row["tail_loss_rate"],
                "runtime_status": "NO_RUNTIME_EFFECT",
                "promotion_status": "NOT_PROMOTABLE",
                "profitability_status": "NO_PROFIT_CLAIM",
            }
        )
    return output


def _recommendation(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    has_watchlist = any(
        row["candidate_decision"]
        == "SLICE_POLICY_RESEARCH_WATCHLIST_POSITIVE_NET_PROXY"
        for row in decisions
    )
    recommendation = RECOMMEND_MODEL_FACTORY if has_watchlist else RECOMMEND_REFINE
    next_action = NEXT_MODEL_FACTORY if has_watchlist else NEXT_REFINE
    return {
        "recommendation": recommendation,
        "next_required_action": next_action,
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
            "rationale": "Continue through generic research tooling only.",
        },
        {
            "priority": "2",
            "action": "KEEP_SLICE_POLICIES_RESEARCH_ONLY",
            "rationale": "No runtime, registry, promotion, backtest, trading, or profit claim.",
        },
    ]


def _tail_risk(policy_metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy_name": row["policy_name"],
            "selected_rows": row["selected_rows"],
            "mean_net_proxy": row["mean_net_proxy"],
            "max_drawdown_proxy": row["max_drawdown_proxy"],
            "tail_loss_rate": row["tail_loss_rate"],
            "policy_classification": row["policy_classification"],
        }
        for row in policy_metrics
    ]


def _filter_family(
    policy_metrics: Sequence[Mapping[str, Any]],
    slice_family: str,
) -> list[dict[str, Any]]:
    return [row for row in policy_metrics if row["slice_family"] == slice_family]


def _time_rows(policy_metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in policy_metrics
        if row["slice_family"] in ("month", "quarter")
    ]


def _policy_name(row: Mapping[str, str]) -> str:
    return (
        f"{row.get('strategy_family', '')}:{row.get('candidate_name', '')}:"
        f"{row.get('slice_family', '')}={row.get('slice_value', '')}"
    )


def _policy_family(slice_family: str) -> str:
    if slice_family == "symbol":
        return "SYMBOL_FILTER"
    if slice_family in ("month", "quarter"):
        return "TIME_FILTER"
    if slice_family in ("volatility_bucket", "range_bucket"):
        return "REGIME_FILTER"
    if slice_family == "volume_bucket":
        return "VOLUME_FILTER"
    return "GENERIC_SLICE_FILTER"


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "strategy_slice_policy_report_json": str(
            output_dir / "strategy_slice_policy_report.json"
        ),
        "strategy_slice_policy_report_md": str(
            output_dir / "strategy_slice_policy_report.md"
        ),
        "policy_metrics_csv": str(output_dir / "policy_metrics.csv"),
        "policy_candidates_csv": str(output_dir / "policy_candidates.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "tail_risk_csv": str(output_dir / "tail_risk.csv"),
        "candidate_decisions_csv": str(output_dir / "candidate_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> str:
    counts: dict[str, int] = {}
    for row in decisions:
        decision = str(row["candidate_decision"])
        counts[decision] = counts.get(decision, 0) + 1
    lines = [
        "# M20 Strategy Slice Policy Evaluator",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Policy count: `{report['policy_count']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Decision Counts",
    ]
    for decision, count in sorted(counts.items()):
        lines.append(f"- `{decision}`: `{count}`")
    lines.extend(
        [
            "",
            "Slice policies are research diagnostics only and are not profitability, "
            "runtime, promotion, trading, or backtest evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _empty_or_float(value: Any) -> float | str:
    if value in ("", None):
        return ""
    return _to_float(value)


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


__all__ = ["analyze_m20_strategy_slice_policy"]
