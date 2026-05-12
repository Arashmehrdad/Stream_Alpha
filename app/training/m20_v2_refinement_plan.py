"""Research-only planner for refining enriched M20 v2 strategy candidates."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_FACTORY_NAME = "strategy_candidate_v2_factory"
DEFAULT_OUTPUT_NAME = "v2_refinement_plan"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
BLOCKERS = ("NO_POSITIVE_PROXY_RESEARCH_CANDIDATE", "NO_RUNTIME_OR_PROMOTION_DECISION")


def plan_m20_v2_refinement(
    *,
    source_run_dir: Path,
    candidate_factory_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Analyze enriched v2 outputs and propose reusable refinement directions."""
    source_dir = Path(source_run_dir).resolve()
    vol_scaled_dir = source_dir / "research_labels" / "vol_scaled"
    factory_dir = (
        Path(candidate_factory_dir).resolve()
        if candidate_factory_dir
        else vol_scaled_dir / DEFAULT_FACTORY_NAME
    )
    output_dir = vol_scaled_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_csv(factory_dir / "candidate_metrics.csv")
    decisions = _read_csv(factory_dir / "candidate_decisions.csv")
    failure_modes = _failure_modes(metrics, decisions)
    directions = _refinement_directions(failure_modes)
    recommendation = _recommendation()
    output_files = _output_files(output_dir)
    report = {
        "source_run_dir": str(source_dir),
        "candidate_factory_dir": str(factory_dir),
        "candidate_count": len(metrics),
        "failure_modes": [row["failure_mode"] for row in failure_modes],
        "evidence_blockers": list(BLOCKERS),
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
        "candidate_factory_dir": str(factory_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["v2_refinement_plan_json"]), report)
    Path(output_files["v2_refinement_plan_md"]).write_text(
        _markdown(report, failure_modes, directions),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["failure_modes_csv"]), failure_modes)
    write_csv_artifact(Path(output_files["refinement_directions_csv"]), directions)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "failure_mode_diagnostics": failure_modes,
            "refinement_directions": directions,
            "recommendation_payload": recommendation,
        }
    )


def _failure_modes(
    metrics: Sequence[Mapping[str, str]],
    decisions: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    negative_count = sum(
        1
        for row in decisions
        if row.get("candidate_decision") == "V2_STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE"
    )
    broad = [
        row for row in metrics
        if _to_float(row.get("coverage")) >= 0.10
    ]
    low_turnover_needed = [
        row for row in metrics
        if _to_float(row.get("selected_rows")) >= 10000
    ]
    return [
        _mode("NEGATIVE_NET_PROXY", negative_count, "All ready candidates are negative."),
        _mode("BROAD_OR_HIGH_TURNOVER_COVERAGE", len(broad), "Coverage remains too broad."),
        _mode(
            "LOWER_TURNOVER_REQUIRED",
            len(low_turnover_needed),
            "Selected-row counts remain high.",
        ),
        _mode(
            "NO_POSITIVE_PROXY_RESEARCH_CANDIDATE",
            negative_count,
            "No positive proxy candidate.",
        ),
        _mode(
            "NO_RUNTIME_OR_PROMOTION_DECISION",
            len(metrics),
            "Research-only status is preserved.",
        ),
    ]


def _mode(name: str, count: int, rationale: str) -> dict[str, Any]:
    return {
        "failure_mode": name,
        "affected_count": count,
        "severity": "HIGH" if count else "INFO",
        "rationale": rationale,
    }


def _refinement_directions(failure_modes: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    del failure_modes
    return [
        {
            "direction": "LOWER_TURNOVER_COMPOSITES",
            "description": "Require regime, ADX, momentum, volatility, and volume agreement.",
        },
        {
            "direction": "ABSTENTION_FIRST_TAIL_FILTERS",
            "description": "Exclude high volatility, high displacement, and weak-trend contexts.",
        },
        {
            "direction": "STRONGER_REGIME_CONDITIONING",
            "description": "Use research-only regime labels as gates, not as promotion evidence.",
        },
        {
            "direction": "VOLATILITY_ADJUSTED_GATING",
            "description": "Constrain realized volatility and return dispersion before selection.",
        },
    ]


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": "BUILD_REFINED_V2_STRATEGY_DEFINITIONS",
        "next_required_action": "BUILD_REFINED_V2_STRATEGY_DEFINITIONS",
        "evidence_blockers": list(BLOCKERS),
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "v2_refinement_plan_json": str(output_dir / "v2_refinement_plan.json"),
        "v2_refinement_plan_md": str(output_dir / "v2_refinement_plan.md"),
        "failure_modes_csv": str(output_dir / "failure_modes.csv"),
        "refinement_directions_csv": str(output_dir / "refinement_directions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    modes: Sequence[Mapping[str, Any]],
    directions: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 V2 Refinement Plan",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Failure Modes",
    ]
    lines.extend(f"- `{row['failure_mode']}`: {row['rationale']}" for row in modes)
    lines.extend(["", "## Reusable Refinement Directions"])
    lines.extend(f"- `{row['direction']}`: {row['description']}" for row in directions)
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
