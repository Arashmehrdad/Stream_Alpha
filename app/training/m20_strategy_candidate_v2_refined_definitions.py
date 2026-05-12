"""Research-only refined M20 v2 strategy definition generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_OUTPUT_NAME = "strategy_candidate_v2_refined_definitions"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "DEFINITION_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
FORBIDDEN_FEATURE_NAMES = (
    "fee_exceedance_label",
    "triple_barrier_label",
    "future_return",
    "gross_value_proxy",
    "net_value_proxy",
    "economic_outcome",
)


def build_m20_strategy_candidate_v2_refined_definitions(
    *,
    source_run_dir: Path,
    refinement_plan_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Write reusable refined v2 candidate definitions with predicate specs."""
    source_dir = Path(source_run_dir).resolve()
    vol_scaled_dir = source_dir / "research_labels" / "vol_scaled"
    plan_dir = (
        Path(refinement_plan_dir).resolve()
        if refinement_plan_dir
        else vol_scaled_dir / "v2_refinement_plan"
    )
    output_dir = vol_scaled_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = _definitions()
    blocked = _blocked_definitions(definitions)
    recommendation = _recommendation(blocked)
    output_files = _output_files(output_dir)
    report = {
        "source_run_dir": str(source_dir),
        "refinement_plan_dir": str(plan_dir),
        "definition_count": len(definitions),
        "blocked_definition_count": len(blocked),
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
        "refinement_plan_dir": str(plan_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["refined_definitions_report_json"]), report)
    Path(output_files["refined_definitions_report_md"]).write_text(
        _markdown(report, definitions),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["candidate_definition_specs_csv"]), definitions)
    write_csv_artifact(Path(output_files["blocked_definitions_csv"]), blocked)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "candidate_definition_specs": definitions,
            "blocked_definitions": blocked,
            "recommendation_payload": recommendation,
        }
    )


def _definitions() -> list[dict[str, Any]]:
    return [
        _definition(
            "refined_composite",
            "refined_regime_adx_momentum_low_turnover",
            ("regime_label", "adx_14", "momentum_3", "realized_vol_12", "volume_zscore_12"),
            {
                "all": [
                    {"field": "regime_label", "eq": "TREND_UP"},
                    {"field": "adx_14", "gte": 25.0},
                    {"field": "momentum_3", "gt": 0.0},
                    {"field": "realized_vol_12", "between": [0.004, 0.025]},
                    {"field": "volume_zscore_12", "gt": 0.5},
                ]
            },
            "regime, trend-strength, momentum, volatility, and volume agreement",
        ),
        _definition(
            "refined_tail_risk_aware",
            "refined_non_extreme_trend_context",
            ("regime_label", "adx_14", "momentum_3", "realized_vol_12", "close_zscore_12"),
            {
                "all": [
                    {"field": "regime_label", "eq": "TREND_UP"},
                    {"field": "adx_14", "gte": 20.0},
                    {"field": "momentum_3", "gt": 0.0},
                    {"field": "realized_vol_12", "lte": 0.02},
                    {"field": "close_zscore_12", "abs_lte": 1.0},
                ]
            },
            "abstention-first exclusion of high-displacement and high-volatility rows",
        ),
        _definition(
            "refined_volatility_adjusted",
            "refined_volatility_adjusted_macd_adx",
            ("macd_line_12_26", "adx_14", "return_std_12", "realized_vol_12"),
            {
                "all": [
                    {"field": "macd_line_12_26", "gt": 0.0},
                    {"field": "adx_14", "gte": 25.0},
                    {"field": "return_std_12", "lte": 0.01},
                    {"field": "realized_vol_12", "between": [0.004, 0.025]},
                ]
            },
            "volatility-adjusted trend filter with ADX confirmation",
        ),
        _definition(
            "refined_abstention",
            "refined_hold_avoid_extreme_context",
            ("realized_vol_12", "return_std_12", "close_zscore_12", "volume_zscore_12"),
            {
                "any": [
                    {"field": "realized_vol_12", "gte": 0.035},
                    {"field": "return_std_12", "gte": 0.015},
                    {"field": "close_zscore_12", "abs_gte": 2.0},
                    {"field": "volume_zscore_12", "abs_gte": 2.5},
                ]
            },
            "research-only abstention context for later policy evaluation",
        ),
    ]


def _definition(
    family: str,
    name: str,
    required_features: Sequence[str],
    predicate: Mapping[str, Any],
    rationale: str,
) -> dict[str, Any]:
    status = "READY_FOR_V2_FACTORY"
    missing = ""
    if _has_forbidden_features(required_features):
        status = "BLOCKED_FORBIDDEN_FEATURE"
        missing = "FORBIDDEN_LEAKAGE_FEATURE"
    return {
        "redesign_family": family,
        "candidate_name": name,
        "candidate_version": "v2_refined",
        "required_features": "|".join(required_features),
        "predicate_spec_json": json.dumps(predicate, sort_keys=True),
        "rationale": rationale,
        "uses_economic_outcome_as_feature": False,
        "evaluates_candidate_now": False,
        "missing_features": missing,
        "definition_status": status,
    }


def _has_forbidden_features(features: Sequence[str]) -> bool:
    return any(feature.lower() in FORBIDDEN_FEATURE_NAMES for feature in features)


def _blocked_definitions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row["definition_status"] != "READY_FOR_V2_FACTORY"]


def _recommendation(blocked: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recommendation = (
        "RUN_REFINED_V2_CANDIDATE_FACTORY"
        if not blocked
        else "RESOLVE_REFINED_DEFINITION_BLOCKERS"
    )
    return {
        "recommendation": recommendation,
        "next_required_action": recommendation,
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "refined_definitions_report_json": str(output_dir / "refined_definitions_report.json"),
        "refined_definitions_report_md": str(output_dir / "refined_definitions_report.md"),
        "candidate_definition_specs_csv": str(output_dir / "candidate_definition_specs.csv"),
        "blocked_definitions_csv": str(output_dir / "blocked_definitions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], definitions: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# M20 Refined V2 Strategy Definitions",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Definition count: `{report['definition_count']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, `NO_PROFIT_CLAIM`",
        "",
        "## Definitions",
    ]
    lines.extend(f"- `{row['candidate_name']}`" for row in definitions)
    return "\n".join(lines) + "\n"
