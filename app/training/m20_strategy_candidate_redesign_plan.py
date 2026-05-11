"""Research-only M20 strategy candidate redesign plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_OUTPUT_NAME = "strategy_candidate_redesign_plan"
FEATURE_COLUMNS_FILE = "training_frame/m20_training_frame_feature_columns.json"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "EXISTING_ARTIFACTS_ONLY",
    "DESIGN_ONLY",
    "NO_CANDIDATE_EVALUATION",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
SAFE_FEATURE_FAMILIES = {
    "price": ("open_price", "high_price", "low_price", "close_price", "vwap"),
    "volume": ("volume", "volume_mean_12", "volume_std_12", "volume_zscore_12"),
    "returns": ("log_return_1", "log_return_3", "momentum_3"),
    "volatility": ("return_std_12", "realized_vol_12"),
    "oscillator": ("rsi_14",),
    "momentum": ("macd_line_12_26", "momentum_3"),
    "zscore": ("close_zscore_12", "volume_zscore_12"),
    "lags": ("lag_log_return_1", "lag_log_return_2", "lag_log_return_3"),
}
MISSING_FEATURE_FAMILIES = {
    "regime": ("regime_label", "market_regime"),
    "trend_strength": ("adx_14", "trend_strength"),
    "atr": ("atr_14", "true_range"),
    "bollinger": ("bollinger_upper", "bollinger_lower", "bollinger_width"),
    "moving_average": ("ema_12", "ema_26", "sma_20", "ema_slope"),
    "time_session": ("hour_of_day", "day_of_week", "session_label"),
    "market_microstructure": ("spread", "order_book_imbalance", "liquidity_depth"),
    "funding": ("funding_rate",),
}


def plan_m20_strategy_candidate_redesign(
    *,
    source_run_dir: Path,
    prediction_run_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Write v2 strategy candidate redesign specs from existing artifacts only."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    prediction_dir = Path(prediction_run_dir).resolve() if prediction_run_dir else None
    vol_scaled_dir = source_dir / "research_labels" / "vol_scaled"
    output_dir = vol_scaled_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = _feature_columns(source_dir / FEATURE_COLUMNS_FILE)
    available_families = _available_feature_families(feature_columns)
    missing_families = _missing_feature_families(feature_columns)
    failure_modes = _failure_modes(vol_scaled_dir)
    specs = _candidate_specs(feature_columns)
    blocked = [row for row in specs if row["definition_status"] == "BLOCKED_MISSING_FEATURES"]
    runnable = [row for row in specs if row["definition_status"] == "READY_FOR_V2_FACTORY"]
    recommendation = _recommendation(runnable)
    output_files = _output_files(output_dir)
    plan = {
        "source_run_dir": str(source_dir),
        "prediction_run_dir": str(prediction_dir) if prediction_dir else "",
        "candidate_definition_count": len(specs),
        "ready_definition_count": len(runnable),
        "blocked_definition_count": len(blocked),
        "failure_modes": failure_modes,
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
        "prediction_run_dir": str(prediction_dir) if prediction_dir else "",
        "source_artifacts": _source_artifacts(vol_scaled_dir, source_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    contract = _candidate_contract()

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["strategy_candidate_redesign_plan_json"]), plan)
    Path(output_files["strategy_candidate_redesign_plan_md"]).write_text(
        _markdown(plan, specs, blocked),
        encoding="utf-8",
    )
    write_csv_artifact(
        Path(output_files["available_feature_families_csv"]),
        available_families,
    )
    write_csv_artifact(
        Path(output_files["missing_feature_families_csv"]),
        missing_families,
    )
    write_csv_artifact(Path(output_files["candidate_definition_specs_csv"]), specs)
    write_json_artifact(Path(output_files["candidate_contract_json"]), contract)
    write_csv_artifact(Path(output_files["blocked_definitions_csv"]), blocked)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **plan,
            "manifest": manifest,
            "available_feature_families": available_families,
            "missing_feature_families": missing_families,
            "candidate_definition_specs": specs,
            "candidate_contract": contract,
            "blocked_definitions": blocked,
            "recommendation_payload": recommendation,
        }
    )


def _feature_columns(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("feature_columns", []))


def _available_feature_families(columns: set[str]) -> list[dict[str, str]]:
    output = []
    for family, required in sorted(SAFE_FEATURE_FAMILIES.items()):
        present = [column for column in required if column in columns]
        output.append(
            {
                "feature_family": family,
                "required_or_supported_columns": "|".join(required),
                "present_columns": "|".join(present),
                "status": "AVAILABLE" if present else "MISSING",
            }
        )
    return output


def _missing_feature_families(columns: set[str]) -> list[dict[str, str]]:
    output = []
    for family, required in sorted(MISSING_FEATURE_FAMILIES.items()):
        present = [column for column in required if column in columns]
        output.append(
            {
                "feature_family": family,
                "needed_columns": "|".join(required),
                "present_columns": "|".join(present),
                "status": "AVAILABLE" if present else "MISSING_FOR_REDESIGN",
            }
        )
    return output


def _failure_modes(vol_scaled_dir: Path) -> list[str]:
    comparator = _optional_json(
        vol_scaled_dir / "research_candidate_comparator" / "recommendation.json"
    )
    dashboard = _optional_json(
        vol_scaled_dir / "m20_research_dashboard" / "recommendation.json"
    )
    modes = [
        "BROAD_NOISY_COVERAGE",
        "WEAK_PRECISION_SEPARATION",
        "NEGATIVE_NET_PROXY",
        "POOR_TAIL_BEHAVIOR",
        "SLICE_INSTABILITY",
        "LOW_SAMPLE_SIZE",
        "MISSING_CONTEXT_FILTERS",
    ]
    blockers = comparator.get("evidence_blockers", []) + dashboard.get("evidence_blockers", [])
    if "NO_POSITIVE_PROXY_RESEARCH_CANDIDATE" in blockers:
        modes.append("NO_POSITIVE_PROXY_RESEARCH_CANDIDATE")
    return sorted(set(modes))


def _candidate_specs(columns: set[str]) -> list[dict[str, Any]]:
    definitions = [
        _definition(
            "multi_condition",
            "momentum_volume_confirmed",
            ("momentum_3", "volume_zscore_12", "realized_vol_12"),
            "momentum direction with volume confirmation and non-extreme volatility gate",
        ),
        _definition(
            "volatility_adjusted",
            "macd_volatility_adjusted_direction",
            ("macd_line_12_26", "realized_vol_12", "return_std_12"),
            "MACD direction normalized by recent volatility context",
        ),
        _definition(
            "range_vol_volume_composite",
            "range_expansion_volume_confirmed",
            ("high_price", "low_price", "close_price", "volume_zscore_12", "log_return_1"),
            "range expansion with volume confirmation and return direction",
        ),
        _definition(
            "lower_turnover",
            "high_agreement_low_turnover_setup",
            ("macd_line_12_26", "momentum_3", "rsi_14", "volume_zscore_12"),
            "require multiple feature families to agree before candidate selection",
        ),
        _definition(
            "abstention_hold",
            "tail_risk_avoidance_context",
            ("realized_vol_12", "return_std_12", "close_zscore_12", "volume_zscore_12"),
            "research-only HOLD/avoid setup using ex-ante volatility and z-score context",
        ),
        _definition(
            "tail_risk_aware",
            "non_extreme_volatility_momentum",
            ("momentum_3", "realized_vol_12", "close_zscore_12"),
            "momentum setup gated away from extreme volatility and price displacement",
        ),
        _definition(
            "regime_conditioned",
            "regime_conditioned_momentum",
            ("regime_label", "momentum_3", "realized_vol_12"),
            "regime-conditioned setup only if explicit safe regime labels are exported",
        ),
        _definition(
            "trend_strength_conditioned",
            "trend_strength_filtered_momentum",
            ("adx_14", "momentum_3", "realized_vol_12"),
            "trend-strength filter for future feature-engineering batch",
        ),
    ]
    output = []
    for definition in definitions:
        missing = [column for column in definition["required_features"] if column not in columns]
        output.append(
            {
                **definition,
                "required_features": "|".join(definition["required_features"]),
                "missing_features": "|".join(missing),
                "definition_status": (
                    "READY_FOR_V2_FACTORY"
                    if not missing
                    else "BLOCKED_MISSING_FEATURES"
                ),
            }
        )
    return output


def _definition(
    family: str,
    name: str,
    required_features: Sequence[str],
    rationale: str,
) -> dict[str, Any]:
    return {
        "redesign_family": family,
        "candidate_name": name,
        "candidate_version": "v2_design",
        "required_features": tuple(required_features),
        "rationale": rationale,
        "uses_economic_outcome_as_feature": False,
        "evaluates_candidate_now": False,
    }


def _candidate_contract() -> dict[str, Any]:
    return {
        "contract_name": "m20_strategy_candidate_definition_v2",
        "candidate_keys": ["strategy_family", "candidate_name", "candidate_version"],
        "row_keys": ["symbol", "interval_begin", "fold_index"],
        "allowed_feature_inputs": "training_frame safe feature columns only",
        "forbidden_feature_inputs": [
            "fee_exceedance_label",
            "triple_barrier_label",
            "future_return",
            "gross_value_proxy",
            "net_value_proxy",
            "runtime_registry_fields",
        ],
        "downstream_flow": [
            "generic_strategy_candidate_factory_v2",
            "generic_edge_evaluator",
            "cost_aware_policy_evaluator",
            "research_candidate_comparator",
            "m20_research_dashboard",
        ],
        "required_statuses": list(HONESTY_FLAGS),
    }


def _recommendation(ready_definitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if ready_definitions:
        recommendation = "BUILD_GENERIC_V2_STRATEGY_CANDIDATES"
        next_action = "BUILD_GENERIC_V2_STRATEGY_CANDIDATE_FACTORY"
    else:
        recommendation = "ADD_REQUIRED_FEATURES_BEFORE_V2_CANDIDATES"
        next_action = "PLAN_SAFE_M20_FEATURE_ENGINEERING"
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
            "rationale": "Proceed only through generic candidate tooling.",
        },
        {
            "priority": "2",
            "action": "KEEP_REDESIGN_PLAN_RESEARCH_ONLY",
            "rationale": "No evaluation, runtime, registry, promotion, backtest, or profit claim.",
        },
    ]


def _source_artifacts(vol_scaled_dir: Path, source_dir: Path) -> dict[str, str]:
    return {
        "candidate_metrics": str(
            vol_scaled_dir / "strategy_candidate_factory" / "candidate_metrics.csv"
        ),
        "refined_candidate_metrics": str(
            vol_scaled_dir
            / "strategy_candidate_refinement"
            / "refined_candidate_metrics.csv"
        ),
        "policy_metrics": str(
            vol_scaled_dir / "strategy_slice_policy_evaluator" / "policy_metrics.csv"
        ),
        "comparator_recommendation": str(
            vol_scaled_dir / "research_candidate_comparator" / "recommendation.json"
        ),
        "dashboard_recommendation": str(
            vol_scaled_dir / "m20_research_dashboard" / "recommendation.json"
        ),
        "feature_columns": str(source_dir / FEATURE_COLUMNS_FILE),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "strategy_candidate_redesign_plan_json": str(
            output_dir / "strategy_candidate_redesign_plan.json"
        ),
        "strategy_candidate_redesign_plan_md": str(
            output_dir / "strategy_candidate_redesign_plan.md"
        ),
        "available_feature_families_csv": str(output_dir / "available_feature_families.csv"),
        "missing_feature_families_csv": str(output_dir / "missing_feature_families.csv"),
        "candidate_definition_specs_csv": str(output_dir / "candidate_definition_specs.csv"),
        "candidate_contract_json": str(output_dir / "candidate_contract.json"),
        "blocked_definitions_csv": str(output_dir / "blocked_definitions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    plan: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# M20 Strategy Candidate Redesign Plan",
        "",
        f"- Recommendation: `{plan['recommendation']}`",
        f"- Next required action: `{plan['next_required_action']}`",
        f"- Ready definitions: `{plan['ready_definition_count']}`",
        f"- Blocked definitions: `{plan['blocked_definition_count']}`",
        "- Status: `RESEARCH_ONLY`, `DESIGN_ONLY`, `NO_CANDIDATE_EVALUATION`, "
        "`NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, `NOT_RUNTIME_READY`, "
        "`NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Candidate Definitions",
    ]
    for row in specs:
        lines.append(
            f"- `{row['redesign_family']}:{row['candidate_name']}` -> "
            f"`{row['definition_status']}`"
        )
    if blocked:
        lines.append("")
        lines.append("## Blocked Definitions")
        for row in blocked:
            lines.append(f"- `{row['candidate_name']}` missing `{row['missing_features']}`")
    lines.append("")
    return "\n".join(lines)


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["plan_m20_strategy_candidate_redesign"]
