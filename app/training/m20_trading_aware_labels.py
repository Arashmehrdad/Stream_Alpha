"""Research-only M20 trading-aware label artifact builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_policy_research_common import (
    HONESTY_FLAGS,
    KEY_WITH_FOLD,
    keyed_rows,
    present,
    read_csv_rows,
    row_key,
    to_float,
    vol_scaled_dir,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_OUTPUT_NAME = "trading_aware_labels"
DEFAULT_ECONOMIC_DIR_NAME = "economic_outcome_artifacts"
DEFAULT_FEATURE_DIR_NAME = "research_feature_enrichment"


def build_m20_trading_aware_labels(
    *,
    source_run_dir: Path,
    economic_outcome_dir: Path | None = None,
    research_feature_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Build deterministic research-only trading-aware labels."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    research_dir = vol_scaled_dir(source_dir)
    outcome_dir = (
        Path(economic_outcome_dir).resolve()
        if economic_outcome_dir
        else research_dir / DEFAULT_ECONOMIC_DIR_NAME
    )
    feature_dir = (
        Path(research_feature_dir).resolve()
        if research_feature_dir
        else research_dir / DEFAULT_FEATURE_DIR_NAME
    )
    output_dir = research_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = read_csv_rows(outcome_dir / "economic_outcomes.csv")
    features = read_csv_rows(feature_dir / "research_features.csv")
    if not outcomes:
        raise ValueError(f"Missing economic outcomes: {outcome_dir / 'economic_outcomes.csv'}")
    feature_index = keyed_rows(features, KEY_WITH_FOLD)
    label_rows = _label_rows(outcomes, feature_index)
    blocked = _blocked_labels()
    lineage = _lineage(outcome_dir, feature_dir)
    leakage = _leakage_audit()
    recommendation = _recommendation(label_rows)
    output_files = _output_files(output_dir)
    schema = _label_schema()
    manifest = {
        "source_run_dir": str(source_dir),
        "economic_outcome_dir": str(outcome_dir),
        "research_feature_dir": str(feature_dir),
        "row_count": len(label_rows),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    report = {
        "summary": "Research-only M20 trading-aware labels.",
        "row_count": len(label_rows),
        "blocked_label_count": len(blocked),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_csv_artifact(Path(output_files["trading_aware_labels_csv"]), label_rows)
    write_json_artifact(Path(output_files["label_schema_json"]), schema)
    write_csv_artifact(Path(output_files["label_lineage_csv"]), lineage)
    write_csv_artifact(Path(output_files["leakage_audit_csv"]), leakage)
    write_csv_artifact(Path(output_files["blocked_labels_csv"]), blocked)
    write_json_artifact(Path(output_files["trading_aware_label_report_json"]), report)
    Path(output_files["trading_aware_label_report_md"]).write_text(
        _markdown(report, blocked),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "label_schema": schema,
            "blocked_labels": blocked,
            "label_lineage": lineage,
            "leakage_audit": leakage,
            "recommendation_payload": recommendation,
        }
    )


def _label_rows(
    outcomes: Sequence[Mapping[str, str]],
    feature_index: Mapping[tuple[str, ...], Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for outcome in outcomes:
        feature = feature_index.get(row_key(outcome, KEY_WITH_FOLD), {})
        realized_vol = to_float(feature.get("realized_vol_12"))
        future_return = to_float(outcome.get("future_return"))
        vol_adjusted = future_return / realized_vol if realized_vol > 0.0 else ""
        rows.append(
            {
                "fold_index": outcome.get("fold_index", ""),
                "symbol": outcome.get("symbol", ""),
                "interval_begin": outcome.get("interval_begin", ""),
                "fee_plus_slippage_exceedance_label": (
                    1 if to_float(outcome.get("net_value_proxy")) > 0.0 else 0
                ),
                "triple_barrier_label": outcome.get("triple_barrier_label", ""),
                "future_return": outcome.get("future_return", ""),
                "net_value_proxy": outcome.get("net_value_proxy", ""),
                "realized_vol_12": feature.get("realized_vol_12", ""),
                "volatility_adjusted_forward_return": vol_adjusted,
                "volatility_adjusted_return_bucket": _vol_bucket(vol_adjusted),
                "label_usage": "RESEARCH_ONLY_POLICY_DIAGNOSTIC",
            }
        )
    return rows


def _vol_bucket(value: Any) -> str:
    if not present(value):
        return "UNAVAILABLE"
    numeric = to_float(value)
    if numeric >= 1.0:
        return "POSITIVE_STRONG"
    if numeric > 0.0:
        return "POSITIVE"
    if numeric <= -1.0:
        return "NEGATIVE_STRONG"
    return "NEGATIVE"


def _blocked_labels() -> list[dict[str, str]]:
    return [
        {
            "label_name": "forward_return_6_candles",
            "blocker": "MISSING_SAFE_FORWARD_RETURN_6_SOURCE",
        },
        {
            "label_name": "forward_return_12_candles",
            "blocker": "MISSING_SAFE_FORWARD_RETURN_12_SOURCE",
        },
    ]


def _lineage(outcome_dir: Path, feature_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "label_name": "fee_plus_slippage_exceedance_label",
            "source_artifact": str(outcome_dir / "economic_outcomes.csv"),
            "source_columns": "net_value_proxy",
            "uses_future_data": "True",
            "runtime_effect": "False",
        },
        {
            "label_name": "volatility_adjusted_forward_return",
            "source_artifact": (
                f"{outcome_dir / 'economic_outcomes.csv'}|"
                f"{feature_dir / 'research_features.csv'}"
            ),
            "source_columns": "future_return|realized_vol_12",
            "uses_future_data": "True",
            "runtime_effect": "False",
        },
    ]


def _leakage_audit() -> list[dict[str, str]]:
    return [
        {
            "audit_name": "LABEL_ARTIFACT_USAGE",
            "status": "LABELS_ARE_OUTCOMES_NOT_POLICY_SELECTION_FEATURES",
            "runtime_effect": "False",
        },
        {
            "audit_name": "ORIGINAL_TRAINING_FRAME_MUTATION",
            "status": "NOT_MUTATED",
            "runtime_effect": "False",
        },
    ]


def _label_schema() -> dict[str, Any]:
    return {
        "schema_version": "m20_trading_aware_labels_v1",
        "required_keys": list(KEY_WITH_FOLD),
        "label_columns": [
            "fee_plus_slippage_exceedance_label",
            "triple_barrier_label",
            "volatility_adjusted_forward_return",
            "volatility_adjusted_return_bucket",
        ],
        "honesty_flags": list(HONESTY_FLAGS),
        "selection_input_allowed": False,
    }


def _recommendation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recommendation = (
        "RE_RUN_DECISION_POLICY_EVALUATOR_WITH_TRADING_AWARE_LABELS"
        if rows
        else "BLOCKED_MISSING_TRADING_AWARE_LABEL_INPUTS"
    )
    return {
        "recommendation": recommendation,
        "next_required_action": recommendation,
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
            "rationale": "Use labels only for research diagnostics and calibration.",
        }
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "trading_aware_labels_csv": str(output_dir / "trading_aware_labels.csv"),
        "label_schema_json": str(output_dir / "label_schema.json"),
        "label_lineage_csv": str(output_dir / "label_lineage.csv"),
        "leakage_audit_csv": str(output_dir / "leakage_audit.csv"),
        "blocked_labels_csv": str(output_dir / "blocked_labels.csv"),
        "trading_aware_label_report_json": str(
            output_dir / "trading_aware_label_report.json"
        ),
        "trading_aware_label_report_md": str(
            output_dir / "trading_aware_label_report.md"
        ),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    blocked: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Trading-Aware Research Labels",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Rows: `{report['row_count']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Blocked Labels",
    ]
    lines.extend(f"- `{row['label_name']}`: `{row['blocker']}`" for row in blocked)
    return "\n".join(lines) + "\n"


__all__ = ["build_m20_trading_aware_labels"]
