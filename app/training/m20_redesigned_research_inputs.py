"""Build redesigned research-only M20 inputs from safe existing artifacts."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_policy_research_common import (
    HONESTY_FLAGS,
    present,
    read_csv_rows,
    to_float,
    vol_scaled_dir,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_OUTPUT_NAME = "m20_redesigned_research_inputs"
HORIZONS = (6, 12)


def build_m20_redesigned_research_inputs(
    *,
    source_run_dir: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Compute safe redesigned research labels and event masks."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    research_dir = vol_scaled_dir(source_dir)
    output_dir = research_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    features = read_csv_rows(research_dir / "research_feature_enrichment" / "research_features.csv")
    outcomes = read_csv_rows(research_dir / "economic_outcome_artifacts" / "economic_outcomes.csv")
    if not features:
        raise ValueError("Missing research feature enrichment rows")
    fee_bps, slippage_bps = _cost_assumptions(outcomes)
    rows = _redesigned_rows(features, fee_bps, slippage_bps)
    labels = [_label_row(row) for row in rows]
    masks = [_mask_row(row) for row in rows]
    diagnostics = _label_diagnostics(labels)
    coverage = _feature_coverage(features)
    lineage = _lineage(fee_bps, slippage_bps)
    leakage = _leakage_audit()
    blocked = _blocked_inputs(rows)
    recommendation = _recommendation(blocked)
    output_files = _output_files(output_dir)
    report = {
        "summary": "M20 redesigned research-only inputs.",
        "row_count": len(rows),
        "blocked_input_count": len(blocked),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "source_run_dir": str(source_dir),
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "horizons": list(HORIZONS),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_csv_artifact(Path(output_files["redesigned_research_inputs_csv"]), rows)
    write_csv_artifact(Path(output_files["multi_horizon_labels_csv"]), labels)
    write_csv_artifact(Path(output_files["event_sampling_masks_csv"]), masks)
    write_csv_artifact(Path(output_files["label_diagnostics_csv"]), diagnostics)
    write_csv_artifact(Path(output_files["feature_coverage_diagnostics_csv"]), coverage)
    write_csv_artifact(Path(output_files["lineage_csv"]), lineage)
    write_csv_artifact(Path(output_files["leakage_audit_csv"]), leakage)
    write_csv_artifact(Path(output_files["blocked_inputs_csv"]), blocked)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(_markdown(report, diagnostics, blocked), "utf-8")
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "label_diagnostics": diagnostics,
            "blocked_inputs": blocked,
            "recommendation_payload": recommendation,
        }
    )


def _redesigned_rows(
    features: Sequence[Mapping[str, str]],
    fee_bps: float,
    slippage_bps: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in sorted(
        features,
        key=lambda item: (item.get("symbol", ""), item.get("interval_begin", "")),
    ):
        grouped[str(row.get("symbol", ""))].append(row)
    output = []
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    for symbol_rows in grouped.values():
        for index, row in enumerate(symbol_rows):
            payload = _base_row(row)
            payload.update(_event_masks(row))
            for horizon in HORIZONS:
                payload.update(_horizon_payload(symbol_rows, index, horizon, cost_rate))
            output.append(payload)
    return output


def _base_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "fold_index": row.get("fold_index", ""),
        "symbol": row.get("symbol", ""),
        "interval_begin": row.get("interval_begin", ""),
        "regime_label": row.get("regime_label", ""),
        "adx_14": row.get("adx_14", ""),
        "realized_vol_12": row.get("realized_vol_12", ""),
        "close_price": row.get("close_price", ""),
    }


def _event_masks(row: Mapping[str, str]) -> dict[str, int]:
    realized_vol = to_float(row.get("realized_vol_12"))
    close_z = abs(to_float(row.get("close_zscore_12")))
    volume_z = abs(to_float(row.get("volume_zscore_12")))
    adx = to_float(row.get("adx_14"))
    momentum = to_float(row.get("momentum_3"))
    return {
        "event_mask_low_turnover": int(
            0.004 <= realized_vol <= 0.025 and close_z <= 1.5
        ),
        "event_mask_regime_trend": int(
            str(row.get("regime_label", "")).startswith("TREND") and adx >= 20.0
        ),
        "event_mask_cost_aware_min_move": int(abs(momentum) >= max(realized_vol, 0.002)),
        "event_mask_tail_risk_exclusion": int(
            realized_vol <= 0.03 and close_z <= 2.0 and volume_z <= 2.5
        ),
    }


def _horizon_payload(
    rows: Sequence[Mapping[str, str]],
    index: int,
    horizon: int,
    cost_rate: float,
) -> dict[str, Any]:
    current = rows[index]
    future_index = index + horizon
    prefix = f"{horizon}"
    if future_index >= len(rows):
        return _blank_horizon(prefix)
    close = to_float(current.get("close_price"))
    future_close = to_float(rows[future_index].get("close_price"))
    if close <= 0.0 or future_close <= 0.0:
        return _blank_horizon(prefix)
    future_return = (future_close - close) / close
    net = future_return - cost_rate
    realized_vol = to_float(current.get("realized_vol_12"))
    vol_adjusted = future_return / realized_vol if realized_vol > 0.0 else ""
    return {
        f"future_return_{prefix}": future_return,
        f"gross_value_proxy_{prefix}": future_return,
        f"net_value_proxy_{prefix}": net,
        f"fee_plus_slippage_exceedance_{prefix}": int(net > 0.0),
        f"volatility_adjusted_forward_return_{prefix}": vol_adjusted,
        f"volatility_adjusted_return_bucket_{prefix}": _vol_bucket(vol_adjusted),
        f"volatility_scaled_triple_barrier_{prefix}": _triple_barrier(rows, index, horizon),
    }


def _blank_horizon(prefix: str) -> dict[str, str]:
    return {
        f"future_return_{prefix}": "",
        f"gross_value_proxy_{prefix}": "",
        f"net_value_proxy_{prefix}": "",
        f"fee_plus_slippage_exceedance_{prefix}": "",
        f"volatility_adjusted_forward_return_{prefix}": "",
        f"volatility_adjusted_return_bucket_{prefix}": "UNAVAILABLE",
        f"volatility_scaled_triple_barrier_{prefix}": "",
    }


def _triple_barrier(rows: Sequence[Mapping[str, str]], index: int, horizon: int) -> int:
    current = rows[index]
    close = to_float(current.get("close_price"))
    realized_vol = to_float(current.get("realized_vol_12"))
    if close <= 0.0 or realized_vol <= 0.0:
        return 0
    barrier = max(realized_vol * sqrt(horizon), 0.001)
    upper = close * (1.0 + barrier)
    lower = close * (1.0 - barrier)
    for future in rows[index + 1 : index + horizon + 1]:
        if to_float(future.get("high_price")) >= upper:
            return 1
        if to_float(future.get("low_price")) <= lower:
            return -1
    return 0


def _label_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = ["fold_index", "symbol", "interval_begin"]
    for horizon in HORIZONS:
        fields.extend(
            [
                f"future_return_{horizon}",
                f"net_value_proxy_{horizon}",
                f"fee_plus_slippage_exceedance_{horizon}",
                f"volatility_adjusted_forward_return_{horizon}",
                f"volatility_adjusted_return_bucket_{horizon}",
                f"volatility_scaled_triple_barrier_{horizon}",
            ]
        )
    return {field: row.get(field, "") for field in fields}


def _mask_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = [
        "fold_index",
        "symbol",
        "interval_begin",
        "event_mask_low_turnover",
        "event_mask_regime_trend",
        "event_mask_cost_aware_min_move",
        "event_mask_tail_risk_exclusion",
    ]
    return {field: row.get(field, "") for field in fields}


def _label_diagnostics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for horizon in HORIZONS:
        label = f"fee_plus_slippage_exceedance_{horizon}"
        usable = [row for row in rows if present(row.get(label))]
        positives = sum(1 for row in usable if int(row[label]) == 1)
        output.append(
            {
                "label_name": label,
                "usable_rows": len(usable),
                "missing_rows": len(rows) - len(usable),
                "positive_rate": positives / len(usable) if usable else "",
            }
        )
    return output


def _feature_coverage(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    columns = (
        "close_price",
        "high_price",
        "low_price",
        "realized_vol_12",
        "regime_label",
        "adx_14",
    )
    return [
        {
            "feature_name": column,
            "available_rows": sum(1 for row in rows if present(row.get(column))),
            "total_rows": len(rows),
        }
        for column in columns
    ]


def _lineage(fee_bps: float, slippage_bps: float) -> list[dict[str, str]]:
    return [
        {
            "artifact": "multi_horizon_labels",
            "source": "research_feature_enrichment/research_features.csv",
            "uses_future_data": "True",
            "selection_input_allowed": "False",
            "fee_bps": str(fee_bps),
            "slippage_bps": str(slippage_bps),
        },
        {
            "artifact": "event_sampling_masks",
            "source": "research_feature_enrichment/research_features.csv",
            "uses_future_data": "False",
            "selection_input_allowed": "True",
            "fee_bps": "",
            "slippage_bps": "",
        },
    ]


def _leakage_audit() -> list[dict[str, str]]:
    return [
        {
            "audit_name": "MULTI_HORIZON_LABEL_USAGE",
            "status": "OUTCOME_LABELS_ONLY_NOT_SELECTION_INPUTS",
        },
        {
            "audit_name": "EVENT_MASK_USAGE",
            "status": "USES_CURRENT_OR_PAST_FEATURES_ONLY",
        },
        {"audit_name": "TRAINING_FRAME_MUTATION", "status": "NOT_MUTATED"},
    ]


def _blocked_inputs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    output = []
    for horizon in HORIZONS:
        missing = sum(1 for row in rows if not present(row.get(f"future_return_{horizon}")))
        if missing == len(rows):
            output.append(
                {
                    "input_name": f"future_return_{horizon}",
                    "blocker": "NO_COMPUTABLE_ROWS",
                }
            )
    return output


def _cost_assumptions(outcomes: Sequence[Mapping[str, str]]) -> tuple[float, float]:
    for row in outcomes:
        if present(row.get("fee_bps")):
            return to_float(row.get("fee_bps")), to_float(row.get("slippage_bps"))
    return 20.0, 0.0


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


def _recommendation(blocked: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    recommendation = (
        "RE_RUN_DECISION_POLICY_EVALUATOR_WITH_REDESIGNED_INPUTS"
        if not blocked
        else "M20_BLOCKED_MISSING_SAFE_INPUTS"
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
    return [{"priority": "1", "action": str(recommendation["next_required_action"])}]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "redesigned_research_inputs_csv": str(output_dir / "redesigned_research_inputs.csv"),
        "multi_horizon_labels_csv": str(output_dir / "multi_horizon_labels.csv"),
        "event_sampling_masks_csv": str(output_dir / "event_sampling_masks.csv"),
        "label_diagnostics_csv": str(output_dir / "label_diagnostics.csv"),
        "feature_coverage_diagnostics_csv": str(
            output_dir / "feature_coverage_diagnostics.csv"
        ),
        "lineage_csv": str(output_dir / "lineage.csv"),
        "leakage_audit_csv": str(output_dir / "leakage_audit.csv"),
        "blocked_inputs_csv": str(output_dir / "blocked_inputs.csv"),
        "report_json": str(output_dir / "m20_redesigned_research_inputs.json"),
        "report_md": str(output_dir / "m20_redesigned_research_inputs.md"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Redesigned Research Inputs",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Rows: `{report['row_count']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Label Diagnostics",
    ]
    lines.extend(
        f"- `{row['label_name']}`: usable `{row['usable_rows']}`, "
        f"positive rate `{row['positive_rate']}`"
        for row in diagnostics
    )
    if blocked:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- `{row['input_name']}`: `{row['blocker']}`" for row in blocked)
    return "\n".join(lines) + "\n"


__all__ = ["build_m20_redesigned_research_inputs"]
