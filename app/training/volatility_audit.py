"""Research-only volatility source audit and vol-scaled labels for M20."""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.policy_eval import (  # pylint: disable=protected-access
    _build_cost_scenario_specs,
    _load_completed_summary,
    _resolve_completed_winner_model_name,
    resolve_completed_run_dir,
)
from app.training.research_labels import (
    DEFAULT_LABEL_HORIZON,
    _build_fee_exceedance_artifact_rows,  # pylint: disable=protected-access
    _build_label_diagnostics_markdown,  # pylint: disable=protected-access
    _build_research_label_diagnostics,  # pylint: disable=protected-access
    _copy_slice_columns,  # pylint: disable=protected-access
    _load_oof_rows,  # pylint: disable=protected-access
    _resolve_fee_rate,  # pylint: disable=protected-access
    _try_float,  # pylint: disable=protected-access
)
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_AUDIT_DIR_NAME = "volatility_audit"
DEFAULT_VOL_SCALED_DIR_NAME = "vol_scaled"
DEFAULT_VOLATILITY_LOOKBACK = 12
VOLATILITY_NAME_TOKENS = (
    "volatility",
    "realized_volatility",
    "rolling_std",
    "return_std",
    "log_return_std",
    "atr",
    "range",
    "true_range",
    "zscore",
    "vol",
)


def audit_completed_run_volatility_sources(
    *,
    run_dir: Path | None,
    model_name: str | None = None,
    lookback: int = DEFAULT_VOLATILITY_LOOKBACK,
    generate_labels: bool = True,
) -> dict[str, Any]:
    """Audit volatility sources and optionally write vol-scaled research labels."""
    # pylint: disable=too-many-locals,too-many-statements
    resolved_run_dir = resolve_completed_run_dir(run_dir)
    summary_payload = _load_completed_summary(resolved_run_dir)
    resolved_model_name = model_name or _resolve_completed_winner_model_name(summary_payload)
    raw_rows, oof_columns = _load_oof_rows(resolved_run_dir / "oof_predictions.csv")
    model_rows = [
        row for row in raw_rows
        if str(row.get("model_name", "")) == resolved_model_name
    ]
    ordered_rows = _order_rows(model_rows)
    manifest_columns = _load_manifest_columns(resolved_run_dir)
    candidate_rows = _candidate_column_rows(
        oof_columns=oof_columns,
        manifest_columns=manifest_columns,
    )
    selected_existing = _select_existing_oof_volatility_column(candidate_rows)
    flags: list[str] = ["FIXED_BPS_FALLBACK_RETAINED", "NO_RUNTIME_EFFECT", "NO_PROMOTION_EFFECT"]
    if selected_existing:
        flags.append("EXISTING_VOLATILITY_COLUMN_FOUND")
        vol_rows = _attach_existing_volatility(ordered_rows, selected_existing)
        volatility_source = "existing_oof_column"
        selected_column = selected_existing
    elif any(row["available_in_manifest"] for row in candidate_rows):
        flags.append("VOLATILITY_SOURCE_NOT_ALIGNED")
        vol_rows = _attach_computed_volatility_proxy(ordered_rows, lookback=lookback)
        volatility_source = "computed_proxy_from_past_future_return_3"
        selected_column = "research_volatility_proxy"
    else:
        flags.append("VOLATILITY_SOURCE_MISSING")
        vol_rows = _attach_computed_volatility_proxy(ordered_rows, lookback=lookback)
        volatility_source = "computed_proxy_from_past_future_return_3"
        selected_column = "research_volatility_proxy"
    if any(_try_float(row.get(selected_column)) is None for row in vol_rows):
        flags.append("VOLATILITY_HAS_MISSING_VALUES")
    if any(
        (value := _try_float(row.get(selected_column))) is not None and value <= 0.0
        for row in vol_rows
    ):
        flags.append("VOLATILITY_HAS_NON_POSITIVE_VALUES")
    usable_vol_rows = [
        row for row in vol_rows
        if (value := _try_float(row.get(selected_column))) is not None and value > 0.0
    ]
    if volatility_source.startswith("computed_proxy") and usable_vol_rows:
        flags.append("RESEARCH_COMPUTED_VOLATILITY_PROXY")

    audit_dir = resolved_run_dir / "research_labels" / DEFAULT_AUDIT_DIR_NAME
    audit_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _volatility_summary_by_slice(vol_rows, selected_column)
    label_result: dict[str, Any] | None = None
    if generate_labels and usable_vol_rows:
        label_result = _write_vol_scaled_labels(
            run_dir=resolved_run_dir,
            model_name=resolved_model_name,
            summary_payload=summary_payload,
            rows=usable_vol_rows,
            volatility_column=selected_column,
            volatility_source=volatility_source,
        )
        flags.append("VOLATILITY_SCALED_LABELS_GENERATED")
        if "LABELS_NOT_TRAINING_READY" in label_result["diagnostics"]["honesty_flags"]:
            flags.append("VOLATILITY_SCALED_LABELS_NOT_READY")
    else:
        flags.append("VOLATILITY_SCALED_LABELS_NOT_READY")

    output_files = {
        "volatility_source_audit_json": str(audit_dir / "volatility_source_audit.json"),
        "volatility_source_audit_md": str(audit_dir / "volatility_source_audit.md"),
        "volatility_candidate_columns_csv": str(audit_dir / "volatility_candidate_columns.csv"),
        "volatility_summary_by_slice_csv": str(audit_dir / "volatility_summary_by_slice.csv"),
        "volatility_label_regeneration_manifest_json": str(
            audit_dir / "volatility_label_regeneration_manifest.json"
        ),
    }
    audit = {
        "run_dir": str(resolved_run_dir),
        "model_name": resolved_model_name,
        "audit_dir": str(audit_dir),
        "oof_row_count": len(raw_rows),
        "winner_model_row_count": len(ordered_rows),
        "lookback": lookback,
        "volatility_source": volatility_source,
        "selected_volatility_column": selected_column,
        "usable_volatility_row_count": len(usable_vol_rows),
        "candidate_columns": candidate_rows,
        "honesty_flags": sorted(dict.fromkeys(flags)),
        "vol_scaled_label_result": label_result,
        "output_files": output_files,
        "recommendation": _recommend(flags, label_result),
    }
    write_json_artifact(Path(output_files["volatility_source_audit_json"]), audit)
    write_csv_artifact(Path(output_files["volatility_candidate_columns_csv"]), candidate_rows)
    write_csv_artifact(Path(output_files["volatility_summary_by_slice_csv"]), summary_rows)
    write_json_artifact(
        Path(output_files["volatility_label_regeneration_manifest_json"]),
        _regeneration_manifest(audit),
    )
    Path(output_files["volatility_source_audit_md"]).write_text(
        _build_audit_markdown(audit),
        encoding="utf-8",
    )
    return make_json_safe(audit)


def _write_vol_scaled_labels(
    *,
    run_dir: Path,
    model_name: str,
    summary_payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    volatility_column: str,
    volatility_source: str,
) -> dict[str, Any]:
    # pylint: disable=too-many-arguments
    label_dir = run_dir / "research_labels" / DEFAULT_VOL_SCALED_DIR_NAME
    label_dir.mkdir(parents=True, exist_ok=True)
    triple_rows = _build_vol_scaled_triple_rows(rows, volatility_column=volatility_column)
    fee_scenarios = _build_cost_scenario_specs(
        fee_rate=_resolve_fee_rate(summary_payload, rows),
        slippage_rates=(0.0, 0.001),
    )
    fee_rows = _build_fee_exceedance_artifact_rows(
        rows,
        fee_scenarios=fee_scenarios,
        has_return_column=True,
        horizon=DEFAULT_LABEL_HORIZON,
    )
    diagnostics = _build_research_label_diagnostics(
        triple_rows=triple_rows,
        fee_rows=fee_rows,
        meta_rows=[],
        warnings=["LABELS_NOT_TRAINING_READY"],
        skipped_horizon_count=len(rows) - len(triple_rows),
        source_row_count=len(rows),
    )
    fixed_rows = _read_csv_rows(run_dir / "research_labels" / "triple_barrier_labels.csv")
    comparison = _compare_fixed_and_vol_scaled(fixed_rows, triple_rows)
    diagnostics["fixed_bps_comparison"] = comparison
    output_files = {
        "research_labels_manifest_vol_scaled_json": str(
            label_dir / "research_labels_manifest_vol_scaled.json"
        ),
        "triple_barrier_labels_vol_scaled_csv": str(
            label_dir / "triple_barrier_labels_vol_scaled.csv"
        ),
        "fee_exceedance_labels_vol_scaled_csv": str(
            label_dir / "fee_exceedance_labels_vol_scaled.csv"
        ),
        "label_diagnostics_vol_scaled_json": str(label_dir / "label_diagnostics_vol_scaled.json"),
        "label_diagnostics_vol_scaled_md": str(label_dir / "label_diagnostics_vol_scaled.md"),
        "label_distribution_by_slice_vol_scaled_csv": str(
            label_dir / "label_distribution_by_slice_vol_scaled.csv"
        ),
    }
    manifest = {
        "run_dir": str(run_dir),
        "label_dir": str(label_dir),
        "model_name": model_name,
        "source": "completed_m20_oof_predictions_with_research_volatility",
        "volatility_source": volatility_source,
        "volatility_column": volatility_column,
        "winner_model_row_count": len(rows),
        "horizon": DEFAULT_LABEL_HORIZON,
        "honesty_flags": diagnostics["honesty_flags"],
        "fixed_bps_fallback_retained": True,
        "output_files": output_files,
        "runtime_effect": "none_research_only",
    }
    write_json_artifact(Path(output_files["research_labels_manifest_vol_scaled_json"]), manifest)
    write_csv_artifact(Path(output_files["triple_barrier_labels_vol_scaled_csv"]), triple_rows)
    write_csv_artifact(Path(output_files["fee_exceedance_labels_vol_scaled_csv"]), fee_rows)
    write_json_artifact(Path(output_files["label_diagnostics_vol_scaled_json"]), diagnostics)
    write_csv_artifact(
        Path(output_files["label_distribution_by_slice_vol_scaled_csv"]),
        diagnostics["label_distribution_by_slice"],
    )
    Path(output_files["label_diagnostics_vol_scaled_md"]).write_text(
        _build_label_diagnostics_markdown(manifest, diagnostics),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **manifest,
            "diagnostics": diagnostics,
        }
    )


def _build_vol_scaled_triple_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    volatility_column: str,
) -> list[dict[str, Any]]:
    output_rows = []
    for source_index, row in enumerate(rows):
        future_return = _try_float(row.get("future_return_3"))
        volatility = _try_float(row.get(volatility_column))
        if future_return is None or volatility is None or volatility <= 0.0:
            continue
        if future_return >= volatility:
            label = 1
            barrier_hit = "upper"
        elif future_return <= -volatility:
            label = -1
            barrier_hit = "lower"
        else:
            label = 0
            barrier_hit = "vertical"
        output_rows.append(
            {
                **_copy_slice_columns(row),
                "row_id": str(row.get("row_id", source_index)),
                "source_index": source_index,
                "label": label,
                "barrier_hit": barrier_hit,
                "future_return": future_return,
                "volatility": volatility,
                "upper_barrier_return": volatility,
                "lower_barrier_return": -volatility,
                "horizon": DEFAULT_LABEL_HORIZON,
                "label_source": "return_proxy_volatility_scaled",
            }
        )
    return output_rows


def _attach_computed_volatility_proxy(
    rows: Sequence[Mapping[str, str]],
    *,
    lookback: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("symbol", "all")), []).append(row)
    output_rows: list[dict[str, Any]] = []
    for symbol_rows in grouped.values():
        past_returns: list[float] = []
        for row in symbol_rows:
            enriched = dict(row)
            if len(past_returns) >= lookback:
                window = past_returns[-lookback:]
                enriched["research_volatility_proxy"] = _std(window)
            else:
                enriched["research_volatility_proxy"] = ""
            current_return = _try_float(row.get("future_return_3"))
            if current_return is not None:
                past_returns.append(current_return)
            output_rows.append(enriched)
    return _order_rows(output_rows)


def _attach_existing_volatility(
    rows: Sequence[Mapping[str, str]],
    column: str,
) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if column in row]


def _candidate_column_rows(
    *,
    oof_columns: Sequence[str],
    manifest_columns: Sequence[str],
) -> list[dict[str, Any]]:
    all_columns = sorted(set(oof_columns) | set(manifest_columns))
    rows = []
    for column in all_columns:
        lowered = column.lower()
        matched_tokens = [token for token in VOLATILITY_NAME_TOKENS if token in lowered]
        if not matched_tokens:
            continue
        rows.append(
            {
                "column_name": column,
                "matched_tokens": ";".join(matched_tokens),
                "available_in_oof": column in oof_columns,
                "available_in_manifest": column in manifest_columns,
                "preference_rank": _volatility_preference_rank(column),
            }
        )
    return sorted(rows, key=lambda row: (int(row["preference_rank"]), str(row["column_name"])))


def _select_existing_oof_volatility_column(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> str | None:
    available = [row for row in candidate_rows if row["available_in_oof"]]
    if not available:
        return None
    selected = sorted(
        available,
        key=lambda row: (row["preference_rank"], row["column_name"]),
    )[0]
    return str(selected["column_name"])


def _volatility_preference_rank(column: str) -> int:
    lowered = column.lower()
    preferences = (
        "realized_vol_12",
        "realized_volatility",
        "return_std",
        "log_return_std",
        "rolling_std",
        "true_range",
        "atr",
        "range",
        "vol",
        "zscore",
    )
    for index, token in enumerate(preferences):
        if token in lowered:
            return index
    return len(preferences)


def _load_manifest_columns(run_dir: Path) -> list[str]:
    columns: list[str] = []
    for filename in (
        "feature_columns.json",
        "dataset_manifest.json",
        "summary.json",
        "run_config.json",
    ):
        path = run_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        _collect_strings(payload, columns)
    return sorted(dict.fromkeys(columns))


def _collect_strings(value: Any, output: list[str]) -> None:
    if isinstance(value, str):
        output.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            _collect_strings(item, output)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, output)


def _volatility_summary_by_slice(
    rows: Sequence[Mapping[str, Any]],
    volatility_column: str,
) -> list[dict[str, Any]]:
    output_rows = [_summary_row("overall", "all", rows, volatility_column)]
    for slice_column in ("symbol", "fold_index", "regime_label"):
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            value = row.get(slice_column)
            if value is None or value == "":
                continue
            grouped.setdefault(str(value), []).append(row)
        for value, group_rows in sorted(grouped.items()):
            output_rows.append(_summary_row(slice_column, value, group_rows, volatility_column))
    return output_rows


def _summary_row(
    slice_column: str,
    slice_value: str,
    rows: Sequence[Mapping[str, Any]],
    volatility_column: str,
) -> dict[str, Any]:
    values = sorted(
        value for row in rows
        if (value := _try_float(row.get(volatility_column))) is not None
    )
    return {
        "slice_column": slice_column,
        "slice_value": slice_value,
        "row_count": len(rows),
        "finite_volatility_count": len(values),
        "missing_volatility_count": len(rows) - len(values),
        "non_positive_volatility_count": sum(1 for value in values if value <= 0.0),
        "min": min(values) if values else None,
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "max": max(values) if values else None,
    }


def _compare_fixed_and_vol_scaled(
    fixed_rows: Sequence[Mapping[str, str]],
    vol_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fixed_by_id = {str(row.get("row_id")): row for row in fixed_rows}
    paired = [
        (fixed_by_id[str(row.get("row_id"))], row)
        for row in vol_rows
        if str(row.get("row_id")) in fixed_by_id
    ]
    agreement_count = sum(
        1 for fixed, vol in paired
        if int(float(fixed.get("label", 999))) == int(vol.get("label", 999))
    )
    fixed_distribution = Counter(str(row.get("label")) for row in fixed_rows)
    vol_distribution = Counter(str(row.get("label")) for row in vol_rows)
    return {
        "paired_row_count": len(paired),
        "agreement_count": agreement_count,
        "label_agreement_rate": agreement_count / len(paired) if paired else 0.0,
        "fixed_bps_label_counts": dict(sorted(fixed_distribution.items())),
        "vol_scaled_label_counts": dict(sorted(vol_distribution.items())),
        "event_rate_stability_by_slice": _event_rate_stability_by_slice(fixed_rows, vol_rows),
    }


def _event_rate_stability_by_slice(
    fixed_rows: Sequence[Mapping[str, str]],
    vol_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for slice_column in ("symbol", "fold_index", "regime_label"):
        fixed_groups = _group_rows(fixed_rows, slice_column)
        vol_groups = _group_rows(vol_rows, slice_column)
        for value in sorted(set(fixed_groups) | set(vol_groups)):
            fixed_rate = _event_rate(fixed_groups.get(value, []))
            vol_rate = _event_rate(vol_groups.get(value, []))
            rows.append(
                {
                    "slice_column": slice_column,
                    "slice_value": value,
                    "fixed_event_rate": fixed_rate,
                    "vol_scaled_event_rate": vol_rate,
                    "absolute_delta": abs(vol_rate - fixed_rate),
                }
            )
    return rows


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    column: str,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get(column) is not None:
            grouped.setdefault(str(row[column]), []).append(row)
    return grouped


def _event_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    event_count = sum(1 for row in rows if int(float(row.get("label", 0))) != 0)
    return event_count / len(rows)


def _regeneration_manifest(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": audit["run_dir"],
        "model_name": audit["model_name"],
        "volatility_source": audit["volatility_source"],
        "selected_volatility_column": audit["selected_volatility_column"],
        "honesty_flags": audit["honesty_flags"],
        "vol_scaled_labels_generated": (
            "VOLATILITY_SCALED_LABELS_GENERATED" in audit["honesty_flags"]
        ),
        "vol_scaled_label_result": audit.get("vol_scaled_label_result"),
        "runtime_effect": "none_research_only",
    }


def _build_audit_markdown(audit: Mapping[str, Any]) -> str:
    label_result = audit.get("vol_scaled_label_result") or {}
    diagnostics = label_result.get("diagnostics", {})
    comparison = diagnostics.get("fixed_bps_comparison", {})
    return "\n".join(
        [
            "# M20 Volatility Source Audit",
            "",
            f"- Run directory: `{audit['run_dir']}`",
            f"- Volatility source: `{audit['volatility_source']}`",
            f"- Selected column: `{audit['selected_volatility_column']}`",
            f"- Honesty flags: `{', '.join(audit['honesty_flags'])}`",
            f"- Recommendation: `{audit['recommendation']}`",
            "",
            "## Fixed-BPS Comparison",
            "",
            f"- Paired rows: `{comparison.get('paired_row_count', 0)}`",
            f"- Label agreement rate: `{float(comparison.get('label_agreement_rate', 0.0)):.6f}`",
            "",
            "This is research-only. No runtime inference, promotion, registry, "
            "execution, training, or roster behavior changed.",
            "",
        ]
    )


def _recommend(flags: Sequence[str], label_result: Mapping[str, Any] | None) -> str:
    if "VOLATILITY_SCALED_LABELS_GENERATED" in flags and label_result:
        return "A. train a tiny research-only triple-barrier baseline next"
    if "RESEARCH_COMPUTED_VOLATILITY_PROXY" in flags:
        return "D. improve feature export to include volatility"
    if "VOLATILITY_SOURCE_MISSING" in flags:
        return "C. keep using fixed-bps labels only as limited research diagnostics"
    return "E. reject this M20 target path for now"


def _order_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in rows],
        key=lambda row: (
            str(row.get("symbol", "")),
            int(float(row.get("fold_index", 0) or 0)),
            str(row.get("interval_begin", "")),
            str(row.get("row_id", "")),
        ),
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def _quantile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return values[lower_index]
    fraction = position - lower_index
    return values[lower_index] + ((values[upper_index] - values[lower_index]) * fraction)
