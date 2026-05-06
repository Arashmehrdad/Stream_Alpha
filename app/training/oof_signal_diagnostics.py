"""Research-only OOF signal diagnostics for completed M20 runs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.policy_eval import (
    DEFAULT_ANALYSIS_DIR_NAME,
    _build_cost_scenario_specs,  # pylint: disable=protected-access
    _load_completed_summary,  # pylint: disable=protected-access
    _normalize_threshold_grid,  # pylint: disable=protected-access
    _resolve_completed_winner_model_name,  # pylint: disable=protected-access
    resolve_completed_run_dir,
)
from app.training.threshold_analysis import (
    resolve_fee_rate,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_DIAGNOSTICS_DIR_NAME = "diagnostics"
DEFAULT_DIAGNOSTIC_THRESHOLDS = tuple(round(value / 100.0, 2) for value in range(5, 100, 5))
DEFAULT_SCORE_COLUMN_CANDIDATES = ("prob_up", "confidence", "score", "raw_score", "y_pred")
DEFAULT_SCENARIO_SLIPPAGE_RATES = (0.0, 0.001)
QUANTILE_POINTS = (
    ("p01", 0.01),
    ("p05", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)


def diagnose_completed_run(
    *,
    run_dir: Path | None,
    model_name: str | None = None,
    thresholds: Iterable[float] = DEFAULT_DIAGNOSTIC_THRESHOLDS,
    score_columns: Sequence[str] | None = None,
    symbol: str | None = None,
    fold_index: int | None = None,
    regime_label: str | None = None,
    analysis_dir_name: str = DEFAULT_ANALYSIS_DIR_NAME,
    diagnostics_dir_name: str = DEFAULT_DIAGNOSTICS_DIR_NAME,
    scenario_slippage_rates: Sequence[float] = DEFAULT_SCENARIO_SLIPPAGE_RATES,
) -> dict[str, Any]:
    """Diagnose OOF score observability for a completed M20 run."""
    # pylint: disable=too-many-arguments,too-many-locals
    resolved_run_dir = resolve_completed_run_dir(run_dir)
    summary_payload = _load_completed_summary(resolved_run_dir)
    resolved_model_name = model_name or _resolve_completed_winner_model_name(summary_payload)
    oof_path = resolved_run_dir / "oof_predictions.csv"
    raw_rows, field_names = _load_oof_rows(oof_path)
    candidate_score_columns = _resolve_score_columns(field_names, score_columns)
    model_rows = [
        row for row in raw_rows
        if str(row.get("model_name", "")) == resolved_model_name
    ]
    filtered_rows = _apply_optional_filters(
        model_rows,
        symbol=symbol,
        fold_index=fold_index,
        regime_label=regime_label,
    )
    primary_score_column = candidate_score_columns[0] if candidate_score_columns else None
    finite_primary_rows = (
        [
            row for row in filtered_rows
            if _try_float(row.get(primary_score_column)) is not None
        ]
        if primary_score_column is not None
        else []
    )
    label_rows = [row for row in finite_primary_rows if _has_any_label(row)]
    fee_rate = _resolve_fee_rate_from_rows(summary_payload, filtered_rows)
    scenario_specs = _build_cost_scenario_specs(
        fee_rate=fee_rate,
        slippage_rates=scenario_slippage_rates,
    )
    threshold_grid = _normalize_threshold_grid(thresholds)
    quantile_rows = [
        _build_score_quantile_row(filtered_rows, score_column)
        for score_column in candidate_score_columns
    ]
    crossing_rows = _build_threshold_crossing_rows(
        filtered_rows,
        score_columns=candidate_score_columns,
        thresholds=threshold_grid,
        scenario_specs=scenario_specs,
    )
    funnel_rows = _build_filter_funnel_rows(
        raw_rows=raw_rows,
        field_names=field_names,
        model_rows=model_rows,
        filtered_rows=filtered_rows,
        finite_primary_rows=finite_primary_rows,
        label_rows=label_rows,
        primary_score_column=primary_score_column,
        filters={"symbol": symbol, "fold_index": fold_index, "regime_label": regime_label},
    )
    label_diagnostics = _build_label_diagnostics(finite_primary_rows, primary_score_column)
    cost_diagnostics = _build_cost_diagnostics(
        filtered_rows,
        score_column=primary_score_column,
        thresholds=threshold_grid,
        scenario_specs=scenario_specs,
    )
    honesty_flags = _build_honesty_flags(
        field_names=field_names,
        quantile_rows=quantile_rows,
        crossing_rows=crossing_rows,
        funnel_rows=funnel_rows,
        label_diagnostics=label_diagnostics,
        primary_score_column=primary_score_column,
    )
    diagnostics_dir = resolved_run_dir / analysis_dir_name / diagnostics_dir_name
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    json_path = diagnostics_dir / "oof_signal_diagnostics.json"
    quantiles_path = diagnostics_dir / "oof_score_quantiles.csv"
    crossings_path = diagnostics_dir / "oof_threshold_crossing_counts.csv"
    funnel_path = diagnostics_dir / "oof_filter_funnel.csv"
    markdown_path = diagnostics_dir / "oof_signal_diagnostics.md"

    summary = {
        "run_dir": str(resolved_run_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "model_name": resolved_model_name,
        "field_names": list(field_names),
        "score_columns": candidate_score_columns,
        "primary_score_column": primary_score_column,
        "threshold_grid": threshold_grid,
        "filters": {
            "symbol": symbol,
            "fold_index": fold_index,
            "regime_label": regime_label,
        },
        "filter_funnel": funnel_rows,
        "score_quantiles": quantile_rows,
        "label_diagnostics": label_diagnostics,
        "cost_gating_diagnostics": cost_diagnostics,
        "honesty_flags": honesty_flags,
        "output_files": {
            "oof_signal_diagnostics_json": str(json_path),
            "oof_score_quantiles_csv": str(quantiles_path),
            "oof_threshold_crossing_counts_csv": str(crossings_path),
            "oof_filter_funnel_csv": str(funnel_path),
            "oof_signal_diagnostics_md": str(markdown_path),
        },
    }
    write_json_artifact(json_path, summary)
    write_csv_artifact(quantiles_path, quantile_rows)
    write_csv_artifact(crossings_path, crossing_rows)
    write_csv_artifact(funnel_path, funnel_rows)
    markdown_path.write_text(_build_markdown(summary), encoding="utf-8")
    return make_json_safe(summary)


def _load_oof_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise ValueError(f"Completed run is missing oof_predictions.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        field_names = list(reader.fieldnames or ())
        if not field_names:
            raise ValueError(f"OOF predictions file has no header: {path}")
        return [dict(row) for row in reader], field_names


def _resolve_score_columns(
    field_names: Sequence[str],
    score_columns: Sequence[str] | None,
) -> list[str]:
    if score_columns is not None:
        missing = [column for column in score_columns if column not in field_names]
        if missing:
            raise ValueError(f"Requested score columns are missing from OOF predictions: {missing}")
        return list(score_columns)
    resolved = [
        column for column in DEFAULT_SCORE_COLUMN_CANDIDATES
        if column in field_names
    ]
    resolved.extend(
        column
        for column in field_names
        if column not in resolved and _looks_like_score_column(column)
    )
    return resolved


def _looks_like_score_column(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in ("prob", "score", "confidence"))


def _apply_optional_filters(
    rows: Sequence[Mapping[str, str]],
    *,
    symbol: str | None,
    fold_index: int | None,
    regime_label: str | None,
) -> list[dict[str, str]]:
    filtered_rows = [dict(row) for row in rows]
    if symbol is not None:
        filtered_rows = [row for row in filtered_rows if row.get("symbol") == symbol]
    if fold_index is not None:
        filtered_rows = [
            row for row in filtered_rows
            if _try_int(row.get("fold_index")) == fold_index
        ]
    if regime_label is not None:
        filtered_rows = [row for row in filtered_rows if row.get("regime_label") == regime_label]
    return filtered_rows


def _build_filter_funnel_rows(
    *,
    raw_rows: Sequence[Mapping[str, str]],
    field_names: Sequence[str],
    model_rows: Sequence[Mapping[str, str]],
    filtered_rows: Sequence[Mapping[str, str]],
    finite_primary_rows: Sequence[Mapping[str, str]],
    label_rows: Sequence[Mapping[str, str]],
    primary_score_column: str | None,
    filters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    # pylint: disable=too-many-arguments
    has_required_columns = {"model_name", "fold_index", "row_id"}.issubset(set(field_names))
    rows = [
        _funnel_row("raw_oof_rows_loaded", len(raw_rows), "all CSV data rows"),
        _funnel_row(
            "after_required_column_validation",
            len(raw_rows) if has_required_columns else 0,
            "requires model_name, fold_index, and row_id",
        ),
        _funnel_row(
            "after_candidate_model_filter",
            len(model_rows),
            "winner/candidate model filter",
        ),
        _funnel_row(
            "after_finite_score_filter",
            len(finite_primary_rows),
            f"finite {primary_score_column}" if primary_score_column else "no score column",
        ),
        _funnel_row(
            "after_label_availability_filter",
            len(label_rows),
            "any known label/target column",
        ),
    ]
    if any(value is not None for value in filters.values()):
        rows.append(
            _funnel_row(
                "after_fold_symbol_regime_filters",
                len(filtered_rows),
                f"filters={dict(filters)}",
            )
        )
    else:
        rows.append(
            _funnel_row(
                "after_fold_symbol_regime_filters",
                len(filtered_rows),
                "no optional fold/symbol/regime filters applied",
            )
        )
    rows.append(
        _funnel_row(
            "eligible_before_thresholding",
            len(finite_primary_rows),
            "finite primary score rows before threshold crossings",
        )
    )
    return rows


def _funnel_row(stage: str, row_count: int, note: str) -> dict[str, Any]:
    return {"stage": stage, "row_count": row_count, "note": note}


def _build_score_quantile_row(
    rows: Sequence[Mapping[str, str]],
    score_column: str,
) -> dict[str, Any]:
    values = [_try_float(row.get(score_column)) for row in rows]
    finite_values = sorted(value for value in values if value is not None)
    missing_count = len(values) - len(finite_values)
    row = {
        "score_column": score_column,
        "row_count": len(values),
        "finite_count": len(finite_values),
        "missing_or_non_finite_count": missing_count,
        "min": _min_or_none(finite_values),
        "max": _max_or_none(finite_values),
        "mean": _mean_or_none(finite_values),
        "std": _std_or_none(finite_values),
        "count_below_0": sum(1 for value in finite_values if value < 0.0),
        "count_between_0_and_1": sum(1 for value in finite_values if 0.0 <= value <= 1.0),
        "count_above_1": sum(1 for value in finite_values if value > 1.0),
    }
    row.update(
        {
            label: _quantile(finite_values, quantile)
            for label, quantile in QUANTILE_POINTS
        }
    )
    return row


def _build_threshold_crossing_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    score_columns: Sequence[str],
    thresholds: Sequence[float],
    scenario_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    crossing_rows: list[dict[str, Any]] = []
    for score_column in score_columns:
        crossing_rows.extend(
            _threshold_crossing_rows_for_group(
                rows,
                score_column=score_column,
                thresholds=thresholds,
                group_type="overall",
                group_value="all",
                scenario_specs=scenario_specs,
            )
        )
        for group_column in ("fold_index", "symbol", "regime_label"):
            if not any(group_column in row for row in rows):
                continue
            for group_value, group_rows in sorted(_group_by_column(rows, group_column).items()):
                crossing_rows.extend(
                    _threshold_crossing_rows_for_group(
                        group_rows,
                        score_column=score_column,
                        thresholds=thresholds,
                        group_type=group_column,
                        group_value=group_value,
                        scenario_specs=scenario_specs,
                    )
                )
    return crossing_rows


def _threshold_crossing_rows_for_group(
    rows: Sequence[Mapping[str, str]],
    *,
    score_column: str,
    thresholds: Sequence[float],
    group_type: str,
    group_value: str,
    scenario_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    # pylint: disable=too-many-arguments
    finite_rows = [
        row for row in rows
        if _try_float(row.get(score_column)) is not None
    ]
    output_rows = []
    for threshold in thresholds:
        crossing = [
            row for row in finite_rows
            if float(_try_float(row.get(score_column))) >= float(threshold)
        ]
        base = {
            "score_column": score_column,
            "threshold": threshold,
            "group_type": group_type,
            "group_value": group_value,
            "eligible_count": len(finite_rows),
            "crossing_count_before_cost": len(crossing),
            "coverage_before_cost": _safe_ratio(len(crossing), len(finite_rows)),
        }
        for scenario in scenario_specs:
            scenario_name = str(scenario["scenario_name"])
            cost_per_trade = float(scenario["cost_per_trade"])
            after_cost_count = sum(
                1
                for row in crossing
                if _net_after_cost(row, cost_per_trade) is not None
                and float(_net_after_cost(row, cost_per_trade)) > 0.0
            )
            output_rows.append(
                {
                    **base,
                    "cost_scenario": scenario_name,
                    "cost_per_trade": cost_per_trade,
                    "crossing_count_after_cost": after_cost_count,
                    "coverage_after_cost": _safe_ratio(after_cost_count, len(finite_rows)),
                }
            )
    return output_rows


def _build_label_diagnostics(
    rows: Sequence[Mapping[str, str]],
    primary_score_column: str | None,
) -> dict[str, Any]:
    y_true_values = [
        int(_try_int(row.get("y_true")))
        for row in rows
        if _try_int(row.get("y_true")) is not None
    ]
    return_values = [
        _try_float(row.get("future_return_3"))
        for row in rows
        if _try_float(row.get("future_return_3")) is not None
    ]
    paired_score_label = [
        (float(_try_float(row.get(primary_score_column))), float(_try_int(row.get("y_true"))))
        for row in rows
        if primary_score_column is not None
        and _try_float(row.get(primary_score_column)) is not None
        and _try_int(row.get("y_true")) is not None
    ]
    paired_score_return = [
        (
            float(_try_float(row.get(primary_score_column))),
            float(_try_float(row.get("future_return_3"))),
        )
        for row in rows
        if primary_score_column is not None
        and _try_float(row.get(primary_score_column)) is not None
        and _try_float(row.get("future_return_3")) is not None
    ]
    label_correlation = _pearson_from_pairs(paired_score_label)
    return_correlation = _pearson_from_pairs(paired_score_return)
    return {
        "classification_label_column": "y_true" if y_true_values else None,
        "classification_class_balance": _class_balance(y_true_values),
        "realized_return_column": "future_return_3" if return_values else None,
        "realized_return_summary": _numeric_summary(return_values),
        "score_label_correlation": label_correlation,
        "score_label_rank_correlation": _rank_correlation(paired_score_label),
        "score_return_correlation": return_correlation,
        "score_return_rank_correlation": _rank_correlation(paired_score_return),
        "possible_inverted_score": (
            (label_correlation is not None and label_correlation < -0.10)
            or (return_correlation is not None and return_correlation < -0.10)
        ),
    }


def _build_cost_diagnostics(
    rows: Sequence[Mapping[str, str]],
    *,
    score_column: str | None,
    thresholds: Sequence[float],
    scenario_specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if score_column is None:
        return {"score_column": None, "thresholds": []}
    threshold_rows = []
    finite_rows = [
        row for row in rows
        if _try_float(row.get(score_column)) is not None
    ]
    for threshold in thresholds:
        crossing = [
            row for row in finite_rows
            if float(_try_float(row.get(score_column))) >= float(threshold)
        ]
        scenario_counts = []
        for scenario in scenario_specs:
            cost_per_trade = float(scenario["cost_per_trade"])
            after_cost_count = sum(
                1
                for row in crossing
                if _net_after_cost(row, cost_per_trade) is not None
                and float(_net_after_cost(row, cost_per_trade)) > 0.0
            )
            scenario_counts.append(
                {
                    "scenario_name": scenario["scenario_name"],
                    "cost_per_trade": cost_per_trade,
                    "count_after_cost": after_cost_count,
                }
            )
        threshold_rows.append(
            {
                "threshold": threshold,
                "count_before_cost": len(crossing),
                "zero_before_cost": len(crossing) == 0,
                "scenario_counts": scenario_counts,
                "zero_after_all_cost_scenarios": all(
                    int(row["count_after_cost"]) == 0 for row in scenario_counts
                ),
            }
        )
    return {"score_column": score_column, "thresholds": threshold_rows}


def _build_honesty_flags(
    *,
    field_names: Sequence[str],
    quantile_rows: Sequence[Mapping[str, Any]],
    crossing_rows: Sequence[Mapping[str, Any]],
    funnel_rows: Sequence[Mapping[str, Any]],
    label_diagnostics: Mapping[str, Any],
    primary_score_column: str | None,
) -> list[str]:
    # pylint: disable=too-many-arguments
    flags: list[str] = []
    overall_crossings = [
        row for row in crossing_rows
        if row["group_type"] == "overall"
        and row["cost_scenario"] == "current_fee"
        and row["score_column"] == primary_score_column
    ]
    max_before_cost = (
        max(int(row["crossing_count_before_cost"]) for row in overall_crossings)
        if overall_crossings
        else 0
    )
    max_after_cost = (
        max(int(row["crossing_count_after_cost"]) for row in overall_crossings)
        if overall_crossings
        else 0
    )
    if overall_crossings and max_before_cost == 0:
        flags.append("NO_THRESHOLD_CROSSINGS")
    if overall_crossings and max_after_cost < 5:
        flags.append("LOW_TRADE_COUNT")
    if any(int(row["count_below_0"]) or int(row["count_above_1"]) for row in quantile_rows):
        flags.append("SCORE_OUT_OF_RANGE")
    if "y_true" not in field_names and "future_return_3" not in field_names:
        flags.append("MISSING_LABELS")
    if label_diagnostics.get("possible_inverted_score"):
        flags.append("POSSIBLE_INVERTED_SCORE")
    if _funnel_collapsed(funnel_rows):
        flags.append("FILTER_FUNNEL_COLLAPSE")
    primary_quantile = next(
        (row for row in quantile_rows if row["score_column"] == primary_score_column),
        None,
    )
    if primary_quantile and (
        int(primary_quantile["count_below_0"]) > 0
        or int(primary_quantile["count_above_1"]) > 0
    ):
        flags.append("UNCALIBRATED_OR_REGRESSION_SCORE")
    return sorted(dict.fromkeys(flags))


def _funnel_collapsed(funnel_rows: Sequence[Mapping[str, Any]]) -> bool:
    raw_count = _funnel_count(funnel_rows, "raw_oof_rows_loaded")
    eligible_count = _funnel_count(funnel_rows, "eligible_before_thresholding")
    return raw_count > 0 and eligible_count == 0


def _funnel_count(funnel_rows: Sequence[Mapping[str, Any]], stage: str) -> int:
    for row in funnel_rows:
        if row["stage"] == stage:
            return int(row["row_count"])
    return 0


def _build_markdown(summary: Mapping[str, Any]) -> str:
    primary_score = summary.get("primary_score_column") or "none"
    flags = ", ".join(summary["honesty_flags"]) or "none"
    quantile = next(
        (
            row for row in summary["score_quantiles"]
            if row["score_column"] == summary.get("primary_score_column")
        ),
        {},
    )
    cost_rows = summary["cost_gating_diagnostics"].get("thresholds", [])
    first_crossing = next(
        (row for row in cost_rows if int(row["count_before_cost"]) > 0),
        None,
    )
    lines = [
        "# M20 OOF Signal Diagnostics",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Model analyzed: `{summary['model_name']}`",
        f"- Primary score column: `{primary_score}`",
        f"- Honesty flags: `{flags}`",
        "",
        "## Score Distribution",
        "",
        (
            f"- min=`{_format_optional_number(quantile.get('min'))}`, "
            f"max=`{_format_optional_number(quantile.get('max'))}`, "
            f"p50=`{_format_optional_number(quantile.get('p50'))}`, "
            f"p95=`{_format_optional_number(quantile.get('p95'))}`"
        ),
        "",
        "## Threshold Diagnosis",
        "",
        (
            "- First threshold with crossings before cost: "
            f"`{first_crossing['threshold'] if first_crossing else 'none'}`"
        ),
        "- Zero trades before cost means the score distribution never reaches "
        "the tested thresholds.",
        "- Zero trades only after cost means score crossings exist but modeled "
        "economics remove them.",
        "",
        "## Label Diagnosis",
        "",
        (
            "- Class balance: "
            f"`{summary['label_diagnostics'].get('classification_class_balance')}`"
        ),
        (
            "- Possible inverted score: "
            f"`{summary['label_diagnostics'].get('possible_inverted_score')}`"
        ),
        "",
        "This diagnostic is research-only. It does not change runtime inference, "
        "promotion, registry authority, or paper/live execution.",
        "",
        "## Output Files",
        "",
    ]
    for label, path in summary["output_files"].items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _resolve_fee_rate_from_rows(
    summary_payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
) -> float:
    proxy_rows = [
        _FeeProxyRow(
            gross=float(gross),
            net=float(net),
        )
        for gross, net in (
            (
                _try_float(row.get("long_only_gross_value_proxy")),
                _try_float(row.get("long_only_net_value_proxy")),
            )
            for row in rows
        )
        if gross is not None and net is not None
    ]
    return resolve_fee_rate(summary_payload, proxy_rows)  # type: ignore[arg-type]


class _FeeProxyRow:  # pylint: disable=too-few-public-methods
    """Small adapter for shared fee-rate resolver."""

    def __init__(self, *, gross: float, net: float) -> None:
        self.long_only_gross_value_proxy = gross
        self.long_only_net_value_proxy = net


def _has_any_label(row: Mapping[str, str]) -> bool:
    return any(
        _try_float(row.get(column)) is not None
        for column in ("y_true", "future_return_3", "long_only_net_value_proxy")
    )


def _group_by_column(
    rows: Sequence[Mapping[str, str]],
    column: str,
) -> dict[str, list[Mapping[str, str]]]:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        value = row.get(column)
        if value is None or value == "":
            continue
        grouped.setdefault(str(value), []).append(row)
    return grouped


def _net_after_cost(row: Mapping[str, str], cost_per_trade: float) -> float | None:
    gross = _try_float(row.get("long_only_gross_value_proxy"))
    if gross is None:
        return None
    return gross - cost_per_trade


def _class_balance(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {}
    total = len(values)
    labels = sorted(set(values))
    return {
        str(label): {
            "count": sum(1 for value in values if value == label),
            "fraction": sum(1 for value in values if value == label) / total,
        }
        for label in labels
    }


def _numeric_summary(values: Sequence[float | None]) -> dict[str, Any]:
    finite_values = sorted(value for value in values if value is not None)
    return {
        "count": len(finite_values),
        "min": _min_or_none(finite_values),
        "max": _max_or_none(finite_values),
        "mean": _mean_or_none(finite_values),
        "std": _std_or_none(finite_values),
        "p50": _quantile(finite_values, 0.50),
    }


def _pearson_from_pairs(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x_value - left_mean) * (y_value - right_mean) for x_value, y_value in pairs)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_var * right_var)
    if denominator == 0.0:
        return None
    return numerator / denominator


def _rank_correlation(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left_ranks = _ranks([pair[0] for pair in pairs])
    right_ranks = _ranks([pair[1] for pair in pairs])
    return _pearson_from_pairs(list(zip(left_ranks, right_ranks)))


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        next_index = index + 1
        while next_index < len(indexed) and indexed[next_index][1] == indexed[index][1]:
            next_index += 1
        average_rank = (index + next_index - 1) / 2.0
        for ranked_index in range(index, next_index):
            ranks[indexed[ranked_index][0]] = average_rank
        index = next_index
    return ranks


def _try_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _try_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _min_or_none(values: Sequence[float]) -> float | None:
    return min(values) if values else None


def _max_or_none(values: Sequence[float]) -> float | None:
    return max(values) if values else None


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std_or_none(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
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


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _format_optional_number(value: Any) -> str:
    if value is None:
        return "none"
    return f"{float(value):.6f}"
