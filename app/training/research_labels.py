"""Research-only M20 trading-aware labels and diagnostics."""

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
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact

# pylint: disable=too-many-arguments,too-many-lines

DEFAULT_RESEARCH_LABEL_DIR_NAME = "research_labels"
DEFAULT_LABEL_HORIZON = 3
DEFAULT_FIXED_BARRIER_BPS = 20.0
DEFAULT_META_SIGNAL_THRESHOLD = 0.50
DEFAULT_MIN_POSITIVE_EVENT_RATE = 0.01
DEFAULT_MIN_TOTAL_EVENT_COUNT = 25
DEFAULT_HIGH_NEUTRAL_RATE = 0.95
DEFAULT_SCENARIO_SLIPPAGE_RATES = (0.0, 0.001)
DEFAULT_TRAINING_FRAME_SOURCE = "training_frame"
DEFAULT_VOL_SCALED_LABEL_DIR_NAME = "vol_scaled"
DEFAULT_FIXED_BPS_COMPARISON_DIR_NAME = "fixed_bps_comparison"


def build_triple_barrier_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    price_column: str = "close_price",
    high_column: str = "high_price",
    low_column: str = "low_price",
    volatility_column: str = "realized_vol_12",
    row_id_column: str = "row_id",
    profit_take_multiple: float = 1.0,
    stop_loss_multiple: float = 1.0,
    fixed_barrier_bps: float = DEFAULT_FIXED_BARRIER_BPS,
) -> list[dict[str, Any]]:
    """Build deterministic triple-barrier event labels from ordered OHLC rows."""
    # pylint: disable=too-many-locals
    if horizon < 1:
        raise ValueError("Triple-barrier horizon must be at least 1")
    labels: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        if source_index + horizon >= len(rows):
            continue
        entry_price = _positive_float(row, price_column)
        volatility = max(
            abs(float(row.get(volatility_column, 0.0) or 0.0)),
            float(fixed_barrier_bps) / 10000.0,
        )
        upper_barrier = entry_price * (1.0 + (volatility * profit_take_multiple))
        lower_barrier = entry_price * (1.0 - (volatility * stop_loss_multiple))
        event = _resolve_triple_barrier_event(
            rows=rows,
            source_index=source_index,
            horizon=horizon,
            high_column=high_column,
            low_column=low_column,
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
        )
        end_row = rows[event["event_end_index"]]
        labels.append(
            {
                "row_id": str(row.get(row_id_column, source_index)),
                "source_index": source_index,
                "event_end_index": event["event_end_index"],
                "event_end_row_id": str(end_row.get(row_id_column, event["event_end_index"])),
                "label": event["label"],
                "barrier_hit": event["barrier_hit"],
                "entry_price": entry_price,
                "upper_barrier": upper_barrier,
                "lower_barrier": lower_barrier,
                "horizon": horizon,
            }
        )
    return labels


def build_fee_exceedance_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    fee_rate: float,
    slippage_rate: float = 0.0,
    price_column: str = "close_price",
    high_column: str = "high_price",
    row_id_column: str = "row_id",
) -> list[dict[str, Any]]:
    """Label whether forward highs exceed a fee/slippage threshold within horizon."""
    if horizon < 1:
        raise ValueError("Fee-exceedance horizon must be at least 1")
    threshold = float(fee_rate) + float(slippage_rate)
    if threshold < 0.0:
        raise ValueError("Fee/slippage threshold must be non-negative")
    labels: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        if source_index + horizon >= len(rows):
            continue
        entry_price = _positive_float(row, price_column)
        future_rows = rows[source_index + 1 : source_index + horizon + 1]
        best_forward_return = max(
            (_positive_float(future_row, high_column) / entry_price) - 1.0
            for future_row in future_rows
        )
        labels.append(
            {
                "row_id": str(row.get(row_id_column, source_index)),
                "source_index": source_index,
                "label": int(best_forward_return >= threshold),
                "best_forward_return": best_forward_return,
                "threshold_return": threshold,
                "horizon": horizon,
            }
        )
    return labels


def build_return_proxy_triple_barrier_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    return_column: str = "future_return_3",
    row_id_column: str = "row_id",
    fixed_barrier_bps: float = DEFAULT_FIXED_BARRIER_BPS,
) -> list[dict[str, Any]]:
    """Build fixed-bps triple-barrier-like labels from realized forward returns."""
    if horizon < 1:
        raise ValueError("Return-proxy triple-barrier horizon must be at least 1")
    barrier_return = float(fixed_barrier_bps) / 10000.0
    labels: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        future_return = _try_float(row.get(return_column))
        if future_return is None:
            continue
        if future_return >= barrier_return:
            label = 1
            barrier_hit = "upper"
        elif future_return <= -barrier_return:
            label = -1
            barrier_hit = "lower"
        else:
            label = 0
            barrier_hit = "vertical"
        labels.append(
            {
                **_copy_slice_columns(row),
                "row_id": str(row.get(row_id_column, source_index)),
                "source_index": source_index,
                "event_end_index": source_index + horizon,
                "event_end_row_id": "",
                "label": label,
                "barrier_hit": barrier_hit,
                "future_return": future_return,
                "upper_barrier_return": barrier_return,
                "lower_barrier_return": -barrier_return,
                "horizon": horizon,
                "label_source": "return_proxy_fixed_bps",
            }
        )
    return labels


def build_return_fee_exceedance_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    fee_rate: float,
    slippage_rate: float = 0.0,
    return_column: str = "future_return_3",
    row_id_column: str = "row_id",
) -> list[dict[str, Any]]:
    """Label whether realized forward return exceeds fee plus slippage."""
    threshold = float(fee_rate) + float(slippage_rate)
    labels: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        future_return = _try_float(row.get(return_column))
        if future_return is None:
            continue
        labels.append(
            {
                **_copy_slice_columns(row),
                "row_id": str(row.get(row_id_column, source_index)),
                "source_index": source_index,
                "label": int(future_return > threshold),
                "future_return": future_return,
                "threshold_return": threshold,
                "fee_rate": float(fee_rate),
                "slippage_rate": float(slippage_rate),
                "horizon": horizon,
            }
        )
    return labels


def build_incumbent_meta_labels(
    signal_rows: Sequence[Mapping[str, Any]],
    *,
    signal_threshold: float = 0.50,
    probability_column: str = "prob_up",
    net_value_column: str = "long_only_net_value_proxy",
    row_id_column: str = "row_id",
) -> list[dict[str, Any]]:
    """Label whether an incumbent BUY signal should have been taken after costs."""
    labels: list[dict[str, Any]] = []
    for source_index, row in enumerate(signal_rows):
        if probability_column not in row:
            raise ValueError(f"Incumbent signal row is missing {probability_column}: {row}")
        if net_value_column not in row:
            raise ValueError(f"Incumbent signal row is missing {net_value_column}: {row}")
        probability = float(row[probability_column])
        base_signal = probability >= float(signal_threshold)
        net_value = float(row[net_value_column])
        labels.append(
            {
                "row_id": str(row.get(row_id_column, source_index)),
                "source_index": source_index,
                "base_signal": base_signal,
                "meta_label": int(base_signal and net_value > 0.0),
                "probability": probability,
                "net_value": net_value,
                "signal_threshold": float(signal_threshold),
            }
        )
    return labels


def build_label_diagnostics(
    label_rows: Sequence[Mapping[str, Any]],
    *,
    label_column: str = "label",
    skipped_count: int = 0,
) -> dict[str, Any]:
    """Summarize research-label distribution for artifact diagnostics."""
    counter = Counter(str(row.get(label_column)) for row in label_rows)
    total_count = len(label_rows)
    return {
        "label_column": label_column,
        "total_count": total_count,
        "skipped_count": int(skipped_count),
        "label_distribution": dict(sorted(counter.items())),
        "class_balance": {
            label: count / total_count if total_count else 0.0
            for label, count in sorted(counter.items())
        },
    }


def write_label_diagnostics(path: Path, diagnostics: Mapping[str, Any]) -> None:
    """Persist research-label diagnostics as deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(dict(diagnostics)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_completed_run_research_labels(
    *,
    run_dir: Path | None,
    model_name: str | None = None,
    output_dir_name: str = DEFAULT_RESEARCH_LABEL_DIR_NAME,
    horizon: int = DEFAULT_LABEL_HORIZON,
    fixed_barrier_bps: float = DEFAULT_FIXED_BARRIER_BPS,
    meta_signal_threshold: float = DEFAULT_META_SIGNAL_THRESHOLD,
    scenario_slippage_rates: Sequence[float] = DEFAULT_SCENARIO_SLIPPAGE_RATES,
) -> dict[str, Any]:
    """Generate deterministic research-only M20 label artifacts for a completed run."""
    # pylint: disable=too-many-locals
    resolved_run_dir = resolve_completed_run_dir(run_dir)
    summary_payload = _load_completed_summary(resolved_run_dir)
    resolved_model_name = model_name or _resolve_completed_winner_model_name(summary_payload)
    raw_rows, field_names = _load_oof_rows(resolved_run_dir / "oof_predictions.csv")
    model_rows = [
        row for row in raw_rows
        if str(row.get("model_name", "")) == resolved_model_name
    ]
    ordered_rows = sorted(
        model_rows,
        key=lambda row: (
            str(row.get("symbol", "")),
            int(float(row.get("fold_index", 0) or 0)),
            str(row.get("interval_begin", "")),
            str(row.get("row_id", "")),
        ),
    )
    warnings: list[str] = []
    has_price_columns = {"close_price", "high_price", "low_price"}.issubset(field_names)
    has_return_column = "future_return_3" in field_names
    has_volatility_column = any(
        column in field_names
        for column in ("realized_vol_12", "volatility")
    )
    if not has_price_columns and not has_return_column:
        warnings.append("MISSING_PRICE_OR_RETURN_COLUMN")
    if not has_volatility_column:
        warnings.append("MISSING_VOLATILITY_COLUMN_USING_FIXED_BPS")

    label_dir = resolved_run_dir / output_dir_name
    label_dir.mkdir(parents=True, exist_ok=True)
    triple_rows, triple_skipped = _build_triple_barrier_artifact_rows(
        ordered_rows,
        has_price_columns=has_price_columns,
        has_return_column=has_return_column,
        horizon=horizon,
        fixed_barrier_bps=fixed_barrier_bps,
    )
    fee_scenarios = _build_cost_scenario_specs(
        fee_rate=_resolve_fee_rate(summary_payload, ordered_rows),
        slippage_rates=scenario_slippage_rates,
    )
    fee_rows = _build_fee_exceedance_artifact_rows(
        ordered_rows,
        fee_scenarios=fee_scenarios,
        has_return_column=has_return_column,
        horizon=horizon,
    )
    meta_rows = _build_meta_label_artifact_rows(
        ordered_rows,
        meta_signal_threshold=meta_signal_threshold,
        warnings=warnings,
    )
    if triple_skipped > 0:
        warnings.append("INSUFFICIENT_FUTURE_HORIZON_ROWS")

    diagnostics = _build_research_label_diagnostics(
        triple_rows=triple_rows,
        fee_rows=fee_rows,
        meta_rows=meta_rows,
        warnings=warnings,
        skipped_horizon_count=triple_skipped,
        source_row_count=len(ordered_rows),
    )
    output_files = {
        "research_labels_manifest_json": str(label_dir / "research_labels_manifest.json"),
        "triple_barrier_labels_csv": str(label_dir / "triple_barrier_labels.csv"),
        "fee_exceedance_labels_csv": str(label_dir / "fee_exceedance_labels.csv"),
        "label_diagnostics_json": str(label_dir / "label_diagnostics.json"),
        "label_diagnostics_md": str(label_dir / "label_diagnostics.md"),
        "label_distribution_by_slice_csv": str(label_dir / "label_distribution_by_slice.csv"),
    }
    if meta_rows:
        output_files["incumbent_meta_labels_csv"] = str(label_dir / "incumbent_meta_labels.csv")

    manifest = {
        "run_dir": str(resolved_run_dir),
        "label_dir": str(label_dir),
        "model_name": resolved_model_name,
        "source": "completed_m20_oof_predictions",
        "source_row_count": len(raw_rows),
        "winner_model_row_count": len(ordered_rows),
        "horizon": horizon,
        "fixed_barrier_bps": fixed_barrier_bps,
        "meta_signal_threshold": meta_signal_threshold,
        "price_columns_available": has_price_columns,
        "return_column_available": has_return_column,
        "volatility_column_available": has_volatility_column,
        "fee_scenarios": fee_scenarios,
        "honesty_flags": diagnostics["honesty_flags"],
        "output_files": output_files,
        "runtime_effect": "none_research_only",
    }
    write_json_artifact(Path(output_files["research_labels_manifest_json"]), manifest)
    write_csv_artifact(Path(output_files["triple_barrier_labels_csv"]), triple_rows)
    write_csv_artifact(Path(output_files["fee_exceedance_labels_csv"]), fee_rows)
    if meta_rows:
        write_csv_artifact(Path(output_files["incumbent_meta_labels_csv"]), meta_rows)
    write_json_artifact(Path(output_files["label_diagnostics_json"]), diagnostics)
    write_csv_artifact(
        Path(output_files["label_distribution_by_slice_csv"]),
        diagnostics["label_distribution_by_slice"],
    )
    Path(output_files["label_diagnostics_md"]).write_text(
        _build_label_diagnostics_markdown(manifest, diagnostics),
        encoding="utf-8",
    )
    return make_json_safe({**manifest, "diagnostics": diagnostics})


def generate_training_frame_research_labels(
    *,
    run_dir: Path,
    horizon: int = DEFAULT_LABEL_HORIZON,
    fixed_barrier_bps: float = DEFAULT_FIXED_BARRIER_BPS,
    price_column: str = "close_price",
    volatility_column: str = "realized_vol_12",
    scenario_slippage_rates: Sequence[float] = DEFAULT_SCENARIO_SLIPPAGE_RATES,
    write_fixed_bps_comparison: bool = True,
) -> dict[str, Any]:
    """Generate research-only labels from an exported M20 training_frame artifact."""
    # pylint: disable=too-many-locals
    resolved_run_dir = Path(run_dir).resolve()
    training_frame_dir = resolved_run_dir / DEFAULT_TRAINING_FRAME_SOURCE
    rows, field_names = _load_training_frame_rows(training_frame_dir)
    ordered_rows = _order_training_frame_rows(rows)
    label_dir = (
        resolved_run_dir / DEFAULT_RESEARCH_LABEL_DIR_NAME / DEFAULT_VOL_SCALED_LABEL_DIR_NAME
    )
    label_dir.mkdir(parents=True, exist_ok=True)
    warnings = [
        "RESEARCH_ONLY_LABELS",
        "SOURCE_TRAINING_FRAME",
        "VOLATILITY_SCALED_LABELS_GENERATED",
        "ROW_ALIGNED_REALIZED_VOLATILITY_USED",
        "FIXED_BPS_COMPARISON_RETAINED",
        "META_LABEL_NOT_APPLICABLE_NO_OOF_SIGNALS",
        "NO_RUNTIME_EFFECT",
        "NO_PROMOTION_EFFECT",
        "NO_REGISTRY_WRITE",
        "LABELS_NOT_RUNTIME_READY",
    ]
    if volatility_column not in field_names:
        raise ValueError(
            f"Training-frame source is missing required volatility column "
            f"{volatility_column}: {training_frame_dir}"
        )
    if price_column not in field_names:
        raise ValueError(
            f"Training-frame source is missing required price column "
            f"{price_column}: {training_frame_dir}"
        )
    vol_missing_count = sum(_try_float(row.get(volatility_column)) is None for row in ordered_rows)
    vol_non_positive_count = sum(
        (value := _try_float(row.get(volatility_column))) is not None and value <= 0.0
        for row in ordered_rows
    )
    if vol_missing_count or vol_non_positive_count:
        warnings.append("VOLATILITY_HAS_MISSING_OR_NON_POSITIVE_VALUES")
    usable_rows = [
        row for row in ordered_rows
        if _try_float(row.get(volatility_column)) is not None
        and (_try_float(row.get(volatility_column)) or 0.0) > 0.0
    ]
    triple_rows, skipped_count = _build_training_frame_vol_scaled_triple_rows(
        usable_rows,
        horizon=horizon,
        price_column=price_column,
        volatility_column=volatility_column,
    )
    fee_rate = _resolve_training_frame_fee_rate(resolved_run_dir)
    fee_scenarios = _build_cost_scenario_specs(
        fee_rate=fee_rate,
        slippage_rates=scenario_slippage_rates,
    )
    fee_rows = _build_training_frame_fee_rows(
        usable_rows,
        horizon=horizon,
        price_column=price_column,
        fee_scenarios=fee_scenarios,
    )
    if skipped_count:
        warnings.append("INSUFFICIENT_FUTURE_HORIZON_ROWS")
    diagnostics = _build_research_label_diagnostics(
        triple_rows=triple_rows,
        fee_rows=fee_rows,
        meta_rows=[],
        warnings=warnings,
        skipped_horizon_count=skipped_count,
        source_row_count=len(ordered_rows),
    )
    diagnostics.update(
        {
            "source": DEFAULT_TRAINING_FRAME_SOURCE,
            "rows_labeled": len(triple_rows),
            "timestamp_min": _min_row_value(ordered_rows, "interval_begin"),
            "timestamp_max": _max_row_value(ordered_rows, "interval_begin"),
            "volatility_column_used": volatility_column,
            "volatility_missing_count": vol_missing_count,
            "volatility_non_positive_count": vol_non_positive_count,
            "price_column_used": price_column,
        }
    )
    fixed_comparison = {}
    if write_fixed_bps_comparison:
        fixed_comparison = _write_fixed_bps_comparison(
            run_dir=resolved_run_dir,
            rows=usable_rows,
            vol_scaled_rows=triple_rows,
            horizon=horizon,
            price_column=price_column,
            fixed_barrier_bps=fixed_barrier_bps,
        )
        diagnostics["fixed_bps_comparison"] = fixed_comparison["comparison"]
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
        "run_dir": str(resolved_run_dir),
        "label_dir": str(label_dir),
        "model_name": "not_applicable_export_only",
        "source": DEFAULT_TRAINING_FRAME_SOURCE,
        "source_row_count": len(ordered_rows),
        "rows_labeled": len(triple_rows),
        "horizon": horizon,
        "price_column": price_column,
        "volatility_column": volatility_column,
        "volatility_source": "row_aligned_training_frame_realized_volatility",
        "fee_scenarios": fee_scenarios,
        "fixed_bps_comparison_retained": bool(write_fixed_bps_comparison),
        "honesty_flags": diagnostics["honesty_flags"],
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
            "fixed_bps_comparison": fixed_comparison,
        }
    )


def _resolve_triple_barrier_event(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_index: int,
    horizon: int,
    high_column: str,
    low_column: str,
    upper_barrier: float,
    lower_barrier: float,
) -> dict[str, Any]:
    for future_index in range(source_index + 1, source_index + horizon + 1):
        future_row = rows[future_index]
        if _positive_float(future_row, high_column) >= upper_barrier:
            return {"event_end_index": future_index, "label": 1, "barrier_hit": "upper"}
        if _positive_float(future_row, low_column) <= lower_barrier:
            return {"event_end_index": future_index, "label": -1, "barrier_hit": "lower"}
    return {
        "event_end_index": source_index + horizon,
        "label": 0,
        "barrier_hit": "vertical",
    }


def _load_training_frame_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    features_path = path / "m20_training_frame_features.csv"
    keys_path = path / "m20_training_frame_keys.csv"
    if not features_path.exists():
        raise ValueError(f"Training-frame source is missing features CSV: {features_path}")
    if not keys_path.exists():
        raise ValueError(f"Training-frame source is missing keys CSV: {keys_path}")
    with features_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        field_names = list(reader.fieldnames or ())
        if not field_names:
            raise ValueError(f"Training-frame features file has no header: {features_path}")
        rows = [dict(row) for row in reader]
    return rows, field_names


def _order_training_frame_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (
                str(row.get("symbol", "")),
                str(row.get("interval_begin", row.get("timestamp", ""))),
                int(float(row.get("fold_index", row.get("fold", 0)) or 0)),
                str(row.get("row_id", "")),
            ),
        )
    ]


def _build_training_frame_vol_scaled_triple_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    price_column: str,
    volatility_column: str,
) -> tuple[list[dict[str, Any]], int]:
    # pylint: disable=too-many-locals
    output_rows: list[dict[str, Any]] = []
    skipped = 0
    for group_rows in _group_ordered_rows_by_symbol(rows).values():
        if len(group_rows) <= horizon:
            skipped += len(group_rows)
            continue
        skipped += horizon
        for source_index, row in enumerate(group_rows[:-horizon]):
            entry_price = _positive_float(row, price_column)
            future_row = group_rows[source_index + horizon]
            future_price = _positive_float(future_row, price_column)
            future_return = (future_price / entry_price) - 1.0
            volatility = float(row[volatility_column])
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
                    "event_end_index": source_index + horizon,
                    "event_end_row_id": str(future_row.get("row_id", "")),
                    "label": label,
                    "barrier_hit": barrier_hit,
                    "entry_price": entry_price,
                    "event_end_price": future_price,
                    "future_return": future_return,
                    "volatility": volatility,
                    "upper_barrier_return": volatility,
                    "lower_barrier_return": -volatility,
                    "horizon": horizon,
                    "label_source": "training_frame_volatility_scaled",
                }
            )
    return output_rows, skipped


def _build_training_frame_fee_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    price_column: str,
    fee_scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for group_rows in _group_ordered_rows_by_symbol(rows).values():
        if len(group_rows) <= horizon:
            continue
        for source_index, row in enumerate(group_rows[:-horizon]):
            entry_price = _positive_float(row, price_column)
            future_price = _positive_float(group_rows[source_index + horizon], price_column)
            future_return = (future_price / entry_price) - 1.0
            for scenario in fee_scenarios:
                threshold = float(scenario["cost_per_trade"])
                output_rows.append(
                    {
                        **_copy_slice_columns(row),
                        "row_id": str(row.get("row_id", source_index)),
                        "source_index": source_index,
                        "label": int(future_return > threshold),
                        "future_return": future_return,
                        "threshold_return": threshold,
                        "scenario_name": scenario["scenario_name"],
                        "cost_per_trade": threshold,
                        "horizon": horizon,
                    }
                )
    return output_rows


def _write_fixed_bps_comparison(
    *,
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    vol_scaled_rows: Sequence[Mapping[str, Any]],
    horizon: int,
    price_column: str,
    fixed_barrier_bps: float,
) -> dict[str, Any]:
    comparison_dir = (
        run_dir / DEFAULT_RESEARCH_LABEL_DIR_NAME / DEFAULT_FIXED_BPS_COMPARISON_DIR_NAME
    )
    comparison_dir.mkdir(parents=True, exist_ok=True)
    fixed_rows, _ = _build_training_frame_fixed_bps_rows(
        rows,
        horizon=horizon,
        price_column=price_column,
        fixed_barrier_bps=fixed_barrier_bps,
    )
    comparison = _compare_label_agreement(fixed_rows, vol_scaled_rows)
    diagnostics = _build_research_label_diagnostics(
        triple_rows=fixed_rows,
        fee_rows=[],
        meta_rows=[],
        warnings=[
            "RESEARCH_ONLY_LABELS",
            "SOURCE_TRAINING_FRAME",
            "FIXED_BPS_COMPARISON_RETAINED",
            "NO_RUNTIME_EFFECT",
            "NO_PROMOTION_EFFECT",
            "NO_REGISTRY_WRITE",
        ],
        skipped_horizon_count=len(rows) - len(fixed_rows),
        source_row_count=len(rows),
    )
    files = {
        "triple_barrier_labels_fixed_bps_csv": str(
            comparison_dir / "triple_barrier_labels_fixed_bps.csv"
        ),
        "label_diagnostics_fixed_bps_json": str(
            comparison_dir / "label_diagnostics_fixed_bps.json"
        ),
        "fixed_vs_vol_scaled_comparison_json": str(
            comparison_dir / "fixed_vs_vol_scaled_comparison.json"
        ),
        "fixed_vs_vol_scaled_comparison_md": str(
            comparison_dir / "fixed_vs_vol_scaled_comparison.md"
        ),
    }
    write_csv_artifact(Path(files["triple_barrier_labels_fixed_bps_csv"]), fixed_rows)
    write_json_artifact(Path(files["label_diagnostics_fixed_bps_json"]), diagnostics)
    write_json_artifact(Path(files["fixed_vs_vol_scaled_comparison_json"]), comparison)
    Path(files["fixed_vs_vol_scaled_comparison_md"]).write_text(
        "\n".join(
            [
                "# M20 Fixed-BPS vs Volatility-Scaled Label Comparison",
                "",
                f"- Paired rows: `{comparison['paired_row_count']}`",
                f"- Agreement rate: `{float(comparison['label_agreement_rate']):.6f}`",
                "",
                "Fixed-bps labels are retained only as a research comparison artifact.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"comparison_dir": str(comparison_dir), "comparison": comparison, "output_files": files}


def _build_training_frame_fixed_bps_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    price_column: str,
    fixed_barrier_bps: float,
) -> tuple[list[dict[str, Any]], int]:
    barrier = float(fixed_barrier_bps) / 10000.0
    output_rows: list[dict[str, Any]] = []
    skipped = 0
    for group_rows in _group_ordered_rows_by_symbol(rows).values():
        if len(group_rows) <= horizon:
            skipped += len(group_rows)
            continue
        skipped += horizon
        for source_index, row in enumerate(group_rows[:-horizon]):
            entry_price = _positive_float(row, price_column)
            future_price = _positive_float(group_rows[source_index + horizon], price_column)
            future_return = (future_price / entry_price) - 1.0
            if future_return >= barrier:
                label = 1
                barrier_hit = "upper"
            elif future_return <= -barrier:
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
                    "upper_barrier_return": barrier,
                    "lower_barrier_return": -barrier,
                    "horizon": horizon,
                    "label_source": "training_frame_fixed_bps_comparison",
                }
            )
    return output_rows, skipped


def _compare_label_agreement(
    fixed_rows: Sequence[Mapping[str, Any]],
    vol_scaled_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fixed_by_id = {str(row.get("row_id")): row for row in fixed_rows}
    paired = [
        (fixed_by_id[str(row.get("row_id"))], row)
        for row in vol_scaled_rows
        if str(row.get("row_id")) in fixed_by_id
    ]
    agreement_count = sum(
        1 for fixed, vol in paired
        if int(fixed.get("label", 999)) == int(vol.get("label", 999))
    )
    return {
        "paired_row_count": len(paired),
        "agreement_count": agreement_count,
        "label_agreement_rate": agreement_count / len(paired) if paired else 0.0,
        "fixed_bps_label_counts": dict(
            sorted(Counter(str(row.get("label")) for row in fixed_rows).items())
        ),
        "vol_scaled_label_counts": dict(
            sorted(Counter(str(row.get("label")) for row in vol_scaled_rows).items())
        ),
    }


def _group_ordered_rows_by_symbol(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("symbol", "all")), []).append(row)
    return grouped


def _resolve_training_frame_fee_rate(run_dir: Path) -> float:
    for filename in ("run_config.json", "summary.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            if "round_trip_fee_bps" in payload:
                return float(payload["round_trip_fee_bps"]) / 10000.0
            economics_contract = payload.get("economics_contract")
            if isinstance(economics_contract, Mapping) and "fee_rate" in economics_contract:
                return float(economics_contract["fee_rate"])
    return 0.0


def _min_row_value(rows: Sequence[Mapping[str, Any]], column: str) -> str:
    values = sorted(str(row.get(column, "")) for row in rows if row.get(column, "") != "")
    return values[0] if values else ""


def _max_row_value(rows: Sequence[Mapping[str, Any]], column: str) -> str:
    values = sorted(str(row.get(column, "")) for row in rows if row.get(column, "") != "")
    return values[-1] if values else ""


def _load_oof_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise ValueError(f"Completed run is missing oof_predictions.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        field_names = list(reader.fieldnames or ())
        if not field_names:
            raise ValueError(f"OOF predictions file has no header: {path}")
        return [dict(row) for row in reader], field_names


def _build_triple_barrier_artifact_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    has_price_columns: bool,
    has_return_column: bool,
    horizon: int,
    fixed_barrier_bps: float,
) -> tuple[list[dict[str, Any]], int]:
    if has_price_columns:
        eligible_rows, skipped = _rows_with_future_horizon(rows, horizon)
        labels = build_triple_barrier_labels(
            eligible_rows,
            horizon=horizon,
            fixed_barrier_bps=fixed_barrier_bps,
        )
        return [
            _with_slice_columns(label, eligible_rows[int(label["source_index"])])
            for label in labels
        ], skipped
    if has_return_column:
        labels = build_return_proxy_triple_barrier_labels(
            rows,
            horizon=horizon,
            fixed_barrier_bps=fixed_barrier_bps,
        )
        return labels, 0
    return [], len(rows)


def _build_fee_exceedance_artifact_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fee_scenarios: Sequence[Mapping[str, Any]],
    has_return_column: bool,
    horizon: int,
) -> list[dict[str, Any]]:
    if not has_return_column:
        return []
    output_rows: list[dict[str, Any]] = []
    for scenario in fee_scenarios:
        output_rows.extend(
            {
                **row,
                "scenario_name": scenario["scenario_name"],
                "cost_per_trade": scenario["cost_per_trade"],
            }
            for row in build_return_fee_exceedance_labels(
                rows,
                horizon=horizon,
                fee_rate=float(scenario["cost_per_trade"]),
                slippage_rate=0.0,
            )
        )
    return output_rows


def _build_meta_label_artifact_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    meta_signal_threshold: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    required_columns = {"prob_up", "long_only_net_value_proxy"}
    if not required_columns.issubset(set(rows[0].keys()) if rows else set()):
        warnings.append("MISSING_INCUMBENT_SIGNAL")
        return []
    return [
        {**_copy_slice_columns(source_row), **label}
        for source_row, label in zip(
            rows,
            build_incumbent_meta_labels(
                rows,
                signal_threshold=meta_signal_threshold,
            ),
        )
    ]


def _build_research_label_diagnostics(
    *,
    triple_rows: Sequence[Mapping[str, Any]],
    fee_rows: Sequence[Mapping[str, Any]],
    meta_rows: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
    skipped_horizon_count: int,
    source_row_count: int,
) -> dict[str, Any]:
    triple_distribution = _label_distribution(triple_rows, "label")
    fee_distribution = _label_distribution(fee_rows, "label")
    meta_distribution = _label_distribution(meta_rows, "meta_label")
    slice_rows = _build_slice_distribution_rows(triple_rows)
    honesty_flags = _build_research_label_honesty_flags(
        triple_distribution=triple_distribution,
        fee_distribution=fee_distribution,
        meta_distribution=meta_distribution,
        warnings=warnings,
        skipped_horizon_count=skipped_horizon_count,
        source_row_count=source_row_count,
    )
    return {
        "source_row_count": source_row_count,
        "skipped_due_to_insufficient_future_horizon": skipped_horizon_count,
        "missing_or_invalid_row_count": max(source_row_count - len(triple_rows), 0),
        "triple_barrier": {
            **triple_distribution,
            "neutral_hold_rate": _label_rate(triple_rows, "label", 0),
            "positive_event_rate": _label_rate(triple_rows, "label", 1),
            "negative_event_rate": _label_rate(triple_rows, "label", -1),
        },
        "fee_exceedance": {
            **fee_distribution,
            "after_cost_positive_event_rate": _label_rate(fee_rows, "label", 1),
        },
        "incumbent_meta": meta_distribution,
        "label_distribution_by_slice": slice_rows,
        "honesty_flags": honesty_flags,
        "warnings": sorted(dict.fromkeys(warnings)),
    }


def _build_research_label_honesty_flags(
    *,
    triple_distribution: Mapping[str, Any],
    fee_distribution: Mapping[str, Any],
    meta_distribution: Mapping[str, Any],
    warnings: Sequence[str],
    skipped_horizon_count: int,
    source_row_count: int,
) -> list[str]:
    flags = list(warnings)
    positive_count = int(triple_distribution["label_counts"].get("1", 0))
    total_count = int(triple_distribution["total_count"])
    positive_rate = positive_count / total_count if total_count else 0.0
    neutral_rate = (
        int(triple_distribution["label_counts"].get("0", 0)) / total_count
        if total_count
        else 0.0
    )
    fee_positive_count = int(fee_distribution["label_counts"].get("1", 0))
    if positive_rate < DEFAULT_MIN_POSITIVE_EVENT_RATE:
        flags.append("LOW_POSITIVE_EVENT_RATE")
    if positive_count < DEFAULT_MIN_TOTAL_EVENT_COUNT:
        flags.append("LOW_TOTAL_EVENT_COUNT")
    if neutral_rate >= DEFAULT_HIGH_NEUTRAL_RATE:
        flags.append("HIGH_NEUTRAL_RATE")
    if fee_positive_count == 0 and positive_count > 0:
        flags.append("COSTS_ELIMINATE_POSITIVE_CLASS")
    if skipped_horizon_count > 0 or total_count < source_row_count:
        flags.append("INSUFFICIENT_FUTURE_HORIZON_ROWS")
    if (
        not meta_distribution.get("total_count")
        and "META_LABEL_NOT_APPLICABLE_NO_OOF_SIGNALS" not in warnings
    ):
        flags.append("MISSING_INCUMBENT_SIGNAL")
    flags.append("LABELS_NOT_TRAINING_READY")
    return sorted(dict.fromkeys(flags))


def _build_slice_distribution_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for slice_column in ("symbol", "fold_index", "regime_label"):
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            value = row.get(slice_column)
            if value is None or value == "":
                continue
            grouped.setdefault(str(value), []).append(row)
        for slice_value, slice_rows in sorted(grouped.items()):
            distribution = _label_distribution(slice_rows, "label")
            output_rows.append(
                {
                    "slice_column": slice_column,
                    "slice_value": slice_value,
                    "total_count": distribution["total_count"],
                    "positive_event_rate": _label_rate(slice_rows, "label", 1),
                    "negative_event_rate": _label_rate(slice_rows, "label", -1),
                    "neutral_hold_rate": _label_rate(slice_rows, "label", 0),
                    "positive_count": distribution["label_counts"].get("1", 0),
                    "negative_count": distribution["label_counts"].get("-1", 0),
                    "neutral_count": distribution["label_counts"].get("0", 0),
                }
            )
    return output_rows


def _label_distribution(
    rows: Sequence[Mapping[str, Any]],
    label_column: str,
) -> dict[str, Any]:
    counter = Counter(str(row.get(label_column)) for row in rows)
    total_count = len(rows)
    return {
        "label_column": label_column,
        "total_count": total_count,
        "label_counts": dict(sorted(counter.items())),
        "class_balance": {
            label: count / total_count if total_count else 0.0
            for label, count in sorted(counter.items())
        },
    }


def _label_rate(
    rows: Sequence[Mapping[str, Any]],
    label_column: str,
    label_value: int,
) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if int(row.get(label_column, 999)) == label_value) / len(rows)


def _build_label_diagnostics_markdown(
    manifest: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    triple = diagnostics["triple_barrier"]
    fee = diagnostics["fee_exceedance"]
    flags = ", ".join(diagnostics["honesty_flags"]) or "none"
    if "COSTS_ELIMINATE_POSITIVE_CLASS" in diagnostics["honesty_flags"]:
        next_batch = "candidate rejection or revised label design before classifier training"
    elif float(fee.get("after_cost_positive_event_rate", 0.0)) > 0.01:
        next_batch = "fee-exceedance classifier training research"
    elif float(triple.get("positive_event_rate", 0.0)) > 0.01:
        next_batch = "triple-barrier model training research"
    else:
        next_batch = "full M20 candidate rejection or incumbent-only abstention review"
    return "\n".join(
        [
            "# M20 Research Label Diagnostics",
            "",
            f"- Run directory: `{manifest['run_dir']}`",
            f"- Model analyzed: `{manifest['model_name']}`",
            f"- Label source: `{manifest['source']}`",
            f"- Honesty flags: `{flags}`",
            "",
            "## Triple-Barrier Labels",
            "",
            f"- Total labels: `{triple['total_count']}`",
            f"- Positive-event rate: `{float(triple['positive_event_rate']):.6f}`",
            f"- Negative-event rate: `{float(triple['negative_event_rate']):.6f}`",
            f"- Neutral/HOLD rate: `{float(triple['neutral_hold_rate']):.6f}`",
            "",
            "## Fee-Exceedance Labels",
            "",
            f"- Total labels: `{fee['total_count']}`",
            "- After-cost positive-event rate: "
            f"`{float(fee['after_cost_positive_event_rate']):.6f}`",
            "",
            "## Operator Interpretation",
            "",
            f"- Suggested next research branch: `{next_batch}`",
            "- These are research-only artifacts, not runtime training labels.",
            "- No runtime inference, registry authority, promotion, execution, "
            "thresholds, or roster behavior changed.",
            "",
        ]
    )


def _rows_with_future_horizon(
    rows: Sequence[Mapping[str, Any]],
    horizon: int,
) -> tuple[list[Mapping[str, Any]], int]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("symbol", "all")), []).append(row)
    eligible: list[Mapping[str, Any]] = []
    skipped = 0
    for group_rows in grouped.values():
        if len(group_rows) <= horizon:
            skipped += len(group_rows)
            continue
        eligible.extend(group_rows[:-horizon])
        skipped += horizon
    return eligible, skipped


def _resolve_fee_rate(
    summary_payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> float:
    economics_contract = summary_payload.get("economics_contract")
    if isinstance(economics_contract, Mapping) and "fee_rate" in economics_contract:
        return float(economics_contract["fee_rate"])
    deltas = [
        float(gross) - float(net)
        for gross, net in (
            (row.get("long_only_gross_value_proxy"), row.get("long_only_net_value_proxy"))
            for row in rows
        )
        if _try_float(gross) is not None and _try_float(net) is not None
    ]
    non_zero_deltas = [delta for delta in deltas if abs(delta) > 1e-12]
    return sorted(non_zero_deltas)[len(non_zero_deltas) // 2] if non_zero_deltas else 0.0


def _with_slice_columns(
    label: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> dict[str, Any]:
    return {**_copy_slice_columns(source_row), **dict(label)}


def _copy_slice_columns(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        column: row[column]
        for column in (
            "model_name",
            "fold_index",
            "symbol",
            "interval_begin",
            "as_of_time",
            "regime_label",
        )
        if column in row
    }


def _try_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _positive_float(row: Mapping[str, Any], column: str) -> float:
    if column not in row:
        raise ValueError(f"Research label row is missing {column}: {row}")
    value = float(row[column])
    if value <= 0.0:
        raise ValueError(f"Research label row has non-positive {column}: {value}")
    return value
