"""Research-only nested/held-out M20 rank-gate tuning."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_rank_gated_selector import (
    BASELINE_NAME,
    DISABLE_GAPS,
    SCENARIO_NAME,
    _condition_weights as load_condition_weights,  # pylint: disable=protected-access
    _key as row_key,  # pylint: disable=protected-access
    _quarter as month_to_quarter,  # pylint: disable=protected-access
    _read_csv_rows as read_csv_rows,  # pylint: disable=protected-access
    _to_float as to_float,  # pylint: disable=protected-access
)
from app.training.m20_strategy_selector_simulation import (
    _conditions as selector_conditions,  # pylint: disable=protected-access
)
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "rank_gate_nested_tuning"
K_GRID = (0.25, 0.5, 1.0, 2.0, 5.0)
CONDITION_MODES = (
    "condition_then_top_k",
    "top_k_with_n_conditions",
    "disable_gap_filtered_top_k",
)
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "NESTED_HELDOUT_TUNING",
    "VALIDATION_TUNED_TEST_LOCKED",
    "NOT_RUNTIME",
    "NOT_BACKTEST",
    "NOT_PROMOTABLE",
    "NO_REGISTRY_WRITE",
    "NO_PROFIT_CLAIM",
    "DISABLE_GAPS_UNTESTED",
)


def tune_m20_rank_gate_nested(
    *,
    original_run_dir: Path,
    confirmation_run_dir: Path,
) -> dict[str, Any]:
    """Tune rank-gate params on validation and evaluate locked params on test."""
    # pylint: disable=too-many-locals
    original = Path(original_run_dir).resolve()
    confirmation = Path(confirmation_run_dir).resolve()
    output_dir = original / "research_labels" / "vol_scaled" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_weights = load_condition_weights(original)
    validation_rows = _joined_rows(original, "validation", condition_weights)
    original_test_rows = _joined_rows(original, "test", condition_weights)
    confirmation_test_rows = _joined_rows(confirmation, "test", condition_weights)

    validation_grid = _validation_grid(validation_rows)
    selected = _select_params(validation_grid)
    original_test = _metric_for_params("original_test", original_test_rows, selected)
    confirmation_test = _metric_for_params(
        "confirmation_test",
        confirmation_test_rows,
        selected,
    )
    locked_metrics = [original_test]
    confirmation_metrics = [confirmation_test]
    previous = _previous_condition_then_top_one(original)
    stability = _stability(original_test, confirmation_test)
    gaps = _gap_rows("original_test", original_test_rows) + _gap_rows(
        "confirmation_test",
        confirmation_test_rows,
    )
    recommendation = _recommendation(original_test, confirmation_test, previous)
    output_files = _output_files(output_dir)

    manifest = {
        "original_run_dir": str(original),
        "confirmation_run_dir": str(confirmation),
        "output_dir": str(output_dir),
        "k_grid": list(K_GRID),
        "condition_modes": list(CONDITION_MODES),
        "selected_params": selected,
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "output_files": output_files,
    }
    report = {
        "output_dir": str(output_dir),
        "selected_params": selected,
        "validation_selected_metric": selected.get("validation_policy_name", ""),
        "original_test": original_test,
        "confirmation_test": confirmation_test,
        "previous_condition_then_top_1": previous,
        "recommendation": recommendation["recommendation"],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["validation_grid_csv"]), validation_grid)
    write_csv_artifact(Path(output_files["locked_test_metrics_csv"]), locked_metrics)
    write_csv_artifact(Path(output_files["confirmation_metrics_csv"]), confirmation_metrics)
    write_csv_artifact(Path(output_files["stability_csv"]), [stability])
    write_csv_artifact(Path(output_files["disable_gap_exposure_csv"]), gaps)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "validation_grid": validation_grid,
            "locked_test_metrics": locked_metrics,
            "confirmation_metrics": confirmation_metrics,
            "stability": stability,
            "disable_gap_exposure": gaps,
        }
    )


def _validation_grid(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = []
    for top_k in K_GRID:
        for mode in CONDITION_MODES:
            metrics.append(_metric_for_params("validation", rows, _params(mode, top_k)))
    return metrics


def _select_params(validation_grid: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = max(
        validation_grid,
        key=lambda row: (row["lift"], row["precision"], -row["coverage"]),
        default={},
    )
    return {
        "mode": selected.get("mode", ""),
        "top_k": selected.get("top_k", 0.0),
        "min_conditions": selected.get("min_conditions", 0),
        "validation_policy_name": selected.get("policy_name", ""),
        "validation_lift": selected.get("lift", 0.0),
        "validation_precision": selected.get("precision", 0.0),
        "validation_coverage": selected.get("coverage", 0.0),
    }


def _params(mode: str, top_k: float) -> dict[str, Any]:
    return {
        "mode": mode,
        "top_k": top_k,
        "min_conditions": 2 if mode == "top_k_with_n_conditions" else 0,
    }


def _metric_for_params(
    run_label: str,
    rows: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    mask = _mask(rows, params)
    selected = [row for row, keep in zip(rows, mask) if keep]
    positives = sum(int(row["y_true"]) for row in rows)
    selected_positives = sum(int(row["y_true"]) for row in selected)
    total = len(rows)
    base_rate = positives / total if total else 0.0
    precision = selected_positives / len(selected) if selected else 0.0
    return {
        "run_label": run_label,
        "policy_name": _policy_name(params),
        "mode": params.get("mode", ""),
        "top_k": params.get("top_k", 0.0),
        "min_conditions": params.get("min_conditions", 0),
        "total_rows": total,
        "selected_rows": len(selected),
        "coverage": len(selected) / total if total else 0.0,
        "precision": precision,
        "lift": precision / base_rate if base_rate else 0.0,
        "recall": selected_positives / positives if positives else 0.0,
        "false_positive_count": len(selected) - selected_positives,
        "avg_prob": _mean([row["probability"] for row in selected]),
        "median_prob": _median([row["probability"] for row in selected]),
        "symbol_mix": _mix(selected, "symbol"),
        "month_mix": _mix(selected, "month"),
        "quarter_mix": _mix(selected, "quarter"),
        "disable_gap_exposure": sum(bool(row["disable_gap_match"]) for row in selected),
    }


def _mask(rows: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> list[bool]:
    mode = str(params.get("mode", ""))
    top_k = to_float(params.get("top_k"))
    selected_count = max(1, int(len(rows) * top_k / 100.0))
    if mode == "condition_then_top_k":
        eligible = [
            (index, row)
            for index, row in enumerate(rows)
            if row["confirmed_conditions"] and not row["disable_gap_match"]
        ]
    elif mode == "top_k_with_n_conditions":
        min_conditions = int(params.get("min_conditions", 2))
        eligible = [
            (index, row)
            for index, row in enumerate(rows)
            if len(row["confirmed_conditions"]) >= min_conditions
            and not row["disable_gap_match"]
        ]
    else:
        eligible = [
            (index, row)
            for index, row in enumerate(rows)
            if not row["disable_gap_match"]
        ]
    ranked = sorted(eligible, key=lambda item: item[1]["probability"], reverse=True)
    selected = {index for index, _row in ranked[:selected_count]}
    return [index in selected for index in range(len(rows))]


def _joined_rows(
    run_dir: Path,
    split: str,
    condition_weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    predictions = read_csv_rows(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_baselines"
        / f"predictions_{BASELINE_NAME}_{split}_full.csv"
    )
    features = {
        row_key(row): row
        for row in read_csv_rows(
            run_dir / "training_frame" / "m20_training_frame_features.csv"
        )
    }
    labels = {
        row_key(row): row
        for row in read_csv_rows(
            run_dir
            / "research_labels"
            / "vol_scaled"
            / "fee_exceedance_labels_vol_scaled.csv"
        )
        if row.get("scenario_name") == SCENARIO_NAME
    }
    joined = []
    for prediction in predictions:
        key = row_key(prediction)
        feature = features.get(key)
        label = labels.get(key)
        if not feature or not label:
            continue
        timestamp = str(prediction.get("interval_begin", ""))
        month = timestamp[:7]
        row = {
            **feature,
            "split": split,
            "y_true": int(float(prediction.get("y_true", label.get("label", 0)))),
            "probability": to_float(prediction.get("probability")),
            "month": month,
            "quarter": month_to_quarter(month),
        }
        row["conditions"] = selector_conditions(row)
        row["confirmed_conditions"] = [
            condition
            for condition in row["conditions"]
            if condition_weights.get(condition, 0.0) > 0.0
        ]
        row["disable_gap_match"] = sorted(set(row["conditions"]).intersection(DISABLE_GAPS))
        joined.append(row)
    return joined


def _previous_condition_then_top_one(original_run_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(
        original_run_dir
        / "research_labels"
        / "vol_scaled"
        / "rank_gated_selector"
        / "comparison.csv"
    )
    for row in rows:
        if row.get("policy_name") == "CONDITION_THEN_TOP_1":
            return {
                "policy_name": row.get("policy_name", ""),
                "original_coverage": to_float(row.get("original_coverage")),
                "original_precision": to_float(row.get("original_precision")),
                "original_lift": to_float(row.get("original_lift")),
                "confirmation_coverage": to_float(row.get("confirmation_coverage")),
                "confirmation_precision": to_float(row.get("confirmation_precision")),
                "confirmation_lift": to_float(row.get("confirmation_lift")),
            }
    return {}


def _stability(
    original_test: Mapping[str, Any],
    confirmation_test: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "policy_name": original_test.get("policy_name", ""),
        "original_coverage": original_test.get("coverage", 0.0),
        "confirmation_coverage": confirmation_test.get("coverage", 0.0),
        "original_precision": original_test.get("precision", 0.0),
        "confirmation_precision": confirmation_test.get("precision", 0.0),
        "original_lift": original_test.get("lift", 0.0),
        "confirmation_lift": confirmation_test.get("lift", 0.0),
        "lift_min": min(
            original_test.get("lift", 0.0),
            confirmation_test.get("lift", 0.0),
        ),
        "lift_delta": confirmation_test.get("lift", 0.0) - original_test.get("lift", 0.0),
        "coverage_delta": confirmation_test.get("coverage", 0.0)
        - original_test.get("coverage", 0.0),
    }


def _recommendation(
    original_test: Mapping[str, Any],
    confirmation_test: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> dict[str, Any]:
    lift_min = min(original_test.get("lift", 0.0), confirmation_test.get("lift", 0.0))
    coverage = original_test.get("coverage", 0.0)
    if lift_min >= 1.5 and 0.002 <= coverage <= 0.05:
        recommendation = (
            "A. keep nested rank-gate as research candidate and confirm on another "
            "window before any policy simulation."
        )
    elif lift_min >= 1.2:
        recommendation = "B. retune with stricter coverage or additional confirmation"
    else:
        recommendation = "E. reject current nested rank-gate candidate as unstable"
    return {
        "recommendation": recommendation,
        "selected_policy": original_test.get("policy_name", ""),
        "original_test_lift": original_test.get("lift", 0.0),
        "confirmation_test_lift": confirmation_test.get("lift", 0.0),
        "previous_condition_then_top_1": previous,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _gap_rows(run_label: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_label": run_label,
            "disable_gap_condition": gap,
            "encountered_rows": sum(gap in row["conditions"] for row in rows),
            "coverage": sum(gap in row["conditions"] for row in rows) / len(rows)
            if rows
            else 0.0,
        }
        for gap in sorted(DISABLE_GAPS)
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "validation_grid_csv": str(output_dir / "validation_grid.csv"),
        "locked_test_metrics_csv": str(output_dir / "locked_test_metrics.csv"),
        "confirmation_metrics_csv": str(output_dir / "confirmation_metrics.csv"),
        "stability_csv": str(output_dir / "stability.csv"),
        "disable_gap_exposure_csv": str(output_dir / "disable_gap_exposure.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    selected = report["selected_params"]
    original = report["original_test"]
    confirmation = report["confirmation_test"]
    return "\n".join(
        [
            "# M20 Nested Rank-Gate Tuning",
            "",
            f"- Selected validation policy: `{selected['validation_policy_name']}`",
            f"- Original test lift: `{original['lift']}`",
            f"- Confirmation test lift: `{confirmation['lift']}`",
            f"- Recommendation: `{report['recommendation']}`",
            f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
            "",
            "Validation selected the params; test and confirmation are locked evaluations.",
            "This is research-only, not runtime logic and not a backtest.",
            "",
        ]
    )


def _policy_name(params: Mapping[str, Any]) -> str:
    top_k = str(params.get("top_k", "")).rstrip("0").rstrip(".")
    if params.get("mode") == "top_k_with_n_conditions":
        return f"TOP_{top_k}_WITH_{params.get('min_conditions', 2)}_CONDITIONS"
    if params.get("mode") == "condition_then_top_k":
        return f"CONDITION_THEN_TOP_{top_k}"
    if params.get("mode") == "disable_gap_filtered_top_k":
        return f"DISABLE_GAP_FILTERED_TOP_{top_k}"
    mode = str(params.get("mode", "")).upper()
    return f"{mode}_{top_k}"


def _mix(rows: Sequence[Mapping[str, Any]], column: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(column, ""))
        counts[value] = counts.get(value, 0) + 1
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else 0.0


def _median(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return median(finite) if finite else 0.0
