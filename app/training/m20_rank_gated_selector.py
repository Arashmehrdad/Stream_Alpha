"""Research-only M20 rank-gated selector evaluation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_strategy_selector_simulation import (
    _conditions as selector_conditions,  # pylint: disable=protected-access
)
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "rank_gated_selector"
BASELINE_NAME = "logistic_regression_tiny"
SCENARIO_NAME = "current_fee"
TOP_K_VALUES = (1, 2, 5, 10)
DISABLE_GAPS = {"month=2026-04", "quarter=2026Q2"}
HONESTY_FLAGS = (
    "RESEARCH_ONLY_RANK_GATED_SELECTOR",
    "NOT_RUNTIME_SELECTOR",
    "NOT_BACKTEST",
    "NOT_PROMOTABLE",
    "NO_REGISTRY_WRITE",
    "NO_RUNTIME_EFFECT",
    "NO_PROMOTION_EFFECT",
    "NO_PROFITABILITY_CLAIM",
    "OPPORTUNITY_GATE_ONLY",
    "DOES_NOT_DECIDE_LONG_SHORT",
    "ORIGINAL_AND_CONFIRMATION_RUNS_USED",
    "DISABLE_GAPS_UNTESTED",
)


def tune_m20_rank_gated_selector(
    *,
    original_run_dir: Path,
    confirmation_run_dir: Path,
) -> dict[str, Any]:
    """Evaluate rank-gated selector policies on original and confirmation runs."""
    original = Path(original_run_dir).resolve()
    confirmation = Path(confirmation_run_dir).resolve()
    output_dir = original / "research_labels" / "vol_scaled" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_weights = _condition_weights(original)
    run_results = [
        _evaluate_run(original, "original", condition_weights),
        _evaluate_run(confirmation, "confirmation", condition_weights),
    ]
    metrics = [row for result in run_results for row in result["metrics"]]
    gaps = [row for result in run_results for row in result["gaps"]]
    comparison = _comparison(metrics)
    stability = _stability(comparison)
    recommendation = _recommendation(comparison)
    output_files = _output_files(output_dir)
    manifest = {
        "original_run_dir": str(original),
        "confirmation_run_dir": str(confirmation),
        "output_dir": str(output_dir),
        "top_k_values": list(TOP_K_VALUES),
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "output_files": output_files,
    }
    report = {
        "output_dir": str(output_dir),
        "policy_count": len({row["policy_name"] for row in metrics}),
        "honesty_flags": list(HONESTY_FLAGS),
        "recommendation": recommendation["recommendation"],
        "best_stable_policy": recommendation["best_stable_policy"],
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _report_markdown(report, comparison),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["metrics_csv"]), metrics)
    write_csv_artifact(Path(output_files["comparison_csv"]), comparison)
    write_csv_artifact(Path(output_files["stability_csv"]), stability)
    write_csv_artifact(Path(output_files["gaps_csv"]), gaps)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "metrics": metrics,
            "comparison": comparison,
            "stability": stability,
            "gaps": gaps,
        }
    )


def _evaluate_run(
    run_dir: Path,
    run_label: str,
    condition_weights: Mapping[str, float],
) -> dict[str, Any]:
    rows = _joined_rows(run_dir)
    for row in rows:
        row["conditions"] = selector_conditions(row)
        row["confirmed_conditions"] = [
            condition for condition in row["conditions"]
            if condition_weights.get(condition, 0.0) > 0.0
        ]
        row["disable_gap_match"] = sorted(set(row["conditions"]).intersection(DISABLE_GAPS))
    metrics = []
    for top_k in TOP_K_VALUES:
        metrics.append(
            _metric_row(
                run_label,
                f"GLOBAL_TOP_{top_k}",
                rows,
                _top_k_mask(rows, top_k),
            )
        )
        metrics.append(
            _metric_row(
                run_label,
                f"CONDITION_THEN_TOP_{top_k}",
                rows,
                _condition_then_top_k_mask(rows, top_k),
            )
        )
        metrics.append(
            _metric_row(
                run_label,
                f"TOP_{top_k}_WITH_2_CONDITIONS",
                rows,
                _top_k_with_n_conditions_mask(rows, top_k, 2),
            )
        )
        metrics.append(
            _metric_row(
                run_label,
                f"PER_CONDITION_TOP_{top_k}",
                rows,
                _per_condition_top_k_mask(rows, top_k),
            )
        )
        metrics.append(
            _metric_row(
                run_label,
                f"DISABLE_GAP_FILTERED_TOP_{top_k}",
                rows,
                _disable_gap_filtered_top_k_mask(rows, top_k),
            )
        )
    return {"metrics": metrics, "gaps": _gap_rows(run_label, rows)}


def _metric_row(
    run_label: str,
    policy_name: str,
    rows: Sequence[Mapping[str, Any]],
    mask: Sequence[bool],
) -> dict[str, Any]:
    total = len(rows)
    selected = [row for row, keep in zip(rows, mask) if keep]
    positives = sum(int(row["y_true"]) for row in rows)
    selected_positives = sum(int(row["y_true"]) for row in selected)
    base_rate = positives / total if total else 0.0
    precision = selected_positives / len(selected) if selected else 0.0
    return {
        "run_label": run_label,
        "policy_name": policy_name,
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


def _top_k_mask(rows: Sequence[Mapping[str, Any]], top_k: int) -> list[bool]:
    selected_count = max(1, int(len(rows) * top_k / 100.0))
    ranked = sorted(enumerate(rows), key=lambda item: item[1]["probability"], reverse=True)
    selected = {index for index, _row in ranked[:selected_count]}
    return [index in selected for index in range(len(rows))]


def _condition_then_top_k_mask(rows: Sequence[Mapping[str, Any]], top_k: int) -> list[bool]:
    eligible = [
        (index, row) for index, row in enumerate(rows)
        if row["confirmed_conditions"] and not row["disable_gap_match"]
    ]
    selected_count = max(1, int(len(rows) * top_k / 100.0))
    ranked = sorted(eligible, key=lambda item: item[1]["probability"], reverse=True)
    selected = {index for index, _row in ranked[:selected_count]}
    return [index in selected for index in range(len(rows))]


def _top_k_with_n_conditions_mask(
    rows: Sequence[Mapping[str, Any]],
    top_k: int,
    min_conditions: int,
) -> list[bool]:
    eligible = [
        (index, row) for index, row in enumerate(rows)
        if len(row["confirmed_conditions"]) >= min_conditions and not row["disable_gap_match"]
    ]
    selected_count = max(1, int(len(rows) * top_k / 100.0))
    ranked = sorted(eligible, key=lambda item: item[1]["probability"], reverse=True)
    selected = {index for index, _row in ranked[:selected_count]}
    return [index in selected for index in range(len(rows))]


def _per_condition_top_k_mask(rows: Sequence[Mapping[str, Any]], top_k: int) -> list[bool]:
    selected: set[int] = set()
    conditions = sorted(
        {
            condition for row in rows
            for condition in row["confirmed_conditions"]
            if not row["disable_gap_match"]
        }
    )
    per_condition_count = max(1, int(len(rows) * top_k / 100.0 / max(len(conditions), 1)))
    for condition in conditions:
        eligible = [
            (index, row) for index, row in enumerate(rows)
            if condition in row["confirmed_conditions"] and not row["disable_gap_match"]
        ]
        ranked = sorted(eligible, key=lambda item: item[1]["probability"], reverse=True)
        selected.update(index for index, _row in ranked[:per_condition_count])
    return [index in selected for index in range(len(rows))]


def _disable_gap_filtered_top_k_mask(
    rows: Sequence[Mapping[str, Any]],
    top_k: int,
) -> list[bool]:
    eligible = [
        (index, row) for index, row in enumerate(rows) if not row["disable_gap_match"]
    ]
    selected_count = max(1, int(len(rows) * top_k / 100.0))
    ranked = sorted(eligible, key=lambda item: item[1]["probability"], reverse=True)
    selected = {index for index, _row in ranked[:selected_count]}
    return [index in selected for index in range(len(rows))]


def _comparison(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["run_label"], row["policy_name"]): row for row in metrics}
    policies = sorted({row["policy_name"] for row in metrics})
    rows = []
    for policy in policies:
        original = by_key.get(("original", policy), {})
        confirmation = by_key.get(("confirmation", policy), {})
        rows.append(
            {
                "policy_name": policy,
                "original_coverage": original.get("coverage", 0.0),
                "confirmation_coverage": confirmation.get("coverage", 0.0),
                "original_precision": original.get("precision", 0.0),
                "confirmation_precision": confirmation.get("precision", 0.0),
                "original_lift": original.get("lift", 0.0),
                "confirmation_lift": confirmation.get("lift", 0.0),
                "lift_min": min(original.get("lift", 0.0), confirmation.get("lift", 0.0)),
                "lift_delta": confirmation.get("lift", 0.0) - original.get("lift", 0.0),
                "coverage_delta": confirmation.get("coverage", 0.0)
                - original.get("coverage", 0.0),
            }
        )
    return rows


def _stability(comparison: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy_name": row["policy_name"],
            "stable_lift_above_one": row["lift_min"] > 1.0,
            "coverage_shift_abs": abs(row["coverage_delta"]),
            "lift_shift_abs": abs(row["lift_delta"]),
        }
        for row in comparison
    ]


def _recommendation(comparison: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stable = [
        row for row in comparison
        if row["lift_min"] > 1.2 and 0.005 <= row["original_coverage"] <= 0.15
    ]
    best = max(stable, key=lambda row: row["lift_min"], default={})
    if best:
        recommendation = (
            "C. tune selector thresholds in a nested/held-out way around the "
            f"stable candidate {best['policy_name']}."
        )
    else:
        recommendation = "B. run selector simulation on another confirmation window"
    return {
        "recommendation": recommendation,
        "best_stable_policy": best.get("policy_name", ""),
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _joined_rows(run_dir: Path) -> list[dict[str, Any]]:
    predictions = _read_csv_rows(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_baselines"
        / f"predictions_{BASELINE_NAME}_test_full.csv"
    )
    features = _by_key(
        _read_csv_rows(run_dir / "training_frame" / "m20_training_frame_features.csv")
    )
    labels = _by_key(
        [
            row for row in _read_csv_rows(
                run_dir
                / "research_labels"
                / "vol_scaled"
                / "fee_exceedance_labels_vol_scaled.csv"
            )
            if row.get("scenario_name") == SCENARIO_NAME
        ]
    )
    joined = []
    for prediction in predictions:
        key = _key(prediction)
        feature = features.get(key)
        label = labels.get(key)
        if not feature or not label:
            continue
        timestamp = str(prediction.get("interval_begin", ""))
        month = timestamp[:7]
        row = {
            **feature,
            "y_true": int(float(prediction.get("y_true", label.get("label", 0)))),
            "probability": _to_float(prediction.get("probability")),
            "month": month,
            "quarter": _quarter(month),
        }
        joined.append(row)
    return joined


def _condition_weights(original_run_dir: Path) -> dict[str, float]:
    rows = _read_csv_rows(
        original_run_dir
        / "research_labels"
        / "vol_scaled"
        / "strategy_selector_design"
        / "strategy_selector_condition_weights.csv"
    )
    return {
        f"{row['slice_family']}={row['slice_value']}": _to_float(row.get("evidence_weight"))
        for row in rows
        if row.get("proposed_selector_action") == "ENABLE_RESEARCH_CANDIDATE"
    }


def _gap_rows(run_label: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_label": run_label,
            "disable_gap_condition": gap,
            "encountered_rows": sum(gap in row["conditions"] for row in rows),
            "coverage": sum(gap in row["conditions"] for row in rows) / len(rows)
            if rows else 0.0,
        }
        for gap in sorted(DISABLE_GAPS)
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "metrics_csv": str(output_dir / "metrics.csv"),
        "comparison_csv": str(output_dir / "comparison.csv"),
        "stability_csv": str(output_dir / "stability.csv"),
        "gaps_csv": str(output_dir / "gaps.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _report_markdown(report: Mapping[str, Any], comparison: Sequence[Mapping[str, Any]]) -> str:
    best = max(comparison, key=lambda row: row["lift_min"], default={})
    return "\n".join(
        [
            "# M20 Rank-Gated Selector Evaluation",
            "",
            f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
            f"- Recommendation: `{report['recommendation']}`",
            f"- Best min-lift policy: `{best.get('policy_name', '')}`",
            "",
            "This is research-only opportunity-gate evaluation, not a backtest.",
            "",
        ]
    )


def _by_key(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str], Mapping[str, str]]:
    return {_key(row): row for row in rows}


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("symbol", "")),
        str(row.get("interval_begin", "")),
        str(row.get("fold_index", "")),
    )


def _mix(rows: Sequence[Mapping[str, Any]], column: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(column, ""))
        counts[value] = counts.get(value, 0) + 1
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _quarter(month: str) -> str:
    try:
        year, month_number = month.split("-")
        return f"{year}Q{((int(month_number) - 1) // 3) + 1}"
    except (ValueError, IndexError):
        return ""


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else 0.0


def _median(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return median(finite) if finite else 0.0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def _to_float(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 0.0
    return converted if math.isfinite(converted) else 0.0
