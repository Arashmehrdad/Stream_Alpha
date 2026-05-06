"""Research-only M20 strategy selector simulation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


SIMULATION_DIR_NAME = "strategy_selector_simulation"
DESIGN_DIR_NAME = "strategy_selector_design"
BASELINE_NAME = "logistic_regression_tiny"
SCENARIO_NAME = "current_fee"
WEIGHTED_ALLOW_THRESHOLD = 1.0
WEIGHTED_WATCHLIST_THRESHOLD = 0.5
STRICT_REQUIRED_CONDITIONS = 2
DISABLE_GAPS = {"month=2026-04", "quarter=2026Q2"}
HONESTY_FLAGS = (
    "RESEARCH_ONLY_SELECTOR_SIMULATION",
    "NOT_RUNTIME_SELECTOR",
    "NOT_BACKTEST",
    "NOT_PROMOTABLE",
    "NO_REGISTRY_WRITE",
    "NO_RUNTIME_EFFECT",
    "NO_PROMOTION_EFFECT",
    "NO_PROFITABILITY_CLAIM",
    "STRATEGY_ENSEMBLE_SIMULATION_ONLY",
    "OPPORTUNITY_GATE_ONLY",
    "DOES_NOT_DECIDE_LONG_SHORT",
    "RESEARCH_THRESHOLD_NOT_OPTIMIZED",
    "DISABLE_GAPS_UNTESTED",
    "ORIGINAL_AND_CONFIRMATION_RUNS_USED",
    "FUTURE_CONFIRMATION_REQUIRED",
)


def simulate_m20_strategy_selector(
    *,
    original_run_dir: Path,
    confirmation_run_dir: Path,
) -> dict[str, Any]:
    """Simulate research-only selector behavior on original and confirmation runs."""
    # pylint: disable=too-many-locals
    original = Path(original_run_dir).resolve()
    confirmation = Path(confirmation_run_dir).resolve()
    design_dir = original / "research_labels" / "vol_scaled" / DESIGN_DIR_NAME
    if not (design_dir / "strategy_selector_candidate_spec.json").exists():
        raise ValueError("MISSING_SELECTOR_SPEC")
    output_dir = original / "research_labels" / "vol_scaled" / SIMULATION_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_weights = _condition_weights(design_dir)
    runs = [
        _simulate_one_run(
            run_dir=original,
            run_label="original",
            condition_weights=condition_weights,
        ),
        _simulate_one_run(
            run_dir=confirmation,
            run_label="confirmation",
            condition_weights=condition_weights,
        ),
    ]
    policy_rows = [row for run in runs for row in run["policy_metrics"]]
    selected_summary = [row for run in runs for row in run["selected_rows_summary"]]
    contribution_rows = [row for run in runs for row in run["condition_contributions"]]
    skipped_rows = [row for run in runs for row in run["skipped_reason_summary"]]
    disable_gap_rows = [row for run in runs for row in run["disable_gap_exposure"]]
    comparison_rows = _policy_comparison_rows(runs)
    stability_rows = _cross_run_stability_rows(runs)
    recommendation = _recommendation(policy_rows)
    output_files = _output_files(output_dir)
    report = {
        "selector_id": "fee_exceedance_gate_v0_research",
        "simulation_dir": str(output_dir),
        "original_run_dir": str(original),
        "confirmation_run_dir": str(confirmation),
        "thresholds": _thresholds(),
        "honesty_flags": list(HONESTY_FLAGS),
        "policy_count": len({row["policy_name"] for row in policy_rows}),
        "recommendation": recommendation,
        "output_files": output_files,
    }
    manifest = {
        "selector_id": report["selector_id"],
        "simulation_dir": str(output_dir),
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "thresholds": _thresholds(),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["selector_simulation_manifest_json"]), manifest)
    write_json_artifact(Path(output_files["selector_simulation_report_json"]), report)
    Path(output_files["selector_simulation_report_md"]).write_text(
        _report_markdown(report, policy_rows, stability_rows),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["selector_policy_metrics_by_run_csv"]), policy_rows)
    write_csv_artifact(
        Path(output_files["selector_policy_metrics_comparison_csv"]),
        comparison_rows,
    )
    write_csv_artifact(
        Path(output_files["selector_selected_rows_summary_csv"]),
        selected_summary,
    )
    write_csv_artifact(
        Path(output_files["selector_condition_contributions_csv"]),
        contribution_rows,
    )
    write_csv_artifact(
        Path(output_files["selector_skipped_reason_summary_csv"]),
        skipped_rows,
    )
    write_csv_artifact(
        Path(output_files["selector_cross_run_stability_csv"]),
        stability_rows,
    )
    write_csv_artifact(Path(output_files["selector_policy_sensitivity_csv"]), policy_rows)
    write_csv_artifact(
        Path(output_files["selector_disable_gap_exposure_csv"]),
        disable_gap_rows,
    )
    write_json_artifact(
        Path(output_files["selector_simulation_recommendation_json"]),
        {"recommendation": recommendation, "honesty_flags": list(HONESTY_FLAGS)},
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "policy_metrics": policy_rows,
            "cross_run_stability": stability_rows,
        }
    )


def _simulate_one_run(
    *,
    run_dir: Path,
    run_label: str,
    condition_weights: Mapping[str, float],
) -> dict[str, Any]:
    rows = _joined_rows(run_dir)
    if not rows:
        raise ValueError(f"MISSING_RUN_INPUTS:{run_dir}")
    for row in rows:
        row["conditions"] = _conditions(row)
        row["matched_confirmed_conditions"] = [
            condition for condition in row["conditions"]
            if condition_weights.get(condition, 0.0) > 0.0
        ]
        row["selector_score"] = sum(
            condition_weights[condition]
            for condition in row["matched_confirmed_conditions"]
        )
        row["disable_gap_match"] = sorted(set(row["conditions"]).intersection(DISABLE_GAPS))
    policies = {
        "GLOBAL_LOGISTIC_TOP5": _global_top5_mask(rows),
        "SELECTOR_WEIGHTED_CONFIRMED": [
            _weighted_action(row) == "ALLOW_STRATEGY_SEARCH" for row in rows
        ],
        "SELECTOR_ANY_CONFIRMED": [
            bool(row["matched_confirmed_conditions"]) and not row["disable_gap_match"]
            for row in rows
        ],
        "SELECTOR_STRICT_MULTI_CONDITION": [
            len(row["matched_confirmed_conditions"]) >= STRICT_REQUIRED_CONDITIONS
            and not row["disable_gap_match"]
            for row in rows
        ],
    }
    policy_metrics = [
        _policy_metrics(run_label, policy_name, rows, mask)
        for policy_name, mask in policies.items()
    ]
    return {
        "run_label": run_label,
        "rows": rows,
        "policy_metrics": policy_metrics,
        "selected_rows_summary": _selected_rows_summary(run_label, rows, policies),
        "condition_contributions": _condition_contributions(run_label, rows, condition_weights),
        "skipped_reason_summary": _skipped_reason_summary(run_label, rows),
        "disable_gap_exposure": _disable_gap_exposure(run_label, rows),
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
        joined.append(
            {
                **feature,
                "y_true": int(float(prediction.get("y_true", label.get("label", 0)))),
                "probability": _to_float(prediction.get("probability")),
                "split": prediction.get("split", "test"),
            }
        )
    return joined


def _conditions(row: Mapping[str, Any]) -> list[str]:
    timestamp = str(row.get("interval_begin", ""))
    month = timestamp[:7]
    quarter = _quarter(month)
    conditions = [
        f"symbol={row.get('symbol', '')}",
        f"month={month}",
        f"quarter={quarter}",
        f"momentum={_momentum_bucket(_to_float(row.get('log_return_1')))}",
        f"range={_range_bucket(row)}",
        f"macd={_macd_bucket(_to_float(row.get('macd_line_12_26')))}",
        f"volume={_tertile_bucket(_to_float(row.get('volume_zscore_12')))}",
        f"volatility={_tertile_bucket(_to_float(row.get('realized_vol_12')))}",
        f"rsi={_rsi_bucket(_to_float(row.get('rsi_14')))}",
    ]
    return conditions


def _policy_metrics(
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
    top5 = _global_top5_mask(rows)
    top5_selected = sum(1 for keep, top in zip(mask, top5) if keep and top)
    false_positive_count = len(selected) - selected_positives
    return {
        "run_label": run_label,
        "policy_name": policy_name,
        "total_test_rows": total,
        "selected_row_count": len(selected),
        "coverage": len(selected) / total if total else 0.0,
        "selected_positive_count": selected_positives,
        "selected_positive_rate_precision": precision,
        "base_positive_rate": base_rate,
        "lift_vs_base": precision / base_rate if base_rate else 0.0,
        "recall_of_positive_events": selected_positives / positives if positives else 0.0,
        "false_positive_count": false_positive_count,
        "false_positive_rate": false_positive_count / len(selected) if selected else 0.0,
        "average_predicted_probability_selected": _mean([row["probability"] for row in selected]),
        "median_predicted_probability_selected": _median([row["probability"] for row in selected]),
        "top5_overlap_count": top5_selected,
        "top5_overlap_rate": top5_selected / sum(top5) if sum(top5) else 0.0,
        "disable_gap_unknown_count": sum(bool(row["disable_gap_match"]) for row in selected),
        "month_2026_04_encountered": sum("month=2026-04" in row["conditions"] for row in rows),
        "quarter_2026Q2_encountered": sum("quarter=2026Q2" in row["conditions"] for row in rows),
    }


def _global_top5_mask(rows: Sequence[Mapping[str, Any]]) -> list[bool]:
    selected_count = max(1, int(len(rows) * 0.05))
    ranked = sorted(
        enumerate(rows),
        key=lambda item: item[1]["probability"],
        reverse=True,
    )
    selected_indices = {index for index, _row in ranked[:selected_count]}
    return [index in selected_indices for index in range(len(rows))]


def _weighted_action(row: Mapping[str, Any]) -> str:
    if row["disable_gap_match"]:
        return "DISABLE_GAP_UNKNOWN"
    if row["selector_score"] >= WEIGHTED_ALLOW_THRESHOLD:
        return "ALLOW_STRATEGY_SEARCH"
    if row["selector_score"] >= WEIGHTED_WATCHLIST_THRESHOLD:
        return "WATCHLIST_ONLY"
    return "HOLD_OR_SKIP"


def _selected_rows_summary(
    run_label: str,
    rows: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Sequence[bool]],
) -> list[dict[str, Any]]:
    summary = []
    for policy_name, mask in policies.items():
        selected = [row for row, keep in zip(rows, mask) if keep]
        for column in ("symbol",):
            for value, count in _counts(row.get(column, "") for row in selected).items():
                summary.append(
                    {
                        "run_label": run_label,
                        "policy_name": policy_name,
                        "summary_column": column,
                        "summary_value": value,
                        "selected_row_count": count,
                    }
                )
    return summary


def _condition_contributions(
    run_label: str,
    rows: Sequence[Mapping[str, Any]],
    condition_weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    positives: dict[str, int] = {}
    for row in rows:
        for condition in row["matched_confirmed_conditions"]:
            counts[condition] = counts.get(condition, 0) + 1
            positives[condition] = positives.get(condition, 0) + int(row["y_true"])
    return [
        {
            "run_label": run_label,
            "condition": condition,
            "matched_row_count": counts[condition],
            "matched_positive_count": positives[condition],
            "matched_positive_rate": positives[condition] / counts[condition],
            "evidence_weight": condition_weights.get(condition, 0.0),
        }
        for condition in sorted(counts)
    ]


def _skipped_reason_summary(
    run_label: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reasons = {"NO_CONFIRMED_CONDITION": 0, "DISABLE_GAP_UNKNOWN": 0}
    for row in rows:
        action = _weighted_action(row)
        if action == "DISABLE_GAP_UNKNOWN":
            reasons["DISABLE_GAP_UNKNOWN"] += 1
        elif action == "HOLD_OR_SKIP":
            reasons["NO_CONFIRMED_CONDITION"] += 1
    return [
        {"run_label": run_label, "reason": reason, "row_count": count}
        for reason, count in sorted(reasons.items())
    ]


def _disable_gap_exposure(
    run_label: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "run_label": run_label,
            "disable_gap_condition": gap,
            "encountered_row_count": sum(gap in row["conditions"] for row in rows),
            "coverage": sum(gap in row["conditions"] for row in rows) / len(rows) if rows else 0.0,
            "status": "DISABLE_GAP_UNTESTED",
        }
        for gap in sorted(DISABLE_GAPS)
    ]


def _policy_comparison_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_run = {run["run_label"]: run["policy_metrics"] for run in runs}
    original = {row["policy_name"]: row for row in by_run.get("original", [])}
    confirmation = {row["policy_name"]: row for row in by_run.get("confirmation", [])}
    rows = []
    for policy in sorted(set(original).intersection(confirmation)):
        rows.append(
            {
                "policy_name": policy,
                "original_precision": original[policy]["selected_positive_rate_precision"],
                "confirmation_precision": confirmation[policy]["selected_positive_rate_precision"],
                "precision_delta": (
                    confirmation[policy]["selected_positive_rate_precision"]
                    - original[policy]["selected_positive_rate_precision"]
                ),
                "original_lift": original[policy]["lift_vs_base"],
                "confirmation_lift": confirmation[policy]["lift_vs_base"],
                "lift_delta": (
                    confirmation[policy]["lift_vs_base"] - original[policy]["lift_vs_base"]
                ),
                "original_coverage": original[policy]["coverage"],
                "confirmation_coverage": confirmation[policy]["coverage"],
            }
        )
    return rows


def _cross_run_stability_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return _policy_comparison_rows(runs)


def _recommendation(policy_rows: Sequence[Mapping[str, Any]]) -> str:
    weighted = [
        row for row in policy_rows
        if row["policy_name"] == "SELECTOR_WEIGHTED_CONFIRMED"
    ]
    if all(row["lift_vs_base"] > 1.0 and row["coverage"] >= 0.01 for row in weighted):
        return (
            "A. design research-only strategy family modules next; also confirm "
            "disable/gap slices separately."
        )
    return "B. run selector simulation on another confirmation window"


def _condition_weights(design_dir: Path) -> dict[str, float]:
    rows = _read_csv_rows(design_dir / "strategy_selector_condition_weights.csv")
    return {
        f"{row['slice_family']}={row['slice_value']}": _to_float(row.get("evidence_weight"))
        for row in rows
        if row.get("proposed_selector_action") == "ENABLE_RESEARCH_CANDIDATE"
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    names = {
        "selector_simulation_manifest_json": "selector_simulation_manifest.json",
        "selector_simulation_report_json": "selector_simulation_report.json",
        "selector_simulation_report_md": "selector_simulation_report.md",
        "selector_policy_metrics_by_run_csv": "selector_policy_metrics_by_run.csv",
        "selector_policy_metrics_comparison_csv": "selector_policy_metrics_comparison.csv",
        "selector_selected_rows_summary_csv": "selector_selected_rows_summary.csv",
        "selector_condition_contributions_csv": "selector_condition_contributions.csv",
        "selector_skipped_reason_summary_csv": "selector_skipped_reason_summary.csv",
        "selector_cross_run_stability_csv": "selector_cross_run_stability.csv",
        "selector_policy_sensitivity_csv": "selector_policy_sensitivity.csv",
        "selector_disable_gap_exposure_csv": "selector_disable_gap_exposure.csv",
        "selector_simulation_recommendation_json": "selector_simulation_recommendation.json",
    }
    return {key: str(output_dir / name) for key, name in names.items()}


def _report_markdown(
    report: Mapping[str, Any],
    policy_rows: Sequence[Mapping[str, Any]],
    stability_rows: Sequence[Mapping[str, Any]],
) -> str:
    weighted = [
        row for row in policy_rows
        if row["policy_name"] == "SELECTOR_WEIGHTED_CONFIRMED"
    ]
    return "\n".join(
        [
            "# M20 Selector Simulation Report",
            "",
            f"- Selector: `{report['selector_id']}`",
            f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
            f"- Recommendation: `{report['recommendation']}`",
            "",
            "## Executive Summary",
            "",
            "- This is an opportunity-gate simulation only, not a backtest.",
            "- It does not decide LONG/SHORT and is not runtime-ready.",
            "",
            "## Weighted Selector",
            "",
            *[
                (
                    f"- `{row['run_label']}`: coverage `{row['coverage']:.6f}`, "
                    f"precision `{row['selected_positive_rate_precision']:.6f}`, "
                    f"lift `{row['lift_vs_base']:.6f}`"
                )
                for row in weighted
            ],
            "",
            "## Cross-Run Stability",
            "",
            *[
                (
                    f"- `{row['policy_name']}`: lift "
                    f"`{row['original_lift']:.6f}` -> `{row['confirmation_lift']:.6f}`"
                )
                for row in stability_rows
            ],
            "",
            "Disable gaps remain untested and require separate confirmation.",
            "",
        ]
    )


def _quarter(month: str) -> str:
    try:
        year, month_number = month.split("-")
        quarter = (int(month_number) - 1) // 3 + 1
        return f"{year}Q{quarter}"
    except (ValueError, IndexError):
        return ""


def _momentum_bucket(value: float) -> str:
    if value < -0.0001:
        return "negative"
    if value > 0.0001:
        return "positive"
    return "flat"


def _range_bucket(row: Mapping[str, Any]) -> str:
    close = _to_float(row.get("close_price"))
    if close <= 0:
        return "mid"
    range_pct = (_to_float(row.get("high_price")) - _to_float(row.get("low_price"))) / close
    if range_pct <= 0.001:
        return "low"
    if range_pct >= 0.003:
        return "high"
    return "mid"


def _macd_bucket(value: float) -> str:
    if value < -0.0001:
        return "negative"
    if value > 0.0001:
        return "positive"
    return "near_zero"


def _rsi_bucket(value: float) -> str:
    if value < 30:
        return "oversold"
    if value > 70:
        return "overbought"
    return "neutral"


def _tertile_bucket(value: float) -> str:
    if value < -0.25:
        return "low"
    if value > 0.25:
        return "high"
    return "mid"


def _thresholds() -> dict[str, Any]:
    return {
        "weighted_allow_threshold": WEIGHTED_ALLOW_THRESHOLD,
        "weighted_watchlist_threshold": WEIGHTED_WATCHLIST_THRESHOLD,
        "strict_required_conditions": STRICT_REQUIRED_CONDITIONS,
    }


def _by_key(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str], Mapping[str, str]]:
    return {_key(row): row for row in rows}


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("symbol", "")),
        str(row.get("interval_begin", "")),
        str(row.get("fold_index", "")),
    )


def _counts(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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
