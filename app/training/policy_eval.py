"""Research-only offline threshold policy evaluation for completed M20 runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import (
    load_summary_payload,
    resolve_fee_rate,
    resolve_winner_model_name,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_ANALYSIS_DIR_NAME = "policy_eval"
DEFAULT_THRESHOLD_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
DEFAULT_BASELINE_THRESHOLD = 0.50
DEFAULT_M20_ARTIFACT_ROOT = Path("artifacts") / "training" / "m20"
DEFAULT_MIN_HONEST_TRADES = 5
DEFAULT_MIN_HONEST_COVERAGE = 0.01
DEFAULT_SCENARIO_SLIPPAGE_RATES = (0.0, 0.001)
_REQUIRED_OOF_COLUMNS = (
    "model_name",
    "fold_index",
    "row_id",
    "y_true",
    "prob_up",
    "future_return_3",
    "long_only_gross_value_proxy",
    "long_only_net_value_proxy",
)


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True, slots=True)
class PolicyEvalRow:
    """Winner-model OOF row used by the offline policy evaluator."""

    model_name: str
    fold_index: int
    row_id: str
    y_true: int
    prob_up: float
    future_return_3: float
    long_only_gross_value_proxy: float
    long_only_net_value_proxy: float
    source_index: int
    regime_label: str | None = None
    interval_begin: str | None = None
    as_of_time: str | None = None


def resolve_completed_run_dir(run_dir: Path | None) -> Path:
    """Resolve a completed M20 run directory, defaulting to the newest known run."""
    if run_dir is not None:
        resolved = Path(run_dir).resolve()
        if not resolved.exists():
            raise ValueError(f"Completed run directory does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"Completed run path is not a directory: {resolved}")
        return resolved

    artifact_root = (_repo_root() / DEFAULT_M20_ARTIFACT_ROOT).resolve()
    if not artifact_root.exists():
        raise ValueError(f"No M20 artifact root exists yet at {artifact_root}")
    run_dirs = sorted(
        (
            path
            for path in artifact_root.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise ValueError(f"No completed M20 run directories were found under {artifact_root}")
    return run_dirs[0].resolve()


def load_policy_eval_rows(path: Path, *, model_name: str) -> list[PolicyEvalRow]:
    """Load winner-model OOF rows needed for offline policy evaluation."""
    if not path.exists():
        raise ValueError(f"Completed run is missing oof_predictions.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        field_names = reader.fieldnames or ()
        missing_columns = [column for column in _REQUIRED_OOF_COLUMNS if column not in field_names]
        if missing_columns:
            raise ValueError(
                "Out-of-fold predictions are missing required columns for policy evaluation: "
                f"{missing_columns}"
            )
        rows = [
            _row_from_csv(raw_row, source_index)
            for source_index, raw_row in enumerate(reader)
            if raw_row["model_name"] == model_name
        ]
    if not rows:
        raise ValueError(
            f"No out-of-fold predictions were found for model {model_name!r} in {path}"
        )
    return rows


def analyze_completed_run(
    *,
    run_dir: Path | None,
    thresholds: Iterable[float] = DEFAULT_THRESHOLD_GRID,
    model_name: str | None = None,
    baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD,
    analysis_dir_name: str = DEFAULT_ANALYSIS_DIR_NAME,
    min_honest_trades: int = DEFAULT_MIN_HONEST_TRADES,
    min_honest_coverage: float = DEFAULT_MIN_HONEST_COVERAGE,
    scenario_slippage_rates: Sequence[float] = DEFAULT_SCENARIO_SLIPPAGE_RATES,
) -> dict[str, Any]:
    """Analyze a completed M20 run directory and persist policy-evaluation artifacts."""
    # pylint: disable=too-many-arguments,too-many-locals
    resolved_run_dir = resolve_completed_run_dir(run_dir)
    summary_payload = _load_completed_summary(resolved_run_dir)
    resolved_model_name = model_name or _resolve_completed_winner_model_name(summary_payload)
    rows = load_policy_eval_rows(
        resolved_run_dir / "oof_predictions.csv",
        model_name=resolved_model_name,
    )
    fee_rate = resolve_fee_rate(summary_payload, rows)
    threshold_grid = _normalize_threshold_grid(thresholds)
    normalized_baseline = _normalize_threshold(baseline_threshold)

    baseline_result = _evaluate_threshold(rows, threshold=normalized_baseline, fee_rate=fee_rate)
    candidate_results = [
        _evaluate_threshold(rows, threshold=threshold, fee_rate=fee_rate)
        for threshold in threshold_grid
    ]
    scenario_specs = _build_cost_scenario_specs(
        fee_rate=fee_rate,
        slippage_rates=scenario_slippage_rates,
    )
    candidate_results = [
        _attach_research_diagnostics(
            result,
            baseline_result,
            scenario_specs=scenario_specs,
            min_honest_trades=min_honest_trades,
            min_honest_coverage=min_honest_coverage,
        )
        for result in candidate_results
    ]
    best_candidate = select_best_policy_result(candidate_results)
    analysis_dir = resolved_run_dir / analysis_dir_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv_path = analysis_dir / "policy_candidates.csv"
    report_json_path = analysis_dir / "policy_report.json"
    report_md_path = analysis_dir / "policy_report.md"

    analysis_summary = {
        "run_dir": str(resolved_run_dir),
        "analysis_dir": str(analysis_dir),
        "model_name": resolved_model_name,
        "fee_rate": fee_rate,
        "threshold_grid": threshold_grid,
        "baseline_threshold": normalized_baseline,
        "baseline_result": baseline_result,
        "candidate_results": candidate_results,
        "best_candidate": best_candidate,
        "regime_summary_available": any(row.regime_label for row in rows),
        "low_trade_policy": {
            "min_honest_trades": min_honest_trades,
            "min_honest_coverage": min_honest_coverage,
        },
        "cost_scenarios": [dict(scenario) for scenario in scenario_specs],
        "output_files": {
            "policy_candidates_csv": str(candidates_csv_path),
            "policy_report_json": str(report_json_path),
            "policy_report_md": str(report_md_path),
        },
    }
    write_csv_artifact(
        candidates_csv_path,
        [
            {
                **_flatten_candidate_row(result),
                "is_best_candidate": (
                    float(result["threshold"]) == float(best_candidate["threshold"])
                ),
                "is_baseline_threshold": float(result["threshold"]) == float(normalized_baseline),
            }
            for result in candidate_results
        ],
    )
    write_json_artifact(report_json_path, analysis_summary)
    report_md_path.write_text(_build_report_markdown(analysis_summary), encoding="utf-8")
    return make_json_safe(analysis_summary)


def select_best_policy_result(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose a deterministic best threshold while avoiding zero-trade winners when possible."""
    result_list = [dict(result) for result in results]
    if not result_list:
        raise ValueError("No policy-evaluation results were available for selection")
    trading_results = [result for result in result_list if int(result["trade_count"]) > 0]
    candidate_results = trading_results or result_list
    return sorted(candidate_results, key=_policy_result_sort_key)[0]


def _evaluate_threshold(
    rows: list[PolicyEvalRow],
    *,
    threshold: float,
    fee_rate: float,
) -> dict[str, Any]:
    trade_rows = [row for row in rows if row.prob_up >= threshold]
    per_fold_breakdown = [
        {
            "fold_index": fold_index,
            **_compute_metrics(fold_rows, _select_trade_rows(fold_rows, threshold)),
        }
        for fold_index, fold_rows in sorted(_group_rows_by_fold(rows).items())
    ]
    metrics = _compute_metrics(rows, trade_rows)
    return {
        "threshold": threshold,
        **metrics,
        "fee_rate": fee_rate,
        "per_fold_breakdown": per_fold_breakdown,
        "per_regime_breakdown": _build_regime_breakdown(rows, threshold),
    }


def _compute_metrics(
    rows: list[PolicyEvalRow],
    trade_rows: list[PolicyEvalRow],
) -> dict[str, Any]:
    prediction_count = len(rows)
    trade_count = len(trade_rows)
    abstention_count = prediction_count - trade_count
    gross_sum = sum(row.long_only_gross_value_proxy for row in trade_rows)
    net_sum = sum(row.long_only_net_value_proxy for row in trade_rows)
    precision_on_trades = (
        sum(int(row.y_true == 1) for row in trade_rows) / trade_count
        if trade_count > 0
        else None
    )
    ordered_trades = _order_rows_for_drawdown(trade_rows)
    drawdown = _compute_max_drawdown_proxy(ordered_trades)
    if prediction_count == 0:
        return {
            "prediction_count": 0,
            "trade_count": 0,
            "coverage": 0.0,
            "trade_rate": 0.0,
            "abstention_count": 0,
            "abstention_rate": 0.0,
            "precision_on_trades": precision_on_trades,
            "directional_accuracy_on_trades": precision_on_trades,
            "mean_long_only_gross_value_proxy": 0.0,
            "mean_long_only_net_value_proxy": 0.0,
            "cumulative_long_only_gross_value_proxy": 0.0,
            "cumulative_long_only_net_value_proxy": 0.0,
            "max_drawdown_proxy": 0.0,
            "after_cost_positive": False,
            "trade_row_ids": [],
        }
    mean_gross = gross_sum / prediction_count
    mean_net = net_sum / prediction_count
    return {
        "prediction_count": prediction_count,
        "trade_count": trade_count,
        "coverage": trade_count / prediction_count,
        "trade_rate": trade_count / prediction_count,
        "abstention_count": abstention_count,
        "abstention_rate": abstention_count / prediction_count,
        "precision_on_trades": precision_on_trades,
        "directional_accuracy_on_trades": precision_on_trades,
        "mean_long_only_gross_value_proxy": mean_gross,
        "mean_long_only_net_value_proxy": mean_net,
        "cumulative_long_only_gross_value_proxy": gross_sum,
        "cumulative_long_only_net_value_proxy": net_sum,
        "max_drawdown_proxy": drawdown,
        "after_cost_positive": mean_net > 0.0,
        "trade_row_ids": [row.row_id for row in ordered_trades],
    }


def _build_regime_breakdown(
    rows: list[PolicyEvalRow],
    threshold: float,
) -> list[dict[str, Any]] | None:
    regime_rows = [row for row in rows if row.regime_label]
    if not regime_rows:
        return None
    grouped: dict[str, list[PolicyEvalRow]] = {}
    for row in regime_rows:
        grouped.setdefault(str(row.regime_label), []).append(row)
    return [
        {
            "regime_label": regime_label,
            **_compute_metrics(group, _select_trade_rows(group, threshold)),
        }
        for regime_label, group in sorted(grouped.items())
    ]


def _attach_baseline_comparison(
    result: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(result),
        "beats_baseline_mean_net": (
            float(result["mean_long_only_net_value_proxy"])
            > float(baseline_result["mean_long_only_net_value_proxy"])
        ),
        "delta_vs_baseline_mean_long_only_net_value_proxy": (
            float(result["mean_long_only_net_value_proxy"])
            - float(baseline_result["mean_long_only_net_value_proxy"])
        ),
        "delta_vs_baseline_trade_count": (
            int(result["trade_count"]) - int(baseline_result["trade_count"])
        ),
        "delta_vs_baseline_max_drawdown_proxy": (
            float(result["max_drawdown_proxy"]) - float(baseline_result["max_drawdown_proxy"])
        ),
    }


def _attach_research_diagnostics(
    result: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    *,
    scenario_specs: Sequence[Mapping[str, Any]],
    min_honest_trades: int,
    min_honest_coverage: float,
) -> dict[str, Any]:
    compared = _attach_baseline_comparison(result, baseline_result)
    fold_stability = _build_fold_stability(compared, baseline_result)
    cost_scenarios = _evaluate_cost_scenarios(compared, scenario_specs)
    honesty_flags = _build_honesty_flags(
        compared,
        min_honest_trades=min_honest_trades,
        min_honest_coverage=min_honest_coverage,
    )
    return {
        **compared,
        **fold_stability,
        "cost_scenario_results": cost_scenarios,
        "honesty_flags": honesty_flags,
        "has_honesty_flags": bool(honesty_flags),
    }


def _flatten_candidate_row(result: Mapping[str, Any]) -> dict[str, Any]:
    weakest_fold = result.get("weakest_fold") or {}
    current_scenario = _find_cost_scenario(result, "current_fee")
    double_fee_scenario = _find_cost_scenario(result, "double_fee")
    slippage_scenario = _find_cost_scenario(result, "current_fee_plus_10bps_slippage")
    return {
        "threshold": result["threshold"],
        "prediction_count": result["prediction_count"],
        "trade_count": result["trade_count"],
        "coverage": result["coverage"],
        "abstention_count": result["abstention_count"],
        "abstention_rate": result["abstention_rate"],
        "precision_on_trades": result["precision_on_trades"],
        "mean_long_only_gross_value_proxy": result["mean_long_only_gross_value_proxy"],
        "mean_long_only_net_value_proxy": result["mean_long_only_net_value_proxy"],
        "cumulative_long_only_gross_value_proxy": result["cumulative_long_only_gross_value_proxy"],
        "cumulative_long_only_net_value_proxy": result["cumulative_long_only_net_value_proxy"],
        "max_drawdown_proxy": result["max_drawdown_proxy"],
        "after_cost_positive": result["after_cost_positive"],
        "beats_baseline_mean_net": result["beats_baseline_mean_net"],
        "delta_vs_baseline_mean_long_only_net_value_proxy": (
            result["delta_vs_baseline_mean_long_only_net_value_proxy"]
        ),
        "delta_vs_baseline_trade_count": result["delta_vs_baseline_trade_count"],
        "delta_vs_baseline_max_drawdown_proxy": result["delta_vs_baseline_max_drawdown_proxy"],
        "weakest_fold_index": weakest_fold.get("fold_index"),
        "weakest_fold_mean_net": weakest_fold.get("mean_long_only_net_value_proxy"),
        "fold_win_count_vs_baseline": result["fold_win_count_vs_baseline"],
        "fold_count": result["fold_count"],
        "gains_concentrated_in_one_fold": result["gains_concentrated_in_one_fold"],
        "honesty_flags": ";".join(result["honesty_flags"]),
        "current_fee_mean_net": current_scenario.get("mean_long_only_net_value_proxy"),
        "double_fee_mean_net": double_fee_scenario.get("mean_long_only_net_value_proxy"),
        "current_fee_plus_10bps_slippage_mean_net": slippage_scenario.get(
            "mean_long_only_net_value_proxy"
        ),
    }


def _build_report_markdown(summary: Mapping[str, Any]) -> str:
    best_candidate = summary["best_candidate"]
    baseline_result = summary["baseline_result"]
    lines = [
        "# M20 Policy Evaluation",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Model analyzed: `{summary['model_name']}`",
        f"- Baseline threshold: `{float(summary['baseline_threshold']):.2f}`",
        f"- Fee rate: `{float(summary['fee_rate']):.6f}`",
        "",
        "## Best Candidate",
        "",
        (
            f"- Best threshold: `{float(best_candidate['threshold']):.2f}` "
            f"(trade_count={int(best_candidate['trade_count'])}, "
            f"coverage={float(best_candidate['coverage']):.4f}, "
            f"mean_net={float(best_candidate['mean_long_only_net_value_proxy']):.6f}, "
            f"max_drawdown={float(best_candidate['max_drawdown_proxy']):.6f})"
        ),
        (
            "- Delta vs baseline mean net: "
            f"`{float(best_candidate['delta_vs_baseline_mean_long_only_net_value_proxy']):.6f}`"
        ),
        (
            "- Fold stability: "
            f"`{int(best_candidate['fold_win_count_vs_baseline'])}/"
            f"{int(best_candidate['fold_count'])}` folds beat baseline; "
            f"weakest_fold=`{_format_optional_fold(best_candidate.get('weakest_fold'))}`"
        ),
        f"- Honesty flags: `{_format_honesty_flags(best_candidate['honesty_flags'])}`",
        "",
        "## Baseline",
        "",
        (
            f"- Baseline threshold `{float(summary['baseline_threshold']):.2f}`: "
            f"(trade_count={int(baseline_result['trade_count'])}, "
            f"coverage={float(baseline_result['coverage']):.4f}, "
            f"mean_net={float(baseline_result['mean_long_only_net_value_proxy']):.6f}, "
            f"max_drawdown={float(baseline_result['max_drawdown_proxy']):.6f})"
        ),
        "",
        "## Cost Scenarios",
        "",
        "| Scenario | Cost per trade | Mean net | Cumulative net | After-cost positive |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for scenario in best_candidate["cost_scenario_results"]:
        lines.append(
            "| "
            f"{scenario['scenario_name']} | "
            f"{float(scenario['cost_per_trade']):.6f} | "
            f"{float(scenario['mean_long_only_net_value_proxy']):.6f} | "
            f"{float(scenario['cumulative_long_only_net_value_proxy']):.6f} | "
            f"{scenario['after_cost_positive']} |"
        )
    lines.extend(
        [
            "",
            "## Recovery Interpretation",
            "",
            "This report is a refusal-to-trade research aid. A threshold can look better by ",
            "trading less, so coverage, trade count, fold stability, regime slices, and ",
            "cost scenarios must be read together before drawing conclusions.",
            "",
            "This evaluator is research-only M20 recovery infrastructure. It does not change ",
            "runtime rosters, promotion semantics, inference behavior, or execution policy.",
            "",
        "## Output Files",
        "",
        ]
    )
    for label, path in summary["output_files"].items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _build_fold_stability(
    result: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_by_fold = {
        int(row["fold_index"]): row
        for row in baseline_result.get("per_fold_breakdown", [])
    }
    per_fold = [dict(row) for row in result.get("per_fold_breakdown", [])]
    fold_win_count = 0
    for row in per_fold:
        baseline_row = baseline_by_fold.get(int(row["fold_index"]))
        baseline_mean_net = (
            float(baseline_row["mean_long_only_net_value_proxy"])
            if baseline_row is not None
            else 0.0
        )
        row["delta_vs_baseline_mean_long_only_net_value_proxy"] = (
            float(row["mean_long_only_net_value_proxy"]) - baseline_mean_net
        )
        if float(row["mean_long_only_net_value_proxy"]) > baseline_mean_net:
            fold_win_count += 1
    weakest_fold = _select_weakest_fold(per_fold)
    positive_deltas = [
        float(row["delta_vs_baseline_mean_long_only_net_value_proxy"])
        for row in per_fold
        if float(row["delta_vs_baseline_mean_long_only_net_value_proxy"]) > 0.0
    ]
    total_positive_delta = sum(positive_deltas)
    biggest_positive_delta = max(positive_deltas) if positive_deltas else 0.0
    return {
        "per_fold_breakdown": per_fold,
        "weakest_fold": weakest_fold,
        "fold_count": len(per_fold),
        "fold_win_count_vs_baseline": fold_win_count,
        "gains_concentrated_in_one_fold": (
            len(positive_deltas) > 1
            and total_positive_delta > 0.0
            and biggest_positive_delta / total_positive_delta >= 0.80
        ),
    }


def _select_weakest_fold(per_fold: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not per_fold:
        return None
    return dict(
        sorted(
            per_fold,
            key=lambda row: (
                float(row["mean_long_only_net_value_proxy"]),
                float(row["cumulative_long_only_net_value_proxy"]),
                int(row["fold_index"]),
            ),
        )[0]
    )


def _build_honesty_flags(
    result: Mapping[str, Any],
    *,
    min_honest_trades: int,
    min_honest_coverage: float,
) -> list[str]:
    flags: list[str] = []
    if int(result["trade_count"]) == 0:
        flags.append("ZERO_TRADES")
    elif int(result["trade_count"]) < int(min_honest_trades):
        flags.append("LOW_TRADE_COUNT")
    if float(result["coverage"]) < float(min_honest_coverage):
        flags.append("LOW_COVERAGE")
    if result.get("gains_concentrated_in_one_fold"):
        flags.append("GAINS_CONCENTRATED_IN_ONE_FOLD")
    return flags


def _build_cost_scenario_specs(
    *,
    fee_rate: float,
    slippage_rates: Sequence[float],
) -> list[dict[str, Any]]:
    normalized_slippage = sorted(
        {
            round(float(slippage_rate), 6)
            for slippage_rate in slippage_rates
            if float(slippage_rate) >= 0.0
        }
    )
    scenarios = [
        {
            "scenario_name": "current_fee",
            "fee_multiplier": 1.0,
            "slippage_rate": 0.0,
            "cost_per_trade": float(fee_rate),
        },
        {
            "scenario_name": "double_fee",
            "fee_multiplier": 2.0,
            "slippage_rate": 0.0,
            "cost_per_trade": float(fee_rate) * 2.0,
        },
    ]
    for slippage_rate in normalized_slippage:
        if slippage_rate <= 0.0:
            continue
        scenarios.append(
            {
                "scenario_name": f"current_fee_plus_{int(slippage_rate * 10000)}bps_slippage",
                "fee_multiplier": 1.0,
                "slippage_rate": slippage_rate,
                "cost_per_trade": float(fee_rate) + slippage_rate,
            }
        )
    return scenarios


def _evaluate_cost_scenarios(
    result: Mapping[str, Any],
    scenario_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    trade_count = int(result["trade_count"])
    prediction_count = int(result["prediction_count"])
    gross_sum = float(result["cumulative_long_only_gross_value_proxy"])
    scenario_rows = []
    for scenario in scenario_specs:
        cost_per_trade = float(scenario["cost_per_trade"])
        cumulative_net = gross_sum - (trade_count * cost_per_trade)
        mean_net = cumulative_net / prediction_count if prediction_count else 0.0
        scenario_rows.append(
            {
                **dict(scenario),
                "trade_count": trade_count,
                "prediction_count": prediction_count,
                "cumulative_long_only_net_value_proxy": cumulative_net,
                "mean_long_only_net_value_proxy": mean_net,
                "after_cost_positive": mean_net > 0.0,
            }
        )
    return scenario_rows


def _find_cost_scenario(
    result: Mapping[str, Any],
    scenario_name: str,
) -> dict[str, Any]:
    for scenario in result.get("cost_scenario_results", []):
        if scenario.get("scenario_name") == scenario_name:
            return dict(scenario)
    return {}


def _format_optional_fold(fold: Any) -> str:
    if not isinstance(fold, Mapping):
        return "none"
    return (
        f"{int(fold['fold_index'])}:"
        f"{float(fold['mean_long_only_net_value_proxy']):.6f}"
    )


def _format_honesty_flags(flags: Sequence[str]) -> str:
    return ", ".join(flags) if flags else "none"


def _normalize_threshold_grid(thresholds: Iterable[float]) -> list[float]:
    normalized = [_normalize_threshold(threshold) for threshold in thresholds]
    if not normalized:
        raise ValueError("Policy evaluation requires at least one threshold candidate")
    return sorted(dict.fromkeys(normalized))


def _normalize_threshold(value: float) -> float:
    normalized = round(float(value), 6)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"Invalid threshold candidate outside [0, 1]: {normalized}")
    return normalized


def _policy_result_sort_key(result: Mapping[str, Any]) -> tuple[float, float, int, float]:
    return (
        -float(result["mean_long_only_net_value_proxy"]),
        -float(result["cumulative_long_only_net_value_proxy"]),
        -int(result["trade_count"]),
        float(result["threshold"]),
    )


def _select_trade_rows(rows: list[PolicyEvalRow], threshold: float) -> list[PolicyEvalRow]:
    return [row for row in rows if row.prob_up >= threshold]


def _group_rows_by_fold(rows: list[PolicyEvalRow]) -> dict[int, list[PolicyEvalRow]]:
    grouped: dict[int, list[PolicyEvalRow]] = {}
    for row in rows:
        grouped.setdefault(row.fold_index, []).append(row)
    return grouped


def _order_rows_for_drawdown(rows: list[PolicyEvalRow]) -> list[PolicyEvalRow]:
    return sorted(
        rows,
        key=lambda row: (
            row.interval_begin or "",
            row.as_of_time or "",
            row.source_index,
        ),
    )


def _compute_max_drawdown_proxy(rows: list[PolicyEvalRow]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in rows:
        cumulative += float(row.long_only_net_value_proxy)
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return max_drawdown


def _row_from_csv(raw_row: Mapping[str, str], source_index: int) -> PolicyEvalRow:
    regime_label = (raw_row.get("regime_label") or "").strip() or None
    interval_begin = (raw_row.get("interval_begin") or "").strip() or None
    as_of_time = (raw_row.get("as_of_time") or "").strip() or None
    return PolicyEvalRow(
        model_name=str(raw_row["model_name"]),
        fold_index=int(raw_row["fold_index"]),
        row_id=str(raw_row["row_id"]),
        y_true=int(raw_row["y_true"]),
        prob_up=float(raw_row["prob_up"]),
        future_return_3=float(raw_row["future_return_3"]),
        long_only_gross_value_proxy=float(raw_row["long_only_gross_value_proxy"]),
        long_only_net_value_proxy=float(raw_row["long_only_net_value_proxy"]),
        source_index=source_index,
        regime_label=regime_label,
        interval_begin=interval_begin,
        as_of_time=as_of_time,
    )


def _load_completed_summary(run_dir: Path) -> dict[str, Any]:
    try:
        summary_payload = load_summary_payload(run_dir)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Completed run summary.json is not valid JSON: {run_dir / 'summary.json'}"
        ) from error
    if not isinstance(summary_payload, Mapping):
        raise ValueError(
            f"Completed run summary.json must contain a JSON object: {run_dir / 'summary.json'}"
        )
    return dict(summary_payload)


def _resolve_completed_winner_model_name(summary_payload: Mapping[str, Any]) -> str:
    winner_model_name = resolve_winner_model_name(summary_payload).strip()
    if not winner_model_name:
        raise ValueError("Completed run summary winner.model_name is blank")
    return winner_model_name


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
