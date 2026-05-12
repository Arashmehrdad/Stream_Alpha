"""Generic research-only M20 decision-policy evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_policy_research_common import (
    HONESTY_FLAGS,
    KEY_WITH_FOLD,
    MIN_POLICY_ROWS,
    economics_metrics,
    group_rows,
    keyed_rows,
    present,
    read_csv_rows,
    row_key,
    to_float,
    to_int,
    truthy,
    vol_scaled_dir,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_OUTPUT_NAME = "decision_policy_eval"
DEFAULT_CANDIDATE_DIR_NAME = "strategy_candidate_v2_refined_factory"
DEFAULT_ECONOMIC_DIR_NAME = "economic_outcome_artifacts"
DEFAULT_LABEL_DIR_NAME = "trading_aware_labels"
DEFAULT_RESEARCH_INPUT_FILE = "multi_horizon_labels.csv"
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
INTERSECTION_THRESHOLDS = (0.60, 0.70)


def evaluate_m20_decision_policies(
    *,
    source_run_dir: Path,
    prediction_run_dir: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
    candidate_dir: Path | None = None,
    economic_outcome_dir: Path | None = None,
    trading_aware_label_dir: Path | None = None,
    research_input_dir: Path | None = None,
    label_column: str = "fee_plus_slippage_exceedance_label",
) -> dict[str, Any]:
    """Evaluate generic TAKE/HOLD policies from existing offline artifacts."""
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    source_dir = Path(source_run_dir).resolve()
    prediction_dir = Path(prediction_run_dir).resolve()
    research_dir = vol_scaled_dir(source_dir)
    output_dir = research_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _input_paths(
        research_dir,
        prediction_dir,
        candidate_dir,
        economic_outcome_dir,
        trading_aware_label_dir,
        research_input_dir,
    )
    oof_rows = read_csv_rows(paths["oof_predictions"])
    outcome_rows = read_csv_rows(paths["economic_outcomes"])
    candidate_rows = read_csv_rows(paths["strategy_candidates"])
    label_rows = read_csv_rows(paths["trading_aware_labels"])
    research_input_rows = read_csv_rows(paths["research_inputs"])
    if not oof_rows:
        raise ValueError(f"Missing OOF predictions: {paths['oof_predictions']}")
    if not outcome_rows:
        raise ValueError(f"Missing economic outcomes: {paths['economic_outcomes']}")
    outcome_index = keyed_rows(outcome_rows, KEY_WITH_FOLD)
    label_index = keyed_rows(label_rows, KEY_WITH_FOLD)
    research_input_index = keyed_rows(research_input_rows, KEY_WITH_FOLD)
    enriched_rows = _enrich_oof_rows(
        oof_rows,
        outcome_index,
        label_index,
        research_input_index,
        label_column,
    )
    candidate_keys = _candidate_keys(candidate_rows)
    policies = _policy_specs(enriched_rows, candidate_keys)
    rows_by_model = _rows_by_model(enriched_rows)
    evaluated = [_evaluate_policy(policy, rows_by_model) for policy in policies]
    metrics = [row["metric"] for row in evaluated]
    decisions = _policy_decisions(metrics)
    baseline_rows = [row for row in metrics if row["policy_family"] == "BASELINE"]
    baseline_comparison = _baseline_comparison(metrics, baseline_rows)
    calibration = _calibration_metrics(enriched_rows)
    by_symbol = _slice_metrics(evaluated, "symbol")
    by_regime = _slice_metrics(evaluated, "regime_label")
    by_time = _slice_metrics(evaluated, "month")
    by_time.extend(_slice_metrics(evaluated, "quarter"))
    search_breadth = _search_breadth(policies, candidate_rows)
    recommendation = _recommendation(decisions)
    output_files = _output_files(output_dir)
    manifest = {
        "source_run_dir": str(source_dir),
        "prediction_run_dir": str(prediction_dir),
        "input_paths": {name: str(path) for name, path in paths.items()},
        "diagnostic_label_column": label_column,
        "policy_count": len(metrics),
        "search_breadth": search_breadth,
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    report = {
        "summary": "Generic M20 research-only decision-policy evaluator.",
        "policy_count": len(metrics),
        "event_count": len(enriched_rows),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "best_policy_candidate": _best_policy(metrics),
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["decision_policy_report_json"]), report)
    Path(output_files["decision_policy_report_md"]).write_text(
        _markdown(report, decisions),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["policy_metrics_csv"]), metrics)
    write_csv_artifact(Path(output_files["policy_candidates_csv"]), _policy_rows(policies))
    write_csv_artifact(Path(output_files["baseline_comparison_csv"]), baseline_comparison)
    write_csv_artifact(Path(output_files["calibration_metrics_csv"]), calibration)
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_regime_csv"]), by_regime)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_csv_artifact(Path(output_files["search_breadth_csv"]), [search_breadth])
    write_csv_artifact(Path(output_files["candidate_decisions_csv"]), decisions)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "policy_metrics": metrics,
            "candidate_decisions": decisions,
            "baseline_comparison": baseline_comparison,
            "calibration_metrics": calibration,
            "search_breadth": search_breadth,
            "recommendation_payload": recommendation,
        }
    )


def _input_paths(
    research_dir: Path,
    prediction_dir: Path,
    candidate_dir: Path | None,
    economic_outcome_dir: Path | None,
    trading_aware_label_dir: Path | None,
    research_input_dir: Path | None,
) -> dict[str, Path]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    candidates = (
        Path(candidate_dir).resolve()
        if candidate_dir
        else research_dir / DEFAULT_CANDIDATE_DIR_NAME
    )
    outcomes = (
        Path(economic_outcome_dir).resolve()
        if economic_outcome_dir
        else research_dir / DEFAULT_ECONOMIC_DIR_NAME
    )
    labels = (
        Path(trading_aware_label_dir).resolve()
        if trading_aware_label_dir
        else research_dir / DEFAULT_LABEL_DIR_NAME
    )
    research_inputs = Path(research_input_dir).resolve() if research_input_dir else None
    return {
        "oof_predictions": prediction_dir / "oof_predictions.csv",
        "economic_outcomes": outcomes / "economic_outcomes.csv",
        "strategy_candidates": candidates / "strategy_candidates_v2.csv",
        "trading_aware_labels": labels / "trading_aware_labels.csv",
        "research_inputs": (
            research_inputs / DEFAULT_RESEARCH_INPUT_FILE
            if research_inputs
            else Path("__missing_optional_research_inputs__.csv")
        ),
    }


def _enrich_oof_rows(
    oof_rows: Sequence[Mapping[str, str]],
    outcome_index: Mapping[tuple[str, ...], Mapping[str, Any]],
    label_index: Mapping[tuple[str, ...], Mapping[str, Any]],
    research_input_index: Mapping[tuple[str, ...], Mapping[str, Any]],
    label_column: str,
) -> list[dict[str, Any]]:
    output = []
    for row in oof_rows:
        key = row_key(row, KEY_WITH_FOLD)
        outcome = outcome_index.get(key)
        if outcome is None:
            continue
        label = label_index.get(key, {})
        research_input = research_input_index.get(key, {})
        output.append(
            {
                **row,
                "net_value_proxy": outcome.get("net_value_proxy", ""),
                "gross_value_proxy": outcome.get("gross_value_proxy", ""),
                "fee_exceedance_label": outcome.get("fee_exceedance_label", ""),
                "triple_barrier_label": outcome.get("triple_barrier_label", ""),
                "trading_aware_label": label.get("fee_plus_slippage_exceedance_label", ""),
                "redesigned_label": research_input.get(label_column, ""),
            }
        )
    return output


def _candidate_keys(
    candidate_rows: Sequence[Mapping[str, str]],
) -> dict[str, set[tuple[str, ...]]]:
    output: dict[str, set[tuple[str, ...]]] = {}
    for row in candidate_rows:
        name = str(row.get("candidate_name", ""))
        if not name:
            continue
        output.setdefault(name, set()).add(row_key(row, KEY_WITH_FOLD))
    return output


def _policy_specs(
    rows: Sequence[Mapping[str, Any]],
    candidate_keys: Mapping[str, set[tuple[str, ...]]],
) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    models = sorted({str(row.get("model_name", "")) for row in rows if row.get("model_name")})
    for model_name in models:
        policies.extend(_model_policy_specs(model_name, rows, candidate_keys))
    return policies


def _model_policy_specs(
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
    candidate_keys: Mapping[str, set[tuple[str, ...]]],
) -> list[dict[str, Any]]:
    policies = [
        _policy(
            "BASELINE",
            f"{model_name}:BASELINE_LONG_TRADE_TAKEN",
            model_name,
            lambda row: truthy(row.get("long_trade_taken")),
            "Default/incumbent-equivalent baseline when available.",
        )
    ]
    for threshold in THRESHOLDS:
        policies.append(
            _policy(
                "PROBABILITY_THRESHOLD",
                f"{model_name}:PROB_UP_GTE_{threshold:.2f}",
                model_name,
                lambda row, value=threshold: to_float(row.get("prob_up")) >= value,
                "Generic probability threshold.",
            )
        )
        policies.append(
            _policy(
                "CONFIDENCE_THRESHOLD",
                f"{model_name}:CONFIDENCE_GTE_{threshold:.2f}",
                model_name,
                lambda row, value=threshold: to_float(row.get("confidence")) >= value,
                "Generic confidence threshold.",
            )
        )
    regimes = sorted({str(row.get("regime_label", "")) for row in rows if row.get("regime_label")})
    for regime in regimes:
        policies.append(
            _policy(
                "REGIME_CONDITIONAL_THRESHOLD",
                f"{model_name}:REGIME_{regime}_PROB_UP_GTE_0.60",
                model_name,
                lambda row, value=regime: (
                    row.get("regime_label") == value
                    and to_float(row.get("prob_up")) >= 0.60
                ),
                "Regime-specific probability threshold.",
            )
        )
    for candidate_name, keys in sorted(candidate_keys.items()):
        policies.append(
            _candidate_policy(model_name, candidate_name, keys, None)
        )
        for threshold in INTERSECTION_THRESHOLDS:
            policies.append(
                _candidate_policy(model_name, candidate_name, keys, threshold)
            )
    return policies


def _rows_by_model(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("model_name", "")), []).append(row)
    return grouped


def _policy(
    family: str,
    name: str,
    model_name: str,
    selector: Callable[[Mapping[str, Any]], bool],
    description: str,
) -> dict[str, Any]:
    return {
        "policy_family": family,
        "policy_name": name,
        "model_name": model_name,
        "selector": selector,
        "description": description,
    }


def _candidate_policy(
    model_name: str,
    candidate_name: str,
    keys: set[tuple[str, ...]],
    threshold: float | None,
) -> dict[str, Any]:
    if threshold is None:
        return _policy(
            "CANDIDATE_EVENT_POLICY",
            f"{model_name}:CANDIDATE_{candidate_name}",
            model_name,
            lambda row, candidate_keys=keys: row_key(row, KEY_WITH_FOLD) in candidate_keys,
            "Candidate-event-conditioned TAKE policy.",
        )
    return _policy(
        "CANDIDATE_PLUS_SCORE_POLICY",
        f"{model_name}:CANDIDATE_{candidate_name}_PROB_UP_GTE_{threshold:.2f}",
        model_name,
        lambda row, candidate_keys=keys, value=threshold: (
            row_key(row, KEY_WITH_FOLD) in candidate_keys
            and to_float(row.get("prob_up")) >= value
        ),
        "Candidate-event policy intersected with probability threshold.",
    )


def _evaluate_policy(
    policy: Mapping[str, Any],
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    model_rows = rows_by_model.get(str(policy["model_name"]), [])
    selector = policy["selector"]
    selected = [row for row in model_rows if selector(row)]
    return {
        "policy": policy,
        "selected_rows": selected,
        "metric": _policy_metric(policy, len(model_rows), selected),
    }


def _policy_metric(
    policy: Mapping[str, Any],
    event_count: int,
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    economics = economics_metrics(selected, "net_value_proxy")
    selected_count = len(selected)
    return {
        "policy_family": policy["policy_family"],
        "policy_name": policy["policy_name"],
        "model_name": policy["model_name"],
        "event_count": event_count,
        "selected_rows": selected_count,
        "coverage": selected_count / event_count if event_count else 0.0,
        "abstention_rate": 1.0 - (selected_count / event_count) if event_count else 0.0,
        "low_sample_warning": str(0 < selected_count < MIN_POLICY_ROWS),
        "classification": _classify_policy(selected_count, economics),
        **economics,
    }


def _classify_policy(selected_count: int, economics: Mapping[str, Any]) -> str:
    if selected_count == 0:
        return "POLICY_INSUFFICIENT_EVIDENCE"
    if selected_count < MIN_POLICY_ROWS:
        return "POLICY_LOW_SAMPLE"
    if present(economics.get("mean_net_value_proxy")) and float(
        economics["mean_net_value_proxy"]
    ) > 0.0:
        return "POLICY_RESEARCH_WATCHLIST_POSITIVE_PROXY"
    return "POLICY_ECONOMICS_NEGATIVE"


def _policy_decisions(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy_name": row["policy_name"],
            "policy_family": row["policy_family"],
            "model_name": row["model_name"],
            "policy_decision": row["classification"],
            "selected_rows": row["selected_rows"],
            "coverage": row["coverage"],
            "mean_net_value_proxy": row["mean_net_value_proxy"],
            "runtime_status": "NO_RUNTIME_EFFECT",
            "promotion_status": "NOT_PROMOTABLE",
            "profitability_status": "NO_PROFIT_CLAIM",
        }
        for row in metrics
    ]


def _baseline_comparison(
    metrics: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_model = {row["model_name"]: row for row in baseline_rows}
    output = []
    for row in metrics:
        baseline = baseline_by_model.get(row["model_name"])
        if baseline is None or row["policy_family"] == "BASELINE":
            continue
        output.append(
            {
                "policy_name": row["policy_name"],
                "model_name": row["model_name"],
                "baseline_policy_name": baseline["policy_name"],
                "policy_mean_net_value_proxy": row["mean_net_value_proxy"],
                "baseline_mean_net_value_proxy": baseline["mean_net_value_proxy"],
                "mean_net_delta_vs_baseline": _delta(
                    row["mean_net_value_proxy"],
                    baseline["mean_net_value_proxy"],
                ),
                "paired_comparison_status": "PROXY_COMPARISON_ONLY_NOT_BACKTEST",
            }
        )
    return output


def _calibration_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for label_column in ("y_true", "trading_aware_label", "redesigned_label"):
        usable = [
            row for row in rows
            if present(row.get("prob_up")) and present(row.get(label_column))
        ]
        if not usable:
            continue
        brier = sum(
            (to_float(row["prob_up"]) - to_int(row[label_column])) ** 2
            for row in usable
        ) / len(usable)
        output.append(
            {
                "label_column": label_column,
                "row_count": len(usable),
                "brier_score": brier,
                "calibration_status": "RESEARCH_DIAGNOSTIC_ONLY",
            }
        )
    return output


def _slice_metrics(
    evaluated: Sequence[Mapping[str, Any]],
    slice_field: str,
) -> list[dict[str, Any]]:
    output = []
    for item in evaluated:
        metric = item["metric"]
        selected = item["selected_rows"]
        for value, slice_rows in sorted(group_rows(selected, slice_field).items()):
            economics = economics_metrics(slice_rows, "net_value_proxy")
            output.append(
                {
                    "policy_name": metric["policy_name"],
                    "model_name": metric["model_name"],
                    "slice_field": slice_field,
                    "slice_value": value,
                    "selected_rows": len(slice_rows),
                    **economics,
                }
            )
    return output


def _search_breadth(
    policies: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "policy_configurations_tried": len(policies),
        "candidate_definitions_referenced": len(
            {row.get("candidate_name", "") for row in candidate_rows if row.get("candidate_name")}
        ),
        "threshold_values_tried": len(THRESHOLDS),
        "intersection_threshold_values_tried": len(INTERSECTION_THRESHOLDS),
        "search_breadth_warning": "MULTIPLE_COMPARISON_RESEARCH_ONLY",
    }


def _policy_rows(policies: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy_family": row["policy_family"],
            "policy_name": row["policy_name"],
            "model_name": row["model_name"],
            "description": row["description"],
            "selection_uses_outcome_columns": "False",
        }
        for row in policies
    ]


def _recommendation(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = any(
        row["policy_decision"] == "POLICY_RESEARCH_WATCHLIST_POSITIVE_PROXY"
        for row in decisions
    )
    recommendation = (
        "RUN_POLICY_VALIDATION_AUDIT"
        if positive
        else "DESIGN_TRADING_AWARE_RESEARCH_LABELS"
    )
    return {
        "recommendation": recommendation,
        "next_required_action": recommendation,
        "evidence_blockers": [
            "NOT_BACKTEST",
            "NOT_RUNTIME_READY",
            "NOT_PROMOTABLE",
            "NO_PROFIT_CLAIM",
        ],
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _best_policy(metrics: Sequence[Mapping[str, Any]]) -> str:
    candidates = [
        row for row in metrics
        if present(row.get("mean_net_value_proxy"))
    ]
    if not candidates:
        return ""
    best = max(candidates, key=lambda row: float(row["mean_net_value_proxy"]))
    return str(best["policy_name"])


def _next_actions(recommendation: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action": str(recommendation["next_required_action"]),
            "rationale": "Continue generic research-only policy evaluation.",
        }
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "decision_policy_report_json": str(output_dir / "decision_policy_report.json"),
        "decision_policy_report_md": str(output_dir / "decision_policy_report.md"),
        "policy_metrics_csv": str(output_dir / "policy_metrics.csv"),
        "policy_candidates_csv": str(output_dir / "policy_candidates.csv"),
        "baseline_comparison_csv": str(output_dir / "baseline_comparison.csv"),
        "calibration_metrics_csv": str(output_dir / "calibration_metrics.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_regime_csv": str(output_dir / "by_regime.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "search_breadth_csv": str(output_dir / "search_breadth.csv"),
        "candidate_decisions_csv": str(output_dir / "candidate_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> str:
    grouped: dict[str, int] = {}
    for row in decisions:
        grouped[row["policy_decision"]] = grouped.get(row["policy_decision"], 0) + 1
    lines = [
        "# M20 Decision Policy Evaluation",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Policy count: `{report['policy_count']}`",
        f"- Best proxy policy candidate: `{report['best_policy_candidate']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Policy Decisions",
    ]
    lines.extend(f"- `{decision}`: `{count}`" for decision, count in sorted(grouped.items()))
    return "\n".join(lines) + "\n"


def _delta(left: Any, right: Any) -> Any:
    if not present(left) or not present(right):
        return ""
    return to_float(left) - to_float(right)


__all__ = ["evaluate_m20_decision_policies"]
