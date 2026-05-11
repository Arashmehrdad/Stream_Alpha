"""Generic research-only M20 cost-aware specialist policy evaluator."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_LABEL_FILE = "research_labels/vol_scaled/fee_exceedance_labels_vol_scaled.csv"
DEFAULT_OUTPUT_NAME = "cost_aware_specialist_policy_evaluator"
DEFAULT_EDGE_EVALUATOR_NAME = "specialist_edge_evaluator"
TOP_K_POLICIES = (
    ("TOP_1_PERCENT", 0.01),
    ("TOP_2_PERCENT", 0.02),
    ("TOP_5_PERCENT", 0.05),
    ("TOP_10_PERCENT", 0.10),
)
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
SAFE_NET_COLUMNS = (
    "long_only_net_value_proxy",
    "net_value_proxy",
    "safe_net_value_proxy",
    "net_proxy",
)
ECONOMIC_OUTCOME_FILE = "economic_outcomes.csv"
NO_NET_PROXY = "NET_PROXY_NOT_AVAILABLE"
ECONOMIC_REQUIRED = "ECONOMIC_POLICY_EVALUATION_REQUIRED"
NEXT_ACTION_NO_ECONOMICS = "DESIGN_SAFE_ECONOMIC_OUTCOME_ARTIFACTS_FOR_SPECIALIST_POLICIES"
OVERALL_STATUS = (
    "RESEARCH_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
    "NOT_BACKTEST",
)
HONESTY_FLAGS = (
    "RESEARCH_ONLY_COST_AWARE_SPECIALIST_POLICY_EVALUATOR",
    "EXISTING_ARTIFACTS_ONLY",
    "NO_MODEL_RETRAIN",
    "NO_RUNTIME_EFFECT",
    "NO_REGISTRY_WRITE",
    "NO_PROMOTION_EFFECT",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NOT_BACKTEST",
    "NO_PROFIT_CLAIM",
)


def analyze_m20_cost_aware_specialist_policy(
    *,
    prediction_run_dir: Path,
    label_source_run_dir: Path,
    prediction_source: str,
    models: Sequence[str] | None = None,
    edge_evaluator_dir: Path | None = None,
    economic_outcome_dir: Path | None = None,
    label_file: str = DEFAULT_LABEL_FILE,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Evaluate generic specialist policies from existing artifacts."""
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    prediction_dir = Path(prediction_run_dir).resolve()
    label_run_dir = Path(label_source_run_dir).resolve()
    specialist_dir = prediction_dir / "research_labels" / "vol_scaled" / "specialist_predictions"
    output_dir = prediction_dir / "research_labels" / "vol_scaled" / output_name
    edge_dir = (
        Path(edge_evaluator_dir).resolve()
        if edge_evaluator_dir is not None
        else prediction_dir / "research_labels" / "vol_scaled" / DEFAULT_EDGE_EVALUATOR_NAME
    )
    outcome_dir = (
        Path(economic_outcome_dir).resolve()
        if economic_outcome_dir is not None
        else label_run_dir / "research_labels" / "vol_scaled" / "economic_outcome_artifacts"
    )
    label_path = label_run_dir / label_file

    discovered_models = _discover_models(specialist_dir, prediction_source)
    selected_models = _selected_models(models, discovered_models)
    prediction_files = {
        model_name: specialist_dir / f"predictions_{model_name}_{prediction_source}.csv"
        for model_name in selected_models
    }
    _assert_prediction_files(prediction_files)

    labels = _read_csv(label_path, "Missing label file")
    safe_net_column = _safe_net_column(labels)
    label_index, fallback_label_index = _label_indexes(labels)
    outcome_index, outcome_fallback_index = _economic_outcome_indexes(outcome_dir)
    economics_available = safe_net_column is not None or bool(outcome_index)
    edge_slices = _load_edge_slices(edge_dir)

    joined_by_model = {}
    source_rows_by_model = {}
    for model_name, prediction_file in prediction_files.items():
        predictions = _read_csv(prediction_file, "Missing specialist prediction file")
        source_rows_by_model[model_name] = len(predictions)
        joined_by_model[model_name] = _join_rows(
            model_name=model_name,
            predictions=predictions,
            label_index=label_index,
            fallback_label_index=fallback_label_index,
            outcome_index=outcome_index,
            outcome_fallback_index=outcome_fallback_index,
            safe_net_column=safe_net_column,
        )
    economics_available = _joined_economics_available(joined_by_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    topk_policy_metrics = [
        row
        for model_name, rows in sorted(joined_by_model.items())
        for row in _topk_policy_metrics(model_name, rows, economics_available)
    ]
    threshold_policy_metrics = [
        row
        for model_name, rows in sorted(joined_by_model.items())
        for row in _threshold_policy_metrics(model_name, rows, economics_available)
    ]
    slice_policy_metrics = [
        row
        for model_name, rows in sorted(joined_by_model.items())
        for row in _slice_policy_metrics(
            model_name,
            rows,
            edge_slices.get(model_name, ()),
            economics_available,
        )
    ]
    policy_candidates = topk_policy_metrics + threshold_policy_metrics + slice_policy_metrics
    model_policy_metrics = _model_policy_metrics(policy_candidates)
    by_symbol = [
        row
        for model_name, rows in sorted(joined_by_model.items())
        for row in _group_metrics(model_name, rows, "symbol", "symbol", economics_available)
    ]
    by_time = [
        row
        for model_name, rows in sorted(joined_by_model.items())
        for family, key in (("month", "month"), ("quarter", "quarter"))
        for row in _group_metrics(model_name, rows, family, key, economics_available)
    ]
    candidate_decisions = _candidate_decisions(policy_candidates)
    recommendation = _recommendation(candidate_decisions, economics_available)
    evidence_blockers = _evidence_blockers(economics_available)
    output_files = _output_files(output_dir, bool(threshold_policy_metrics))
    economics_availability = {
        "economics_available": economics_available,
        "safe_net_column": safe_net_column or "",
        "economic_outcome_dir": str(outcome_dir),
        "economic_outcomes_present": bool(outcome_index),
        "evidence_blockers": evidence_blockers,
        "safe_source": _safe_source(economics_available, outcome_index, safe_net_column),
    }

    manifest = {
        "prediction_run_dir": str(prediction_dir),
        "label_source_run_dir": str(label_run_dir),
        "specialist_prediction_dir": str(specialist_dir),
        "edge_evaluator_dir": str(edge_dir),
        "edge_evaluator_present": edge_dir.exists(),
        "economic_outcome_dir": str(outcome_dir),
        "economic_outcomes_present": bool(outcome_index),
        "label_file": str(label_path),
        "prediction_source": prediction_source,
        "models": list(selected_models),
        "discovered_models": list(discovered_models),
        "joined_rows": sum(len(rows) for rows in joined_by_model.values()),
        "source_rows_by_model": source_rows_by_model,
        "joined_rows_by_model": {
            model_name: len(rows) for model_name, rows in sorted(joined_by_model.items())
        },
        "skipped_unlabeled_rows": sum(source_rows_by_model.values())
        - sum(len(rows) for rows in joined_by_model.values()),
        "economics_available": economics_available,
        "evidence_blockers": evidence_blockers,
        "overall_status": list(OVERALL_STATUS),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    report = {
        "summary": "Generic M20 cost-aware specialist policy evaluation from artifacts only.",
        "best_policy_candidate": _best_policy(candidate_decisions),
        "economics_available": economics_available,
        "evidence_blockers": evidence_blockers,
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "overall_status": list(OVERALL_STATUS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "candidate_decisions": candidate_decisions,
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["cost_aware_policy_report_json"]), report)
    Path(output_files["cost_aware_policy_report_md"]).write_text(
        _markdown(report, candidate_decisions),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["policy_candidates_csv"]), policy_candidates)
    write_csv_artifact(Path(output_files["model_policy_metrics_csv"]), model_policy_metrics)
    write_csv_artifact(Path(output_files["topk_policy_metrics_csv"]), topk_policy_metrics)
    if threshold_policy_metrics:
        write_csv_artifact(
            Path(output_files["threshold_policy_metrics_csv"]),
            threshold_policy_metrics,
        )
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_json_artifact(
        Path(output_files["economics_availability_json"]),
        economics_availability,
    )
    write_csv_artifact(Path(output_files["candidate_decisions_csv"]), candidate_decisions)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)

    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "economics_availability": economics_availability,
            "policy_candidates": policy_candidates,
            "model_policy_metrics": model_policy_metrics,
            "topk_policy_metrics": topk_policy_metrics,
            "threshold_policy_metrics": threshold_policy_metrics,
            "by_symbol": by_symbol,
            "by_time": by_time,
            "recommendation_payload": recommendation,
        }
    )


def _discover_models(specialist_dir: Path, prediction_source: str) -> tuple[str, ...]:
    if not specialist_dir.exists():
        raise ValueError(f"Missing specialist prediction directory: {specialist_dir}")
    suffix = f"_{prediction_source}.csv"
    models = sorted(
        path.name.removeprefix("predictions_").removesuffix(suffix)
        for path in specialist_dir.glob(f"predictions_*{suffix}")
        if path.name.startswith("predictions_")
    )
    if not models:
        raise ValueError(
            "No specialist prediction files found for prediction_source="
            f"{prediction_source!r} in {specialist_dir}"
        )
    return tuple(models)


def _selected_models(
    requested_models: Sequence[str] | None,
    discovered_models: Sequence[str],
) -> tuple[str, ...]:
    if requested_models is None:
        return tuple(discovered_models)
    selected = tuple(model.strip() for model in requested_models if model.strip())
    if not selected:
        raise ValueError("--models did not contain any model names")
    return selected


def _assert_prediction_files(prediction_files: Mapping[str, Path]) -> None:
    missing = [str(path) for path in prediction_files.values() if not path.exists()]
    if missing:
        raise ValueError("Missing specialist prediction file(s): " + "; ".join(missing))


def _read_csv(path: Path, missing_prefix: str) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"{missing_prefix}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_net_column(rows: Sequence[Mapping[str, str]]) -> str | None:
    if not rows:
        return None
    columns = set(rows[0])
    for column in SAFE_NET_COLUMNS:
        if column in columns:
            return column
    return None


def _label_indexes(
    rows: Sequence[Mapping[str, str]],
) -> tuple[
    dict[tuple[str, str, str], Mapping[str, str]],
    dict[tuple[str, str], Mapping[str, str]],
]:
    model_index = {}
    fallback_index = {}
    for row in rows:
        symbol = str(row.get("symbol", ""))
        interval_begin = str(row.get("interval_begin", ""))
        model_name = str(row.get("model_name", ""))
        if model_name:
            model_index[(model_name, symbol, interval_begin)] = row
        fallback_index[(symbol, interval_begin)] = row
    return model_index, fallback_index


def _join_rows(
    *,
    model_name: str,
    predictions: Sequence[Mapping[str, str]],
    label_index: Mapping[tuple[str, str, str], Mapping[str, str]],
    fallback_label_index: Mapping[tuple[str, str], Mapping[str, str]],
    outcome_index: Mapping[tuple[str, str, str], Mapping[str, str]],
    outcome_fallback_index: Mapping[tuple[str, str], Mapping[str, str]],
    safe_net_column: str | None,
) -> list[dict[str, Any]]:
    # pylint: disable=too-many-arguments,too-many-locals
    joined = []
    for row in predictions:
        symbol = str(row.get("symbol", ""))
        interval_begin = str(row.get("interval_begin", ""))
        row_model = str(row.get("model_name", model_name) or model_name)
        label = label_index.get((row_model, symbol, interval_begin))
        if label is None:
            label = fallback_label_index.get((symbol, interval_begin))
        if label is None:
            continue
        fold_index = str(row.get("fold_index", label.get("fold_index", "")))
        outcome = outcome_index.get((fold_index, symbol, interval_begin))
        if outcome is None:
            outcome = outcome_fallback_index.get((symbol, interval_begin))
        probability = _first_float(row, ("prob_up", "probability", "score"))
        joined.append(
            {
                "model_name": model_name,
                "symbol": symbol,
                "interval_begin": interval_begin,
                "probability": probability,
                "has_probability": probability is not None,
                "target": _to_int(label.get("label", label.get("y_true", "0"))),
                "month": interval_begin[:7],
                "quarter": _quarter(interval_begin),
                "net_proxy": _joined_float(outcome, label, safe_net_column, "net_value_proxy"),
                "gross_proxy": _joined_float(
                    outcome,
                    label,
                    safe_net_column,
                    "gross_value_proxy",
                ),
            }
        )
    return joined


def _topk_policy_metrics(
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
    economics_available: bool,
) -> list[dict[str, Any]]:
    return [
        _policy_metric(
            model_name,
            policy_name,
            "top_k",
            policy_value,
            rows,
            _topk_selection(rows, policy_value),
            economics_available,
        )
        for policy_name, policy_value in TOP_K_POLICIES
    ]


def _threshold_policy_metrics(
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
    economics_available: bool,
) -> list[dict[str, Any]]:
    if not any(bool(row.get("has_probability")) for row in rows):
        return []
    return [
        _policy_metric(
            model_name,
            f"THRESHOLD_{threshold:.2f}",
            "threshold",
            threshold,
            rows,
            [
                row for row in rows
                if row.get("probability") is not None
                and float(row["probability"]) >= threshold
            ],
            economics_available,
        )
        for threshold in THRESHOLDS
    ]


def _slice_policy_metrics(
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
    edge_slices: Sequence[Mapping[str, str]],
    economics_available: bool,
) -> list[dict[str, Any]]:
    output = []
    for edge_slice in edge_slices:
        family = str(edge_slice.get("slice_family", ""))
        value = str(edge_slice.get("slice_value", ""))
        if family not in ("symbol", "month", "quarter"):
            continue
        slice_rows = [row for row in rows if str(row.get(family, "")) == value]
        diagnostic = family in ("month", "quarter")
        output.append(
            _policy_metric(
                model_name,
                f"EDGE_SLICE_{family.upper()}_{value}",
                "edge_slice_diagnostic" if diagnostic else "edge_slice",
                0.05,
                rows,
                _topk_selection(slice_rows, 0.05),
                economics_available,
            )
        )
    return output


def _group_metrics(
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
    family: str,
    key: str,
    economics_available: bool,
) -> list[dict[str, Any]]:
    output = []
    for value, group_rows in sorted(_group_by(rows, key).items()):
        metric = _policy_metric(
            model_name,
            f"{family.upper()}_{value}",
            family,
            0.0,
            group_rows,
            group_rows,
            economics_available,
        )
        output.append({"slice_family": family, "slice_value": value, **metric})
    return output


def _policy_metric(
    model_name: str,
    policy_name: str,
    policy_type: str,
    policy_value: float,
    rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    economics_available: bool,
) -> dict[str, Any]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    selected_count = len(selected_rows)
    positive_count = sum(int(row["target"]) for row in rows)
    selected_positive = sum(int(row["target"]) for row in selected_rows)
    base_rate = _positive_rate(rows)
    precision = selected_positive / selected_count if selected_count else 0.0
    economics = _economic_metrics(selected_rows) if economics_available else _empty_economics()
    return {
        "model_name": model_name,
        "policy_name": policy_name,
        "policy_type": policy_type,
        "policy_value": policy_value,
        "selected_rows": selected_count,
        "coverage": selected_count / len(rows) if rows else 0.0,
        "base_positive_rate": base_rate,
        "selected_positive_rate": precision,
        "precision": precision,
        "lift_vs_base": precision / base_rate if base_rate > 0 else 0.0,
        "recall": selected_positive / positive_count if positive_count else 0.0,
        "false_positives": selected_count - selected_positive,
        "avg_probability": _mean(
            float(row["probability"]) for row in selected_rows
            if row.get("probability") is not None
        ),
        "policy_classification": _classify_policy(
            selected_count,
            precision / base_rate if base_rate > 0 else 0.0,
            economics_available,
            economics,
        ),
        **economics,
    }


def _economic_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    net_values = [float(row["net_proxy"]) for row in rows if row.get("net_proxy") is not None]
    gross_values = [
        float(row["gross_proxy"]) for row in rows if row.get("gross_proxy") is not None
    ]
    if not net_values:
        return _empty_economics()
    worst_tail, best_tail = _tail_means(net_values, 5)
    return {
        "economics_status": "NET_PROXY_AVAILABLE",
        "mean_net_proxy": _mean(net_values),
        "cumulative_net_proxy": sum(net_values),
        "max_drawdown_proxy": _max_drawdown(net_values),
        "expectancy_proxy": _mean(net_values),
        "win_rate_proxy": sum(1 for value in net_values if value > 0.0) / len(net_values),
        "mean_gross_proxy": _mean(gross_values),
        "best_5_net_proxy": best_tail,
        "worst_5_net_proxy": worst_tail,
    }


def _empty_economics() -> dict[str, Any]:
    return {
        "economics_status": NO_NET_PROXY,
        "mean_net_proxy": "",
        "cumulative_net_proxy": "",
        "max_drawdown_proxy": "",
        "expectancy_proxy": "",
        "win_rate_proxy": "",
        "mean_gross_proxy": "",
        "best_5_net_proxy": "",
        "worst_5_net_proxy": "",
    }


def _classify_policy(
    selected_count: int,
    lift: float,
    economics_available: bool,
    economics: Mapping[str, Any],
) -> str:
    classification = "WEAK_OR_UNSTABLE_POLICY"
    if selected_count == 0:
        classification = "INSUFFICIENT_EVIDENCE"
    elif economics_available and economics.get("mean_net_proxy") not in ("", None):
        mean_net = float(economics["mean_net_proxy"])
        win_rate = float(economics["win_rate_proxy"])
        if mean_net > 0.0 and lift >= 1.1 and win_rate >= 0.5:
            classification = "ECONOMICALLY_PROMISING_RESEARCH_CANDIDATE"
        elif lift >= 1.1 and mean_net > 0.0:
            classification = "SIGNAL_CONFIRMED_ECONOMICS_MIXED"
        elif lift >= 1.1 and mean_net <= 0.0:
            classification = "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE"
        elif lift >= 1.1:
            classification = "WATCHLIST_CONDITIONAL_POLICY"
    elif lift >= 1.5:
        classification = "SIGNAL_CONFIRMED_ECONOMICS_UNKNOWN"
    elif lift >= 1.1:
        classification = "WATCHLIST_CONDITIONAL_POLICY"
    return classification


def _candidate_decisions(
    policy_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for model_name, model_rows in sorted(_group_by(policy_candidates, "model_name").items()):
        best = _best_row(model_rows)
        output.append(
            {
                "model_name": model_name,
                "best_policy": best.get("policy_name", ""),
                "candidate_decision": best.get(
                    "policy_classification",
                    "INSUFFICIENT_EVIDENCE",
                ),
                "selected_rows": best.get("selected_rows", 0),
                "coverage": best.get("coverage", 0.0),
                "precision": best.get("precision", 0.0),
                "lift_vs_base": best.get("lift_vs_base", 0.0),
                "economics_status": best.get("economics_status", NO_NET_PROXY),
                "mean_net_proxy": best.get("mean_net_proxy", ""),
                "cumulative_net_proxy": best.get("cumulative_net_proxy", ""),
                "runtime_status": "NO_RUNTIME_EFFECT",
                "promotion_status": "NOT_PROMOTABLE",
                "profitability_status": "NO_PROFIT_CLAIM",
            }
        )
    return output


def _model_policy_metrics(
    policy_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for model_name, model_rows in sorted(_group_by(policy_candidates, "model_name").items()):
        best = _best_row(model_rows)
        output.append(
            {
                "model_name": model_name,
                "policy_count": len(model_rows),
                "best_policy": best.get("policy_name", ""),
                "best_policy_classification": best.get("policy_classification", ""),
                "best_lift_vs_base": best.get("lift_vs_base", 0.0),
                "best_precision": best.get("precision", 0.0),
                "economics_status": best.get("economics_status", NO_NET_PROXY),
                "mean_net_proxy": best.get("mean_net_proxy", ""),
                "cumulative_net_proxy": best.get("cumulative_net_proxy", ""),
            }
        )
    return output


def _recommendation(
    candidate_decisions: Sequence[Mapping[str, Any]],
    economics_available: bool,
) -> dict[str, Any]:
    if not economics_available:
        recommendation = "ADD_SAFE_NET_PROXY_OR_ECONOMIC_OUTCOME_ARTIFACTS"
        next_action = NEXT_ACTION_NO_ECONOMICS
    elif any(
        row["candidate_decision"] == "ECONOMICALLY_PROMISING_RESEARCH_CANDIDATE"
        for row in candidate_decisions
    ):
        recommendation = "PLAN_STRICT_OUT_OF_SAMPLE_POLICY_CONFIRMATION"
        next_action = "PLAN_STRICT_OUT_OF_SAMPLE_POLICY_CONFIRMATION"
    elif any(
        row["candidate_decision"]
        in ("SIGNAL_CONFIRMED_ECONOMICS_MIXED", "WATCHLIST_CONDITIONAL_POLICY")
        for row in candidate_decisions
    ):
        recommendation = "REFINE_OR_PAUSE_COST_AWARE_SPECIALIST_POLICY"
        next_action = "REFINE_OR_PAUSE_COST_AWARE_SPECIALIST_POLICY"
    else:
        recommendation = "REJECT_OR_WATCHLIST_SPECIALIST_POLICY"
        next_action = "REJECT_OR_WATCHLIST_SPECIALIST_POLICY"
    return {
        "recommendation": recommendation,
        "next_required_action": next_action,
        "best_policy_candidate": _best_policy(candidate_decisions),
        "economics_available": economics_available,
        "evidence_blockers": _evidence_blockers(economics_available),
        "overall_status": list(OVERALL_STATUS),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _evidence_blockers(economics_available: bool) -> list[str]:
    if economics_available:
        return ["NOT_BACKTEST", "NOT_RUNTIME_READY", "NOT_PROMOTABLE", "NO_PROFIT_CLAIM"]
    return [
        NO_NET_PROXY,
        ECONOMIC_REQUIRED,
        "NOT_BACKTEST",
        "NOT_RUNTIME_READY",
        "NOT_PROMOTABLE",
        "NO_PROFIT_CLAIM",
    ]


def _next_actions(recommendation: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action": str(recommendation["next_required_action"]),
            "rationale": (
                "Policy economics require safe outcome artifacts before runtime, "
                "promotion, or profitability claims."
            ),
        },
        {
            "priority": "2",
            "action": "KEEP_SPECIALIST_POLICIES_RESEARCH_ONLY",
            "rationale": "No runtime, registry, promotion, backtest, or trading change.",
        },
    ]


def _best_policy(candidate_decisions: Sequence[Mapping[str, Any]]) -> str:
    if not candidate_decisions:
        return ""
    best = _best_row(candidate_decisions)
    return f"{best.get('model_name', '')}:{best.get('best_policy', '')}"


def _best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda row: (
            _classification_rank(
                str(row.get("policy_classification", row.get("candidate_decision", "")))
            ),
            float(row.get("mean_net_proxy") or 0.0),
            float(row.get("lift_vs_base") or 0.0),
            float(row.get("precision") or 0.0),
            str(row.get("policy_name", row.get("best_policy", ""))),
        ),
        reverse=True,
    )[0]


def _classification_rank(classification: str) -> int:
    ranks = {
        "ECONOMICALLY_PROMISING_RESEARCH_CANDIDATE": 5,
        "SIGNAL_CONFIRMED_ECONOMICS_MIXED": 4,
        "SIGNAL_CONFIRMED_ECONOMICS_UNKNOWN": 3,
        "WATCHLIST_CONDITIONAL_POLICY": 3,
        "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE": 2,
        "WEAK_OR_UNSTABLE_POLICY": 2,
        "INSUFFICIENT_EVIDENCE": 1,
    }
    return ranks.get(classification, 0)


def _load_edge_slices(edge_dir: Path) -> dict[str, tuple[dict[str, str], ...]]:
    slices: dict[str, list[dict[str, str]]] = {}
    if not edge_dir.exists():
        return {}
    for csv_name in ("by_symbol.csv", "by_time.csv"):
        path = edge_dir / csv_name
        if not path.exists():
            continue
        for row in _read_csv(path, "Missing edge evaluator file"):
            if row.get("classification") != "ENABLE_CONDITIONAL_RESEARCH_SLICE":
                continue
            slices.setdefault(str(row.get("model_name", "")), []).append(row)
    return {model_name: tuple(rows) for model_name, rows in slices.items()}


def _economic_outcome_indexes(
    outcome_dir: Path,
) -> tuple[
    dict[tuple[str, str, str], Mapping[str, str]],
    dict[tuple[str, str], Mapping[str, str]],
]:
    path = outcome_dir / ECONOMIC_OUTCOME_FILE
    if not path.exists():
        return {}, {}
    fold_index = {}
    fallback_index = {}
    for row in _read_csv(path, "Missing economic outcome file"):
        symbol = str(row.get("symbol", ""))
        interval_begin = str(row.get("interval_begin", ""))
        fold_index[(str(row.get("fold_index", "")), symbol, interval_begin)] = row
        fallback_index[(symbol, interval_begin)] = row
    return fold_index, fallback_index


def _joined_float(
    outcome: Mapping[str, str] | None,
    label: Mapping[str, str],
    safe_net_column: str | None,
    outcome_column: str,
) -> float | None:
    if outcome is not None and outcome.get(outcome_column) not in ("", None):
        return _to_float(outcome.get(outcome_column))
    if outcome_column == "net_value_proxy" and safe_net_column is not None:
        return _to_float(label.get(safe_net_column))
    return None


def _safe_source(
    economics_available: bool,
    outcome_index: Mapping[tuple[str, str, str], Mapping[str, str]],
    safe_net_column: str | None,
) -> str:
    if not economics_available:
        return ""
    if outcome_index:
        return "economic_outcome_artifacts"
    if safe_net_column is not None:
        return "label_or_evaluation_artifact"
    return ""


def _joined_economics_available(
    joined_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    return any(
        row.get("net_proxy") is not None
        for rows in joined_by_model.values()
        for row in rows
    )


def _topk_selection(
    rows: Sequence[Mapping[str, Any]],
    fraction: float,
) -> list[Mapping[str, Any]]:
    ranked = sorted(
        (row for row in rows if row.get("probability") is not None),
        key=lambda row: (
            -float(row["probability"]),
            str(row["interval_begin"]),
            str(row["symbol"]),
        ),
    )
    count = max(1, int(len(ranked) * fraction)) if ranked else 0
    return ranked[:count]


def _max_drawdown(values: Sequence[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
    return max_drawdown


def _tail_means(values: Sequence[float], count: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    sorted_values = sorted(values)
    tail_count = min(count, len(sorted_values))
    return _mean(sorted_values[:tail_count]), _mean(sorted_values[-tail_count:])


def _group_by(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    return groups


def _positive_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(int(row["target"]) for row in rows) / len(rows) if rows else 0.0


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _first_float(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        if name in row and row.get(name) not in ("", None):
            return _to_float(row.get(name))
    return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _quarter(timestamp: str) -> str:
    if len(timestamp) < 7:
        return ""
    try:
        month = int(timestamp[5:7])
    except ValueError:
        return ""
    return f"{timestamp[:4]}Q{((month - 1) // 3) + 1}"


def _output_files(output_dir: Path, include_thresholds: bool) -> dict[str, str]:
    files = {
        "manifest_json": str(output_dir / "manifest.json"),
        "cost_aware_policy_report_json": str(output_dir / "cost_aware_policy_report.json"),
        "cost_aware_policy_report_md": str(output_dir / "cost_aware_policy_report.md"),
        "policy_candidates_csv": str(output_dir / "policy_candidates.csv"),
        "model_policy_metrics_csv": str(output_dir / "model_policy_metrics.csv"),
        "topk_policy_metrics_csv": str(output_dir / "topk_policy_metrics.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "economics_availability_json": str(output_dir / "economics_availability.json"),
        "candidate_decisions_csv": str(output_dir / "candidate_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }
    if include_thresholds:
        files["threshold_policy_metrics_csv"] = str(
            output_dir / "threshold_policy_metrics.csv"
        )
    return files


def _markdown(
    report: Mapping[str, Any],
    candidate_decisions: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# M20 Cost-Aware Specialist Policy Evaluator",
        "",
        f"- Best policy candidate: `{report['best_policy_candidate']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Economics available: `{report['economics_available']}`",
        "- Evidence blockers: "
        + ", ".join(f"`{blocker}`" for blocker in report["evidence_blockers"]),
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_RUNTIME_READY`, "
        "`NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`, `NOT_BACKTEST`",
        "",
        "## Candidate Decisions",
    ]
    for row in candidate_decisions:
        lines.append(
            f"- `{row['model_name']}` `{row['best_policy']}` -> "
            f"`{row['candidate_decision']}`"
        )
    lines.extend(
        [
            "",
            "Existing artifacts only. No training, scoring, runtime, registry, "
            "promotion, backtest, trading, or profit claim was added.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["analyze_m20_cost_aware_specialist_policy"]
