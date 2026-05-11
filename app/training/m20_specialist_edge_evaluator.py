"""Generic research-only M20 specialist edge evaluator from saved artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_LABEL_FILE = "research_labels/vol_scaled/fee_exceedance_labels_vol_scaled.csv"
DEFAULT_OUTPUT_NAME = "specialist_edge_evaluator"
TOP_K_FRACTIONS = (0.01, 0.02, 0.05, 0.10)
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
NO_NET_PROXY = "NET_PROXY_NOT_AVAILABLE"
NEXT_REQUIRED_ACTION = "DESIGN_COST_AWARE_SPECIALIST_POLICY_EVALUATOR"
OVERALL_STATUS = (
    "RESEARCH_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
HONESTY_FLAGS = (
    "RESEARCH_ONLY_SPECIALIST_EDGE_EVALUATOR",
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
ECONOMIC_BLOCKERS = (
    NO_NET_PROXY,
    "ECONOMIC_POLICY_EVALUATION_REQUIRED",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)


def analyze_m20_specialist_edge(
    *,
    prediction_run_dir: Path,
    label_source_run_dir: Path,
    prediction_source: str,
    models: Sequence[str] | None = None,
    label_file: str = DEFAULT_LABEL_FILE,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Evaluate conditional specialist edge for all requested/discovered models."""
    # pylint: disable=too-many-arguments,too-many-locals
    prediction_dir = Path(prediction_run_dir).resolve()
    label_run_dir = Path(label_source_run_dir).resolve()
    specialist_dir = prediction_dir / "research_labels" / "vol_scaled" / "specialist_predictions"
    output_dir = prediction_dir / "research_labels" / "vol_scaled" / output_name
    label_path = label_run_dir / label_file

    discovered_models = _discover_models(specialist_dir, prediction_source)
    selected_models = _selected_models(models, discovered_models)
    prediction_files = {
        model_name: specialist_dir / f"predictions_{model_name}_{prediction_source}.csv"
        for model_name in selected_models
    }
    missing_predictions = [
        str(path) for path in prediction_files.values() if not path.exists()
    ]
    if missing_predictions:
        raise ValueError(
            "Missing specialist prediction file(s): " + "; ".join(missing_predictions)
        )

    labels = _read_csv(label_path, "Missing label file")
    label_index, fallback_label_index = _label_indexes(labels)
    joined_by_model = {}
    source_rows_by_model = {}
    net_proxy_status_by_model = {}
    for model_name, prediction_file in prediction_files.items():
        predictions = _read_csv(prediction_file, "Missing specialist prediction file")
        source_rows_by_model[model_name] = len(predictions)
        joined_by_model[model_name] = _join_rows(
            model_name=model_name,
            predictions=predictions,
            label_index=label_index,
            fallback_label_index=fallback_label_index,
        )
        net_proxy_status_by_model[model_name] = _net_proxy_status(joined_by_model[model_name])

    output_dir.mkdir(parents=True, exist_ok=True)
    model_metrics = [
        _model_metrics(model_name, rows, net_proxy_status_by_model[model_name])
        for model_name, rows in sorted(joined_by_model.items())
    ]
    topk_policy_metrics = [
        metric
        for model_name, rows in sorted(joined_by_model.items())
        for metric in _topk_policy_metrics(model_name, rows)
    ]
    threshold_policy_metrics = [
        metric
        for model_name, rows in sorted(joined_by_model.items())
        for metric in _threshold_policy_metrics(model_name, rows)
    ]
    by_symbol = [
        metric
        for model_name, rows in sorted(joined_by_model.items())
        for metric in _slice_metrics(model_name, rows, "symbol", "symbol")
    ]
    by_time = [
        metric
        for model_name, rows in sorted(joined_by_model.items())
        for family, key in (("month", "month"), ("quarter", "quarter"))
        for metric in _slice_metrics(model_name, rows, family, key)
    ]
    candidate_decisions = _candidate_decisions(
        model_metrics=model_metrics,
        topk_metrics=topk_policy_metrics,
        by_symbol=by_symbol,
        by_time=by_time,
    )
    best_candidate = _best_candidate(candidate_decisions)
    evidence_blockers = _evidence_blockers(model_metrics)
    next_actions = _next_actions(candidate_decisions, evidence_blockers)
    recommendation = _recommendation(
        best_candidate,
        candidate_decisions,
        evidence_blockers,
    )
    output_files = _output_files(output_dir, bool(threshold_policy_metrics))

    manifest = {
        "prediction_run_dir": str(prediction_dir),
        "label_source_run_dir": str(label_run_dir),
        "specialist_prediction_dir": str(specialist_dir),
        "label_file": str(label_path),
        "prediction_source": prediction_source,
        "requested_models": list(models or []),
        "models": list(selected_models),
        "discovered_models": list(discovered_models),
        "joined_rows": sum(len(rows) for rows in joined_by_model.values()),
        "source_rows_by_model": source_rows_by_model,
        "joined_rows_by_model": {
            model_name: len(rows) for model_name, rows in sorted(joined_by_model.items())
        },
        "skipped_unlabeled_rows": sum(source_rows_by_model.values())
        - sum(len(rows) for rows in joined_by_model.values()),
        "top_k_fractions": list(TOP_K_FRACTIONS),
        "thresholds": list(THRESHOLDS) if threshold_policy_metrics else [],
        "overall_status": list(OVERALL_STATUS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "evidence_blockers": evidence_blockers,
        "next_required_action": NEXT_REQUIRED_ACTION,
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    report = {
        "summary": (
            "Generic M20 specialist edge evaluation from existing prediction "
            "and label artifacts only."
        ),
        "best_candidate": best_candidate,
        "prediction_source": prediction_source,
        "joined_rows": manifest["joined_rows"],
        "overall_status": list(OVERALL_STATUS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "evidence_blockers": evidence_blockers,
        "next_required_action": NEXT_REQUIRED_ACTION,
        "honesty_flags": list(HONESTY_FLAGS),
        "candidate_decisions": candidate_decisions,
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["specialist_edge_report_json"]), report)
    Path(output_files["specialist_edge_report_md"]).write_text(
        _markdown(report, model_metrics, candidate_decisions),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["model_edge_metrics_csv"]), model_metrics)
    write_csv_artifact(Path(output_files["topk_policy_metrics_csv"]), topk_policy_metrics)
    if threshold_policy_metrics:
        write_csv_artifact(
            Path(output_files["threshold_policy_metrics_csv"]),
            threshold_policy_metrics,
        )
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_csv_artifact(Path(output_files["candidate_decisions_csv"]), candidate_decisions)
    write_csv_artifact(Path(output_files["next_actions_csv"]), next_actions)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)

    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "model_edge_metrics": model_metrics,
            "topk_policy_metrics": topk_policy_metrics,
            "threshold_policy_metrics": threshold_policy_metrics,
            "by_symbol": by_symbol,
            "by_time": by_time,
            "evidence_blockers": evidence_blockers,
            "next_actions": next_actions,
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


def _read_csv(path: Path, missing_prefix: str) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"{missing_prefix}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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
) -> list[dict[str, Any]]:
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
        probability = _first_float(row, ("prob_up", "probability", "score"))
        joined.append(
            {
                "model_name": model_name,
                "row_model_name": row_model,
                "symbol": symbol,
                "interval_begin": interval_begin,
                "probability": probability,
                "has_probability": probability is not None,
                "prediction": _to_int(row.get("y_pred", row.get("prediction", "0"))),
                "target": _to_int(label.get("label", label.get("y_true", "0"))),
                "month": interval_begin[:7],
                "quarter": _quarter(interval_begin),
                "net_proxy": _first_float(
                    label,
                    (
                        "long_only_net_value_proxy",
                        "net_value_proxy",
                        "safe_net_value_proxy",
                    ),
                ),
            }
        )
    return joined


def _model_metrics(
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
    net_proxy_status: str,
) -> dict[str, Any]:
    base_metrics = _classification_metrics(rows)
    top5 = _policy_metrics(model_name, rows, _topk_selection(rows, 0.05), "top_k", 0.05)
    return {
        "model_name": model_name,
        "row_count": len(rows),
        **base_metrics,
        "roc_auc": _roc_auc(rows),
        "pr_auc": _average_precision(rows),
        "top5_precision": top5["precision"],
        "top5_lift": top5["lift_vs_base"],
        "best_slice": "",
        "enable_slice_count": 0,
        "net_proxy_status": net_proxy_status,
        **_economics_fields(rows),
    }


def _topk_policy_metrics(
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _policy_metrics(model_name, rows, _topk_selection(rows, fraction), "top_k", fraction)
        for fraction in TOP_K_FRACTIONS
    ]


def _threshold_policy_metrics(
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not any(bool(row.get("has_probability")) for row in rows):
        return []
    return [
        _policy_metrics(
            model_name,
            rows,
            [
                row for row in rows
                if row.get("probability") is not None
                and float(row["probability"]) >= threshold
            ],
            "threshold",
            threshold,
        )
        for threshold in THRESHOLDS
    ]


def _slice_metrics(
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
    slice_family: str,
    key: str,
) -> list[dict[str, Any]]:
    output = []
    for slice_value, slice_rows in sorted(_group_by(rows, key).items()):
        top5 = _policy_metrics(
            model_name,
            slice_rows,
            _topk_selection(slice_rows, 0.05),
            "top_k",
            0.05,
        )
        output.append(
            {
                "model_name": model_name,
                "slice_family": slice_family,
                "slice_value": slice_value,
                "row_count": len(slice_rows),
                "positive_rate": _positive_rate(slice_rows),
                "top5_precision": top5["precision"],
                "top5_lift": top5["lift_vs_base"],
                "classification": _classify_slice(slice_rows, top5),
                **_economics_fields(slice_rows),
            }
        )
    return output


def _candidate_decisions(
    *,
    model_metrics: Sequence[Mapping[str, Any]],
    topk_metrics: Sequence[Mapping[str, Any]],
    by_symbol: Sequence[Mapping[str, Any]],
    by_time: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    slices = list(by_symbol) + list(by_time)
    for metric in model_metrics:
        model_name = str(metric["model_name"])
        model_topk = [
            row for row in topk_metrics if str(row["model_name"]) == model_name
        ]
        top5 = _find_fraction(model_topk, 0.05)
        top10 = _find_fraction(model_topk, 0.10)
        enable_slices = [
            row for row in slices
            if str(row["model_name"]) == model_name
            and row["classification"] == "ENABLE_CONDITIONAL_RESEARCH_SLICE"
        ]
        decision = _classify_model(metric, top5, top10, enable_slices)
        best_slice = _best_slice(enable_slices)
        output.append(
            {
                "model_name": model_name,
                "candidate_decision": decision,
                "row_count": metric["row_count"],
                "base_positive_rate": metric["positive_rate"],
                "top5_precision": top5["precision"],
                "top5_lift": top5["lift_vs_base"],
                "top10_precision": top10["precision"],
                "top10_lift": top10["lift_vs_base"],
                "pr_auc": metric["pr_auc"],
                "roc_auc": metric["roc_auc"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "enable_slice_count": len(enable_slices),
                "best_slice": best_slice,
                "runtime_status": "NO_RUNTIME_EFFECT",
                "promotion_status": "NOT_PROMOTABLE",
                "profitability_status": "NO_PROFIT_CLAIM",
                "decision_rationale": _decision_rationale(
                    decision,
                    top5,
                    metric,
                    len(enable_slices),
                ),
            }
        )
    return sorted(output, key=lambda row: str(row["model_name"]))


def _classify_model(
    metric: Mapping[str, Any],
    top5: Mapping[str, Any],
    top10: Mapping[str, Any],
    enable_slices: Sequence[Mapping[str, Any]],
) -> str:
    if int(metric["row_count"]) < 10 or int(metric["positive_count"]) == 0:
        return "INSUFFICIENT_EVIDENCE"
    if float(top5["lift_vs_base"]) >= 1.5 and enable_slices:
        return "CONFIRMED_SELECTIVE_EDGE_RESEARCH_CANDIDATE"
    if float(top5["lift_vs_base"]) >= 1.1 or float(top10["lift_vs_base"]) >= 1.1:
        return "WATCHLIST_CONDITIONAL_SIGNAL"
    return "WEAK_OR_UNSTABLE"


def _classify_slice(
    rows: Sequence[Mapping[str, Any]],
    top5: Mapping[str, Any],
) -> str:
    if len(rows) < 3 or sum(int(row["target"]) for row in rows) == 0:
        return "INSUFFICIENT_EVIDENCE"
    if float(top5["lift_vs_base"]) >= 1.2 and float(top5["selected_rows"]) > 0:
        return "ENABLE_CONDITIONAL_RESEARCH_SLICE"
    if float(top5["lift_vs_base"]) >= 1.0:
        return "WATCHLIST_CONDITIONAL_SIGNAL"
    return "WEAK_OR_UNSTABLE"


def _classification_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    true_positive = sum(
        1 for row in rows if int(row["prediction"]) == 1 and int(row["target"]) == 1
    )
    true_negative = sum(
        1 for row in rows if int(row["prediction"]) == 0 and int(row["target"]) == 0
    )
    false_positive = sum(
        1 for row in rows if int(row["prediction"]) == 1 and int(row["target"]) == 0
    )
    false_negative = sum(
        1 for row in rows if int(row["prediction"]) == 0 and int(row["target"]) == 1
    )
    total = len(rows)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    return {
        "positive_count": true_positive + false_negative,
        "positive_rate": (true_positive + false_negative) / total if total else 0.0,
        "accuracy_diagnostic": (true_positive + true_negative) / total if total else 0.0,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "false_positives": false_positive,
        "true_positives": true_positive,
    }


def _policy_metrics(
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    policy_type: str,
    policy_value: float,
) -> dict[str, Any]:
    selected_count = len(selected_rows)
    positive_count = sum(int(row["target"]) for row in rows)
    selected_positive = sum(int(row["target"]) for row in selected_rows)
    base_rate = _positive_rate(rows)
    precision = selected_positive / selected_count if selected_count else 0.0
    return {
        "model_name": model_name,
        "policy_type": policy_type,
        "policy_value": policy_value,
        "top_k_fraction": policy_value if policy_type == "top_k" else "",
        "probability_threshold": policy_value if policy_type == "threshold" else "",
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
        **_economics_fields(selected_rows),
    }


def _topk_selection(
    rows: Sequence[Mapping[str, Any]],
    fraction: float,
) -> list[Mapping[str, Any]]:
    probability_rows = [row for row in rows if row.get("probability") is not None]
    ranked = sorted(
        probability_rows,
        key=lambda row: (
            -float(row["probability"]),
            str(row["interval_begin"]),
            str(row["symbol"]),
        ),
    )
    count = max(1, int(len(ranked) * fraction)) if ranked else 0
    return ranked[:count]


def _economics_fields(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    net_values = [
        float(row["net_proxy"]) for row in rows if row.get("net_proxy") is not None
    ]
    if not net_values:
        return {
            "net_proxy_status": NO_NET_PROXY,
            "avg_net_proxy": "",
            "cumulative_net_proxy": "",
        }
    return {
        "net_proxy_status": "NET_PROXY_AVAILABLE",
        "avg_net_proxy": _mean(net_values),
        "cumulative_net_proxy": sum(net_values),
    }


def _net_proxy_status(rows: Sequence[Mapping[str, Any]]) -> str:
    return (
        "NET_PROXY_AVAILABLE"
        if any(row.get("net_proxy") is not None for row in rows)
        else NO_NET_PROXY
    )


def _find_fraction(
    rows: Sequence[Mapping[str, Any]],
    fraction: float,
) -> Mapping[str, Any]:
    for row in rows:
        if float(row.get("top_k_fraction") or 0.0) == fraction:
            return row
    return {
        "precision": 0.0,
        "lift_vs_base": 0.0,
    }


def _best_candidate(candidate_decisions: Sequence[Mapping[str, Any]]) -> str:
    ranked = sorted(
        candidate_decisions,
        key=lambda row: (
            row["candidate_decision"] == "CONFIRMED_SELECTIVE_EDGE_RESEARCH_CANDIDATE",
            float(row["top5_lift"]),
            float(row["pr_auc"] or 0.0) - float(row["base_positive_rate"]),
            str(row["model_name"]),
        ),
        reverse=True,
    )
    return str(ranked[0]["model_name"]) if ranked else ""


def _best_slice(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    best = max(rows, key=lambda row: float(row["top5_lift"]))
    return f"{best['slice_family']}={best['slice_value']}"


def _decision_rationale(
    decision: str,
    top5: Mapping[str, Any],
    metric: Mapping[str, Any],
    enable_slice_count: int,
) -> str:
    return (
        f"{decision}: top5_lift={float(top5['lift_vs_base']):.6f}; "
        f"pr_auc={float(metric['pr_auc'] or 0.0):.6f}; "
        f"roc_auc={float(metric['roc_auc'] or 0.0):.6f}; "
        f"enable_slice_count={enable_slice_count}"
    )


def _next_actions(
    candidate_decisions: Sequence[Mapping[str, Any]],
    evidence_blockers: Sequence[str],
) -> list[dict[str, str]]:
    has_candidate = any(
        row["candidate_decision"] == "CONFIRMED_SELECTIVE_EDGE_RESEARCH_CANDIDATE"
        for row in candidate_decisions
    )
    action = (
        NEXT_REQUIRED_ACTION
        if has_candidate
        else "KEEP_SPECIALISTS_RESEARCH_ONLY_AND_COLLECT_STRONGER_EVIDENCE"
    )
    return [
        {
            "priority": "1",
            "action": action,
            "rationale": (
                "Design cost-aware policy evaluation before any runtime or promotion "
                "claim; current evidence blockers: " + "|".join(evidence_blockers)
            ),
        },
        {
            "priority": "2",
            "action": "DO_NOT_PROMOTE_FROM_EDGE_REPORT_ONLY",
            "rationale": (
                "This report has no runtime safety, registry, backtest, or "
                "profitability evidence."
            ),
        },
    ]


def _recommendation(
    best_candidate: str,
    candidate_decisions: Sequence[Mapping[str, Any]],
    evidence_blockers: Sequence[str],
) -> dict[str, Any]:
    decision_by_model = {
        str(row["model_name"]): str(row["candidate_decision"])
        for row in candidate_decisions
    }
    return {
        "recommendation": NEXT_REQUIRED_ACTION,
        "best_candidate": best_candidate,
        "candidate_decisions": decision_by_model,
        "evidence_blockers": list(evidence_blockers),
        "next_required_action": NEXT_REQUIRED_ACTION,
        "overall_status": list(OVERALL_STATUS),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _evidence_blockers(model_metrics: Sequence[Mapping[str, Any]]) -> list[str]:
    if any(row.get("net_proxy_status") == NO_NET_PROXY for row in model_metrics):
        return list(ECONOMIC_BLOCKERS)
    return ["ECONOMIC_POLICY_EVALUATION_REQUIRED", "NOT_BACKTEST", "NOT_RUNTIME_READY"]


def _roc_auc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    pairs = sorted(
        (float(row["probability"]), int(row["target"]))
        for row in rows
        if row.get("probability") is not None
    )
    positive_count = sum(target for _, target in pairs)
    negative_count = len(pairs) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        positives_in_tie = sum(target for _, target in pairs[index:end])
        rank_sum += positives_in_tie * average_rank
        index = end
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _average_precision(rows: Sequence[Mapping[str, Any]]) -> float | None:
    ranked = sorted(
        (row for row in rows if row.get("probability") is not None),
        key=lambda row: (
            -float(row["probability"]),
            str(row["interval_begin"]),
            str(row["symbol"]),
        ),
    )
    positives = sum(int(row["target"]) for row in ranked)
    if positives == 0:
        return None
    hit_count = 0
    precision_sum = 0.0
    for index, row in enumerate(ranked, start=1):
        if int(row["target"]) == 1:
            hit_count += 1
            precision_sum += hit_count / index
    return precision_sum / positives


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
            try:
                return float(row[name])
            except (TypeError, ValueError):
                return None
    return None


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
    quarter = ((month - 1) // 3) + 1
    return f"{timestamp[:4]}Q{quarter}"


def _output_files(output_dir: Path, include_thresholds: bool) -> dict[str, str]:
    files = {
        "manifest_json": str(output_dir / "manifest.json"),
        "specialist_edge_report_json": str(output_dir / "specialist_edge_report.json"),
        "specialist_edge_report_md": str(output_dir / "specialist_edge_report.md"),
        "model_edge_metrics_csv": str(output_dir / "model_edge_metrics.csv"),
        "topk_policy_metrics_csv": str(output_dir / "topk_policy_metrics.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
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
    model_metrics: Sequence[Mapping[str, Any]],
    candidate_decisions: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# M20 Specialist Edge Evaluator",
        "",
        f"- Best candidate: `{report['best_candidate']}`",
        f"- Joined rows: {report['joined_rows']}",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_RUNTIME_READY`, "
        "`NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        f"- Next required action: `{report['next_required_action']}`",
        "- Evidence blockers: "
        + ", ".join(f"`{blocker}`" for blocker in report["evidence_blockers"]),
        "",
        "## Model Metrics",
    ]
    for row in model_metrics:
        lines.append(
            f"- `{row['model_name']}`: rows={row['row_count']}, "
            f"top5_lift={float(row['top5_lift']):.6f}, "
            f"PR-AUC={row['pr_auc']}, ROC-AUC={row['roc_auc']}, "
            f"net_proxy_status=`{row['net_proxy_status']}`"
        )
    lines.extend(["", "## Candidate Decisions"])
    for row in candidate_decisions:
        lines.append(
            f"- `{row['model_name']}` -> `{row['candidate_decision']}`; "
            f"best_slice=`{row['best_slice']}`"
        )
    lines.extend(
        [
            "",
            "Existing artifacts only. No runtime, registry, promotion, backtest, "
            "trading, model-retrain, or profit claim was added.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["analyze_m20_specialist_edge"]
