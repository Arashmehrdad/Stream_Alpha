"""Research-only conditional usefulness for M20 specialist OOF predictions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "specialist_conditional_usefulness"
SPECIALIST_MODELS = ("neuralforecast_nhits", "neuralforecast_patchtst")
TOP_K_FRACTIONS = (0.01, 0.02, 0.05, 0.10)

_HONESTY_FLAGS_OOF = (
    "RESEARCH_ONLY_SPECIALIST_CONDITIONAL_USEFULNESS",
    "EXISTING_OOF_ONLY",
    "NOT_RUNTIME_COMPARABLE",
    "NOT_PROMOTABLE",
    "NO_SCORE_ONLY_RERUN_EXECUTED",
    "NO_MODEL_RETRAIN",
    "NO_REGISTRY_WRITE",
    "NO_RUNTIME_EFFECT",
    "NO_PROMOTION_EFFECT",
    "NOT_BACKTEST",
    "NO_PROFIT_CLAIM",
    "SPECIALIST_CONDITIONAL_ANALYSIS_ONLY",
)

_HONESTY_FLAGS_SCORE_ONLY_CONFIRMATION = (
    "RESEARCH_ONLY_SPECIALIST_CONDITIONAL_USEFULNESS",
    "SCORE_ONLY_CONFIRMATION_PREDICTIONS",
    "NOT_RUNTIME_COMPARABLE",
    "NOT_PROMOTABLE",
    "MANUAL_CONFIRMATION_RUN_ONLY",
    "NO_MODEL_RETRAIN",
    "NO_REGISTRY_WRITE",
    "NO_RUNTIME_EFFECT",
    "NO_PROMOTION_EFFECT",
    "NOT_BACKTEST",
    "NO_PROFIT_CLAIM",
    "SPECIALIST_CONDITIONAL_ANALYSIS_ONLY",
)

def analyze_m20_specialist_conditional_usefulness(
    *,
    base_run_dir: Path,
    previous_run_dir: Path,
    prediction_source: str = "oof",
) -> dict[str, Any]:
    """Analyze sanitized NHITS/PatchTST predictions against fee labels.

    Args:
        base_run_dir: Directory containing specialist_predictions export.
        previous_run_dir: Directory containing fee_exceedance labels.
        prediction_source: Either 'oof' or 'score_only_confirmation'.
    """
    # pylint: disable=too-many-locals
    if prediction_source not in ("oof", "score_only_confirmation"):
        raise ValueError(
            f"prediction_source must be 'oof' or 'score_only_confirmation', "
            f"got {prediction_source}"
        )
    base_dir = Path(base_run_dir).resolve()
    previous_dir = Path(previous_run_dir).resolve()
    specialist_dir = base_dir / "research_labels" / "vol_scaled" / "specialist_predictions"
    output_dir = base_dir / "research_labels" / "vol_scaled" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _read_csv(
        previous_dir
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_labels_vol_scaled.csv"
    )
    label_index, fallback_label_index = _label_indexes(labels)
    source_counts = {}
    joined_by_model = {}
    prediction_suffix = (
        "score_only_confirmation" if prediction_source == "score_only_confirmation" else "oof"
    )
    for model_name in SPECIALIST_MODELS:
        prediction_rows = _read_csv(
            specialist_dir / f"predictions_{model_name}_{prediction_suffix}.csv"
        )
        source_counts[model_name] = len(prediction_rows)
        joined_by_model[model_name] = _joined_model_rows(
            prediction_rows,
            label_index,
            fallback_label_index,
        )
    model_metrics = [
        _model_metric(model_name, rows)
        for model_name, rows in sorted(joined_by_model.items())
    ]
    topk_metrics = [
        row
        for model_name, rows in sorted(joined_by_model.items())
        for row in _topk_metrics(model_name, rows)
    ]
    by_slice = [
        row
        for model_name, rows in sorted(joined_by_model.items())
        for row in _slice_metrics(model_name, rows)
    ]
    honesty_flags = (
        _HONESTY_FLAGS_SCORE_ONLY_CONFIRMATION
        if prediction_source == "score_only_confirmation"
        else _HONESTY_FLAGS_OOF
    )
    comparison = _comparison(model_metrics, topk_metrics, by_slice)
    recommendation = _recommendation(comparison, honesty_flags=honesty_flags)
    output_files = _output_files(output_dir)
    summary_text = (
        "Sanitized NHITS/PatchTST score-only confirmation predictions were "
        "conditionally analyzed against existing volatility-scaled fee-exceedance labels."
        if prediction_source == "score_only_confirmation"
        else (
            "Sanitized NHITS/PatchTST OOF predictions were conditionally analyzed "
            "against existing volatility-scaled fee-exceedance labels."
        )
    )
    manifest = {
        "base_run_dir": str(base_dir),
        "previous_run_dir": str(previous_dir),
        "specialist_prediction_dir": str(specialist_dir),
        "prediction_source": prediction_source,
        "output_dir": str(output_dir),
        "models": list(SPECIALIST_MODELS),
        "joined_rows": sum(len(rows) for rows in joined_by_model.values()),
        "source_rows_by_model": source_counts,
        "joined_rows_by_model": {
            model_name: len(rows) for model_name, rows in sorted(joined_by_model.items())
        },
        "skipped_unlabeled_rows": sum(source_counts.values())
        - sum(len(rows) for rows in joined_by_model.values()),
        "honesty_flags": list(honesty_flags),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "output_files": output_files,
    }
    report = {
        "summary": summary_text,
        "recommendation": recommendation["recommendation"],
        "best_candidate": recommendation["best_candidate"],
        "prediction_source": prediction_source,
        "joined_rows": manifest["joined_rows"],
        "honesty_flags": list(honesty_flags),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    Path(output_files["report_md"]).write_text(
        _markdown(report, model_metrics, comparison, prediction_source=prediction_source),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["model_metrics_csv"]), model_metrics)
    write_csv_artifact(Path(output_files["by_slice_csv"]), by_slice)
    write_csv_artifact(Path(output_files["topk_metrics_csv"]), topk_metrics)
    write_csv_artifact(Path(output_files["comparison_csv"]), comparison)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "model_metrics": model_metrics,
            "topk_metrics": topk_metrics,
            "by_slice": by_slice,
            "comparison": comparison,
            "recommendation_payload": recommendation,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
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
        model_index[
            (
                row.get("model_name", ""),
                row.get("symbol", ""),
                row.get("interval_begin", ""),
            )
        ] = row
        fallback_index[(row.get("symbol", ""), row.get("interval_begin", ""))] = row
    return model_index, fallback_index


def _joined_model_rows(
    predictions: Sequence[Mapping[str, str]],
    label_index: Mapping[tuple[str, str, str], Mapping[str, str]],
    fallback_label_index: Mapping[tuple[str, str], Mapping[str, str]],
) -> list[dict[str, Any]]:
    joined = []
    for row in predictions:
        key = (row.get("model_name", ""), row.get("symbol", ""), row.get("interval_begin", ""))
        fallback_key = (row.get("symbol", ""), row.get("interval_begin", ""))
        label = label_index.get(key) or fallback_label_index.get(fallback_key)
        if label is None:
            continue
        joined.append(
            {
                "model_name": row.get("model_name", ""),
                "symbol": row.get("symbol", ""),
                "interval_begin": row.get("interval_begin", ""),
                "fold_index": row.get("fold_index", ""),
                "regime_label": row.get("regime_label", ""),
                "probability": _to_float(row.get("prob_up", "0")),
                "prediction": _to_int(row.get("y_pred", "0")),
                "target": _to_int(label.get("label", "0")),
                "month": row.get("interval_begin", "")[:7],
                "quarter": _quarter(row.get("interval_begin", "")),
            }
        )
    return joined


def _model_metric(model_name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "row_count": len(rows),
        **_classification_metrics(rows),
        "roc_auc": _roc_auc(rows),
        "pr_auc": _average_precision(rows),
    }


def _topk_metrics(model_name: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    base_rate = _positive_rate(rows)
    positives = sum(int(row["target"]) for row in rows)
    ranked = sorted(rows, key=lambda row: (-float(row["probability"]), row["interval_begin"]))
    for fraction in TOP_K_FRACTIONS:
        count = max(1, int(len(ranked) * fraction)) if ranked else 0
        selected = ranked[:count]
        selected_positive = sum(int(row["target"]) for row in selected)
        precision = selected_positive / count if count else 0.0
        output.append(
            {
                "model_name": model_name,
                "top_k_fraction": fraction,
                "selected_rows": count,
                "coverage": count / len(rows) if rows else 0.0,
                "precision": precision,
                "base_positive_rate": base_rate,
                "lift": precision / base_rate if base_rate > 0 else 0.0,
                "recall": selected_positive / positives if positives else 0.0,
                "false_positives": count - selected_positive,
                "avg_probability": _mean(float(row["probability"]) for row in selected),
            }
        )
    return output


def _slice_metrics(model_name: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for family, key in (
        ("symbol", "symbol"),
        ("month", "month"),
        ("quarter", "quarter"),
        ("regime", "regime_label"),
    ):
        for value, slice_rows in _group_by(rows, key).items():
            metric = _classification_metrics(slice_rows)
            output.append(
                {
                    "model_name": model_name,
                    "slice_family": family,
                    "slice_value": value,
                    "row_count": len(slice_rows),
                    "positive_rate": _positive_rate(slice_rows),
                    "top5_lift": _topk_lift(slice_rows, 0.05),
                    "classification": _classify_slice(slice_rows),
                    **metric,
                }
            )
    return sorted(
        output,
        key=lambda row: (row["model_name"], row["slice_family"], str(row["slice_value"])),
    )


def _classification_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if int(row["prediction"]) == 1 and int(row["target"]) == 1)
    tn = sum(1 for row in rows if int(row["prediction"]) == 0 and int(row["target"]) == 0)
    fp = sum(1 for row in rows if int(row["prediction"]) == 1 and int(row["target"]) == 0)
    fn = sum(1 for row in rows if int(row["prediction"]) == 0 and int(row["target"]) == 1)
    total = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "positive_count": tp + fn,
        "positive_rate": (tp + fn) / total if total else 0.0,
        "accuracy_diagnostic": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positives": fp,
        "true_positives": tp,
    }


def _comparison(
    model_metrics: Sequence[Mapping[str, Any]],
    topk_metrics: Sequence[Mapping[str, Any]],
    by_slice: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for metric in model_metrics:
        model_name = str(metric["model_name"])
        top5 = next(
            row for row in topk_metrics
            if row["model_name"] == model_name and float(row["top_k_fraction"]) == 0.05
        )
        enable_slices = [
            row for row in by_slice
            if row["model_name"] == model_name
            and row["classification"] == "KEEP_CONDITIONAL_RESEARCH_CANDIDATE"
        ]
        output.append(
            {
                "model_name": model_name,
                "rows": metric["row_count"],
                "base_positive_rate": metric["positive_rate"],
                "pr_auc": metric["pr_auc"],
                "roc_auc": metric["roc_auc"],
                "top5_precision": top5["precision"],
                "top5_lift": top5["lift"],
                "enable_slice_count": len(enable_slices),
                "best_slice": _best_slice(enable_slices),
                "recommendation_basis": _basis(metric, top5, enable_slices),
            }
        )
    return output


def _recommendation(
    comparison: Sequence[Mapping[str, Any]],
    *,
    honesty_flags: Sequence[str],
) -> dict[str, Any]:
    ranked = sorted(
        comparison,
        key=lambda row: (
            float(row["top5_lift"]),
            float(row["pr_auc"] or 0.0) - float(row["base_positive_rate"]),
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else {}
    recommendation = (
        "RUN_SPECIALIST_CONFIRMATION_EXPORT"
        if best and float(best["top5_lift"]) >= 1.2 and int(best["enable_slice_count"]) > 0
        else "KEEP_SPECIALISTS_CONDITIONALLY_UNKNOWN_OR_WEAK"
    )
    return {
        "recommendation": recommendation,
        "best_candidate": best.get("model_name", ""),
        "rationale": best.get("recommendation_basis", "no specialist comparison available"),
        "runtime_ready": False,
        "promotable": False,
        "honesty_flags": list(honesty_flags),
    }


def _classify_slice(rows: Sequence[Mapping[str, Any]]) -> str:
    if len(rows) < 1000:
        return "INSUFFICIENT_SAMPLE"
    if sum(int(row["target"]) for row in rows) < 50:
        return "INSUFFICIENT_POSITIVES"
    base_rate = _positive_rate(rows)
    top5_lift = _topk_lift(rows, 0.05)
    if top5_lift >= 1.2 and _average_precision(rows) > base_rate + 0.015:
        return "KEEP_CONDITIONAL_RESEARCH_CANDIDATE"
    if top5_lift <= 1.0:
        return "WEAK_OR_UNSTABLE"
    return "WATCHLIST_ONLY"


def _basis(
    metric: Mapping[str, Any],
    top5: Mapping[str, Any],
    enable_slices: Sequence[Mapping[str, Any]],
) -> str:
    pr_auc = metric["pr_auc"] if metric["pr_auc"] is not None else 0.0
    return (
        f"top5_lift={float(top5['lift']):.6f}; "
        f"pr_auc={float(pr_auc):.6f}; "
        f"enable_slices={len(enable_slices)}"
    )


def _best_slice(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    best = max(rows, key=lambda row: float(row["top5_lift"]))
    return f"{best['slice_family']}={best['slice_value']}"


def _topk_lift(rows: Sequence[Mapping[str, Any]], fraction: float) -> float:
    base_rate = _positive_rate(rows)
    if not rows or base_rate <= 0:
        return 0.0
    count = max(1, int(len(rows) * fraction))
    ranked = sorted(rows, key=lambda row: (-float(row["probability"]), row["interval_begin"]))
    selected = ranked[:count]
    precision = sum(int(row["target"]) for row in selected) / count
    return precision / base_rate


def _roc_auc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    pairs = sorted((float(row["probability"]), int(row["target"])) for row in rows)
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
    positives = sum(int(row["target"]) for row in rows)
    if positives == 0:
        return None
    ranked = sorted(rows, key=lambda row: (-float(row["probability"]), row["interval_begin"]))
    hit_count = 0
    precision_sum = 0.0
    for index, row in enumerate(ranked, start=1):
        if int(row["target"]) == 1:
            hit_count += 1
            precision_sum += hit_count / index
    return precision_sum / positives


def _positive_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(int(row["target"]) for row in rows) / len(rows) if rows else 0.0


def _group_by(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    return groups


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _quarter(timestamp: str) -> str:
    if len(timestamp) < 7:
        return ""
    month = int(timestamp[5:7])
    quarter = ((month - 1) // 3) + 1
    return f"{timestamp[:4]}Q{quarter}"


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "model_metrics_csv": str(output_dir / "model_metrics.csv"),
        "by_slice_csv": str(output_dir / "by_slice.csv"),
        "topk_metrics_csv": str(output_dir / "topk_metrics.csv"),
        "comparison_csv": str(output_dir / "comparison.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    model_metrics: Sequence[Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    *,
    prediction_source: str,
) -> str:
    lines = [
        "# M20 Specialist Conditional Usefulness",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Best candidate: `{report['best_candidate']}`",
        f"- Joined rows: {report['joined_rows']}",
        (
            "- Status: research-only; score-only confirmation predictions only."
            if prediction_source == "score_only_confirmation"
            else "- Status: research-only; existing OOF only."
        ),
        "",
        "## Model Metrics",
    ]
    for row in model_metrics:
        lines.append(
            f"- `{row['model_name']}`: rows={row['row_count']}, "
            f"PR-AUC={row['pr_auc']}, ROC-AUC={row['roc_auc']}, "
            f"balanced_accuracy={row['balanced_accuracy']:.6f}"
        )
    lines.extend(["", "## Comparison"])
    for row in comparison:
        lines.append(
            f"- `{row['model_name']}`: top5_lift={float(row['top5_lift']):.6f}, "
            f"best_slice=`{row['best_slice']}`"
        )
    lines.extend(
        [
            "",
            "No score-only rerun, model retrain, runtime path, registry write, "
            "promotion, backtest, or profit claim was added.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["analyze_m20_specialist_conditional_usefulness"]
