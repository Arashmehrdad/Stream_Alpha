"""Conditional usefulness diagnostics for M20 fee-exceedance baselines."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


CONDITIONAL_DIR_NAME = "conditional_usefulness"
FULL_TEST_CONDITIONAL_DIR_NAME = "conditional_usefulness_full_test"
DEFAULT_BASELINE_NAME = "logistic_regression_tiny"
MIN_ROWS_PER_SLICE = 1000
MIN_POSITIVES_PER_SLICE = 50
TOP_K_FRACTIONS = (0.01, 0.05, 0.10)
HONESTY_FLAGS = (
    "RESEARCH_ONLY_CONDITIONAL_USEFULNESS",
    "NOT_RUNTIME_COMPARABLE",
    "NOT_PROMOTABLE",
    "NO_REGISTRY_WRITE",
    "NO_RUNTIME_EFFECT",
    "NO_PROMOTION_EFFECT",
    "SINGLE_RECENT_FOLD_ONLY",
    "ENABLE_DISABLE_RESEARCH_ONLY",
    "NO_PROFITABILITY_CLAIM",
    "STRATEGY_ENSEMBLE_INPUT_ONLY",
    "TEST_SPLIT_PRIMARY",
    "TRAIN_SPLIT_NOT_HEADLINE",
    "SLICE_METRICS_FULLY_REPORTED",
    "WEAK_OR_UNSTABLE_SLICES_INCLUDED",
)


def analyze_conditional_usefulness(
    *,
    run_dir: Path,
    prediction_source: str = "auto",
) -> dict[str, Any]:
    """Analyze where the fee-exceedance baseline is useful or harmful."""
    # pylint: disable=too-many-locals
    resolved_run_dir = Path(run_dir).resolve()
    baseline_dir = resolved_run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_baselines"
    baseline_metrics = _load_json(baseline_dir / "fee_baseline_metrics.json")
    manifest = _load_json(baseline_dir / "fee_baseline_manifest.json")
    prediction_path, resolved_prediction_source = _resolve_prediction_path(
        baseline_dir,
        manifest,
        baseline_metrics,
        prediction_source=prediction_source,
    )
    output_name = (
        FULL_TEST_CONDITIONAL_DIR_NAME
        if resolved_prediction_source == "full_test" else CONDITIONAL_DIR_NAME
    )
    output_dir = resolved_run_dir / "research_labels" / "vol_scaled" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = _read_csv_rows(prediction_path)
    if not predictions or "probability" not in predictions[0]:
        raise ValueError("CONDITIONAL_ANALYSIS_BLOCKED_MISSING_PREDICTIONS")
    features = _read_csv_rows(
        resolved_run_dir / "training_frame" / "m20_training_frame_features.csv"
    )
    labels = _read_csv_rows(
        resolved_run_dir
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_labels_vol_scaled.csv"
    )
    joined = _join_rows(predictions, features, labels)
    if not joined:
        raise ValueError("CONDITIONAL_ANALYSIS_BLOCKED_MISSING_PREDICTIONS")
    bucket_definitions = _bucket_definitions(joined)
    slice_rows = _build_all_slice_rows(joined, bucket_definitions)
    search_breadth = _search_breadth(slice_rows)
    flags = list(HONESTY_FLAGS)
    flags.append(
        "FULL_TEST_PREDICTIONS_USED"
        if resolved_prediction_source == "full_test"
        else "SAMPLED_TEST_PREDICTIONS_USED"
    )
    if resolved_prediction_source == "full_test":
        flags[0] = "RESEARCH_ONLY_FULL_TEST_CONDITIONAL_USEFULNESS"
    if search_breadth["enable_candidate_count"]:
        flags.append("CONDITIONAL_SEARCH_REQUIRES_CONFIRMATION")
    recommendation = _recommend(search_breadth, slice_rows)
    output_files = _output_files(output_dir)
    report = {
        "run_dir": str(resolved_run_dir),
        "baseline_dir": str(baseline_dir),
        "conditional_dir": str(output_dir),
        "prediction_path": str(prediction_path),
        "prediction_source": resolved_prediction_source,
        "prediction_rows_analyzed": len(joined),
        "baseline_name": str(joined[0].get("baseline_name", DEFAULT_BASELINE_NAME)),
        "global_baseline_recap": _global_recap(baseline_metrics),
        "search_breadth": search_breadth,
        "bucket_definitions": bucket_definitions,
        "honesty_flags": sorted(dict.fromkeys(flags)),
        "recommendation": recommendation,
        "strategy_ensemble_implications": _strategy_implications(slice_rows),
        "output_files": output_files,
    }
    conditional_manifest = {
        "run_dir": str(resolved_run_dir),
        "conditional_dir": str(output_dir),
        "source_prediction_file": str(prediction_path),
        "prediction_source": resolved_prediction_source,
        "prediction_rows_analyzed": len(joined),
        "honesty_flags": report["honesty_flags"],
        "recommendation": recommendation,
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "output_files": output_files,
    }
    if resolved_prediction_source == "full_test":
        comparison = _write_full_vs_sample_comparison(
            resolved_run_dir,
            output_dir,
            slice_rows,
            report,
        )
        report["full_vs_sample_comparison"] = comparison
        conditional_manifest["full_vs_sample_comparison"] = comparison
    _write_outputs(report, conditional_manifest, slice_rows, bucket_definitions, output_files)
    return make_json_safe({**report, "manifest": conditional_manifest})


def _resolve_prediction_path(
    _baseline_dir: Path,
    manifest: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    *,
    prediction_source: str,
) -> tuple[Path, str]:
    output_files = manifest.get("output_files", {})
    full = output_files.get("predictions_logistic_regression_tiny_test_full_csv")
    if prediction_source in {"auto", "full-test"} and full and Path(str(full)).exists():
        return Path(str(full)), "full_test"
    if prediction_source == "full-test":
        raise ValueError("CONDITIONAL_ANALYSIS_BLOCKED_MISSING_PREDICTIONS")
    preferred = output_files.get("predictions_logistic_regression_tiny_csv")
    if preferred and Path(str(preferred)).exists():
        return Path(str(preferred)), "sampled_test"
    best = max(
        baseline_metrics.get("baselines", []),
        key=lambda row: (
            row.get("average_precision") or 0.0,
            row.get("balanced_accuracy") or 0.0,
        ),
        default={},
    )
    best_name = best.get("baseline_name")
    if best_name:
        path = output_files.get(f"predictions_{best_name}_csv")
        if path and Path(str(path)).exists():
            return Path(str(path)), "sampled_test"
    raise ValueError("CONDITIONAL_ANALYSIS_BLOCKED_MISSING_PREDICTIONS")


def _join_rows(
    predictions: Sequence[Mapping[str, str]],
    features: Sequence[Mapping[str, str]],
    labels: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    feature_by_key = {
        _key(row): row for row in features
    }
    label_by_key = {
        _key(row): row for row in labels
        if str(row.get("scenario_name", "")) == "current_fee"
    }
    joined = []
    for prediction in predictions:
        key = _key(prediction)
        feature = feature_by_key.get(key)
        label = label_by_key.get(key, {})
        if feature is None:
            continue
        joined.append(
            {
                **dict(feature),
                "baseline_name": prediction.get("baseline_name", prediction.get("model_name", "")),
                "label": int(
                    float(
                        prediction.get("label", prediction.get("y_true", label.get("label", 0)))
                        or 0
                    )
                ),
                "prediction": int(
                    float(prediction.get("prediction", prediction.get("y_pred", 0)) or 0)
                ),
                "probability": float(prediction.get("probability", 0.0) or 0.0),
                "split": prediction.get("split", "test"),
            }
        )
    return sorted(joined, key=lambda row: (str(row["interval_begin"]), str(row["symbol"])))


def _bucket_definitions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    definitions: dict[str, Any] = {}
    if _has_column(rows, "realized_vol_12"):
        definitions["volatility"] = _quantile_definition(rows, "realized_vol_12")
    momentum_column = _first_available(rows, ("log_return_1", "momentum_3", "lag_log_return_1"))
    if momentum_column:
        definitions["momentum"] = {
            "column": momentum_column,
            "buckets": {"negative": "<0", "flat": "==0", "positive": ">0"},
        }
    if _has_column(rows, "rsi_14"):
        definitions["rsi"] = {
            "column": "rsi_14",
            "buckets": {"oversold": "<30", "neutral": "30_to_70", "overbought": ">70"},
            "quantile_buckets": _quantile_definition(rows, "rsi_14")["thresholds"],
        }
    if _has_column(rows, "macd_line_12_26"):
        values = [abs(float(row["macd_line_12_26"])) for row in rows]
        threshold = _quantile(values, 0.25)
        definitions["macd"] = {"column": "macd_line_12_26", "near_zero_threshold": threshold}
    if _has_column(rows, "volume"):
        definitions["volume"] = _quantile_definition(rows, "volume")
    if all(_has_column(rows, column) for column in ("high_price", "low_price", "close_price")):
        range_values = [_range_pct(row) for row in rows]
        definitions["range"] = {
            "column": "range_pct",
            "thresholds": _low_high_thresholds(range_values),
        }
    return definitions


def _build_all_slice_rows(
    rows: Sequence[Mapping[str, Any]],
    definitions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    families = [
        ("symbol", lambda row: str(row.get("symbol", ""))),
        ("month", lambda row: str(row.get("interval_begin", ""))[:7]),
        ("quarter", _quarter_bucket),
    ]
    if "volatility" in definitions:
        families.append(
            (
                "volatility",
                lambda row: _quantile_bucket(row, "realized_vol_12", definitions["volatility"]),
            )
        )
    if "momentum" in definitions:
        column = definitions["momentum"]["column"]
        families.append(("momentum", lambda row, col=column: _sign_bucket(row, col)))
    if "rsi" in definitions:
        families.append(("rsi", _rsi_bucket))
    if "macd" in definitions:
        families.append(("macd", lambda row: _macd_bucket(row, definitions["macd"])))
    if "volume" in definitions:
        families.append(
            ("volume", lambda row: _quantile_bucket(row, "volume", definitions["volume"]))
        )
    if "range" in definitions:
        families.append(("range", lambda row: _range_bucket(row, definitions["range"])))
    families.append(("regime_lite", lambda row: _regime_lite_bucket(row, definitions)))
    output = []
    for family, resolver in families:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(resolver(row)), []).append(row)
        for value, slice_rows in sorted(grouped.items()):
            output.append(_slice_metrics(family, value, slice_rows))
    return output


def _slice_metrics(
    family: str,
    value: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    positive_count = sum(labels)
    base_rate = positive_count / len(labels) if labels else 0.0
    matrix = _confusion(labels, predictions)
    both_classes = len(set(labels)) == 2
    row = {
        "slice_family": family,
        "slice_value": value,
        "row_count": len(rows),
        "positive_count": positive_count,
        "positive_rate": base_rate,
        "predicted_positive_mean_probability": (
            sum(probabilities) / len(probabilities) if probabilities else 0.0
        ),
        "roc_auc": _roc_auc(labels, probabilities) if both_classes else "",
        "average_precision": _average_precision(labels, probabilities) if both_classes else "",
        "balanced_accuracy": _balanced_accuracy(labels, predictions) if both_classes else "",
        "precision": _safe_ratio(matrix["tp"], matrix["tp"] + matrix["fp"]),
        "recall": _safe_ratio(matrix["tp"], matrix["tp"] + matrix["fn"]),
        "f1": _safe_ratio(2 * matrix["tp"], (2 * matrix["tp"]) + matrix["fp"] + matrix["fn"]),
        "false_positive_rate": _safe_ratio(matrix["fp"], matrix["fp"] + matrix["tn"]),
        "confidence_interval": "not_computed",
    }
    for fraction in TOP_K_FRACTIONS:
        topk = _topk(labels, probabilities, fraction)
        pct = int(fraction * 100)
        row[f"top_{pct}_precision"] = topk["precision"]
        row[f"top_{pct}_lift"] = topk["lift"]
        row[f"top_{pct}_recall"] = topk["recall"]
        row[f"top_{pct}_trade_count"] = topk["count"]
        row[f"top_{pct}_coverage"] = topk["coverage"]
    row["classification"] = _classify_slice(row)
    return row


def _classify_slice(row: Mapping[str, Any]) -> str:
    # pylint: disable=too-many-return-statements
    row_count = int(row["row_count"])
    positive_count = int(row["positive_count"])
    if row_count < MIN_ROWS_PER_SLICE:
        return "INSUFFICIENT_SAMPLE"
    if row["average_precision"] == "":
        return "METRIC_UNDEFINED"
    if positive_count < MIN_POSITIVES_PER_SLICE:
        return "INSUFFICIENT_POSITIVES"
    ap_delta = float(row["average_precision"]) - float(row["positive_rate"])
    top5_lift = float(row["top_5_lift"])
    top5_precision = float(row["top_5_precision"])
    if top5_lift >= 1.5 and top5_precision > float(row["positive_rate"]) and ap_delta >= 0.03:
        return "ENABLE_CANDIDATE"
    if top5_lift >= 1.2 or ap_delta >= 0.015:
        return "WATCHLIST_CANDIDATE"
    if top5_lift <= 1.0 or ap_delta <= 0.0:
        return "DISABLE_CANDIDATE"
    return "WATCHLIST_CANDIDATE"


def _search_breadth(slice_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classifications = [str(row["classification"]) for row in slice_rows]
    return {
        "slice_family_count": len({row["slice_family"] for row in slice_rows}),
        "slice_count": len(slice_rows),
        "enable_candidate_count": classifications.count("ENABLE_CANDIDATE"),
        "watchlist_candidate_count": classifications.count("WATCHLIST_CANDIDATE"),
        "disable_candidate_count": classifications.count("DISABLE_CANDIDATE"),
        "insufficient_sample_count": classifications.count("INSUFFICIENT_SAMPLE"),
        "insufficient_positives_count": classifications.count("INSUFFICIENT_POSITIVES"),
        "metric_undefined_count": classifications.count("METRIC_UNDEFINED"),
    }


def _recommend(search_breadth: Mapping[str, Any], slice_rows: Sequence[Mapping[str, Any]]) -> str:
    del slice_rows
    if int(search_breadth["enable_candidate_count"]) > 0:
        return "A. confirm promising slices on another run/fold/window"
    if int(search_breadth["watchlist_candidate_count"]) > 0:
        return "C. train a tiny stronger tabular model but keep same conditional diagnostics"
    if int(search_breadth["disable_candidate_count"]) > 0:
        return "D. add richer features/regime labels before modeling"
    return "E. reject this fee-exceedance path as weak"


def _strategy_implications(slice_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    enabled = [row for row in slice_rows if row["classification"] == "ENABLE_CANDIDATE"]
    if not enabled:
        return ["watchlist_or_disable_only_until_confirmed"]
    families = {row["slice_family"] for row in enabled}
    implications = ["opportunity_gate"]
    if "symbol" in families:
        implications.append("symbol_specific_gate")
    if "volatility" in families:
        implications.append("volatility_specific_gate")
    if "momentum" in families or "macd" in families:
        implications.append("trend_or_spark_gate")
    if "range" in families or "rsi" in families:
        implications.append("range_or_sideways_context_gate")
    implications.append("disable_in_bad_slices")
    return implications


def _write_outputs(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    slice_rows: Sequence[Mapping[str, Any]],
    bucket_definitions: Mapping[str, Any],
    output_files: Mapping[str, str],
) -> None:
    write_json_artifact(Path(output_files["conditional_usefulness_manifest_json"]), manifest)
    write_json_artifact(Path(output_files["conditional_usefulness_report_json"]), report)
    Path(output_files["conditional_usefulness_report_md"]).write_text(
        _markdown_report(report, slice_rows),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["conditional_usefulness_by_slice_csv"]), slice_rows)
    for family, key in (
        ("symbol", "conditional_usefulness_by_symbol_csv"),
        ("month", "conditional_usefulness_by_time_csv"),
        ("quarter", "conditional_usefulness_by_time_csv"),
        ("volatility", "conditional_usefulness_by_volatility_csv"),
        ("momentum", "conditional_usefulness_by_momentum_csv"),
        ("rsi", "conditional_usefulness_by_rsi_csv"),
        ("macd", "conditional_usefulness_by_macd_csv"),
        ("volume", "conditional_usefulness_by_volume_csv"),
        ("range", "conditional_usefulness_by_range_csv"),
        ("regime_lite", "conditional_usefulness_regime_lite_csv"),
    ):
        rows = [row for row in slice_rows if row["slice_family"] == family]
        if rows:
            path = Path(output_files[key])
            existing = _read_csv_rows(path) if path.exists() and key.endswith("time_csv") else []
            write_csv_artifact(path, [*existing, *rows])
    write_csv_artifact(
        Path(output_files["conditional_enable_disable_summary_csv"]),
        _enable_disable_rows(slice_rows),
    )
    write_json_artifact(
        Path(output_files["conditional_search_breadth_json"]),
        report["search_breadth"],
    )
    write_json_artifact(
        Path(output_files["conditional_bucket_definitions_json"]),
        bucket_definitions,
    )


def _write_full_vs_sample_comparison(
    run_dir: Path,
    output_dir: Path,
    full_slice_rows: Sequence[Mapping[str, Any]],
    full_report: Mapping[str, Any],
) -> dict[str, Any]:
    # pylint: disable=too-many-locals
    sample_dir = run_dir / "research_labels" / "vol_scaled" / CONDITIONAL_DIR_NAME
    if not (sample_dir / "conditional_usefulness_report.json").exists():
        comparison = {
            "comparison_status": "SAMPLE_CONDITIONAL_OUTPUT_MISSING",
            "rows_analyzed_sample": 0,
            "rows_analyzed_full": full_report.get("prediction_rows_analyzed", 0),
            "slice_families_sample": 0,
            "slice_families_full": full_report.get("search_breadth", {}).get(
                "slice_family_count", 0
            ),
            "slices_evaluated_sample": 0,
            "slices_evaluated_full": full_report.get("search_breadth", {}).get(
                "slice_count", 0
            ),
            "enable_candidates_sample": 0,
            "enable_candidates_full": full_report.get("search_breadth", {}).get(
                "enable_candidate_count", 0
            ),
            "watchlist_candidates_sample": 0,
            "watchlist_candidates_full": full_report.get("search_breadth", {}).get(
                "watchlist_candidate_count", 0
            ),
            "disable_candidates_sample": 0,
            "disable_candidates_full": full_report.get("search_breadth", {}).get(
                "disable_candidate_count", 0
            ),
            "insufficient_sample_slices_sample": 0,
            "insufficient_sample_slices_full": full_report.get("search_breadth", {}).get(
                "insufficient_sample_count", 0
            ),
            "candidate_slices_survived": [],
            "candidate_slices_downgraded": [],
            "candidate_slices_disappeared": [],
            "new_candidate_slices": [],
            "metric_deltas_by_matching_slice": [],
            "recommendation_sample": "",
            "recommendation_full": full_report.get("recommendation", ""),
        }
        write_json_artifact(
            output_dir / "full_vs_sample_conditional_comparison.json",
            comparison,
        )
        (output_dir / "full_vs_sample_conditional_comparison.md").write_text(
            _comparison_markdown(comparison),
            encoding="utf-8",
        )
        return comparison
    sample_report = _load_json(sample_dir / "conditional_usefulness_report.json")
    sample_rows = _read_csv_rows(sample_dir / "conditional_usefulness_by_slice.csv")
    sample_by_key = {
        (str(row["slice_family"]), str(row["slice_value"])): row for row in sample_rows
    }
    full_by_key = {
        (str(row["slice_family"]), str(row["slice_value"])): row for row in full_slice_rows
    }
    survived = []
    downgraded = []
    disappeared = []
    new_candidates = []
    deltas = []
    for key, sample in sample_by_key.items():
        full = full_by_key.get(key)
        if full is None:
            disappeared.append(_slice_key_dict(key))
            continue
        if sample.get("classification") == "ENABLE_CANDIDATE":
            if full.get("classification") == "ENABLE_CANDIDATE":
                survived.append(_slice_key_dict(key))
            else:
                downgraded.append(
                    {
                        **_slice_key_dict(key),
                        "sample_classification": sample.get("classification"),
                        "full_classification": full.get("classification"),
                    }
                )
        deltas.append(_slice_delta(key, sample, full))
    for key, full in full_by_key.items():
        if key not in sample_by_key and full.get("classification") == "ENABLE_CANDIDATE":
            new_candidates.append(_slice_key_dict(key))
    sample_breadth = sample_report.get("search_breadth", {})
    full_breadth = full_report.get("search_breadth", {})
    comparison = {
        "rows_analyzed_sample": sample_report.get("prediction_rows_analyzed", 0),
        "rows_analyzed_full": full_report.get("prediction_rows_analyzed", 0),
        "slice_families_sample": sample_breadth.get("slice_family_count", 0),
        "slice_families_full": full_breadth.get("slice_family_count", 0),
        "slices_evaluated_sample": sample_breadth.get("slice_count", 0),
        "slices_evaluated_full": full_breadth.get("slice_count", 0),
        "enable_candidates_sample": sample_breadth.get("enable_candidate_count", 0),
        "enable_candidates_full": full_breadth.get("enable_candidate_count", 0),
        "watchlist_candidates_sample": sample_breadth.get("watchlist_candidate_count", 0),
        "watchlist_candidates_full": full_breadth.get("watchlist_candidate_count", 0),
        "disable_candidates_sample": sample_breadth.get("disable_candidate_count", 0),
        "disable_candidates_full": full_breadth.get("disable_candidate_count", 0),
        "insufficient_sample_slices_sample": sample_breadth.get("insufficient_sample_count", 0),
        "insufficient_sample_slices_full": full_breadth.get("insufficient_sample_count", 0),
        "candidate_slices_survived": survived,
        "candidate_slices_downgraded": downgraded,
        "candidate_slices_disappeared": disappeared,
        "new_candidate_slices": new_candidates,
        "metric_deltas_by_matching_slice": deltas,
        "recommendation_sample": sample_report.get("recommendation", ""),
        "recommendation_full": full_report.get("recommendation", ""),
    }
    write_json_artifact(output_dir / "full_vs_sample_conditional_comparison.json", comparison)
    (output_dir / "full_vs_sample_conditional_comparison.md").write_text(
        _comparison_markdown(comparison),
        encoding="utf-8",
    )
    return comparison


def _slice_delta(
    key: tuple[str, str],
    sample: Mapping[str, Any],
    full: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **_slice_key_dict(key),
        "top_5_lift_sample": _as_float(sample.get("top_5_lift")),
        "top_5_lift_full": _as_float(full.get("top_5_lift")),
        "top_5_lift_delta": _as_float(full.get("top_5_lift"))
        - _as_float(sample.get("top_5_lift")),
        "average_precision_sample": _as_float(sample.get("average_precision")),
        "average_precision_full": _as_float(full.get("average_precision")),
        "average_precision_delta": _as_float(full.get("average_precision"))
        - _as_float(sample.get("average_precision")),
    }


def _slice_key_dict(key: tuple[str, str]) -> dict[str, str]:
    return {"slice_family": key[0], "slice_value": key[1]}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _comparison_markdown(comparison: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# M20 Full-Test vs Sample Conditional Comparison",
            "",
            f"- Rows analyzed sample/full: `{comparison['rows_analyzed_sample']}` / "
            f"`{comparison['rows_analyzed_full']}`",
            f"- Enable candidates sample/full: `{comparison['enable_candidates_sample']}` / "
            f"`{comparison['enable_candidates_full']}`",
            f"- Surviving enable candidates: `{len(comparison['candidate_slices_survived'])}`",
            f"- Downgraded enable candidates: `{len(comparison['candidate_slices_downgraded'])}`",
            "",
            "Slice changes remain research-only and require confirmation on another window.",
            "",
        ]
    )


def _output_files(output_dir: Path) -> dict[str, str]:
    names = {
        "conditional_usefulness_manifest_json": "conditional_usefulness_manifest.json",
        "conditional_usefulness_report_json": "conditional_usefulness_report.json",
        "conditional_usefulness_report_md": "conditional_usefulness_report.md",
        "conditional_usefulness_by_slice_csv": "conditional_usefulness_by_slice.csv",
        "conditional_usefulness_by_symbol_csv": "conditional_usefulness_by_symbol.csv",
        "conditional_usefulness_by_time_csv": "conditional_usefulness_by_time.csv",
        "conditional_usefulness_by_volatility_csv": "conditional_usefulness_by_volatility.csv",
        "conditional_usefulness_by_momentum_csv": "conditional_usefulness_by_momentum.csv",
        "conditional_usefulness_by_rsi_csv": "conditional_usefulness_by_rsi.csv",
        "conditional_usefulness_by_macd_csv": "conditional_usefulness_by_macd.csv",
        "conditional_usefulness_by_volume_csv": "conditional_usefulness_by_volume.csv",
        "conditional_usefulness_by_range_csv": "conditional_usefulness_by_range.csv",
        "conditional_usefulness_regime_lite_csv": "conditional_usefulness_regime_lite.csv",
        "conditional_enable_disable_summary_csv": "conditional_enable_disable_summary.csv",
        "conditional_search_breadth_json": "conditional_search_breadth.json",
        "conditional_bucket_definitions_json": "conditional_bucket_definitions.json",
        "full_vs_sample_conditional_comparison_json": "full_vs_sample_conditional_comparison.json",
        "full_vs_sample_conditional_comparison_md": "full_vs_sample_conditional_comparison.md",
    }
    return {key: str(output_dir / name) for key, name in names.items()}


def _markdown_report(
    report: Mapping[str, Any],
    slice_rows: Sequence[Mapping[str, Any]],
) -> str:
    enabled = [row for row in slice_rows if row["classification"] == "ENABLE_CANDIDATE"]
    disabled = [row for row in slice_rows if row["classification"] == "DISABLE_CANDIDATE"]
    best = sorted(slice_rows, key=lambda row: float(row["top_5_lift"]), reverse=True)[:5]
    worst = sorted(slice_rows, key=lambda row: float(row["top_5_lift"]))[:5]
    recap = report["global_baseline_recap"]
    return "\n".join(
        [
            "# M20 Conditional Usefulness Report",
            "",
            f"- Prediction rows analyzed: `{report['prediction_rows_analyzed']}`",
            f"- Baseline: `{report['baseline_name']}`",
            f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
            f"- Recommendation: `{report['recommendation']}`",
            "",
            "## Executive Summary",
            "",
            "- The fee-exceedance baseline is research-positive globally but not runtime-ready.",
            "- Conditional findings are strategy-ensemble inputs only.",
            f"- Enable candidates found: `{len(enabled)}`",
            f"- Disable candidates found: `{len(disabled)}`",
            "- Any promising slice requires confirmation on another fold/window.",
            "",
            "## Global Baseline Recap",
            "",
            f"- Base positive rate: `{float(recap['base_positive_rate']):.6f}`",
            f"- PR-AUC / average precision: `{float(recap['average_precision']):.6f}`",
            f"- ROC-AUC: `{float(recap['roc_auc']):.6f}`",
            f"- Balanced accuracy: `{float(recap['balanced_accuracy']):.6f}`",
            "",
            "## Best Candidate Slices",
            "",
            *[
                "- "
                f"`{row['slice_family']}={row['slice_value']}` "
                f"top5_lift=`{float(row['top_5_lift']):.6f}` "
                f"ap=`{float(row['average_precision'] or 0.0):.6f}` "
                f"rows=`{row['row_count']}` positives=`{row['positive_count']}`"
                for row in best
            ],
            "",
            "## Worst Slices",
            "",
            *[
                "- "
                f"`{row['slice_family']}={row['slice_value']}` "
                f"top5_lift=`{float(row['top_5_lift']):.6f}` "
                f"ap=`{float(row['average_precision'] or 0.0):.6f}` "
                f"classification=`{row['classification']}`"
                for row in worst
            ],
            "",
            "## Strategy-Ensemble Implications",
            "",
            f"- Future role candidates: `{', '.join(report['strategy_ensemble_implications'])}`",
            "- No runtime ensemble behavior was implemented.",
            "- This is not enough for runtime use.",
            "",
        ]
    )


def _enable_disable_rows(slice_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, int] = {}
    for row in slice_rows:
        grouped[str(row["classification"])] = grouped.get(str(row["classification"]), 0) + 1
    return [
        {"classification": key, "slice_count": value}
        for key, value in sorted(grouped.items())
    ]


def _global_recap(metrics: Mapping[str, Any]) -> dict[str, Any]:
    best = max(
        metrics.get("baselines", []),
        key=lambda row: (row.get("average_precision") or 0.0, row.get("balanced_accuracy") or 0.0),
        default={},
    )
    return {
        "baseline_name": best.get("baseline_name", ""),
        "base_positive_rate": best.get("positive_rate", 0.0),
        "average_precision": best.get("average_precision", 0.0),
        "roc_auc": best.get("roc_auc", 0.0),
        "balanced_accuracy": best.get("balanced_accuracy", 0.0),
    }


def _quarter_bucket(row: Mapping[str, Any]) -> str:
    stamp = str(row.get("interval_begin", ""))
    month = int(stamp[5:7]) if len(stamp) >= 7 and stamp[5:7].isdigit() else 1
    return f"{stamp[:4]}Q{((month - 1) // 3) + 1}"


def _quantile_definition(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, Any]:
    values = sorted(float(row[column]) for row in rows if _try_float(row.get(column)) is not None)
    return {"column": column, "thresholds": _low_high_thresholds(values)}


def _low_high_thresholds(values: Sequence[float]) -> dict[str, float]:
    return {"low_mid": _quantile(values, 1 / 3), "mid_high": _quantile(values, 2 / 3)}


def _quantile_bucket(row: Mapping[str, Any], column: str, definition: Mapping[str, Any]) -> str:
    value = float(row.get(column, 0.0) or 0.0)
    thresholds = definition["thresholds"]
    if value <= float(thresholds["low_mid"]):
        return "low"
    if value <= float(thresholds["mid_high"]):
        return "mid"
    return "high"


def _sign_bucket(row: Mapping[str, Any], column: str) -> str:
    value = float(row.get(column, 0.0) or 0.0)
    if value < 0.0:
        return "negative"
    if value > 0.0:
        return "positive"
    return "flat"


def _rsi_bucket(row: Mapping[str, Any]) -> str:
    value = float(row.get("rsi_14", 50.0) or 50.0)
    if value < 30.0:
        return "oversold"
    if value > 70.0:
        return "overbought"
    return "neutral"


def _macd_bucket(row: Mapping[str, Any], definition: Mapping[str, Any]) -> str:
    value = float(row.get("macd_line_12_26", 0.0) or 0.0)
    threshold = float(definition["near_zero_threshold"])
    if abs(value) <= threshold:
        return "near_zero"
    return "positive" if value > 0.0 else "negative"


def _range_bucket(row: Mapping[str, Any], definition: Mapping[str, Any]) -> str:
    value = _range_pct(row)
    thresholds = definition["thresholds"]
    if value <= float(thresholds["low_mid"]):
        return "low"
    if value <= float(thresholds["mid_high"]):
        return "mid"
    return "high"


def _regime_lite_bucket(row: Mapping[str, Any], definitions: Mapping[str, Any]) -> str:
    vol = (
        _quantile_bucket(row, "realized_vol_12", definitions["volatility"])
        if "volatility" in definitions else "unknown_vol"
    )
    momentum = _sign_bucket(row, definitions.get("momentum", {}).get("column", "log_return_1"))
    volume = (
        _quantile_bucket(row, "volume", definitions["volume"])
        if "volume" in definitions else "unknown_volume"
    )
    range_bucket = (
        _range_bucket(row, definitions["range"])
        if "range" in definitions else "unknown_range"
    )
    if vol == "high" and momentum == "positive":
        return "high_vol_positive_momentum"
    if vol == "high" and momentum == "negative":
        return "high_vol_negative_momentum"
    if vol == "low" and range_bucket == "low":
        return "low_vol_range_like"
    if volume == "high" and range_bucket == "high":
        return "high_volume_high_range"
    return "other"


def _topk(labels: Sequence[int], probabilities: Sequence[float], fraction: float) -> dict[str, Any]:
    count = max(1, int(len(labels) * fraction))
    ordered = sorted(range(len(labels)), key=lambda index: probabilities[index], reverse=True)
    selected = ordered[:count]
    positives = sum(labels[index] for index in selected)
    base_rate = sum(labels) / len(labels) if labels else 0.0
    precision = positives / count if count else 0.0
    return {
        "count": count,
        "coverage": count / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": _safe_ratio(positives, sum(labels)),
        "lift": precision / base_rate if base_rate else 0.0,
    }


def _balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    matrix = _confusion(labels, predictions)
    return (
        _safe_ratio(matrix["tp"], matrix["tp"] + matrix["fn"])
        + _safe_ratio(matrix["tn"], matrix["tn"] + matrix["fp"])
    ) / 2.0


def _confusion(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, int]:
    return {
        "tp": _count_pair(labels, predictions, 1, 1),
        "tn": _count_pair(labels, predictions, 0, 0),
        "fp": _count_pair(labels, predictions, 0, 1),
        "fn": _count_pair(labels, predictions, 1, 0),
    }


def _count_pair(
    labels: Sequence[int],
    predictions: Sequence[int],
    label_value: int,
    prediction_value: int,
) -> int:
    return sum(
        1 for label, prediction in zip(labels, predictions)
        if label == label_value and prediction == prediction_value
    )


def _roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, probabilities) if label == 1]
    negatives = [score for label, score in zip(labels, probabilities) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    if sum(labels) == 0:
        return None
    ordered = sorted(range(len(labels)), key=lambda index: probabilities[index], reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(ordered, start=1):
        if labels[index] == 1:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / sum(labels)


def _range_pct(row: Mapping[str, Any]) -> float:
    high = float(row.get("high_price", 0.0) or 0.0)
    low = float(row.get("low_price", 0.0) or 0.0)
    close = float(row.get("close_price", 1.0) or 1.0)
    return (high - low) / close if close else 0.0


def _has_column(rows: Sequence[Mapping[str, Any]], column: str) -> bool:
    return bool(rows) and column in rows[0] and _try_float(rows[0].get(column)) is not None


def _first_available(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    for column in columns:
        if _has_column(rows, column):
            return column
    return ""


def _quantile(values: Sequence[float], q_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("symbol", "")), str(row.get("interval_begin", "")))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing required conditional usefulness input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(payload)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _try_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None
