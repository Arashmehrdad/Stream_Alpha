"""Research-only data and regime diagnostics for completed M7 runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.common.serialization import make_json_safe
from app.training.policy_candidates import (
    LOW_TRADE_COUNT_CAUTION_THRESHOLD,
    LongOnlyPolicyCandidate,
    build_default_policy_candidates,
    low_trade_count_caution,
)
from app.training.threshold_analysis import (
    compute_policy_metrics,
    group_rows_by_fold,
    load_summary_payload,
    ordered_regime_groups,
    resolve_fee_rate,
    resolve_winner_model_name,
    select_best_policy_result,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_ANALYSIS_DIR_NAME = "data_regime_diagnostics"
DEFAULT_M7_ARTIFACT_ROOT = Path("artifacts") / "training" / "m7"
DEFAULT_OPPORTUNITY_THRESHOLDS_BPS = (0, 5, 10, 20)
OPPORTUNITY_SPARSITY_RATE_THRESHOLD = 0.05
OPPORTUNITY_SPARSITY_COUNT_THRESHOLD = 20
_REQUIRED_OOF_COLUMNS = (
    "model_name",
    "fold_index",
    "row_id",
    "y_true",
    "prob_up",
    "regime_label",
    "future_return_3",
    "long_only_gross_value_proxy",
    "long_only_net_value_proxy",
)
_REQUIRED_FOLD_METRIC_COLUMNS = (
    "model_name",
    "fold_index",
    "directional_accuracy",
    "brier_score",
    "trade_count",
    "trade_rate",
    "mean_long_only_net_value_proxy",
)


@dataclass(frozen=True, slots=True)
class DiagnosticOofRow:
    """Winner-model OOF row enriched with symbol when present."""

    model_name: str
    fold_index: int
    row_id: str
    symbol: str | None
    y_true: int
    prob_up: float
    regime_label: str
    future_return_3: float
    long_only_gross_value_proxy: float
    long_only_net_value_proxy: float


def analyze_completed_run(
    *,
    run_dir: Path | None = None,
    opportunity_thresholds_bps: Iterable[int] = DEFAULT_OPPORTUNITY_THRESHOLDS_BPS,
    analysis_dir_name: str = DEFAULT_ANALYSIS_DIR_NAME,
) -> dict[str, Any]:
    """Analyze one completed M7 run for data quality, regime routing, and fold drift."""
    resolved_run_dir = _resolve_completed_run_dir(run_dir)
    required_files = _require_artifact_files(resolved_run_dir)
    summary_payload = load_summary_payload(resolved_run_dir)
    winner_model_name = resolve_winner_model_name(summary_payload)
    rows = _load_diagnostic_oof_predictions(
        required_files["oof_predictions"],
        model_name=winner_model_name,
    )
    fee_rate = resolve_fee_rate(summary_payload, rows)
    dataset_manifest = _load_json(required_files["dataset_manifest"])
    winner_fold_metrics = _load_winner_fold_metrics(
        required_files["fold_metrics"],
        model_name=winner_model_name,
    )
    normalized_thresholds_bps = _normalize_opportunity_thresholds(opportunity_thresholds_bps)

    label_diagnostics = _build_label_diagnostics(rows)
    opportunity_density = _build_opportunity_density(
        rows,
        thresholds_bps=normalized_thresholds_bps,
    )
    candidate_results = _evaluate_named_candidates(rows, fee_rate=fee_rate)
    best_named_candidate = select_best_policy_result(candidate_results)
    regime_routing = _build_regime_routing_summary(
        rows=rows,
        candidate_results=candidate_results,
        best_named_candidate=best_named_candidate,
    )
    fold_diagnostics = _build_fold_diagnostics(
        rows=rows,
        winner_fold_metrics=winner_fold_metrics,
        candidate_results=candidate_results,
        best_named_candidate=best_named_candidate,
        fee_rate=fee_rate,
    )
    feature_shift_support = _build_feature_shift_support(dataset_manifest)
    warnings = _build_warnings(
        opportunity_density=opportunity_density,
        best_named_candidate=best_named_candidate,
        regime_routing=regime_routing,
        feature_shift_support=feature_shift_support,
    )

    analysis_dir = resolved_run_dir / analysis_dir_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_json_path = analysis_dir / "diagnostics.json"
    label_csv_path = analysis_dir / "label_diagnostics.csv"
    opportunity_csv_path = analysis_dir / "opportunity_density.csv"
    regime_routing_csv_path = analysis_dir / "regime_routing.csv"
    fold_csv_path = analysis_dir / "fold_diagnostics.csv"
    summary_md_path = analysis_dir / "summary.md"

    diagnostics = {
        "run_dir": str(resolved_run_dir),
        "analysis_dir": str(analysis_dir),
        "model_name": winner_model_name,
        "fee_rate": fee_rate,
        "required_files": {name: str(path) for name, path in required_files.items()},
        "dataset_manifest": dataset_manifest,
        "label_diagnostics": label_diagnostics,
        "opportunity_density": opportunity_density,
        "candidate_results": candidate_results,
        "best_named_candidate": best_named_candidate,
        "regime_routing": regime_routing,
        "fold_diagnostics": fold_diagnostics,
        "feature_shift_support": feature_shift_support,
        "warnings": warnings,
        "output_files": {
            "diagnostics_json": str(diagnostics_json_path),
            "label_diagnostics_csv": str(label_csv_path),
            "opportunity_density_csv": str(opportunity_csv_path),
            "regime_routing_csv": str(regime_routing_csv_path),
            "fold_diagnostics_csv": str(fold_csv_path),
            "summary_md": str(summary_md_path),
        },
    }

    write_json_artifact(diagnostics_json_path, diagnostics)
    write_csv_artifact(label_csv_path, _flatten_label_diagnostics(label_diagnostics))
    write_csv_artifact(opportunity_csv_path, _flatten_opportunity_density(opportunity_density))
    write_csv_artifact(regime_routing_csv_path, _flatten_regime_routing(regime_routing))
    write_csv_artifact(fold_csv_path, fold_diagnostics["rows"])
    summary_md_path.write_text(_build_summary_markdown(diagnostics), encoding="utf-8")
    return make_json_safe(diagnostics)


def _resolve_completed_run_dir(run_dir: Path | None) -> Path:
    if run_dir is not None:
        resolved = Path(run_dir).resolve()
        if not resolved.exists():
            raise ValueError(f"Completed run directory does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"Completed run path is not a directory: {resolved}")
        return resolved

    artifact_root = (_repo_root() / DEFAULT_M7_ARTIFACT_ROOT).resolve()
    if not artifact_root.exists():
        raise ValueError(f"No M7 artifact root exists yet at {artifact_root}")
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
        raise ValueError(f"No completed M7 run directories were found under {artifact_root}")
    return run_dirs[0].resolve()


def _require_artifact_files(run_dir: Path) -> dict[str, Path]:
    required_files = {
        "summary": run_dir / "summary.json",
        "oof_predictions": run_dir / "oof_predictions.csv",
        "fold_metrics": run_dir / "fold_metrics.csv",
        "dataset_manifest": run_dir / "dataset_manifest.json",
    }
    missing_files = [str(path) for path in required_files.values() if not path.exists()]
    if missing_files:
        raise ValueError(
            "Completed run is missing required data/regime diagnostics files: "
            f"{missing_files}"
        )
    return required_files


def _load_diagnostic_oof_predictions(
    path: Path,
    *,
    model_name: str,
) -> list[DiagnosticOofRow]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        field_names = tuple(reader.fieldnames or ())
        missing_columns = [column for column in _REQUIRED_OOF_COLUMNS if column not in field_names]
        if missing_columns:
            raise ValueError(
                "Out-of-fold predictions are missing required columns for data/regime diagnostics: "
                f"{missing_columns}"
            )
        rows = [
            DiagnosticOofRow(
                model_name=str(raw_row["model_name"]),
                fold_index=int(raw_row["fold_index"]),
                row_id=str(raw_row["row_id"]),
                symbol=str(raw_row["symbol"]) if "symbol" in field_names and raw_row.get("symbol") else None,
                y_true=int(raw_row["y_true"]),
                prob_up=float(raw_row["prob_up"]),
                regime_label=str(raw_row["regime_label"]),
                future_return_3=float(raw_row["future_return_3"]),
                long_only_gross_value_proxy=float(raw_row["long_only_gross_value_proxy"]),
                long_only_net_value_proxy=float(raw_row["long_only_net_value_proxy"]),
            )
            for raw_row in reader
            if raw_row["model_name"] == model_name
        ]
    if not rows:
        raise ValueError(
            f"No out-of-fold predictions were found for model {model_name!r} in {path}"
        )
    return rows


def _load_winner_fold_metrics(
    path: Path,
    *,
    model_name: str,
) -> dict[int, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        field_names = tuple(reader.fieldnames or ())
        missing_columns = [
            column
            for column in _REQUIRED_FOLD_METRIC_COLUMNS
            if column not in field_names
        ]
        if missing_columns:
            raise ValueError(
                "Fold metrics are missing required columns for data/regime diagnostics: "
                f"{missing_columns}"
            )
        winner_rows = {
            int(raw_row["fold_index"]): {
                "directional_accuracy": float(raw_row["directional_accuracy"]),
                "brier_score": float(raw_row["brier_score"]),
                "trade_count": int(raw_row["trade_count"]),
                "trade_rate": float(raw_row["trade_rate"]),
                "mean_long_only_net_value_proxy": float(
                    raw_row["mean_long_only_net_value_proxy"]
                ),
            }
            for raw_row in reader
            if raw_row["model_name"] == model_name
        }
    if not winner_rows:
        raise ValueError(
            f"No fold metrics were found for model {model_name!r} in {path}"
        )
    return winner_rows


def _normalize_opportunity_thresholds(thresholds_bps: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(threshold) for threshold in thresholds_bps}))
    if not normalized:
        raise ValueError("At least one opportunity threshold is required")
    if any(threshold < 0 for threshold in normalized):
        raise ValueError("Opportunity thresholds must be non-negative bps values")
    return normalized


def _build_label_diagnostics(rows: list[DiagnosticOofRow]) -> dict[str, Any]:
    overall = _label_summary(rows)
    by_fold = [
        {
            "fold_index": fold_index,
            **_label_summary(fold_rows),
        }
        for fold_index, fold_rows in sorted(group_rows_by_fold(rows).items())
    ]
    by_regime = [
        {
            "regime_label": regime_label,
            **_label_summary(regime_rows),
        }
        for regime_label, regime_rows in ordered_regime_groups(rows)
    ]
    by_symbol = None
    if any(row.symbol is not None for row in rows):
        grouped_symbols: dict[str, list[DiagnosticOofRow]] = {}
        for row in rows:
            if row.symbol is None:
                continue
            grouped_symbols.setdefault(row.symbol, []).append(row)
        by_symbol = [
            {
                "symbol": symbol,
                **_label_summary(symbol_rows),
            }
            for symbol, symbol_rows in sorted(grouped_symbols.items())
        ]
    return {
        "overall": overall,
        "by_fold": by_fold,
        "by_regime": by_regime,
        "by_symbol": by_symbol,
    }


def _label_summary(rows: list[DiagnosticOofRow]) -> dict[str, Any]:
    prediction_count = len(rows)
    positive_label_count = sum(int(row.y_true == 1) for row in rows)
    return {
        "prediction_count": prediction_count,
        "positive_label_count": positive_label_count,
        "positive_label_rate": (
            positive_label_count / prediction_count if prediction_count > 0 else 0.0
        ),
    }


def _build_opportunity_density(
    rows: list[DiagnosticOofRow],
    *,
    thresholds_bps: tuple[int, ...],
) -> dict[str, Any]:
    overall = [
        _opportunity_summary(rows, threshold_bps)
        for threshold_bps in thresholds_bps
    ]
    by_regime = [
        {
            "regime_label": regime_label,
            **_opportunity_summary(regime_rows, threshold_bps),
        }
        for regime_label, regime_rows in ordered_regime_groups(rows)
        for threshold_bps in thresholds_bps
    ]
    twenty_bps_summary = next(
        row for row in overall if int(row["threshold_bps"]) == 20
    )
    return {
        "overall": overall,
        "by_regime": by_regime,
        "twenty_bps_sparse": _is_sparse_twenty_bps_opportunity(twenty_bps_summary),
        "twenty_bps_summary": twenty_bps_summary,
    }


def _opportunity_summary(
    rows: list[DiagnosticOofRow],
    threshold_bps: int,
) -> dict[str, Any]:
    threshold_rate = threshold_bps / 10_000.0
    opportunity_count = sum(int(row.future_return_3 > threshold_rate) for row in rows)
    prediction_count = len(rows)
    return {
        "threshold_bps": int(threshold_bps),
        "prediction_count": prediction_count,
        "opportunity_count": opportunity_count,
        "opportunity_rate": (
            opportunity_count / prediction_count if prediction_count > 0 else 0.0
        ),
    }


def _is_sparse_twenty_bps_opportunity(summary: Mapping[str, Any]) -> bool:
    return (
        int(summary["opportunity_count"]) < OPPORTUNITY_SPARSITY_COUNT_THRESHOLD
        or float(summary["opportunity_rate"]) < OPPORTUNITY_SPARSITY_RATE_THRESHOLD
    )


def _evaluate_named_candidates(
    rows: list[DiagnosticOofRow],
    *,
    fee_rate: float,
) -> list[dict[str, Any]]:
    return [
        _evaluate_candidate(rows=rows, candidate=candidate, fee_rate=fee_rate)
        for candidate in build_default_policy_candidates()
    ]


def _evaluate_candidate(
    *,
    rows: list[DiagnosticOofRow],
    candidate: LongOnlyPolicyCandidate,
    fee_rate: float,
) -> dict[str, Any]:
    trade_rows = _select_trade_rows(rows, candidate)
    per_regime_breakdown = [
        {
            "regime_label": regime_label,
            **compute_policy_metrics(regime_rows, _select_trade_rows(regime_rows, candidate), fee_rate),
        }
        for regime_label, regime_rows in ordered_regime_groups(rows)
    ]
    metrics = compute_policy_metrics(rows, trade_rows, fee_rate)
    return {
        "policy_name": candidate.name,
        "policy_description": candidate.description,
        "prob_up_min": float(candidate.prob_up_min),
        "blocked_regimes": sorted(candidate.blocked_regimes),
        "allowed_regimes": (
            sorted(candidate.allowed_regimes)
            if candidate.allowed_regimes is not None
            else None
        ),
        **metrics,
        "caution_text": low_trade_count_caution(int(metrics["trade_count"])) or "",
        "per_regime_breakdown": per_regime_breakdown,
    }


def _select_trade_rows(
    rows: list[DiagnosticOofRow],
    candidate: LongOnlyPolicyCandidate,
) -> list[DiagnosticOofRow]:
    selected_rows: list[DiagnosticOofRow] = []
    for row in rows:
        effective_threshold = candidate.threshold_for_regime(row.regime_label)
        if effective_threshold is None:
            continue
        if row.prob_up >= effective_threshold:
            selected_rows.append(row)
    return selected_rows


def _build_regime_routing_summary(
    *,
    rows: list[DiagnosticOofRow],
    candidate_results: list[dict[str, Any]],
    best_named_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    findings: list[str] = []
    policy_rows: list[dict[str, Any]] = []
    for result in candidate_results:
        total_trade_count = int(result["trade_count"])
        regime_breakdown = []
        for regime_row in result["per_regime_breakdown"]:
            trade_share = (
                int(regime_row["trade_count"]) / total_trade_count
                if total_trade_count > 0
                else 0.0
            )
            regime_breakdown.append(
                {
                    **regime_row,
                    "trade_share_of_policy": trade_share,
                    "net_direction": _net_direction(regime_row["mean_long_only_net_value_proxy"]),
                }
            )
        policy_rows.append(
            {
                "policy_name": result["policy_name"],
                "policy_description": result["policy_description"],
                "trade_count": total_trade_count,
                "trade_rate": float(result["trade_rate"]),
                "mean_long_only_net_value_proxy": float(
                    result["mean_long_only_net_value_proxy"]
                ),
                "after_cost_positive": bool(result["after_cost_positive"]),
                "caution_text": result["caution_text"],
                "per_regime_breakdown": regime_breakdown,
            }
        )

    default_policy = _find_policy(policy_rows, "default_long_only_050")
    best_policy = _find_policy(policy_rows, str(best_named_candidate["policy_name"]))
    default_trend_down = _find_regime_breakdown(default_policy, "TREND_DOWN")
    if (
        default_policy is not None
        and int(default_policy["trade_count"]) > 0
        and default_trend_down is not None
        and float(default_trend_down["trade_share_of_policy"]) >= 0.5
    ):
        findings.append(
            "TREND_DOWN accounts for at least half of default long-only trades."
        )
    best_policy_trend_up = _find_regime_breakdown(best_policy, "TREND_UP")
    total_trend_up_rows = sum(int(row.regime_label == "TREND_UP") for row in rows)
    if (
        best_policy is not None
        and total_trend_up_rows > 0
        and best_policy_trend_up is not None
        and int(best_policy_trend_up["trade_count"]) == 0
    ):
        findings.append(
            "Best named candidate takes no TREND_UP longs despite available TREND_UP rows."
        )
    return {
        "best_named_candidate_policy_name": str(best_named_candidate["policy_name"]),
        "policy_results": policy_rows,
        "suspicious_findings": findings,
    }


def _find_policy(
    candidate_results: list[dict[str, Any]],
    policy_name: str,
) -> dict[str, Any] | None:
    for result in candidate_results:
        if result["policy_name"] == policy_name:
            return result
    return None


def _find_regime_breakdown(
    policy_result: Mapping[str, Any] | None,
    regime_label: str,
) -> dict[str, Any] | None:
    if policy_result is None:
        return None
    for row in policy_result["per_regime_breakdown"]:
        if row["regime_label"] == regime_label:
            return dict(row)
    return None


def _build_fold_diagnostics(
    *,
    rows: list[DiagnosticOofRow],
    winner_fold_metrics: Mapping[int, Mapping[str, Any]],
    candidate_results: list[dict[str, Any]],
    best_named_candidate: Mapping[str, Any],
    fee_rate: float,
) -> dict[str, Any]:
    default_candidate = _find_candidate_definition("default_long_only_050")
    best_policy_name = str(best_named_candidate["policy_name"])
    best_policy_definition = _find_candidate_definition(best_policy_name)

    fold_rows = []
    for fold_index, fold_group in sorted(group_rows_by_fold(rows).items()):
        default_metrics = compute_policy_metrics(
            fold_group,
            _select_trade_rows(fold_group, default_candidate),
            fee_rate,
        )
        best_metrics = compute_policy_metrics(
            fold_group,
            _select_trade_rows(fold_group, best_policy_definition),
            fee_rate,
        )
        winner_metric_row = winner_fold_metrics.get(fold_index, {})
        fold_rows.append(
            {
                "fold_index": fold_index,
                "prediction_count": len(fold_group),
                "positive_label_rate": _label_summary(fold_group)["positive_label_rate"],
                "mean_future_return_3": (
                    sum(row.future_return_3 for row in fold_group) / len(fold_group)
                ),
                "default_trade_count": int(default_metrics["trade_count"]),
                "default_trade_rate": float(default_metrics["trade_rate"]),
                "default_mean_long_only_net_value_proxy": float(
                    default_metrics["mean_long_only_net_value_proxy"]
                ),
                "best_named_policy_name": best_policy_name,
                "best_named_trade_count": int(best_metrics["trade_count"]),
                "best_named_trade_rate": float(best_metrics["trade_rate"]),
                "best_named_mean_long_only_net_value_proxy": float(
                    best_metrics["mean_long_only_net_value_proxy"]
                ),
                "winner_directional_accuracy": winner_metric_row.get("directional_accuracy"),
                "winner_brier_score": winner_metric_row.get("brier_score"),
                "winner_saved_trade_count": winner_metric_row.get("trade_count"),
                "winner_saved_trade_rate": winner_metric_row.get("trade_rate"),
                "winner_saved_mean_long_only_net_value_proxy": winner_metric_row.get(
                    "mean_long_only_net_value_proxy"
                ),
            }
        )

    weakest_fold = sorted(
        fold_rows,
        key=lambda row: (
            float(row["best_named_mean_long_only_net_value_proxy"]),
            float(row["default_mean_long_only_net_value_proxy"]),
            float(row["mean_future_return_3"]),
            int(row["fold_index"]),
        ),
    )[0]
    return {
        "best_named_candidate_policy_name": best_policy_name,
        "rows": fold_rows,
        "weakest_fold": weakest_fold,
    }


def _find_candidate_definition(policy_name: str) -> LongOnlyPolicyCandidate:
    for candidate in build_default_policy_candidates():
        if candidate.name == policy_name:
            return candidate
    raise ValueError(f"Unknown named diagnostic policy candidate: {policy_name}")


def _build_feature_shift_support(dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
    del dataset_manifest
    return {
        "available": False,
        "reason": (
            "Completed run artifacts persist feature names and manifest metadata, but not "
            "per-row feature values, so feature distribution shift diagnostics are not "
            "available from this artifact alone."
        ),
    }


def _build_warnings(
    *,
    opportunity_density: Mapping[str, Any],
    best_named_candidate: Mapping[str, Any],
    regime_routing: Mapping[str, Any],
    feature_shift_support: Mapping[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if bool(opportunity_density["twenty_bps_sparse"]):
        warnings.append(">=20 bps opportunities are rare in the completed OOF rows.")
    sparse_caution = low_trade_count_caution(
        int(best_named_candidate["trade_count"]),
        threshold=LOW_TRADE_COUNT_CAUTION_THRESHOLD,
    )
    if sparse_caution is not None:
        warnings.append(sparse_caution)
    if any(
        finding == "TREND_DOWN accounts for at least half of default long-only trades."
        for finding in regime_routing["suspicious_findings"]
    ):
        warnings.append("TREND_DOWN dominates default long-only trades.")
    if not bool(feature_shift_support["available"]):
        warnings.append("Completed artifacts do not support feature-shift diagnostics.")
    return warnings


def _flatten_label_diagnostics(label_diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "dimension": "overall",
            "group_key": "overall",
            **label_diagnostics["overall"],
        }
    ]
    rows.extend(
        {
            "dimension": "fold",
            "group_key": str(row["fold_index"]),
            **{key: value for key, value in row.items() if key != "fold_index"},
        }
        for row in label_diagnostics["by_fold"]
    )
    rows.extend(
        {
            "dimension": "regime",
            "group_key": str(row["regime_label"]),
            **{key: value for key, value in row.items() if key != "regime_label"},
        }
        for row in label_diagnostics["by_regime"]
    )
    if label_diagnostics["by_symbol"] is not None:
        rows.extend(
            {
                "dimension": "symbol",
                "group_key": str(row["symbol"]),
                **{key: value for key, value in row.items() if key != "symbol"},
            }
            for row in label_diagnostics["by_symbol"]
        )
    return rows


def _flatten_opportunity_density(opportunity_density: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "group_type": "overall",
            "group_key": "overall",
            **row,
        }
        for row in opportunity_density["overall"]
    ]
    rows.extend(
        {
            "group_type": "regime",
            "group_key": str(row["regime_label"]),
            **{key: value for key, value in row.items() if key != "regime_label"},
        }
        for row in opportunity_density["by_regime"]
    )
    return rows


def _flatten_regime_routing(regime_routing: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in regime_routing["policy_results"]:
        for regime_row in policy["per_regime_breakdown"]:
            rows.append(
                {
                    "policy_name": policy["policy_name"],
                    "regime_label": regime_row["regime_label"],
                    "prediction_count": regime_row["prediction_count"],
                    "trade_count": regime_row["trade_count"],
                    "trade_rate": regime_row["trade_rate"],
                    "trade_share_of_policy": regime_row["trade_share_of_policy"],
                    "mean_long_only_gross_value_proxy": regime_row["mean_long_only_gross_value_proxy"],
                    "mean_long_only_net_value_proxy": regime_row["mean_long_only_net_value_proxy"],
                    "cumulative_long_only_net_value_proxy": regime_row["cumulative_long_only_net_value_proxy"],
                    "after_cost_positive": regime_row["after_cost_positive"],
                    "net_direction": regime_row["net_direction"],
                }
            )
    return rows


def _net_direction(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "flat"


def _build_summary_markdown(diagnostics: Mapping[str, Any]) -> str:
    label_rate = diagnostics["label_diagnostics"]["overall"]["positive_label_rate"]
    twenty_bps_summary = diagnostics["opportunity_density"]["twenty_bps_summary"]
    best_named_candidate = diagnostics["best_named_candidate"]
    weakest_fold = diagnostics["fold_diagnostics"]["weakest_fold"]
    suspicious_findings = diagnostics["regime_routing"]["suspicious_findings"]
    lines = [
        "# M7 Data and Regime Diagnostics",
        "",
        f"- Run directory: `{diagnostics['run_dir']}`",
        f"- Model analyzed: `{diagnostics['model_name']}`",
        f"- Overall positive label rate: `{float(label_rate):.4f}`",
        (
            f"- >=20 bps opportunities: `{int(twenty_bps_summary['opportunity_count'])}` / "
            f"`{int(twenty_bps_summary['prediction_count'])}` "
            f"(`{float(twenty_bps_summary['opportunity_rate']):.4f}`)"
        ),
        (
            f"- Best named policy candidate: `{best_named_candidate['policy_name']}` "
            f"(trade_count={int(best_named_candidate['trade_count'])}, "
            f"trade_rate={float(best_named_candidate['trade_rate']):.4f}, "
            f"mean_net={float(best_named_candidate['mean_long_only_net_value_proxy']):.6f}, "
            f"after_cost_positive={bool(best_named_candidate['after_cost_positive'])})"
        ),
        (
            f"- Weakest fold: `fold {int(weakest_fold['fold_index'])}` "
            f"(best_named_mean_net={float(weakest_fold['best_named_mean_long_only_net_value_proxy']):.6f}, "
            f"default_mean_net={float(weakest_fold['default_mean_long_only_net_value_proxy']):.6f})"
        ),
        "",
        "## Warnings",
        "",
    ]
    if diagnostics["warnings"]:
        lines.extend(f"- {warning}" for warning in diagnostics["warnings"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Suspicious Regime Routing",
            "",
        ]
    )
    if suspicious_findings:
        lines.extend(f"- {finding}" for finding in suspicious_findings)
    else:
        lines.append("- none detected from the completed-run artifact.")
    lines.extend(
        [
            "",
            "## Feature Shift Support",
            "",
            (
                f"- Available: `{bool(diagnostics['feature_shift_support']['available'])}`"
            ),
            f"- Detail: {diagnostics['feature_shift_support']['reason']}",
            "",
            "## Output Files",
            "",
        ]
    )
    for label, path in diagnostics["output_files"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "This diagnostics output is research support only. It does not change runtime "
            "policy, promotion semantics, or production behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    """Run completed-run M7 data and regime diagnostics."""
    parser = argparse.ArgumentParser(
        description="Analyze Stream Alpha completed M7 data and regime diagnostics",
    )
    parser.add_argument(
        "--run-dir",
        help="Optional completed M7 run directory. Defaults to the newest completed M7 run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the diagnostics summary as JSON.",
    )
    arguments = parser.parse_args()

    try:
        diagnostics = analyze_completed_run(
            run_dir=Path(arguments.run_dir) if arguments.run_dir else None,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(diagnostics), sort_keys=True))
        return

    twenty_bps_summary = diagnostics["opportunity_density"]["twenty_bps_summary"]
    weakest_fold = diagnostics["fold_diagnostics"]["weakest_fold"]
    suspicious_findings = diagnostics["regime_routing"]["suspicious_findings"]

    print(f"run_dir={diagnostics['run_dir']}")
    print(f"analysis_dir={diagnostics['analysis_dir']}")
    print(
        "overall_positive_label_rate="
        f"{float(diagnostics['label_diagnostics']['overall']['positive_label_rate']):.4f}"
    )
    print(
        "twenty_bps_opportunities="
        f"{int(twenty_bps_summary['opportunity_count'])}/"
        f"{int(twenty_bps_summary['prediction_count'])}"
        f"(rate={float(twenty_bps_summary['opportunity_rate']):.4f},"
        f" sparse={bool(diagnostics['opportunity_density']['twenty_bps_sparse'])})"
    )
    print(
        "weakest_fold="
        f"{int(weakest_fold['fold_index'])}"
        f"(best_named_mean_net={float(weakest_fold['best_named_mean_long_only_net_value_proxy']):.6f})"
    )
    print(
        "suspicious_regime_routing="
        f"{suspicious_findings[0] if suspicious_findings else 'none'}"
    )


if __name__ == "__main__":
    main()
