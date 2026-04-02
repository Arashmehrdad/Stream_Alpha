"""Cost-aware post-training threshold and regime analysis for completed M7 runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from app.common.serialization import make_json_safe
from app.regime.service import REGIME_LABELS


DEFAULT_THRESHOLD_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
DEFAULT_ANALYSIS_DIR_NAME = "threshold_analysis"
_DEFAULT_M7_ARTIFACT_ROOT = Path("artifacts") / "training" / "m7"
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


@dataclass(frozen=True, slots=True)
class OofPredictionRow:
    """Winner-model out-of-fold row with the fields needed for policy analysis."""

    model_name: str
    fold_index: int
    row_id: str
    y_true: int
    prob_up: float
    regime_label: str
    future_return_3: float
    long_only_gross_value_proxy: float
    long_only_net_value_proxy: float


@dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    """One post-training research policy variant for long-only threshold analysis."""

    name: str
    description: str
    blocked_regimes: frozenset[str] = frozenset()
    per_regime_thresholds: Mapping[str, float] | None = None

    def threshold_for_row(self, row: OofPredictionRow, default_threshold: float) -> float | None:
        """Return the effective threshold for this row, or None when longs are blocked."""
        if row.regime_label in self.blocked_regimes:
            return None
        if self.per_regime_thresholds is None:
            return default_threshold
        if row.regime_label in self.per_regime_thresholds:
            return float(self.per_regime_thresholds[row.regime_label])
        return default_threshold

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe policy description."""
        return {
            "name": self.name,
            "description": self.description,
            "blocked_regimes": sorted(self.blocked_regimes),
            "per_regime_thresholds": (
                {
                    regime_label: float(threshold)
                    for regime_label, threshold in sorted(self.per_regime_thresholds.items())
                }
                if self.per_regime_thresholds is not None
                else None
            ),
        }


def load_summary_payload(run_dir: Path) -> dict[str, Any]:
    """Load the completed-run summary payload."""
    return _load_summary_payload(run_dir)


def resolve_winner_model_name(summary_payload: Mapping[str, Any]) -> str:
    """Return the winner model_name from a completed-run summary payload."""
    return _resolve_winner_model_name(summary_payload)


def load_oof_predictions(
    path: Path,
    *,
    model_name: str,
) -> list[OofPredictionRow]:
    """Load winner-model OOF predictions needed for post-training analysis."""
    return _load_oof_predictions(path, model_name=model_name)


def resolve_fee_rate(
    summary_payload: Mapping[str, Any],
    predictions: list[OofPredictionRow],
) -> float:
    """Resolve the long-only fee rate for post-training economics analysis."""
    return _resolve_fee_rate(summary_payload, predictions)


def group_rows_by_fold(rows: list[OofPredictionRow]) -> dict[int, list[OofPredictionRow]]:
    """Group OOF rows by walk-forward fold."""
    return _group_rows_by_fold(rows)


def ordered_regime_groups(
    rows: list[OofPredictionRow],
) -> list[tuple[str, list[OofPredictionRow]]]:
    """Group OOF rows by regime label in a stable operator-facing order."""
    return _ordered_regime_groups(rows)


def compute_policy_metrics(
    rows: list[OofPredictionRow],
    trade_rows: list[OofPredictionRow],
    fee_rate: float,
) -> dict[str, Any]:
    """Compute cost-aware long-only metrics for one policy slice."""
    return _compute_policy_metrics(rows, trade_rows, fee_rate)


def select_worst_fold(per_fold_breakdown: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the weakest fold row from a policy-fold breakdown."""
    return _select_worst_fold(per_fold_breakdown)


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON analysis artifact deterministically."""
    _write_json(path, payload)


def write_csv_artifact(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one CSV analysis artifact deterministically."""
    _write_csv(path, rows)


def analyze_completed_run(
    *,
    run_dir: Path | None,
    thresholds: Iterable[float] = DEFAULT_THRESHOLD_GRID,
    model_name: str | None = None,
    per_regime_thresholds: Mapping[str, float] | None = None,
    analysis_dir_name: str = DEFAULT_ANALYSIS_DIR_NAME,
) -> dict[str, Any]:
    """Analyze a completed run directory and persist threshold-analysis artifacts."""
    resolved_run_dir = resolve_completed_run_dir(run_dir)
    summary_payload = _load_summary_payload(resolved_run_dir)
    resolved_model_name = model_name or _resolve_winner_model_name(summary_payload)
    predictions = _load_oof_predictions(
        resolved_run_dir / "oof_predictions.csv",
        model_name=resolved_model_name,
    )
    fee_rate = _resolve_fee_rate(summary_payload, predictions)
    normalized_thresholds = _normalize_threshold_grid(thresholds)
    policies = _build_policy_variants(per_regime_thresholds)

    sweep_results = [
        _analyze_policy_threshold(
            rows=predictions,
            policy=policy,
            threshold=threshold,
            fee_rate=fee_rate,
        )
        for policy in policies
        for threshold in normalized_thresholds
    ]
    best_by_policy = _select_best_results_by_policy(sweep_results, policies)
    best_overall = select_best_policy_result(best_by_policy)
    after_cost_positive_results = [
        result
        for result in sweep_results
        if result["after_cost_positive"]
    ]
    worst_fold_for_best_overall = _select_worst_fold(best_overall["per_fold_breakdown"])

    analysis_dir = resolved_run_dir / analysis_dir_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    threshold_sweep_payload = {
        "run_dir": str(resolved_run_dir),
        "analysis_dir": str(analysis_dir),
        "model_name": resolved_model_name,
        "fee_rate": fee_rate,
        "threshold_grid": normalized_thresholds,
        "policies": [policy.to_dict() for policy in policies],
        "results": sweep_results,
    }
    threshold_sweep_json_path = analysis_dir / "threshold_sweep.json"
    threshold_sweep_csv_path = analysis_dir / "threshold_sweep.csv"
    regime_policy_json_path = analysis_dir / "regime_policy_comparison.json"
    regime_policy_csv_path = analysis_dir / "regime_policy_comparison.csv"
    fold_breakdown_csv_path = analysis_dir / "fold_policy_breakdown.csv"
    summary_md_path = analysis_dir / "summary.md"

    _write_json(threshold_sweep_json_path, threshold_sweep_payload)
    _write_csv(
        threshold_sweep_csv_path,
        [_flatten_policy_result(result) for result in sweep_results],
    )

    regime_policy_payload = {
        "run_dir": str(resolved_run_dir),
        "analysis_dir": str(analysis_dir),
        "model_name": resolved_model_name,
        "fee_rate": fee_rate,
        "best_by_policy": best_by_policy,
        "best_overall_policy": best_overall,
        "any_after_cost_positive": bool(after_cost_positive_results),
        "after_cost_positive_policies": [
            _format_policy_threshold_reference(result)
            for result in after_cost_positive_results
        ],
        "worst_fold_for_best_overall": worst_fold_for_best_overall,
    }
    _write_json(regime_policy_json_path, regime_policy_payload)
    _write_csv(
        regime_policy_csv_path,
        [
            {
                **_flatten_policy_result(result),
                "is_best_overall_policy": result["policy_name"] == best_overall["policy_name"]
                and float(result["threshold"]) == float(best_overall["threshold"]),
            }
            for result in best_by_policy
        ],
    )
    _write_csv(
        fold_breakdown_csv_path,
        _flatten_fold_breakdown_rows(sweep_results),
    )

    analysis_summary = {
        "run_dir": str(resolved_run_dir),
        "analysis_dir": str(analysis_dir),
        "model_name": resolved_model_name,
        "fee_rate": fee_rate,
        "threshold_grid": normalized_thresholds,
        "best_global_threshold_policy": _find_policy_result(
            best_by_policy,
            "baseline_threshold_only",
        ),
        "best_no_long_in_trend_down_policy": _find_policy_result(
            best_by_policy,
            "no_long_in_trend_down",
        ),
        "best_no_long_in_trend_down_and_high_vol_policy": _find_policy_result(
            best_by_policy,
            "no_long_in_trend_down_and_high_vol",
        ),
        "best_per_regime_threshold_policy": _find_policy_result(
            best_by_policy,
            "per_regime_thresholds",
        ),
        "best_overall_policy": best_overall,
        "any_after_cost_positive": bool(after_cost_positive_results),
        "after_cost_positive_policies": [
            _format_policy_threshold_reference(result)
            for result in after_cost_positive_results
        ],
        "worst_fold_for_best_overall": worst_fold_for_best_overall,
        "output_files": {
            "threshold_sweep_json": str(threshold_sweep_json_path),
            "threshold_sweep_csv": str(threshold_sweep_csv_path),
            "regime_policy_comparison_json": str(regime_policy_json_path),
            "regime_policy_comparison_csv": str(regime_policy_csv_path),
            "fold_policy_breakdown_csv": str(fold_breakdown_csv_path),
            "summary_md": str(summary_md_path),
        },
    }
    summary_md_path.write_text(_build_summary_markdown(analysis_summary), encoding="utf-8")
    return make_json_safe(analysis_summary)


def resolve_completed_run_dir(run_dir: Path | None) -> Path:
    """Resolve a completed M7 run directory, defaulting to the newest known run."""
    if run_dir is not None:
        resolved = Path(run_dir).resolve()
        if not resolved.exists():
            raise ValueError(f"Completed run directory does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"Completed run path is not a directory: {resolved}")
        return resolved

    artifact_root = (_repo_root() / _DEFAULT_M7_ARTIFACT_ROOT).resolve()
    if not artifact_root.exists():
        raise ValueError(f"No M7 artifact root exists yet at {artifact_root}")
    run_dirs = sorted(
        (path for path in artifact_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise ValueError(f"No completed M7 run directories were found under {artifact_root}")
    return run_dirs[0].resolve()


def select_best_policy_result(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose a deterministic best-result row while avoiding zero-trade winners when possible."""
    result_list = [dict(result) for result in results]
    if not result_list:
        raise ValueError("No threshold-analysis results were available for selection")
    trading_results = [result for result in result_list if int(result["trade_count"]) > 0]
    candidate_results = trading_results or result_list
    return sorted(candidate_results, key=_policy_result_sort_key)[0]


def _policy_result_sort_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    threshold_value = result.get("threshold", result.get("prob_up_min", 0.0))
    return (
        -float(result["mean_long_only_net_value_proxy"]),
        -float(result["cumulative_long_only_net_value_proxy"]),
        -int(result["trade_count"]),
        float(threshold_value),
        str(result["policy_name"]),
        json.dumps(result.get("per_regime_thresholds") or {}, sort_keys=True),
        "|".join(result.get("blocked_regimes") or ()),
    )


def _select_best_results_by_policy(
    sweep_results: list[dict[str, Any]],
    policies: list[ThresholdPolicy],
) -> list[dict[str, Any]]:
    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in sweep_results:
        grouped_results[str(result["policy_name"])].append(result)
    ordered_best_results: list[dict[str, Any]] = []
    for policy in policies:
        policy_results = grouped_results.get(policy.name, [])
        if not policy_results:
            continue
        ordered_best_results.append(select_best_policy_result(policy_results))
    return ordered_best_results


def _find_policy_result(
    best_results: list[dict[str, Any]],
    policy_name: str,
) -> dict[str, Any] | None:
    for result in best_results:
        if result["policy_name"] == policy_name:
            return result
    return None


def _analyze_policy_threshold(
    *,
    rows: list[OofPredictionRow],
    policy: ThresholdPolicy,
    threshold: float,
    fee_rate: float,
) -> dict[str, Any]:
    trade_rows = _select_trade_rows(rows, policy=policy, threshold=threshold)
    per_fold_breakdown = [
        {
            "fold_index": fold_index,
            **_compute_policy_metrics(fold_rows, _select_trade_rows(fold_rows, policy, threshold), fee_rate),
        }
        for fold_index, fold_rows in sorted(_group_rows_by_fold(rows).items())
    ]
    per_regime_breakdown = [
        {
            "regime_label": regime_label,
            **_compute_policy_metrics(
                regime_rows,
                _select_trade_rows(regime_rows, policy, threshold),
                fee_rate,
            ),
        }
        for regime_label, regime_rows in _ordered_regime_groups(rows)
    ]
    return {
        "policy_name": policy.name,
        "policy_description": policy.description,
        "threshold": threshold,
        "blocked_regimes": sorted(policy.blocked_regimes),
        "per_regime_thresholds": (
            {
                regime_label: float(regime_threshold)
                for regime_label, regime_threshold in sorted(policy.per_regime_thresholds.items())
            }
            if policy.per_regime_thresholds is not None
            else None
        ),
        **_compute_policy_metrics(rows, trade_rows, fee_rate),
        "per_fold_breakdown": per_fold_breakdown,
        "per_regime_breakdown": per_regime_breakdown,
    }


def _group_rows_by_fold(rows: list[OofPredictionRow]) -> dict[int, list[OofPredictionRow]]:
    grouped_rows: dict[int, list[OofPredictionRow]] = defaultdict(list)
    for row in rows:
        grouped_rows[row.fold_index].append(row)
    return grouped_rows


def _ordered_regime_groups(
    rows: list[OofPredictionRow],
) -> list[tuple[str, list[OofPredictionRow]]]:
    grouped_rows: dict[str, list[OofPredictionRow]] = defaultdict(list)
    for row in rows:
        grouped_rows[row.regime_label].append(row)
    ordered_labels = list(REGIME_LABELS) + sorted(set(grouped_rows) - set(REGIME_LABELS))
    return [
        (regime_label, grouped_rows[regime_label])
        for regime_label in ordered_labels
        if regime_label in grouped_rows
    ]


def _select_trade_rows(
    rows: list[OofPredictionRow],
    policy: ThresholdPolicy,
    threshold: float,
) -> list[OofPredictionRow]:
    selected_rows: list[OofPredictionRow] = []
    for row in rows:
        effective_threshold = policy.threshold_for_row(row, threshold)
        if effective_threshold is None:
            continue
        if row.prob_up >= effective_threshold:
            selected_rows.append(row)
    return selected_rows


def _compute_policy_metrics(
    rows: list[OofPredictionRow],
    trade_rows: list[OofPredictionRow],
    fee_rate: float,
) -> dict[str, Any]:
    prediction_count = len(rows)
    trade_count = len(trade_rows)
    gross_sum = sum(row.future_return_3 for row in trade_rows)
    net_sum = sum(row.future_return_3 - fee_rate for row in trade_rows)
    precision_on_trades = (
        sum(int(row.y_true == 1) for row in trade_rows) / trade_count
        if trade_count > 0
        else None
    )
    if prediction_count == 0:
        return {
            "prediction_count": 0,
            "trade_count": 0,
            "trade_rate": 0.0,
            "precision_on_trades": precision_on_trades,
            "directional_accuracy_on_trades": precision_on_trades,
            "mean_long_only_gross_value_proxy": 0.0,
            "mean_long_only_net_value_proxy": 0.0,
            "cumulative_long_only_gross_value_proxy": 0.0,
            "cumulative_long_only_net_value_proxy": 0.0,
            "after_cost_positive": False,
        }
    mean_gross_value_proxy = gross_sum / prediction_count
    mean_net_value_proxy = net_sum / prediction_count
    return {
        "prediction_count": prediction_count,
        "trade_count": trade_count,
        "trade_rate": trade_count / prediction_count,
        "precision_on_trades": precision_on_trades,
        "directional_accuracy_on_trades": precision_on_trades,
        "mean_long_only_gross_value_proxy": mean_gross_value_proxy,
        "mean_long_only_net_value_proxy": mean_net_value_proxy,
        "cumulative_long_only_gross_value_proxy": gross_sum,
        "cumulative_long_only_net_value_proxy": net_sum,
        "after_cost_positive": mean_net_value_proxy > 0.0,
    }


def _flatten_policy_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy_name": result["policy_name"],
        "policy_description": result["policy_description"],
        "threshold": result["threshold"],
        "blocked_regimes": "|".join(result["blocked_regimes"]),
        "per_regime_thresholds_json": json.dumps(
            result["per_regime_thresholds"],
            sort_keys=True,
        )
        if result["per_regime_thresholds"] is not None
        else "",
        "prediction_count": result["prediction_count"],
        "trade_count": result["trade_count"],
        "trade_rate": result["trade_rate"],
        "precision_on_trades": result["precision_on_trades"],
        "directional_accuracy_on_trades": result["directional_accuracy_on_trades"],
        "mean_long_only_gross_value_proxy": result["mean_long_only_gross_value_proxy"],
        "mean_long_only_net_value_proxy": result["mean_long_only_net_value_proxy"],
        "cumulative_long_only_gross_value_proxy": result["cumulative_long_only_gross_value_proxy"],
        "cumulative_long_only_net_value_proxy": result["cumulative_long_only_net_value_proxy"],
        "after_cost_positive": result["after_cost_positive"],
    }


def _flatten_fold_breakdown_rows(
    sweep_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in sweep_results:
        for fold_row in result["per_fold_breakdown"]:
            rows.append(
                {
                    "policy_name": result["policy_name"],
                    "threshold": result["threshold"],
                    "blocked_regimes": "|".join(result["blocked_regimes"]),
                    "per_regime_thresholds_json": json.dumps(
                        result["per_regime_thresholds"],
                        sort_keys=True,
                    )
                    if result["per_regime_thresholds"] is not None
                    else "",
                    "fold_index": fold_row["fold_index"],
                    "prediction_count": fold_row["prediction_count"],
                    "trade_count": fold_row["trade_count"],
                    "trade_rate": fold_row["trade_rate"],
                    "precision_on_trades": fold_row["precision_on_trades"],
                    "directional_accuracy_on_trades": (
                        fold_row["directional_accuracy_on_trades"]
                    ),
                    "mean_long_only_gross_value_proxy": (
                        fold_row["mean_long_only_gross_value_proxy"]
                    ),
                    "mean_long_only_net_value_proxy": (
                        fold_row["mean_long_only_net_value_proxy"]
                    ),
                    "cumulative_long_only_gross_value_proxy": (
                        fold_row["cumulative_long_only_gross_value_proxy"]
                    ),
                    "cumulative_long_only_net_value_proxy": (
                        fold_row["cumulative_long_only_net_value_proxy"]
                    ),
                    "after_cost_positive": fold_row["after_cost_positive"],
                }
            )
    return rows


def _build_policy_variants(
    per_regime_thresholds: Mapping[str, float] | None,
) -> list[ThresholdPolicy]:
    policies = [
        ThresholdPolicy(
            name="baseline_threshold_only",
            description="Long-only threshold on prob_up with no regime blocks",
        ),
        ThresholdPolicy(
            name="no_long_in_trend_down",
            description="Global prob_up threshold plus a hard TREND_DOWN long block",
            blocked_regimes=frozenset({"TREND_DOWN"}),
        ),
        ThresholdPolicy(
            name="no_long_in_trend_down_and_high_vol",
            description="Global prob_up threshold plus hard TREND_DOWN and HIGH_VOL long blocks",
            blocked_regimes=frozenset({"TREND_DOWN", "HIGH_VOL"}),
        ),
    ]
    if per_regime_thresholds is not None:
        policies.append(
            ThresholdPolicy(
                name="per_regime_thresholds",
                description="Global prob_up threshold with explicit per-regime threshold overrides",
                per_regime_thresholds=dict(per_regime_thresholds),
            )
        )
    return policies


def _normalize_threshold_grid(thresholds: Iterable[float]) -> list[float]:
    normalized = [round(float(threshold), 6) for threshold in thresholds]
    if not normalized:
        raise ValueError("Threshold analysis requires at least one threshold candidate")
    for threshold in normalized:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid threshold candidate outside [0, 1]: {threshold}")
    return sorted(dict.fromkeys(normalized))


def _resolve_fee_rate(
    summary_payload: Mapping[str, Any],
    predictions: list[OofPredictionRow],
) -> float:
    economics_contract = summary_payload.get("economics_contract")
    if isinstance(economics_contract, Mapping) and "fee_rate" in economics_contract:
        return float(economics_contract["fee_rate"])
    fee_deltas = [
        row.long_only_gross_value_proxy - row.long_only_net_value_proxy
        for row in predictions
        if abs(row.long_only_gross_value_proxy - row.long_only_net_value_proxy) > 1e-12
    ]
    if not fee_deltas:
        return 0.0
    return float(median(fee_deltas))


def _load_summary_payload(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"Completed run is missing summary.json: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_winner_model_name(summary_payload: Mapping[str, Any]) -> str:
    winner = summary_payload.get("winner")
    if not isinstance(winner, Mapping) or "model_name" not in winner:
        raise ValueError("Completed run summary does not expose winner.model_name")
    return str(winner["model_name"])


def _load_oof_predictions(
    path: Path,
    *,
    model_name: str,
) -> list[OofPredictionRow]:
    if not path.exists():
        raise ValueError(f"Completed run is missing oof_predictions.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing_columns = [column for column in _REQUIRED_OOF_COLUMNS if column not in (reader.fieldnames or ())]
        if missing_columns:
            raise ValueError(
                "Out-of-fold predictions are missing required columns for threshold analysis: "
                f"{missing_columns}"
            )
        rows = [
            _row_from_oof_csv(raw_row)
            for raw_row in reader
            if raw_row["model_name"] == model_name
        ]
    if not rows:
        raise ValueError(
            f"No out-of-fold predictions were found for model {model_name!r} in {path}"
        )
    return rows


def _row_from_oof_csv(raw_row: Mapping[str, str]) -> OofPredictionRow:
    return OofPredictionRow(
        model_name=str(raw_row["model_name"]),
        fold_index=int(raw_row["fold_index"]),
        row_id=str(raw_row["row_id"]),
        y_true=int(raw_row["y_true"]),
        prob_up=float(raw_row["prob_up"]),
        regime_label=str(raw_row["regime_label"]),
        future_return_3=float(raw_row["future_return_3"]),
        long_only_gross_value_proxy=float(raw_row["long_only_gross_value_proxy"]),
        long_only_net_value_proxy=float(raw_row["long_only_net_value_proxy"]),
    )


def _select_worst_fold(per_fold_breakdown: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_fold_breakdown:
        raise ValueError("Fold breakdown is required to locate the weakest fold")
    return sorted(
        per_fold_breakdown,
        key=lambda row: (
            float(row["mean_long_only_net_value_proxy"]),
            float(row["cumulative_long_only_net_value_proxy"]),
            -int(row["trade_count"]),
            int(row["fold_index"]),
        ),
    )[0]


def _format_policy_threshold_reference(result: Mapping[str, Any]) -> str:
    return (
        f"{result['policy_name']}@{float(result['threshold']):.2f}"
        f"(net={float(result['mean_long_only_net_value_proxy']):.6f},"
        f" trades={int(result['trade_count'])})"
    )


def _build_summary_markdown(summary: Mapping[str, Any]) -> str:
    best_global = summary["best_global_threshold_policy"]
    best_trend_down_block = summary["best_no_long_in_trend_down_policy"]
    best_overall = summary["best_overall_policy"]
    worst_fold = summary["worst_fold_for_best_overall"]
    lines = [
        "# M7 Threshold Analysis",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Model analyzed: `{summary['model_name']}`",
        f"- Fee rate: `{float(summary['fee_rate']):.6f}`",
        "",
        "## Best Policies",
        "",
        _format_policy_summary_line("Best global threshold", best_global),
        _format_policy_summary_line("Best TREND_DOWN block", best_trend_down_block),
        _format_policy_summary_line("Best overall traded policy", best_overall),
        "",
        "## Economics Answer",
        "",
        f"- Any tested policy after-cost positive: `{summary['any_after_cost_positive']}`",
        (
            f"- After-cost positive policies: {', '.join(summary['after_cost_positive_policies'])}"
            if summary["after_cost_positive_policies"]
            else "- After-cost positive policies: none"
        ),
        "",
        "## Fold Stability",
        "",
        (
            "- Worst fold under the best overall policy: "
            f"`fold {int(worst_fold['fold_index'])}` "
            f"(trade_count={int(worst_fold['trade_count'])}, "
            f"trade_rate={float(worst_fold['trade_rate']):.4f}, "
            f"mean_net={float(worst_fold['mean_long_only_net_value_proxy']):.6f})"
        ),
        "",
        "## Output Files",
        "",
    ]
    for label, path in summary["output_files"].items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")
    lines.append(
        "This analysis is research support only. It does not change promotion, runtime, or "
        "live/paper/shadow threshold behavior."
    )
    lines.append("")
    return "\n".join(lines)


def _format_policy_summary_line(label: str, result: Mapping[str, Any] | None) -> str:
    if result is None:
        return f"- {label}: none"
    return (
        f"- {label}: `{result['policy_name']}` at `{float(result['threshold']):.2f}` "
        f"(trade_count={int(result['trade_count'])}, "
        f"trade_rate={float(result['trade_rate']):.4f}, "
        f"mean_net={float(result['mean_long_only_net_value_proxy']):.6f}, "
        f"after_cost_positive={bool(result['after_cost_positive'])})"
    )


def _parse_per_regime_thresholds(raw_values: list[str]) -> dict[str, float] | None:
    if not raw_values:
        return None
    thresholds: dict[str, float] = {}
    for raw_value in raw_values:
        if "=" not in raw_value:
            raise ValueError(
                "Per-regime threshold overrides must use REGIME_LABEL=THRESHOLD format"
            )
        regime_label, raw_threshold = raw_value.split("=", 1)
        thresholds[regime_label.strip()] = float(raw_threshold)
    return thresholds


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(make_json_safe(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    field_names = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    """Run threshold analysis for a completed M7 artifact directory."""
    parser = argparse.ArgumentParser(
        description="Analyze Stream Alpha completed M7 threshold and regime policies",
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the completed M7 artifact directory. Defaults to the newest M7 run.",
    )
    parser.add_argument(
        "--model-name",
        help="Optional model_name inside oof_predictions.csv. Defaults to summary winner.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help="Optional threshold grid. Defaults to 0.50 through 0.90 in 0.05 steps.",
    )
    parser.add_argument(
        "--per-regime-threshold",
        action="append",
        default=[],
        help="Optional override in REGIME_LABEL=THRESHOLD form. Can be repeated.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the saved analysis summary as JSON.",
    )
    arguments = parser.parse_args()
    try:
        analysis_summary = analyze_completed_run(
            run_dir=Path(arguments.run_dir) if arguments.run_dir else None,
            thresholds=arguments.thresholds or DEFAULT_THRESHOLD_GRID,
            model_name=arguments.model_name,
            per_regime_thresholds=_parse_per_regime_thresholds(
                list(arguments.per_regime_threshold)
            ),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(analysis_summary), sort_keys=True))
        return

    print(f"run_dir={analysis_summary['run_dir']}")
    print(f"analysis_dir={analysis_summary['analysis_dir']}")
    print(f"model_name={analysis_summary['model_name']}")
    print(
        "best_global_threshold="
        f"{_format_policy_threshold_reference(analysis_summary['best_global_threshold_policy'])}"
    )
    print(
        "best_no_long_in_trend_down="
        f"{_format_policy_threshold_reference(analysis_summary['best_no_long_in_trend_down_policy'])}"
    )
    print(
        "best_overall_policy="
        f"{_format_policy_threshold_reference(analysis_summary['best_overall_policy'])}"
    )
    print(f"any_after_cost_positive={analysis_summary['any_after_cost_positive']}")
    print(
        "worst_fold_for_best_overall="
        f"{int(analysis_summary['worst_fold_for_best_overall']['fold_index'])}"
    )


if __name__ == "__main__":
    main()
