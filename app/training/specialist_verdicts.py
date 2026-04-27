"""M20 recent-window specialist verdict helpers."""
# pylint: disable=too-many-arguments,too-many-locals,too-many-return-statements

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sklearn.metrics import accuracy_score, brier_score_loss

from app.common.time import to_rfc3339


DEFAULT_MIN_REGIME_PREDICTIONS_FOR_VERDICT = 100


def filter_recent_predictions(
    predictions: list[Any],
    window_days: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Return OOF predictions within the most recent window and metadata."""
    if not predictions:
        return [], {}

    max_time = max(_parse_time(prediction.as_of_time) for prediction in predictions)
    cutoff = max_time - timedelta(days=window_days)
    recent = [
        prediction
        for prediction in predictions
        if _parse_time(prediction.as_of_time) >= cutoff
    ]
    if not recent:
        return [], {}

    min_recent = min(_parse_time(prediction.as_of_time) for prediction in recent)
    return recent, {
        "window_days": window_days,
        "cutoff": to_rfc3339(cutoff),
        "min_as_of_time": to_rfc3339(min_recent),
        "max_as_of_time": to_rfc3339(max_time),
        "eligible_rows": len(recent),
        "total_oof_rows": len(predictions),
    }


def compute_max_drawdown(predictions: list[Any]) -> float:
    """Max drawdown of the cumulative net-value-proxy equity curve."""
    if not predictions:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for prediction in sorted(predictions, key=lambda row: row.as_of_time):
        cumulative += prediction.long_only_net_value_proxy
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return max_drawdown


def build_specialist_verdicts(
    *,
    recent_predictions: list[Any],
    model_configs: dict[str, dict[str, Any]],
    required_baselines: tuple[str, ...],
    incumbent_predictions: list[Any] | None = None,
    incumbent_model_version: str | None = None,
    max_drawdown_tolerance: float | None = None,
    min_regime_predictions: int = DEFAULT_MIN_REGIME_PREDICTIONS_FOR_VERDICT,
) -> dict[str, dict[str, Any]]:
    """Per-specialist accept/reject verdicts on target-regime recent evidence."""
    verdicts: dict[str, dict[str, Any]] = {}
    for model_name, model_config in model_configs.items():
        candidate_role = model_config.get("candidate_role")
        scope_regimes = model_config.get("scope_regimes")
        if not candidate_role or not scope_regimes:
            continue
        scope_set = set(scope_regimes)
        target_preds = [
            row
            for row in recent_predictions
            if row.model_name == model_name and row.regime_label in scope_set
        ]

        if len(target_preds) < min_regime_predictions:
            verdicts[model_name] = {
                "candidate_role": candidate_role,
                "target_regimes": list(scope_regimes),
                "target_prediction_count": len(target_preds),
                "verdict": "inconclusive",
                "reason": (
                    "Too few predictions in target regimes "
                    f"({len(target_preds)} < {min_regime_predictions})"
                ),
            }
            continue

        target_metrics = _compute_verdict_metrics(target_preds)
        target_net = target_metrics["mean_long_only_net_value_proxy"]
        target_drawdown = compute_max_drawdown(target_preds)
        baseline_comparisons, beats_all_baselines = _compare_baselines(
            recent_predictions=recent_predictions,
            required_baselines=required_baselines,
            scope_set=scope_set,
            target_net=target_net,
        )
        after_cost_positive = target_net > 0.0
        incumbent_comparison = _compare_incumbent(
            incumbent_predictions=incumbent_predictions,
            incumbent_model_version=incumbent_model_version,
            scope_set=scope_set,
            target_net=target_net,
            target_drawdown=target_drawdown,
            max_drawdown_tolerance=max_drawdown_tolerance,
        )
        verdict, reason, verdict_basis = _decide_verdict(
            incumbent_comparison=incumbent_comparison,
            after_cost_positive=after_cost_positive,
            beats_all_baselines=beats_all_baselines,
        )

        verdict_entry: dict[str, Any] = {
            "candidate_role": candidate_role,
            "target_regimes": list(scope_regimes),
            "target_prediction_count": len(target_preds),
            "verdict_basis": verdict_basis,
            "target_regime_metrics": {
                "mean_long_only_net_value_proxy": target_net,
                "mean_long_only_gross_value_proxy": target_metrics[
                    "mean_long_only_gross_value_proxy"
                ],
                "directional_accuracy": target_metrics["directional_accuracy"],
                "brier_score": target_metrics["brier_score"],
                "trade_count": target_metrics["trade_count"],
                "trade_rate": target_metrics["trade_rate"],
                "max_drawdown": target_drawdown,
            },
            "baseline_comparisons": baseline_comparisons,
            "after_cost_positive": after_cost_positive,
            "beats_all_baselines": beats_all_baselines,
            "verdict": verdict,
            "reason": reason,
        }
        if incumbent_comparison is not None:
            verdict_entry["incumbent_comparison"] = incumbent_comparison
        verdicts[model_name] = verdict_entry
    return verdicts


def _compare_baselines(
    *,
    recent_predictions: list[Any],
    required_baselines: tuple[str, ...],
    scope_set: set[str],
    target_net: float,
) -> tuple[dict[str, dict[str, Any]], bool]:
    baseline_comparisons: dict[str, dict[str, Any]] = {}
    beats_all_baselines = True
    for baseline_name in required_baselines:
        baseline_target_preds = [
            row
            for row in recent_predictions
            if row.model_name == baseline_name and row.regime_label in scope_set
        ]
        if not baseline_target_preds:
            baseline_comparisons[baseline_name] = {
                "available": False,
                "beats": False,
            }
            beats_all_baselines = False
            continue
        baseline_net = _compute_verdict_metrics(baseline_target_preds)[
            "mean_long_only_net_value_proxy"
        ]
        beats = target_net > baseline_net
        baseline_comparisons[baseline_name] = {
            "available": True,
            "baseline_net_value_proxy": baseline_net,
            "delta_vs_baseline": target_net - baseline_net,
            "beats": beats,
        }
        if not beats:
            beats_all_baselines = False
    return baseline_comparisons, beats_all_baselines


def _compare_incumbent(
    *,
    incumbent_predictions: list[Any] | None,
    incumbent_model_version: str | None,
    scope_set: set[str],
    target_net: float,
    target_drawdown: float,
    max_drawdown_tolerance: float | None,
) -> dict[str, Any] | None:
    if incumbent_predictions is None:
        return None
    incumbent_target_preds = [
        row for row in incumbent_predictions if row.regime_label in scope_set
    ]
    if not incumbent_target_preds:
        return None
    incumbent_metrics = _compute_verdict_metrics(incumbent_target_preds)
    incumbent_net = incumbent_metrics["mean_long_only_net_value_proxy"]
    incumbent_drawdown = compute_max_drawdown(incumbent_target_preds)
    drawdown_delta = target_drawdown - incumbent_drawdown
    drawdown_acceptable = True
    if max_drawdown_tolerance is not None:
        drawdown_acceptable = drawdown_delta <= max_drawdown_tolerance
    return {
        "incumbent_model_version": incumbent_model_version,
        "incumbent_net_value_proxy": incumbent_net,
        "incumbent_directional_accuracy": incumbent_metrics[
            "directional_accuracy"
        ],
        "incumbent_trade_count": incumbent_metrics["trade_count"],
        "incumbent_trade_rate": incumbent_metrics["trade_rate"],
        "incumbent_max_drawdown": incumbent_drawdown,
        "delta_net_value_proxy": target_net - incumbent_net,
        "delta_drawdown": drawdown_delta,
        "beats_incumbent": target_net > incumbent_net,
        "drawdown_acceptable": drawdown_acceptable,
    }


def _decide_verdict(
    *,
    incumbent_comparison: dict[str, Any] | None,
    after_cost_positive: bool,
    beats_all_baselines: bool,
) -> tuple[str, str, str]:
    if incumbent_comparison is not None:
        if (
            incumbent_comparison["beats_incumbent"]
            and after_cost_positive
            and incumbent_comparison["drawdown_acceptable"]
        ):
            return (
                "accepted",
                "Beats incumbent and positive after costs in target regimes",
                "incumbent_comparison",
            )
        if incumbent_comparison["beats_incumbent"] and not after_cost_positive:
            return (
                "rejected",
                "Beats incumbent but not positive after costs in target regimes",
                "incumbent_comparison",
            )
        if not incumbent_comparison["beats_incumbent"]:
            return (
                "rejected",
                "Does not beat incumbent in target regimes",
                "incumbent_comparison",
            )
        return (
            "rejected",
            "Drawdown worsened beyond tolerance vs incumbent",
            "incumbent_comparison",
        )

    if after_cost_positive and beats_all_baselines:
        return (
            "accepted",
            "Positive after costs and beats all baselines in target regimes "
            "(no incumbent available)",
            "baseline_only",
        )
    if beats_all_baselines and not after_cost_positive:
        return (
            "rejected",
            "Beats baselines but not positive after costs in target regimes",
            "baseline_only",
        )
    return (
        "rejected",
        "Does not beat all baselines in target regimes",
        "baseline_only",
    )


def _compute_verdict_metrics(predictions: list[Any]) -> dict[str, Any]:
    y_true = [prediction.y_true for prediction in predictions]
    y_pred = [prediction.y_pred for prediction in predictions]
    prob_up = [prediction.prob_up for prediction in predictions]
    prediction_count = len(predictions)
    trade_count = sum(prediction.long_trade_taken for prediction in predictions)
    return {
        "directional_accuracy": accuracy_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, prob_up),
        "trade_count": trade_count,
        "trade_rate": trade_count / prediction_count,
        "mean_long_only_gross_value_proxy": (
            sum(prediction.long_only_gross_value_proxy for prediction in predictions)
            / prediction_count
        ),
        "mean_long_only_net_value_proxy": (
            sum(prediction.long_only_net_value_proxy for prediction in predictions)
            / prediction_count
        ),
    }


def _parse_time(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
