"""Named research-only policy-candidate evaluation for completed M7 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.common.serialization import make_json_safe
from app.training.policy_candidates import (
    LOW_TRADE_COUNT_CAUTION_THRESHOLD,
    LongOnlyPolicyCandidate,
    build_default_policy_candidates,
    find_policy_candidate,
    low_trade_count_caution,
)
from app.training.threshold_analysis import (
    OofPredictionRow,
    compute_policy_metrics,
    group_rows_by_fold,
    load_oof_predictions,
    load_summary_payload,
    ordered_regime_groups,
    resolve_completed_run_dir,
    resolve_fee_rate,
    resolve_winner_model_name,
    select_best_policy_result,
    select_worst_fold,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_ANALYSIS_DIR_NAME = "policy_candidate_analysis"


def evaluate_policy_candidates(
    *,
    run_dir: Path | None,
    candidate_names: Iterable[str] | None = None,
    model_name: str | None = None,
    analysis_dir_name: str = DEFAULT_ANALYSIS_DIR_NAME,
) -> dict[str, Any]:
    """Evaluate named research-only long-only candidates against one completed M7 run."""
    resolved_run_dir = resolve_completed_run_dir(run_dir)
    summary_payload = load_summary_payload(resolved_run_dir)
    resolved_model_name = model_name or resolve_winner_model_name(summary_payload)
    predictions = load_oof_predictions(
        resolved_run_dir / "oof_predictions.csv",
        model_name=resolved_model_name,
    )
    fee_rate = resolve_fee_rate(summary_payload, predictions)
    candidates = _resolve_candidates(candidate_names)
    candidate_results = [
        _evaluate_candidate(
            rows=predictions,
            candidate=candidate,
            fee_rate=fee_rate,
        )
        for candidate in candidates
    ]
    best_candidate = select_best_policy_result(candidate_results)
    worst_fold_for_best_candidate = select_worst_fold(best_candidate["per_fold_breakdown"])
    after_cost_positive_candidates = [
        result["policy_name"]
        for result in candidate_results
        if result["after_cost_positive"]
    ]
    candidate_family_summary = _build_candidate_family_summary(candidate_results)

    analysis_dir = resolved_run_dir / analysis_dir_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = analysis_dir / "policy_candidate_summary.json"
    summary_csv_path = analysis_dir / "policy_candidate_summary.csv"
    fold_breakdown_csv_path = analysis_dir / "policy_candidate_fold_breakdown.csv"
    summary_md_path = analysis_dir / "summary.md"

    analysis_summary = {
        "run_dir": str(resolved_run_dir),
        "analysis_dir": str(analysis_dir),
        "model_name": resolved_model_name,
        "fee_rate": fee_rate,
        "caution_trade_count_threshold": LOW_TRADE_COUNT_CAUTION_THRESHOLD,
        "candidate_definitions": [candidate.to_dict() for candidate in candidates],
        "candidate_results": candidate_results,
        "best_candidate": best_candidate,
        "candidate_family_summary": candidate_family_summary,
        "any_after_cost_positive": bool(after_cost_positive_candidates),
        "after_cost_positive_candidates": after_cost_positive_candidates,
        "worst_fold_for_best_candidate": worst_fold_for_best_candidate,
        "output_files": {
            "policy_candidate_summary_json": str(summary_json_path),
            "policy_candidate_summary_csv": str(summary_csv_path),
            "policy_candidate_fold_breakdown_csv": str(fold_breakdown_csv_path),
            "summary_md": str(summary_md_path),
        },
    }
    write_json_artifact(summary_json_path, analysis_summary)
    write_csv_artifact(
        summary_csv_path,
        [_flatten_candidate_result(result, best_candidate) for result in candidate_results],
    )
    write_csv_artifact(
        fold_breakdown_csv_path,
        _flatten_candidate_fold_rows(candidate_results),
    )
    summary_md_path.write_text(_build_summary_markdown(analysis_summary), encoding="utf-8")
    return make_json_safe(analysis_summary)


def _resolve_candidates(
    candidate_names: Iterable[str] | None,
) -> tuple[LongOnlyPolicyCandidate, ...]:
    if candidate_names is None:
        return build_default_policy_candidates()
    resolved_names = tuple(candidate_names)
    if not resolved_names:
        return build_default_policy_candidates()
    return tuple(find_policy_candidate(candidate_name) for candidate_name in resolved_names)


def _evaluate_candidate(
    *,
    rows: list[OofPredictionRow],
    candidate: LongOnlyPolicyCandidate,
    fee_rate: float,
) -> dict[str, Any]:
    trade_rows = _select_trade_rows(rows, candidate)
    per_fold_breakdown = [
        {
            "fold_index": fold_index,
            **compute_policy_metrics(fold_rows, _select_trade_rows(fold_rows, candidate), fee_rate),
        }
        for fold_index, fold_rows in sorted(group_rows_by_fold(rows).items())
    ]
    per_regime_breakdown = [
        {
            "regime_label": regime_label,
            **compute_policy_metrics(regime_rows, _select_trade_rows(regime_rows, candidate), fee_rate),
        }
        for regime_label, regime_rows in ordered_regime_groups(rows)
    ]
    metrics = compute_policy_metrics(rows, trade_rows, fee_rate)
    caution_text = low_trade_count_caution(int(metrics["trade_count"]))
    routing_flags = _build_candidate_routing_flags(
        rows=rows,
        trade_count=int(metrics["trade_count"]),
        per_regime_breakdown=per_regime_breakdown,
        positive_but_sparse=caution_text is not None and bool(metrics["after_cost_positive"]),
    )
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
        "per_regime_thresholds": (
            {
                regime_label: float(threshold)
                for regime_label, threshold in sorted(candidate.per_regime_thresholds.items())
            }
            if candidate.per_regime_thresholds is not None
            else None
        ),
        **metrics,
        "caution_text": caution_text,
        **routing_flags,
        "per_fold_breakdown": per_fold_breakdown,
        "per_regime_breakdown": per_regime_breakdown,
    }


def _select_trade_rows(
    rows: list[OofPredictionRow],
    candidate: LongOnlyPolicyCandidate,
) -> list[OofPredictionRow]:
    selected_rows: list[OofPredictionRow] = []
    for row in rows:
        effective_threshold = candidate.threshold_for_regime(row.regime_label)
        if effective_threshold is None:
            continue
        if row.prob_up >= effective_threshold:
            selected_rows.append(row)
    return selected_rows


def _flatten_candidate_result(
    result: Mapping[str, Any],
    best_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "policy_name": result["policy_name"],
        "policy_description": result["policy_description"],
        "prob_up_min": result["prob_up_min"],
        "blocked_regimes": "|".join(result["blocked_regimes"]),
        "allowed_regimes": "|".join(result["allowed_regimes"] or ()),
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
        "caution_text": result["caution_text"] or "",
        "trades_in_trend_down": result["trades_in_trend_down"],
        "trades_in_trend_up": result["trades_in_trend_up"],
        "trades_in_range": result["trades_in_range"],
        "trades_in_high_vol": result["trades_in_high_vol"],
        "trend_up_blocked_entirely": result["trend_up_blocked_entirely"],
        "positive_but_sparse": result["positive_but_sparse"],
        "is_best_candidate": result["policy_name"] == best_candidate["policy_name"],
    }


def _flatten_candidate_fold_rows(
    candidate_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in candidate_results:
        for fold_row in result["per_fold_breakdown"]:
            rows.append(
                {
                    "policy_name": result["policy_name"],
                    "prob_up_min": result["prob_up_min"],
                    "blocked_regimes": "|".join(result["blocked_regimes"]),
                    "allowed_regimes": "|".join(result["allowed_regimes"] or ()),
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
                    "directional_accuracy_on_trades": fold_row["directional_accuracy_on_trades"],
                    "mean_long_only_gross_value_proxy": fold_row["mean_long_only_gross_value_proxy"],
                    "mean_long_only_net_value_proxy": fold_row["mean_long_only_net_value_proxy"],
                    "cumulative_long_only_gross_value_proxy": fold_row["cumulative_long_only_gross_value_proxy"],
                    "cumulative_long_only_net_value_proxy": fold_row["cumulative_long_only_net_value_proxy"],
                    "after_cost_positive": fold_row["after_cost_positive"],
                }
            )
    return rows


def _build_summary_markdown(summary: Mapping[str, Any]) -> str:
    best_candidate = summary["best_candidate"]
    family_summary = summary["candidate_family_summary"]
    worst_fold = summary["worst_fold_for_best_candidate"]
    lines = [
        "# M7 Policy Candidate Evaluation",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Model analyzed: `{summary['model_name']}`",
        f"- Fee rate: `{float(summary['fee_rate']):.6f}`",
        "",
        "## Best Candidate",
        "",
        (
            f"- Best named candidate: `{best_candidate['policy_name']}` "
            f"(trade_count={int(best_candidate['trade_count'])}, "
            f"trade_rate={float(best_candidate['trade_rate']):.4f}, "
            f"mean_net={float(best_candidate['mean_long_only_net_value_proxy']):.6f}, "
            f"after_cost_positive={bool(best_candidate['after_cost_positive'])})"
        ),
        (
            f"- Routing flags: TREND_UP trades={int(best_candidate['trades_in_trend_up'])}, "
            f"TREND_DOWN trades={int(best_candidate['trades_in_trend_down'])}, "
            f"RANGE trades={int(best_candidate['trades_in_range'])}, "
            f"HIGH_VOL trades={int(best_candidate['trades_in_high_vol'])}, "
            f"trend_up_blocked_entirely={bool(best_candidate['trend_up_blocked_entirely'])}, "
            f"positive_but_sparse={bool(best_candidate['positive_but_sparse'])}"
        ),
    ]
    if best_candidate["caution_text"]:
        lines.extend(
            [
                "",
                "## Caution",
                "",
                f"- {best_candidate['caution_text']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Candidate Family Summary",
            "",
            _format_family_summary_line(
                "Best candidate by mean net proxy",
                family_summary["best_candidate_by_mean_net_proxy"],
            ),
            _format_family_summary_line(
                "Best positive candidate by trade count",
                family_summary["best_positive_candidate_by_trade_count"],
            ),
            (
                "- Positive candidates depend on blocking TREND_UP entirely: "
                f"`{bool(family_summary['positivity_depends_on_trend_up_blocking'])}`"
            ),
            (
                "- RANGE-only behavior dominates the positive candidate set: "
                f"`{bool(family_summary['range_only_behavior_dominates_positive_candidates'])}`"
            ),
            "",
            "## Economics Answer",
            "",
            f"- Any named candidate after-cost positive: `{summary['any_after_cost_positive']}`",
            (
                f"- After-cost positive named candidates: {', '.join(summary['after_cost_positive_candidates'])}"
                if summary["after_cost_positive_candidates"]
                else "- After-cost positive named candidates: none"
            ),
            "",
            "## Fold Stability",
            "",
            (
                f"- Weakest fold under the best candidate: `fold {int(worst_fold['fold_index'])}` "
                f"(trade_count={int(worst_fold['trade_count'])}, "
                f"trade_rate={float(worst_fold['trade_rate']):.4f}, "
                f"mean_net={float(worst_fold['mean_long_only_net_value_proxy']):.6f})"
            ),
            "",
            "## Output Files",
            "",
        ]
    )
    for label, path in summary["output_files"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "This evaluation is research support only. It does not change promotion, runtime, "
            "or live/paper/shadow policy behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_candidate_routing_flags(
    *,
    rows: list[OofPredictionRow],
    trade_count: int,
    per_regime_breakdown: list[dict[str, Any]],
    positive_but_sparse: bool,
) -> dict[str, Any]:
    available_regime_counts = {
        regime_label: len(regime_rows)
        for regime_label, regime_rows in ordered_regime_groups(rows)
    }
    trade_counts_by_regime = {
        str(row["regime_label"]): int(row["trade_count"])
        for row in per_regime_breakdown
    }
    trades_in_trend_up = trade_counts_by_regime.get("TREND_UP", 0)
    trades_in_trend_down = trade_counts_by_regime.get("TREND_DOWN", 0)
    trades_in_range = trade_counts_by_regime.get("RANGE", 0)
    trades_in_high_vol = trade_counts_by_regime.get("HIGH_VOL", 0)
    return {
        "trades_in_trend_down": trades_in_trend_down,
        "trades_in_trend_up": trades_in_trend_up,
        "trades_in_range": trades_in_range,
        "trades_in_high_vol": trades_in_high_vol,
        "trend_up_blocked_entirely": (
            available_regime_counts.get("TREND_UP", 0) > 0 and trades_in_trend_up == 0
        ),
        "range_only_behavior": trade_count > 0 and trades_in_range == trade_count,
        "positive_but_sparse": positive_but_sparse,
    }


def _build_candidate_family_summary(
    candidate_results: list[dict[str, Any]],
) -> dict[str, Any]:
    positive_candidates = [
        result for result in candidate_results if bool(result["after_cost_positive"])
    ]
    best_positive_by_trade_count = (
        sorted(
            positive_candidates,
            key=lambda result: (
                -int(result["trade_count"]),
                -float(result["mean_long_only_net_value_proxy"]),
                str(result["policy_name"]),
            ),
        )[0]
        if positive_candidates
        else None
    )
    return {
        "positive_candidate_names": [
            str(result["policy_name"]) for result in positive_candidates
        ],
        "best_candidate_by_mean_net_proxy": select_best_policy_result(candidate_results),
        "best_positive_candidate_by_trade_count": best_positive_by_trade_count,
        "positivity_depends_on_trend_up_blocking": (
            bool(positive_candidates)
            and all(bool(result["trend_up_blocked_entirely"]) for result in positive_candidates)
        ),
        "range_only_behavior_dominates_positive_candidates": (
            bool(positive_candidates)
            and all(bool(result["range_only_behavior"]) for result in positive_candidates)
        ),
    }


def _format_family_summary_line(
    label: str,
    result: Mapping[str, Any] | None,
) -> str:
    if result is None:
        return f"- {label}: none"
    return (
        f"- {label}: `{result['policy_name']}` "
        f"(trade_count={int(result['trade_count'])}, "
        f"mean_net={float(result['mean_long_only_net_value_proxy']):.6f}, "
        f"after_cost_positive={bool(result['after_cost_positive'])})"
    )


def main() -> None:
    """Run named research-only policy-candidate evaluation for one completed M7 run."""
    parser = argparse.ArgumentParser(
        description="Evaluate Stream Alpha named M7 research policy candidates",
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
        "--candidate",
        action="append",
        default=[],
        help="Optional candidate name. Can be repeated. Defaults to the bounded built-in set.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the saved candidate-analysis summary as JSON.",
    )
    arguments = parser.parse_args()
    try:
        analysis_summary = evaluate_policy_candidates(
            run_dir=Path(arguments.run_dir) if arguments.run_dir else None,
            candidate_names=list(arguments.candidate) or None,
            model_name=arguments.model_name,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(analysis_summary), sort_keys=True))
        return

    best_candidate = analysis_summary["best_candidate"]
    print(f"run_dir={analysis_summary['run_dir']}")
    print(f"analysis_dir={analysis_summary['analysis_dir']}")
    print(f"model_name={analysis_summary['model_name']}")
    print(
        "best_candidate="
        f"{best_candidate['policy_name']}(net={float(best_candidate['mean_long_only_net_value_proxy']):.6f},"
        f" trades={int(best_candidate['trade_count'])})"
    )
    print(f"any_after_cost_positive={analysis_summary['any_after_cost_positive']}")
    print(
        "worst_fold_for_best_candidate="
        f"{int(analysis_summary['worst_fold_for_best_candidate']['fold_index'])}"
    )
    if best_candidate["caution_text"]:
        print(f"caution={best_candidate['caution_text']}")


if __name__ == "__main__":
    main()
