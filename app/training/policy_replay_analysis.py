"""Research-only trade-ledger replay analysis for completed M7 policy candidates."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from app.common.serialization import make_json_safe
from app.training.policy_candidates import LongOnlyPolicyCandidate, find_policy_candidate
from app.training.threshold_analysis import (
    load_summary_payload,
    resolve_completed_run_dir,
    resolve_winner_model_name,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_REPLAY_ANALYSIS_DIR_NAME = "policy_replay_analysis"
DEFAULT_MULTI_RUN_ANALYSIS_DIR = Path("_analysis") / "policy_replay"
DEFAULT_M7_ARTIFACT_ROOT = Path("artifacts") / "training" / "m7"
DEFAULT_REPLAY_CANDIDATE_NAMES = (
    "default_long_only_050",
    "range_only_080",
    "m7_research_long_only_v1",
    "no_long_in_trend_down_high_vol_080",
    "range_or_trend_up_080",
    "per_regime_thresholds_v1",
)
LOW_TRADE_COUNT_WARNING_THRESHOLD = 20
MULTI_RUN_TRADE_COUNT_WARNING_THRESHOLD = 50
POSITIVE_RUN_RATE_WARNING_THRESHOLD = 0.60

_REQUIRED_REPLAY_COLUMNS = (
    "model_name",
    "fold_index",
    "row_id",
    "symbol",
    "interval_begin",
    "as_of_time",
    "y_true",
    "prob_up",
    "regime_label",
    "future_return_3",
    "long_only_gross_value_proxy",
    "long_only_net_value_proxy",
)


@dataclass(frozen=True, slots=True)
class ReplayOofRow:
    """Completed-run OOF row with the fields needed for replay evaluation."""

    model_name: str
    fold_index: int
    row_id: str
    symbol: str
    interval_begin: str
    as_of_time: str
    y_true: int
    prob_up: float
    regime_label: str
    future_return_3: float
    long_only_gross_value_proxy: float
    long_only_net_value_proxy: float
    source_index: int


def analyze_policy_replay(
    *,
    run_dir: Path | None,
    candidate_names: Iterable[str] | None = None,
    model_name: str | None = None,
    analysis_dir_name: str = DEFAULT_REPLAY_ANALYSIS_DIR_NAME,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Replay a bounded set of named policy candidates over one completed M7 run."""
    resolved_run_dir = resolve_completed_run_dir(run_dir)
    summary_payload = load_summary_payload(resolved_run_dir)
    resolved_model_name = model_name or resolve_winner_model_name(summary_payload)
    replay_rows = _load_replay_rows(
        resolved_run_dir / "oof_predictions.csv",
        model_name=resolved_model_name,
    )
    candidates = _resolve_candidates(candidate_names)
    evaluated_candidates = [
        _evaluate_replay_candidate(rows=replay_rows, candidate=candidate)
        for candidate in candidates
    ]
    candidate_results = [entry["summary"] for entry in evaluated_candidates]
    trade_ledger_rows = [
        ledger_row
        for entry in evaluated_candidates
        for ledger_row in entry["ledger_rows"]
    ]
    best_candidate = _select_best_replay_candidate(candidate_results)
    positive_candidates = [
        result["policy_name"]
        for result in candidate_results
        if bool(result["cumulative_net_positive"])
    ]

    analysis_dir = resolved_run_dir / analysis_dir_name
    summary_json_path = analysis_dir / "replay_summary.json"
    summary_csv_path = analysis_dir / "replay_summary.csv"
    ledger_csv_path = analysis_dir / "replay_trade_ledger.csv"
    summary_md_path = analysis_dir / "summary.md"

    analysis_summary = {
        "run_dir": str(resolved_run_dir),
        "analysis_dir": str(analysis_dir),
        "model_name": resolved_model_name,
        "candidate_definitions": [candidate.to_dict() for candidate in candidates],
        "candidate_results": candidate_results,
        "best_candidate": best_candidate,
        "any_positive_cumulative_candidate": bool(positive_candidates),
        "positive_cumulative_candidates": positive_candidates,
        "output_files": {
            "replay_summary_json": str(summary_json_path),
            "replay_summary_csv": str(summary_csv_path),
            "replay_trade_ledger_csv": str(ledger_csv_path),
            "summary_md": str(summary_md_path),
        },
    }

    if write_artifacts:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        write_json_artifact(summary_json_path, analysis_summary)
        write_csv_artifact(
            summary_csv_path,
            [
                {
                    **_flatten_replay_candidate(result),
                    "is_best_candidate": result["policy_name"] == best_candidate["policy_name"],
                }
                for result in candidate_results
            ],
        )
        write_csv_artifact(ledger_csv_path, trade_ledger_rows)
        summary_md_path.write_text(
            _build_single_run_summary_markdown(analysis_summary),
            encoding="utf-8",
        )
    return make_json_safe(analysis_summary)


def analyze_policy_replay_across_runs(
    *,
    artifact_root: Path | None = None,
    candidate_names: Iterable[str] | None = None,
    analysis_dir: Path | None = None,
) -> dict[str, Any]:
    """Aggregate replay metrics across completed M7 runs for the bounded candidate set."""
    resolved_artifact_root = _resolve_artifact_root(artifact_root)
    run_directories = _discover_run_directories(resolved_artifact_root)
    complete_runs, skipped_runs = _partition_complete_runs(run_directories)
    if not complete_runs:
        raise ValueError(
            f"No complete M7 runs with summary.json and oof_predictions.csv were found under {resolved_artifact_root}"
        )

    per_run_rows: list[dict[str, Any]] = []
    analyzable_run_count = 0
    for run_dir in complete_runs:
        try:
            replay_summary = analyze_policy_replay(
                run_dir=run_dir,
                candidate_names=candidate_names,
                write_artifacts=False,
            )
        except ValueError as error:
            if _is_skippable_replay_error(error):
                skipped_runs.append(
                    {
                        "run_id": run_dir.name,
                        "reason": f"incompatible for policy replay analysis: {error}",
                    }
                )
                continue
            raise
        analyzable_run_count += 1
        run_id = run_dir.name
        for candidate_result in replay_summary["candidate_results"]:
            per_run_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    **_flatten_replay_candidate(candidate_result),
                }
            )

    if not per_run_rows:
        raise ValueError(
            "No analyzable M7 runs with the required replay OOF columns were found under "
            f"{resolved_artifact_root}"
        )

    candidate_summaries = _aggregate_multi_run_replay_rows(
        per_run_rows=per_run_rows,
        complete_run_count=len(complete_runs),
    )
    best_candidate = _select_best_multi_run_candidate(candidate_summaries)
    resolved_analysis_dir = (
        analysis_dir.resolve()
        if analysis_dir is not None
        else (resolved_artifact_root / DEFAULT_MULTI_RUN_ANALYSIS_DIR).resolve()
    )
    resolved_analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = resolved_analysis_dir / "multi_run_replay_summary.json"
    summary_csv_path = resolved_analysis_dir / "multi_run_replay_summary.csv"
    summary_md_path = resolved_analysis_dir / "summary.md"

    analysis_summary = {
        "artifact_root": str(resolved_artifact_root),
        "analysis_dir": str(resolved_analysis_dir),
        "scanned_run_count": len(run_directories),
        "complete_run_count": len(complete_runs),
        "analyzable_run_count": analyzable_run_count,
        "skipped_runs": skipped_runs,
        "candidate_summaries": candidate_summaries,
        "run_breakdown_rows": per_run_rows,
        "best_candidate": best_candidate,
        "output_files": {
            "multi_run_replay_summary_json": str(summary_json_path),
            "multi_run_replay_summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
        },
    }

    write_json_artifact(summary_json_path, analysis_summary)
    write_csv_artifact(
        summary_csv_path,
        [
            {
                **summary,
                "warnings": " | ".join(summary["warnings"]),
            }
            for summary in candidate_summaries
        ],
    )
    summary_md_path.write_text(
        _build_multi_run_summary_markdown(analysis_summary),
        encoding="utf-8",
    )
    return make_json_safe(analysis_summary)


def _resolve_candidates(
    candidate_names: Iterable[str] | None,
) -> tuple[LongOnlyPolicyCandidate, ...]:
    resolved_names = tuple(candidate_names or DEFAULT_REPLAY_CANDIDATE_NAMES)
    if not resolved_names:
        resolved_names = DEFAULT_REPLAY_CANDIDATE_NAMES
    return tuple(find_policy_candidate(candidate_name) for candidate_name in resolved_names)


def _evaluate_replay_candidate(
    *,
    rows: list[ReplayOofRow],
    candidate: LongOnlyPolicyCandidate,
) -> dict[str, Any]:
    ordered_rows = _order_replay_rows(rows)
    trade_rows = [
        row
        for row in ordered_rows
        if _row_passes_candidate(row, candidate)
    ]
    available_regime_counts = _available_regime_counts(ordered_rows)
    ledger_rows, max_drawdown_proxy, longest_loss_streak = _build_trade_ledger(
        trade_rows=trade_rows,
        policy_name=candidate.name,
    )
    trade_count = len(trade_rows)
    cumulative_gross_proxy = sum(row.long_only_gross_value_proxy for row in trade_rows)
    cumulative_net_proxy = sum(row.long_only_net_value_proxy for row in trade_rows)
    mean_net_proxy = cumulative_net_proxy / trade_count if trade_count > 0 else 0.0
    median_net_proxy_per_trade = (
        float(median(row.long_only_net_value_proxy for row in trade_rows))
        if trade_count > 0
        else None
    )
    win_rate_on_trades = (
        sum(int(row.long_only_net_value_proxy > 0.0) for row in trade_rows) / trade_count
        if trade_count > 0
        else None
    )
    trades_in_trend_down = sum(int(row.regime_label == "TREND_DOWN") for row in trade_rows)
    trades_in_trend_up = sum(int(row.regime_label == "TREND_UP") for row in trade_rows)
    trades_in_range = sum(int(row.regime_label == "RANGE") for row in trade_rows)
    trades_in_high_vol = sum(int(row.regime_label == "HIGH_VOL") for row in trade_rows)
    never_trades_trend_up = (
        available_regime_counts.get("TREND_UP", 0) > 0 and trades_in_trend_up == 0
    )
    warnings = _build_single_run_warnings(
        trade_count=trade_count,
        cumulative_net_proxy=cumulative_net_proxy,
        max_drawdown_proxy=max_drawdown_proxy,
        never_trades_trend_up=never_trades_trend_up,
    )
    return {
        "summary": {
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
            "trade_count": trade_count,
            "cumulative_gross_proxy": cumulative_gross_proxy,
            "cumulative_net_proxy": cumulative_net_proxy,
            "mean_net_proxy": mean_net_proxy,
            "median_net_proxy_per_trade": median_net_proxy_per_trade,
            "win_rate_on_trades": win_rate_on_trades,
            "max_drawdown_proxy": max_drawdown_proxy,
            "longest_loss_streak": longest_loss_streak,
            "first_trade_time": trade_rows[0].as_of_time if trade_rows else None,
            "last_trade_time": trade_rows[-1].as_of_time if trade_rows else None,
            "trades_in_trend_down": trades_in_trend_down,
            "trades_in_trend_up": trades_in_trend_up,
            "trades_in_range": trades_in_range,
            "trades_in_high_vol": trades_in_high_vol,
            "available_trend_up_rows": available_regime_counts.get("TREND_UP", 0),
            "never_trades_trend_up": never_trades_trend_up,
            "cumulative_net_positive": cumulative_net_proxy > 0.0,
            "warnings": warnings,
            "evidence_still_thin": bool(warnings),
        },
        "ledger_rows": ledger_rows,
    }


def _row_passes_candidate(row: ReplayOofRow, candidate: LongOnlyPolicyCandidate) -> bool:
    effective_threshold = candidate.threshold_for_regime(row.regime_label)
    if effective_threshold is None:
        return False
    return row.prob_up >= effective_threshold


def _order_replay_rows(rows: list[ReplayOofRow]) -> list[ReplayOofRow]:
    return sorted(
        rows,
        key=lambda row: (
            row.as_of_time,
            row.interval_begin,
            row.symbol,
            row.row_id,
            row.source_index,
        ),
    )


def _build_trade_ledger(
    *,
    trade_rows: list[ReplayOofRow],
    policy_name: str,
) -> tuple[list[dict[str, Any]], float, int]:
    ledger_rows: list[dict[str, Any]] = []
    cumulative_gross_proxy = 0.0
    cumulative_net_proxy = 0.0
    running_peak = 0.0
    max_drawdown_proxy = 0.0
    longest_loss_streak = 0
    current_loss_streak = 0
    for trade_index, row in enumerate(trade_rows, start=1):
        cumulative_gross_proxy += row.long_only_gross_value_proxy
        cumulative_net_proxy += row.long_only_net_value_proxy
        running_peak = max(running_peak, cumulative_net_proxy)
        drawdown_proxy = cumulative_net_proxy - running_peak
        max_drawdown_proxy = min(max_drawdown_proxy, drawdown_proxy)
        if row.long_only_net_value_proxy < 0.0:
            current_loss_streak += 1
            longest_loss_streak = max(longest_loss_streak, current_loss_streak)
        else:
            current_loss_streak = 0
        ledger_rows.append(
            {
                "policy_name": policy_name,
                "trade_index": trade_index,
                "row_id": row.row_id,
                "symbol": row.symbol,
                "interval_begin": row.interval_begin,
                "as_of_time": row.as_of_time,
                "fold_index": row.fold_index,
                "regime_label": row.regime_label,
                "prob_up": row.prob_up,
                "y_true": row.y_true,
                "future_return_3": row.future_return_3,
                "long_only_gross_value_proxy": row.long_only_gross_value_proxy,
                "long_only_net_value_proxy": row.long_only_net_value_proxy,
                "cumulative_gross_proxy": cumulative_gross_proxy,
                "cumulative_net_proxy": cumulative_net_proxy,
                "drawdown_proxy": drawdown_proxy,
            }
        )
    return ledger_rows, max_drawdown_proxy, longest_loss_streak


def _available_regime_counts(rows: list[ReplayOofRow]) -> dict[str, int]:
    regime_counts: dict[str, int] = {}
    for row in rows:
        regime_counts[row.regime_label] = regime_counts.get(row.regime_label, 0) + 1
    return regime_counts


def _build_single_run_warnings(
    *,
    trade_count: int,
    cumulative_net_proxy: float,
    max_drawdown_proxy: float,
    never_trades_trend_up: bool,
) -> list[str]:
    warnings: list[str] = []
    if trade_count < LOW_TRADE_COUNT_WARNING_THRESHOLD:
        warnings.append("Trade count remains below 20 in replay.")
    if cumulative_net_proxy > 0.0 and abs(max_drawdown_proxy) >= cumulative_net_proxy:
        warnings.append("Positive cumulative net but max drawdown is at least as large as total gain.")
    if never_trades_trend_up:
        warnings.append("Candidate never trades TREND_UP in this run.")
    return warnings


def _select_best_replay_candidate(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not results:
        raise ValueError("No replay results were available for selection")
    trading_results = [result for result in results if int(result["trade_count"]) > 0]
    candidate_results = trading_results or results
    return sorted(
        candidate_results,
        key=lambda result: (
            -float(result["cumulative_net_proxy"]),
            -int(result["trade_count"]),
            -float(result["mean_net_proxy"]),
            -float(result["win_rate_on_trades"] or 0.0),
            -float(result["max_drawdown_proxy"]),
            str(result["policy_name"]),
        ),
    )[0]


def _flatten_replay_candidate(result: Mapping[str, Any]) -> dict[str, Any]:
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
        "trade_count": int(result["trade_count"]),
        "cumulative_gross_proxy": float(result["cumulative_gross_proxy"]),
        "cumulative_net_proxy": float(result["cumulative_net_proxy"]),
        "mean_net_proxy": float(result["mean_net_proxy"]),
        "median_net_proxy_per_trade": result["median_net_proxy_per_trade"],
        "win_rate_on_trades": result["win_rate_on_trades"],
        "max_drawdown_proxy": float(result["max_drawdown_proxy"]),
        "longest_loss_streak": int(result["longest_loss_streak"]),
        "first_trade_time": result["first_trade_time"] or "",
        "last_trade_time": result["last_trade_time"] or "",
        "trades_in_trend_down": int(result["trades_in_trend_down"]),
        "trades_in_trend_up": int(result["trades_in_trend_up"]),
        "trades_in_range": int(result["trades_in_range"]),
        "trades_in_high_vol": int(result["trades_in_high_vol"]),
        "available_trend_up_rows": int(result["available_trend_up_rows"]),
        "never_trades_trend_up": bool(result["never_trades_trend_up"]),
        "cumulative_net_positive": bool(result["cumulative_net_positive"]),
        "evidence_still_thin": bool(result["evidence_still_thin"]),
        "warnings": " | ".join(result["warnings"]),
    }


def _load_replay_rows(
    path: Path,
    *,
    model_name: str,
) -> list[ReplayOofRow]:
    if not path.exists():
        raise ValueError(f"Completed run is missing oof_predictions.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing_columns = [
            column
            for column in _REQUIRED_REPLAY_COLUMNS
            if column not in (reader.fieldnames or ())
        ]
        if missing_columns:
            raise ValueError(
                "Out-of-fold predictions are missing required columns for policy replay analysis: "
                f"{missing_columns}"
            )
        rows = [
            _row_from_replay_csv(raw_row, source_index)
            for source_index, raw_row in enumerate(reader)
            if raw_row["model_name"] == model_name
        ]
    if not rows:
        raise ValueError(
            f"No out-of-fold predictions were found for model {model_name!r} in {path}"
        )
    return rows


def _row_from_replay_csv(raw_row: Mapping[str, str], source_index: int) -> ReplayOofRow:
    return ReplayOofRow(
        model_name=str(raw_row["model_name"]),
        fold_index=int(raw_row["fold_index"]),
        row_id=str(raw_row["row_id"]),
        symbol=str(raw_row["symbol"]),
        interval_begin=str(raw_row["interval_begin"]),
        as_of_time=str(raw_row["as_of_time"]),
        y_true=int(raw_row["y_true"]),
        prob_up=float(raw_row["prob_up"]),
        regime_label=str(raw_row["regime_label"]),
        future_return_3=float(raw_row["future_return_3"]),
        long_only_gross_value_proxy=float(raw_row["long_only_gross_value_proxy"]),
        long_only_net_value_proxy=float(raw_row["long_only_net_value_proxy"]),
        source_index=source_index,
    )


def _resolve_artifact_root(artifact_root: Path | None) -> Path:
    if artifact_root is None:
        return (Path(__file__).resolve().parents[2] / DEFAULT_M7_ARTIFACT_ROOT).resolve()
    resolved = artifact_root.resolve()
    if not resolved.exists():
        raise ValueError(f"M7 artifact root does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"M7 artifact root is not a directory: {resolved}")
    return resolved


def _discover_run_directories(artifact_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in artifact_root.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        ),
        key=lambda path: path.name,
    )


def _partition_complete_runs(
    run_directories: list[Path],
) -> tuple[list[Path], list[dict[str, str]]]:
    complete_runs: list[Path] = []
    skipped_runs: list[dict[str, str]] = []
    for run_dir in run_directories:
        missing_files = [
            file_name
            for file_name in ("summary.json", "oof_predictions.csv")
            if not (run_dir / file_name).exists()
        ]
        if missing_files:
            skipped_runs.append(
                {
                    "run_id": run_dir.name,
                    "reason": f"missing required files: {', '.join(missing_files)}",
                }
            )
            continue
        complete_runs.append(run_dir)
    return complete_runs, skipped_runs


def _aggregate_multi_run_replay_rows(
    *,
    per_run_rows: list[dict[str, Any]],
    complete_run_count: int,
) -> list[dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in per_run_rows:
        grouped_rows.setdefault(str(row["policy_name"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for policy_name, rows in sorted(grouped_rows.items()):
        cumulative_values = [float(row["cumulative_net_proxy"]) for row in rows]
        max_drawdowns = [float(row["max_drawdown_proxy"]) for row in rows]
        total_trades_in_trend_up = sum(int(row["trades_in_trend_up"]) for row in rows)
        total_available_trend_up_rows = sum(
            int(row["available_trend_up_rows"]) for row in rows
        )
        positive_cumulative_run_count = sum(
            int(bool(row["cumulative_net_positive"])) for row in rows
        )
        summary = {
            "policy_name": policy_name,
            "run_count": len(rows),
            "complete_run_count": complete_run_count,
            "total_trade_count": sum(int(row["trade_count"]) for row in rows),
            "mean_cumulative_net_proxy": sum(cumulative_values) / len(rows),
            "median_cumulative_net_proxy": float(median(cumulative_values)),
            "positive_cumulative_run_count": positive_cumulative_run_count,
            "positive_cumulative_run_rate": positive_cumulative_run_count / len(rows),
            "worst_run_cumulative_net_proxy": min(cumulative_values),
            "best_run_cumulative_net_proxy": max(cumulative_values),
            "worst_run_max_drawdown_proxy": min(max_drawdowns),
            "average_max_drawdown_proxy": sum(max_drawdowns) / len(rows),
            "total_trades_in_trend_down": sum(
                int(row["trades_in_trend_down"]) for row in rows
            ),
            "total_trades_in_trend_up": total_trades_in_trend_up,
            "total_trades_in_range": sum(int(row["trades_in_range"]) for row in rows),
            "total_trades_in_high_vol": sum(
                int(row["trades_in_high_vol"]) for row in rows
            ),
            "never_trades_trend_up_across_runs": (
                total_available_trend_up_rows > 0 and total_trades_in_trend_up == 0
            ),
        }
        summary["warnings"] = _build_multi_run_warnings(summary)
        summary["evidence_still_thin"] = bool(summary["warnings"])
        summaries.append(summary)
    return summaries


def _build_multi_run_warnings(summary: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if int(summary["total_trade_count"]) < MULTI_RUN_TRADE_COUNT_WARNING_THRESHOLD:
        warnings.append("Total trade count remains below 50 across replayed runs.")
    if float(summary["positive_cumulative_run_rate"]) < POSITIVE_RUN_RATE_WARNING_THRESHOLD:
        warnings.append("Positive cumulative run rate remains below 0.60 across runs.")
    if int(summary["run_count"]) < 3:
        warnings.append("Fewer than 3 analyzable runs were available.")
    if bool(summary["never_trades_trend_up_across_runs"]):
        warnings.append("Candidate never trades TREND_UP across replayed runs.")
    return warnings


def _select_best_multi_run_candidate(
    candidate_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not candidate_summaries:
        raise ValueError("No multi-run replay candidate summaries were available")
    return sorted(
        candidate_summaries,
        key=lambda summary: (
            -float(summary["median_cumulative_net_proxy"]),
            -float(summary["mean_cumulative_net_proxy"]),
            -float(summary["positive_cumulative_run_rate"]),
            -int(summary["total_trade_count"]),
            -float(summary["average_max_drawdown_proxy"]),
            str(summary["policy_name"]),
        ),
    )[0]


def _is_skippable_replay_error(error: ValueError) -> bool:
    message = str(error)
    return any(
        indicator in message
        for indicator in (
            "Completed run summary does not expose winner.model_name",
            "Completed run is missing summary.json",
            "Completed run is missing oof_predictions.csv",
            "No out-of-fold predictions were found for model",
            "Out-of-fold predictions are missing required columns for policy replay analysis",
        )
    )


def _build_single_run_summary_markdown(summary: Mapping[str, Any]) -> str:
    best_candidate = summary["best_candidate"]
    lines = [
        "# M7 Policy Replay Analysis",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Model analyzed: `{summary['model_name']}`",
        "",
        "## Best Candidate",
        "",
        (
            f"- Best candidate by cumulative net proxy: `{best_candidate['policy_name']}` "
            f"(trade_count={int(best_candidate['trade_count'])}, "
            f"cumulative_net={float(best_candidate['cumulative_net_proxy']):.6f}, "
            f"max_drawdown={float(best_candidate['max_drawdown_proxy']):.6f})"
        ),
        (
            f"- Routing totals: `TREND_UP={int(best_candidate['trades_in_trend_up'])}`, "
            f"`TREND_DOWN={int(best_candidate['trades_in_trend_down'])}`, "
            f"`RANGE={int(best_candidate['trades_in_range'])}`, "
            f"`HIGH_VOL={int(best_candidate['trades_in_high_vol'])}`"
        ),
        f"- Evidence still thin: `{bool(best_candidate['evidence_still_thin'])}`",
    ]
    if best_candidate["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in best_candidate["warnings"])
    lines.extend(["", "## Output Files", ""])
    for label, path in summary["output_files"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "This replay is research support only. It uses saved proxy fields from completed "
            "runs and does not change production behavior, runtime policy, or promotion semantics.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_multi_run_summary_markdown(summary: Mapping[str, Any]) -> str:
    best_candidate = summary["best_candidate"]
    lines = [
        "# M7 Multi-Run Policy Replay Analysis",
        "",
        f"- Artifact root: `{summary['artifact_root']}`",
        f"- Scanned run directories: `{int(summary['scanned_run_count'])}`",
        f"- Complete runs with required files: `{int(summary['complete_run_count'])}`",
        f"- Analyzable runs: `{int(summary['analyzable_run_count'])}`",
        "",
        "## Best Candidate",
        "",
        (
            f"- Best candidate by cumulative net proxy across runs: `{best_candidate['policy_name']}` "
            f"(median_cumulative_net={float(best_candidate['median_cumulative_net_proxy']):.6f}, "
            f"mean_cumulative_net={float(best_candidate['mean_cumulative_net_proxy']):.6f}, "
            f"total_trade_count={int(best_candidate['total_trade_count'])})"
        ),
        (
            f"- Drawdown summary: `average_max_drawdown={float(best_candidate['average_max_drawdown_proxy']):.6f}`, "
            f"`worst_run_max_drawdown={float(best_candidate['worst_run_max_drawdown_proxy']):.6f}`"
        ),
        f"- Evidence still thin: `{bool(best_candidate['evidence_still_thin'])}`",
    ]
    if best_candidate["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in best_candidate["warnings"])
    lines.extend(["", "## Output Files", ""])
    for label, path in summary["output_files"].items():
        lines.append(f"- {label}: `{path}`")
    if summary["skipped_runs"]:
        lines.extend(["", "## Skipped Runs", ""])
        lines.extend(
            f"- `{row['run_id']}`: {row['reason']}" for row in summary["skipped_runs"]
        )
    lines.extend(
        [
            "",
            "This replay aggregation is research support only. It does not change runtime "
            "behavior or promotion semantics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run single-run or multi-run M7 policy replay analysis."""
    parser = argparse.ArgumentParser(
        description="Replay Stream Alpha named M7 policy candidates over completed runs",
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the completed M7 artifact directory. Defaults to the newest M7 run.",
    )
    parser.add_argument(
        "--artifact-root",
        help="Path to the M7 artifact root. Used with --multi-run.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Optional candidate name. Can be repeated.",
    )
    parser.add_argument(
        "--model-name",
        help="Optional model_name inside oof_predictions.csv. Defaults to summary winner.",
    )
    parser.add_argument(
        "--multi-run",
        action="store_true",
        help="Aggregate replay metrics across completed runs instead of one run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the saved replay summary as JSON.",
    )
    arguments = parser.parse_args()
    try:
        if arguments.multi_run:
            analysis_summary = analyze_policy_replay_across_runs(
                artifact_root=Path(arguments.artifact_root) if arguments.artifact_root else None,
                candidate_names=list(arguments.candidate) or None,
            )
        else:
            analysis_summary = analyze_policy_replay(
                run_dir=Path(arguments.run_dir) if arguments.run_dir else None,
                candidate_names=list(arguments.candidate) or None,
                model_name=arguments.model_name,
            )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(analysis_summary), sort_keys=True))
        return

    if arguments.multi_run:
        best_candidate = analysis_summary["best_candidate"]
        print(f"artifact_root={analysis_summary['artifact_root']}")
        print(f"analysis_dir={analysis_summary['analysis_dir']}")
        print(
            "best_candidate="
            f"{best_candidate['policy_name']}(median_cumulative_net={float(best_candidate['median_cumulative_net_proxy']):.6f},"
            f" total_trade_count={int(best_candidate['total_trade_count'])},"
            f" average_max_drawdown={float(best_candidate['average_max_drawdown_proxy']):.6f})"
        )
        print(f"evidence_still_thin={bool(best_candidate['evidence_still_thin'])}")
        if best_candidate["warnings"]:
            print(f"warnings={' | '.join(best_candidate['warnings'])}")
        return

    best_candidate = analysis_summary["best_candidate"]
    print(f"run_dir={analysis_summary['run_dir']}")
    print(f"analysis_dir={analysis_summary['analysis_dir']}")
    print(
        "best_candidate="
        f"{best_candidate['policy_name']}(cumulative_net={float(best_candidate['cumulative_net_proxy']):.6f},"
        f" trade_count={int(best_candidate['trade_count'])},"
        f" max_drawdown={float(best_candidate['max_drawdown_proxy']):.6f})"
    )
    print(f"evidence_still_thin={bool(best_candidate['evidence_still_thin'])}")
    if best_candidate["warnings"]:
        print(f"warnings={' | '.join(best_candidate['warnings'])}")


if __name__ == "__main__":
    main()
