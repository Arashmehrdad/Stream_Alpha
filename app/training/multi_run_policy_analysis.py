"""Research-only multi-run evaluation of named M7 policy candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from app.common.serialization import make_json_safe
from app.training.policy_candidate_analysis import evaluate_policy_candidates


DEFAULT_MULTI_RUN_ANALYSIS_DIR = Path("_analysis") / "policy_candidates"
DEFAULT_M7_ARTIFACT_ROOT = Path("artifacts") / "training" / "m7"


def analyze_policy_candidates_across_runs(
    *,
    artifact_root: Path | None = None,
    candidate_names: Iterable[str] | None = None,
    min_run_count: int = 1,
    analysis_dir: Path | None = None,
) -> dict[str, Any]:
    """Aggregate named policy-candidate results across completed M7 runs."""
    if min_run_count < 1:
        raise ValueError("min_run_count must be at least 1")

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
            run_analysis = evaluate_policy_candidates(
                run_dir=run_dir,
                candidate_names=candidate_names,
            )
        except ValueError as error:
            if _is_skippable_run_evaluation_error(error):
                skipped_runs.append(
                    {
                        "run_id": run_dir.name,
                        "reason": f"incompatible for policy-candidate analysis: {error}",
                    }
                )
                continue
            raise
        analyzable_run_count += 1
        run_id = run_dir.name
        for candidate_result in run_analysis["candidate_results"]:
            weakest_fold = _select_weakest_fold(candidate_result["per_fold_breakdown"])
            per_run_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "policy_name": candidate_result["policy_name"],
                    "prediction_count": int(candidate_result["prediction_count"]),
                    "trade_count": int(candidate_result["trade_count"]),
                    "trade_rate": float(candidate_result["trade_rate"]),
                    "mean_long_only_net_value_proxy": float(
                        candidate_result["mean_long_only_net_value_proxy"]
                    ),
                    "mean_long_only_gross_value_proxy": float(
                        candidate_result["mean_long_only_gross_value_proxy"]
                    ),
                    "after_cost_positive": bool(candidate_result["after_cost_positive"]),
                    "trades_in_trend_down": int(candidate_result["trades_in_trend_down"]),
                    "trades_in_trend_up": int(candidate_result["trades_in_trend_up"]),
                    "trades_in_range": int(candidate_result["trades_in_range"]),
                    "trades_in_high_vol": int(candidate_result["trades_in_high_vol"]),
                    "trend_up_blocked_entirely": bool(
                        candidate_result["trend_up_blocked_entirely"]
                    ),
                    "positive_but_sparse": bool(candidate_result["positive_but_sparse"]),
                    "weakest_fold": int(weakest_fold["fold_index"]) if weakest_fold else None,
                    "caution_flag": bool(candidate_result["caution_text"]),
                    "caution_text": candidate_result["caution_text"] or "",
                }
            )

    if not per_run_rows:
        raise ValueError(
            "No analyzable M7 runs with the required research OOF columns were found under "
            f"{resolved_artifact_root}"
        )

    candidate_summaries = _aggregate_candidate_rows(
        per_run_rows=per_run_rows,
        complete_run_count=len(complete_runs),
    )
    eligible_candidates = [
        summary
        for summary in candidate_summaries
        if int(summary["run_count"]) >= min_run_count
    ]
    if not eligible_candidates:
        raise ValueError(
            f"No policy candidates met min_run_count={min_run_count} across complete M7 runs"
        )
    best_candidate = sorted(
        eligible_candidates,
        key=lambda summary: (
            -float(summary["median_net_value_proxy_across_runs"]),
            -float(summary["mean_net_value_proxy_across_runs"]),
            -float(summary["positive_run_rate"]),
            -int(summary["total_trade_count"]),
            str(summary["policy_name"]),
        ),
    )[0]

    resolved_analysis_dir = (
        analysis_dir.resolve()
        if analysis_dir is not None
        else (resolved_artifact_root / DEFAULT_MULTI_RUN_ANALYSIS_DIR).resolve()
    )
    resolved_analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = resolved_analysis_dir / "multi_run_policy_summary.json"
    summary_csv_path = resolved_analysis_dir / "multi_run_policy_summary.csv"
    run_breakdown_csv_path = resolved_analysis_dir / "multi_run_policy_run_breakdown.csv"
    summary_md_path = resolved_analysis_dir / "summary.md"

    summary_payload = {
        "artifact_root": str(resolved_artifact_root),
        "analysis_dir": str(resolved_analysis_dir),
        "scanned_run_count": len(run_directories),
        "complete_run_count": len(complete_runs),
        "analyzable_run_count": analyzable_run_count,
        "skipped_runs": skipped_runs,
        "candidate_summaries": candidate_summaries,
        "best_candidate": best_candidate,
        "min_run_count": int(min_run_count),
        "output_files": {
            "multi_run_policy_summary_json": str(summary_json_path),
            "multi_run_policy_summary_csv": str(summary_csv_path),
            "multi_run_policy_run_breakdown_csv": str(run_breakdown_csv_path),
            "summary_md": str(summary_md_path),
        },
    }

    _write_json(summary_json_path, summary_payload)
    _write_csv(
        summary_csv_path,
        [
            {
                **summary,
                "is_best_candidate": summary["policy_name"] == best_candidate["policy_name"],
                "warnings": " | ".join(summary["warnings"]),
            }
            for summary in candidate_summaries
        ],
    )
    _write_csv(run_breakdown_csv_path, per_run_rows)
    summary_md_path.write_text(_build_summary_markdown(summary_payload), encoding="utf-8")
    return make_json_safe(summary_payload)


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


def _aggregate_candidate_rows(
    *,
    per_run_rows: list[dict[str, Any]],
    complete_run_count: int,
) -> list[dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in per_run_rows:
        grouped_rows.setdefault(str(row["policy_name"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for policy_name, rows in sorted(grouped_rows.items()):
        net_values = [float(row["mean_long_only_net_value_proxy"]) for row in rows]
        total_trade_count = sum(int(row["trade_count"]) for row in rows)
        positive_run_count = sum(int(bool(row["after_cost_positive"])) for row in rows)
        summary = {
            "policy_name": policy_name,
            "run_count": len(rows),
            "complete_run_count": complete_run_count,
            "total_prediction_count": sum(int(row["prediction_count"]) for row in rows),
        }
        summary["total_trade_count"] = total_trade_count
        summary["mean_trade_rate"] = sum(float(row["trade_rate"]) for row in rows) / len(rows)
        summary["mean_net_value_proxy_across_runs"] = sum(net_values) / len(rows)
        summary["median_net_value_proxy_across_runs"] = float(median(net_values))
        summary["positive_run_count"] = positive_run_count
        summary["positive_run_rate"] = positive_run_count / len(rows)
        summary["worst_run_net_value_proxy"] = min(net_values)
        summary["best_run_net_value_proxy"] = max(net_values)
        summary["total_trades_in_trend_down"] = sum(
            int(row["trades_in_trend_down"]) for row in rows
        )
        summary["total_trades_in_trend_up"] = sum(
            int(row["trades_in_trend_up"]) for row in rows
        )
        summary["total_trades_in_range"] = sum(
            int(row["trades_in_range"]) for row in rows
        )
        summary["total_trades_in_high_vol"] = sum(
            int(row["trades_in_high_vol"]) for row in rows
        )
        summary["positive_but_sparse_run_count"] = sum(
            int(bool(row["positive_but_sparse"])) for row in rows
        )
        summary["never_trades_trend_up_across_runs"] = (
            int(summary["total_trades_in_trend_up"]) == 0
        )
        warnings = _build_multi_run_warnings(summary)
        summary["warnings"] = warnings
        summary["evidence_too_sparse"] = bool(warnings)
        summaries.append(summary)
    return summaries


def _build_multi_run_warnings(summary: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if int(summary["total_trade_count"]) < 50:
        warnings.append("Total trade count remains below 50 across runs.")
    if float(summary["positive_run_rate"]) < 0.6:
        warnings.append("Positive run rate remains below 0.60 across runs.")
    if int(summary["run_count"]) < 3:
        warnings.append("Fewer than 3 analyzable runs were available.")
    if bool(summary["never_trades_trend_up_across_runs"]):
        warnings.append("Candidate never trades TREND_UP across analyzable runs.")
    return warnings


def _is_skippable_run_evaluation_error(error: ValueError) -> bool:
    message = str(error)
    return any(
        indicator in message
        for indicator in (
            "Completed run summary does not expose winner.model_name",
            "Completed run is missing summary.json",
            "Completed run is missing oof_predictions.csv",
            "No out-of-fold predictions were found for model",
            "Out-of-fold predictions are missing required columns",
        )
    )


def _select_weakest_fold(per_fold_breakdown: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not per_fold_breakdown:
        return None
    return sorted(
        per_fold_breakdown,
        key=lambda row: (
            float(row["mean_long_only_net_value_proxy"]),
            -int(row["trade_count"]),
            int(row["fold_index"]),
        ),
    )[0]


def _build_summary_markdown(summary: Mapping[str, Any]) -> str:
    best_candidate = summary["best_candidate"]
    lines = [
        "# M7 Multi-Run Policy Candidate Evaluation",
        "",
        f"- Artifact root: `{summary['artifact_root']}`",
        f"- Complete runs with required files: `{int(summary['complete_run_count'])}`",
        f"- Analyzable runs: `{int(summary['analyzable_run_count'])}`",
        f"- Scanned run directories: `{int(summary['scanned_run_count'])}`",
        "",
        "## Best Candidate",
        "",
        (
            f"- Best candidate by median net value proxy: `{best_candidate['policy_name']}` "
            f"(median_net={float(best_candidate['median_net_value_proxy_across_runs']):.6f}, "
            f"positive_run_rate={float(best_candidate['positive_run_rate']):.2f}, "
            f"total_trade_count={int(best_candidate['total_trade_count'])})"
        ),
        (
            f"- Routing totals for the best candidate: "
            f"`TREND_UP={int(best_candidate['total_trades_in_trend_up'])}`, "
            f"`TREND_DOWN={int(best_candidate['total_trades_in_trend_down'])}`, "
            f"`RANGE={int(best_candidate['total_trades_in_range'])}`, "
            f"`HIGH_VOL={int(best_candidate['total_trades_in_high_vol'])}`"
        ),
    ]
    if best_candidate["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in best_candidate["warnings"])
    lines.extend(
        [
            "",
            "## Output Files",
            "",
        ]
    )
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
            "This evaluation is research support only. It does not change production behavior "
            "or promotion semantics.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(make_json_safe(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv

    field_names = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run multi-run named policy-candidate analysis across completed M7 artifacts."""
    parser = argparse.ArgumentParser(
        description="Evaluate Stream Alpha named M7 policy candidates across completed runs",
    )
    parser.add_argument(
        "--artifact-root",
        help="Path to the M7 artifact root. Defaults to artifacts/training/m7.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Optional candidate name. Can be repeated.",
    )
    parser.add_argument(
        "--min-run-count",
        type=int,
        default=1,
        help="Minimum run_count required when selecting the best candidate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the saved multi-run summary as JSON.",
    )
    arguments = parser.parse_args()
    try:
        analysis_summary = analyze_policy_candidates_across_runs(
            artifact_root=Path(arguments.artifact_root) if arguments.artifact_root else None,
            candidate_names=list(arguments.candidate) or None,
            min_run_count=arguments.min_run_count,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(analysis_summary), sort_keys=True))
        return

    best_candidate = analysis_summary["best_candidate"]
    print(f"artifact_root={analysis_summary['artifact_root']}")
    print(f"analysis_dir={analysis_summary['analysis_dir']}")
    print(
        "best_candidate="
        f"{best_candidate['policy_name']}(median_net={float(best_candidate['median_net_value_proxy_across_runs']):.6f},"
        f" positive_run_rate={float(best_candidate['positive_run_rate']):.2f},"
        f" total_trade_count={int(best_candidate['total_trade_count'])})"
    )
    if best_candidate["warnings"]:
        print(f"warnings={' | '.join(best_candidate['warnings'])}")


if __name__ == "__main__":
    main()
