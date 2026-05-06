"""Research-only M20 rank-gate tail and condition concentration analysis."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_rank_gate_economics import LOCKED_POLICY
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "rank_gate_tail_analysis"
NET_DIAGNOSTIC_DIR = "rank_gate_net_diagnostics"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "TAIL_DIAGNOSTIC_ONLY",
    "NOT_PNL",
    "NO_RUNTIME",
    "NO_REGISTRY",
    "NO_PROMOTION",
    "NO_PROFIT_CLAIM",
    "SPARSE_SELECTION",
    "NET_PROXY_MIXED",
)
TOP_LIMITS = (5, 10)


def analyze_m20_rank_gate_tail(*, base_run_dir: Path) -> dict[str, Any]:
    """Analyze tail and condition concentration for locked rank-gate selected rows."""
    # pylint: disable=too-many-locals
    base_dir = Path(base_run_dir).resolve()
    source_dir = base_dir / "research_labels" / "vol_scaled" / NET_DIAGNOSTIC_DIR
    output_dir = base_dir / "research_labels" / "vol_scaled" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = source_dir / "selected_row_diagnostics.csv"
    output_files = _output_files(output_dir)
    if not selected_path.exists():
        return _write_blocker(base_dir, output_files, selected_path)

    rows = _read_csv_rows(selected_path)
    tail_contribution = _tail_contribution(rows)
    by_symbol = _group_summary(rows, "symbol")
    by_time = _group_summary(rows, "month") + _group_summary(rows, "quarter")
    by_condition = []
    for column in (
        "probability_bin",
        "volatility_bucket",
        "range_bucket",
        "volume_bucket",
        "momentum_bucket",
        "macd_bucket",
    ):
        by_condition.extend(_group_summary(rows, column))
    worst_rows = _top_rows(rows, reverse=False)
    best_rows = _top_rows(rows, reverse=True)
    recommendation = _recommendation(tail_contribution, by_symbol, by_condition)
    report = {
        "policy": LOCKED_POLICY,
        "selected_rows": len(rows),
        "recommendation": recommendation["recommendation"],
        "blockers": recommendation["blockers"],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "base_run_dir": str(base_dir),
        "source_selected_rows": str(selected_path),
        "output_dir": str(output_dir),
        "policy": LOCKED_POLICY,
        "runtime_status": "NOT_RUNTIME_READY",
        "promotion_status": "NOT_PROMOTABLE",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _markdown(report, tail_contribution),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["tail_contribution_csv"]), tail_contribution)
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_csv_artifact(Path(output_files["by_condition_csv"]), by_condition)
    write_csv_artifact(Path(output_files["worst_rows_csv"]), worst_rows)
    write_csv_artifact(Path(output_files["best_rows_csv"]), best_rows)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe({**report, "tail_contribution": tail_contribution})


def _write_blocker(
    base_dir: Path,
    output_files: Mapping[str, str],
    selected_path: Path,
) -> dict[str, Any]:
    recommendation = {
        "recommendation": "BLOCKED_ROW_LEVEL_NET_DIAGNOSTICS_MISSING",
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": ["ROW_LEVEL_ECONOMICS_MISSING", "RUN_NET_DIAGNOSTICS_FIRST"],
    }
    report = {
        "policy": LOCKED_POLICY,
        "selected_rows": 0,
        "recommendation": recommendation["recommendation"],
        "blockers": recommendation["blockers"],
        "honesty_flags": list(HONESTY_FLAGS),
        "missing_input": str(selected_path),
        "output_files": dict(output_files),
    }
    manifest = {
        "base_run_dir": str(base_dir),
        "missing_input": str(selected_path),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    Path(output_files["report_md"]).write_text(
        "# M20 Rank-Gate Tail Analysis\n\nBlocked: row-level net diagnostics are missing.\n",
        encoding="utf-8",
    )
    return make_json_safe(report)


def _tail_contribution(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output = []
    for run_label in sorted({row["run_label"] for row in rows}):
        run_rows = [row for row in rows if row["run_label"] == run_label]
        net_values = [_float(row.get("net_return_proxy")) for row in run_rows]
        total_net = sum(net_values)
        output.append(
            {
                "run_label": run_label,
                "selected_rows": len(run_rows),
                "true_positive_count": sum(_int(row.get("label")) for row in run_rows),
                "false_positive_count": sum(not _int(row.get("label")) for row in run_rows),
                "net_value_proxy": total_net,
                "mean_net_proxy": _mean(net_values),
                "median_net_proxy": _median(net_values),
                "p10_net_proxy": _quantile(net_values, 0.10),
                "p90_net_proxy": _quantile(net_values, 0.90),
                "tp_net_contribution": sum(
                    _float(row.get("net_return_proxy"))
                    for row in run_rows
                    if _int(row.get("label"))
                ),
                "fp_net_contribution": sum(
                    _float(row.get("net_return_proxy"))
                    for row in run_rows
                    if not _int(row.get("label"))
                ),
                "worst_5_contribution": _tail_sum(run_rows, 5, reverse=False),
                "worst_10_contribution": _tail_sum(run_rows, 10, reverse=False),
                "best_5_contribution": _tail_sum(run_rows, 5, reverse=True),
                "best_10_contribution": _tail_sum(run_rows, 10, reverse=True),
                "worst_5_tail_concentration_ratio": _tail_ratio(run_rows, 5),
                "worst_10_tail_concentration_ratio": _tail_ratio(run_rows, 10),
                "unstable_tail_concentration_flag": _tail_ratio(run_rows, 5) > 0.5,
            }
        )
    return output


def _group_summary(
    rows: Sequence[Mapping[str, str]],
    column: str,
) -> list[dict[str, Any]]:
    output = []
    for run_label in sorted({row["run_label"] for row in rows}):
        run_rows = [row for row in rows if row["run_label"] == run_label]
        for value in sorted({row.get(column, "") for row in run_rows}):
            group = [row for row in run_rows if row.get(column, "") == value]
            net_values = [_float(row.get("net_return_proxy")) for row in group]
            output.append(
                {
                    "run_label": run_label,
                    "slice_family": column,
                    "slice_value": value,
                    "selected_rows": len(group),
                    "true_positive_count": sum(_int(row.get("label")) for row in group),
                    "false_positive_count": sum(not _int(row.get("label")) for row in group),
                    "precision": (
                        sum(_int(row.get("label")) for row in group) / len(group)
                        if group else 0.0
                    ),
                    "net_value_proxy": sum(net_values),
                    "mean_net_proxy": _mean(net_values),
                    "median_net_proxy": _median(net_values),
                    "worst_5_contribution": _tail_sum(group, 5, reverse=False),
                    "tail_concentration_ratio": _tail_ratio(group, 5),
                }
            )
    return output


def _top_rows(rows: Sequence[Mapping[str, str]], *, reverse: bool) -> list[dict[str, Any]]:
    output = []
    for run_label in sorted({row["run_label"] for row in rows}):
        run_rows = [row for row in rows if row["run_label"] == run_label]
        ranked = sorted(
            run_rows,
            key=lambda row: _float(row.get("net_return_proxy")),
            reverse=reverse,
        )
        for rank, row in enumerate(ranked[:10], start=1):
            output.append(
                {
                    "run_label": run_label,
                    "rank": rank,
                    "symbol": row.get("symbol", ""),
                    "interval_begin": row.get("interval_begin", ""),
                    "month": row.get("month", ""),
                    "quarter": row.get("quarter", ""),
                    "label": row.get("label", ""),
                    "probability": row.get("probability", ""),
                    "future_return": row.get("future_return", ""),
                    "net_return_proxy": row.get("net_return_proxy", ""),
                    "probability_bin": row.get("probability_bin", ""),
                    "range_bucket": row.get("range_bucket", ""),
                    "momentum_bucket": row.get("momentum_bucket", ""),
                    "macd_bucket": row.get("macd_bucket", ""),
                }
            )
    return output


def _recommendation(
    tail_rows: Sequence[Mapping[str, Any]],
    by_symbol: Sequence[Mapping[str, Any]],
    by_condition: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    negative_runs = [row for row in tail_rows if _float(row.get("net_value_proxy")) < 0.0]
    concentrated = [
        row for row in list(by_symbol) + list(by_condition)
        if _float(row.get("tail_concentration_ratio")) > 0.5
        and int(row.get("selected_rows", 0)) >= 5
    ]
    recommendation = (
        "REVIEW_TAIL_CONCENTRATION_BEFORE_ANY_POLICY_OR_STRATEGY_STEP"
        if negative_runs or concentrated
        else "KEEP_RESEARCH_ONLY_TAIL_MONITORING"
    )
    return {
        "recommendation": recommendation,
        "negative_net_runs": [row["run_label"] for row in negative_runs],
        "unstable_concentration_count": len(concentrated),
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": [
            "NOT_PNL",
            "NOT_PROFIT_EVIDENCE",
            "SPARSE_SELECTION",
            "NET_PROXY_MIXED",
            "TAIL_DIAGNOSTIC_ONLY",
        ],
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "tail_contribution_csv": str(output_dir / "tail_contribution.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "by_condition_csv": str(output_dir / "by_condition.csv"),
        "worst_rows_csv": str(output_dir / "worst_rows.csv"),
        "best_rows_csv": str(output_dir / "best_rows.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    tail_contribution: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# M20 Rank-Gate Tail/Condition Concentration Analysis",
        "",
        f"- Policy: `{LOCKED_POLICY}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Honesty flags: `{', '.join(HONESTY_FLAGS)}`",
        "",
        "This is a research-only tail diagnostic, not PnL and not a backtest.",
        "",
        "## Tail Contribution",
        "",
    ]
    for row in tail_contribution:
        lines.append(
            f"- `{row['run_label']}`: net `{row['net_value_proxy']}`, "
            f"worst-5 `{row['worst_5_contribution']}`, "
            f"best-5 `{row['best_5_contribution']}`"
        )
    lines.append("")
    return "\n".join(lines)


def _tail_sum(rows: Sequence[Mapping[str, str]], limit: int, *, reverse: bool) -> float:
    ranked = sorted(
        rows,
        key=lambda row: _float(row.get("net_return_proxy")),
        reverse=reverse,
    )
    return sum(_float(row.get("net_return_proxy")) for row in ranked[:limit])


def _tail_ratio(rows: Sequence[Mapping[str, str]], limit: int) -> float:
    losses = [
        abs(_float(row.get("net_return_proxy")))
        for row in rows
        if _float(row.get("net_return_proxy")) < 0.0
    ]
    if not losses:
        return 0.0
    return sum(sorted(losses, reverse=True)[:limit]) / sum(losses)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def _float(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 0.0
    return converted if math.isfinite(converted) else 0.0


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else 0.0


def _median(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return median(finite) if finite else 0.0


def _quantile(values: Sequence[float], quantile: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return 0.0
    index = min(len(finite) - 1, max(0, int((len(finite) - 1) * quantile)))
    return finite[index]
