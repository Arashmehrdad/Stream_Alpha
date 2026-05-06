"""Research-only M20 rank-gate tail-risk filter simulation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_rank_gate_economics import LOCKED_POLICY
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "rank_gate_tail_filter"
SOURCE_DIR_NAME = "rank_gate_net_diagnostics"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "FILTER_SIM_ONLY",
    "NOT_BACKTEST",
    "NOT_PNL",
    "NO_RUNTIME",
    "NO_REGISTRY",
    "NO_PROMOTION",
    "NO_PROFIT_CLAIM",
    "SPARSE_SELECTION",
    "TAIL_RISK_FILTER_TEST",
)


def simulate_m20_rank_gate_tail_filter(*, base_run_dir: Path) -> dict[str, Any]:
    """Run exploratory tail-risk filters on locked rank-gate selected rows."""
    # pylint: disable=too-many-locals
    base_dir = Path(base_run_dir).resolve()
    source_path = (
        base_dir
        / "research_labels"
        / "vol_scaled"
        / SOURCE_DIR_NAME
        / "selected_row_diagnostics.csv"
    )
    output_dir = base_dir / "research_labels" / "vol_scaled" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = _output_files(output_dir)
    if not source_path.exists():
        return _write_blocker(base_dir, output_files, source_path)

    rows = _read_csv_rows(source_path)
    filters = _filter_specs(rows)
    metrics = [
        _metric_row(run_label, filter_name, run_rows, keep)
        for run_label, run_rows in _by_run(rows).items()
        for filter_name, keep in filters
    ]
    stability = _stability(metrics)
    tail_comparison = _tail_comparison(metrics)
    recommendation = _recommendation(stability)
    report = {
        "policy": LOCKED_POLICY,
        "selected_input_rows": len(rows),
        "filter_count": len(filters),
        "recommendation": recommendation["recommendation"],
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": recommendation["blockers"],
        "output_files": output_files,
    }
    manifest = {
        "base_run_dir": str(base_dir),
        "source_selected_rows": str(source_path),
        "output_dir": str(output_dir),
        "policy": LOCKED_POLICY,
        "exploratory_test_sweep": True,
        "runtime_status": "NOT_RUNTIME_READY",
        "promotion_status": "NOT_PROMOTABLE",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _markdown(report, stability),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["filter_metrics_csv"]), metrics)
    write_csv_artifact(Path(output_files["stability_csv"]), stability)
    write_csv_artifact(Path(output_files["tail_comparison_csv"]), tail_comparison)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe({**report, "stability": stability})


def _write_blocker(
    base_dir: Path,
    output_files: Mapping[str, str],
    source_path: Path,
) -> dict[str, Any]:
    recommendation = {
        "recommendation": "BLOCKED_ROW_LEVEL_NET_DIAGNOSTICS_MISSING",
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": ["ROW_LEVEL_ECONOMICS_MISSING", "RUN_NET_DIAGNOSTICS_FIRST"],
    }
    report = {
        "policy": LOCKED_POLICY,
        "base_run_dir": str(base_dir),
        "missing_input": str(source_path),
        "recommendation": recommendation["recommendation"],
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": recommendation["blockers"],
        "output_files": dict(output_files),
    }
    write_json_artifact(Path(output_files["manifest_json"]), report)
    write_json_artifact(Path(output_files["report_json"]), report)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    Path(output_files["report_md"]).write_text(
        "# M20 Rank-Gate Tail Filter\n\nBlocked: row-level selected rows are missing.\n",
        encoding="utf-8",
    )
    return make_json_safe(report)


def _filter_specs(
    rows: Sequence[Mapping[str, str]],
) -> list[tuple[str, Callable[[Mapping[str, str]], bool]]]:
    negative_buckets = _negative_buckets(rows)
    unstable_buckets = _unstable_buckets(rows)
    p50 = _quantile([_float(row.get("probability")) for row in rows], 0.50)
    p75 = _quantile([_float(row.get("probability")) for row in rows], 0.75)
    p90 = _quantile([_float(row.get("probability")) for row in rows], 0.90)
    return [
        ("BASE_LOCKED_SELECTION", lambda _row: True),
        ("EXCLUDE_HIGH_RANGE", lambda row: row.get("range_bucket") != "high"),
        ("EXCLUDE_HIGH_VOL", lambda row: row.get("volatility_bucket") != "high"),
        (
            "EXCLUDE_UNSTABLE_CONCENTRATION_SLICES",
            lambda row: not _matches_buckets(row, unstable_buckets),
        ),
        (
            "EXCLUDE_NEGATIVE_SYMBOL_TIME_BUCKETS",
            lambda row: not _matches_buckets(row, negative_buckets),
        ),
        (f"MIN_PROB_P50_{p50:.6f}", lambda row: _float(row.get("probability")) >= p50),
        (f"MIN_PROB_P75_{p75:.6f}", lambda row: _float(row.get("probability")) >= p75),
        (f"MIN_PROB_P90_{p90:.6f}", lambda row: _float(row.get("probability")) >= p90),
        (
            "COMBO_EXCLUDE_NEGATIVE_AND_MIN_PROB_P75",
            lambda row: (
                not _matches_buckets(row, negative_buckets)
                and _float(row.get("probability")) >= p75
            ),
        ),
        (
            "COMBO_EXCLUDE_UNSTABLE_AND_MIN_PROB_P75",
            lambda row: (
                not _matches_buckets(row, unstable_buckets)
                and _float(row.get("probability")) >= p75
            ),
        ),
    ]


def _metric_row(
    run_label: str,
    filter_name: str,
    run_rows: Sequence[Mapping[str, str]],
    keep: Callable[[Mapping[str, str]], bool],
) -> dict[str, Any]:
    selected = [row for row in run_rows if keep(row)]
    positives = sum(_int(row.get("label")) for row in run_rows)
    selected_positives = sum(_int(row.get("label")) for row in selected)
    base_precision = positives / len(run_rows) if run_rows else 0.0
    precision = selected_positives / len(selected) if selected else 0.0
    net_values = [_float(row.get("net_return_proxy")) for row in selected]
    return {
        "run_label": run_label,
        "filter_name": filter_name,
        "input_selected_rows": len(run_rows),
        "selected_rows": len(selected),
        "coverage": len(selected) / len(run_rows) if run_rows else 0.0,
        "precision": precision,
        "lift": precision / base_precision if base_precision else 0.0,
        "recall": selected_positives / positives if positives else 0.0,
        "true_positive_count": selected_positives,
        "false_positive_count": len(selected) - selected_positives,
        "net_proxy": sum(net_values),
        "worst5": _tail_sum(selected, 5, reverse=False),
        "best5": _tail_sum(selected, 5, reverse=True),
        "tail_ratio": _tail_ratio(selected, 5),
        "avg_prob": _mean([_float(row.get("probability")) for row in selected]),
        "disable_gap_exposure": sum(_disable_gap(row) for row in selected),
        "symbol_mix": _mix(selected, "symbol"),
        "month_mix": _mix(selected, "month"),
        "quarter_mix": _mix(selected, "quarter"),
    }


def _stability(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for filter_name in sorted({row["filter_name"] for row in metrics}):
        rows = [row for row in metrics if row["filter_name"] == filter_name]
        net_values = [_float(row.get("net_proxy")) for row in rows]
        output.append(
            {
                "filter_name": filter_name,
                "run_count": len(rows),
                "min_selected_rows": min((int(row["selected_rows"]) for row in rows), default=0),
                "min_precision": min((_float(row.get("precision")) for row in rows), default=0.0),
                "min_lift": min((_float(row.get("lift")) for row in rows), default=0.0),
                "min_net_proxy": min(net_values, default=0.0),
                "max_net_proxy": max(net_values, default=0.0),
                "positive_net_run_count": sum(value > 0.0 for value in net_values),
                "negative_net_run_count": sum(value < 0.0 for value in net_values),
                "max_tail_ratio": max((_float(row.get("tail_ratio")) for row in rows), default=0.0),
                "total_disable_gap_exposure": sum(int(row["disable_gap_exposure"]) for row in rows),
            }
        )
    return output


def _tail_comparison(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["run_label"], row["filter_name"]): row for row in metrics}
    output = []
    for row in metrics:
        base = by_key.get((row["run_label"], "BASE_LOCKED_SELECTION"), {})
        output.append(
            {
                "run_label": row["run_label"],
                "filter_name": row["filter_name"],
                "net_delta_vs_base": _float(row.get("net_proxy")) - _float(base.get("net_proxy")),
                "worst5_delta_vs_base": _float(row.get("worst5")) - _float(base.get("worst5")),
                "coverage_delta_vs_base": (
                    _float(row.get("coverage")) - _float(base.get("coverage"))
                ),
                "precision_delta_vs_base": (
                    _float(row.get("precision")) - _float(base.get("precision"))
                ),
            }
        )
    return output


def _recommendation(stability: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [
        row for row in stability
        if row["filter_name"] != "BASE_LOCKED_SELECTION"
        and _float(row.get("min_net_proxy")) > -0.05
        and int(row.get("min_selected_rows", 0)) >= 5
    ]
    best = max(candidates, key=lambda row: _float(row.get("min_net_proxy")), default={})
    recommendation = (
        "KEEP_BEST_FILTER_AS_RESEARCH_ONLY_FOLLOW_UP_CANDIDATE"
        if best else "NO_STABLE_TAIL_FILTER_FOUND"
    )
    return {
        "recommendation": recommendation,
        "best_research_filter": best.get("filter_name", ""),
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": [
            "EXPLORATORY_TEST_SWEEP_NOT_FINAL_OPTIMIZATION",
            "NOT_BACKTEST",
            "NOT_PNL",
            "NO_PROFIT_CLAIM",
            "SPARSE_SELECTION",
        ],
    }


def _negative_buckets(rows: Sequence[Mapping[str, str]]) -> set[tuple[str, str]]:
    buckets: set[tuple[str, str]] = set()
    for column in ("symbol", "month", "quarter"):
        for run_rows in _by_run(rows).values():
            for value in sorted({row.get(column, "") for row in run_rows}):
                group = [
                    row for row in run_rows
                    if row.get(column, "") == value
                ]
                if sum(_float(row.get("net_return_proxy")) for row in group) < 0.0:
                    buckets.add((column, value))
    return buckets


def _unstable_buckets(rows: Sequence[Mapping[str, str]]) -> set[tuple[str, str]]:
    buckets: set[tuple[str, str]] = set()
    for column in (
        "symbol",
        "month",
        "quarter",
        "range_bucket",
        "volatility_bucket",
        "volume_bucket",
        "momentum_bucket",
        "macd_bucket",
    ):
        for run_rows in _by_run(rows).values():
            for value in sorted({row.get(column, "") for row in run_rows}):
                group = [row for row in run_rows if row.get(column, "") == value]
                if len(group) >= 5 and _tail_ratio(group, 5) > 0.5:
                    buckets.add((column, value))
    return buckets


def _matches_buckets(row: Mapping[str, str], buckets: set[tuple[str, str]]) -> bool:
    return any(row.get(column, "") == value for column, value in buckets)


def _disable_gap(row: Mapping[str, str]) -> int:
    return int(
        row.get("month") == "2026-04"
        or row.get("quarter") == "2026Q2"
    )


def _by_run(rows: Sequence[Mapping[str, str]]) -> dict[str, list[Mapping[str, str]]]:
    output: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        output.setdefault(row.get("run_label", ""), []).append(row)
    return dict(sorted(output.items()))


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "filter_metrics_csv": str(output_dir / "filter_metrics.csv"),
        "stability_csv": str(output_dir / "stability.csv"),
        "tail_comparison_csv": str(output_dir / "tail_comparison.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], stability: Sequence[Mapping[str, Any]]) -> str:
    best = max(stability, key=lambda row: _float(row.get("min_net_proxy")), default={})
    return "\n".join(
        [
            "# M20 Rank-Gate Tail-Risk Filter Simulation",
            "",
            f"- Policy: `{LOCKED_POLICY}`",
            f"- Recommendation: `{report['recommendation']}`",
            f"- Best exploratory filter: `{best.get('filter_name', '')}`",
            f"- Honesty flags: `{', '.join(HONESTY_FLAGS)}`",
            "",
            "This is an exploratory test-split filter sweep, not final optimization.",
            "",
        ]
    )


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
    return sum(sorted(losses, reverse=True)[:limit]) / sum(losses) if losses else 0.0


def _mix(rows: Sequence[Mapping[str, str]], column: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(column, "")
        counts[value] = counts.get(value, 0) + 1
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))


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


def _quantile(values: Sequence[float], quantile: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return 0.0
    index = min(len(finite) - 1, max(0, int((len(finite) - 1) * quantile)))
    return finite[index]


def _median(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return median(finite) if finite else 0.0
