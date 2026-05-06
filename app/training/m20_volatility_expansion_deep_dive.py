"""Research-only deep diagnostics for M20 volatility-expansion setups."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_volatility_expansion_research import (
    _joined_rows as _vol_joined_rows,  # pylint: disable=protected-access
)
from app.training.m20_volatility_expansion_research import (
    _rank_gate_keys as _vol_rank_gate_keys,  # pylint: disable=protected-access
)
from app.training.m20_volatility_expansion_research import (
    _rank_gate_key as _vol_rank_gate_key,  # pylint: disable=protected-access
)
from app.training.m20_volatility_expansion_research import (
    _runs as _vol_runs,  # pylint: disable=protected-access
)
from app.training.m20_volatility_expansion_research import (
    _setup_specs as _vol_setup_specs,  # pylint: disable=protected-access
)
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "volatility_expansion_deep_dive"
VOL_DIR_NAME = "volatility_expansion_research"
TOP_SETUPS = (
    "vol_plus_range_high",
    "vol_plus_volume_high",
    "range_plus_volume_high",
    "realized_vol_high",
    "range_high",
    "volume_high",
)
HONESTY_FLAGS = (
    "RESEARCH_ONLY_VOLATILITY_EXPANSION_DEEP_DIVE",
    "EXISTING_ARTIFACTS_ONLY",
    "DIAGNOSTIC_ONLY",
    "NOT_BACKTEST",
    "NOT_PNL",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_RUNTIME_EFFECT",
    "NO_REGISTRY_WRITE",
    "NO_PROMOTION_EFFECT",
    "NO_PROFIT_CLAIM",
)


def analyze_m20_volatility_expansion_deep_dive(*, base_run_dir: Path) -> dict[str, Any]:
    """Run a research-only deep dive over top volatility-expansion setups."""
    # pylint: disable=too-many-locals
    base_dir = Path(base_run_dir).resolve()
    vol_dir = base_dir / "research_labels" / "vol_scaled"
    output_dir = vol_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = _output_files(output_dir)

    runs = _vol_runs(base_dir)
    rank_gate_keys = _vol_rank_gate_keys(base_dir)
    setup_metrics: list[dict[str, Any]] = []
    rank_gate_overlap: list[dict[str, Any]] = []
    by_run_inputs: list[dict[str, Any]] = []
    by_symbol = []
    by_time = []
    intersections = []
    for run_label, run_dir in runs:
        rows = _vol_joined_rows(run_dir)
        setup_predicates = _setup_predicates(rows)
        by_run_inputs.extend(_run_rows(run_label, rows))
        setup_metrics.extend(
            _setup_deep_metrics(run_label, rows, setup_predicates, rank_gate_keys)
        )
        rank_gate_overlap.extend(
            _rank_gate_overlap(run_label, rows, setup_predicates, rank_gate_keys)
        )
        by_symbol.extend(_slice_rows(run_label, rows, setup_predicates, "symbol"))
        by_time.extend(_slice_rows(run_label, rows, setup_predicates, "month"))
        by_time.extend(_slice_rows(run_label, rows, setup_predicates, "quarter"))
        intersections.extend(_intersection_rows(run_label, rows, setup_predicates))
    recommendation = _recommendation(setup_metrics)
    report = {
        "base_run_dir": str(base_dir),
        "run_count": len(runs),
        "setup_count": len({row["setup_name"] for row in setup_metrics}),
        "recommendation": recommendation["recommendation"],
        "primary_setup": recommendation["primary_setup"],
        "secondary_setups": recommendation["secondary_setups"],
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": recommendation["blockers"],
        "output_files": output_files,
    }
    manifest = {
        "base_run_dir": str(base_dir),
        "output_dir": str(output_dir),
        "source": "existing_volatility_expansion_research_artifacts",
        "top_setups": list(TOP_SETUPS),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _markdown(report, setup_metrics),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["setup_deep_metrics_csv"]), setup_metrics)
    write_csv_artifact(Path(output_files["by_run_csv"]), _by_run(by_run_inputs, setup_metrics))
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_csv_artifact(Path(output_files["rank_gate_overlap_deep_csv"]), rank_gate_overlap)
    write_csv_artifact(Path(output_files["condition_intersection_matrix_csv"]), intersections)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(report)


def _setup_predicates(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Callable[[Mapping[str, Any]], bool]]:
    specs = dict(_vol_setup_specs(rows))
    return {setup: predicate for setup, predicate in specs.items() if setup in TOP_SETUPS}


def _setup_deep_metrics(
    run_label: str,
    rows: Sequence[Mapping[str, Any]],
    predicates: Mapping[str, Callable[[Mapping[str, Any]], bool]],
    rank_gate_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    output = []
    positives = sum(_int(row.get("label")) for row in rows)
    base_rate = positives / len(rows) if rows else 0.0
    for setup_name in TOP_SETUPS:
        predicate = predicates.get(setup_name)
        selected = [row for row in rows if predicate and predicate(row)]
        selected_positive = sum(_int(row.get("label")) for row in selected)
        selected_negative = len(selected) - selected_positive
        positive_rate = selected_positive / len(selected) if selected else 0.0
        rank_overlap = [
            row for row in selected if _vol_rank_gate_key(run_label, row) in rank_gate_keys
        ]
        output.append(
            {
                "run_label": run_label,
                "setup_name": setup_name,
                "total_rows": len(rows),
                "setup_rows": len(selected),
                "coverage": len(selected) / len(rows) if rows else 0.0,
                "positive_count": selected_positive,
                "false_positive_count": selected_negative,
                "base_positive_rate": base_rate,
                "positive_rate": positive_rate,
                "lift_vs_base": positive_rate / base_rate if base_rate else 0.0,
                "recall": selected_positive / positives if positives else 0.0,
                "net_proxy_sum": sum(_float(row.get("net_proxy")) for row in selected),
                "net_proxy_mean": (
                    sum(_float(row.get("net_proxy")) for row in selected) / len(selected)
                    if selected else 0.0
                ),
                "rank_gate_overlap_rows": len(rank_overlap),
                "rank_gate_overlap_rate": len(rank_overlap) / len(selected) if selected else 0.0,
                "classification": _classify_setup(
                    positive_rate / base_rate if base_rate else 0.0,
                    len(rank_overlap) / len(selected) if selected else 0.0,
                    len(selected) / len(rows) if rows else 0.0,
                ),
            }
        )
    return output


def _rank_gate_overlap(
    run_label: str,
    rows: Sequence[Mapping[str, Any]],
    predicates: Mapping[str, Callable[[Mapping[str, Any]], bool]],
    rank_gate_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    output = []
    for setup_name in TOP_SETUPS:
        predicate = predicates.get(setup_name)
        selected = [row for row in rows if predicate and predicate(row)]
        overlap = [row for row in selected if _vol_rank_gate_key(run_label, row) in rank_gate_keys]
        output.append(
            {
                "run_label": run_label,
                "setup_name": setup_name,
                "setup_rows": len(selected),
                "rank_gate_overlap_rows": len(overlap),
                "rank_gate_overlap_rate": len(overlap) / len(selected) if selected else 0.0,
                "overlap_positive_rate": (
                    sum(_int(row.get("label")) for row in overlap) / len(overlap)
                    if overlap else 0.0
                ),
            }
        )
    return output


def _slice_rows(
    run_label: str,
    rows: Sequence[Mapping[str, Any]],
    predicates: Mapping[str, Callable[[Mapping[str, Any]], bool]],
    column: str,
) -> list[dict[str, Any]]:
    output = []
    for setup_name in TOP_SETUPS:
        predicate = predicates.get(setup_name)
        selected = [row for row in rows if predicate and predicate(row)]
        for value in sorted({str(row.get(column, "")) for row in selected}):
            group = [row for row in selected if str(row.get(column, "")) == value]
            positives = sum(_int(row.get("label")) for row in group)
            output.append(
                {
                    "run_label": run_label,
                    "setup_name": setup_name,
                    "slice_family": column,
                    "slice_value": value,
                    "rows": len(group),
                    "positive_rate": positives / len(group) if group else 0.0,
                }
            )
    return output


def _intersection_rows(
    run_label: str,
    rows: Sequence[Mapping[str, Any]],
    predicates: Mapping[str, Callable[[Mapping[str, Any]], bool]],
) -> list[dict[str, Any]]:
    output = []
    selected_by_setup = {
        setup: {
            _row_key(row)
            for row in rows
            if predicates.get(setup) and predicates[setup](row)
        }
        for setup in TOP_SETUPS
    }
    for setup_a in TOP_SETUPS:
        for setup_b in TOP_SETUPS:
            keys_a = selected_by_setup.get(setup_a, set())
            keys_b = selected_by_setup.get(setup_b, set())
            intersection = keys_a & keys_b
            union = keys_a | keys_b
            output.append(
                {
                    "run_label": run_label,
                    "setup_a": setup_a,
                    "setup_b": setup_b,
                    "setup_a_rows": len(keys_a),
                    "setup_b_rows": len(keys_b),
                    "intersection_rows": len(intersection),
                    "intersection_over_a": len(intersection) / len(keys_a) if keys_a else 0.0,
                    "jaccard": len(intersection) / len(union) if union else 0.0,
                }
            )
    return output


def _run_rows(run_label: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    positives = sum(_int(row.get("label")) for row in rows)
    return [
        {
            "run_label": run_label,
            "total_rows": len(rows),
            "positive_count": positives,
            "base_positive_rate": positives / len(rows) if rows else 0.0,
        }
    ]


def _by_run(
    run_inputs: Sequence[Mapping[str, Any]],
    setup_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in run_inputs:
        run_label = str(row["run_label"])
        setup_rows = [metric for metric in setup_metrics if metric["run_label"] == run_label]
        best = max(setup_rows, key=lambda metric: _float(metric["lift_vs_base"]), default={})
        output.append(
            {
                **dict(row),
                "best_setup": best.get("setup_name", ""),
                "best_lift": best.get("lift_vs_base", 0.0),
                "best_coverage": best.get("coverage", 0.0),
            }
        )
    return output


def _recommendation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = {}
    for setup_name in TOP_SETUPS:
        setup_rows = [row for row in rows if row["setup_name"] == setup_name]
        if not setup_rows:
            continue
        grouped[setup_name] = {
            "min_lift": min(_float(row.get("lift_vs_base")) for row in setup_rows),
            "avg_lift": sum(_float(row.get("lift_vs_base")) for row in setup_rows)
            / len(setup_rows),
            "avg_coverage": sum(_float(row.get("coverage")) for row in setup_rows)
            / len(setup_rows),
        }
    ranked = sorted(grouped.items(), key=lambda item: -item[1]["min_lift"])
    primary_setup = ranked[0][0] if ranked else ""
    secondary = [setup for setup, data in ranked[1:] if data["min_lift"] >= 1.45]
    recommendation = (
        "TEST_VOLATILITY_EXPANSION_COMBO_NEXT"
        if primary_setup and grouped[primary_setup]["min_lift"] >= 1.50
        else "KEEP_VOLATILITY_EXPANSION_AS_STANDALONE_DIAGNOSTIC"
    )
    return {
        "recommendation": recommendation,
        "primary_setup": primary_setup,
        "secondary_setups": secondary,
        "setup_scores": grouped,
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": [
            "SETUP_LIFT_IS_NOT_PNL",
            "NOT_BACKTEST",
            "NO_PROFIT_CLAIM",
            "NOT_RUNTIME_READY",
            "NOT_PROMOTABLE",
        ],
    }


def _classify_setup(lift: float, overlap_rate: float, coverage: float) -> str:
    if lift >= 1.55 and coverage >= 0.05:
        return "PRIMARY_VOL_EXPANSION_CANDIDATE"
    if lift >= 1.45 and coverage >= 0.05:
        return "SECONDARY_VOL_EXPANSION_CANDIDATE"
    if overlap_rate > 0.0 and coverage < 0.05:
        return "OVERLAPS_RANK_GATE_ONLY"
    if lift >= 1.15:
        return "BROAD_LABEL_LIFT_ONLY"
    return "WEAK_OR_UNSTABLE"


def _markdown(report: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]) -> str:
    best = max(metrics, key=lambda row: _float(row.get("lift_vs_base")), default={})
    return "\n".join(
        [
            "# M20 Volatility Expansion Deep Dive",
            "",
            f"- Recommendation: `{report['recommendation']}`",
            f"- Primary setup: `{report['primary_setup']}`",
            (
                f"- Best observed setup/run: `{best.get('setup_name', '')}` / "
                f"`{best.get('run_label', '')}`"
            ),
            f"- Honesty flags: `{', '.join(HONESTY_FLAGS)}`",
            "",
            "This is diagnostic-only setup analysis. It is not a backtest, not PnL,",
            "not a runtime selector, and not profitability evidence.",
            "",
        ]
    )


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "setup_deep_metrics_csv": str(output_dir / "setup_deep_metrics.csv"),
        "by_run_csv": str(output_dir / "by_run.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "rank_gate_overlap_deep_csv": str(output_dir / "rank_gate_overlap_deep.csv"),
        "condition_intersection_matrix_csv": str(
            output_dir / "condition_intersection_matrix.csv"
        ),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("symbol", "")),
        str(row.get("interval_begin", "")),
        str(row.get("fold_index", "")),
    )


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


def read_volatility_deep_dive_csv(path: Path) -> list[dict[str, str]]:
    """Read volatility deep-dive CSV for tests."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
