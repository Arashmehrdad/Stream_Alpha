"""Research-only M20 rank-gate economics diagnostics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from app.training.m20_rank_gated_selector import (
    BASELINE_NAME,
    DISABLE_GAPS,
    SCENARIO_NAME,
    _condition_weights as load_condition_weights,  # pylint: disable=protected-access
    _key as row_key,  # pylint: disable=protected-access
    _quarter as month_to_quarter,  # pylint: disable=protected-access
    _read_csv_rows as read_csv_rows,  # pylint: disable=protected-access
    _to_float as to_float,  # pylint: disable=protected-access
)
from app.training.m20_strategy_selector_simulation import (
    _conditions as selector_conditions,  # pylint: disable=protected-access
)
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "rank_gate_economics"
LOCKED_POLICY = "CONDITION_THEN_TOP_0.25"
GLOBAL_KS = (0.25, 1.0, 5.0)
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "NOT_BACKTEST",
    "NOT_PROFIT_EVIDENCE",
    "NO_RUNTIME",
    "NO_REGISTRY",
    "NO_PROMOTION",
    "SPARSE_COVERAGE",
    "FORWARD_RETURN_PROXY_LIMITED",
)


def simulate_m20_rank_gate_economics(*, base_run_dir: Path) -> dict[str, Any]:
    """Simulate research-only opportunity-filter economics diagnostics."""
    # pylint: disable=too-many-locals
    base_dir = Path(base_run_dir).resolve()
    output_dir = base_dir / "research_labels" / "vol_scaled" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = _load_packet(base_dir)
    condition_weights = load_condition_weights(base_dir)
    runs = _runs_from_packet(packet, base_dir)

    policy_metrics: list[dict[str, Any]] = []
    for run in runs:
        rows = _joined_rows(run["run_dir"], condition_weights)
        policy_metrics.extend(_run_policy_metrics(run, rows))

    by_run = _summarize(policy_metrics, "run_label")
    by_symbol = _slice_rows(policy_metrics, "symbol")
    by_time = _slice_rows(policy_metrics, "month") + _slice_rows(policy_metrics, "quarter")
    comparison = _comparison(policy_metrics)
    recommendation = _recommendation(comparison)
    output_files = _output_files(output_dir)
    manifest = {
        "base_run_dir": str(base_dir),
        "output_dir": str(output_dir),
        "policy": LOCKED_POLICY,
        "evidence_status": packet.get("evidence_status", "UNKNOWN"),
        "runtime_status": "NOT_RUNTIME_READY",
        "promotion_status": "NOT_PROMOTABLE",
        "honesty_flags": list(HONESTY_FLAGS),
        "source_artifacts": packet.get("source_artifacts", []),
        "output_files": output_files,
    }
    report = {
        "policy": LOCKED_POLICY,
        "policy_count": len({row["policy_name"] for row in policy_metrics}),
        "run_count": len(runs),
        "recommendation": recommendation["recommendation"],
        "blockers": recommendation["blockers"],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(_markdown(report, comparison), encoding="utf-8")
    write_csv_artifact(Path(output_files["policy_metrics_csv"]), policy_metrics)
    write_csv_artifact(Path(output_files["by_run_csv"]), by_run)
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_csv_artifact(Path(output_files["comparison_csv"]), comparison)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return {
        **report,
        "output_dir": str(output_dir),
        "policy_metrics": policy_metrics,
        "comparison": comparison,
    }


def _run_policy_metrics(
    run: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    specs = [
        ("NO_GATE", _all_mask(rows)),
        ("CONDITION_THEN_TOP_0.25", _condition_then_top_mask(rows, 0.25)),
    ]
    for top_k in GLOBAL_KS:
        specs.append((f"GLOBAL_TOP_{_k_name(top_k)}", _top_mask(rows, top_k)))
    return [
        _metric_row(str(run["window_label"]), policy_name, rows, mask)
        for policy_name, mask in specs
    ]


def _metric_row(
    run_label: str,
    policy_name: str,
    rows: Sequence[Mapping[str, Any]],
    mask: Sequence[bool],
) -> dict[str, Any]:
    selected = [row for row, keep in zip(rows, mask) if keep]
    total = len(rows)
    positives = sum(int(row["label"]) for row in rows)
    selected_pos = sum(int(row["label"]) for row in selected)
    base_rate = positives / total if total else 0.0
    precision = selected_pos / len(selected) if selected else 0.0
    returns = [row["future_return"] for row in selected if math.isfinite(row["future_return"])]
    net_returns = [
        row["net_return_proxy"]
        for row in selected
        if math.isfinite(row["net_return_proxy"])
    ]
    net_curve = _cumulative(net_returns)
    return {
        "run_label": run_label,
        "policy_name": policy_name,
        "total_rows": total,
        "selected_rows": len(selected),
        "coverage": len(selected) / total if total else 0.0,
        "positive_rate": precision,
        "lift": precision / base_rate if base_rate else 0.0,
        "recall": selected_pos / positives if positives else 0.0,
        "false_positive_count": len(selected) - selected_pos,
        "false_positive_cost_proxy": sum(
            row["cost_per_trade"] for row in selected if not int(row["label"])
        ),
        "true_positive_value_proxy": sum(
            row["net_return_proxy"] for row in selected if int(row["label"])
        ),
        "net_value_proxy": sum(net_returns) if net_returns else 0.0,
        "avg_forward_return": _mean(returns),
        "median_forward_return": _median(returns),
        "avg_net_return_proxy": _mean(net_returns),
        "median_net_return_proxy": _median(net_returns),
        "drawdown_proxy": _max_drawdown(net_curve),
        "avg_prob": _mean([row["probability"] for row in selected]),
        "symbol_mix": _mix(selected, "symbol"),
        "month_mix": _mix(selected, "month"),
        "quarter_mix": _mix(selected, "quarter"),
        "disable_gap_exposure": sum(bool(row["disable_gap_match"]) for row in selected),
    }


def _joined_rows(
    run_dir: Path,
    condition_weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    predictions = read_csv_rows(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_baselines"
        / f"predictions_{BASELINE_NAME}_test_full.csv"
    )
    labels = {
        row_key(row): row
        for row in read_csv_rows(
            run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_labels_vol_scaled.csv"
        )
        if row.get("scenario_name") == SCENARIO_NAME
    }
    features = {
        row_key(row): row
        for row in read_csv_rows(
            run_dir / "training_frame" / "m20_training_frame_features.csv"
        )
    }
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        label = labels.get(row_key(prediction))
        feature = features.get(row_key(prediction), {})
        if not label:
            continue
        future_return = to_float(label.get("future_return"))
        cost = to_float(label.get("cost_per_trade"))
        month = str(prediction.get("interval_begin", ""))[:7]
        row = {
            **feature,
            **prediction,
            "label": int(float(label.get("label", 0))),
            "probability": to_float(prediction.get("probability")),
            "future_return": future_return,
            "cost_per_trade": cost,
            "net_return_proxy": future_return - cost,
            "month": month,
            "quarter": month_to_quarter(month),
        }
        row["conditions"] = selector_conditions(row)
        row["confirmed_conditions"] = [
            condition
            for condition in row["conditions"]
            if condition_weights.get(condition, 0.0) > 0.0
        ]
        row["disable_gap_match"] = sorted(set(row["conditions"]).intersection(DISABLE_GAPS))
        rows.append(row)
    return sorted(rows, key=lambda item: str(item.get("interval_begin", "")))


def _condition_then_top_mask(rows: Sequence[Mapping[str, Any]], top_k: float) -> list[bool]:
    eligible = [
        (index, row)
        for index, row in enumerate(rows)
        if row["confirmed_conditions"] and not row["disable_gap_match"]
    ]
    return _select_ranked(rows, eligible, top_k)


def _top_mask(rows: Sequence[Mapping[str, Any]], top_k: float) -> list[bool]:
    return _select_ranked(rows, list(enumerate(rows)), top_k)


def _all_mask(rows: Sequence[Mapping[str, Any]]) -> list[bool]:
    return [True for _row in rows]


def _select_ranked(
    rows: Sequence[Mapping[str, Any]],
    eligible: Sequence[tuple[int, Mapping[str, Any]]],
    top_k: float,
) -> list[bool]:
    selected_count = max(1, int(len(rows) * top_k / 100.0))
    ranked = sorted(eligible, key=lambda item: item[1]["probability"], reverse=True)
    selected = {index for index, _row in ranked[:selected_count]}
    return [index in selected for index in range(len(rows))]


def _runs_from_packet(packet: Mapping[str, Any], _base_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for metric in packet.get("window_metrics", []):
        run_dir = Path(str(metric.get("run_dir", "")))
        if not run_dir.is_absolute():
            run_dir = run_dir.resolve()
        runs.append({**metric, "run_dir": run_dir})
    return runs


def _load_packet(base_dir: Path) -> dict[str, Any]:
    packet_path = (
        base_dir
        / "research_labels"
        / "vol_scaled"
        / "rank_gate_evidence_packet"
        / "rank_gate_evidence_packet.json"
    )
    if not packet_path.exists():
        return {
            "evidence_status": "PACKET_MISSING",
            "window_metrics": [
                {"window_label": "original_locked_test", "run_dir": str(base_dir)}
            ],
        }
    return json.loads(packet_path.read_text(encoding="utf-8"))


def _summarize(rows: Sequence[Mapping[str, Any]], column: str) -> list[dict[str, Any]]:
    groups = sorted({str(row[column]) for row in rows})
    return [
        {
            "group": group,
            "policy_count": sum(str(row[column]) == group for row in rows),
            "best_lift": max(
                (row["lift"] for row in rows if str(row[column]) == group),
                default=0.0,
            ),
        }
        for group in groups
    ]


def _slice_rows(rows: Sequence[Mapping[str, Any]], mix_column: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        for part in str(row.get(f"{mix_column}_mix", "")).split(";"):
            if not part:
                continue
            value, count = part.rsplit(":", 1)
            output.append(
                {
                    "run_label": row["run_label"],
                    "policy_name": row["policy_name"],
                    "slice_column": mix_column,
                    "slice_value": value,
                    "selected_rows": int(count),
                }
            )
    return output


def _comparison(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    policies = sorted({row["policy_name"] for row in rows})
    output = []
    for policy in policies:
        policy_rows = [row for row in rows if row["policy_name"] == policy]
        output.append(
            {
                "policy_name": policy,
                "window_count": len(policy_rows),
                "min_lift": min((row["lift"] for row in policy_rows), default=0.0),
                "max_lift": max((row["lift"] for row in policy_rows), default=0.0),
                "avg_net_value_proxy": _mean([row["net_value_proxy"] for row in policy_rows]),
                "total_selected_rows": sum(row["selected_rows"] for row in policy_rows),
            }
        )
    return output


def _recommendation(comparison: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    locked = [
        row for row in comparison if row.get("policy_name") == LOCKED_POLICY
    ]
    min_lift = locked[0]["min_lift"] if locked else 0.0
    if min_lift > 1.5:
        recommendation = "KEEP_RESEARCH_ONLY_RANK_GATE_ECONOMICS_CANDIDATE"
    else:
        recommendation = "DO_NOT_ADVANCE_RANK_GATE_ECONOMICS"
    return {
        "recommendation": recommendation,
        "locked_policy_min_lift": min_lift,
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": [
            "NOT_BACKTEST",
            "NOT_PROFIT_EVIDENCE",
            "SPARSE_COVERAGE",
            "FORWARD_RETURN_PROXY_LIMITED",
            "NOT_RUNTIME_READY",
        ],
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "policy_metrics_csv": str(output_dir / "policy_metrics.csv"),
        "by_run_csv": str(output_dir / "by_run.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "comparison_csv": str(output_dir / "comparison.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], comparison: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# M20 Rank-Gate Economics Diagnostics",
        "",
        f"- Policy: `{LOCKED_POLICY}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Honesty flags: `{', '.join(HONESTY_FLAGS)}`",
        "",
        "This is research-only opportunity-filter economics diagnostics, not a backtest.",
        "",
        "## Policy Comparison",
        "",
    ]
    for row in comparison:
        lines.append(
            f"- `{row['policy_name']}`: min lift `{row['min_lift']}`, "
            f"avg net value proxy `{row['avg_net_value_proxy']}`"
        )
    lines.append("")
    return "\n".join(lines)


def _k_name(value: float) -> str:
    return str(value).rstrip("0").rstrip(".")


def _mix(rows: Sequence[Mapping[str, Any]], column: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(column, ""))
        counts[value] = counts.get(value, 0) + 1
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _cumulative(values: Sequence[float]) -> list[float]:
    total = 0.0
    output = []
    for value in values:
        total += value
        output.append(total)
    return output


def _max_drawdown(values: Sequence[float]) -> float:
    peak = 0.0
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = min(drawdown, value - peak)
    return drawdown


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else 0.0


def _median(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return median(finite) if finite else 0.0


def read_metric_csv(path: Path) -> list[dict[str, str]]:
    """Read metric CSV rows for tests."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
