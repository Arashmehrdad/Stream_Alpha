"""Research-only M20 rank-gate plus momentum-breakout combo diagnostics."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_momentum_breakout_research import (
    _momentum as momentum_value,  # pylint: disable=protected-access
    _quarter as month_to_quarter,  # pylint: disable=protected-access
    _range_pct as range_pct,  # pylint: disable=protected-access
    _tertiles as tertiles,  # pylint: disable=protected-access
)
from app.training.m20_rank_gate_economics import (
    LOCKED_POLICY,
    _condition_then_top_mask as gate_mask,  # pylint: disable=protected-access
    _joined_rows as joined_rank_rows,  # pylint: disable=protected-access
    _load_packet as load_packet,  # pylint: disable=protected-access
    _runs_from_packet as runs_from_packet,  # pylint: disable=protected-access
)
from app.training.m20_rank_gated_selector import (
    _condition_weights as load_condition_weights,  # pylint: disable=protected-access
)
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "gate_momentum_combo"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "COMBO_DIAGNOSTIC_ONLY",
    "NOT_BACKTEST",
    "NOT_PNL",
    "NO_RUNTIME",
    "NO_REGISTRY",
    "NO_PROMOTION",
    "NO_PROFIT_CLAIM",
)
MOMENTUM_SETUPS = ("realized_vol_high", "range_high", "volume_high")


def analyze_m20_gate_momentum_combo(*, base_run_dir: Path) -> dict[str, Any]:
    """Analyze locked rank gate with momentum-breakout setup combinations."""
    # pylint: disable=too-many-locals
    base_dir = Path(base_run_dir).resolve()
    output_dir = base_dir / "research_labels" / "vol_scaled" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = load_packet(base_dir)
    condition_weights = load_condition_weights(base_dir)
    runs = runs_from_packet(packet, base_dir)

    metrics: list[dict[str, Any]] = []
    for run in runs:
        run_label = str(run["window_label"])
        rows = _annotated_rows(Path(run["run_dir"]), condition_weights)
        masks = _policy_masks(rows)
        for policy_name, mask in masks.items():
            metrics.append(_metric_row(run_label, policy_name, rows, mask))
    by_run = _by_group(metrics, "run_label")
    by_symbol = _slice_rows(metrics, "symbol")
    by_time = _slice_rows(metrics, "month") + _slice_rows(metrics, "quarter")
    tail_summary = _tail_summary(metrics)
    recommendation = _recommendation(metrics)
    output_files = _output_files(output_dir)
    report = {
        "policy_root": LOCKED_POLICY,
        "run_count": len(runs),
        "policy_count": len({row["policy_name"] for row in metrics}),
        "recommendation": recommendation["recommendation"],
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": recommendation["blockers"],
        "output_files": output_files,
    }
    manifest = {
        "base_run_dir": str(base_dir),
        "output_dir": str(output_dir),
        "source_policy": LOCKED_POLICY,
        "momentum_setups": list(MOMENTUM_SETUPS),
        "rank_gate_usage": "COMBO_DIAGNOSTIC_ONLY",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(_markdown(report, metrics), encoding="utf-8")
    write_csv_artifact(Path(output_files["policy_metrics_csv"]), metrics)
    write_csv_artifact(Path(output_files["by_run_csv"]), by_run)
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_csv_artifact(Path(output_files["tail_summary_csv"]), tail_summary)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(report)


def _annotated_rows(
    run_dir: Path,
    condition_weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows = joined_rank_rows(run_dir, condition_weights)
    _vol_mid, vol_high = tertiles([_float(row.get("realized_vol_12")) for row in rows])
    _volume_mid, volume_high = tertiles([_float(row.get("volume")) for row in rows])
    _range_mid, range_high = tertiles([range_pct(row) for row in rows])
    for row in rows:
        timestamp = str(row.get("interval_begin", ""))
        month = timestamp[:7]
        row["month"] = month
        row["quarter"] = month_to_quarter(month)
        row["range_pct"] = range_pct(row)
        row["realized_vol_high"] = _float(row.get("realized_vol_12")) >= vol_high
        row["range_high"] = row["range_pct"] >= range_high
        row["volume_high"] = _float(row.get("volume")) >= volume_high
        row["momentum_setup"] = any(bool(row[setup]) for setup in MOMENTUM_SETUPS)
        row["momentum_positive"] = momentum_value(row) > 0.0
        row["momentum_flat"] = abs(momentum_value(row)) <= 0.0005
        row["disable_gap_exposure"] = int(month == "2026-04" or row["quarter"] == "2026Q2")
    return rows


def _policy_masks(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[bool]]:
    gate = gate_mask(rows, 0.25)
    momentum = [bool(row.get("momentum_setup")) for row in rows]
    masks = {
        "MOMENTUM_ONLY_ALL_STABLE_SETUPS": momentum,
        "MOMENTUM_ONLY_REALIZED_VOL_HIGH": [
            bool(row.get("realized_vol_high")) for row in rows
        ],
        "MOMENTUM_ONLY_RANGE_HIGH": [bool(row.get("range_high")) for row in rows],
        "MOMENTUM_ONLY_VOLUME_HIGH": [bool(row.get("volume_high")) for row in rows],
        "GATE_ONLY_CONDITION_THEN_TOP_0.25": gate,
        "GATE_AND_MOMENTUM": [keep and mom for keep, mom in zip(gate, momentum)],
        "GATE_OR_MOMENTUM": [keep or mom for keep, mom in zip(gate, momentum)],
        "GATE_THEN_MOMENTUM_TOPK": _gate_then_momentum_topk(rows, gate, momentum),
    }
    return masks


def _gate_then_momentum_topk(
    rows: Sequence[Mapping[str, Any]],
    gate: Sequence[bool],
    momentum: Sequence[bool],
) -> list[bool]:
    selected_count = max(1, int(len(rows) * 0.25 / 100.0))
    eligible = [
        (index, row)
        for index, (row, gate_keep, momentum_keep) in enumerate(zip(rows, gate, momentum))
        if gate_keep and momentum_keep
    ]
    ranked = sorted(eligible, key=lambda item: _float(item[1].get("probability")), reverse=True)
    selected = {index for index, _row in ranked[:selected_count]}
    return [index in selected for index in range(len(rows))]


def _metric_row(
    run_label: str,
    policy_name: str,
    rows: Sequence[Mapping[str, Any]],
    mask: Sequence[bool],
) -> dict[str, Any]:
    selected = [row for row, keep in zip(rows, mask) if keep]
    positives = sum(_int(row.get("label")) for row in rows)
    selected_pos = sum(_int(row.get("label")) for row in selected)
    base_rate = positives / len(rows) if rows else 0.0
    precision = selected_pos / len(selected) if selected else 0.0
    net_values = [_float(row.get("net_return_proxy")) for row in selected]
    return {
        "run_label": run_label,
        "policy_name": policy_name,
        "total_rows": len(rows),
        "selected_rows": len(selected),
        "coverage": len(selected) / len(rows) if rows else 0.0,
        "precision": precision,
        "lift": precision / base_rate if base_rate else 0.0,
        "recall": selected_pos / positives if positives else 0.0,
        "false_positive_count": len(selected) - selected_pos,
        "true_positive_count": selected_pos,
        "net_proxy": sum(net_values),
        "worst5": _tail_sum(selected, 5, reverse=False),
        "best5": _tail_sum(selected, 5, reverse=True),
        "avg_prob": _mean([_float(row.get("probability")) for row in selected]),
        "disable_gap_exposure": sum(_int(row.get("disable_gap_exposure")) for row in selected),
        "symbol_mix": _mix(selected, "symbol"),
        "month_mix": _mix(selected, "month"),
        "quarter_mix": _mix(selected, "quarter"),
    }


def _by_group(rows: Sequence[Mapping[str, Any]], column: str) -> list[dict[str, Any]]:
    output = []
    for value in sorted({str(row.get(column, "")) for row in rows}):
        group = [row for row in rows if str(row.get(column, "")) == value]
        output.append(
            {
                "group": value,
                "policy_count": len(group),
                "best_lift": max((_float(row.get("lift")) for row in group), default=0.0),
                "best_policy": (
                    max(group, key=lambda row: _float(row.get("lift")))["policy_name"]
                    if group else ""
                ),
            }
        )
    return output


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
                    "slice_family": mix_column,
                    "slice_value": value,
                    "selected_rows": int(count),
                }
            )
    return output


def _tail_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_label": row["run_label"],
            "policy_name": row["policy_name"],
            "net_proxy": row["net_proxy"],
            "worst5": row["worst5"],
            "best5": row["best5"],
            "selected_rows": row["selected_rows"],
        }
        for row in rows
    ]


def _recommendation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stable = []
    for policy_name in sorted({row["policy_name"] for row in rows}):
        policy_rows = [row for row in rows if row["policy_name"] == policy_name]
        if policy_rows and min(_float(row.get("lift")) for row in policy_rows) > 1.2:
            stable.append(policy_name)
    non_gate_stable = [
        policy for policy in stable
        if policy != "GATE_ONLY_CONDITION_THEN_TOP_0.25"
    ]
    gate_equivalent = _gate_equivalent(rows, "GATE_AND_MOMENTUM")
    best = non_gate_stable[0] if non_gate_stable and not gate_equivalent else ""
    recommendation = (
        "KEEP_COMBO_AS_RESEARCH_DIAGNOSTIC_CANDIDATE"
        if best else "NO_INCREMENTAL_COMBO_EDGE_OVER_PAUSED_GATE"
    )
    return {
        "recommendation": recommendation,
        "stable_combo_candidates": stable,
        "best_research_combo": best,
        "gate_and_momentum_equivalent_to_gate_only": gate_equivalent,
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": ["COMBO_DIAGNOSTIC_ONLY", "NOT_BACKTEST", "NO_PROFIT_CLAIM"],
    }


def _gate_equivalent(rows: Sequence[Mapping[str, Any]], policy_name: str) -> bool:
    by_key = {(row["run_label"], row["policy_name"]): row for row in rows}
    for run_label in {str(row["run_label"]) for row in rows}:
        gate = by_key.get((run_label, "GATE_ONLY_CONDITION_THEN_TOP_0.25"), {})
        combo = by_key.get((run_label, policy_name), {})
        if gate.get("selected_rows") != combo.get("selected_rows"):
            return False
        if abs(_float(gate.get("precision")) - _float(combo.get("precision"))) > 1e-12:
            return False
    return True


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "policy_metrics_csv": str(output_dir / "policy_metrics.csv"),
        "by_run_csv": str(output_dir / "by_run.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "tail_summary_csv": str(output_dir / "tail_summary.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]) -> str:
    best = max(metrics, key=lambda row: _float(row.get("lift")), default={})
    return "\n".join(
        [
            "# M20 Gate + Momentum Combo Diagnostic",
            "",
            f"- Recommendation: `{report['recommendation']}`",
            f"- Best observed policy: `{best.get('policy_name', '')}`",
            f"- Honesty flags: `{', '.join(HONESTY_FLAGS)}`",
            "",
            "This is combo diagnostics only, not a backtest or trading logic.",
            "",
        ]
    )


def _tail_sum(rows: Sequence[Mapping[str, Any]], limit: int, *, reverse: bool) -> float:
    ranked = sorted(
        rows,
        key=lambda row: _float(row.get("net_return_proxy")),
        reverse=reverse,
    )
    return sum(_float(row.get("net_return_proxy")) for row in ranked[:limit])


def _mix(rows: Sequence[Mapping[str, Any]], column: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(column, ""))
        counts[value] = counts.get(value, 0) + 1
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else 0.0


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


def read_combo_csv(path: Path) -> list[dict[str, str]]:
    """Read combo diagnostic CSV for tests."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
