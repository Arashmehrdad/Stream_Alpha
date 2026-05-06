"""Research-only M20 volatility combo economics diagnostics."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_momentum_breakout_research import (
    _momentum as momentum_value,  # pylint: disable=protected-access
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


OUTPUT_DIR_NAME = "volatility_combo_economics"
VOLATILITY_SETUPS = (
    "vol_plus_range_high",
    "vol_plus_volume_high",
    "range_plus_volume_high",
    "realized_vol_high",
)
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "EXISTING_ARTIFACTS_ONLY",
    "COMBO_ECONOMICS_DIAGNOSTIC_ONLY",
    "NOT_BACKTEST",
    "NOT_PNL",
    "NO_RUNTIME",
    "NO_REGISTRY",
    "NO_PROMOTION",
    "NO_PROFIT_CLAIM",
)


def analyze_m20_volatility_combo_economics(*, base_run_dir: Path) -> dict[str, Any]:
    """Analyze rank-gate, momentum, and volatility combo economics diagnostics."""
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
        for policy_name, mask in _policy_masks(rows).items():
            metrics.append(_metric_row(run_label, policy_name, rows, mask))

    stability = _stability(metrics)
    recommendation = _recommendation(stability)
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
        "volatility_setups": list(VOLATILITY_SETUPS),
        "source": "existing_artifacts_only",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(_markdown(report, stability), encoding="utf-8")
    write_csv_artifact(Path(output_files["policy_metrics_csv"]), metrics)
    write_csv_artifact(Path(output_files["by_run_csv"]), _by_group(metrics, "run_label"))
    write_csv_artifact(Path(output_files["by_symbol_csv"]), _slice_rows(metrics, "symbol"))
    write_csv_artifact(
        Path(output_files["by_time_csv"]),
        _slice_rows(metrics, "month") + _slice_rows(metrics, "quarter"),
    )
    write_csv_artifact(Path(output_files["tail_summary_csv"]), _tail_summary(metrics))
    write_csv_artifact(Path(output_files["stability_csv"]), stability)
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
        row["range_pct"] = range_pct(row)
        row["realized_vol_high"] = _float(row.get("realized_vol_12")) >= vol_high
        row["range_high"] = row["range_pct"] >= range_high
        row["volume_high"] = _float(row.get("volume")) >= volume_high
        row["vol_plus_range_high"] = bool(row["realized_vol_high"] and row["range_high"])
        row["vol_plus_volume_high"] = bool(row["realized_vol_high"] and row["volume_high"])
        row["range_plus_volume_high"] = bool(row["range_high"] and row["volume_high"])
        row["volatility_setup"] = any(bool(row[setup]) for setup in VOLATILITY_SETUPS)
        row["momentum_setup"] = any(
            bool(row.get(setup)) for setup in ("realized_vol_high", "range_high", "volume_high")
        )
        row["momentum_positive_or_flat"] = momentum_value(row) >= -0.0005
    return rows


def _policy_masks(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[bool]]:
    gate = gate_mask(rows, 0.25)
    volatility = [bool(row.get("volatility_setup")) for row in rows]
    momentum = [bool(row.get("momentum_setup")) for row in rows]
    masks = {
        f"VOLATILITY_ONLY_{setup.upper()}": [bool(row.get(setup)) for row in rows]
        for setup in VOLATILITY_SETUPS
    }
    masks.update(
        {
            "RANK_GATE_ONLY_CONDITION_THEN_TOP_0.25": gate,
            "RANK_GATE_AND_VOLATILITY": [
                keep and vol for keep, vol in zip(gate, volatility)
            ],
            "RANK_GATE_OR_VOLATILITY": [
                keep or vol for keep, vol in zip(gate, volatility)
            ],
            "MOMENTUM_AND_VOLATILITY": [
                mom and vol for mom, vol in zip(momentum, volatility)
            ],
            "RANK_GATE_AND_MOMENTUM_AND_VOLATILITY": [
                keep and mom and vol
                for keep, mom, vol in zip(gate, momentum, volatility)
            ],
        }
    )
    return masks


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
        "net_proxy_mean": sum(net_values) / len(net_values) if net_values else 0.0,
        "worst5": _tail_sum(selected, 5, reverse=False),
        "best5": _tail_sum(selected, 5, reverse=True),
        "avg_prob": _mean([_float(row.get("probability")) for row in selected]),
        "disable_gap_exposure": sum(bool(row.get("disable_gap_match")) for row in selected),
        "symbol_mix": _mix(selected, "symbol"),
        "month_mix": _mix(selected, "month"),
        "quarter_mix": _mix(selected, "quarter"),
        "classification": "",
    }


def _stability(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    gate = _policy_rows(rows, "RANK_GATE_ONLY_CONDITION_THEN_TOP_0.25")
    gate_min_lift = min((_float(row.get("lift")) for row in gate), default=0.0)
    gate_negative_windows = sum(_float(row.get("net_proxy")) < 0.0 for row in gate)
    for policy_name in sorted({str(row["policy_name"]) for row in rows}):
        policy_rows = _policy_rows(rows, policy_name)
        min_lift = min((_float(row.get("lift")) for row in policy_rows), default=0.0)
        avg_lift = _mean([_float(row.get("lift")) for row in policy_rows])
        negative_windows = sum(_float(row.get("net_proxy")) < 0.0 for row in policy_rows)
        avg_net = _mean([_float(row.get("net_proxy")) for row in policy_rows])
        avg_coverage = _mean([_float(row.get("coverage")) for row in policy_rows])
        output.append(
            {
                "policy_name": policy_name,
                "window_count": len(policy_rows),
                "min_lift": min_lift,
                "avg_lift": avg_lift,
                "avg_coverage": avg_coverage,
                "avg_net_proxy": avg_net,
                "negative_net_windows": negative_windows,
                "disable_gap_exposure": sum(
                    _int(row.get("disable_gap_exposure")) for row in policy_rows
                ),
                "improves_min_lift_vs_gate": min_lift > gate_min_lift,
                "worsens_net_stability_vs_gate": negative_windows > gate_negative_windows,
                "classification": _classify_policy(
                    policy_name,
                    min_lift,
                    avg_coverage,
                    negative_windows,
                    gate_negative_windows,
                ),
            }
        )
    return output


def _recommendation(stability: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    combo_candidates = [
        row for row in stability
        if str(row.get("policy_name", "")).startswith(("RANK_GATE_AND", "MOMENTUM_AND"))
    ]
    stable_combo = [
        row for row in combo_candidates
        if row.get("classification") == "KEEP_RESEARCH_CANDIDATE"
    ]
    if stable_combo:
        recommendation = "TEST_VOLATILITY_COMBO_CONFIRMATION_WINDOW"
    elif any(row.get("classification") == "WATCHLIST_ONLY" for row in combo_candidates):
        recommendation = "TRY_VOLATILITY_AS_OPTIONAL_GATE_FILTER"
    elif any(str(row.get("policy_name", "")).startswith("VOLATILITY_ONLY") for row in stability):
        recommendation = "KEEP_VOLATILITY_LABEL_LIFT_ONLY"
    else:
        recommendation = "REJECT_VOLATILITY_COMBO_FOR_NOW"
    return {
        "recommendation": recommendation,
        "best_combo_candidate": stable_combo[0]["policy_name"] if stable_combo else "",
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": [
            "COMBO_ECONOMICS_DIAGNOSTIC_ONLY",
            "NOT_BACKTEST",
            "NOT_PNL",
            "NO_PROFIT_CLAIM",
            "NOT_RUNTIME_READY",
        ],
    }


def _classify_policy(
    policy_name: str,
    min_lift: float,
    avg_coverage: float,
    negative_windows: int,
    gate_negative_windows: int,
) -> str:
    if negative_windows > gate_negative_windows:
        return "UNSTABLE_NET_PROXY"
    if min_lift >= 1.5 and avg_coverage <= 0.25 and negative_windows < gate_negative_windows:
        return "KEEP_RESEARCH_CANDIDATE"
    if min_lift >= 1.3 and avg_coverage <= 0.35:
        return "WATCHLIST_ONLY"
    if policy_name.startswith("VOLATILITY_ONLY") and min_lift >= 1.2:
        return "BROAD_LABEL_LIFT_NO_ECONOMICS"
    return "REJECT_FOR_NOW"


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
                "best_net_proxy": max(
                    (_float(row.get("net_proxy")) for row in group),
                    default=0.0,
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
            "selected_rows": row["selected_rows"],
            "net_proxy": row["net_proxy"],
            "worst5": row["worst5"],
            "best5": row["best5"],
        }
        for row in rows
    ]


def _policy_rows(rows: Sequence[Mapping[str, Any]], policy_name: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("policy_name") == policy_name]


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
        "stability_csv": str(output_dir / "stability.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], stability: Sequence[Mapping[str, Any]]) -> str:
    best = max(stability, key=lambda row: _float(row.get("min_lift")), default={})
    return "\n".join(
        [
            "# M20 Volatility Combo Economics Diagnostic",
            "",
            f"- Recommendation: `{report['recommendation']}`",
            f"- Best min-lift policy: `{best.get('policy_name', '')}`",
            f"- Honesty flags: `{', '.join(HONESTY_FLAGS)}`",
            "",
            "This is research-only combo economics diagnostics, not a backtest,",
            "not PnL, not runtime logic, and not profitability evidence.",
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


def read_volatility_combo_csv(path: Path) -> list[dict[str, str]]:
    """Read volatility combo economics CSV for tests."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
