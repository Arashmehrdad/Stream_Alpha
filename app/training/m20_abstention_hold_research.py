"""Research-only M20 abstention/HOLD diagnostics."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_momentum_breakout_research import (
    _range_pct as range_pct,  # pylint: disable=protected-access
    _tertiles as tertiles,  # pylint: disable=protected-access
)
from app.training.m20_rank_gate_economics import (
    _condition_then_top_mask as gate_mask,  # pylint: disable=protected-access
    _joined_rows as joined_rank_rows,  # pylint: disable=protected-access
    _load_packet as load_packet,  # pylint: disable=protected-access
    _runs_from_packet as runs_from_packet,  # pylint: disable=protected-access
)
from app.training.m20_rank_gated_selector import (
    _condition_weights as load_condition_weights,  # pylint: disable=protected-access
)
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "abstention_hold_research"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "EXISTING_ARTIFACTS_ONLY",
    "ABSTENTION_DIAGNOSTIC_ONLY",
    "NOT_BACKTEST",
    "NOT_PNL",
    "NO_RUNTIME",
    "NO_REGISTRY",
    "NO_PROMOTION",
    "NO_PROFIT_CLAIM",
)


def analyze_m20_abstention_hold(*, base_run_dir: Path) -> dict[str, Any]:
    """Analyze research-only HOLD/skip candidates from existing M20 artifacts."""
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
        selected_mask = gate_mask(rows, 0.25)
        for rule_name, hold_mask in _hold_masks(rows, selected_mask).items():
            metrics.append(_metric_row(run_label, rule_name, rows, selected_mask, hold_mask))

    by_run = _by_group(metrics, "run_label")
    by_symbol = _slice_rows(metrics, "symbol")
    by_time = _slice_rows(metrics, "month") + _slice_rows(metrics, "quarter")
    avoided_loss = _proxy_rows(metrics, "avoided_negative_net_proxy")
    missed_positive = _proxy_rows(metrics, "missed_positive_count")
    recommendation = _recommendation(_stability(metrics))
    output_files = _output_files(output_dir)
    report = {
        "run_count": len(runs),
        "rule_count": len({row["rule_name"] for row in metrics}),
        "recommendation": recommendation["recommendation"],
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": recommendation["blockers"],
        "output_files": output_files,
    }
    manifest = {
        "base_run_dir": str(base_dir),
        "output_dir": str(output_dir),
        "source": "existing_artifacts_only",
        "rank_gate_usage": "ABSTENTION_INPUT_ONLY",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _markdown(report, metrics),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["hold_rule_metrics_csv"]), metrics)
    write_csv_artifact(Path(output_files["by_run_csv"]), by_run)
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_csv_artifact(Path(output_files["avoided_loss_proxy_csv"]), avoided_loss)
    write_csv_artifact(Path(output_files["missed_positive_proxy_csv"]), missed_positive)
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
    probability_low = _quantile([_float(row.get("probability")) for row in rows], 0.25)
    for row in rows:
        row["range_pct"] = range_pct(row)
        row["realized_vol_high"] = _float(row.get("realized_vol_12")) >= vol_high
        row["range_high"] = row["range_pct"] >= range_high
        row["volume_high"] = _float(row.get("volume")) >= volume_high
        row["vol_plus_range_high"] = bool(row["realized_vol_high"] and row["range_high"])
        row["confirmed_condition_count"] = len(row.get("confirmed_conditions", []))
        row["probability_low"] = _float(row.get("probability")) <= probability_low
        row["broad_unstable_volatility"] = bool(row["realized_vol_high"] or row["range_high"])
    return rows


def _hold_masks(
    rows: Sequence[Mapping[str, Any]],
    selected_mask: Sequence[bool],
) -> dict[str, list[bool]]:
    selected_rows = [row for row, keep in zip(rows, selected_mask) if keep]
    selected_net_median = _median([_float(row.get("net_return_proxy")) for row in selected_rows])
    return {
        "HOLD_LOW_PROBABILITY": [bool(row.get("probability_low")) for row in rows],
        "HOLD_NO_CONFIRMED_CONDITIONS": [
            _int(row.get("confirmed_condition_count")) == 0 for row in rows
        ],
        "HOLD_DISABLE_GAP_EXPOSURE": [bool(row.get("disable_gap_match")) for row in rows],
        "HOLD_BROAD_UNSTABLE_VOLATILITY": [
            bool(row.get("broad_unstable_volatility")) and not bool(row.get("vol_plus_range_high"))
            for row in rows
        ],
        "HOLD_SELECTED_NEGATIVE_NET_PROXY": [
            keep and _float(row.get("net_return_proxy")) < 0.0
            for row, keep in zip(rows, selected_mask)
        ],
        "HOLD_SELECTED_BELOW_MEDIAN_NET_PROXY": [
            keep and _float(row.get("net_return_proxy")) <= selected_net_median
            for row, keep in zip(rows, selected_mask)
        ],
    }


def _metric_row(
    run_label: str,
    rule_name: str,
    rows: Sequence[Mapping[str, Any]],
    selected_mask: Sequence[bool],
    hold_mask: Sequence[bool],
) -> dict[str, Any]:
    selected = [row for row, keep in zip(rows, selected_mask) if keep]
    held = [row for row, hold in zip(rows, hold_mask) if hold]
    remaining = [
        row for row, keep, hold in zip(rows, selected_mask, hold_mask)
        if keep and not hold
    ]
    positives = sum(_int(row.get("label")) for row in rows)
    remaining_pos = sum(_int(row.get("label")) for row in remaining)
    selected_pos = sum(_int(row.get("label")) for row in selected)
    base_rate = positives / len(rows) if rows else 0.0
    remaining_precision = remaining_pos / len(remaining) if remaining else 0.0
    avoided_negative = sum(
        -_float(row.get("net_return_proxy"))
        for row, keep, hold in zip(rows, selected_mask, hold_mask)
        if keep and hold and _float(row.get("net_return_proxy")) < 0.0
    )
    missed_positive = sum(
        _int(row.get("label"))
        for row, keep, hold in zip(rows, selected_mask, hold_mask)
        if keep and hold
    )
    return {
        "run_label": run_label,
        "rule_name": rule_name,
        "total_rows": len(rows),
        "selected_rows_before_hold": len(selected),
        "selected_rows_after_hold": len(remaining),
        "skipped_rows": len(held),
        "skip_coverage": len(held) / len(rows) if rows else 0.0,
        "selected_skip_count": len(selected) - len(remaining),
        "positive_events_skipped": sum(_int(row.get("label")) for row in held),
        "false_positives_avoided": sum(
            1
            for row, keep, hold in zip(rows, selected_mask, hold_mask)
            if keep and hold and not _int(row.get("label"))
        ),
        "missed_positive_count": missed_positive,
        "avoided_negative_net_proxy": avoided_negative,
        "remaining_precision": remaining_precision,
        "remaining_lift": remaining_precision / base_rate if base_rate else 0.0,
        "remaining_net_proxy": sum(_float(row.get("net_return_proxy")) for row in remaining),
        "before_precision": selected_pos / len(selected) if selected else 0.0,
        "before_net_proxy": sum(_float(row.get("net_return_proxy")) for row in selected),
        "symbol_mix": _mix(held, "symbol"),
        "month_mix": _mix(held, "month"),
        "quarter_mix": _mix(held, "quarter"),
    }


def _stability(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for rule_name in sorted({str(row["rule_name"]) for row in rows}):
        rule_rows = [row for row in rows if row["rule_name"] == rule_name]
        positive_miss_rate = _mean([
            _float(row.get("missed_positive_count"))
            / max(1.0, _float(row.get("selected_skip_count")))
            for row in rule_rows
        ])
        output.append(
            {
                "rule_name": rule_name,
                "window_count": len(rule_rows),
                "avg_avoided_negative_net_proxy": _mean([
                    _float(row.get("avoided_negative_net_proxy")) for row in rule_rows
                ]),
                "avg_remaining_net_proxy": _mean([
                    _float(row.get("remaining_net_proxy")) for row in rule_rows
                ]),
                "avg_remaining_lift": _mean([
                    _float(row.get("remaining_lift")) for row in rule_rows
                ]),
                "avg_skip_coverage": _mean([
                    _float(row.get("skip_coverage")) for row in rule_rows
                ]),
                "avg_positive_miss_rate": positive_miss_rate,
                "positive_remaining_net_windows": sum(
                    _float(row.get("remaining_net_proxy")) > 0.0 for row in rule_rows
                ),
                "implementability": _implementability(rule_name),
                "classification": _classify_rule(rule_rows, positive_miss_rate),
            }
        )
    return output


def _recommendation(stability: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    implementable = [
        row for row in stability if row.get("implementability") == "PREDICTION_TIME_PROXY"
    ]
    keep = [
        row for row in implementable if row["classification"] == "KEEP_ABSTENTION_CANDIDATE"
    ]
    watchlist = [
        row for row in implementable if row["classification"] == "WATCHLIST_ABSTENTION_RULE"
    ]
    oracle = [
        row for row in stability if row.get("implementability") == "ORACLE_NET_PROXY_ONLY"
    ]
    if keep:
        recommendation = "TEST_ABSTENTION_WITH_STRATEGY_FAMILIES_NEXT"
    elif watchlist:
        recommendation = "KEEP_ABSTENTION_AS_RESEARCH_FILTER"
    elif stability:
        recommendation = "NEED_RICHER_BAD_REGIME_LABELS"
    else:
        recommendation = "REJECT_CURRENT_ABSTENTION_RULES"
    return {
        "recommendation": recommendation,
        "kept_rules": [row["rule_name"] for row in keep],
        "watchlist_rules": [row["rule_name"] for row in watchlist],
        "oracle_diagnostic_rules": [row["rule_name"] for row in oracle],
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": [
            "ABSTENTION_DIAGNOSTIC_ONLY",
            "ORACLE_NET_PROXY_RULES_NOT_IMPLEMENTABLE",
            "NOT_BACKTEST",
            "NOT_PNL",
            "NO_PROFIT_CLAIM",
            "NOT_RUNTIME_READY",
        ],
    }


def _classify_rule(
    rows: Sequence[Mapping[str, Any]],
    positive_miss_rate: float,
) -> str:
    avg_avoided = _mean([_float(row.get("avoided_negative_net_proxy")) for row in rows])
    positive_remaining_windows = sum(_float(row.get("remaining_net_proxy")) > 0.0 for row in rows)
    avg_skip = _mean([_float(row.get("skip_coverage")) for row in rows])
    if avg_skip > 0.5 or positive_miss_rate > 0.5:
        return "TOO_BROAD_MISSES_TOO_MUCH"
    if avg_avoided > 0.0 and positive_remaining_windows == len(rows):
        return "KEEP_ABSTENTION_CANDIDATE"
    if avg_avoided > 0.0 and positive_remaining_windows >= 1:
        return "WATCHLIST_ABSTENTION_RULE"
    if avg_avoided > 0.0:
        return "UNSTABLE_ABSTENTION_RULE"
    return "REJECT_FOR_NOW"


def _implementability(rule_name: str) -> str:
    if rule_name in (
        "HOLD_SELECTED_NEGATIVE_NET_PROXY",
        "HOLD_SELECTED_BELOW_MEDIAN_NET_PROXY",
    ):
        return "ORACLE_NET_PROXY_ONLY"
    return "PREDICTION_TIME_PROXY"


def _by_group(rows: Sequence[Mapping[str, Any]], column: str) -> list[dict[str, Any]]:
    output = []
    for value in sorted({str(row.get(column, "")) for row in rows}):
        group = [row for row in rows if str(row.get(column, "")) == value]
        output.append(
            {
                "group": value,
                "rule_count": len(group),
                "best_avoided_negative_net_proxy": max(
                    (_float(row.get("avoided_negative_net_proxy")) for row in group),
                    default=0.0,
                ),
                "best_rule": (
                    max(group, key=lambda row: _float(row.get("avoided_negative_net_proxy")))[
                        "rule_name"
                    ]
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
                    "rule_name": row["rule_name"],
                    "slice_family": mix_column,
                    "slice_value": value,
                    "skipped_rows": int(count),
                }
            )
    return output


def _proxy_rows(rows: Sequence[Mapping[str, Any]], column: str) -> list[dict[str, Any]]:
    return [
        {
            "run_label": row["run_label"],
            "rule_name": row["rule_name"],
            column: row[column],
            "selected_skip_count": row["selected_skip_count"],
        }
        for row in rows
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "hold_rule_metrics_csv": str(output_dir / "hold_rule_metrics.csv"),
        "by_run_csv": str(output_dir / "by_run.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "avoided_loss_proxy_csv": str(output_dir / "avoided_loss_proxy.csv"),
        "missed_positive_proxy_csv": str(output_dir / "missed_positive_proxy.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]) -> str:
    best = max(
        metrics,
        key=lambda row: _float(row.get("avoided_negative_net_proxy")),
        default={},
    )
    return "\n".join(
        [
            "# M20 Abstention/HOLD Research Diagnostic",
            "",
            f"- Recommendation: `{report['recommendation']}`",
            f"- Best avoided-loss rule: `{best.get('rule_name', '')}`",
            f"- Honesty flags: `{', '.join(HONESTY_FLAGS)}`",
            "",
            "This is research-only abstention diagnostics, not runtime HOLD logic,",
            "not a backtest, not PnL, and not profitability evidence.",
            "",
        ]
    )


def _mix(rows: Sequence[Mapping[str, Any]], column: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(column, ""))
        counts[value] = counts.get(value, 0) + 1
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _quantile(values: Sequence[float], quantile: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return 0.0
    index = min(len(finite) - 1, max(0, int((len(finite) - 1) * quantile)))
    return finite[index]


def _median(values: Sequence[float]) -> float:
    return _quantile(values, 0.5)


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


def read_abstention_csv(path: Path) -> list[dict[str, str]]:
    """Read abstention/HOLD diagnostic CSV for tests."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
