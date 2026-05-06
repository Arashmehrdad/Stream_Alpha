"""Research-only M20 rank-gate net-proxy diagnostics."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_rank_gate_economics import (
    LOCKED_POLICY,
    _condition_then_top_mask as condition_then_top_mask,  # pylint: disable=protected-access
    _joined_rows as joined_rows,  # pylint: disable=protected-access
    _load_packet as load_packet,  # pylint: disable=protected-access
    _runs_from_packet as runs_from_packet,  # pylint: disable=protected-access
)
from app.training.m20_rank_gated_selector import (
    _condition_weights as load_condition_weights,  # pylint: disable=protected-access
)
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "rank_gate_net_diagnostics"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "NET_PROXY_MIXED",
    "NOT_PNL",
    "NO_RUNTIME",
    "NO_REGISTRY",
    "NO_PROMOTION",
    "NO_PROFIT_CLAIM",
    "SPARSE_SELECTION",
)
TOP_ROW_LIMIT = 25


def diagnose_m20_rank_gate_net(*, base_run_dir: Path) -> dict[str, Any]:
    """Explain locked rank-gate net-proxy sign flips using existing artifacts."""
    # pylint: disable=too-many-locals
    base_dir = Path(base_run_dir).resolve()
    output_dir = base_dir / "research_labels" / "vol_scaled" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = load_packet(base_dir)
    condition_weights = load_condition_weights(base_dir)
    runs = runs_from_packet(packet, base_dir)

    selected_rows: list[dict[str, Any]] = []
    by_run: list[dict[str, Any]] = []
    by_symbol: list[dict[str, Any]] = []
    by_time: list[dict[str, Any]] = []
    by_feature_bucket: list[dict[str, Any]] = []
    tail_events: list[dict[str, Any]] = []
    for run in runs:
        run_label = str(run["window_label"])
        rows = joined_rows(Path(run["run_dir"]), condition_weights)
        mask = condition_then_top_mask(rows, 0.25)
        selected = [
            _selected_row(run_label, index, row)
            for index, (row, keep) in enumerate(zip(rows, mask))
            if keep
        ]
        selected_rows.extend(selected)
        by_run.append(_summary_row(run_label, "run", run_label, selected))
        by_symbol.extend(_group_summary(run_label, selected, "symbol"))
        by_time.extend(_group_summary(run_label, selected, "month"))
        by_time.extend(_group_summary(run_label, selected, "quarter"))
        by_feature_bucket.extend(_feature_bucket_rows(run_label, selected))
        tail_events.extend(_tail_rows(run_label, selected))

    recommendation = _recommendation(by_run)
    output_files = _output_files(output_dir)
    report = {
        "policy": LOCKED_POLICY,
        "run_count": len(runs),
        "selected_rows": len(selected_rows),
        "recommendation": recommendation["recommendation"],
        "blockers": recommendation["blockers"],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "base_run_dir": str(base_dir),
        "output_dir": str(output_dir),
        "policy": LOCKED_POLICY,
        "evidence_status": packet.get("evidence_status", "UNKNOWN"),
        "runtime_status": "NOT_RUNTIME_READY",
        "promotion_status": "NOT_PROMOTABLE",
        "honesty_flags": list(HONESTY_FLAGS),
        "source_artifacts": [
            str(base_dir / "research_labels" / "vol_scaled" / "rank_gate_economics"),
            str(base_dir / "research_labels" / "vol_scaled" / "rank_gate_evidence_packet"),
        ],
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _markdown(report, by_run),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["by_run_csv"]), by_run)
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_csv_artifact(Path(output_files["by_feature_bucket_csv"]), by_feature_bucket)
    write_csv_artifact(Path(output_files["tail_events_csv"]), tail_events)
    write_csv_artifact(Path(output_files["selected_row_diagnostics_csv"]), selected_rows)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe({**report, "by_run": by_run, "tail_events": tail_events})


def _selected_row(run_label: str, selected_rank: int, row: Mapping[str, Any]) -> dict[str, Any]:
    net_proxy = _float(row.get("net_return_proxy"))
    return {
        "run_label": run_label,
        "selected_rank": selected_rank,
        "symbol": row.get("symbol", ""),
        "interval_begin": row.get("interval_begin", ""),
        "month": row.get("month", ""),
        "quarter": row.get("quarter", ""),
        "label": int(row.get("label", 0)),
        "probability": _float(row.get("probability")),
        "future_return": _float(row.get("future_return")),
        "cost_per_trade": _float(row.get("cost_per_trade")),
        "net_return_proxy": net_proxy,
        "is_tail_loss": net_proxy <= 0.0,
        "realized_vol_12": _float(row.get("realized_vol_12")),
        "range_pct": _range_pct(row),
        "volume": _float(row.get("volume")),
        "macd_line_12_26": _float(row.get("macd_line_12_26")),
        "log_return_1": _float(row.get("log_return_1")),
        "probability_bin": _probability_bin(_float(row.get("probability"))),
        "volatility_bucket": _bucket(_float(row.get("realized_vol_12"))),
        "range_bucket": _bucket(_range_pct(row)),
        "volume_bucket": _bucket(_float(row.get("volume"))),
        "macd_bucket": _signed_bucket(_float(row.get("macd_line_12_26"))),
        "momentum_bucket": _signed_bucket(_float(row.get("log_return_1"))),
        "confirmed_condition_count": len(row.get("confirmed_conditions", [])),
    }


def _summary_row(
    run_label: str,
    slice_family: str,
    slice_value: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    positives = [row for row in rows if int(row.get("label", 0))]
    negatives = [row for row in rows if not int(row.get("label", 0))]
    net_values = [_float(row.get("net_return_proxy")) for row in rows]
    forward_values = [_float(row.get("future_return")) for row in rows]
    tail_losses = [value for value in net_values if value < 0.0]
    tail_loss_sum = sum(tail_losses)
    net_sum = sum(net_values)
    return {
        "run_label": run_label,
        "slice_family": slice_family,
        "slice_value": slice_value,
        "selected_rows": len(rows),
        "true_positive_count": len(positives),
        "false_positive_count": len(negatives),
        "precision": len(positives) / len(rows) if rows else 0.0,
        "mean_forward_return": _mean(forward_values),
        "median_forward_return": _median(forward_values),
        "p10_forward_return": _quantile(forward_values, 0.10),
        "p90_forward_return": _quantile(forward_values, 0.90),
        "mean_net_proxy": _mean(net_values),
        "median_net_proxy": _median(net_values),
        "p10_net_proxy": _quantile(net_values, 0.10),
        "p90_net_proxy": _quantile(net_values, 0.90),
        "winner_contribution": sum(value for value in net_values if value > 0.0),
        "loser_contribution": tail_loss_sum,
        "tail_loss_contribution": tail_loss_sum,
        "net_value_proxy": net_sum,
        "cost_scenario_contribution": sum(_float(row.get("cost_per_trade")) for row in rows),
        "tail_loss_share_of_abs_net": abs(tail_loss_sum) / sum(abs(value) for value in net_values)
        if net_values else 0.0,
    }


def _group_summary(
    run_label: str,
    rows: Sequence[Mapping[str, Any]],
    column: str,
) -> list[dict[str, Any]]:
    return [
        _summary_row(
            run_label,
            column,
            value,
            [row for row in rows if str(row.get(column, "")) == value],
        )
        for value in sorted({str(row.get(column, "")) for row in rows})
    ]


def _feature_bucket_rows(run_label: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for column in (
        "probability_bin",
        "volatility_bucket",
        "range_bucket",
        "volume_bucket",
        "macd_bucket",
        "momentum_bucket",
    ):
        output.extend(_group_summary(run_label, rows, column))
    return output


def _tail_rows(run_label: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: _float(row.get("net_return_proxy")))
    tail_rows = ranked[:TOP_ROW_LIMIT] + list(reversed(ranked[-TOP_ROW_LIMIT:]))
    output = []
    for row in tail_rows:
        output.append(
            {
                "run_label": run_label,
                "tail_side": _tail_side(row),
                "symbol": row.get("symbol", ""),
                "interval_begin": row.get("interval_begin", ""),
                "month": row.get("month", ""),
                "quarter": row.get("quarter", ""),
                "label": row.get("label", 0),
                "probability": row.get("probability", 0.0),
                "future_return": row.get("future_return", 0.0),
                "net_return_proxy": row.get("net_return_proxy", 0.0),
            }
        )
    return output


def _tail_side(row: Mapping[str, Any]) -> str:
    return "negative" if _float(row.get("net_return_proxy")) < 0.0 else "positive"


def _recommendation(by_run: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    net_values = [_float(row.get("net_value_proxy")) for row in by_run]
    has_positive = any(value > 0.0 for value in net_values)
    has_negative = any(value < 0.0 for value in net_values)
    recommendation = (
        "DIAGNOSE_TAIL_AND_CONDITION_CONCENTRATION_BEFORE_ANY_POLICY_STEP"
        if has_positive and has_negative
        else "KEEP_RESEARCH_ONLY_NET_PROXY_MONITORING"
    )
    return {
        "recommendation": recommendation,
        "net_proxy_signs_mixed": has_positive and has_negative,
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": [
            "NOT_PNL",
            "NOT_PROFIT_EVIDENCE",
            "SPARSE_SELECTION",
            "NET_PROXY_MIXED",
            "NOT_RUNTIME_READY",
        ],
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "by_run_csv": str(output_dir / "by_run.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "by_feature_bucket_csv": str(output_dir / "by_feature_bucket.csv"),
        "tail_events_csv": str(output_dir / "tail_events.csv"),
        "selected_row_diagnostics_csv": str(output_dir / "selected_row_diagnostics.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], by_run: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# M20 Rank-Gate Net-Proxy Diagnostics",
        "",
        f"- Policy: `{LOCKED_POLICY}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Honesty flags: `{', '.join(HONESTY_FLAGS)}`",
        "",
        "This is research-only net-proxy decomposition, not PnL and not a backtest.",
        "",
        "## Run Summary",
        "",
    ]
    for row in by_run:
        lines.append(
            f"- `{row['run_label']}`: selected `{row['selected_rows']}`, "
            f"precision `{row['precision']}`, net proxy `{row['net_value_proxy']}`"
        )
    lines.append("")
    return "\n".join(lines)


def _range_pct(row: Mapping[str, Any]) -> float:
    high = _float(row.get("high_price"))
    low = _float(row.get("low_price"))
    close = _float(row.get("close_price"))
    return (high - low) / close if close else 0.0


def _probability_bin(value: float) -> str:
    if value >= 0.99:
        return "p99_plus"
    if value >= 0.95:
        return "p95_p99"
    if value >= 0.90:
        return "p90_p95"
    return "below_p90"


def _bucket(value: float) -> str:
    if value <= 0.0:
        return "missing_or_non_positive"
    if value < 0.001:
        return "low"
    if value < 0.003:
        return "mid"
    return "high"


def _signed_bucket(value: float) -> str:
    if value < -0.000001:
        return "negative"
    if value > 0.000001:
        return "positive"
    return "flat"


def _float(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 0.0
    return converted if math.isfinite(converted) else 0.0


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


def read_net_diagnostic_csv(path: Path) -> list[dict[str, str]]:
    """Read diagnostic CSV rows for tests."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
