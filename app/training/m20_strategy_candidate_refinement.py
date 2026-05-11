"""Generic research-only refinement analysis for M20 strategy candidates."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_CANDIDATE_FACTORY_NAME = "strategy_candidate_factory"
DEFAULT_TRAINING_FRAME_DIR = "training_frame"
DEFAULT_OUTPUT_NAME = "strategy_candidate_refinement"
FEATURE_FILE = "m20_training_frame_features.csv"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
RECOMMEND_SLICE_POLICY = "EVALUATE_REFINED_SLICES_WITH_GENERIC_POLICY_TOOL"
RECOMMEND_REFINE = "REFINE_STRATEGY_CANDIDATE_DEFINITIONS"
NEXT_SLICE_POLICY = "RUN_GENERIC_STRATEGY_SLICE_POLICY_EVALUATOR"
NEXT_REFINE = "REFINE_STRATEGY_CANDIDATE_DEFINITIONS"


def analyze_m20_strategy_candidate_refinement(
    *,
    source_run_dir: Path,
    candidate_factory_dir: Path | None = None,
    training_frame_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
    min_slice_rows: int = 100,
) -> dict[str, Any]:
    """Analyze generic refinements for existing strategy candidate artifacts."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    vol_scaled_dir = source_dir / "research_labels" / "vol_scaled"
    factory_dir = (
        Path(candidate_factory_dir).resolve()
        if candidate_factory_dir is not None
        else vol_scaled_dir / DEFAULT_CANDIDATE_FACTORY_NAME
    )
    frame_dir = (
        Path(training_frame_dir).resolve()
        if training_frame_dir is not None
        else source_dir / DEFAULT_TRAINING_FRAME_DIR
    )
    _assert_factory_artifacts(factory_dir)
    output_dir = vol_scaled_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_metrics = _read_csv(factory_dir / "candidate_metrics.csv")
    context = _training_context(frame_dir / FEATURE_FILE)
    slice_aggregates = _aggregate_candidate_events(
        factory_dir / "strategy_candidates.csv",
        context,
    )
    slice_diagnostics = _slice_diagnostics(slice_aggregates, min_slice_rows)
    refined_metrics = _refined_metrics(candidate_metrics, slice_diagnostics)
    tail_loss = _tail_loss_diagnostics(slice_diagnostics)
    sample_sizes = _sample_size_diagnostics(slice_diagnostics, min_slice_rows)
    decisions = _candidate_decisions(refined_metrics)
    recommendation = _recommendation(decisions)
    output_files = _output_files(output_dir)
    report = {
        "summary": "Generic M20 strategy candidate refinement analysis.",
        "source_run_dir": str(source_dir),
        "candidate_factory_dir": str(factory_dir),
        "candidate_count": len(refined_metrics),
        "slice_count": len(slice_diagnostics),
        "min_slice_rows": min_slice_rows,
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "source_run_dir": str(source_dir),
        "candidate_factory_dir": str(factory_dir),
        "training_frame_dir": str(frame_dir),
        "min_slice_rows": min_slice_rows,
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(
        Path(output_files["strategy_candidate_refinement_report_json"]),
        report,
    )
    Path(output_files["strategy_candidate_refinement_report_md"]).write_text(
        _markdown(report, decisions),
        encoding="utf-8",
    )
    write_csv_artifact(
        Path(output_files["refined_candidate_metrics_csv"]),
        refined_metrics,
    )
    write_csv_artifact(Path(output_files["slice_diagnostics_csv"]), slice_diagnostics)
    write_csv_artifact(Path(output_files["tail_loss_diagnostics_csv"]), tail_loss)
    write_csv_artifact(Path(output_files["sample_size_diagnostics_csv"]), sample_sizes)
    write_csv_artifact(Path(output_files["candidate_decisions_csv"]), decisions)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "refined_candidate_metrics": refined_metrics,
            "slice_diagnostics": slice_diagnostics,
            "tail_loss_diagnostics": tail_loss,
            "sample_size_diagnostics": sample_sizes,
            "candidate_decisions": decisions,
            "recommendation_payload": recommendation,
        }
    )


def _assert_factory_artifacts(factory_dir: Path) -> None:
    missing = [
        path.name for path in (
            factory_dir / "strategy_candidates.csv",
            factory_dir / "candidate_metrics.csv",
            factory_dir / "manifest.json",
        )
        if not path.exists()
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing strategy candidate factory artifacts: {joined}")


def _training_context(feature_path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not feature_path.exists():
        return {}
    rows = _read_csv(feature_path)
    ranges = [_range_ratio(row) for row in rows]
    vols = [_to_float(row.get("realized_vol_12")) for row in rows]
    volumes = [_to_float(row.get("volume")) for row in rows]
    range_low, range_high = _quantiles(ranges)
    vol_low, vol_high = _quantiles(vols)
    volume_low, volume_high = _quantiles(volumes)
    output = {}
    for row in rows:
        key = _key(row)
        output[key] = {
            "range_bucket": _bucket(_range_ratio(row), range_low, range_high),
            "volatility_bucket": _bucket(
                _to_float(row.get("realized_vol_12")),
                vol_low,
                vol_high,
            ),
            "volume_bucket": _bucket(_to_float(row.get("volume")), volume_low, volume_high),
        }
    return output


def _aggregate_candidate_events(
    candidate_path: Path,
    context: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    aggregates: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in _iter_csv(candidate_path):
        candidate_key = _candidate_key(row)
        context_row = context.get(_key(row), {})
        slices = (
            ("symbol", row.get("symbol", "")),
            ("month", row.get("interval_begin", "")[:7]),
            ("quarter", _quarter(row.get("interval_begin", ""))),
            ("volatility_bucket", context_row.get("volatility_bucket", "UNKNOWN")),
            ("range_bucket", context_row.get("range_bucket", "UNKNOWN")),
            ("volume_bucket", context_row.get("volume_bucket", "UNKNOWN")),
        )
        for slice_family, slice_value in slices:
            aggregate = aggregates.setdefault(
                (*candidate_key, slice_family, slice_value),
                _new_aggregate(row, slice_family, slice_value),
            )
            _add_row(aggregate, row)
    return aggregates


def _new_aggregate(
    row: Mapping[str, str],
    slice_family: str,
    slice_value: str,
) -> dict[str, Any]:
    return {
        "strategy_family": row.get("strategy_family", ""),
        "candidate_name": row.get("candidate_name", ""),
        "slice_family": slice_family,
        "slice_value": slice_value,
        "selected_rows": 0,
        "positive_rows": 0,
        "net_values": [],
    }


def _add_row(aggregate: dict[str, Any], row: Mapping[str, str]) -> None:
    aggregate["selected_rows"] += 1
    aggregate["positive_rows"] += _to_int(row.get("fee_exceedance_label"))
    if row.get("net_value_proxy") not in ("", None):
        aggregate["net_values"].append(_to_float(row.get("net_value_proxy")))


def _slice_diagnostics(
    aggregates: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    min_slice_rows: int,
) -> list[dict[str, Any]]:
    rows = []
    for aggregate in sorted(
        aggregates.values(),
        key=lambda row: (
            str(row["strategy_family"]),
            str(row["candidate_name"]),
            str(row["slice_family"]),
            str(row["slice_value"]),
        ),
    ):
        net_values = list(aggregate["net_values"])
        selected_rows = int(aggregate["selected_rows"])
        mean_net = _mean(net_values) if net_values else ""
        rows.append(
            {
                "strategy_family": aggregate["strategy_family"],
                "candidate_name": aggregate["candidate_name"],
                "slice_family": aggregate["slice_family"],
                "slice_value": aggregate["slice_value"],
                "selected_rows": selected_rows,
                "selected_positive_rate": (
                    aggregate["positive_rows"] / selected_rows
                    if selected_rows
                    else 0.0
                ),
                "mean_net_proxy": mean_net,
                "cumulative_net_proxy": sum(net_values) if net_values else "",
                "max_drawdown_proxy": _max_drawdown(net_values) if net_values else "",
                "win_rate_proxy": (
                    sum(1 for value in net_values if value > 0.0) / len(net_values)
                    if net_values
                    else ""
                ),
                "worst_5_net_proxy": sum(sorted(net_values)[:5]) if net_values else "",
                "tail_loss_rate": _tail_loss_rate(net_values),
                "sample_size_status": _sample_size_status(selected_rows, min_slice_rows),
                "slice_decision": _slice_decision(selected_rows, min_slice_rows, mean_net),
            }
        )
    return rows


def _refined_metrics(
    candidate_metrics: Sequence[Mapping[str, str]],
    slice_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    slices_by_candidate: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in slice_rows:
        slices_by_candidate[(row["strategy_family"], row["candidate_name"])].append(row)
    output = []
    for metric in candidate_metrics:
        key = (metric.get("strategy_family", ""), metric.get("candidate_name", ""))
        slices = slices_by_candidate.get(key, [])
        eligible = [
            row for row in slices
            if row["slice_decision"] == "REFINED_SLICE_WATCHLIST_POSITIVE_NET_PROXY"
        ]
        best_slice = _best_slice(eligible or slices)
        output.append(
            {
                "strategy_family": key[0],
                "candidate_name": key[1],
                "source_selected_rows": metric.get("selected_rows", ""),
                "source_mean_net_proxy": metric.get("mean_net_proxy", ""),
                "source_classification": metric.get("classification", ""),
                "slice_count": len(slices),
                "positive_net_slice_count": sum(
                    1 for row in slices
                    if _is_positive_number(row.get("mean_net_proxy"))
                ),
                "eligible_positive_slice_count": len(eligible),
                "best_slice_family": best_slice.get("slice_family", ""),
                "best_slice_value": best_slice.get("slice_value", ""),
                "best_slice_rows": best_slice.get("selected_rows", ""),
                "best_slice_mean_net_proxy": best_slice.get("mean_net_proxy", ""),
                "best_slice_tail_loss_rate": best_slice.get("tail_loss_rate", ""),
                "refinement_decision": _refinement_decision(eligible, metric),
                "runtime_status": "NO_RUNTIME_EFFECT",
                "promotion_status": "NOT_PROMOTABLE",
                "profitability_status": "NO_PROFIT_CLAIM",
            }
        )
    return output


def _tail_loss_diagnostics(
    slice_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "strategy_family": row["strategy_family"],
            "candidate_name": row["candidate_name"],
            "slice_family": row["slice_family"],
            "slice_value": row["slice_value"],
            "selected_rows": row["selected_rows"],
            "worst_5_net_proxy": row["worst_5_net_proxy"],
            "max_drawdown_proxy": row["max_drawdown_proxy"],
            "tail_loss_rate": row["tail_loss_rate"],
        }
        for row in slice_rows
    ]


def _sample_size_diagnostics(
    slice_rows: Sequence[Mapping[str, Any]],
    min_slice_rows: int,
) -> list[dict[str, Any]]:
    return [
        {
            "strategy_family": row["strategy_family"],
            "candidate_name": row["candidate_name"],
            "slice_family": row["slice_family"],
            "slice_value": row["slice_value"],
            "selected_rows": row["selected_rows"],
            "min_slice_rows": min_slice_rows,
            "sample_size_status": row["sample_size_status"],
        }
        for row in slice_rows
    ]


def _candidate_decisions(
    refined_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "strategy_family": row["strategy_family"],
            "candidate_name": row["candidate_name"],
            "candidate_decision": row["refinement_decision"],
            "best_slice": _format_slice(row),
            "eligible_positive_slice_count": row["eligible_positive_slice_count"],
            "runtime_status": "NO_RUNTIME_EFFECT",
            "promotion_status": "NOT_PROMOTABLE",
            "profitability_status": "NO_PROFIT_CLAIM",
        }
        for row in refined_metrics
    ]


def _recommendation(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    has_watchlist = any(
        row["candidate_decision"] == "REFINED_SLICE_POLICY_WATCHLIST"
        for row in decisions
    )
    recommendation = RECOMMEND_SLICE_POLICY if has_watchlist else RECOMMEND_REFINE
    next_action = NEXT_SLICE_POLICY if has_watchlist else NEXT_REFINE
    return {
        "recommendation": recommendation,
        "next_required_action": next_action,
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _next_actions(recommendation: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action": str(recommendation["next_required_action"]),
            "rationale": "Continue through generic slice tooling, not one-off strategy work.",
        },
        {
            "priority": "2",
            "action": "KEEP_REFINED_CANDIDATES_RESEARCH_ONLY",
            "rationale": "No runtime, registry, promotion, backtest, trading, or profit claim.",
        },
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "strategy_candidate_refinement_report_json": str(
            output_dir / "strategy_candidate_refinement_report.json"
        ),
        "strategy_candidate_refinement_report_md": str(
            output_dir / "strategy_candidate_refinement_report.md"
        ),
        "refined_candidate_metrics_csv": str(output_dir / "refined_candidate_metrics.csv"),
        "slice_diagnostics_csv": str(output_dir / "slice_diagnostics.csv"),
        "tail_loss_diagnostics_csv": str(output_dir / "tail_loss_diagnostics.csv"),
        "sample_size_diagnostics_csv": str(output_dir / "sample_size_diagnostics.csv"),
        "candidate_decisions_csv": str(output_dir / "candidate_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# M20 Strategy Candidate Refinement",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Candidate count: `{report['candidate_count']}`",
        f"- Slice count: `{report['slice_count']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Candidate Decisions",
    ]
    for row in decisions:
        lines.append(
            f"- `{row['strategy_family']}:{row['candidate_name']}` -> "
            f"`{row['candidate_decision']}`"
        )
    lines.extend(
        [
            "",
            "Positive net-proxy slices are research watchlist evidence only; this is not "
            "a backtest, runtime policy, promotion, or profit claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (
        row.get("fold_index", ""),
        row.get("symbol", ""),
        row.get("interval_begin", ""),
    )


def _candidate_key(row: Mapping[str, str]) -> tuple[str, str]:
    return row.get("strategy_family", ""), row.get("candidate_name", "")


def _range_ratio(row: Mapping[str, str]) -> float:
    close = _to_float(row.get("close_price"))
    if close == 0.0:
        return 0.0
    return (_to_float(row.get("high_price")) - _to_float(row.get("low_price"))) / close


def _quantiles(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    low_index = int((len(ordered) - 1) * 0.25)
    high_index = int((len(ordered) - 1) * 0.75)
    return ordered[low_index], ordered[high_index]


def _bucket(value: float, low: float, high: float) -> str:
    if value <= low:
        return "LOW"
    if value >= high:
        return "HIGH"
    return "MID"


def _quarter(timestamp: str) -> str:
    if len(timestamp) < 7:
        return ""
    try:
        month = int(timestamp[5:7])
    except ValueError:
        return ""
    return f"{timestamp[:4]}Q{((month - 1) // 3) + 1}"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _max_drawdown(values: Sequence[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = min(drawdown, cumulative - peak)
    return drawdown


def _tail_loss_rate(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    return sum(1 for value in values if value < 0.0) / len(values)


def _sample_size_status(selected_rows: int, min_slice_rows: int) -> str:
    if selected_rows >= min_slice_rows:
        return "ADEQUATE_SAMPLE"
    if selected_rows > 0:
        return "LOW_SAMPLE"
    return "NO_SAMPLE"


def _slice_decision(selected_rows: int, min_slice_rows: int, mean_net: Any) -> str:
    if selected_rows < min_slice_rows:
        return "REFINED_SLICE_LOW_SAMPLE"
    if mean_net == "":
        return "REFINED_SLICE_ECONOMICS_UNKNOWN"
    if float(mean_net) > 0.0:
        return "REFINED_SLICE_WATCHLIST_POSITIVE_NET_PROXY"
    return "REFINED_SLICE_NEGATIVE_OR_FLAT_NET_PROXY"


def _best_slice(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda row: (
            _to_float(row.get("mean_net_proxy")),
            _to_int(row.get("selected_rows")),
            str(row.get("slice_family", "")),
            str(row.get("slice_value", "")),
        ),
        reverse=True,
    )[0]


def _refinement_decision(
    eligible_slices: Sequence[Mapping[str, Any]],
    source_metric: Mapping[str, str],
) -> str:
    if eligible_slices:
        return "REFINED_SLICE_POLICY_WATCHLIST"
    if _to_float(source_metric.get("mean_net_proxy")) < 0.0:
        return "REFINE_OR_WATCHLIST_NEGATIVE_ECONOMICS"
    return "INSUFFICIENT_REFINEMENT_EVIDENCE"


def _format_slice(row: Mapping[str, Any]) -> str:
    family = row.get("best_slice_family", "")
    value = row.get("best_slice_value", "")
    if not family:
        return ""
    return f"{family}={value}"


def _is_positive_number(value: Any) -> bool:
    if value in ("", None):
        return False
    return _to_float(value) > 0.0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


__all__ = ["analyze_m20_strategy_candidate_refinement"]
