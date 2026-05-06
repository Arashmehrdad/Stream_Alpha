"""Research-only M20 range mean-reversion setup diagnostics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "range_mean_reversion_research"
SCENARIO_NAME = "current_fee"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "DIAGNOSTIC_ONLY",
    "NOT_BACKTEST",
    "NO_RUNTIME",
    "NO_REGISTRY",
    "NO_PROMOTION",
    "NO_PROFIT_CLAIM",
)
REQUIRED_FEATURES = ("macd_line_12_26", "log_return_1", "close_price")


def analyze_m20_range_mean_reversion(*, base_run_dir: Path) -> dict[str, Any]:
    """Analyze momentum-breakout setup diagnostics across available M20 windows."""
    # pylint: disable=too-many-locals
    base_dir = Path(base_run_dir).resolve()
    output_dir = base_dir / "research_labels" / "vol_scaled" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = _output_files(output_dir)
    runs = _runs(base_dir)
    missing = _missing_required_features(base_dir)
    if missing:
        return _write_blocker(base_dir, output_files, missing)

    rank_gate_keys = _rank_gate_keys(base_dir)
    setup_metrics: list[dict[str, Any]] = []
    rank_gate_overlap: list[dict[str, Any]] = []
    for run_label, run_dir in runs:
        rows = _joined_rows(run_dir)
        setup_specs = _setup_specs(rows)
        for setup_name, predicate in setup_specs:
            metrics = _metric_row(run_label, setup_name, rows, predicate)
            setup_metrics.append(metrics)
            rank_gate_overlap.append(
                _overlap_row(run_label, setup_name, rows, predicate, rank_gate_keys)
            )
    by_run = _by_group(setup_metrics, "run_label")
    by_symbol = _slice_metrics(runs, "symbol")
    by_time = _slice_metrics(runs, "month") + _slice_metrics(runs, "quarter")
    recommendation = _recommendation(setup_metrics)
    report = {
        "family_id": "range_mean_reversion",
        "run_count": len(runs),
        "setup_count": len({row["setup_name"] for row in setup_metrics}),
        "recommendation": recommendation["recommendation"],
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": recommendation["blockers"],
        "output_files": output_files,
    }
    manifest = {
        "base_run_dir": str(base_dir),
        "output_dir": str(output_dir),
        "source_runs": [{"run_label": label, "run_dir": str(path)} for label, path in runs],
        "strategy_family": "range_mean_reversion",
        "rank_gate_usage": "OVERLAP_DIAGNOSTIC_ONLY",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _markdown(report, setup_metrics),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["setup_metrics_csv"]), setup_metrics)
    write_csv_artifact(Path(output_files["by_run_csv"]), by_run)
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_csv_artifact(Path(output_files["rank_gate_overlap_csv"]), rank_gate_overlap)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(report)


def _write_blocker(
    base_dir: Path,
    output_files: Mapping[str, str],
    missing: Sequence[str],
) -> dict[str, Any]:
    recommendation = {
        "recommendation": "BLOCKED_REQUIRED_FEATURES_MISSING",
        "missing_required_features": list(missing),
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": ["REQUIRED_FEATURES_MISSING"],
    }
    report = {
        "family_id": "range_mean_reversion",
        "base_run_dir": str(base_dir),
        "recommendation": recommendation["recommendation"],
        "missing_required_features": list(missing),
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": recommendation["blockers"],
        "output_files": dict(output_files),
    }
    write_json_artifact(Path(output_files["manifest_json"]), report)
    write_json_artifact(Path(output_files["report_json"]), report)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    Path(output_files["report_md"]).write_text(
        "# M20 Range Mean-Reversion Research\n\nBlocked: required features are missing.\n",
        encoding="utf-8",
    )
    return make_json_safe(report)


def _runs(base_dir: Path) -> list[tuple[str, Path]]:
    packet_path = (
        base_dir
        / "research_labels"
        / "vol_scaled"
        / "rank_gate_evidence_packet"
        / "rank_gate_evidence_packet.json"
    )
    if not packet_path.exists():
        return [("base", base_dir)]
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    return [
        (str(row.get("window_label", f"run_{index}")), Path(str(row.get("run_dir"))).resolve())
        for index, row in enumerate(packet.get("window_metrics", []))
        if row.get("run_dir")
    ]


def _missing_required_features(base_dir: Path) -> list[str]:
    path = base_dir / "training_frame" / "m20_training_frame_feature_columns.json"
    if not path.exists():
        return list(REQUIRED_FEATURES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = payload.get("feature_columns", []) if isinstance(payload, dict) else payload
    available = {str(column) for column in columns}
    return [column for column in REQUIRED_FEATURES if column not in available]


def _joined_rows(run_dir: Path) -> list[dict[str, Any]]:
    features = {
        _key(row): row
        for row in _read_csv(
            run_dir / "training_frame" / "m20_training_frame_features.csv"
        )
    }
    labels = {
        _key(row): row
        for row in _read_csv(
            run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_labels_vol_scaled.csv"
        )
        if row.get("scenario_name") == SCENARIO_NAME
    }
    rows = []
    for key, feature in features.items():
        label = labels.get(key)
        if not label:
            continue
        timestamp = str(feature.get("interval_begin", ""))
        month = timestamp[:7]
        rows.append(
            {
                **feature,
                "label": _int(label.get("label")),
                "month": month,
                "quarter": _quarter(month),
                "range_pct": _range_pct(feature),
            }
        )
    return rows


def _setup_specs(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, Callable[[Mapping[str, Any]], bool]]]:
    vol_mid, vol_high = _tertiles([_float(row.get("realized_vol_12")) for row in rows])
    volume_mid, _volume_high = _tertiles([_float(row.get("volume")) for row in rows])
    range_mid, _range_high = _tertiles([_float(row.get("range_pct")) for row in rows])
    macd_near_zero = max(
        _quantile([abs(_float(row.get("macd_line_12_26"))) for row in rows], 1 / 3),
        0.000001,
    )
    return [
        ("range_low", lambda row: _float(row.get("range_pct")) <= range_mid),
        ("realized_vol_low", lambda row: _float(row.get("realized_vol_12")) <= vol_mid),
        (
            "realized_vol_mid",
            lambda row: vol_mid < _float(row.get("realized_vol_12")) < vol_high,
        ),
        ("volume_low", lambda row: _float(row.get("volume")) <= volume_mid),
        (
            "macd_near_zero",
            lambda row: abs(_float(row.get("macd_line_12_26"))) <= macd_near_zero,
        ),
        ("return_reversal_proxy", _return_reversal_proxy),
        ("range_mean_reversion_proxy", lambda row: (
            _float(row.get("range_pct")) <= range_mid
            and _float(row.get("realized_vol_12")) <= vol_high
            and abs(_float(row.get("macd_line_12_26"))) <= macd_near_zero
        )),
    ]


def _metric_row(
    run_label: str,
    setup_name: str,
    rows: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    setup_rows = [row for row in rows if predicate(row)]
    positives = sum(_int(row.get("label")) for row in rows)
    setup_positives = sum(_int(row.get("label")) for row in setup_rows)
    base_rate = positives / len(rows) if rows else 0.0
    positive_rate = setup_positives / len(setup_rows) if setup_rows else 0.0
    return {
        "run_label": run_label,
        "setup_name": setup_name,
        "total_rows": len(rows),
        "setup_rows": len(setup_rows),
        "setup_frequency": len(setup_rows) / len(rows) if rows else 0.0,
        "positive_count": setup_positives,
        "base_positive_rate": base_rate,
        "setup_positive_rate": positive_rate,
        "lift_vs_base": positive_rate / base_rate if base_rate else 0.0,
    }


def _overlap_row(
    run_label: str,
    setup_name: str,
    rows: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
    rank_gate_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    setup_rows = [row for row in rows if predicate(row)]
    overlap = [row for row in setup_rows if _rank_gate_key(row) in rank_gate_keys]
    return {
        "run_label": run_label,
        "setup_name": setup_name,
        "setup_rows": len(setup_rows),
        "rank_gate_overlap_rows": len(overlap),
        "rank_gate_overlap_rate": len(overlap) / len(setup_rows) if setup_rows else 0.0,
        "overlap_positive_rate": (
            sum(_int(row.get("label")) for row in overlap) / len(overlap)
            if overlap else 0.0
        ),
    }


def _slice_metrics(
    runs: Sequence[tuple[str, Path]],
    column: str,
) -> list[dict[str, Any]]:
    output = []
    for run_label, run_dir in runs:
        rows = _joined_rows(run_dir)
        for value in sorted({str(row.get(column, "")) for row in rows}):
            group = [row for row in rows if str(row.get(column, "")) == value]
            positives = sum(_int(row.get("label")) for row in group)
            output.append(
                {
                    "run_label": run_label,
                    "slice_family": column,
                    "slice_value": value,
                    "rows": len(group),
                    "positive_rate": positives / len(group) if group else 0.0,
                }
            )
    return output


def _by_group(rows: Sequence[Mapping[str, Any]], column: str) -> list[dict[str, Any]]:
    output = []
    for value in sorted({str(row.get(column, "")) for row in rows}):
        group = [row for row in rows if str(row.get(column, "")) == value]
        output.append(
            {
                "group": value,
                "setup_count": len(group),
                "best_lift": max((_float(row.get("lift_vs_base")) for row in group), default=0.0),
                "best_setup": (
                    max(group, key=lambda row: _float(row.get("lift_vs_base")))["setup_name"]
                    if group else ""
                ),
            }
        )
    return output


def _recommendation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stable = []
    for setup_name in sorted({row["setup_name"] for row in rows}):
        setup_rows = [row for row in rows if row["setup_name"] == setup_name]
        if setup_rows and min(_float(row.get("lift_vs_base")) for row in setup_rows) > 1.05:
            stable.append(setup_name)
    recommendation = (
        "KEEP_RANGE_MEAN_REVERSION_AS_RESEARCH_DIAGNOSTIC_CANDIDATE"
        if stable else "NO_STABLE_RANGE_MEAN_REVERSION_SETUP_FOUND"
    )
    return {
        "recommendation": recommendation,
        "stable_setup_candidates": stable,
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": ["DIAGNOSTIC_ONLY", "NOT_BACKTEST", "NO_PROFIT_CLAIM"],
    }


def _rank_gate_keys(base_dir: Path) -> set[tuple[str, str]]:
    path = (
        base_dir
        / "research_labels"
        / "vol_scaled"
        / "rank_gate_net_diagnostics"
        / "selected_row_diagnostics.csv"
    )
    return {_rank_gate_key(row) for row in _read_csv(path)} if path.exists() else set()


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "setup_metrics_csv": str(output_dir / "setup_metrics.csv"),
        "by_run_csv": str(output_dir / "by_run.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "rank_gate_overlap_csv": str(output_dir / "rank_gate_overlap.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]) -> str:
    best = max(metrics, key=lambda row: _float(row.get("lift_vs_base")), default={})
    return "\n".join(
        [
            "# M20 Range Mean-Reversion Research Diagnostic",
            "",
            f"- Recommendation: `{report['recommendation']}`",
            f"- Best observed setup: `{best.get('setup_name', '')}`",
            f"- Honesty flags: `{', '.join(HONESTY_FLAGS)}`",
            "",
            "This is setup diagnostics only, not a backtest or trading logic.",
            "",
        ]
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("symbol", "")),
        str(row.get("interval_begin", "")),
        str(row.get("fold_index", "")),
    )



def _rank_gate_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("symbol", "")),
        str(row.get("interval_begin", "")),
    )

def _range_pct(row: Mapping[str, Any]) -> float:
    close = _float(row.get("close_price"))
    return (_float(row.get("high_price")) - _float(row.get("low_price"))) / close if close else 0.0


def _momentum(row: Mapping[str, Any]) -> float:
    return _float(row.get("momentum_3")) or _float(row.get("log_return_1"))


def _return_reversal_proxy(row: Mapping[str, Any]) -> bool:
    return _float(row.get("log_return_1")) * _momentum(row) < 0.0


def _quarter(month: str) -> str:
    try:
        year, month_number = month.split("-")
        return f"{year}Q{((int(month_number) - 1) // 3) + 1}"
    except (ValueError, IndexError):
        return ""


def _tertiles(values: Sequence[float]) -> tuple[float, float]:
    return (_quantile(values, 1 / 3), _quantile(values, 2 / 3))


def _quantile(values: Sequence[float], quantile: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return 0.0
    index = min(len(finite) - 1, max(0, int((len(finite) - 1) * quantile)))
    return finite[index]


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


def read_range_csv(path: Path) -> list[dict[str, str]]:
    """Read momentum diagnostic CSV for tests."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
