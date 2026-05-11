"""Generic research-only M20 strategy-conditioned candidate factory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_OUTPUT_NAME = "strategy_candidate_factory"
DEFAULT_ECONOMIC_OUTCOME_NAME = "economic_outcome_artifacts"
DEFAULT_TRAINING_FRAME_DIR = "training_frame"
FEATURE_FILE = "m20_training_frame_features.csv"
FEATURE_COLUMNS_FILE = "m20_training_frame_feature_columns.json"
LABEL_FILE = "research_labels/vol_scaled/fee_exceedance_labels_vol_scaled.csv"
TRIPLE_BARRIER_FILE = "research_labels/vol_scaled/triple_barrier_labels_vol_scaled.csv"
ECONOMIC_OUTCOME_FILE = "economic_outcomes.csv"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
OVERALL_STATUS = HONESTY_FLAGS
ECONOMIC_MISSING = "ECONOMIC_OUTCOMES_NOT_AVAILABLE"
LABELS_MISSING = "LABELS_NOT_AVAILABLE"
FEATURES_MISSING = "BLOCKED_REQUIRED_FEATURES_MISSING"


def build_m20_strategy_candidates(
    *,
    source_run_dir: Path,
    economic_outcome_dir: Path | None = None,
    training_frame_dir: Path | None = None,
    label_source_run_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Build generic strategy-conditioned candidate artifacts."""
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    source_dir = Path(source_run_dir).resolve()
    label_dir = Path(label_source_run_dir).resolve() if label_source_run_dir else source_dir
    frame_dir = (
        Path(training_frame_dir).resolve()
        if training_frame_dir is not None
        else source_dir / DEFAULT_TRAINING_FRAME_DIR
    )
    outcome_dir = (
        Path(economic_outcome_dir).resolve()
        if economic_outcome_dir is not None
        else source_dir / "research_labels" / "vol_scaled" / DEFAULT_ECONOMIC_OUTCOME_NAME
    )
    output_dir = source_dir / "research_labels" / "vol_scaled" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = _output_files(output_dir)
    feature_path = frame_dir / FEATURE_FILE
    if not feature_path.exists():
        raise ValueError(f"Missing training frame features: {feature_path}")

    feature_rows = _read_csv(feature_path)
    feature_columns = _feature_columns(frame_dir / FEATURE_COLUMNS_FILE, feature_rows)
    outcome_index = _outcome_index(outcome_dir / ECONOMIC_OUTCOME_FILE)
    label_index = _label_index(label_dir / LABEL_FILE)
    triple_barrier_index = _triple_barrier_index(label_dir / TRIPLE_BARRIER_FILE)
    economics_available = bool(outcome_index)
    labels_available = bool(label_index)
    base_positive_rate = _base_positive_rate(feature_rows, label_index)
    family_specs = _family_specs(feature_rows, feature_columns)
    feature_family_audit = _feature_family_audit(family_specs, feature_columns)

    candidate_rows: list[dict[str, Any]] = []
    candidate_metrics: list[dict[str, Any]] = []
    by_symbol: list[dict[str, Any]] = []
    by_time: list[dict[str, Any]] = []
    for spec in family_specs:
        if not spec["available"]:
            candidate_metrics.append(_blocked_metric(spec))
            continue
        for candidate in spec["candidates"]:
            selected = [
                _candidate_row(
                    feature_row,
                    spec["strategy_family"],
                    candidate["candidate_name"],
                    candidate["feature_columns"],
                    outcome_index,
                    label_index,
                    triple_barrier_index,
                )
                for feature_row in feature_rows
                if candidate["predicate"](feature_row)
            ]
            candidate_rows.extend(selected)
            metric = _candidate_metric(
                strategy_family=spec["strategy_family"],
                candidate_name=candidate["candidate_name"],
                selected_rows=selected,
                total_rows=len(feature_rows),
                labels_available=labels_available,
                economics_available=economics_available,
                base_positive_rate=base_positive_rate,
            )
            candidate_metrics.append(metric)
            by_symbol.extend(
                _slice_metrics(
                    metric,
                    selected,
                    "symbol",
                    labels_available,
                    economics_available,
                    base_positive_rate,
                )
            )
            by_time.extend(
                _slice_metrics(
                    metric,
                    selected,
                    "month",
                    labels_available,
                    economics_available,
                    base_positive_rate,
                )
            )
            by_time.extend(
                _slice_metrics(
                    metric,
                    selected,
                    "quarter",
                    labels_available,
                    economics_available,
                    base_positive_rate,
                )
            )

    blockers = _blockers(feature_family_audit, economics_available, labels_available)
    candidate_decisions = _candidate_decisions(candidate_metrics)
    recommendation = _recommendation(candidate_decisions, economics_available)
    manifest = {
        "source_run_dir": str(source_dir),
        "training_frame_dir": str(frame_dir),
        "economic_outcome_dir": str(outcome_dir),
        "label_source_run_dir": str(label_dir),
        "feature_path": str(feature_path),
        "feature_columns": list(feature_columns),
        "economics_available": economics_available,
        "labels_available": labels_available,
        "strategy_families": [spec["strategy_family"] for spec in family_specs],
        "candidate_count": len(candidate_metrics),
        "candidate_event_rows": len(candidate_rows),
        "blockers": blockers,
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    report = {
        "summary": "Generic M20 strategy-conditioned candidate factory.",
        "best_candidate": _best_candidate(candidate_decisions),
        "candidate_count": len(candidate_metrics),
        "strategy_families_evaluated": [spec["strategy_family"] for spec in family_specs],
        "economics_available": economics_available,
        "labels_available": labels_available,
        "blockers": blockers,
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "overall_status": list(OVERALL_STATUS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["strategy_candidate_report_json"]), report)
    Path(output_files["strategy_candidate_report_md"]).write_text(
        _markdown(report, candidate_decisions),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["strategy_candidates_csv"]), candidate_rows)
    write_csv_artifact(Path(output_files["candidate_metrics_csv"]), candidate_metrics)
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_csv_artifact(Path(output_files["feature_family_audit_csv"]), feature_family_audit)
    write_csv_artifact(Path(output_files["candidate_decisions_csv"]), candidate_decisions)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "candidate_metrics": candidate_metrics,
            "candidate_decisions": candidate_decisions,
            "feature_family_audit": feature_family_audit,
            "recommendation_payload": recommendation,
        }
    )


def _family_specs(
    rows: Sequence[Mapping[str, str]],
    feature_columns: Sequence[str],
) -> list[dict[str, Any]]:
    columns = set(feature_columns)
    range_values = [_range_ratio(row) for row in rows]
    range_low, range_high = _quantiles(range_values)
    vol_low, vol_high = _quantiles(_column_values(rows, "realized_vol_12"))
    volume_low, volume_high = _quantiles(_column_values(rows, "volume"))
    specs = [
        _spec(
            "macd_momentum",
            ("macd_line_12_26",),
            columns,
            [
                _candidate(
                    "macd_positive",
                    ("macd_line_12_26",),
                    lambda row: _to_float(row.get("macd_line_12_26")) > 0.0,
                ),
                _candidate(
                    "macd_negative",
                    ("macd_line_12_26",),
                    lambda row: _to_float(row.get("macd_line_12_26")) < 0.0,
                ),
                _candidate(
                    "macd_near_zero",
                    ("macd_line_12_26",),
                    lambda row: abs(_to_float(row.get("macd_line_12_26"))) <= 1.0,
                ),
            ],
        ),
        _spec(
            "rsi_mean_reversion",
            ("rsi_14",),
            columns,
            [
                _candidate(
                    "rsi_oversold",
                    ("rsi_14",),
                    lambda row: _to_float(row.get("rsi_14")) < 30.0,
                ),
                _candidate(
                    "rsi_overbought",
                    ("rsi_14",),
                    lambda row: _to_float(row.get("rsi_14")) > 70.0,
                ),
                _candidate(
                    "rsi_neutral",
                    ("rsi_14",),
                    lambda row: 40.0 <= _to_float(row.get("rsi_14")) <= 60.0,
                ),
            ],
        ),
        _spec(
            "range_compression",
            ("high_price", "low_price", "close_price"),
            columns,
            [
                _candidate(
                    "range_low",
                    ("high_price", "low_price", "close_price"),
                    lambda row: _range_ratio(row) <= range_low,
                ),
                _candidate(
                    "range_high",
                    ("high_price", "low_price", "close_price"),
                    lambda row: _range_ratio(row) >= range_high,
                ),
            ],
        ),
        _spec(
            "volatility_state",
            ("realized_vol_12",),
            columns,
            [
                _candidate(
                    "realized_vol_low",
                    ("realized_vol_12",),
                    lambda row: _to_float(row.get("realized_vol_12")) <= vol_low,
                ),
                _candidate(
                    "realized_vol_high",
                    ("realized_vol_12",),
                    lambda row: _to_float(row.get("realized_vol_12")) >= vol_high,
                ),
            ],
        ),
        _spec(
            "return_reversal",
            ("log_return_1",),
            columns,
            [
                _candidate(
                    "return_positive",
                    ("log_return_1",),
                    lambda row: _to_float(row.get("log_return_1")) > 0.0,
                ),
                _candidate(
                    "return_negative",
                    ("log_return_1",),
                    lambda row: _to_float(row.get("log_return_1")) < 0.0,
                ),
                _candidate("reversal_proxy", ("log_return_1", "momentum_3"), _reversal_proxy),
            ],
        ),
        _spec(
            "volume_context",
            ("volume",),
            columns,
            [
                _candidate(
                    "volume_low",
                    ("volume",),
                    lambda row: _to_float(row.get("volume")) <= volume_low,
                ),
                _candidate(
                    "volume_high",
                    ("volume",),
                    lambda row: _to_float(row.get("volume")) >= volume_high,
                ),
            ],
        ),
    ]
    return specs


def _spec(
    strategy_family: str,
    required_features: Sequence[str],
    columns: set[str],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    missing = [column for column in required_features if column not in columns]
    filtered_candidates = [
        candidate for candidate in candidates
        if all(column in columns for column in candidate["feature_columns"])
    ]
    return {
        "strategy_family": strategy_family,
        "required_features": tuple(required_features),
        "missing_features": tuple(missing),
        "available": not missing,
        "candidates": tuple(filtered_candidates),
    }


def _candidate(
    name: str,
    feature_columns: Sequence[str],
    predicate: Callable[[Mapping[str, str]], bool],
) -> dict[str, Any]:
    return {
        "candidate_name": name,
        "feature_columns": tuple(feature_columns),
        "predicate": predicate,
    }


def _candidate_row(
    feature_row: Mapping[str, str],
    strategy_family: str,
    candidate_name: str,
    feature_columns: Sequence[str],
    outcome_index: Mapping[tuple[str, str, str], Mapping[str, str]],
    label_index: Mapping[tuple[str, str, str], Mapping[str, str]],
    triple_barrier_index: Mapping[tuple[str, str, str], str],
) -> dict[str, Any]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    symbol = str(feature_row.get("symbol", ""))
    interval_begin = str(feature_row.get("interval_begin", ""))
    fold_index = str(feature_row.get("fold_index", ""))
    key = (fold_index, symbol, interval_begin)
    outcome = outcome_index.get(key, {})
    label = label_index.get(key, {})
    return {
        "strategy_family": strategy_family,
        "candidate_name": candidate_name,
        "symbol": symbol,
        "interval_begin": interval_begin,
        "fold_index": fold_index,
        "setup_passed": True,
        "feature_snapshot_json": json.dumps(
            {column: feature_row.get(column, "") for column in feature_columns},
            sort_keys=True,
        ),
        "fee_exceedance_label": outcome.get("fee_exceedance_label", label.get("label", "")),
        "triple_barrier_label": outcome.get(
            "triple_barrier_label",
            triple_barrier_index.get(key, ""),
        ),
        "gross_value_proxy": outcome.get("gross_value_proxy", ""),
        "net_value_proxy": outcome.get("net_value_proxy", ""),
    }


def _candidate_metric(
    *,
    strategy_family: str,
    candidate_name: str,
    selected_rows: Sequence[Mapping[str, Any]],
    total_rows: int,
    labels_available: bool,
    economics_available: bool,
    base_positive_rate: float,
) -> dict[str, Any]:
    # pylint: disable=too-many-arguments
    selected_count = len(selected_rows)
    positive_count = sum(_to_int(row.get("fee_exceedance_label")) for row in selected_rows)
    selected_rate = positive_count / selected_count if selected_count else 0.0
    economics = _economic_metrics(selected_rows) if economics_available else _empty_economics()
    return {
        "strategy_family": strategy_family,
        "candidate_name": candidate_name,
        "total_rows": total_rows,
        "selected_rows": selected_count,
        "coverage": selected_count / total_rows if total_rows else 0.0,
        "base_positive_rate": base_positive_rate if labels_available else "",
        "selected_positive_rate": selected_rate if labels_available else "",
        "lift_vs_base": (
            selected_rate / base_positive_rate
            if labels_available and base_positive_rate > 0
            else ""
        ),
        "classification": _classify_candidate(selected_count, economics_available, economics),
        **economics,
    }


def _slice_metrics(
    base_metric: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    slice_family: str,
    labels_available: bool,
    economics_available: bool,
    base_positive_rate: float,
) -> list[dict[str, Any]]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    output = []
    key = "symbol" if slice_family == "symbol" else slice_family
    for value, rows in sorted(_group_by(selected_rows, key).items()):
        metric = _candidate_metric(
            strategy_family=str(base_metric["strategy_family"]),
            candidate_name=str(base_metric["candidate_name"]),
            selected_rows=rows,
            total_rows=len(selected_rows),
            labels_available=labels_available,
            economics_available=economics_available,
            base_positive_rate=base_positive_rate,
        )
        output.append({"slice_family": slice_family, "slice_value": value, **metric})
    return output


def _candidate_decisions(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "strategy_family": row["strategy_family"],
            "candidate_name": row["candidate_name"],
            "candidate_decision": row["classification"],
            "selected_rows": row["selected_rows"],
            "coverage": row["coverage"],
            "mean_net_proxy": row["mean_net_proxy"],
            "cumulative_net_proxy": row["cumulative_net_proxy"],
            "runtime_status": "NO_RUNTIME_EFFECT",
            "promotion_status": "NOT_PROMOTABLE",
            "profitability_status": "NO_PROFIT_CLAIM",
        }
        for row in metrics
    ]


def _classify_candidate(
    selected_count: int,
    economics_available: bool,
    economics: Mapping[str, Any],
) -> str:
    if selected_count == 0:
        return "STRATEGY_CANDIDATE_INSUFFICIENT_EVIDENCE"
    if not economics_available:
        return "STRATEGY_CANDIDATE_SIGNAL_ONLY_ECONOMICS_UNKNOWN"
    mean_net = economics.get("mean_net_proxy")
    if mean_net in ("", None):
        return "STRATEGY_CANDIDATE_INSUFFICIENT_EVIDENCE"
    if float(mean_net) > 0.0:
        return "STRATEGY_CANDIDATE_ECONOMICALLY_PROMISING_RESEARCH_ONLY"
    if float(mean_net) < 0.0:
        return "STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE"
    return "STRATEGY_CANDIDATE_WATCHLIST"


def _economic_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    net_values = [
        _to_float(row.get("net_value_proxy")) for row in rows
        if row.get("net_value_proxy") not in ("", None)
    ]
    if not net_values:
        return _empty_economics()
    return {
        "mean_net_proxy": _mean(net_values),
        "cumulative_net_proxy": sum(net_values),
        "max_drawdown_proxy": _max_drawdown(net_values),
        "win_rate_proxy": sum(1 for value in net_values if value > 0.0) / len(net_values),
    }


def _empty_economics() -> dict[str, Any]:
    return {
        "mean_net_proxy": "",
        "cumulative_net_proxy": "",
        "max_drawdown_proxy": "",
        "win_rate_proxy": "",
    }


def _feature_family_audit(
    specs: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> list[dict[str, Any]]:
    columns = set(feature_columns)
    return [
        {
            "strategy_family": spec["strategy_family"],
            "required_features": "|".join(spec["required_features"]),
            "missing_features": "|".join(spec["missing_features"]),
            "status": "AVAILABLE" if spec["available"] else FEATURES_MISSING,
            "available_feature_count": sum(
                1 for column in spec["required_features"] if column in columns
            ),
        }
        for spec in specs
    ]


def _blocked_metric(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strategy_family": spec["strategy_family"],
        "candidate_name": "",
        "total_rows": 0,
        "selected_rows": 0,
        "coverage": 0.0,
        "base_positive_rate": "",
        "selected_positive_rate": "",
        "lift_vs_base": "",
        "classification": "STRATEGY_CANDIDATE_BLOCKED_MISSING_FEATURES",
        **_empty_economics(),
    }


def _blockers(
    audit_rows: Sequence[Mapping[str, Any]],
    economics_available: bool,
    labels_available: bool,
) -> list[str]:
    blockers = []
    if not economics_available:
        blockers.append(ECONOMIC_MISSING)
    if not labels_available:
        blockers.append(LABELS_MISSING)
    if any(row["status"] == FEATURES_MISSING for row in audit_rows):
        blockers.append(FEATURES_MISSING)
    return blockers


def _recommendation(
    decisions: Sequence[Mapping[str, Any]],
    economics_available: bool,
) -> dict[str, Any]:
    if not economics_available:
        recommendation = "BUILD_OR_LINK_SAFE_ECONOMIC_OUTCOMES"
        next_action = "BUILD_SAFE_ECONOMIC_OUTCOME_ARTIFACTS"
    elif any(
        row["candidate_decision"] == "STRATEGY_CANDIDATE_ECONOMICALLY_PROMISING_RESEARCH_ONLY"
        for row in decisions
    ):
        recommendation = "EVALUATE_STRATEGY_CANDIDATES_WITH_GENERIC_MODEL_FACTORY"
        next_action = "DESIGN_GENERIC_STRATEGY_MODEL_FACTORY"
    else:
        recommendation = "WATCHLIST_OR_REFINE_STRATEGY_CANDIDATE_DEFINITIONS"
        next_action = "REFINE_STRATEGY_CANDIDATE_DEFINITIONS"
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
            "rationale": "Continue through generic strategy tooling, not one-off model paths.",
        },
        {
            "priority": "2",
            "action": "KEEP_STRATEGY_CANDIDATES_RESEARCH_ONLY",
            "rationale": "No runtime, registry, promotion, backtest, trading, or profit claim.",
        },
    ]


def _best_candidate(decisions: Sequence[Mapping[str, Any]]) -> str:
    ranked = sorted(
        decisions,
        key=lambda row: (
            row["candidate_decision"] == "STRATEGY_CANDIDATE_ECONOMICALLY_PROMISING_RESEARCH_ONLY",
            _to_float(row.get("mean_net_proxy")),
            _to_float(row.get("coverage")),
            str(row["candidate_name"]),
        ),
        reverse=True,
    )
    if not ranked:
        return ""
    return f"{ranked[0]['strategy_family']}:{ranked[0]['candidate_name']}"


def _feature_columns(path: Path, rows: Sequence[Mapping[str, str]]) -> list[str]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload.get("feature_columns", []))
    if not rows:
        return []
    return [
        column for column in rows[0]
        if column not in ("symbol", "interval_begin", "fold_index", "row_id")
    ]


def _outcome_index(path: Path) -> dict[tuple[str, str, str], Mapping[str, str]]:
    if not path.exists():
        return {}
    return {
        (row.get("fold_index", ""), row.get("symbol", ""), row.get("interval_begin", "")): row
        for row in _read_csv(path)
    }


def _label_index(path: Path) -> dict[tuple[str, str, str], Mapping[str, str]]:
    if not path.exists():
        return {}
    rows = [
        row for row in _read_csv(path)
        if row.get("scenario_name", "current_fee") == "current_fee"
    ]
    return {
        (row.get("fold_index", ""), row.get("symbol", ""), row.get("interval_begin", "")): row
        for row in rows
    }


def _base_positive_rate(
    feature_rows: Sequence[Mapping[str, str]],
    label_index: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> float:
    labels = []
    for row in feature_rows:
        key = (
            row.get("fold_index", ""),
            row.get("symbol", ""),
            row.get("interval_begin", ""),
        )
        label = label_index.get(key)
        if label is not None:
            labels.append(_to_int(label.get("label")))
    return sum(labels) / len(labels) if labels else 0.0


def _triple_barrier_index(path: Path) -> dict[tuple[str, str, str], str]:
    if not path.exists():
        return {}
    return {
        (
            row.get("fold_index", ""),
            row.get("symbol", ""),
            row.get("interval_begin", ""),
        ): row.get("label", "")
        for row in _read_csv(path)
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _column_values(rows: Sequence[Mapping[str, str]], column: str) -> list[float]:
    return [_to_float(row.get(column)) for row in rows if row.get(column) not in ("", None)]


def _quantiles(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    low_index = int((len(ordered) - 1) * 0.25)
    high_index = int((len(ordered) - 1) * 0.75)
    return ordered[low_index], ordered[high_index]


def _range_ratio(row: Mapping[str, str]) -> float:
    close = _to_float(row.get("close_price"))
    if close == 0.0:
        return 0.0
    return (_to_float(row.get("high_price")) - _to_float(row.get("low_price"))) / close


def _reversal_proxy(row: Mapping[str, str]) -> bool:
    return _to_float(row.get("log_return_1")) * _to_float(row.get("momentum_3")) < 0.0


def _group_by(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if key == "month":
            value = str(row.get("interval_begin", ""))[:7]
        elif key == "quarter":
            value = _quarter(str(row.get("interval_begin", "")))
        else:
            value = str(row.get(key, ""))
        grouped.setdefault(value, []).append(row)
    return grouped


def _max_drawdown(values: Sequence[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = min(drawdown, cumulative - peak)
    return drawdown


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def _quarter(timestamp: str) -> str:
    if len(timestamp) < 7:
        return ""
    try:
        month = int(timestamp[5:7])
    except ValueError:
        return ""
    return f"{timestamp[:4]}Q{((month - 1) // 3) + 1}"


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "strategy_candidate_report_json": str(output_dir / "strategy_candidate_report.json"),
        "strategy_candidate_report_md": str(output_dir / "strategy_candidate_report.md"),
        "strategy_candidates_csv": str(output_dir / "strategy_candidates.csv"),
        "candidate_metrics_csv": str(output_dir / "candidate_metrics.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "feature_family_audit_csv": str(output_dir / "feature_family_audit.csv"),
        "candidate_decisions_csv": str(output_dir / "candidate_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# M20 Strategy Candidate Factory",
        "",
        f"- Best candidate: `{report['best_candidate']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Economics available: `{report['economics_available']}`",
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
            "Existing artifacts only. No runtime, registry, promotion, training, scoring, "
            "backtest, trading, or profit claim was added.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["build_m20_strategy_candidates"]
