"""Research-only M20 safe economic outcome artifact builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_OUTPUT_NAME = "economic_outcome_artifacts"
DEFAULT_LABEL_FILE = "research_labels/vol_scaled/fee_exceedance_labels_vol_scaled.csv"
DEFAULT_TRIPLE_BARRIER_FILE = "research_labels/vol_scaled/triple_barrier_labels_vol_scaled.csv"
DEFAULT_TRAINING_FEATURES_FILE = "training_frame/m20_training_frame_features.csv"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "EVALUATION_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
MISSING_SOURCE = "MISSING_SAFE_ECONOMIC_SOURCE"
MISSING_MAGNITUDE = "ECONOMIC_MAGNITUDE_NOT_AVAILABLE"


def build_m20_economic_outcome_artifacts(
    *,
    source_run_dir: Path,
    label_source_run_dir: Path | None = None,
    prediction_run_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
    fee_bps: float = 20.0,
    slippage_bps: float = 0.0,
    horizon_candles: int | None = None,
    price_column: str = "close_price",
) -> dict[str, Any]:
    """Build safe evaluation-only economic outcomes from existing artifacts."""
    # pylint: disable=too-many-arguments,too-many-locals
    source_dir = Path(source_run_dir).resolve()
    label_dir = Path(label_source_run_dir).resolve() if label_source_run_dir else source_dir
    prediction_dir = Path(prediction_run_dir).resolve() if prediction_run_dir else None
    output_dir = source_dir / "research_labels" / "vol_scaled" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = _output_files(output_dir)
    resolved_horizon = horizon_candles or _configured_horizon(source_dir)
    candidates = _source_candidates(label_dir)
    schema_audit = [_schema_row(path) for path in candidates]
    label_path = label_dir / DEFAULT_LABEL_FILE
    triple_barrier_path = label_dir / DEFAULT_TRIPLE_BARRIER_FILE
    features_path = label_dir / DEFAULT_TRAINING_FEATURES_FILE

    blockers: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    source_file = ""
    source_kind = ""
    if label_path.exists():
        label_rows = _read_csv(label_path)
        label_columns = set(label_rows[0]) if label_rows else set()
        if "future_return" in label_columns:
            source_file = str(label_path)
            source_kind = "fee_exceedance_labels"
            triple_barrier = _triple_barrier_index(triple_barrier_path)
            rows = _rows_from_future_return_labels(
                _primary_scenario_rows(label_rows),
                triple_barrier=triple_barrier,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                horizon_candles=resolved_horizon,
            )
        else:
            blockers.append(_blocker(MISSING_MAGNITUDE, str(label_path)))
    elif features_path.exists():
        source_file = str(features_path)
        source_kind = "training_frame_prices"
        rows = _rows_from_training_prices(
            _read_csv(features_path),
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            horizon_candles=resolved_horizon,
            price_column=price_column,
        )
        if not rows:
            blockers.append(_blocker(MISSING_MAGNITUDE, str(features_path)))
    else:
        blockers.append(_blocker(MISSING_SOURCE, str(label_dir)))

    economics_computable = bool(rows)
    if not economics_computable and not blockers:
        blockers.append(_blocker(MISSING_MAGNITUDE, str(label_dir)))
    recommendation = _recommendation(economics_computable, blockers)
    manifest = {
        "source_run_dir": str(source_dir),
        "label_source_run_dir": str(label_dir),
        "prediction_run_dir": str(prediction_dir) if prediction_dir else "",
        "output_dir": str(output_dir),
        "source_file": source_file,
        "source_kind": source_kind,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "horizon_candles": resolved_horizon,
        "price_column": price_column,
        "economics_computable": economics_computable,
        "rows_written": len(rows),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    report = {
        "summary": "Safe research-only economic outcome artifact builder.",
        "economics_computable": economics_computable,
        "rows_written": len(rows),
        "source_file": source_file,
        "blockers": [row["blocker"] for row in blockers],
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["economic_outcome_report_json"]), report)
    Path(output_files["economic_outcome_report_md"]).write_text(
        _markdown(report),
        encoding="utf-8",
    )
    if rows:
        write_csv_artifact(Path(output_files["economic_outcomes_csv"]), rows)
    write_csv_artifact(Path(output_files["schema_audit_csv"]), schema_audit)
    write_csv_artifact(Path(output_files["blockers_csv"]), blockers)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "schema_audit": schema_audit,
            "blocker_rows": blockers,
            "recommendation_payload": recommendation,
        }
    )


def _source_candidates(label_dir: Path) -> list[Path]:
    return [
        label_dir / DEFAULT_LABEL_FILE,
        label_dir / DEFAULT_TRIPLE_BARRIER_FILE,
        label_dir / DEFAULT_TRAINING_FEATURES_FILE,
    ]


def _schema_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "columns": "",
            "has_symbol": False,
            "has_interval_begin": False,
            "has_future_return": False,
            "has_price_column": False,
            "has_label": False,
        }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader, [])
    return {
        "path": str(path),
        "exists": True,
        "columns": "|".join(columns),
        "has_symbol": "symbol" in columns,
        "has_interval_begin": "interval_begin" in columns,
        "has_future_return": "future_return" in columns,
        "has_price_column": "close_price" in columns,
        "has_label": "label" in columns,
    }


def _rows_from_future_return_labels(
    label_rows: Sequence[Mapping[str, str]],
    *,
    triple_barrier: Mapping[tuple[str, str], str],
    fee_bps: float,
    slippage_bps: float,
    horizon_candles: int,
) -> list[dict[str, Any]]:
    fee_rate = fee_bps / 10000.0
    slippage_rate = slippage_bps / 10000.0
    rows = []
    for row in label_rows:
        future_return = _to_float(row.get("future_return"))
        net_value = future_return - fee_rate - slippage_rate
        symbol = str(row.get("symbol", ""))
        interval_begin = str(row.get("interval_begin", ""))
        rows.append(
            {
                "symbol": symbol,
                "interval_begin": interval_begin,
                "fold_index": row.get("fold_index", ""),
                "future_return": future_return,
                "gross_value_proxy": future_return,
                "net_value_proxy": net_value,
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "horizon_candles": _to_int(row.get("horizon")) or horizon_candles,
                "direction": "long_only",
                "scenario_name": row.get("scenario_name", ""),
                "fee_exceedance_label": row.get("label", ""),
                "triple_barrier_label": triple_barrier.get((symbol, interval_begin), ""),
            }
        )
    return rows


def _rows_from_training_prices(
    feature_rows: Sequence[Mapping[str, str]],
    *,
    fee_bps: float,
    slippage_bps: float,
    horizon_candles: int,
    price_column: str,
) -> list[dict[str, Any]]:
    # pylint: disable=too-many-locals
    if not feature_rows or price_column not in feature_rows[0]:
        return []
    rows = []
    fee_rate = fee_bps / 10000.0
    slippage_rate = slippage_bps / 10000.0
    for _, symbol_rows in sorted(_group_by(feature_rows, "symbol").items()):
        ordered = sorted(symbol_rows, key=lambda row: str(row.get("interval_begin", "")))
        for index, row in enumerate(ordered):
            future_index = index + horizon_candles
            if future_index >= len(ordered):
                continue
            entry = _to_float(row.get(price_column))
            future = _to_float(ordered[future_index].get(price_column))
            if entry == 0.0:
                continue
            future_return = (future / entry) - 1.0
            rows.append(
                {
                    "symbol": row.get("symbol", ""),
                    "interval_begin": row.get("interval_begin", ""),
                    "fold_index": row.get("fold_index", ""),
                    "future_return": future_return,
                    "gross_value_proxy": future_return,
                    "net_value_proxy": future_return - fee_rate - slippage_rate,
                    "fee_bps": fee_bps,
                    "slippage_bps": slippage_bps,
                    "horizon_candles": horizon_candles,
                    "direction": "long_only",
                    "scenario_name": "",
                    "fee_exceedance_label": "",
                    "triple_barrier_label": "",
                }
            )
    return rows


def _triple_barrier_index(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    return {
        (row.get("symbol", ""), row.get("interval_begin", "")): row.get("label", "")
        for row in _read_csv(path)
    }


def _primary_scenario_rows(
    label_rows: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    if not label_rows or "scenario_name" not in label_rows[0]:
        return list(label_rows)
    current_fee_rows = [
        row for row in label_rows if row.get("scenario_name") == "current_fee"
    ]
    return current_fee_rows or list(label_rows)


def _configured_horizon(source_dir: Path) -> int:
    config_path = source_dir / "run_config.json"
    if not config_path.exists():
        return 3
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 3
    return _to_int(payload.get("label_horizon_candles")) or 3


def _recommendation(
    economics_computable: bool,
    blockers: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    blocker_names = [row["blocker"] for row in blockers]
    if economics_computable:
        recommendation = "RUN_COST_AWARE_SPECIALIST_POLICY_EVALUATOR_WITH_ECONOMIC_OUTCOMES"
        next_action = "RE_RUN_GENERIC_COST_AWARE_SPECIALIST_POLICY_EVALUATOR"
    elif MISSING_MAGNITUDE in blocker_names:
        recommendation = "ADD_RETURN_MAGNITUDE_SOURCE_FOR_ECONOMIC_OUTCOMES"
        next_action = "EXPORT_OR_BUILD_SAFE_FORWARD_RETURN_OUTCOMES"
    else:
        recommendation = "BLOCKED_MISSING_SAFE_ECONOMIC_SOURCE"
        next_action = "IDENTIFY_SAFE_ECONOMIC_SOURCE_ARTIFACT"
    return {
        "recommendation": recommendation,
        "next_required_action": next_action,
        "economics_computable": economics_computable,
        "blockers": blocker_names,
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
            "rationale": "Use safe evaluation-only economic outcomes before policy economics.",
        },
        {
            "priority": "2",
            "action": "KEEP_ECONOMIC_OUTCOMES_OUT_OF_RUNTIME_AND_PREDICTION_EXPORTS",
            "rationale": "Avoid leakage into prediction exports and runtime paths.",
        },
    ]


def _blocker(blocker: str, source: str) -> dict[str, str]:
    return {"blocker": blocker, "source": source}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _group_by(
    rows: Sequence[Mapping[str, str]],
    key: str,
) -> dict[str, list[Mapping[str, str]]]:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped


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


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "economic_outcome_report_json": str(output_dir / "economic_outcome_report.json"),
        "economic_outcome_report_md": str(output_dir / "economic_outcome_report.md"),
        "economic_outcomes_csv": str(output_dir / "economic_outcomes.csv"),
        "schema_audit_csv": str(output_dir / "schema_audit.csv"),
        "blockers_csv": str(output_dir / "blockers.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M20 Economic Outcome Artifacts",
        "",
        f"- Economics computable: `{report['economics_computable']}`",
        f"- Rows written: `{report['rows_written']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Blockers: `{', '.join(report['blockers'])}`",
        "- Status: `RESEARCH_ONLY`, `EVALUATION_ONLY`, `NO_RUNTIME_EFFECT`, "
        "`NOT_BACKTEST`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "No runtime, registry, promotion, training, scoring, prediction export, "
        "backtest, trading, or profit claim behavior was changed.",
        "",
    ]
    return "\n".join(lines)


__all__ = ["build_m20_economic_outcome_artifacts"]
