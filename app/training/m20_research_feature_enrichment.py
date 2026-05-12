"""Research-only M20 feature enrichment artifact builder."""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.regime.dataset import RegimeSourceRow
from app.regime.service import SymbolThresholds, classify_row
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_OUTPUT_NAME = "research_feature_enrichment"
FEATURE_FILE = "training_frame/m20_training_frame_features.csv"
FEATURE_COLUMNS_FILE = "training_frame/m20_training_frame_feature_columns.json"
SAFE_FEATURE_AUDIT_DIR = "research_labels/vol_scaled/safe_feature_availability"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "EVALUATION_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
KEY_COLUMNS = ("symbol", "interval_begin", "fold_index", "row_id")
REGIME_INPUTS = ("realized_vol_12", "momentum_3", "macd_line_12_26")
ADX_INPUTS = ("high_price", "low_price", "close_price")
ADX_PERIOD = 14


def build_m20_research_feature_enrichment(
    *,
    source_run_dir: Path,
    regime_thresholds_path: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Build research-only enriched features without mutating the training frame."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    output_dir = source_dir / "research_labels" / "vol_scaled" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = source_dir / FEATURE_FILE
    feature_columns_path = source_dir / FEATURE_COLUMNS_FILE
    safe_audit_dir = source_dir / SAFE_FEATURE_AUDIT_DIR
    thresholds_path = Path(regime_thresholds_path).resolve()
    feature_rows = _read_csv(feature_path)
    original_columns = _feature_columns(feature_columns_path, feature_rows)
    threshold_payload = _optional_json(thresholds_path)
    thresholds = _thresholds_by_symbol(threshold_payload)
    blocked = _blocked_features(original_columns, thresholds)
    regime_available = not any(row["feature_name"] == "regime_label" for row in blocked)
    adx_available = not any(row["feature_name"] == "adx_14" for row in blocked)
    regime_values = (
        _regime_values(feature_rows, thresholds, thresholds_path) if regime_available else {}
    )
    adx_values = _adx_values(feature_rows) if adx_available else {}
    enriched_rows = _enriched_rows(
        feature_rows,
        regime_values=regime_values,
        adx_values=adx_values,
        include_regime=regime_available,
        include_adx=adx_available,
    )
    added_features = _added_features(regime_available=regime_available, adx_available=adx_available)
    feature_columns = list(original_columns) + added_features
    output_files = _output_files(output_dir)
    lineage = _feature_lineage(
        feature_path=feature_path,
        safe_audit_dir=safe_audit_dir,
        thresholds_path=thresholds_path,
        threshold_payload=threshold_payload,
        blocked=blocked,
    )
    leakage = _leakage_audit(lineage)
    recommendation = _recommendation(added_features, blocked)
    report = {
        "source_run_dir": str(source_dir),
        "feature_path": str(feature_path),
        "safe_feature_audit_dir": str(safe_audit_dir),
        "regime_thresholds_path": str(thresholds_path),
        "rows_written": len(enriched_rows),
        "original_feature_count": len(original_columns),
        "added_features": added_features,
        "blocked_feature_count": len(blocked),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "source_run_dir": str(source_dir),
        "source_artifacts": {
            "training_frame_features": str(feature_path),
            "training_frame_feature_columns": str(feature_columns_path),
            "safe_feature_availability": str(safe_audit_dir),
            "regime_thresholds": str(thresholds_path),
        },
        "rows_written": len(enriched_rows),
        "feature_columns": feature_columns,
        "added_features": added_features,
        "blocked_features": [row["feature_name"] for row in blocked],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    _write_outputs(
        {
            "output_files": output_files,
            "manifest": manifest,
            "report": report,
            "rows": enriched_rows,
            "feature_columns": feature_columns,
            "lineage": lineage,
            "leakage": leakage,
            "blocked": blocked,
            "recommendation": recommendation,
        }
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "feature_lineage": lineage,
            "leakage_audit": leakage,
            "blocked_features": blocked,
            "recommendation_payload": recommendation,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Missing M20 training-frame features: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _feature_columns(path: Path, rows: Sequence[Mapping[str, str]]) -> list[str]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [str(column) for column in payload.get("feature_columns", [])]
    if not rows:
        return []
    return [column for column in rows[0] if column not in KEY_COLUMNS]


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _thresholds_by_symbol(payload: Mapping[str, Any]) -> dict[str, SymbolThresholds]:
    if payload.get("schema_version") != "m8_thresholds_v1":
        return {}
    threshold_payload = payload.get("thresholds_by_symbol", {})
    if not isinstance(threshold_payload, dict):
        return {}
    output = {}
    for symbol, row in threshold_payload.items():
        output[str(symbol)] = SymbolThresholds(
            symbol=str(row["symbol"]),
            fitted_row_count=int(row["fitted_row_count"]),
            high_vol_threshold=float(row["high_vol_threshold"]),
            trend_abs_threshold=float(row["trend_abs_threshold"]),
        )
    return output


def _blocked_features(
    original_columns: Sequence[str],
    thresholds: Mapping[str, SymbolThresholds],
) -> list[dict[str, str]]:
    columns = set(original_columns)
    blocked = []
    missing_regime = sorted(set(REGIME_INPUTS) - columns)
    if missing_regime or not thresholds:
        blocked.append(
            {
                "feature_name": "regime_label",
                "reason": "|".join(missing_regime or ["M8_THRESHOLDS_NOT_AVAILABLE"]),
                "status": "BLOCKED",
            }
        )
    missing_adx = sorted(set(ADX_INPUTS) - columns)
    if missing_adx:
        blocked.append(
            {
                "feature_name": "adx_14",
                "reason": "|".join(missing_adx),
                "status": "BLOCKED",
            }
        )
    return blocked


def _regime_values(
    rows: Sequence[Mapping[str, str]],
    thresholds: Mapping[str, SymbolThresholds],
    thresholds_path: Path,
) -> dict[tuple[str, str, str], str]:
    output = {}
    for row in rows:
        source_row = RegimeSourceRow(
            symbol=str(row["symbol"]),
            interval_begin=_parse_time(row["interval_begin"]),
            as_of_time=_parse_time(row.get("interval_begin", "")),
            realized_vol_12=float(row["realized_vol_12"]),
            momentum_3=float(row["momentum_3"]),
            macd_line_12_26=float(row["macd_line_12_26"]),
        )
        try:
            output[_key(row)] = classify_row(source_row, dict(thresholds))
        except ValueError:
            output[_key(row)] = ""
    if not thresholds_path.exists():
        return {}
    return output


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _adx_values(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str], str]:
    output: dict[tuple[str, str, str], str] = {}
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["symbol"])].append(row)
    for symbol_rows in grouped.values():
        ordered = sorted(symbol_rows, key=lambda row: row["interval_begin"])
        output.update(_symbol_adx_values(ordered))
    return output


def _symbol_adx_values(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str], str]:
    tr_values: list[float] = []
    plus_dm_values: list[float] = []
    minus_dm_values: list[float] = []
    dx_window: list[float] = []
    dx_window_sum = 0.0
    output = {}
    previous: Mapping[str, str] | None = None
    for row in rows:
        if previous is None:
            output[_key(row)] = ""
            previous = row
            continue
        true_range = _true_range(row, previous)
        plus_dm, minus_dm = _directional_movement(row, previous)
        tr_values.append(true_range)
        plus_dm_values.append(plus_dm)
        minus_dm_values.append(minus_dm)
        dx_value = _dx(tr_values, plus_dm_values, minus_dm_values)
        if dx_value is not None:
            dx_window.append(dx_value)
            dx_window_sum += dx_value
            if len(dx_window) > ADX_PERIOD:
                dx_window_sum -= dx_window.pop(0)
        output[_key(row)] = _adx_from_window(dx_window, dx_window_sum)
        previous = row
    return output


def _true_range(row: Mapping[str, str], previous: Mapping[str, str]) -> float:
    high = float(row["high_price"])
    low = float(row["low_price"])
    previous_close = float(previous["close_price"])
    return max(high - low, abs(high - previous_close), abs(low - previous_close))


def _directional_movement(
    row: Mapping[str, str],
    previous: Mapping[str, str],
) -> tuple[float, float]:
    up_move = float(row["high_price"]) - float(previous["high_price"])
    down_move = float(previous["low_price"]) - float(row["low_price"])
    plus_dm = up_move if up_move > down_move and up_move > 0.0 else 0.0
    minus_dm = down_move if down_move > up_move and down_move > 0.0 else 0.0
    return plus_dm, minus_dm


def _dx(
    tr_values: Sequence[float],
    plus_dm_values: Sequence[float],
    minus_dm_values: Sequence[float],
) -> float | None:
    if len(tr_values) < ADX_PERIOD:
        return None
    tr_sum = sum(tr_values[-ADX_PERIOD:])
    if tr_sum == 0.0:
        return 0.0
    plus_di = 100.0 * sum(plus_dm_values[-ADX_PERIOD:]) / tr_sum
    minus_di = 100.0 * sum(minus_dm_values[-ADX_PERIOD:]) / tr_sum
    denominator = plus_di + minus_di
    return 0.0 if denominator == 0.0 else 100.0 * abs(plus_di - minus_di) / denominator


def _adx_from_window(dx_window: Sequence[float], dx_window_sum: float) -> str:
    if len(dx_window) < ADX_PERIOD:
        return ""
    return f"{dx_window_sum / ADX_PERIOD:.10f}"


def _enriched_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    regime_values: Mapping[tuple[str, str, str], str],
    adx_values: Mapping[tuple[str, str, str], str],
    include_regime: bool,
    include_adx: bool,
) -> list[dict[str, str]]:
    output = []
    for row in rows:
        enriched = dict(row)
        key = _key(row)
        if include_regime:
            enriched["regime_label"] = regime_values.get(key, "")
        if include_adx:
            enriched["adx_14"] = adx_values.get(key, "")
        output.append(enriched)
    return output


def _added_features(*, regime_available: bool, adx_available: bool) -> list[str]:
    candidates = (("regime_label", regime_available), ("adx_14", adx_available))
    return [feature for feature, available in candidates if available]


def _key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("symbol", "")),
        str(row.get("interval_begin", "")),
        str(row.get("fold_index", "")),
    )


def _feature_lineage(
    *,
    feature_path: Path,
    safe_audit_dir: Path,
    thresholds_path: Path,
    threshold_payload: Mapping[str, Any],
    blocked: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    blocked_names = {row["feature_name"]: row["reason"] for row in blocked}
    return [
        _lineage_row(
            {
                "feature_name": "regime_label",
                "source_columns": REGIME_INPUTS,
                "source_artifact": feature_path,
                "auxiliary_artifact": thresholds_path,
                "calculation_scope": "per_symbol_m8_threshold_classification",
                "causal_window": "current_row_only_with_fixed_m8_thresholds",
                "status": "BLOCKED" if "regime_label" in blocked_names else "ADDED",
                "blocker": blocked_names.get("regime_label", ""),
                "threshold_run_id": str(threshold_payload.get("run_id", "")),
            }
        ),
        _lineage_row(
            {
                "feature_name": "adx_14",
                "source_columns": ADX_INPUTS,
                "source_artifact": feature_path,
                "auxiliary_artifact": safe_audit_dir,
                "calculation_scope": "per_symbol_chronological_ohlc",
                "causal_window": "rolling_14_past_and_current_rows_only",
                "status": "BLOCKED" if "adx_14" in blocked_names else "ADDED",
                "blocker": blocked_names.get("adx_14", ""),
                "threshold_run_id": "",
            }
        ),
    ]


def _lineage_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_name": str(payload["feature_name"]),
        "source_columns": "|".join(payload["source_columns"]),
        "source_artifact": str(payload["source_artifact"]),
        "auxiliary_artifact": str(payload["auxiliary_artifact"]),
        "calculation_scope": str(payload["calculation_scope"]),
        "causal_window": str(payload["causal_window"]),
        "uses_future_data": "False",
        "uses_labels": "False",
        "uses_economic_outcomes": "False",
        "runtime_effect": "False",
        "status": str(payload["status"]),
        "blocker": str(payload["blocker"]),
        "threshold_run_id": str(payload["threshold_run_id"]),
    }


def _leakage_audit(lineage: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "feature_name": str(row["feature_name"]),
            "uses_future_data": str(row["uses_future_data"]),
            "uses_labels": str(row["uses_labels"]),
            "uses_economic_outcomes": str(row["uses_economic_outcomes"]),
            "runtime_effect": str(row["runtime_effect"]),
            "leakage_status": "NO_LEAKAGE_INPUTS_USED",
        }
        for row in lineage
    ]


def _recommendation(
    added_features: Sequence[str],
    blocked: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    if added_features:
        recommendation = "RE_RUN_V2_STRATEGY_CANDIDATE_FACTORY_WITH_RESEARCH_FEATURES"
    else:
        recommendation = "BLOCKED_NO_RESEARCH_FEATURES_ADDED"
    return {
        "recommendation": recommendation,
        "next_required_action": recommendation,
        "features_added": list(added_features),
        "blocked_features": [row["feature_name"] for row in blocked],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "research_features_csv": str(output_dir / "research_features.csv"),
        "research_feature_columns_json": str(output_dir / "research_feature_columns.json"),
        "feature_lineage_csv": str(output_dir / "feature_lineage.csv"),
        "leakage_audit_csv": str(output_dir / "leakage_audit.csv"),
        "blocked_features_csv": str(output_dir / "blocked_features.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
        "research_feature_enrichment_report_json": str(
            output_dir / "research_feature_enrichment_report.json"
        ),
        "research_feature_enrichment_report_md": str(
            output_dir / "research_feature_enrichment_report.md"
        ),
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    output_files = payload["output_files"]
    feature_columns = payload["feature_columns"]
    write_json_artifact(Path(output_files["manifest_json"]), payload["manifest"])
    write_csv_artifact(Path(output_files["research_features_csv"]), payload["rows"])
    write_json_artifact(
        Path(output_files["research_feature_columns_json"]),
        {
            "feature_columns": list(feature_columns),
            "feature_schema_version": "m20_research_v1",
        },
    )
    write_csv_artifact(Path(output_files["feature_lineage_csv"]), payload["lineage"])
    write_csv_artifact(Path(output_files["leakage_audit_csv"]), payload["leakage"])
    write_csv_artifact(Path(output_files["blocked_features_csv"]), payload["blocked"])
    write_json_artifact(Path(output_files["recommendation_json"]), payload["recommendation"])
    write_json_artifact(
        Path(output_files["research_feature_enrichment_report_json"]),
        payload["report"],
    )
    Path(output_files["research_feature_enrichment_report_md"]).write_text(
        _markdown(payload["report"], payload["blocked"]),
        encoding="utf-8",
    )


def _markdown(report: Mapping[str, Any], blocked: Sequence[Mapping[str, str]]) -> str:
    lines = [
        "# M20 Research Feature Enrichment",
        "",
        f"- Rows written: `{report['rows_written']}`",
        f"- Features added: `{', '.join(report['added_features'])}`",
        f"- Recommendation: `{report['recommendation']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
    ]
    if blocked:
        lines.extend(["", "## Blocked Features"])
        lines.extend(f"- `{row['feature_name']}`: `{row['reason']}`" for row in blocked)
    return "\n".join(lines) + "\n"
