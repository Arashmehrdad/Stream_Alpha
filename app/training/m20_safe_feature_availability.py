"""Research-only M20 safe feature availability audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_OUTPUT_NAME = "safe_feature_availability"
FEATURE_FILE = "training_frame/m20_training_frame_features.csv"
FEATURE_COLUMNS_FILE = "training_frame/m20_training_frame_feature_columns.json"
DEFAULT_REGIME_THRESHOLDS = "artifacts/regime/m8/20260320T165813Z/thresholds.json"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "EXISTING_ARTIFACTS_ONLY",
    "NO_FEATURE_ENGINEERING_PERFORMED",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
REQUIRED_REGIME_INPUTS = ("realized_vol_12", "momentum_3", "macd_line_12_26")
REQUIRED_ADX_INPUTS = ("high_price", "low_price", "close_price")
LEAKAGE_COLUMNS = (
    "future_return",
    "gross_value_proxy",
    "net_value_proxy",
    "fee_exceedance_label",
    "triple_barrier_label",
)


def audit_m20_safe_feature_availability(
    *,
    source_run_dir: Path,
    regime_thresholds_path: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Write an artifact-backed audit for blocked M20 v2 research features."""
    # pylint: disable=too-many-locals
    source_dir = Path(source_run_dir).resolve()
    output_dir = source_dir / "research_labels" / "vol_scaled" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = source_dir / FEATURE_FILE
    feature_columns_path = source_dir / FEATURE_COLUMNS_FILE
    thresholds_path = _resolve_thresholds_path(regime_thresholds_path)
    feature_columns = _feature_columns(feature_columns_path)
    sample_rows = _read_csv(feature_path, limit=200)
    thresholds_payload = _optional_json(thresholds_path)
    feature_sources = _feature_sources(
        feature_columns=feature_columns,
        feature_path=feature_path,
        thresholds_path=thresholds_path,
        thresholds_payload=thresholds_payload,
    )
    leakage_audit = _leakage_audit(feature_sources, feature_columns, sample_rows)
    blocked = [row for row in feature_sources if row["availability_status"] != "SAFE_COMPUTABLE"]
    recommendation = _recommendation(feature_sources)
    output_files = _output_files(output_dir)
    report = {
        "source_run_dir": str(source_dir),
        "feature_path": str(feature_path),
        "feature_columns_path": str(feature_columns_path),
        "regime_thresholds_path": str(thresholds_path),
        "feature_count": len(feature_columns),
        "audited_feature_count": len(feature_sources),
        "safe_computable_feature_count": sum(
            1 for row in feature_sources if row["availability_status"] == "SAFE_COMPUTABLE"
        ),
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
            "feature_file": str(feature_path),
            "feature_columns_file": str(feature_columns_path),
            "regime_thresholds_file": str(thresholds_path),
        },
        "audited_features": [row["feature_name"] for row in feature_sources],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["safe_feature_availability_report_json"]), report)
    Path(output_files["safe_feature_availability_report_md"]).write_text(
        _markdown(report, feature_sources, blocked),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["feature_sources_csv"]), feature_sources)
    write_csv_artifact(Path(output_files["blocked_features_csv"]), blocked)
    write_csv_artifact(Path(output_files["leakage_risk_audit_csv"]), leakage_audit)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "feature_sources": feature_sources,
            "blocked_features": blocked,
            "leakage_risk_audit": leakage_audit,
            "recommendation_payload": recommendation,
        }
    )


def _resolve_thresholds_path(path: Path | None) -> Path:
    if path is not None:
        return Path(path).resolve()
    return (Path.cwd() / DEFAULT_REGIME_THRESHOLDS).resolve()


def _feature_columns(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(str(column) for column in payload.get("feature_columns", []))


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, *, limit: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader):
            if index >= limit:
                break
            rows.append(dict(row))
        return rows


def _feature_sources(
    *,
    feature_columns: set[str],
    feature_path: Path,
    thresholds_path: Path,
    thresholds_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _regime_feature_source(feature_columns, feature_path, thresholds_path, thresholds_payload),
        _adx_feature_source(feature_columns, feature_path),
    ]


def _regime_feature_source(
    feature_columns: set[str],
    feature_path: Path,
    thresholds_path: Path,
    thresholds_payload: Mapping[str, Any],
) -> dict[str, Any]:
    missing_inputs = sorted(set(REQUIRED_REGIME_INPUTS) - feature_columns)
    threshold_blockers = _threshold_blockers(thresholds_payload)
    blockers = missing_inputs + threshold_blockers
    return {
        "feature_name": "regime_label",
        "already_available": str("regime_label" in feature_columns),
        "availability_status": "SAFE_COMPUTABLE" if not blockers else "BLOCKED",
        "source_columns": "|".join(REQUIRED_REGIME_INPUTS),
        "source_artifact": str(feature_path),
        "auxiliary_artifact": str(thresholds_path),
        "calculation_scope": "per_symbol_m8_threshold_classification",
        "causal_window": "current_row_only_with_fixed_m8_thresholds",
        "uses_future_data": "False",
        "uses_labels": "False",
        "uses_economic_outcomes": "False",
        "runtime_effect": "False",
        "missing_inputs": "|".join(missing_inputs),
        "blockers": "|".join(blockers),
        "threshold_run_id": str(thresholds_payload.get("run_id", "")),
    }


def _adx_feature_source(feature_columns: set[str], feature_path: Path) -> dict[str, Any]:
    missing_inputs = sorted(set(REQUIRED_ADX_INPUTS) - feature_columns)
    return {
        "feature_name": "adx_14",
        "already_available": str("adx_14" in feature_columns),
        "availability_status": "SAFE_COMPUTABLE" if not missing_inputs else "BLOCKED",
        "source_columns": "|".join(REQUIRED_ADX_INPUTS),
        "source_artifact": str(feature_path),
        "auxiliary_artifact": "",
        "calculation_scope": "per_symbol_chronological_ohlc",
        "causal_window": "rolling_14_past_and_current_rows_only",
        "uses_future_data": "False",
        "uses_labels": "False",
        "uses_economic_outcomes": "False",
        "runtime_effect": "False",
        "missing_inputs": "|".join(missing_inputs),
        "blockers": "|".join(missing_inputs),
        "threshold_run_id": "",
    }


def _threshold_blockers(payload: Mapping[str, Any]) -> list[str]:
    if not payload:
        return ["M8_THRESHOLDS_NOT_AVAILABLE"]
    blockers = []
    if payload.get("schema_version") != "m8_thresholds_v1":
        blockers.append("UNSUPPORTED_M8_THRESHOLD_SCHEMA")
    required_inputs = set(str(value) for value in payload.get("required_inputs", []))
    missing_inputs = sorted(set(REQUIRED_REGIME_INPUTS) - required_inputs)
    blockers.extend(f"THRESHOLD_MISSING_{column}" for column in missing_inputs)
    if not payload.get("thresholds_by_symbol"):
        blockers.append("M8_THRESHOLDS_BY_SYMBOL_MISSING")
    return blockers


def _leakage_audit(
    feature_sources: Sequence[Mapping[str, Any]],
    feature_columns: set[str],
    sample_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    sample_columns = set(sample_rows[0]) if sample_rows else set()
    risky_available = sorted(
        column
        for column in LEAKAGE_COLUMNS
        if column in feature_columns or column in sample_columns
    )
    rows = []
    for source in feature_sources:
        rows.append(
            {
                "feature_name": str(source["feature_name"]),
                "uses_future_data": str(source["uses_future_data"]),
                "uses_labels": str(source["uses_labels"]),
                "uses_economic_outcomes": str(source["uses_economic_outcomes"]),
                "runtime_effect": str(source["runtime_effect"]),
                "risky_columns_available": "|".join(risky_available),
                "leakage_status": "NO_LEAKAGE_INPUTS_REQUIRED",
            }
        )
    return rows


def _recommendation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safe = [
        row["feature_name"]
        for row in rows
        if row["availability_status"] == "SAFE_COMPUTABLE"
    ]
    blocked = [
        row["feature_name"]
        for row in rows
        if row["availability_status"] != "SAFE_COMPUTABLE"
    ]
    if safe:
        recommendation = "BUILD_M20_RESEARCH_FEATURE_ENRICHMENT_ARTIFACT"
        next_required_action = "BUILD_M20_RESEARCH_FEATURE_ENRICHMENT_ARTIFACT"
    else:
        recommendation = "BLOCKED_MISSING_SAFE_FEATURE_SOURCES"
        next_required_action = "IDENTIFY_SAFE_FEATURE_SOURCES_FOR_BLOCKED_V2_DEFINITIONS"
    return {
        "recommendation": recommendation,
        "next_required_action": next_required_action,
        "safe_computable_features": safe,
        "blocked_features": blocked,
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "safe_feature_availability_report_json": str(
            output_dir / "safe_feature_availability_report.json"
        ),
        "safe_feature_availability_report_md": str(
            output_dir / "safe_feature_availability_report.md"
        ),
        "feature_sources_csv": str(output_dir / "feature_sources.csv"),
        "blocked_features_csv": str(output_dir / "blocked_features.csv"),
        "leakage_risk_audit_csv": str(output_dir / "leakage_risk_audit.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    feature_sources: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# M20 Safe Feature Availability Audit",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Safe computable features: `{report['safe_computable_feature_count']}`",
        f"- Blocked features: `{report['blocked_feature_count']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "## Feature Sources",
    ]
    for row in feature_sources:
        lines.append(
            f"- `{row['feature_name']}`: `{row['availability_status']}` "
            f"from `{row['source_columns']}`"
        )
    if blocked:
        lines.extend(["", "## Blockers"])
        for row in blocked:
            lines.append(f"- `{row['feature_name']}`: `{row['blockers']}`")
    return "\n".join(lines) + "\n"
