"""Opt-in research-only export hook for M20 row-level market features."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


EXPORT_DIR_NAME = "training_frame"
SCHEMA_VERSION = "m20_training_frame_v1"
PREVIEW_ROW_LIMIT = 50
KEY_COLUMNS = ("symbol", "interval_begin", "fold_index", "row_id")
TIMESTAMP_COLUMN = "interval_begin"
PREDICTION_OUTPUT_COLUMNS = {
    "y_pred",
    "prob_up",
    "confidence",
    "long_trade_taken",
}
LEAKAGE_TOKENS = (
    "label",
    "target",
    "future",
    "forward",
    "barrier",
    "outcome",
    "post",
    "gross",
    "net",
)
HONESTY_BASE_FLAGS = (
    "NO_PROMOTION_EFFECT",
    "NO_REGISTRY_WRITE",
    "NO_RUNTIME_EFFECT",
    "RESEARCH_ONLY_TRAINING_FRAME_EXPORT_HOOK",
    "TRAINING_FRAME_EXPORT_NOT_RUNTIME_READY",
)
FOLD_DIR_NAME = "folds"


def maybe_export_m20_training_frame(
    *,
    enabled: bool,
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    config_path: Path | None = None,
    source: str = "app.training.service.run_training",
) -> dict[str, Any] | None:
    """Write the research-only training-frame export when explicitly enabled."""
    # pylint: disable=too-many-arguments
    if not enabled:
        return None
    return export_m20_training_frame(
        run_dir=run_dir,
        rows=rows,
        feature_columns=feature_columns,
        config_path=config_path,
        source=source,
    )


def export_m20_training_frame_fold(
    *,
    run_dir: Path,
    fold_index: int,
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    config_path: Path | None = None,
    score_only_source_path: Path | None = None,
    parquet_dir: Path | None = None,
    export_mode: str = "early_score_only",
    skipped_folds: Sequence[Mapping[str, Any]] = (),
    confirmation_window: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one fold's market-feature frame before model scoring starts."""
    # pylint: disable=too-many-arguments,too-many-locals
    resolved_run_dir = Path(run_dir).resolve()
    export_dir = resolved_run_dir / EXPORT_DIR_NAME
    fold_dir = export_dir / FOLD_DIR_NAME
    fold_dir.mkdir(parents=True, exist_ok=True)
    audit = _audit_rows(rows=rows, feature_columns=feature_columns)
    export_rows = _export_rows(rows, audit["feature_columns"])
    fold_label = f"fold_{fold_index}"
    feature_path = fold_dir / f"{fold_label}_features.csv"
    key_path = fold_dir / f"{fold_label}_keys.csv"
    manifest_path = fold_dir / f"{fold_label}_manifest.json"
    _write_csv_atomic(feature_path, export_rows)
    _write_csv_atomic(key_path, _key_rows(export_rows))
    manifest = _fold_manifest(
        run_dir=resolved_run_dir,
        fold_index=fold_index,
        export_rows=export_rows,
        feature_columns=audit["feature_columns"],
        excluded=audit["excluded"],
        config_path=config_path,
        score_only_source_path=score_only_source_path,
        parquet_dir=parquet_dir,
        export_mode=export_mode,
        confirmation_window=confirmation_window,
    )
    _write_json_atomic(manifest_path, manifest)
    checkpoint = _checkpoint_payload(
        run_dir=resolved_run_dir,
        export_mode=export_mode,
        feature_columns=audit["feature_columns"],
        skipped_folds=skipped_folds,
    )
    _write_json_atomic(export_dir / "m20_training_frame_export_checkpoint.json", checkpoint)
    return make_json_safe(manifest)


def finalize_m20_training_frame_export(
    *,
    run_dir: Path,
    feature_columns: Sequence[str],
    config_path: Path | None = None,
    score_only_source_path: Path | None = None,
    parquet_dir: Path | None = None,
    export_mode: str = "early_score_only",
    skipped_folds: Sequence[Mapping[str, Any]] = (),
    complete: bool = True,
    confirmation_window: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine exported fold files into final research training-frame artifacts."""
    # pylint: disable=too-many-arguments
    resolved_run_dir = Path(run_dir).resolve()
    export_dir = resolved_run_dir / EXPORT_DIR_NAME
    fold_rows = _read_fold_feature_rows(export_dir / FOLD_DIR_NAME)
    return export_m20_training_frame(
        run_dir=resolved_run_dir,
        rows=fold_rows,
        feature_columns=feature_columns,
        config_path=config_path,
        source="app.training.service.run_training.early_export",
        score_only_source_path=score_only_source_path,
        parquet_dir=parquet_dir,
        export_mode=export_mode,
        skipped_folds=skipped_folds,
        complete=complete,
        confirmation_window=confirmation_window,
    )


def export_m20_training_frame(
    *,
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    config_path: Path | None = None,
    source: str = "app.training.service.run_training",
    score_only_source_path: Path | None = None,
    parquet_dir: Path | None = None,
    export_mode: str = "full_pipeline",
    skipped_folds: Sequence[Mapping[str, Any]] = (),
    complete: bool = True,
    confirmation_window: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write deterministic market-feature frame artifacts for future research."""
    # pylint: disable=too-many-arguments,too-many-locals
    resolved_run_dir = Path(run_dir).resolve()
    export_dir = resolved_run_dir / EXPORT_DIR_NAME
    export_dir.mkdir(parents=True, exist_ok=True)
    audit = _audit_rows(rows=rows, feature_columns=feature_columns)
    export_rows = _export_rows(rows, audit["feature_columns"])
    flags = _honesty_flags(audit, export_rows)
    if export_mode == "export_only":
        flags.extend(["EXPORT_ONLY_MODE", "MODEL_SCORING_SKIPPED_EXPORT_ONLY"])
    if skipped_folds:
        flags.extend(["FOLDS_SKIPPED_BY_RECENT_WINDOW", "RECENT_WINDOW_FILTER_APPLIED"])
    if confirmation_window and confirmation_window.get("override_enabled"):
        flags.extend(
            [
                "CONFIRMATION_WINDOW_OVERRIDE_SUPPORTED",
                "CONFIRMATION_WINDOW_REQUIRES_MANUAL_RUN",
                "RESEARCH_ONLY_CONFIRMATION_WINDOW_OVERRIDE",
            ]
        )
        flags.extend(confirmation_window.get("honesty_flags", []))
    if complete:
        flags.append("MARKET_FEATURE_FRAME_EXPORT_COMPLETE")
    else:
        flags.append("MARKET_FEATURE_FRAME_EXPORT_PARTIAL")
    flags = sorted(dict.fromkeys(flags))
    output_files = _output_files(export_dir)
    report = {
        "run_id": resolved_run_dir.name,
        "run_dir": str(resolved_run_dir),
        "config_path": str(config_path) if config_path else None,
        "score_only_source_path": str(score_only_source_path) if score_only_source_path else None,
        "parquet_dir": str(parquet_dir) if parquet_dir else None,
        "source": source,
        "export_mode": export_mode,
        "export_timestamp": _deterministic_export_timestamp(export_rows),
        "row_count": len(export_rows),
        "feature_count": len(audit["feature_columns"]),
        "key_columns": list(KEY_COLUMNS),
        "fold_coverage": _coverage(export_rows, "fold_index"),
        "symbol_coverage": _coverage(export_rows, "symbol"),
        "timestamp_range": _timestamp_range(export_rows),
        "missing_value_summary": _missing_value_summary(export_rows, audit["feature_columns"]),
        "duplicate_key_count": _duplicate_key_count(export_rows),
        "row_count_by_fold": _row_count_by_fold(export_rows),
        "eligible_folds": sorted(_coverage(export_rows, "fold_index")),
        "skipped_folds": list(skipped_folds),
        "confirmation_window": dict(confirmation_window or {"override_enabled": False}),
        "feature_column_hash": _feature_column_hash(audit["feature_columns"]),
        "feature_schema_version": SCHEMA_VERSION,
        "export_complete": bool(complete),
        "leakage_screen_summary": {
            "input_feature_columns": len(feature_columns),
            "exported_feature_columns": len(audit["feature_columns"]),
            "excluded_columns": len(audit["excluded_columns"]),
        },
        "excluded_columns_by_reason": dict(Counter(row["reason"] for row in audit["excluded"])),
        "excluded_columns": audit["excluded"],
        "honesty_flags": flags,
        "output_files": output_files,
    }
    manifest = {
        "run_id": report["run_id"],
        "run_dir": report["run_dir"],
        "config_path": report["config_path"],
        "score_only_source_path": report["score_only_source_path"],
        "parquet_dir": report["parquet_dir"],
        "source": report["source"],
        "export_mode": report["export_mode"],
        "export_timestamp": report["export_timestamp"],
        "feature_schema_version": SCHEMA_VERSION,
        "feature_columns": audit["feature_columns"],
        "feature_column_hash": report["feature_column_hash"],
        "key_columns": list(KEY_COLUMNS),
        "row_count": len(export_rows),
        "row_count_by_fold": report["row_count_by_fold"],
        "eligible_folds": report["eligible_folds"],
        "skipped_folds": report["skipped_folds"],
        "confirmation_window": report["confirmation_window"],
        "export_complete": report["export_complete"],
        "honesty_flags": flags,
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "output_files": output_files,
    }
    _write_artifacts(
        output_files=output_files,
        rows=export_rows,
        feature_columns=audit["feature_columns"],
        manifest=manifest,
        report=report,
    )
    return make_json_safe({"manifest": manifest, "report": report, "honesty_flags": flags})


def _audit_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    exported_features = []
    excluded = []
    available_columns = set(rows[0].keys()) if rows else set()
    for column in feature_columns:
        reason = _exclusion_reason(column)
        if reason:
            excluded.append({"column": column, "reason": reason})
            continue
        if column in KEY_COLUMNS:
            excluded.append({"column": column, "reason": "key_column"})
            continue
        if rows and column not in available_columns:
            excluded.append({"column": column, "reason": "missing_from_rows"})
            continue
        exported_features.append(str(column))
    return {
        "feature_columns": exported_features,
        "excluded_columns": [row["column"] for row in excluded],
        "excluded": excluded,
    }


def _export_rows(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output_row = {column: row.get(column, "") for column in KEY_COLUMNS}
        for column in feature_columns:
            output_row[column] = row.get(column, "")
        output.append(output_row)
    return output


def _honesty_flags(
    audit: Mapping[str, Any],
    export_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    flags = list(HONESTY_BASE_FLAGS)
    flags.append("EARLY_EXPORT_BEFORE_MODEL_SCORING")
    if export_rows and audit["feature_columns"]:
        flags.extend(
            [
                "FEATURE_SCHEMA_RECORDED",
                "MARKET_FEATURE_FRAME_EXPORTED",
                "TRAINING_FRAME_EXPORT_READY_FOR_RESEARCH_BASELINE",
            ]
        )
    else:
        flags.append("MARKET_FEATURE_FRAME_EXPORT_SKIPPED")
    if any(row["reason"] == "prediction_output" for row in audit["excluded"]):
        flags.append("PREDICTION_OUTPUT_COLUMNS_EXCLUDED")
    if audit["excluded"]:
        flags.append("POSSIBLE_LEAKAGE_COLUMNS_EXCLUDED")
    if export_rows and all(row.get(TIMESTAMP_COLUMN) for row in export_rows):
        flags.append("TIMESTAMP_KEYS_PRESENT")
    if export_rows and any(str(row.get("fold_index", "")) != "" for row in export_rows):
        flags.append("FOLD_KEYS_PRESENT")
    return sorted(dict.fromkeys(flags))


def _fold_manifest(
    *,
    run_dir: Path,
    fold_index: int,
    export_rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    excluded: Sequence[Mapping[str, str]],
    config_path: Path | None,
    score_only_source_path: Path | None,
    parquet_dir: Path | None,
    export_mode: str,
    confirmation_window: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # pylint: disable=too-many-arguments
    return {
        "run_id": run_dir.name,
        "fold_index": fold_index,
        "config_path": str(config_path) if config_path else None,
        "score_only_source_path": str(score_only_source_path) if score_only_source_path else None,
        "parquet_dir": str(parquet_dir) if parquet_dir else None,
        "export_mode": export_mode,
        "confirmation_window": dict(confirmation_window or {"override_enabled": False}),
        "row_count": len(export_rows),
        "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
        "feature_column_hash": _feature_column_hash(feature_columns),
        "key_columns": list(KEY_COLUMNS),
        "symbol_coverage": _coverage(export_rows, "symbol"),
        "timestamp_range": _timestamp_range(export_rows),
        "duplicate_key_count": _duplicate_key_count(export_rows),
        "excluded_columns": list(excluded),
        "honesty_flags": sorted(
            dict.fromkeys(
                list(HONESTY_BASE_FLAGS)
                + [
                    "EARLY_EXPORT_BEFORE_MODEL_SCORING",
                    "MARKET_FEATURE_FRAME_EXPORTED",
                    "TRAINING_FRAME_EXPORT_READY_FOR_RESEARCH_BASELINE",
                ]
            )
        ),
    }


def _checkpoint_payload(
    *,
    run_dir: Path,
    export_mode: str,
    feature_columns: Sequence[str],
    skipped_folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fold_dir = run_dir / EXPORT_DIR_NAME / FOLD_DIR_NAME
    manifests = sorted(fold_dir.glob("fold_*_manifest.json")) if fold_dir.exists() else []
    return {
        "run_id": run_dir.name,
        "export_mode": export_mode,
        "export_complete": False,
        "exported_fold_count": len(manifests),
        "exported_fold_manifests": [str(path) for path in manifests],
        "feature_columns": list(feature_columns),
        "feature_column_hash": _feature_column_hash(feature_columns),
        "skipped_folds": list(skipped_folds),
        "honesty_flags": sorted(
            dict.fromkeys(
                list(HONESTY_BASE_FLAGS)
                + [
                    "EARLY_EXPORT_BEFORE_MODEL_SCORING",
                    "MARKET_FEATURE_FRAME_EXPORT_PARTIAL",
                ]
            )
        ),
    }


def _exclusion_reason(column: str) -> str | None:
    lowered = str(column).lower()
    if lowered in PREDICTION_OUTPUT_COLUMNS:
        return "prediction_output"
    if lowered in {"y_true", "label"}:
        return "target_or_label"
    for token in LEAKAGE_TOKENS:
        if token in lowered:
            return f"leakage_token:{token}"
    return None


def _write_artifacts(
    *,
    output_files: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    write_csv_artifact(Path(output_files["features_csv"]), list(rows))
    write_csv_artifact(Path(output_files["keys_csv"]), _key_rows(rows))
    write_json_artifact(
        Path(output_files["feature_columns_json"]),
        {
            "feature_columns": list(feature_columns),
            "feature_column_hash": _feature_column_hash(feature_columns),
            "feature_schema_version": SCHEMA_VERSION,
        },
    )
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(_report_markdown(report), encoding="utf-8")
    write_csv_artifact(Path(output_files["preview_csv"]), list(rows[:PREVIEW_ROW_LIMIT]))
    _write_json_atomic(
        Path(output_files["checkpoint_json"]),
        {
            "run_id": manifest["run_id"],
            "export_mode": manifest["export_mode"],
            "export_complete": manifest["export_complete"],
            "row_count": manifest["row_count"],
            "row_count_by_fold": manifest["row_count_by_fold"],
            "feature_columns": manifest["feature_columns"],
            "feature_column_hash": manifest["feature_column_hash"],
            "skipped_folds": manifest["skipped_folds"],
            "honesty_flags": manifest["honesty_flags"],
        },
    )


def _output_files(export_dir: Path) -> dict[str, str]:
    return {
        "features_csv": str(export_dir / "m20_training_frame_features.csv"),
        "keys_csv": str(export_dir / "m20_training_frame_keys.csv"),
        "feature_columns_json": str(export_dir / "m20_training_frame_feature_columns.json"),
        "manifest_json": str(export_dir / "m20_training_frame_export_manifest.json"),
        "report_json": str(export_dir / "m20_training_frame_export_report.json"),
        "report_md": str(export_dir / "m20_training_frame_export_report.md"),
        "preview_csv": str(export_dir / "m20_training_frame_preview.csv"),
        "checkpoint_json": str(export_dir / "m20_training_frame_export_checkpoint.json"),
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# M20 Training Frame Export Hook Report",
            "",
            f"- Run id: `{report['run_id']}`",
            f"- Row count: `{report['row_count']}`",
            f"- Export mode: `{report['export_mode']}`",
            f"- Export complete: `{report['export_complete']}`",
            f"- Feature count: `{report['feature_count']}`",
            f"- Feature schema version: `{report['feature_schema_version']}`",
            f"- Feature column hash: `{report['feature_column_hash']}`",
            f"- Duplicate key count: `{report['duplicate_key_count']}`",
            f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
            "",
            "This export is research-only and has no runtime, registry, or promotion effect.",
            "",
        ]
    )


def _key_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{column: row.get(column, "") for column in KEY_COLUMNS} for row in rows]


def _coverage(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row.get(column, "")) for row in rows if row.get(column)).items())
    )


def _row_count_by_fold(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return _coverage(rows, "fold_index")


def _timestamp_range(rows: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    parsed = [
        _parse_timestamp(str(row.get(TIMESTAMP_COLUMN, "")))
        for row in rows
        if row.get(TIMESTAMP_COLUMN)
    ]
    valid = [value for value in parsed if value is not None]
    return {
        "min": min(valid).isoformat() if valid else None,
        "max": max(valid).isoformat() if valid else None,
    }


def _missing_value_summary(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> dict[str, float]:
    return {
        column: _safe_ratio(sum(1 for row in rows if row.get(column) in ("", None)), len(rows))
        for column in feature_columns
    }


def _duplicate_key_count(rows: Sequence[Mapping[str, Any]]) -> int:
    counts = Counter(tuple(str(row.get(column, "")) for column in KEY_COLUMNS) for row in rows)
    return sum(count - 1 for count in counts.values() if count > 1)


def _feature_column_hash(feature_columns: Sequence[str]) -> str:
    payload = json.dumps(list(feature_columns), separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_fold_feature_rows(fold_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not fold_dir.exists():
        return rows
    for path in sorted(fold_dir.glob("fold_*_features.csv")):
        rows.extend(read_csv_rows(path))
    return rows


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    write_csv_artifact(temp_path, list(rows))
    temp_path.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    write_json_artifact(temp_path, payload)
    temp_path.replace(path)


def _deterministic_export_timestamp(rows: Sequence[Mapping[str, Any]]) -> str | None:
    timestamp_range = _timestamp_range(rows)
    return timestamp_range["max"]


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    value = float(numerator) / float(denominator)
    if not math.isfinite(value):
        return 0.0
    return value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read exported CSV rows for tests and lightweight operator checks."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]
