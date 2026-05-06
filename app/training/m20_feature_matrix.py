"""Research-only M20 feature-source audit and label alignment."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_baseline_research import (  # pylint: disable=protected-access
    EXACT_LEAKAGE_COLUMNS,
    IDENTIFIER_COLUMNS,
    LEAKAGE_TOKENS,
    SCORE_COLUMNS,
    _try_float,
)
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


LABEL_FILE = "triple_barrier_labels_vol_scaled.csv"
FEATURE_MATRIX_DIR_NAME = "feature_matrix"
DISCOVERY_TOKENS = (
    "feature_ohlc",
    "features",
    "training_frame",
    "dataset",
    "oof",
    "fold",
    "model_input",
)
JOIN_CANDIDATES = (
    ("symbol", "interval_begin"),
    ("symbol", "timestamp"),
    ("symbol", "time"),
    ("row_id",),
)
WEAK_UNMATCHED_RATE = 0.05
PREVIEW_ROW_LIMIT = 50


def build_m20_research_feature_matrix(*, run_dir: Path) -> dict[str, Any]:
    """Audit feature sources and build a row-aligned research feature matrix if safe."""
    # pylint: disable=too-many-locals
    resolved_run_dir = Path(run_dir).resolve()
    label_path = (
        resolved_run_dir
        / "research_labels"
        / "vol_scaled"
        / LABEL_FILE
    )
    label_rows = _read_csv_rows(label_path)
    if not label_rows:
        raise ValueError(f"Missing or empty vol-scaled label file: {label_path}")

    matrix_dir = (
        resolved_run_dir
        / "research_labels"
        / "vol_scaled"
        / FEATURE_MATRIX_DIR_NAME
    )
    matrix_dir.mkdir(parents=True, exist_ok=True)
    discovered_sources = _discover_sources(resolved_run_dir)
    candidate_sources = _source_summaries(discovered_sources)
    selected = _select_source(candidate_sources, label_rows)
    selected_rows = _filter_to_label_model(
        _read_csv_rows(Path(selected["path"])) if selected else [],
        label_rows,
    )
    alignment = _align_rows(label_rows, selected_rows) if selected else _empty_alignment(label_rows)
    column_audit = _audit_columns(selected_rows)
    matrix_rows = _matrix_rows(label_rows, alignment, column_audit["safe_numeric_feature_columns"])
    flags = _honesty_flags(
        selected=selected,
        alignment=alignment,
        column_audit=column_audit,
        matrix_rows=matrix_rows,
    )
    recommendation = _recommend(flags)
    output_files = _output_files(matrix_dir, include_matrix=bool(matrix_rows))
    source_audit = {
        "run_dir": str(resolved_run_dir),
        "label_path": str(label_path),
        "label_row_count": len(label_rows),
        "discovered_source_files": discovered_sources,
        "candidate_sources": candidate_sources,
        "selected_source_file": selected,
        "honesty_flags": flags,
        "output_files": output_files,
        "recommendation": recommendation,
    }
    alignment_report = {
        "join_keys_used": alignment["join_keys_used"],
        "label_rows": len(label_rows),
        "feature_rows": len(selected_rows),
        "matched_rows": alignment["matched_rows"],
        "unmatched_label_rows": alignment["unmatched_label_rows"],
        "unmatched_label_rate": alignment["unmatched_label_rate"],
        "duplicate_key_count": alignment["duplicate_key_count"],
        "missing_value_rates_by_feature": _missing_rates(
            matrix_rows,
            column_audit["safe_numeric_feature_columns"],
        ),
        "safe_numeric_feature_count": len(column_audit["safe_numeric_feature_columns"]),
        "excluded_feature_count": len(column_audit["excluded_columns"]),
        "exclusion_reason_counts": dict(Counter(row["reason"] for row in column_audit["excluded"])),
        "timestamp_alignment_summary": alignment["timestamp_alignment_summary"],
        "leakage_screen_summary": column_audit["leakage_screen_summary"],
        "coverage_by_slice": _coverage_by_slice(label_rows, alignment),
        "honesty_flags": flags,
        "recommendation": recommendation,
    }
    manifest = {
        "run_dir": str(resolved_run_dir),
        "feature_matrix_dir": str(matrix_dir),
        "source_label_file": str(label_path),
        "selected_source_file": selected["path"] if selected else None,
        "join_keys_used": alignment["join_keys_used"],
        "safe_numeric_feature_columns": column_audit["safe_numeric_feature_columns"],
        "honesty_flags": flags,
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "recommendation": recommendation,
        "output_files": output_files,
    }
    _write_outputs(
        output_files=output_files,
        source_audit=source_audit,
        alignment_report=alignment_report,
        manifest=manifest,
        column_audit=column_audit,
        matrix_rows=matrix_rows,
    )
    return make_json_safe(
        {
            "run_dir": str(resolved_run_dir),
            "feature_matrix_dir": str(matrix_dir),
            "source_audit": source_audit,
            "alignment_report": alignment_report,
            "manifest": manifest,
            "honesty_flags": flags,
            "recommendation": recommendation,
            "output_files": output_files,
        }
    )


def _discover_sources(run_dir: Path) -> list[dict[str, Any]]:
    sources = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        rel_path = path.relative_to(run_dir).as_posix()
        lowered = rel_path.lower()
        if path.suffix.lower() not in {".csv", ".json", ".parquet"}:
            continue
        if any(token in lowered for token in DISCOVERY_TOKENS):
            sources.append(
                {
                    "path": str(path),
                    "relative_path": rel_path,
                    "suffix": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                    "row_source_supported": path.suffix.lower() == ".csv",
                }
            )
    return sources


def _source_summaries(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source in sources:
        path = Path(str(source["path"]))
        columns: list[str] = []
        row_count = 0
        if path.suffix.lower() == ".csv":
            rows_in_file = _read_csv_rows(path, limit=200)
            columns = list(rows_in_file[0].keys()) if rows_in_file else []
            row_count = _count_csv_rows(path)
        elif path.suffix.lower() == ".json":
            columns = _json_feature_columns(path)
        rows.append(
            {
                **dict(source),
                "columns": columns,
                "row_count": row_count,
                "join_key_candidates": [
                    "+".join(keys) for keys in JOIN_CANDIDATES
                    if all(key in columns for key in keys)
                ],
                "safe_numeric_feature_count_sample": len(
                    _audit_columns(_read_csv_rows(path, limit=200))[
                        "safe_numeric_feature_columns"
                    ]
                )
                if path.suffix.lower() == ".csv"
                else 0,
            }
        )
    return rows


def _select_source(
    sources: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any] | None:
    csv_sources = [source for source in sources if source.get("row_source_supported")]
    scored: list[tuple[tuple[int, int, int, str], dict[str, Any]]] = []
    for source in csv_sources:
        path = Path(str(source["path"]))
        sample_rows = _read_csv_rows(path, limit=500)
        alignment = _align_rows(label_rows[:500], sample_rows)
        if not alignment["join_keys_used"]:
            continue
        proper_key = alignment["join_keys_used"] != ["row_order"]
        source_rank = _source_rank(str(source["relative_path"]))
        score = (
            1 if proper_key else 0,
            int(alignment["matched_rows"]),
            int(source.get("safe_numeric_feature_count_sample", 0)),
            f"{source_rank:03d}:{source['relative_path']}",
        )
        scored.append((score, dict(source)))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _filter_to_label_model(
    feature_rows: Sequence[Mapping[str, str]],
    label_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if not feature_rows or "model_name" not in feature_rows[0]:
        return [dict(row) for row in feature_rows]
    label_model_names = sorted(
        {
            str(row.get("model_name", ""))
            for row in label_rows
            if row.get("model_name")
        }
    )
    if len(label_model_names) != 1:
        return [dict(row) for row in feature_rows]
    model_name = label_model_names[0]
    filtered = [
        dict(row) for row in feature_rows
        if str(row.get("model_name", "")) == model_name
    ]
    return filtered or [dict(row) for row in feature_rows]


def _source_rank(relative_path: str) -> int:
    lowered = relative_path.lower()
    if "feature" in lowered or "training_frame" in lowered or "model_input" in lowered:
        return 4
    if "oof" in lowered:
        return 3
    if "dataset" in lowered:
        return 2
    if "fold" in lowered:
        return 1
    return 0


def _align_rows(
    label_rows: Sequence[Mapping[str, str]],
    feature_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    if not label_rows or not feature_rows:
        return _empty_alignment(label_rows)
    label_columns = set(label_rows[0])
    feature_columns = set(feature_rows[0])
    for keys in JOIN_CANDIDATES:
        if all(key in label_columns and key in feature_columns for key in keys):
            return _align_by_keys(label_rows, feature_rows, keys)
    if len(label_rows) == len(feature_rows):
        matches = {index: dict(feature_rows[index]) for index in range(len(label_rows))}
        return {
            "join_keys_used": ["row_order"],
            "matched_by_label_index": matches,
            "matched_rows": len(matches),
            "unmatched_label_rows": 0,
            "unmatched_label_rate": 0.0,
            "duplicate_key_count": 0,
            "timestamp_alignment_summary": {
                "status": "missing",
                "checked_rows": 0,
                "safe_rows": 0,
                "unsafe_rows": 0,
            },
        }
    return _empty_alignment(label_rows)


def _align_by_keys(
    label_rows: Sequence[Mapping[str, str]],
    feature_rows: Sequence[Mapping[str, str]],
    keys: Sequence[str],
) -> dict[str, Any]:
    # pylint: disable=too-many-locals
    feature_by_key: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for row in feature_rows:
        feature_by_key.setdefault(_key(row, keys), []).append(row)
    duplicate_count = sum(len(rows) - 1 for rows in feature_by_key.values() if len(rows) > 1)
    matches = {}
    safe_timestamps = 0
    unsafe_timestamps = 0
    checked_timestamps = 0
    for index, label_row in enumerate(label_rows):
        candidates = feature_by_key.get(_key(label_row, keys), [])
        if len(candidates) != 1:
            continue
        feature_row = dict(candidates[0])
        timestamp_status = _timestamp_safe(label_row, feature_row, keys)
        if timestamp_status is not None:
            checked_timestamps += 1
            if timestamp_status:
                safe_timestamps += 1
            else:
                unsafe_timestamps += 1
                continue
        matches[index] = feature_row
    unmatched = len(label_rows) - len(matches)
    timestamp_status = "confirmed" if checked_timestamps and not unsafe_timestamps else "missing"
    if unsafe_timestamps:
        timestamp_status = "failed"
    return {
        "join_keys_used": list(keys),
        "matched_by_label_index": matches,
        "matched_rows": len(matches),
        "unmatched_label_rows": unmatched,
        "unmatched_label_rate": _safe_ratio(unmatched, len(label_rows)),
        "duplicate_key_count": duplicate_count,
        "timestamp_alignment_summary": {
            "status": timestamp_status,
            "checked_rows": checked_timestamps,
            "safe_rows": safe_timestamps,
            "unsafe_rows": unsafe_timestamps,
        },
    }


def _timestamp_safe(
    label_row: Mapping[str, str],
    feature_row: Mapping[str, str],
    keys: Sequence[str],
) -> bool | None:
    timestamp_key = next(
        (key for key in keys if key in {"interval_begin", "timestamp", "time"}),
        None,
    )
    if timestamp_key is None:
        return None
    label_timestamp = _parse_timestamp(
        label_row.get(timestamp_key) or label_row.get("interval_begin")
    )
    feature_timestamp = _parse_timestamp(feature_row.get(timestamp_key))
    if label_timestamp is None or feature_timestamp is None:
        return None
    return feature_timestamp <= label_timestamp


def _audit_columns(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    columns = list(rows[0].keys()) if rows else []
    numeric_columns = [
        column for column in columns
        if all(_try_float(row.get(column)) is not None for row in rows[: min(len(rows), 100)])
    ]
    excluded = []
    safe = []
    for column in numeric_columns:
        reason = _exclusion_reason(column)
        if reason:
            excluded.append({"column": column, "reason": reason})
        else:
            safe.append(column)
    return {
        "available_columns": columns,
        "numeric_columns": numeric_columns,
        "safe_numeric_feature_columns": safe,
        "score_feature_columns": [column for column in safe if column in SCORE_COLUMNS],
        "excluded_columns": [row["column"] for row in excluded],
        "excluded": excluded,
        "leakage_screen_summary": {
            "numeric_columns": len(numeric_columns),
            "excluded_columns": len(excluded),
            "safe_numeric_feature_columns": len(safe),
        },
    }


def _exclusion_reason(column: str) -> str | None:
    lowered = column.lower()
    if lowered in EXACT_LEAKAGE_COLUMNS:
        return "realized_outcome"
    if column in IDENTIFIER_COLUMNS:
        return "identifier_or_timestamp"
    for token in LEAKAGE_TOKENS:
        if token in lowered:
            return f"leakage_token:{token}"
    if "post" in lowered or "event_end" in lowered:
        return "post_event_or_event_metadata"
    return None


def _matrix_rows(
    label_rows: Sequence[Mapping[str, str]],
    alignment: Mapping[str, Any],
    feature_columns: Sequence[str],
) -> list[dict[str, Any]]:
    if not feature_columns or not alignment["matched_rows"]:
        return []
    matches: Mapping[int, Mapping[str, str]] = alignment["matched_by_label_index"]
    rows = []
    for index, label_row in enumerate(label_rows):
        feature_row = matches.get(index)
        if feature_row is None:
            continue
        output = {
            "row_id": label_row.get("row_id", ""),
            "symbol": label_row.get("symbol", ""),
            "interval_begin": label_row.get("interval_begin", ""),
            "fold_index": label_row.get("fold_index", ""),
            "regime_label": label_row.get("regime_label", ""),
            "label": label_row.get("label", ""),
        }
        for column in feature_columns:
            output[column] = feature_row.get(column, "")
        rows.append(output)
    return rows


def _honesty_flags(
    *,
    selected: Mapping[str, Any] | None,
    alignment: Mapping[str, Any],
    column_audit: Mapping[str, Any],
    matrix_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    flags = ["NO_PROMOTION_EFFECT", "NO_RUNTIME_EFFECT", "RESEARCH_ONLY_FEATURE_MATRIX"]
    if selected:
        flags.append("FEATURE_SOURCE_FOUND")
    else:
        flags.append("FEATURE_SOURCE_MISSING")
    if column_audit["safe_numeric_feature_columns"]:
        flags.append("SAFE_FEATURES_FOUND")
    else:
        flags.append("SAFE_FEATURES_MISSING")
    if alignment["join_keys_used"] == ["row_order"]:
        flags.append("WEAK_ROW_ORDER_ALIGNMENT")
    timestamp_status = alignment["timestamp_alignment_summary"]["status"]
    if timestamp_status == "confirmed":
        flags.append("TIMESTAMP_ALIGNMENT_CONFIRMED")
    else:
        flags.append("TIMESTAMP_ALIGNMENT_MISSING")
    if column_audit["excluded_columns"]:
        flags.append("POSSIBLE_LEAKAGE_COLUMNS_EXCLUDED")
    if alignment["unmatched_label_rate"] > WEAK_UNMATCHED_RATE:
        flags.append("HIGH_UNMATCHED_LABEL_RATE")
    if alignment["duplicate_key_count"]:
        flags.append("DUPLICATE_JOIN_KEYS")
    ready = (
        bool(matrix_rows)
        and "WEAK_ROW_ORDER_ALIGNMENT" not in flags
        and "HIGH_UNMATCHED_LABEL_RATE" not in flags
        and "TIMESTAMP_ALIGNMENT_CONFIRMED" in flags
    )
    flags.append(
        "FEATURE_MATRIX_READY_FOR_RESEARCH_BASELINE"
        if ready
        else "FEATURE_MATRIX_NOT_TRAINING_READY"
    )
    return sorted(dict.fromkeys(flags))


def _recommend(flags: Sequence[str]) -> str:
    if "FEATURE_SOURCE_MISSING" in flags or "SAFE_FEATURES_MISSING" in flags:
        return "B. export row-aligned training features from the original M20 training pipeline"
    if "WEAK_ROW_ORDER_ALIGNMENT" in flags or "TIMESTAMP_ALIGNMENT_MISSING" in flags:
        return "C. fix join keys / timestamp metadata before modeling"
    if "HIGH_UNMATCHED_LABEL_RATE" in flags or "DUPLICATE_JOIN_KEYS" in flags:
        return "C. fix join keys / timestamp metadata before modeling"
    if "FEATURE_MATRIX_READY_FOR_RESEARCH_BASELINE" in flags:
        return "A. train a tiny tabular research baseline on the aligned feature matrix next"
    return "D. keep using score-only baselines only as limited diagnostics"


def _write_outputs(
    *,
    output_files: Mapping[str, str],
    source_audit: Mapping[str, Any],
    alignment_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    column_audit: Mapping[str, Any],
    matrix_rows: Sequence[Mapping[str, Any]],
) -> None:
    # pylint: disable=too-many-arguments
    write_json_artifact(Path(output_files["feature_source_audit_json"]), source_audit)
    Path(output_files["feature_source_audit_md"]).write_text(
        _source_audit_markdown(source_audit),
        encoding="utf-8",
    )
    write_csv_artifact(
        Path(output_files["feature_candidate_columns_csv"]),
        _candidate_column_rows(source_audit),
    )
    write_csv_artifact(
        Path(output_files["feature_exclusion_reasons_csv"]),
        column_audit["excluded"],
    )
    write_json_artifact(Path(output_files["feature_alignment_report_json"]), alignment_report)
    Path(output_files["feature_alignment_report_md"]).write_text(
        _alignment_markdown(alignment_report),
        encoding="utf-8",
    )
    write_json_artifact(Path(output_files["research_feature_matrix_manifest_json"]), manifest)
    write_csv_artifact(
        Path(output_files["feature_matrix_preview_csv"]),
        list(matrix_rows[:PREVIEW_ROW_LIMIT]),
    )
    if matrix_rows and "research_feature_matrix_csv" in output_files:
        write_csv_artifact(Path(output_files["research_feature_matrix_csv"]), list(matrix_rows))


def _candidate_column_rows(source_audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    selected_path = None
    if source_audit["selected_source_file"]:
        selected_path = source_audit["selected_source_file"]["path"]
    for source in source_audit["candidate_sources"]:
        for column in source.get("columns", []):
            rows.append(
                {
                    "source_path": source["relative_path"],
                    "column": column,
                    "selected_source": str(source["path"] == selected_path),
                }
            )
    return rows


def _coverage_by_slice(
    label_rows: Sequence[Mapping[str, str]],
    alignment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matches = set(alignment["matched_by_label_index"])
    rows = []
    for column in ("symbol", "fold_index", "regime_label"):
        if not any(row.get(column) for row in label_rows):
            continue
        grouped: dict[str, list[int]] = {}
        for index, row in enumerate(label_rows):
            grouped.setdefault(str(row.get(column, "")), []).append(index)
        for value, indexes in sorted(grouped.items()):
            matched = sum(1 for index in indexes if index in matches)
            rows.append(
                {
                    "slice_column": column,
                    "slice_value": value,
                    "label_rows": len(indexes),
                    "matched_rows": matched,
                    "match_rate": _safe_ratio(matched, len(indexes)),
                }
            )
    return rows


def _missing_rates(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> dict[str, float]:
    return {
        column: _safe_ratio(
            sum(1 for row in rows if row.get(column) in ("", None)),
            len(rows),
        )
        for column in feature_columns
    }


def _empty_alignment(label_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "join_keys_used": [],
        "matched_by_label_index": {},
        "matched_rows": 0,
        "unmatched_label_rows": len(label_rows),
        "unmatched_label_rate": 1.0 if label_rows else 0.0,
        "duplicate_key_count": 0,
        "timestamp_alignment_summary": {
            "status": "missing",
            "checked_rows": 0,
            "safe_rows": 0,
            "unsafe_rows": 0,
        },
    }


def _output_files(matrix_dir: Path, *, include_matrix: bool) -> dict[str, str]:
    output = {
        "feature_source_audit_json": str(matrix_dir / "feature_source_audit.json"),
        "feature_source_audit_md": str(matrix_dir / "feature_source_audit.md"),
        "feature_candidate_columns_csv": str(matrix_dir / "feature_candidate_columns.csv"),
        "feature_exclusion_reasons_csv": str(matrix_dir / "feature_exclusion_reasons.csv"),
        "feature_alignment_report_json": str(matrix_dir / "feature_alignment_report.json"),
        "feature_alignment_report_md": str(matrix_dir / "feature_alignment_report.md"),
        "research_feature_matrix_manifest_json": str(
            matrix_dir / "research_feature_matrix_manifest.json"
        ),
        "feature_matrix_preview_csv": str(matrix_dir / "feature_matrix_preview.csv"),
    }
    if include_matrix:
        output["research_feature_matrix_csv"] = str(matrix_dir / "research_feature_matrix.csv")
    return output


def _source_audit_markdown(audit: Mapping[str, Any]) -> str:
    selected = audit["selected_source_file"]
    selected_path = selected["relative_path"] if selected else "none"
    return "\n".join(
        [
            "# M20 Research Feature Source Audit",
            "",
            f"- Discovered source files: `{len(audit['discovered_source_files'])}`",
            f"- Selected source file: `{selected_path}`",
            f"- Honesty flags: `{', '.join(audit['honesty_flags'])}`",
            f"- Recommendation: `{audit['recommendation']}`",
            "",
            "This artifact is research-only and does not affect runtime inference, "
            "registry authority, promotion, or execution.",
            "",
        ]
    )


def _alignment_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# M20 Research Feature Alignment Report",
            "",
            f"- Join keys used: `{'+'.join(report['join_keys_used']) or 'none'}`",
            f"- Label rows: `{report['label_rows']}`",
            f"- Feature rows: `{report['feature_rows']}`",
            f"- Matched rows: `{report['matched_rows']}`",
            f"- Unmatched label rows: `{report['unmatched_label_rows']}`",
            f"- Safe numeric feature count: `{report['safe_numeric_feature_count']}`",
            f"- Duplicate key count: `{report['duplicate_key_count']}`",
            f"- Timestamp status: `{report['timestamp_alignment_summary']['status']}`",
            f"- Recommendation: `{report['recommendation']}`",
            "",
        ]
    )


def _read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return max(sum(1 for _ in input_file) - 1, 0)


def _json_feature_columns(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    columns: list[str] = []
    _collect_column_names(payload, columns)
    return sorted(dict.fromkeys(columns))


def _collect_column_names(value: Any, output: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if "column" in lowered and isinstance(nested, list):
                output.extend(str(item) for item in nested if isinstance(item, str))
            else:
                _collect_column_names(nested, output)
    elif isinstance(value, list):
        for item in value:
            _collect_column_names(item, output)


def _key(row: Mapping[str, str], keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in keys)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    value = float(numerator) / float(denominator)
    if not math.isfinite(value):
        return 0.0
    return value
