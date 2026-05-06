"""Research-only M20 training-frame market feature export audit."""

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
    SCORE_COLUMNS,
    _try_float,
)
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


EXPORT_DIR_NAME = "training_frame_export"
DISCOVERY_TOKENS = (
    "training_frame",
    "dataset",
    "feature",
    "feature_ohlc",
    "fold",
    "model_input",
    "score_input",
    "export",
    "oof",
)
KEY_COLUMNS = ("symbol", "interval_begin", "timestamp", "time", "fold_index", "row_id")
TIMESTAMP_COLUMNS = ("interval_begin", "timestamp", "time", "as_of_time")
SCORE_POLICY_COLUMNS = SCORE_COLUMNS | {"long_trade_taken"}
LEAKAGE_NAME_TOKENS = (
    "label",
    "future",
    "forward",
    "gross",
    "net",
    "barrier",
    "hit",
    "horizon",
    "target",
    "outcome",
)
MARKET_HINT_TOKENS = (
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "trade_count",
    "momentum",
    "mean",
    "std",
    "vol",
    "rsi",
    "macd",
    "zscore",
    "lag",
)
PREVIEW_ROW_LIMIT = 50


def export_m20_training_frame_features(*, run_dir: Path) -> dict[str, Any]:
    """Export row-aligned market features or write an explicit blocker."""
    # pylint: disable=too-many-locals
    resolved_run_dir = Path(run_dir).resolve()
    export_dir = (
        resolved_run_dir
        / "research_labels"
        / "vol_scaled"
        / EXPORT_DIR_NAME
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    configured_market_features = _configured_market_features(resolved_run_dir)
    discovered = _discover_candidate_files(resolved_run_dir)
    candidate_summaries = _candidate_summaries(discovered, configured_market_features)
    selected = _select_source(candidate_summaries)
    selected_rows = _read_csv_rows(Path(selected["path"])) if selected else []
    audit = _audit_selected_rows(
        selected_rows,
        configured_market_features=configured_market_features,
    )
    export_rows = _export_rows(selected_rows, audit) if selected else []
    flags = _honesty_flags(
        selected=selected,
        audit=audit,
        export_rows=export_rows,
        candidate_summaries=candidate_summaries,
    )
    recommendation = _recommend(flags)
    output_files = _output_files(export_dir, success=bool(export_rows))
    report = {
        "run_dir": str(resolved_run_dir),
        "export_dir": str(export_dir),
        "discovered_candidate_files": discovered,
        "candidate_sources": candidate_summaries,
        "selected_source_file": selected,
        "selected_join_keys": audit["join_keys"],
        "row_count": len(export_rows),
        "symbol_coverage": _coverage(export_rows, "symbol"),
        "fold_coverage": _coverage(export_rows, "fold_index"),
        "timestamp_coverage": _timestamp_coverage(export_rows),
        "safe_market_feature_count": len(audit["market_feature_columns"]),
        "safe_market_feature_columns": audit["market_feature_columns"],
        "excluded_column_count": len(audit["excluded_columns"]),
        "exclusion_reason_counts": dict(Counter(row["reason"] for row in audit["excluded"])),
        "missing_value_rates": _missing_rates(export_rows, audit["market_feature_columns"]),
        "duplicate_key_count": audit["duplicate_key_count"],
        "oof_score_columns_excluded_from_market_features": audit[
            "oof_score_columns_excluded"
        ],
        "feature_timestamps_prediction_time_safe": audit["timestamp_safe"],
        "honesty_flags": flags,
        "recommendation": recommendation,
        "output_files": output_files,
    }
    schema = _future_export_schema(configured_market_features)
    if export_rows:
        manifest = _manifest(report, configured_market_features)
        _write_success_outputs(output_files, report, manifest, export_rows, audit)
    else:
        blocker = {
            **report,
            "blocker": _blocker_reason(flags),
            "required_future_export_schema": schema,
        }
        write_json_artifact(Path(output_files["m20_training_frame_export_blocker_json"]), blocker)
        Path(output_files["m20_training_frame_export_blocker_md"]).write_text(
            _blocker_markdown(blocker),
            encoding="utf-8",
        )
        write_json_artifact(Path(output_files["m20_required_future_export_schema_json"]), schema)
    return make_json_safe(
        {
            "run_dir": str(resolved_run_dir),
            "export_dir": str(export_dir),
            "report": report,
            "honesty_flags": flags,
            "recommendation": recommendation,
            "output_files": output_files,
        }
    )


def _discover_candidate_files(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative_path = path.relative_to(run_dir).as_posix()
        lowered = relative_path.lower()
        if path.suffix.lower() not in {".csv", ".json", ".parquet"}:
            continue
        if any(token in lowered for token in DISCOVERY_TOKENS):
            rows.append(
                {
                    "path": str(path),
                    "relative_path": relative_path,
                    "suffix": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                    "row_source_supported": path.suffix.lower() == ".csv",
                }
            )
    return rows


def _candidate_summaries(
    discovered: Sequence[Mapping[str, Any]],
    configured_market_features: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for source in discovered:
        path = Path(str(source["path"]))
        sample_rows = _read_csv_rows(path, limit=200) if path.suffix.lower() == ".csv" else []
        columns = list(sample_rows[0].keys()) if sample_rows else _json_columns(path)
        audit = _audit_selected_rows(
            sample_rows,
            configured_market_features=configured_market_features,
        )
        rows.append(
            {
                **dict(source),
                "columns": columns,
                "row_count": _count_csv_rows(path) if path.suffix.lower() == ".csv" else 0,
                "timestamp_keys_present": any(column in columns for column in TIMESTAMP_COLUMNS),
                "symbol_key_present": "symbol" in columns,
                "fold_key_present": "fold_index" in columns or "fold" in columns,
                "safe_market_feature_count_sample": len(audit["market_feature_columns"]),
                "score_policy_feature_count_sample": len(audit["score_policy_columns"]),
            }
        )
    return rows


def _select_source(sources: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    scored: list[tuple[tuple[int, int, int, str], dict[str, Any]]] = []
    for source in sources:
        if not source.get("row_source_supported"):
            continue
        if int(source.get("safe_market_feature_count_sample", 0)) <= 0:
            continue
        has_keys = bool(source.get("timestamp_keys_present") and source.get("symbol_key_present"))
        if not has_keys:
            continue
        score = (
            1,
            int(source.get("safe_market_feature_count_sample", 0)),
            _source_rank(str(source["relative_path"])),
            str(source["relative_path"]),
        )
        scored.append((score, dict(source)))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _source_rank(relative_path: str) -> int:
    lowered = relative_path.lower()
    if "training_frame" in lowered or "model_input" in lowered:
        return 5
    if "feature_ohlc" in lowered or "features" in lowered:
        return 4
    if "dataset" in lowered:
        return 3
    if "fold" in lowered:
        return 2
    if "oof" in lowered:
        return 1
    return 0


def _audit_selected_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    configured_market_features: Sequence[str],
) -> dict[str, Any]:
    columns = list(rows[0].keys()) if rows else []
    numeric_columns = [
        column for column in columns
        if all(_try_float(row.get(column)) is not None for row in rows[: min(len(rows), 100)])
    ]
    configured_set = set(configured_market_features)
    configured_order = {
        column: index for index, column in enumerate(configured_market_features)
    }
    excluded = []
    market_features = []
    score_policy_columns = []
    for column in numeric_columns:
        reason = _exclusion_reason(column)
        if reason:
            excluded.append({"column": column, "reason": reason})
            if reason == "score_policy_output":
                score_policy_columns.append(column)
            continue
        if configured_set and column not in configured_set:
            excluded.append({"column": column, "reason": "not_configured_market_feature"})
            continue
        if not configured_set and not _looks_like_market_feature(column):
            excluded.append({"column": column, "reason": "not_market_feature"})
            continue
        market_features.append(column)
    join_keys = _join_keys(columns)
    duplicate_count = _duplicate_count(rows, join_keys)
    if configured_order:
        market_features = sorted(
            market_features,
            key=lambda column: configured_order.get(column, len(configured_order)),
        )
    return {
        "available_columns": columns,
        "numeric_columns": numeric_columns,
        "market_feature_columns": market_features,
        "score_policy_columns": score_policy_columns,
        "excluded_columns": [row["column"] for row in excluded],
        "excluded": excluded,
        "join_keys": join_keys,
        "duplicate_key_count": duplicate_count,
        "timestamp_safe": bool(any(column in columns for column in TIMESTAMP_COLUMNS)),
        "oof_score_columns_excluded": bool(score_policy_columns),
    }


def _export_rows(
    rows: Sequence[Mapping[str, str]],
    audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not audit["market_feature_columns"]:
        return []
    if "symbol" not in audit["available_columns"]:
        return []
    if not any(column in audit["available_columns"] for column in TIMESTAMP_COLUMNS):
        return []
    output = []
    key_columns = [column for column in KEY_COLUMNS if column in audit["available_columns"]]
    for row in rows:
        output_row = {column: row.get(column, "") for column in key_columns}
        for column in audit["market_feature_columns"]:
            output_row[column] = row.get(column, "")
        output.append(output_row)
    return output


def _honesty_flags(
    *,
    selected: Mapping[str, Any] | None,
    audit: Mapping[str, Any],
    export_rows: Sequence[Mapping[str, Any]],
    candidate_summaries: Sequence[Mapping[str, Any]],
) -> list[str]:
    flags = ["NO_PROMOTION_EFFECT", "NO_RUNTIME_EFFECT", "RESEARCH_ONLY_TRAINING_FRAME_EXPORT"]
    if selected:
        flags.append("ORIGINAL_TRAINING_FRAME_FOUND")
    else:
        flags.append("ORIGINAL_TRAINING_FRAME_MISSING")
    if export_rows:
        flags.extend(["MARKET_FEATURES_EXPORTED", "TRAINING_FRAME_EXPORT_READY"])
    else:
        flags.extend(["MARKET_FEATURES_MISSING", "TRAINING_FRAME_EXPORT_BLOCKED"])
    if not export_rows and any(
        int(source.get("score_policy_feature_count_sample", 0)) > 0
        for source in candidate_summaries
    ):
        flags.append("ONLY_OOF_SCORE_FEATURES_AVAILABLE")
    if audit["score_policy_columns"] or any(
        int(source.get("score_policy_feature_count_sample", 0)) > 0
        for source in candidate_summaries
    ):
        flags.append("SCORE_COLUMNS_EXCLUDED_FROM_MARKET_FEATURES")
    if audit["excluded_columns"] or any(
        int(source.get("safe_market_feature_count_sample", 0)) == 0
        for source in candidate_summaries
    ):
        flags.append("POSSIBLE_LEAKAGE_COLUMNS_EXCLUDED")
    if any(source.get("timestamp_keys_present") for source in candidate_summaries):
        flags.append("TIMESTAMP_KEYS_PRESENT")
    else:
        flags.append("TIMESTAMP_KEYS_MISSING")
    if any(source.get("fold_key_present") for source in candidate_summaries):
        flags.append("FOLD_KEYS_PRESENT")
    else:
        flags.append("FOLD_KEYS_MISSING")
    return sorted(dict.fromkeys(flags))


def _recommend(flags: Sequence[str]) -> str:
    if "TRAINING_FRAME_EXPORT_READY" in flags:
        return "A. train tiny research-only tabular baseline on exported market features next"
    if "ORIGINAL_TRAINING_FRAME_MISSING" in flags:
        return "B. rerun M20 scoring with the new training-frame export hook enabled"
    if "TIMESTAMP_KEYS_MISSING" in flags or "FOLD_KEYS_MISSING" in flags:
        return "C. fix missing timestamp/symbol/fold keys in the training pipeline"
    if "ONLY_OOF_SCORE_FEATURES_AVAILABLE" in flags:
        return "D. keep only score-policy diagnostics for the current artifact"
    return "E. stop M20 model chase and package as negative research"


def _write_success_outputs(
    output_files: Mapping[str, str],
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> None:
    write_csv_artifact(Path(output_files["m20_training_frame_features_csv"]), list(rows))
    write_csv_artifact(Path(output_files["m20_training_frame_keys_csv"]), _key_rows(rows))
    write_json_artifact(
        Path(output_files["m20_training_frame_feature_columns_json"]),
        {"feature_columns": audit["market_feature_columns"]},
    )
    write_json_artifact(Path(output_files["m20_training_frame_export_manifest_json"]), manifest)
    write_json_artifact(Path(output_files["m20_training_frame_export_report_json"]), report)
    Path(output_files["m20_training_frame_export_report_md"]).write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    write_csv_artifact(
        Path(output_files["m20_training_frame_preview_csv"]),
        list(rows[:PREVIEW_ROW_LIMIT]),
    )


def _output_files(export_dir: Path, *, success: bool) -> dict[str, str]:
    if success:
        return {
            "m20_training_frame_features_csv": str(export_dir / "m20_training_frame_features.csv"),
            "m20_training_frame_keys_csv": str(export_dir / "m20_training_frame_keys.csv"),
            "m20_training_frame_feature_columns_json": str(
                export_dir / "m20_training_frame_feature_columns.json"
            ),
            "m20_training_frame_export_manifest_json": str(
                export_dir / "m20_training_frame_export_manifest.json"
            ),
            "m20_training_frame_export_report_json": str(
                export_dir / "m20_training_frame_export_report.json"
            ),
            "m20_training_frame_export_report_md": str(
                export_dir / "m20_training_frame_export_report.md"
            ),
            "m20_training_frame_preview_csv": str(export_dir / "m20_training_frame_preview.csv"),
        }
    return {
        "m20_training_frame_export_blocker_json": str(
            export_dir / "m20_training_frame_export_blocker.json"
        ),
        "m20_training_frame_export_blocker_md": str(
            export_dir / "m20_training_frame_export_blocker.md"
        ),
        "m20_required_future_export_schema_json": str(
            export_dir / "m20_required_future_export_schema.json"
        ),
    }


def _future_export_schema(configured_market_features: Sequence[str]) -> dict[str, Any]:
    return {
        "required_keys": ["symbol", "interval_begin"],
        "recommended_keys": ["row_id", "fold_index", "as_of_time"],
        "feature_columns": list(configured_market_features),
        "excluded_columns": [
            "label",
            "target",
            "future_return_*",
            "barrier_*",
            "y_true",
            "prob_up",
            "confidence",
            "y_pred",
            "long_trade_taken",
            "long_only_*",
        ],
        "notes": [
            "Rows must be prediction-time safe.",
            "Feature timestamps must be less than or equal to the label decision timestamp.",
            "Score/policy outputs may be exported separately, but not as market features.",
        ],
    }


def _manifest(
    report: Mapping[str, Any],
    configured_market_features: Sequence[str],
) -> dict[str, Any]:
    return {
        "run_dir": report["run_dir"],
        "export_dir": report["export_dir"],
        "selected_source_file": (
            report["selected_source_file"]["path"] if report["selected_source_file"] else None
        ),
        "selected_join_keys": report["selected_join_keys"],
        "safe_market_feature_columns": report["safe_market_feature_columns"],
        "configured_market_features": list(configured_market_features),
        "honesty_flags": report["honesty_flags"],
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "recommendation": report["recommendation"],
        "output_files": report["output_files"],
    }


def _blocker_reason(flags: Sequence[str]) -> str:
    if "ORIGINAL_TRAINING_FRAME_MISSING" in flags:
        return "No row-level market feature frame exists in the completed run artifacts."
    if "MARKET_FEATURES_MISSING" in flags:
        return "Candidate row files do not contain safe configured market feature columns."
    return "Training-frame export is blocked by missing safe source rows."


def _blocker_markdown(blocker: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# M20 Training Frame Export Blocker",
            "",
            f"- Blocker: `{blocker['blocker']}`",
            f"- Discovered candidate files: `{len(blocker['discovered_candidate_files'])}`",
            f"- Honesty flags: `{', '.join(blocker['honesty_flags'])}`",
            f"- Recommendation: `{blocker['recommendation']}`",
            "",
            "The completed artifact preserves feature names in manifests but does not preserve "
            "a row-level OHLC-derived feature frame suitable for market-feature baselines.",
            "",
        ]
    )


def _report_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# M20 Training Frame Feature Export",
            "",
            f"- Selected source: `{report['selected_source_file']['relative_path']}`",
            f"- Row count: `{report['row_count']}`",
            f"- Safe market feature count: `{report['safe_market_feature_count']}`",
            f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
            f"- Recommendation: `{report['recommendation']}`",
            "",
            "This export is research-only and does not affect runtime inference, registry "
            "authority, promotion, or execution.",
            "",
        ]
    )


def _configured_market_features(run_dir: Path) -> list[str]:
    for filename in ("feature_columns.json", "run_config.json", "dataset_manifest.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        payload = _read_json(path)
        for key in ("numeric_feature_columns", "configured_feature_columns", "feature_columns"):
            values = payload.get(key)
            if isinstance(values, list):
                return [
                    str(value) for value in values
                    if str(value) != "symbol" and not _exclusion_reason(str(value))
                ]
    return []


def _exclusion_reason(column: str) -> str | None:
    lowered = column.lower()
    if lowered in EXACT_LEAKAGE_COLUMNS:
        return "realized_outcome"
    if lowered in SCORE_POLICY_COLUMNS:
        return "score_policy_output"
    if column in IDENTIFIER_COLUMNS or column in KEY_COLUMNS or column in TIMESTAMP_COLUMNS:
        return "key_or_identifier"
    for token in LEAKAGE_NAME_TOKENS:
        if token in lowered:
            return f"leakage_token:{token}"
    if "barrier" in lowered or "event" in lowered or "post" in lowered:
        return "post_event_or_barrier_metadata"
    return None


def _looks_like_market_feature(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in MARKET_HINT_TOKENS)


def _join_keys(columns: Sequence[str]) -> list[str]:
    if "symbol" not in columns:
        return []
    for timestamp_column in ("interval_begin", "timestamp", "time"):
        if timestamp_column in columns:
            keys = ["symbol", timestamp_column]
            if "fold_index" in columns:
                keys.append("fold_index")
            return keys
    return []


def _duplicate_count(rows: Sequence[Mapping[str, str]], keys: Sequence[str]) -> int:
    if not keys:
        return 0
    counts = Counter(tuple(str(row.get(key, "")) for key in keys) for row in rows)
    return sum(count - 1 for count in counts.values() if count > 1)


def _coverage(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row.get(column, "")) for row in rows if row.get(column)).items())
    )


def _timestamp_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        str(row.get("interval_begin") or row.get("timestamp") or row.get("time") or "")
        for row in rows
        if row.get("interval_begin") or row.get("timestamp") or row.get("time")
    ]
    parsed = [_parse_timestamp(value) for value in values]
    valid = [value for value in parsed if value is not None]
    return {
        "timestamp_rows": len(values),
        "min_timestamp": min(valid).isoformat() if valid else None,
        "max_timestamp": max(valid).isoformat() if valid else None,
    }


def _missing_rates(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> dict[str, float]:
    return {
        column: _safe_ratio(sum(1 for row in rows if row.get(column) in ("", None)), len(rows))
        for column in feature_columns
    }


def _key_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {column: row.get(column, "") for column in KEY_COLUMNS if column in row}
        for row in rows
    ]


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


def _json_columns(path: Path) -> list[str]:
    payload = _read_json(path)
    columns: list[str] = []
    _collect_columns(payload, columns)
    return sorted(dict.fromkeys(columns))


def _collect_columns(value: Any, output: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if "column" in str(key).lower() and isinstance(nested, list):
                output.extend(str(item) for item in nested if isinstance(item, str))
            else:
                _collect_columns(nested, output)
    elif isinstance(value, list):
        for item in value:
            _collect_columns(item, output)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
