"""Shared helpers for M20 research-only policy artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
MIN_POLICY_ROWS = 1000
KEY_WITH_FOLD = ("fold_index", "symbol", "interval_begin")
KEY_NO_FOLD = ("symbol", "interval_begin")
FORBIDDEN_SELECTION_INPUTS = (
    "future_return",
    "future_return_3",
    "gross_value_proxy",
    "net_value_proxy",
    "long_only_gross_value_proxy",
    "long_only_net_value_proxy",
    "fee_exceedance_label",
    "triple_barrier_label",
    "y_true",
)


def vol_scaled_dir(source_run_dir: Path) -> Path:
    """Return the M20 vol-scaled research directory."""
    return Path(source_run_dir).resolve() / "research_labels" / "vol_scaled"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows, returning an empty list when the file is absent."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_csv_header(path: Path) -> list[str]:
    """Read only a CSV header, returning an empty list when absent."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def read_json_payload(path: Path) -> dict[str, Any]:
    """Read JSON payload, returning an empty dict when absent."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_report_bundle(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    report_path: Path,
    report: Mapping[str, Any],
) -> None:
    """Write the standard manifest/report JSON pair."""
    write_json_artifact(manifest_path, manifest)
    write_json_artifact(report_path, report)


def preferred_join_keys(
    left_columns: Sequence[str],
    right_columns: Sequence[str],
) -> tuple[str, ...]:
    """Prefer fold-aware joins when both sides contain fold_index."""
    left = set(left_columns)
    right = set(right_columns)
    if all(column in left and column in right for column in KEY_WITH_FOLD):
        return KEY_WITH_FOLD
    if all(column in left and column in right for column in KEY_NO_FOLD):
        return KEY_NO_FOLD
    return ()


def row_key(row: Mapping[str, Any], keys: Sequence[str] = KEY_WITH_FOLD) -> tuple[str, ...]:
    """Build a stable string key from a row."""
    return tuple(str(row.get(column, "")) for column in keys)


def duplicate_key_count(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> int:
    """Return the number of duplicate keys in rows."""
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in rows:
        key = row_key(row, keys)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def keyed_rows(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    """Index rows by key, keeping the first duplicate deterministically."""
    index: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        index.setdefault(row_key(row, keys), row)
    return index


def matched_key_count(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> int:
    """Count matching keys between two row collections."""
    right_keys = {row_key(row, keys) for row in right_rows}
    return sum(1 for row in left_rows if row_key(row, keys) in right_keys)


def economics_metrics(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, Any]:
    """Compute net-proxy economics metrics for selected rows."""
    values = [to_float(row.get(column)) for row in rows if present(row.get(column))]
    if not values:
        return empty_economics()
    return {
        "mean_net_value_proxy": mean(values),
        "cumulative_net_value_proxy": sum(values),
        "max_drawdown_proxy": max_drawdown(values),
        "win_rate": sum(1 for value in values if value > 0.0) / len(values),
    }


def empty_economics() -> dict[str, Any]:
    """Return blank economics fields."""
    return {
        "mean_net_value_proxy": "",
        "cumulative_net_value_proxy": "",
        "max_drawdown_proxy": "",
        "win_rate": "",
    }


def max_drawdown(values: Sequence[float]) -> float:
    """Compute cumulative-series max drawdown."""
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = min(drawdown, cumulative - peak)
    return drawdown


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean."""
    return sum(values) / len(values) if values else 0.0


def quarter(timestamp: str) -> str:
    """Return YYYYQn from an ISO-like timestamp."""
    if len(timestamp) < 7:
        return ""
    try:
        month = int(timestamp[5:7])
    except ValueError:
        return ""
    return f"{timestamp[:4]}Q{((month - 1) // 3) + 1}"


def present(value: Any) -> bool:
    """Return whether a CSV value is present."""
    return value not in ("", None)


def to_float(value: Any) -> float:
    """Convert a value to float, defaulting to zero."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_int(value: Any) -> int:
    """Convert a value to int, defaulting to zero."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def truthy(value: Any) -> bool:
    """Interpret common CSV boolean/integer truthy values."""
    return str(value).strip().lower() in {"1", "true", "yes", "take", "buy"}


def group_rows(
    rows: Iterable[Mapping[str, Any]],
    field: str,
) -> dict[str, list[Mapping[str, Any]]]:
    """Group rows by a direct or derived slice field."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = str(row.get(field, ""))
        if field == "month":
            value = str(row.get("interval_begin", ""))[:7]
        if field == "quarter":
            value = quarter(str(row.get("interval_begin", "")))
        grouped.setdefault(value, []).append(row)
    return grouped


__all__ = [
    "FORBIDDEN_SELECTION_INPUTS",
    "HONESTY_FLAGS",
    "KEY_NO_FOLD",
    "KEY_WITH_FOLD",
    "MIN_POLICY_ROWS",
    "duplicate_key_count",
    "economics_metrics",
    "empty_economics",
    "group_rows",
    "keyed_rows",
    "matched_key_count",
    "preferred_join_keys",
    "present",
    "read_csv_header",
    "read_csv_rows",
    "read_json_payload",
    "row_key",
    "to_float",
    "to_int",
    "truthy",
    "vol_scaled_dir",
    "write_csv_artifact",
    "write_json_artifact",
    "write_report_bundle",
]
