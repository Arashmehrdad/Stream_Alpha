"""Research-only confirmation-window override helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Iterable, Mapping, Sequence

from app.common.time import to_rfc3339


HONESTY_FLAGS = (
    "RESEARCH_ONLY_CONFIRMATION_WINDOW_OVERRIDE",
    "MANUAL_LONG_RUN_ONLY",
    "NO_RUNTIME_EFFECT",
    "NO_PROMOTION_EFFECT",
    "NO_REGISTRY_WRITE",
    "NOT_PROMOTABLE",
    "NO_PROFITABILITY_CLAIM",
    "DEFAULT_BEHAVIOR_UNCHANGED",
    "CONFIRMATION_WINDOW_OVERRIDE_SUPPORTED",
    "CONFIRMATION_WINDOW_REQUIRES_MANUAL_RUN",
)
SAFE_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


@dataclass(frozen=True, slots=True)
class ConfirmationWindowOverride:
    """Validated confirmation-window override metadata."""

    start: datetime
    end: datetime
    tag: str

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-safe override metadata."""
        return {
            "override_enabled": True,
            "confirmation_window_start": to_rfc3339(self.start),
            "confirmation_window_end": to_rfc3339(self.end),
            "confirmation_tag": self.tag,
            "honesty_flags": list(HONESTY_FLAGS),
        }


def build_confirmation_window_override(
    *,
    start: str | None = None,
    end: str | None = None,
    tag: str | None = None,
) -> ConfirmationWindowOverride | None:
    """Validate CLI/config confirmation-window override inputs."""
    if start is None and end is None and tag is None:
        return None
    if not start or not end:
        raise ValueError(
            "confirmation window override requires both start and end timestamps"
        )
    parsed_start = parse_utc_timestamp(start)
    parsed_end = parse_utc_timestamp(end)
    if parsed_start >= parsed_end:
        raise ValueError("confirmation window start must be before end")
    safe_tag = validate_confirmation_tag(tag or "confirmation")
    return ConfirmationWindowOverride(
        start=parsed_start,
        end=parsed_end,
        tag=safe_tag,
    )


def parse_utc_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to UTC."""
    raw = value.strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as error:
        raise ValueError(f"Invalid confirmation timestamp: {value}") from error
    if parsed.tzinfo is None:
        raise ValueError("confirmation timestamps must include timezone information")
    return parsed.astimezone(timezone.utc)


def validate_confirmation_tag(tag: str) -> str:
    """Return a safe reporting-only confirmation tag."""
    if not SAFE_TAG_PATTERN.fullmatch(tag):
        raise ValueError(
            "confirmation tag may contain only letters, numbers, dot, dash, "
            "and underscore, with max length 64"
        )
    return tag


def filter_rows_for_confirmation_window(
    rows: Sequence[Any],
    override: ConfirmationWindowOverride,
    *,
    timestamp_attr: str = "as_of_time",
) -> list[Any]:
    """Return rows inside the requested half-open confirmation window."""
    selected = [
        row for row in rows
        if override.start <= getattr(row, timestamp_attr) < override.end
    ]
    if not selected:
        raise ValueError("CONFIRMATION_WINDOW_EMPTY")
    return selected


def confirmation_window_validation(
    *,
    selected_rows: Sequence[Any],
    all_symbols: Iterable[str],
    override: ConfirmationWindowOverride,
    original_window: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build validation metadata for a selected confirmation window."""
    symbols = sorted({str(getattr(row, "symbol", "")) for row in selected_rows})
    expected_symbols = sorted(str(symbol) for symbol in all_symbols)
    flags = ["CONFIRMATION_WINDOW_VALIDATED"]
    if set(symbols) != set(expected_symbols):
        flags.append("CONFIRMATION_WINDOW_SYMBOL_COVERAGE_INCOMPLETE")
    distinct = "unknown"
    overlaps = "unknown"
    if original_window:
        original_start = _metadata_timestamp(original_window.get("start"))
        original_end = _metadata_timestamp(original_window.get("end"))
        if original_start and original_end:
            distinct = not (
                override.start == original_start and override.end == original_end
            )
            overlaps = override.start < original_end and original_start < override.end
            if not distinct:
                flags.append("CONFIRMATION_WINDOW_NOT_DISTINCT")
            if overlaps:
                flags.append("CONFIRMATION_WINDOW_OVERLAPS_ORIGINAL")
    return {
        **override.to_metadata(),
        "selected_row_count": len(selected_rows),
        "selected_symbol_coverage": symbols,
        "expected_symbol_coverage": expected_symbols,
        "distinct_from_original": distinct,
        "overlap_with_original": overlaps,
        "honesty_flags": sorted(dict.fromkeys([*override.to_metadata()["honesty_flags"], *flags])),
    }


def _metadata_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return parse_utc_timestamp(str(value))
    except ValueError:
        return None
