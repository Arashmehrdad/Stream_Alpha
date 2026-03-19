"""Time parsing and formatting utilities."""

from __future__ import annotations

import re
from datetime import datetime, timezone


_RFC3339_RE = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?"
    r"(?P<offset>Z|[+-]\d{2}:\d{2})$"
)


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def parse_rfc3339(value: str) -> datetime:
    """Parse RFC3339 timestamps, including nanosecond inputs."""
    match = _RFC3339_RE.match(value)
    if match is None:
        raise ValueError(f"Unsupported RFC3339 timestamp: {value}")
    offset = "+00:00" if match.group("offset") == "Z" else match.group("offset")
    fraction = (match.group("fraction") or "")
    microseconds = int((fraction + "000000")[:6])
    timestamp = datetime.fromisoformat(f"{match.group('base')}{offset}")
    return timestamp.replace(microsecond=microseconds)


def to_rfc3339(value: datetime) -> str:
    """Format a datetime as an RFC3339 UTC string."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
