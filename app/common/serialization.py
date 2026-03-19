"""Serialization helpers for dataclass-backed event payloads."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any

from app.common.time import to_rfc3339


def make_json_safe(value: Any) -> Any:
    """Convert nested values into JSON-safe structures."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return to_rfc3339(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def model_to_dict(model: Any) -> dict[str, Any]:
    """Convert a dataclass model into a JSON-safe dictionary."""
    if not is_dataclass(model):
        raise TypeError("model_to_dict expects a dataclass instance")
    return make_json_safe(asdict(model))


def serialize_model(model: Any) -> bytes:
    """Serialize a dataclass model into a UTF-8 JSON payload."""
    return json.dumps(model_to_dict(model), sort_keys=True).encode("utf-8")
