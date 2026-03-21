"""Explicit reliability artifact writers for M13."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.common.serialization import make_json_safe


def write_json_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one deterministic JSON artifact."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(make_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def append_jsonl_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one deterministic JSONL event row."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(make_json_safe(payload), sort_keys=True))
        output_file.write("\n")
