"""Deterministic artifact writers for the Stream Alpha M19 adaptation layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_adaptation_artifact_root(root_dir: str | Path) -> Path:
    """Create and return the canonical adaptation artifact root."""
    path = Path(root_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write one deterministic JSON artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def append_jsonl_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    """Append one deterministic JSONL row."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
    return target


def write_markdown_artifact(path: str | Path, lines: list[str]) -> Path:
    """Write one deterministic Markdown artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target
