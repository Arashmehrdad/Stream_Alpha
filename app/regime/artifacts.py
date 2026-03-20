"""Artifact helpers for the M8 offline regime workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.common.serialization import make_json_safe


REQUIRED_RUN_ARTIFACTS = (
    "thresholds.json",
    "regime_predictions.csv",
    "by_symbol_summary.csv",
    "overall_summary.json",
    "run_config.json",
    "run_manifest.json",
)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON artifact from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically by replacing the destination file."""
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_path.with_suffix(f"{resolved_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(make_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(resolved_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a CSV artifact with an explicit header order."""
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        resolved_path.write_text("", encoding="utf-8")
        return
    field_names = list(rows[0].keys())
    with resolved_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def write_thresholds_artifact(path: Path, payload: dict[str, Any]) -> Path:
    """Persist the serving-oriented per-symbol thresholds artifact."""
    write_json_atomic(path, payload)
    return Path(path)


def load_thresholds_artifact(path: Path) -> dict[str, Any]:
    """Load the thresholds artifact from disk."""
    return read_json(path)


def required_run_artifact_paths(run_dir: Path) -> dict[str, Path]:
    """Validate the required M8 run artifact files."""
    resolved_run_dir = Path(run_dir).resolve()
    missing_files = [
        file_name
        for file_name in REQUIRED_RUN_ARTIFACTS
        if not (resolved_run_dir / file_name).is_file()
    ]
    if missing_files:
        missing_list = ", ".join(missing_files)
        raise ValueError(
            f"Regime artifact directory is incomplete at {resolved_run_dir}: "
            f"missing {missing_list}"
        )
    return {
        file_name: resolved_run_dir / file_name
        for file_name in REQUIRED_RUN_ARTIFACTS
    }
