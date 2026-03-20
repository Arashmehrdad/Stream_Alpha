"""Rollback workflow for the file-based M7 model registry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.common.time import to_rfc3339, utc_now
from app.training.registry import (
    append_registry_history,
    load_current_registry_entry,
    load_registry_entry,
    write_current_registry_entry,
)


def rollback_to_model_version(
    model_version: str,
    *,
    registry_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically point the current registry champion back to one promoted model version."""
    resolved_model_version = str(model_version).strip()
    if not resolved_model_version:
        raise ValueError("model_version cannot be empty")

    current_entry = load_current_registry_entry(registry_root)
    if current_entry is not None and current_entry["model_version"] == resolved_model_version:
        raise ValueError(f"Model version {resolved_model_version} is already current")

    target_entry = load_registry_entry(resolved_model_version, registry_root=registry_root)
    current_payload = {
        **target_entry,
        "activated_at": to_rfc3339(utc_now()),
        "activation_reason": "ROLLBACK",
    }
    write_current_registry_entry(current_payload, registry_root=registry_root)
    append_registry_history(
        {
            "event_time": to_rfc3339(utc_now()),
            "event_type": "ROLLBACK",
            "model_version": resolved_model_version,
            "previous_model_version": (
                None if current_entry is None else current_entry["model_version"]
            ),
        },
        registry_root=registry_root,
    )
    return current_payload


def main() -> None:
    """Rollback the local registry to one previously promoted model version."""
    parser = argparse.ArgumentParser(description="Rollback Stream Alpha champion model")
    parser.add_argument("--model-version", required=True, help="Promoted model_version to restore")
    arguments = parser.parse_args()

    try:
        current_payload = rollback_to_model_version(arguments.model_version)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(str(current_payload["model_artifact_path"]))


if __name__ == "__main__":
    main()
