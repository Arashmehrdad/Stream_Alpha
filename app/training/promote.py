"""Promotion workflow for immutable registry-backed M7 model versions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.common.time import to_rfc3339, utc_now
from app.training.dataset import LEGACY_ARCHIVED_MODEL_NAMES
from app.training.registry import (
    append_registry_history,
    copy_run_snapshot_to_registry,
    derive_model_version,
    load_current_registry_entry,
    read_json,
    write_current_registry_entry,
)


def promote_run(
    run_dir: Path,
    model_version: str | None = None,
    *,
    registry_root: Path | None = None,
) -> dict[str, Any]:
    """Promote one completed training run into the immutable registry."""
    resolved_run_dir = Path(run_dir).resolve()
    resolved_model_version = (
        derive_model_version(resolved_run_dir)
        if model_version is None
        else str(model_version).strip()
    )
    if not resolved_model_version:
        raise ValueError("model_version cannot be empty")

    previous_entry = load_current_registry_entry(registry_root)
    current_entry = previous_entry
    if current_entry is not None and _is_legacy_archived_current_entry(current_entry):
        current_entry = None
    comparison_payload = _load_comparison_payload(resolved_run_dir)
    if current_entry is not None:
        if comparison_payload is None:
            raise ValueError(
                "comparison_vs_champion.json is required before promoting over "
                "an existing champion",
            )
        if comparison_payload["champion"] is None:
            raise ValueError(
                "comparison_vs_champion.json does not reference the current champion",
            )
        if comparison_payload["champion"]["model_version"] != current_entry["model_version"]:
            raise ValueError(
                "comparison_vs_champion.json was not computed against the current "
                "registry champion",
            )
        if not bool(comparison_payload["passed"]):
            reasons = "; ".join(str(reason) for reason in comparison_payload["reasons"])
            raise ValueError(f"Challenger promotion policy failed: {reasons}")

    entry = copy_run_snapshot_to_registry(
        source_run_dir=resolved_run_dir,
        model_version=resolved_model_version,
        comparison_payload=comparison_payload or _bootstrap_comparison_payload(),
        registry_root=registry_root,
    )
    current_payload = {
        **entry,
        "activated_at": to_rfc3339(utc_now()),
        "activation_reason": "PROMOTE",
    }
    write_current_registry_entry(current_payload, registry_root=registry_root)
    append_registry_history(
        {
            "event_time": to_rfc3339(utc_now()),
            "event_type": "PROMOTE",
            "model_version": resolved_model_version,
            "previous_model_version": (
                None if previous_entry is None else previous_entry["model_version"]
            ),
            "source_run_dir": str(resolved_run_dir),
            "comparison_passed": (
                True if comparison_payload is None else comparison_payload["passed"]
            ),
        },
        registry_root=registry_root,
    )
    return current_payload


def main() -> None:
    """Promote one training run directory into the local registry."""
    parser = argparse.ArgumentParser(description="Promote one Stream Alpha model version")
    parser.add_argument("--run-dir", required=True, help="Path to the training run directory")
    parser.add_argument(
        "--model-version",
        required=False,
        help="Explicit immutable model version; defaults to parent-run_id form",
    )
    arguments = parser.parse_args()

    try:
        current_payload = promote_run(
            Path(arguments.run_dir),
            model_version=arguments.model_version,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(str(current_payload["model_artifact_path"]))


def _load_comparison_payload(run_dir: Path) -> dict[str, Any] | None:
    """Load the existing comparison artifact when present."""
    comparison_path = Path(run_dir).resolve() / "comparison_vs_champion.json"
    if not comparison_path.is_file():
        return None
    return read_json(comparison_path)


def _bootstrap_comparison_payload() -> dict[str, Any]:
    """Return the explicit bootstrap comparison payload for first promotion."""
    return {
        "generated_at": to_rfc3339(utc_now()),
        "passed": True,
        "decision": "bootstrap_allowed",
        "policy": None,
        "challenger": None,
        "champion": None,
        "compatibility_checks": None,
        "metric_deltas": None,
        "reasons": [
            "No current champion was registered at promotion time.",
        ],
    }


def _is_legacy_archived_current_entry(entry: dict[str, Any]) -> bool:
    """Treat the archived sklearn current pointer as non-authoritative bootstrap state."""
    return str(entry.get("model_name", "")).strip() in LEGACY_ARCHIVED_MODEL_NAMES


if __name__ == "__main__":
    main()
