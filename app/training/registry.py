"""File-based model registry and run-manifest helpers for M7."""

# pylint: disable=duplicate-code

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339, utc_now


REQUIRED_RUN_ARTIFACTS = (
    "dataset_manifest.json",
    "feature_columns.json",
    "fold_metrics.csv",
    "model.joblib",
    "oof_predictions.csv",
    "run_config.json",
    "summary.json",
)


def repo_root() -> Path:
    """Return the repository root from the current module location."""
    return Path(__file__).resolve().parents[2]


def default_registry_root() -> Path:
    """Return the default file-based registry root."""
    return repo_root() / "artifacts" / "registry"


def current_registry_path(registry_root: Path | None = None) -> Path:
    """Return the current champion pointer path."""
    root = default_registry_root() if registry_root is None else registry_root
    return root / "current.json"


def history_registry_path(registry_root: Path | None = None) -> Path:
    """Return the registry history log path."""
    root = default_registry_root() if registry_root is None else registry_root
    return root / "history.jsonl"


def registry_models_root(registry_root: Path | None = None) -> Path:
    """Return the immutable promoted-model storage root."""
    root = default_registry_root() if registry_root is None else registry_root
    return root / "models"


def derive_model_version(run_dir: Path) -> str:
    """Return a deterministic model version from the source run directory."""
    resolved_run_dir = Path(run_dir).resolve()
    return f"{resolved_run_dir.parent.name}-{resolved_run_dir.name}"


def required_run_artifact_paths(run_dir: Path) -> dict[str, Path]:
    """Validate the required artifact files for a training run directory."""
    resolved_run_dir = Path(run_dir).resolve()
    missing_files = [
        file_name
        for file_name in REQUIRED_RUN_ARTIFACTS
        if not (resolved_run_dir / file_name).is_file()
    ]
    if missing_files:
        missing_list = ", ".join(missing_files)
        raise ValueError(
            f"Run artifact directory is incomplete at {resolved_run_dir}: missing {missing_list}",
        )
    return {
        file_name: resolved_run_dir / file_name
        for file_name in REQUIRED_RUN_ARTIFACTS
    }


def load_run_manifest(run_dir: Path) -> dict[str, Any]:
    """Load a run manifest if it exists, otherwise build it from source artifacts."""
    manifest_path = Path(run_dir).resolve() / "run_manifest.json"
    if manifest_path.is_file():
        return read_json(manifest_path)
    return build_run_manifest(run_dir)


def build_run_manifest(run_dir: Path) -> dict[str, Any]:
    """Build a deterministic manifest describing a training run's artifacts and protocol."""
    artifact_paths = required_run_artifact_paths(run_dir)
    run_config = read_json(artifact_paths["run_config.json"])
    dataset_manifest = read_json(artifact_paths["dataset_manifest.json"])
    feature_columns = read_json(artifact_paths["feature_columns.json"])
    summary = read_json(artifact_paths["summary.json"])
    model_payload = _load_model_payload(artifact_paths["model.joblib"])

    winner_name = str(summary["winner"]["model_name"])
    winner_metrics = dict(summary["models"][winner_name])
    if model_payload["model_name"] != winner_name:
        raise ValueError(
            "Model artifact winner does not match summary winner for "
            f"{Path(run_dir).resolve()}",
        )

    return make_json_safe(
        {
            "generated_at": to_rfc3339(utc_now()),
            "run_id": Path(run_dir).resolve().name,
            "run_dir": str(Path(run_dir).resolve()),
            "source_run_kind": Path(run_dir).resolve().parent.name,
            "winner": {
                "model_name": winner_name,
                "trained_at": model_payload["trained_at"],
                "metrics": winner_metrics,
                "selection_rule": summary["winner"]["selection_rule"],
            },
            "feature_schema": {
                "configured_feature_columns": list(
                    feature_columns["configured_feature_columns"],
                ),
                "categorical_feature_columns": list(
                    feature_columns["categorical_feature_columns"],
                ),
                "numeric_feature_columns": list(
                    feature_columns["numeric_feature_columns"],
                ),
                "expanded_feature_names": list(
                    feature_columns["expanded_feature_names"],
                ),
            },
            "evaluation_protocol": {
                "source_table": run_config["source_table"],
                "symbols": list(run_config["symbols"]),
                "time_column": run_config["time_column"],
                "interval_column": run_config["interval_column"],
                "close_column": run_config["close_column"],
                "label_horizon_candles": run_config["label_horizon_candles"],
                "purge_gap_candles": run_config["purge_gap_candles"],
                "test_folds": run_config["test_folds"],
                "first_train_fraction": run_config["first_train_fraction"],
                "test_fraction": run_config["test_fraction"],
                "round_trip_fee_bps": run_config["round_trip_fee_bps"],
            },
            "dataset_manifest": dict(dataset_manifest),
            "artifact_files": {
                file_name: str(path.resolve())
                for file_name, path in artifact_paths.items()
            },
        }
    )


def write_run_manifest(run_dir: Path) -> Path:
    """Write the manifest file for one training run and return its path."""
    manifest_path = Path(run_dir).resolve() / "run_manifest.json"
    write_json_atomic(manifest_path, build_run_manifest(run_dir))
    return manifest_path


def resolve_inference_model_path(
    model_override: str,
    *,
    registry_root: Path | None = None,
) -> str:
    """Resolve the model path from an explicit override or the current registry champion."""
    return str(
        resolve_inference_model_metadata(
            model_override,
            registry_root=registry_root,
        )["model_artifact_path"]
    )


def resolve_inference_model_metadata(
    model_override: str,
    *,
    registry_root: Path | None = None,
) -> dict[str, str]:
    """Resolve the active inference model path plus stable version metadata."""
    if model_override.strip():
        override_path = Path(model_override).expanduser().resolve()
        if not override_path.is_file():
            raise ValueError(f"INFERENCE_MODEL_PATH does not exist: {override_path}")
        model_version, model_version_source = _derive_override_model_version(override_path)
        return {
            "model_artifact_path": str(override_path),
            "model_version": model_version,
            "model_version_source": model_version_source,
        }

    current_entry = load_current_registry_entry(registry_root)
    if current_entry is None:
        raise ValueError(
            "INFERENCE_MODEL_PATH is empty and no promoted champion exists at "
            f"{current_registry_path(registry_root).resolve()}",
        )

    model_path = Path(str(current_entry["model_artifact_path"])).expanduser().resolve()
    if not model_path.is_file():
        raise ValueError(f"Registry champion model artifact does not exist: {model_path}")

    model_version = str(current_entry.get("model_version", "")).strip()
    if not model_version:
        model_version, _ = _derive_override_model_version(model_path)
    return {
        "model_artifact_path": str(model_path),
        "model_version": model_version,
        "model_version_source": "REGISTRY_CURRENT",
    }


def load_current_registry_entry(registry_root: Path | None = None) -> dict[str, Any] | None:
    """Return the current registry champion metadata when present."""
    path = current_registry_path(registry_root)
    if not path.is_file():
        return None
    return read_json(path)


def load_registry_entry(
    model_version: str,
    *,
    registry_root: Path | None = None,
) -> dict[str, Any]:
    """Load one immutable promoted-model registry entry by version."""
    entry_path = registry_models_root(registry_root) / model_version / "registry_entry.json"
    if not entry_path.is_file():
        raise ValueError(
            f"Promoted model version {model_version} was not found at {entry_path.resolve()}",
        )
    return read_json(entry_path)


def copy_run_snapshot_to_registry(
    *,
    source_run_dir: Path,
    model_version: str,
    comparison_payload: dict[str, Any] | None = None,
    registry_root: Path | None = None,
) -> dict[str, Any]:
    """Copy one immutable promoted-model snapshot into the registry."""
    resolved_source_run_dir = Path(source_run_dir).resolve()
    artifact_paths = required_run_artifact_paths(resolved_source_run_dir)
    manifest = build_run_manifest(resolved_source_run_dir)
    model_dir = registry_models_root(registry_root) / model_version
    if model_dir.exists():
        raise ValueError(f"Promoted model version already exists: {model_version}")

    model_dir.mkdir(parents=True, exist_ok=False)
    for file_name, source_path in artifact_paths.items():
        shutil.copy2(source_path, model_dir / file_name)

    comparison_path = _copy_or_write_comparison(
        source_run_dir=resolved_source_run_dir,
        target_dir=model_dir,
        comparison_payload=comparison_payload,
    )
    write_json_atomic(model_dir / "run_manifest.json", manifest)

    entry = {
        "model_version": model_version,
        "model_name": manifest["winner"]["model_name"],
        "trained_at": manifest["winner"]["trained_at"],
        "winner_metrics": manifest["winner"]["metrics"],
        "model_artifact_path": str((model_dir / "model.joblib").resolve()),
        "summary_path": str((model_dir / "summary.json").resolve()),
        "feature_columns_path": str((model_dir / "feature_columns.json").resolve()),
        "run_manifest_path": str((model_dir / "run_manifest.json").resolve()),
        "comparison_path": None if comparison_path is None else str(comparison_path.resolve()),
        "source_run_dir": str(resolved_source_run_dir),
        "source_run_id": resolved_source_run_dir.name,
        "source_run_kind": manifest["source_run_kind"],
        "promoted_at": to_rfc3339(utc_now()),
    }
    write_json_atomic(model_dir / "registry_entry.json", entry)
    return entry


def write_current_registry_entry(
    entry: dict[str, Any],
    *,
    registry_root: Path | None = None,
) -> Path:
    """Atomically update the current registry pointer."""
    current_path = current_registry_path(registry_root)
    write_json_atomic(current_path, entry)
    return current_path


def append_registry_history(
    event: dict[str, Any],
    *,
    registry_root: Path | None = None,
) -> Path:
    """Append one registry lifecycle event to history.jsonl."""
    history_path = history_registry_path(registry_root)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(make_json_safe(event), sort_keys=True))
        handle.write("\n")
    return history_path


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically by replacing the destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(make_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _copy_or_write_comparison(
    *,
    source_run_dir: Path,
    target_dir: Path,
    comparison_payload: dict[str, Any] | None,
) -> Path | None:
    """Copy an existing comparison file or write a generated payload when provided."""
    source_path = source_run_dir / "comparison_vs_champion.json"
    target_path = target_dir / "comparison_vs_champion.json"
    if source_path.is_file():
        shutil.copy2(source_path, target_path)
        return target_path
    if comparison_payload is not None:
        write_json_atomic(target_path, comparison_payload)
        return target_path
    return None


def _load_model_payload(model_path: Path) -> dict[str, Any]:
    """Load and validate the saved model payload metadata used by inference."""
    payload = joblib.load(model_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Model artifact must deserialize into a dictionary: {model_path}")
    required_keys = {
        "model_name",
        "trained_at",
        "feature_columns",
        "expanded_feature_names",
        "model",
    }
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        raise ValueError(
            f"Model artifact is missing required keys at {model_path}: {missing_keys}",
        )
    return {
        "model_name": str(payload["model_name"]),
        "trained_at": str(payload["trained_at"]),
        "feature_columns": tuple(str(column) for column in payload["feature_columns"]),
        "expanded_feature_names": tuple(
            str(name) for name in payload["expanded_feature_names"]
        ),
    }


def _derive_override_model_version(artifact_path: Path) -> tuple[str, str]:
    """Derive a stable model version from a direct model override path."""
    registry_entry_path = artifact_path.parent / "registry_entry.json"
    if registry_entry_path.is_file():
        registry_entry = read_json(registry_entry_path)
        model_version = str(registry_entry.get("model_version", "")).strip()
        if model_version:
            return model_version, "REGISTRY_ENTRY"

    parent = artifact_path.parent.resolve()
    grandparent_name = parent.parent.name.lower()
    if artifact_path.name == "model.joblib" and grandparent_name in {"m3", "m7"}:
        return derive_model_version(parent), "RUN_DIR_DERIVED"

    if artifact_path.stem == "model":
        return parent.name, "MODEL_OVERRIDE_PATH"
    return artifact_path.stem, "MODEL_OVERRIDE_PATH"
