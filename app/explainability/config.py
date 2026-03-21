"""Typed checked-in explainability configuration for M14."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def default_explainability_config_path() -> Path:
    """Return the checked-in explainability config path from the repo root."""
    return Path(__file__).resolve().parents[2] / "configs" / "explainability.yaml"


@dataclass(frozen=True, slots=True)
class ExplainabilityReferenceConfig:
    """Reference-vector settings for deterministic M4-side explanations."""

    artifact_root: str
    reference_filename: str
    top_feature_count: int
    explainable_numeric_features: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExplainabilityConfig:
    """Full checked-in M14 explainability configuration."""

    schema_version: str
    reference: ExplainabilityReferenceConfig


def load_explainability_config(config_path: Path) -> ExplainabilityConfig:
    """Load the checked-in explainability config into typed settings."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Explainability config must deserialize into a mapping")

    reference_payload = _require_mapping(payload, "reference")
    config = ExplainabilityConfig(
        schema_version=str(payload.get("schema_version", "")).strip(),
        reference=ExplainabilityReferenceConfig(
            artifact_root=str(reference_payload["artifact_root"]).strip(),
            reference_filename=str(reference_payload["reference_filename"]).strip(),
            top_feature_count=int(reference_payload["top_feature_count"]),
            explainable_numeric_features=tuple(
                str(value).strip()
                for value in reference_payload["explainable_numeric_features"]
            ),
        ),
    )
    _validate_config(config)
    return config


def _require_mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Explainability config section '{key}' must be a mapping")
    return value


def _validate_config(config: ExplainabilityConfig) -> None:
    if not config.schema_version:
        raise ValueError("schema_version must not be empty")
    if not config.reference.artifact_root:
        raise ValueError("reference.artifact_root must not be empty")
    if not config.reference.reference_filename:
        raise ValueError("reference.reference_filename must not be empty")
    if config.reference.top_feature_count <= 0:
        raise ValueError("reference.top_feature_count must be positive")
    if not config.reference.explainable_numeric_features:
        raise ValueError("reference.explainable_numeric_features must not be empty")
    if any(not feature_name for feature_name in config.reference.explainable_numeric_features):
        raise ValueError("reference.explainable_numeric_features must not contain blanks")
