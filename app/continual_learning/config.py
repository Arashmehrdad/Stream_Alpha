"""Typed configuration loading for the Stream Alpha M21 continual learning layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.continual_learning.schemas import ContinualLearningCandidateType


@dataclass(frozen=True, slots=True)
class ArtifactConfig:  # pylint: disable=too-many-instance-attributes
    """Canonical M21 artifact paths."""

    root_dir: str
    summary_path: str
    drift_caps_summary_path: str
    current_profile_path: str
    promotions_history_path: str
    events_history_path: str
    reports_dir: str
    experiments_dir: str


@dataclass(frozen=True, slots=True)
class ContinualLearningConfig:
    """Checked-in M21 continual learning configuration."""

    enabled: bool
    candidate_types: tuple[ContinualLearningCandidateType, ...]
    live_eligible_candidate_types: tuple[ContinualLearningCandidateType, ...]
    shadow_only_candidate_types: tuple[ContinualLearningCandidateType, ...]
    artifacts: ArtifactConfig


def default_continual_learning_config_path() -> Path:
    """Return the checked-in M21 continual learning config path."""
    return Path("configs/continual_learning.yaml")


def load_continual_learning_config(config_path: Path) -> ContinualLearningConfig:
    """Load the checked-in YAML config for M21 continual learning."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_payload = dict(payload["artifacts"])
    config = ContinualLearningConfig(
        enabled=bool(payload.get("enabled", True)),
        candidate_types=tuple(str(value) for value in payload["candidate_types"]),
        live_eligible_candidate_types=tuple(
            str(value) for value in payload["live_eligible_candidate_types"]
        ),
        shadow_only_candidate_types=tuple(
            str(value) for value in payload["shadow_only_candidate_types"]
        ),
        artifacts=ArtifactConfig(
            root_dir=str(artifact_payload["root_dir"]),
            summary_path=str(artifact_payload["summary_path"]),
            drift_caps_summary_path=str(artifact_payload["drift_caps_summary_path"]),
            current_profile_path=str(artifact_payload["current_profile_path"]),
            promotions_history_path=str(artifact_payload["promotions_history_path"]),
            events_history_path=str(artifact_payload["events_history_path"]),
            reports_dir=str(artifact_payload["reports_dir"]),
            experiments_dir=str(artifact_payload["experiments_dir"]),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: ContinualLearningConfig) -> None:
    approved_types = {
        "CALIBRATION_OVERLAY",
        "INCREMENTAL_SHADOW_CHALLENGER",
    }
    configured_types = set(config.candidate_types)
    if configured_types != approved_types:
        raise ValueError(
            f"candidate_types must match the approved M21 set: {sorted(approved_types)}"
        )
    if set(config.live_eligible_candidate_types) != {"CALIBRATION_OVERLAY"}:
        raise ValueError(
            "live_eligible_candidate_types must only contain CALIBRATION_OVERLAY"
        )
    if set(config.shadow_only_candidate_types) != {"INCREMENTAL_SHADOW_CHALLENGER"}:
        raise ValueError(
            "shadow_only_candidate_types must only contain INCREMENTAL_SHADOW_CHALLENGER"
        )
    if not set(config.live_eligible_candidate_types).issubset(configured_types):
        raise ValueError("live_eligible_candidate_types must be a subset of candidate_types")
    if not set(config.shadow_only_candidate_types).issubset(configured_types):
        raise ValueError("shadow_only_candidate_types must be a subset of candidate_types")
