"""Typed configuration loading for the Stream Alpha M20 ensemble layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


CandidateRole = Literal["GENERALIST", "TREND_SPECIALIST", "RANGE_SPECIALIST"]
RegimeLabel = Literal["TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"]
AgreementBand = Literal["HIGH", "MEDIUM", "LOW"]

CANDIDATE_ROLES: tuple[CandidateRole, ...] = (
    "GENERALIST",
    "TREND_SPECIALIST",
    "RANGE_SPECIALIST",
)

REGIME_LABELS: tuple[RegimeLabel, ...] = (
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "HIGH_VOL",
)


@dataclass(frozen=True, slots=True)
class AgreementPolicyConfig:
    """Configurable agreement band thresholds and multipliers."""

    high_ratio_min: float
    high_spread_max: float
    medium_ratio_min: float
    medium_spread_max: float
    high_multiplier: float
    medium_multiplier: float
    low_multiplier: float


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """Checked-in M20 ensemble configuration."""

    enabled: bool
    candidate_roles: tuple[CandidateRole, ...]
    regime_weight_matrix: dict[RegimeLabel, dict[CandidateRole, float]]
    agreement_policy: AgreementPolicyConfig
    artifact_root: str


def default_ensemble_config_path() -> Path:
    """Return the checked-in M20 ensemble config path."""
    return Path("configs/ensemble.yaml")


def load_ensemble_config(config_path: Path) -> EnsembleConfig:
    """Load the checked-in YAML config for M20 ensemble."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    roles = tuple(str(r) for r in payload["candidate_roles"])
    matrix_raw = payload["regime_weight_matrix"]
    matrix: dict[RegimeLabel, dict[CandidateRole, float]] = {}
    for regime, weights in matrix_raw.items():
        matrix[str(regime)] = {str(role): float(w) for role, w in weights.items()}

    ap = payload["agreement_policy"]
    agreement = AgreementPolicyConfig(
        high_ratio_min=float(ap["high_ratio_min"]),
        high_spread_max=float(ap["high_spread_max"]),
        medium_ratio_min=float(ap["medium_ratio_min"]),
        medium_spread_max=float(ap["medium_spread_max"]),
        high_multiplier=float(ap["high_multiplier"]),
        medium_multiplier=float(ap["medium_multiplier"]),
        low_multiplier=float(ap["low_multiplier"]),
    )

    config = EnsembleConfig(
        enabled=bool(payload.get("enabled", True)),
        candidate_roles=roles,
        regime_weight_matrix=matrix,
        agreement_policy=agreement,
        artifact_root=str(payload.get("artifact_root", "artifacts/ensemble")),
    )
    _validate_config(config)
    return config


def _validate_config(config: EnsembleConfig) -> None:
    if not config.candidate_roles:
        raise ValueError("candidate_roles must not be empty")
    for regime, weights in config.regime_weight_matrix.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Regime weight matrix for {regime} must sum to ~1.0, got {total}"
            )
    ap = config.agreement_policy
    if not 0.0 <= ap.high_multiplier <= 1.0:
        raise ValueError("high_multiplier must be in [0.0, 1.0]")
    if not 0.0 <= ap.medium_multiplier <= 1.0:
        raise ValueError("medium_multiplier must be in [0.0, 1.0]")
    if not 0.0 <= ap.low_multiplier <= 1.0:
        raise ValueError("low_multiplier must be in [0.0, 1.0]")
