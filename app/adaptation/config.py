"""Typed configuration loading for the Stream Alpha M19 adaptation layer."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class DriftThresholdConfig:
    """Configured warning and breach thresholds for one drift family."""

    warning: float
    breach: float


@dataclass(frozen=True, slots=True)
class DriftConfig:
    """Concept-drift detector configuration."""

    detector_name: str
    minimum_reference_samples: int
    minimum_live_samples: int
    features: tuple[str, ...]
    prob_up: DriftThresholdConfig
    calibration: DriftThresholdConfig
    feature_default: DriftThresholdConfig
    feature_thresholds: dict[str, DriftThresholdConfig] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RollingWindowConfig:
    """Rolling performance windows tracked for M19."""

    trade_counts: tuple[int, ...]
    day_windows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MinimumSampleConfig:
    """Sample-size gates used across adaptive decisions."""

    threshold_tuning: int
    sizing: int
    calibration: int
    promotion: int
    challenger_shadow: int


@dataclass(frozen=True, slots=True)
class ThresholdBoundsConfig:
    """Bounded additive threshold tuning configuration."""

    max_absolute_delta: float
    min_buy_prob_up: float
    max_buy_prob_up: float
    min_sell_prob_up: float
    max_sell_prob_up: float
    improvement_sensitivity: float


@dataclass(frozen=True, slots=True)
class SizingBoundsConfig:
    """Bounded additive sizing configuration."""

    min_multiplier: float
    max_multiplier: float
    calibration_weight: float
    performance_weight: float
    drawdown_penalty_weight: float


@dataclass(frozen=True, slots=True)
class ChallengerWindowConfig:
    """Recent-window challenger retraining configuration."""

    train_days: int
    validation_days: int
    shadow_days: int
    shadow_only: bool = True


@dataclass(frozen=True, slots=True)
class PromotionThresholdConfig:
    """Evidence-based promotion thresholds for profiles and challengers."""

    min_net_pnl_delta_after_costs: float
    max_drawdown_degradation: float
    min_profit_factor: float
    min_win_rate: float
    max_shadow_divergence_rate: float
    max_blocked_trade_rate: float
    require_reliability_healthy: bool = True


@dataclass(frozen=True, slots=True)
class FreezeRuleConfig:
    """Gates that can freeze adaptive activation."""

    freeze_on_drift_breach: bool
    freeze_on_degraded_reliability: bool
    degraded_health_statuses: tuple[str, ...]
    degraded_freshness_statuses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    """Canonical M19 artifact paths."""

    root_dir: str
    drift_summary_path: str
    performance_summary_path: str
    current_profile_path: str
    promotions_history_path: str
    reports_dir: str
    challengers_dir: str


@dataclass(frozen=True, slots=True)
class AdaptationConfig:
    """Checked-in M19 adaptation configuration."""

    enabled: bool
    rolling_windows: RollingWindowConfig
    minimum_samples: MinimumSampleConfig
    threshold_bounds: ThresholdBoundsConfig
    sizing_bounds: SizingBoundsConfig
    drift: DriftConfig
    challenger_windows: ChallengerWindowConfig
    promotion_thresholds: PromotionThresholdConfig
    freeze_rules: FreezeRuleConfig
    artifacts: ArtifactConfig


def default_adaptation_config_path() -> Path:
    """Return the checked-in M19 adaptation config path."""
    return Path("configs/adaptation.yaml")


def load_adaptation_config(config_path: Path) -> AdaptationConfig:
    """Load the checked-in YAML config for M19 adaptation."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rolling_payload = dict(payload["rolling_windows"])
    minimum_payload = dict(payload["minimum_samples"])
    threshold_payload = dict(payload["threshold_bounds"])
    sizing_payload = dict(payload["sizing_bounds"])
    drift_payload = dict(payload["drift"])
    challenger_payload = dict(payload["challenger_windows"])
    promotion_payload = dict(payload["promotion_thresholds"])
    freeze_payload = dict(payload["freeze_rules"])
    artifact_payload = dict(payload["artifacts"])

    feature_default = DriftThresholdConfig(
        warning=float(drift_payload["feature_default"]["warning"]),
        breach=float(drift_payload["feature_default"]["breach"]),
    )
    feature_thresholds = {
        str(name): DriftThresholdConfig(
            warning=float(value["warning"]),
            breach=float(value["breach"]),
        )
        for name, value in dict(drift_payload.get("feature_thresholds", {})).items()
    }
    config = AdaptationConfig(
        enabled=bool(payload.get("enabled", True)),
        rolling_windows=RollingWindowConfig(
            trade_counts=tuple(int(value) for value in rolling_payload["trade_counts"]),
            day_windows=tuple(int(value) for value in rolling_payload["day_windows"]),
        ),
        minimum_samples=MinimumSampleConfig(
            threshold_tuning=int(minimum_payload["threshold_tuning"]),
            sizing=int(minimum_payload["sizing"]),
            calibration=int(minimum_payload["calibration"]),
            promotion=int(minimum_payload["promotion"]),
            challenger_shadow=int(minimum_payload["challenger_shadow"]),
        ),
        threshold_bounds=ThresholdBoundsConfig(
            max_absolute_delta=float(threshold_payload["max_absolute_delta"]),
            min_buy_prob_up=float(threshold_payload["min_buy_prob_up"]),
            max_buy_prob_up=float(threshold_payload["max_buy_prob_up"]),
            min_sell_prob_up=float(threshold_payload["min_sell_prob_up"]),
            max_sell_prob_up=float(threshold_payload["max_sell_prob_up"]),
            improvement_sensitivity=float(threshold_payload["improvement_sensitivity"]),
        ),
        sizing_bounds=SizingBoundsConfig(
            min_multiplier=float(sizing_payload["min_multiplier"]),
            max_multiplier=float(sizing_payload["max_multiplier"]),
            calibration_weight=float(sizing_payload["calibration_weight"]),
            performance_weight=float(sizing_payload["performance_weight"]),
            drawdown_penalty_weight=float(sizing_payload["drawdown_penalty_weight"]),
        ),
        drift=DriftConfig(
            detector_name=str(drift_payload["detector_name"]),
            minimum_reference_samples=int(drift_payload["minimum_reference_samples"]),
            minimum_live_samples=int(drift_payload["minimum_live_samples"]),
            features=tuple(str(value) for value in drift_payload["features"]),
            prob_up=DriftThresholdConfig(
                warning=float(drift_payload["prob_up"]["warning"]),
                breach=float(drift_payload["prob_up"]["breach"]),
            ),
            calibration=DriftThresholdConfig(
                warning=float(drift_payload["calibration"]["warning"]),
                breach=float(drift_payload["calibration"]["breach"]),
            ),
            feature_default=feature_default,
            feature_thresholds=feature_thresholds,
        ),
        challenger_windows=ChallengerWindowConfig(
            train_days=int(challenger_payload["train_days"]),
            validation_days=int(challenger_payload["validation_days"]),
            shadow_days=int(challenger_payload["shadow_days"]),
            shadow_only=bool(challenger_payload.get("shadow_only", True)),
        ),
        promotion_thresholds=PromotionThresholdConfig(
            min_net_pnl_delta_after_costs=float(
                promotion_payload["min_net_pnl_delta_after_costs"]
            ),
            max_drawdown_degradation=float(promotion_payload["max_drawdown_degradation"]),
            min_profit_factor=float(promotion_payload["min_profit_factor"]),
            min_win_rate=float(promotion_payload["min_win_rate"]),
            max_shadow_divergence_rate=float(
                promotion_payload["max_shadow_divergence_rate"]
            ),
            max_blocked_trade_rate=float(promotion_payload["max_blocked_trade_rate"]),
            require_reliability_healthy=bool(
                promotion_payload.get("require_reliability_healthy", True)
            ),
        ),
        freeze_rules=FreezeRuleConfig(
            freeze_on_drift_breach=bool(freeze_payload["freeze_on_drift_breach"]),
            freeze_on_degraded_reliability=bool(
                freeze_payload["freeze_on_degraded_reliability"]
            ),
            degraded_health_statuses=tuple(
                str(value) for value in freeze_payload["degraded_health_statuses"]
            ),
            degraded_freshness_statuses=tuple(
                str(value) for value in freeze_payload["degraded_freshness_statuses"]
            ),
        ),
        artifacts=ArtifactConfig(
            root_dir=str(artifact_payload["root_dir"]),
            drift_summary_path=str(artifact_payload["drift_summary_path"]),
            performance_summary_path=str(artifact_payload["performance_summary_path"]),
            current_profile_path=str(artifact_payload["current_profile_path"]),
            promotions_history_path=str(artifact_payload["promotions_history_path"]),
            reports_dir=str(artifact_payload["reports_dir"]),
            challengers_dir=str(artifact_payload["challengers_dir"]),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: AdaptationConfig) -> None:
    if not config.rolling_windows.trade_counts:
        raise ValueError("rolling_windows.trade_counts must not be empty")
    if not config.rolling_windows.day_windows:
        raise ValueError("rolling_windows.day_windows must not be empty")
    if config.threshold_bounds.max_absolute_delta < 0.0:
        raise ValueError("threshold_bounds.max_absolute_delta cannot be negative")
    if not 0.0 < config.sizing_bounds.min_multiplier <= config.sizing_bounds.max_multiplier:
        raise ValueError("sizing multiplier bounds are invalid")
    if config.minimum_samples.threshold_tuning <= 0:
        raise ValueError("minimum_samples.threshold_tuning must be positive")
    if config.minimum_samples.sizing <= 0:
        raise ValueError("minimum_samples.sizing must be positive")
    if config.minimum_samples.calibration <= 0:
        raise ValueError("minimum_samples.calibration must be positive")
    if config.minimum_samples.promotion <= 0:
        raise ValueError("minimum_samples.promotion must be positive")
    if config.minimum_samples.challenger_shadow <= 0:
        raise ValueError("minimum_samples.challenger_shadow must be positive")
    if config.drift.minimum_reference_samples <= 0:
        raise ValueError("drift.minimum_reference_samples must be positive")
    if config.drift.minimum_live_samples <= 0:
        raise ValueError("drift.minimum_live_samples must be positive")
    if not config.drift.features:
        raise ValueError("drift.features must not be empty")
