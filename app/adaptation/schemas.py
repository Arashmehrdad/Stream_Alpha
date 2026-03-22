"""Typed schemas shared by the Stream Alpha M19 adaptation package."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


DriftStatus = Literal["HEALTHY", "WATCH", "BREACHED"]
ProfileStatus = Literal[
    "DRAFT",
    "APPROVED",
    "ACTIVE",
    "SUPERSEDED",
    "ROLLED_BACK",
    "REJECTED",
]
ChallengerStatus = Literal[
    "TRAINED",
    "EVALUATED",
    "PROMOTED",
    "REJECTED",
    "ROLLED_BACK",
]
PromotionTargetType = Literal["PROFILE", "CHALLENGER_MODEL"]
PromotionDecisionType = Literal["PROMOTE", "REJECT", "ROLLBACK", "HOLD"]


class ThresholdPolicy(BaseModel):
    """Bounded threshold policy stored inside adaptive profiles."""

    buy_threshold_delta: float = 0.0
    sell_threshold_delta: float = 0.0
    minimum_trade_sample: int = 0
    freeze_on_drift_breach: bool = True
    freeze_on_degraded_reliability: bool = True


class SizingPolicy(BaseModel):
    """Bounded sizing policy stored inside adaptive profiles."""

    size_multiplier: float = 1.0
    minimum_trade_sample: int = 0
    bounded_by_m10: bool = True


class CalibrationProfile(BaseModel):
    """Local calibration profile stored inside adaptive profiles."""

    method: str = "identity"
    x_points: list[float] = Field(default_factory=list)
    y_points: list[float] = Field(default_factory=list)
    trained_sample_count: int = 0
    source_window: str | None = None


class AdaptiveDriftRecord(BaseModel):
    """Persisted adaptive drift state row."""

    symbol: str
    regime_label: str
    detector_name: str
    window_id: str
    reference_window_start: datetime
    reference_window_end: datetime
    live_window_start: datetime
    live_window_end: datetime
    drift_score: float
    warning_threshold: float
    breach_threshold: float
    status: DriftStatus
    reason_code: str
    detail: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AdaptivePerformanceWindow(BaseModel):
    """Persisted adaptive performance summary row."""

    execution_mode: str
    symbol: str
    regime_label: str
    window_id: str
    window_type: str
    window_start: datetime
    window_end: datetime
    trade_count: int
    net_pnl_after_costs: float
    max_drawdown: float
    profit_factor: float
    expectancy: float
    win_rate: float
    precision: float
    avg_slippage_bps: float
    blocked_trade_rate: float
    shadow_divergence_rate: float
    health_context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class AdaptiveProfileRecord(BaseModel):
    """Persisted adaptive profile row."""

    profile_id: str
    status: ProfileStatus
    execution_mode_scope: str
    symbol_scope: str
    regime_scope: str
    threshold_policy_json: ThresholdPolicy = Field(default_factory=ThresholdPolicy)
    sizing_policy_json: SizingPolicy = Field(default_factory=SizingPolicy)
    calibration_profile_json: CalibrationProfile = Field(default_factory=CalibrationProfile)
    source_evidence_json: dict[str, Any] = Field(default_factory=dict)
    rollback_target_profile_id: str | None = None
    created_at: datetime | None = None
    approved_at: datetime | None = None
    activated_at: datetime | None = None
    superseded_at: datetime | None = None


class AdaptiveChallengerRunRecord(BaseModel):
    """Persisted adaptive challenger run row."""

    challenger_run_id: str
    status: ChallengerStatus
    train_window_start: datetime
    train_window_end: datetime
    validation_window_start: datetime
    validation_window_end: datetime
    shadow_window_start: datetime
    shadow_window_end: datetime
    candidate_model_version: str
    config_json: dict[str, Any] = Field(default_factory=dict)
    metrics_json: dict[str, Any] = Field(default_factory=dict)
    shadow_summary_json: dict[str, Any] = Field(default_factory=dict)
    artifact_paths_json: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AdaptivePromotionDecisionRecord(BaseModel):
    """Persisted adaptive promotion decision row."""

    decision_id: str
    target_type: PromotionTargetType
    target_id: str
    incumbent_id: str | None = None
    decision: PromotionDecisionType
    metrics_delta_json: dict[str, Any] = Field(default_factory=dict)
    safety_checks_json: dict[str, Any] = Field(default_factory=dict)
    research_integrity_json: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    summary_text: str
    decided_at: datetime


class AdaptiveRecentPerformanceSummary(BaseModel):
    """Compact recent performance summary used in traces and APIs."""

    window_id: str
    window_type: str
    trade_count: int
    net_pnl_after_costs: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    blocked_trade_rate: float
    shadow_divergence_rate: float


class EffectiveThresholds(BaseModel):
    """Effective thresholds used by M4 after bounded adaptation."""

    buy_prob_up: float
    sell_prob_up: float


class AdaptationContextPayload(BaseModel):
    """Canonical adaptation context attached to decisions and APIs."""

    adaptation_profile_id: str | None = None
    threshold_policy_id: str | None = None
    sizing_policy_id: str | None = None
    calibration_profile_id: str | None = None
    drift_status: str | None = None
    recent_performance_summary: AdaptiveRecentPerformanceSummary | None = None
    adaptation_reason_codes: list[str] = Field(default_factory=list)
    frozen_by_health_gate: bool = False
    calibrated_confidence: float | None = None
    adaptive_size_multiplier: float | None = None
    effective_thresholds: EffectiveThresholds | None = None


class AdaptationSummaryResponse(BaseModel):
    """Read-only adaptation summary endpoint payload."""

    enabled: bool
    active_profile_count: int
    active_profile_id: str | None = None
    adaptation_status: str
    frozen_by_health_gate: bool = False
    latest_drift_status: str | None = None
    latest_promotion_decision: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class AdaptationDriftResponse(BaseModel):
    """Read-only drift collection payload."""

    items: list[AdaptiveDriftRecord] = Field(default_factory=list)


class AdaptationPerformanceResponse(BaseModel):
    """Read-only rolling performance collection payload."""

    items: list[AdaptivePerformanceWindow] = Field(default_factory=list)


class AdaptationProfilesResponse(BaseModel):
    """Read-only adaptive profiles payload."""

    items: list[AdaptiveProfileRecord] = Field(default_factory=list)


class AdaptationPromotionsResponse(BaseModel):
    """Read-only adaptive promotion decisions payload."""

    items: list[AdaptivePromotionDecisionRecord] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AppliedAdaptation:
    """Resolved bounded adaptation applied to one prediction or signal."""

    profile_id: str | None = None
    calibrated_confidence: float | None = None
    effective_thresholds: EffectiveThresholds | None = None
    adaptive_size_multiplier: float = 1.0
    drift_status: str | None = None
    recent_performance_summary: AdaptiveRecentPerformanceSummary | None = None
    adaptation_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    frozen_by_health_gate: bool = False
