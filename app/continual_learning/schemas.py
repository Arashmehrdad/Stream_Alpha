"""Typed schemas shared by the Stream Alpha M21 continual learning package."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


ContinualLearningCandidateType = Literal[
    "CALIBRATION_OVERLAY",
    "INCREMENTAL_SHADOW_CHALLENGER",
]
ContinualLearningProfileStatus = Literal[
    "DRAFT",
    "APPROVED",
    "ACTIVE",
    "SUPERSEDED",
    "ROLLED_BACK",
    "REJECTED",
]
ContinualLearningExperimentStatus = Literal[
    "DRAFT",
    "EVALUATED",
    "APPROVED",
    "REJECTED",
    "ROLLED_BACK",
]
ContinualLearningDecisionType = Literal["PROMOTE", "REJECT", "HOLD", "ROLLBACK"]
ContinualLearningDriftCapStatus = Literal["HEALTHY", "WATCH", "BREACHED"]
ContinualLearningPromotionTargetType = Literal["PROFILE", "EXPERIMENT"]


class CalibrationOverlayProfile(BaseModel):
    """Bounded calibration overlay persisted inside continual-learning profiles."""

    method: str = "identity"
    x_points: list[float] = Field(default_factory=list)
    y_points: list[float] = Field(default_factory=list)
    trained_sample_count: int = 0
    source_window: str | None = None
    notes: str | None = None


class ContinualLearningExperimentRecord(BaseModel):
    """Persisted continual-learning experiment row."""

    experiment_id: str
    candidate_type: ContinualLearningCandidateType
    status: ContinualLearningExperimentStatus
    execution_mode_scope: str = "ALL"
    symbol_scope: str = "ALL"
    regime_scope: str = "ALL"
    base_model_version: str | None = None
    candidate_model_version: str | None = None
    config_json: dict[str, Any] = Field(default_factory=dict)
    metrics_json: dict[str, Any] = Field(default_factory=dict)
    artifact_paths_json: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ContinualLearningProfileRecord(BaseModel):
    """Persisted continual-learning profile row."""

    profile_id: str
    candidate_type: ContinualLearningCandidateType
    status: ContinualLearningProfileStatus
    execution_mode_scope: str = "ALL"
    symbol_scope: str = "ALL"
    regime_scope: str = "ALL"
    calibration_overlay_json: CalibrationOverlayProfile = Field(
        default_factory=CalibrationOverlayProfile
    )
    source_evidence_json: dict[str, Any] = Field(default_factory=dict)
    live_eligible: bool = False
    rollback_target_profile_id: str | None = None
    created_at: datetime | None = None
    approved_at: datetime | None = None
    activated_at: datetime | None = None
    superseded_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_candidate_type_constraints(self) -> "ContinualLearningProfileRecord":
        if (
            self.candidate_type == "INCREMENTAL_SHADOW_CHALLENGER"
            and self.live_eligible
        ):
            raise ValueError(
                "INCREMENTAL_SHADOW_CHALLENGER cannot be marked live_eligible in M21"
            )
        return self


class ContinualLearningDriftCapRecord(BaseModel):
    """Persisted continual-learning drift-cap row."""

    cap_id: str
    execution_mode_scope: str = "ALL"
    symbol_scope: str = "ALL"
    regime_scope: str = "ALL"
    candidate_type: ContinualLearningCandidateType
    status: ContinualLearningDriftCapStatus
    observed_drift_score: float
    warning_threshold: float
    breach_threshold: float
    reason_code: str
    detail: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ContinualLearningPromotionDecisionRecord(BaseModel):
    """Persisted continual-learning promotion decision row."""

    decision_id: str
    target_type: ContinualLearningPromotionTargetType
    target_id: str
    incumbent_id: str | None = None
    candidate_type: ContinualLearningCandidateType
    decision: ContinualLearningDecisionType
    live_eligible_after_decision: bool = False
    metrics_delta_json: dict[str, Any] = Field(default_factory=dict)
    safety_checks_json: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    summary_text: str
    decided_at: datetime

    @model_validator(mode="after")
    def _validate_live_eligibility(self) -> "ContinualLearningPromotionDecisionRecord":
        if (
            self.candidate_type == "INCREMENTAL_SHADOW_CHALLENGER"
            and self.live_eligible_after_decision
        ):
            raise ValueError(
                "INCREMENTAL_SHADOW_CHALLENGER cannot become live-eligible in M21"
            )
        return self


class ContinualLearningEventRecord(BaseModel):
    """Persisted continual-learning audit event row."""

    event_id: str
    event_type: str
    profile_id: str | None = None
    experiment_id: str | None = None
    decision_id: str | None = None
    reason_code: str
    payload_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ContinualLearningSummaryResponse(BaseModel):
    """Read-only continual-learning summary payload."""

    enabled: bool
    active_profile_count: int
    active_profile_id: str | None = None
    continual_learning_status: str
    active_candidate_type: ContinualLearningCandidateType | None = None
    latest_drift_cap_status: ContinualLearningDriftCapStatus | None = None
    latest_promotion_decision: ContinualLearningDecisionType | None = None
    latest_event_type: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class ContinualLearningExperimentsResponse(BaseModel):
    """Read-only continual-learning experiments payload."""

    items: list[ContinualLearningExperimentRecord] = Field(default_factory=list)


class ContinualLearningProfilesResponse(BaseModel):
    """Read-only continual-learning profiles payload."""

    items: list[ContinualLearningProfileRecord] = Field(default_factory=list)


class ContinualLearningDriftCapsResponse(BaseModel):
    """Read-only continual-learning drift-cap payload."""

    items: list[ContinualLearningDriftCapRecord] = Field(default_factory=list)


class ContinualLearningPromotionsResponse(BaseModel):
    """Read-only continual-learning promotions payload."""

    items: list[ContinualLearningPromotionDecisionRecord] = Field(default_factory=list)


class ContinualLearningEventsResponse(BaseModel):
    """Read-only continual-learning events payload."""

    items: list[ContinualLearningEventRecord] = Field(default_factory=list)
