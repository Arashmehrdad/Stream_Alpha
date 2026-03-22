"""Typed schemas shared by the Stream Alpha M20 ensemble package."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


EnsembleProfileStatus = Literal[
    "DRAFT",
    "APPROVED",
    "ACTIVE",
    "SUPERSEDED",
    "ROLLED_BACK",
    "REJECTED",
]
EnsembleApprovalStage = Literal["PENDING", "APPROVED", "ACTIVATED", "SUPERSEDED"]
CandidateParticipationStatus = Literal["ELIGIBLE", "EXCLUDED_SCOPE", "SCORE_FAILED"]


class EnsembleProfileRecord(BaseModel):
    """Persisted ensemble profile row."""

    profile_id: str
    status: EnsembleProfileStatus
    approval_stage: str
    candidate_roster_json: list[dict[str, Any]] = Field(default_factory=list)
    regime_weight_matrix_json: dict[str, dict[str, float]] = Field(default_factory=dict)
    agreement_policy_json: dict[str, Any] = Field(default_factory=dict)
    evidence_summary_json: dict[str, Any] = Field(default_factory=dict)
    rollback_target_profile_id: str | None = None
    created_at: datetime | None = None
    approved_at: datetime | None = None
    activated_at: datetime | None = None
    superseded_at: datetime | None = None


class EnsembleChallengerRunRecord(BaseModel):
    """Persisted ensemble challenger run row."""

    challenger_run_id: str
    status: str
    config_json: dict[str, Any] = Field(default_factory=dict)
    metrics_json: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class EnsemblePromotionDecisionRecord(BaseModel):
    """Persisted ensemble promotion decision row."""

    decision_id: str
    target_type: str
    target_id: str
    incumbent_id: str | None = None
    decision: str
    metrics_delta_json: dict[str, Any] = Field(default_factory=dict)
    safety_checks_json: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    summary_text: str
    decided_at: datetime


class ParticipatingCandidate(BaseModel):
    """One candidate entry inside the ensemble context payload."""

    candidate_id: str
    candidate_role: str
    model_name: str
    model_version: str
    participation_status: str
    scope_regimes: list[str] = Field(default_factory=list)
    applied_weight: float = 0.0
    prob_up: float = 0.0
    prob_down: float = 0.0
    predicted_class: str = "DOWN"


class EnsembleContextPayload(BaseModel):
    """Canonical ensemble context attached to decision traces."""

    ensemble_profile_id: str | None = None
    approval_stage: str | None = None
    resolved_regime_label: str | None = None
    resolved_regime_run_id: str | None = None
    weighting_reason_codes: list[str] = Field(default_factory=list)
    raw_ensemble_confidence: float | None = None
    effective_confidence: float | None = None
    agreement_band: str | None = None
    vote_agreement_ratio: float | None = None
    probability_spread: float | None = None
    agreement_multiplier: float | None = None
    candidate_count: int = 0
    participating_candidates: list[ParticipatingCandidate] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """Resolved ensemble scoring result for one prediction cycle."""

    active: bool = False
    ensemble_profile_id: str | None = None
    approval_stage: str | None = None
    ensemble_prob_up: float | None = None
    ensemble_prob_down: float | None = None
    ensemble_predicted_class: str | None = None
    raw_ensemble_confidence: float | None = None
    effective_confidence: float | None = None
    agreement_band: str | None = None
    vote_agreement_ratio: float | None = None
    probability_spread: float | None = None
    agreement_multiplier: float | None = None
    candidate_count: int = 0
    participating_candidates: tuple = field(default_factory=tuple)
    weighting_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    fallback_reason: str | None = None

    def to_context_payload(
        self,
        *,
        regime_label: str | None = None,
        regime_run_id: str | None = None,
    ) -> EnsembleContextPayload | None:
        """Convert to canonical trace payload shape."""
        if not self.active and self.fallback_reason is None:
            return None
        return EnsembleContextPayload(
            ensemble_profile_id=self.ensemble_profile_id,
            approval_stage=self.approval_stage,
            resolved_regime_label=regime_label,
            resolved_regime_run_id=regime_run_id,
            weighting_reason_codes=list(self.weighting_reason_codes),
            raw_ensemble_confidence=self.raw_ensemble_confidence,
            effective_confidence=self.effective_confidence,
            agreement_band=self.agreement_band,
            vote_agreement_ratio=self.vote_agreement_ratio,
            probability_spread=self.probability_spread,
            agreement_multiplier=self.agreement_multiplier,
            candidate_count=self.candidate_count,
            participating_candidates=[
                ParticipatingCandidate(**c) if isinstance(c, dict) else c
                for c in self.participating_candidates
            ],
        )
