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
EnsembleModelFamily = Literal[
    "AUTOGLUON",
    "NEURALFORECAST_NHITS",
    "NEURALFORECAST_NBEATSX",
    "NEURALFORECAST_TFT",
    "NEURALFORECAST_PATCHTST",
    "AMAZON_CHRONOS_2",
    "GOOGLE_TIMESFM_2_0_500M_PYTORCH",
    "MOIRAI_SMALL",
    "MOIRAI_BASE",
    "RIVER",
    "REGISTRY_CHAMPION_BASELINE",
]
EnsembleResearchSlice = Literal["ALL", "TREND_COMBINED", "RANGE", "HIGH_VOL"]


class EnsembleCandidateRosterEntry(BaseModel):
    """Stable candidate roster contract persisted inside candidate_roster_json."""

    candidate_id: str
    candidate_role: str
    model_version: str
    scope_regimes: list[str] = Field(default_factory=list)
    enabled: bool = True
    expected_model_name: str | None = None
    notes: str | None = None


class EnsembleProfileRecord(BaseModel):
    """Persisted ensemble profile row."""

    profile_id: str
    status: EnsembleProfileStatus
    approval_stage: str
    execution_mode_scope: str = "ALL"
    symbol_scope: str = "ALL"
    regime_scope: str = "ALL"
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
    roster_status: str | None = None
    roster_reason_codes: list[str] = Field(default_factory=list)
    participating_candidates: list[ParticipatingCandidate] = Field(default_factory=list)


class EnsembleResearchCandidate(BaseModel):
    """One registry-backed candidate considered during Packet 2 research."""

    model_version: str
    model_name: str
    model_family: EnsembleModelFamily
    candidate_role: str
    artifact_path: str
    trained_at: str
    scope_regimes: list[str] = Field(default_factory=list)
    entry_metadata: dict[str, Any] = Field(default_factory=dict)


class EnsembleEvaluationSliceMetrics(BaseModel):
    """One evaluation slice used for honest Packet 2 candidate comparison."""

    slice_label: EnsembleResearchSlice
    net_pnl_after_fees_slippage: float
    max_drawdown: float
    calmar_ratio: float | None = None
    profit_factor: float | None = None
    signal_precision: float = 0.0
    trade_count: int = 0
    blocked_trade_rate: float = 0.0
    shadow_divergence: float | None = None


class EnsembleResearchResult(BaseModel):
    """Evaluated candidate plus its regime-conditioned Packet 2 evidence."""

    candidate: EnsembleResearchCandidate
    metrics_by_slice: dict[EnsembleResearchSlice, EnsembleEvaluationSliceMetrics]
    primary_slice: EnsembleResearchSlice
    primary_metric_value: float


class EnsembleRosterSelection(BaseModel):
    """Selected canonical 3-role runtime roster for Packet 2."""

    generalist: EnsembleResearchResult
    trend_specialist: EnsembleResearchResult
    range_specialist: EnsembleResearchResult
    evidence_summary_json: dict[str, Any] = Field(default_factory=dict)


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
    roster_status: str | None = None
    roster_reason_codes: tuple[str, ...] = field(default_factory=tuple)
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
            roster_status=self.roster_status,
            roster_reason_codes=list(self.roster_reason_codes),
            participating_candidates=[
                ParticipatingCandidate(**c) if isinstance(c, dict) else c
                for c in self.participating_candidates
            ],
        )
