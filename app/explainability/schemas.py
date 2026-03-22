"""Typed explainability payloads shared across M14 M4-side consumers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.adaptation.schemas import AdaptationContextPayload
from app.ensemble.schemas import EnsembleContextPayload


ContributionDirection = Literal["UP", "DOWN", "NEUTRAL"]


class TopFeatureContribution(BaseModel):
    """One deterministic one-at-a-time reference-ablation contribution row."""

    feature_name: str
    feature_value: float
    reference_value: float
    signed_contribution: float
    direction: ContributionDirection


class PredictionExplanation(BaseModel):
    """Model-side explanation summary for one M4 prediction response."""

    method: str
    available: bool
    reason_code: str
    summary_text: str
    reference_vector_path: str | None = None
    reference_vector_source: str | None = None
    explainable_feature_count: int = 0
    top_feature_count: int = 0


class ThresholdSnapshot(BaseModel):
    """Inspectable threshold snapshot attached to an M4 signal response."""

    buy_prob_up: float
    sell_prob_up: float
    allow_new_long_entries: bool
    regime_label: str | None = None
    regime_run_id: str | None = None
    regime_artifact_path: str | None = None
    high_vol_threshold: float | None = None
    trend_abs_threshold: float | None = None


class RegimeObservedMetrics(BaseModel):
    """Observed exact-row metrics used to explain the current regime label."""

    realized_vol_12: float
    momentum_3: float
    macd_line_12_26: float


class RegimeThresholds(BaseModel):
    """Exact thresholds that produced the resolved regime label."""

    high_vol_threshold: float
    trend_abs_threshold: float


class RegimeReason(BaseModel):
    """Explicit regime-reason payload for additive M14 signal responses."""

    regime_label: str
    regime_run_id: str
    reason_code: str
    reason_text: str
    observed_metrics: RegimeObservedMetrics
    thresholds: RegimeThresholds
    trade_allowed: bool


class SignalExplanation(BaseModel):
    """Operator-facing summary of how the current signal response was produced."""

    signal: str
    available: bool
    reason_code: str
    summary_text: str
    trade_allowed: bool
    decision_source: str


class ReferenceVectorArtifact(BaseModel):
    """Persisted deterministic reference-vector payload for one model version."""

    schema_version: str
    model_version: str
    generated_at: str
    source_table: str
    interval_minutes: int
    reference_vector_source: str
    explainable_numeric_features: list[str]
    reference_values: dict[str, float]


class ResolvedReferenceVector(BaseModel):
    """In-memory resolved reference-vector bundle used for contribution scoring."""

    reference_vector_path: str
    reference_vector_source: str
    reference_values: dict[str, float]


class DecisionTracePrediction(BaseModel):
    """Canonical prediction section stored inside an M14 decision trace."""

    model_name: str
    model_version: str
    prob_up: float
    prob_down: float
    confidence: float
    predicted_class: str
    top_features: list[TopFeatureContribution] = Field(default_factory=list)
    prediction_explanation: PredictionExplanation | None = None


class DecisionTraceSignal(BaseModel):
    """Canonical signal section stored inside an M14 decision trace."""

    signal: str
    reason: str
    signal_status: str | None = None
    decision_source: str | None = None
    reason_code: str | None = None
    freshness_status: str | None = None
    health_overall_status: str | None = None
    signal_explanation: SignalExplanation | None = None


class DecisionTracePortfolioContext(BaseModel):
    """Portfolio snapshot attached to one M10 rationale."""

    available_cash: float
    open_position_count: int
    current_equity: float
    total_open_exposure_notional: float
    current_symbol_exposure_notional: float


class DecisionTraceServiceRiskState(BaseModel):
    """Service-level M10 state summary attached to one decision trace."""

    trading_day: str
    realized_pnl_today: float
    equity_high_watermark: float
    current_equity: float
    loss_streak_count: int
    loss_streak_cooldown_until_interval_begin: str | None = None
    kill_switch_enabled: bool


class OrderedRiskAdjustment(BaseModel):
    """One ordered M10 adjustment step for a modified trade."""

    step_index: int
    reason_code: str
    reason_text: str
    before_notional: float
    after_notional: float


class DecisionTraceRisk(BaseModel):
    """Canonical risk-rationale section stored inside an M14 decision trace."""

    outcome: str
    primary_reason_code: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    reason_texts: list[str] = Field(default_factory=list)
    requested_notional: float
    approved_notional: float
    portfolio_context: DecisionTracePortfolioContext
    service_risk_state: DecisionTraceServiceRiskState
    ordered_adjustments: list[OrderedRiskAdjustment] = Field(default_factory=list)


class DecisionTraceBlockedTrade(BaseModel):
    """Blocked-trade summary attached when M10 blocks the signal."""

    blocked_stage: str
    reason_code: str | None = None
    reason_texts: list[str] = Field(default_factory=list)


class DecisionTracePayload(BaseModel):
    """Canonical M14 decision-trace payload persisted in PostgreSQL JSONB."""

    schema_version: str
    service_name: str
    execution_mode: str
    symbol: str
    signal_row_id: str
    signal_interval_begin: str
    signal_as_of_time: str
    model_name: str
    model_version: str
    prediction: DecisionTracePrediction
    signal: DecisionTraceSignal
    threshold_snapshot: ThresholdSnapshot | None = None
    regime_reason: RegimeReason | None = None
    adaptation: AdaptationContextPayload | None = None
    ensemble: EnsembleContextPayload | None = None
    risk: DecisionTraceRisk | None = None
    blocked_trade: DecisionTraceBlockedTrade | None = None
