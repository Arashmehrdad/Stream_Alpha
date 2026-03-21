"""Typed explainability payloads shared across M14 M4-side consumers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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
