"""Typed API schemas for the Stream Alpha M4 inference service."""

# pylint: disable=duplicate-code

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.explainability.schemas import (
    PredictionExplanation,
    RegimeReason,
    SignalExplanation,
    ThresholdSnapshot,
    TopFeatureContribution,
)


class HealthResponse(BaseModel):
    """Health endpoint payload."""

    status: str
    service: str
    runtime_profile: str
    execution_mode: str | None = None
    startup_validation_passed: bool | None = None
    startup_report_path: str | None = None
    model_loaded: bool
    model_name: str | None
    model_artifact_path: str | None
    regime_loaded: bool
    regime_run_id: str | None
    regime_artifact_path: str | None
    database: str
    started_at: datetime
    health_overall_status: str | None = None
    reason_code: str | None = None
    freshness_status: str | None = None


class FeatureRowResponse(BaseModel):  # pylint: disable=too-many-instance-attributes
    """Latest canonical feature row payload."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    source_exchange: str
    symbol: str
    interval_minutes: int
    interval_begin: datetime
    interval_end: datetime
    as_of_time: datetime
    computed_at: datetime
    raw_event_id: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    vwap: float
    trade_count: int
    volume: float
    log_return_1: float
    log_return_3: float
    momentum_3: float
    return_mean_12: float
    return_std_12: float
    realized_vol_12: float
    rsi_14: float
    macd_line_12_26: float
    volume_mean_12: float
    volume_std_12: float
    volume_zscore_12: float
    close_zscore_12: float
    lag_log_return_1: float
    lag_log_return_2: float
    lag_log_return_3: float
    created_at: datetime
    updated_at: datetime


class PredictionResponse(BaseModel):
    """Prediction payload returned by `/predict`."""

    symbol: str
    model_name: str
    model_trained_at: str
    model_artifact_path: str
    model_version: str
    row_id: str
    interval_begin: str
    as_of_time: str
    prob_up: float
    prob_down: float
    predicted_class: str
    confidence: float
    regime_label: str
    regime_run_id: str
    decision_source: str | None = None
    reason_code: str | None = None
    freshness_status: str | None = None
    health_overall_status: str | None = None
    top_features: list[TopFeatureContribution] = Field(default_factory=list)
    prediction_explanation: PredictionExplanation


class ThresholdsResponse(BaseModel):
    """Signal threshold payload."""

    buy_prob_up: float
    sell_prob_up: float


class SignalResponse(BaseModel):
    """Signal payload returned by `/signal`."""

    symbol: str
    signal: str
    reason: str
    prob_up: float
    prob_down: float
    confidence: float
    predicted_class: str
    thresholds: ThresholdsResponse
    row_id: str
    as_of_time: str
    model_name: str
    model_version: str
    regime_label: str | None
    regime_run_id: str | None
    trade_allowed: bool
    signal_status: str | None = None
    decision_source: str | None = None
    reason_code: str | None = None
    freshness_status: str | None = None
    health_overall_status: str | None = None
    top_features: list[TopFeatureContribution] = Field(default_factory=list)
    prediction_explanation: PredictionExplanation
    threshold_snapshot: ThresholdSnapshot
    regime_reason: RegimeReason | None = None
    signal_explanation: SignalExplanation


class RegimeResponse(BaseModel):
    """Regime payload returned by `/regime`."""

    symbol: str
    row_id: str
    interval_begin: str
    as_of_time: str
    regime_label: str
    regime_run_id: str
    regime_artifact_path: str
    realized_vol_12: float
    momentum_3: float
    macd_line_12_26: float
    high_vol_threshold: float
    trend_abs_threshold: float
    trade_allowed: bool
    buy_prob_up: float
    sell_prob_up: float
    freshness_status: str | None = None
    health_overall_status: str | None = None


class FreshnessResponse(BaseModel):
    """Exact-row freshness payload returned by `/freshness`."""

    symbol: str
    row_id: str | None
    interval_begin: str | None
    as_of_time: str | None
    health_overall_status: str
    freshness_status: str
    reason_code: str
    feature_freshness_status: str
    feature_reason_code: str
    feature_age_seconds: float | None
    regime_freshness_status: str
    regime_reason_code: str
    regime_age_seconds: float | None
    detail: str | None = None


class LatencyStatsResponse(BaseModel):
    """In-memory latency counters since service startup."""

    count: int
    avg: float
    max: float


class MetricsResponse(BaseModel):
    """JSON metrics payload."""

    requests_total: int
    errors_total: int
    endpoint_counts: dict[str, int]
    latency_ms: LatencyStatsResponse
    service: str
    runtime_profile: str
    execution_mode: str | None = None
    startup_validation_passed: bool | None = None
    startup_report_path: str | None = None
    started_at: datetime
    uptime_seconds: float
    model_name: str | None
    health_overall_status: str | None = None
    reason_code: str | None = None
    freshness_summary: dict[str, dict[str, str | float | None]] | None = None


class ReliabilityEventResponse(BaseModel):
    """Operator-facing reliability event payload."""

    service_name: str
    component_name: str
    event_type: str
    event_time: datetime
    reason_code: str
    health_overall_status: str | None = None
    freshness_status: str | None = None
    breaker_state: str | None = None
    detail: str | None = None


class ServiceReliabilityResponse(BaseModel):
    """Per-service heartbeat summary for the canonical reliability endpoint."""

    service_name: str
    component_name: str
    checked_at: datetime
    heartbeat_at: datetime | None
    heartbeat_age_seconds: float | None
    heartbeat_freshness_status: str
    health_overall_status: str
    reason_code: str
    detail: str | None = None
    feed_freshness_status: str | None = None
    feed_reason_code: str | None = None
    feed_age_seconds: float | None = None


class FeatureLagResponse(BaseModel):
    """Per-symbol feature consumer lag payload."""

    service_name: str
    component_name: str
    symbol: str
    evaluated_at: datetime
    latest_raw_event_received_at: datetime | None
    latest_feature_interval_begin: datetime | None
    latest_feature_as_of_time: datetime | None
    time_lag_seconds: float | None
    processing_lag_seconds: float | None
    time_lag_reason_code: str
    processing_lag_reason_code: str
    lag_breach: bool
    health_overall_status: str
    reason_code: str
    detail: str | None = None


class SystemReliabilityResponse(BaseModel):
    """Canonical cross-service reliability summary."""

    service_name: str
    checked_at: datetime
    runtime_profile: str
    execution_mode: str | None = None
    startup_validation_passed: bool | None = None
    startup_report_path: str | None = None
    health_overall_status: str
    reason_codes: list[str]
    lag_breach_active: bool
    services: list[ServiceReliabilityResponse]
    lag_by_symbol: list[FeatureLagResponse]
    latest_recovery_event: ReliabilityEventResponse | None = None
