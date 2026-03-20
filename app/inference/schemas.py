"""Typed API schemas for the Stream Alpha M4 inference service."""

# pylint: disable=duplicate-code

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Health endpoint payload."""

    status: str
    service: str
    model_loaded: bool
    model_name: str | None
    model_artifact_path: str | None
    regime_loaded: bool
    regime_run_id: str | None
    regime_artifact_path: str | None
    database: str
    started_at: datetime


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
    row_id: str
    interval_begin: str
    as_of_time: str
    prob_up: float
    prob_down: float
    predicted_class: str
    confidence: float
    regime_label: str
    regime_run_id: str


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
    regime_label: str
    regime_run_id: str
    trade_allowed: bool


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
    started_at: datetime
    uptime_seconds: float
    model_name: str | None
