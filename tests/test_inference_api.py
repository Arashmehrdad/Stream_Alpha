"""API tests for the Stream Alpha M4 inference service."""

# pylint: disable=duplicate-code,too-few-public-methods

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi.testclient import TestClient

from app.common.config import (
    FeatureSettings,
    InferenceSettings,
    KafkaSettings,
    KrakenSettings,
    PostgresSettings,
    RetrySettings,
    Settings,
    TableSettings,
    TopicSettings,
)
from app.inference.db import DatabaseUnavailableError
from app.inference.main import create_app
from app.inference.service import InferenceService, load_model_artifact


class SerializableProbabilityModel:
    """Serializable classifier stub for API tests."""

    def __init__(self, prob_up: float) -> None:
        self._prob_up = prob_up

    def predict_proba(self, rows: list[dict]) -> list[list[float]]:
        """Return a fixed probability for each requested row."""
        return [[1.0 - self._prob_up, self._prob_up] for _ in rows]


class FakeDatabase:
    """Minimal async database stub for service and API tests."""

    def __init__(
        self,
        *,
        row: dict | None = None,
        healthy: bool = True,
        fetch_error: Exception | None = None,
    ) -> None:
        self.row = row
        self.healthy = healthy
        self.fetch_error = fetch_error

    async def connect(self) -> None:
        """Open the fake connection."""
        return None

    async def close(self) -> None:
        """Close the fake connection."""
        return None

    async def is_healthy(self) -> bool:
        """Return the configured fake health state."""
        return self.healthy

    async def fetch_latest_feature_row(self, *, symbol: str, interval_minutes: int) -> dict | None:
        """Return the configured row or raise the configured error."""
        del symbol, interval_minutes
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.row


def _build_settings(model_path: str) -> Settings:
    return Settings(
        app_name="streamalpha",
        log_level="INFO",
        service_name="producer",
        heartbeat_interval_seconds=15,
        kraken=KrakenSettings(
            ws_url="wss://ws.kraken.com/v2",
            rest_ohlc_url="https://api.kraken.com/0/public/OHLC",
            symbols=("BTC/USD", "ETH/USD", "SOL/USD"),
            ohlc_interval_minutes=5,
        ),
        kafka=KafkaSettings(
            bootstrap_servers="redpanda:9092",
            client_id="streamalpha-producer",
        ),
        postgres=PostgresSettings(
            host="127.0.0.1",
            port=5432,
            database="streamalpha",
            user="streamalpha",
            password="change-me-local-only",
        ),
        topics=TopicSettings(
            raw_trades="raw.trades",
            raw_ohlc="raw.ohlc",
            raw_health="raw.health",
        ),
        tables=TableSettings(
            raw_trades="raw_trades",
            raw_ohlc="raw_ohlc",
            feature_ohlc="feature_ohlc",
            producer_heartbeat="producer_heartbeat",
        ),
        retry=RetrySettings(
            initial_delay_seconds=1.0,
            max_delay_seconds=30.0,
            multiplier=2.0,
            jitter_seconds=0.5,
        ),
        features=FeatureSettings(
            consumer_group_id="streamalpha-feature-consumer",
            service_name="features",
            finalization_grace_seconds=30,
            bootstrap_candles=64,
        ),
        inference=InferenceSettings(
            model_path=model_path,
            service_name="inference",
            signal_buy_prob_up=0.55,
            signal_sell_prob_up=0.45,
        ),
    )


def _write_artifact(tmp_path: Path, *, prob_up: float) -> Path:
    artifact_path = tmp_path / f"model-{prob_up:.2f}.joblib"
    joblib.dump(
        {
            "model_name": "logistic_regression",
            "trained_at": "2026-03-19T22:30:02Z",
            "feature_columns": ["symbol", "close_price"],
            "expanded_feature_names": ["symbol=BTC/USD", "close_price"],
            "model": SerializableProbabilityModel(prob_up),
        },
        artifact_path,
    )
    return artifact_path


def _feature_row(symbol: str = "BTC/USD") -> dict:
    base_time = datetime(2026, 3, 19, 22, 0, tzinfo=timezone.utc)
    return {
        "id": 1,
        "source_exchange": "kraken",
        "symbol": symbol,
        "interval_minutes": 5,
        "interval_begin": base_time,
        "interval_end": datetime(2026, 3, 19, 22, 5, tzinfo=timezone.utc),
        "as_of_time": datetime(2026, 3, 19, 22, 5, tzinfo=timezone.utc),
        "computed_at": datetime(2026, 3, 19, 22, 6, tzinfo=timezone.utc),
        "raw_event_id": "evt-1",
        "open_price": 70000.0,
        "high_price": 70100.0,
        "low_price": 69900.0,
        "close_price": 70050.0,
        "vwap": 70020.0,
        "trade_count": 120,
        "volume": 12.5,
        "log_return_1": 0.01,
        "log_return_3": 0.02,
        "momentum_3": 0.03,
        "return_mean_12": 0.01,
        "return_std_12": 0.02,
        "realized_vol_12": 0.03,
        "rsi_14": 55.0,
        "macd_line_12_26": 1.2,
        "volume_mean_12": 10.0,
        "volume_std_12": 2.0,
        "volume_zscore_12": 1.25,
        "close_zscore_12": 0.75,
        "lag_log_return_1": 0.005,
        "lag_log_return_2": 0.004,
        "lag_log_return_3": 0.003,
        "created_at": datetime(2026, 3, 19, 22, 6, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 3, 19, 22, 6, tzinfo=timezone.utc),
    }


def _build_client(tmp_path: Path, *, prob_up: float, database: FakeDatabase) -> TestClient:
    artifact = load_model_artifact(str(_write_artifact(tmp_path, prob_up=prob_up)))
    service = InferenceService(
        _build_settings(str(_write_artifact(tmp_path, prob_up=prob_up))),
        database=database,
        model_artifact=artifact,
    )
    return TestClient(create_app(service))


def test_health_reports_success_and_dependency_failure(tmp_path: Path) -> None:
    """`/health` should reflect DB reachability while keeping the model loaded."""
    healthy_client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))
    healthy_response = healthy_client.get("/health")

    assert healthy_response.status_code == 200
    assert healthy_response.json()["status"] == "ok"

    unhealthy_client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row(), healthy=False),
    )
    unhealthy_response = unhealthy_client.get("/health")

    assert unhealthy_response.status_code == 503
    assert unhealthy_response.json()["database"] == "unavailable"


def test_predict_happy_path(tmp_path: Path) -> None:
    """`/predict` should return probabilities and class labels from the saved artifact."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))

    response = client.get("/predict", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "UP"
    assert payload["prob_up"] == 0.7
    assert payload["row_id"].startswith("BTC/USD|")


def test_signal_buy_sell_and_hold(tmp_path: Path) -> None:
    """`/signal` should map probabilities onto BUY, SELL, and HOLD thresholds."""
    buy_client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))
    sell_client = _build_client(tmp_path, prob_up=0.3, database=FakeDatabase(row=_feature_row()))
    hold_client = _build_client(tmp_path, prob_up=0.5, database=FakeDatabase(row=_feature_row()))

    assert buy_client.get("/signal", params={"symbol": "BTC/USD"}).json()["signal"] == "BUY"
    assert sell_client.get("/signal", params={"symbol": "BTC/USD"}).json()["signal"] == "SELL"
    assert hold_client.get("/signal", params={"symbol": "BTC/USD"}).json()["signal"] == "HOLD"


def test_invalid_symbol_and_missing_row_behaviour(tmp_path: Path) -> None:
    """Invalid symbols should 400 and missing rows should 404."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=None))

    invalid_response = client.get("/predict", params={"symbol": "DOGE/USD"})
    missing_response = client.get("/predict", params={"symbol": "BTC/USD"})

    assert invalid_response.status_code == 400
    assert missing_response.status_code == 404


def test_metrics_increment_after_requests(tmp_path: Path) -> None:
    """The in-memory metrics endpoint should count completed requests since startup."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))

    client.get("/health")
    client.get("/predict", params={"symbol": "BTC/USD"})
    metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200
    payload = metrics_response.json()
    assert payload["requests_total"] == 2
    assert payload["endpoint_counts"]["/health"] == 1
    assert payload["endpoint_counts"]["/predict"] == 1


def test_predict_returns_503_when_database_is_unavailable(tmp_path: Path) -> None:
    """DB failures should propagate as 503s from the inference endpoints."""
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(fetch_error=DatabaseUnavailableError("db down")),
    )

    response = client.get("/predict", params={"symbol": "BTC/USD"})

    assert response.status_code == 503
