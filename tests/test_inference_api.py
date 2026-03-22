"""API tests for the Stream Alpha M4 inference service."""

# pylint: disable=duplicate-code,missing-function-docstring,too-few-public-methods

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import joblib
from fastapi.testclient import TestClient

from app.common.time import to_rfc3339, utc_now
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
from app.reliability.schemas import (
    FeatureLagSnapshot,
    RecoveryEvent,
    ReliabilityState,
    ServiceHeartbeat,
)
from app.regime.live import load_live_regime_runtime


class SerializableProbabilityModel:
    """Serializable classifier stub for API tests."""

    def __init__(self, prob_up: float) -> None:
        self._prob_up = prob_up

    def predict_proba(self, rows: list[dict]) -> list[list[float]]:
        """Return a fixed probability for each requested row."""
        return [[1.0 - self._prob_up, self._prob_up] for _ in rows]


class SerializableFeatureAwareModel:
    """Serializable model stub whose probabilities move with numeric features."""

    def predict_proba(self, rows: list[dict]) -> list[list[float]]:
        """Return deterministic binary probabilities from a few explainable inputs."""
        payload: list[list[float]] = []
        for row in rows:
            prob_up = (
                0.50
                + (0.50 * float(row["momentum_3"]))
                - (0.20 * float(row["realized_vol_12"]))
                + (0.05 * float(row["volume_zscore_12"]))
            )
            payload.append([1.0 - prob_up, prob_up])
        return payload


class FakeDatabase:
    """Minimal async database stub for service and API tests."""

    def __init__(
        self,
        *,
        row: dict | None = None,
        reference_vector: dict[str, float] | None = None,
        healthy: bool = True,
        fetch_error: Exception | None = None,
    ) -> None:
        self.row = row
        self.reference_vector = {} if reference_vector is None else reference_vector
        self.healthy = healthy
        self.fetch_error = fetch_error
        self.last_interval_begin = None

    async def connect(self) -> None:
        """Open the fake connection."""
        return None

    async def close(self) -> None:
        """Close the fake connection."""
        return None

    async def is_healthy(self) -> bool:
        """Return the configured fake health state."""
        return self.healthy

    async def fetch_latest_feature_row(
        self,
        *,
        symbol: str,
        interval_minutes: int,
        interval_begin=None,
    ) -> dict | None:
        """Return the configured row or raise the configured error."""
        del symbol, interval_minutes
        self.last_interval_begin = interval_begin
        if self.fetch_error is not None:
            raise self.fetch_error
        if (
            interval_begin is not None
            and self.row is not None
            and self.row["interval_begin"] != interval_begin
        ):
            return None
        return self.row

    async def fetch_feature_reference_vector(
        self,
        *,
        feature_names: tuple[str, ...],
        interval_minutes: int,
    ) -> dict[str, float]:
        """Return configured or row-derived reference values for explainability tests."""
        del interval_minutes
        if self.fetch_error is not None:
            raise self.fetch_error
        if self.reference_vector:
            return {
                feature_name: float(self.reference_vector[feature_name])
                for feature_name in feature_names
                if feature_name in self.reference_vector
            }
        if self.row is None:
            return {}
        return {
            feature_name: float(self.row[feature_name])
            for feature_name in feature_names
            if feature_name in self.row
            and isinstance(self.row[feature_name], (int, float))
            and not isinstance(self.row[feature_name], bool)
        }


class FakeReliabilityStore:
    """Minimal reliability-store stub for inference API tests."""

    def __init__(
        self,
        *,
        heartbeats: dict[tuple[str, str], ServiceHeartbeat] | None = None,
        reliability_states: dict[tuple[str, str], ReliabilityState] | None = None,
        lag_states: dict[tuple[str, str], list[FeatureLagSnapshot]] | None = None,
        latest_recovery_event: RecoveryEvent | None = None,
    ) -> None:
        self.heartbeats = {} if heartbeats is None else heartbeats
        self.reliability_states = (
            {} if reliability_states is None else reliability_states
        )
        self.lag_states = {} if lag_states is None else lag_states
        self.latest_recovery_event = latest_recovery_event
        self.saved_system_snapshots = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def save_service_heartbeat(self, heartbeat: ServiceHeartbeat) -> None:
        self.heartbeats[(heartbeat.service_name, heartbeat.component_name)] = heartbeat

    async def load_latest_service_heartbeat(
        self,
        *,
        service_name: str,
        component_name: str,
    ) -> ServiceHeartbeat | None:
        return self.heartbeats.get((service_name, component_name))

    async def load_reliability_state(
        self,
        *,
        service_name: str,
        component_name: str,
    ) -> ReliabilityState | None:
        return self.reliability_states.get((service_name, component_name))

    async def load_feature_lag_states(
        self,
        *,
        service_name: str,
        component_name: str,
    ) -> list[FeatureLagSnapshot]:
        return list(self.lag_states.get((service_name, component_name), []))

    async def load_latest_recovery_event(self) -> RecoveryEvent | None:
        return self.latest_recovery_event

    async def save_system_reliability_state(self, snapshot) -> None:
        self.saved_system_snapshots.append(snapshot)


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


def _write_thresholds_artifact(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "thresholds.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "m8_thresholds_v1",
                "run_id": "20260320T120000Z",
                "source_table": "feature_ohlc",
                "source_exchange": "kraken",
                "interval_minutes": 5,
                "required_inputs": [
                    "realized_vol_12",
                    "momentum_3",
                    "macd_line_12_26",
                ],
                "regime_labels": [
                    "TREND_UP",
                    "TREND_DOWN",
                    "RANGE",
                    "HIGH_VOL",
                ],
                "thresholds_by_symbol": {
                    "BTC/USD": {
                        "symbol": "BTC/USD",
                        "fitted_row_count": 100,
                        "high_vol_threshold": 0.05,
                        "trend_abs_threshold": 0.02,
                    },
                    "ETH/USD": {
                        "symbol": "ETH/USD",
                        "fitted_row_count": 100,
                        "high_vol_threshold": 0.06,
                        "trend_abs_threshold": 0.03,
                    },
                    "SOL/USD": {
                        "symbol": "SOL/USD",
                        "fitted_row_count": 100,
                        "high_vol_threshold": 0.07,
                        "trend_abs_threshold": 0.04,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def _write_signal_policy(tmp_path: Path) -> Path:
    policy_path = tmp_path / "regime_signal_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "m9_regime_signal_policy_v1",
                "policies": {
                    "TREND_UP": {
                        "buy_prob_up": 0.54,
                        "sell_prob_up": 0.44,
                        "allow_new_long_entries": True,
                    },
                    "RANGE": {
                        "buy_prob_up": 0.58,
                        "sell_prob_up": 0.42,
                        "allow_new_long_entries": True,
                    },
                    "TREND_DOWN": {
                        "buy_prob_up": 0.60,
                        "sell_prob_up": 0.46,
                        "allow_new_long_entries": False,
                    },
                    "HIGH_VOL": {
                        "buy_prob_up": 0.62,
                        "sell_prob_up": 0.48,
                        "allow_new_long_entries": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return policy_path


def _build_regime_runtime(tmp_path: Path):
    return load_live_regime_runtime(
        thresholds_path=str(_write_thresholds_artifact(tmp_path)),
        signal_policy_path=str(_write_signal_policy(tmp_path)),
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


def _write_feature_aware_artifact(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "artifacts" / "training" / "m3" / "20260321T120000Z" / "model.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=False)
    joblib.dump(
        {
            "model_name": "logistic_regression",
            "trained_at": "2026-03-21T12:00:00Z",
            "feature_columns": [
                "symbol",
                "momentum_3",
                "realized_vol_12",
                "volume_zscore_12",
            ],
            "expanded_feature_names": [
                "symbol=BTC/USD",
                "momentum_3",
                "realized_vol_12",
                "volume_zscore_12",
            ],
            "model": SerializableFeatureAwareModel(),
        },
        artifact_path,
    )
    return artifact_path


def _feature_row(
    symbol: str = "BTC/USD",
    *,
    realized_vol_12: float = 0.03,
    momentum_3: float = 0.03,
    macd_line_12_26: float = 1.2,
    base_time: datetime | None = None,
) -> dict:
    if base_time is None:
        base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
            minutes=5
        )
    interval_end = base_time + timedelta(minutes=5)
    as_of_time = interval_end
    computed_at = interval_end + timedelta(minutes=1)
    return {
        "id": 1,
        "source_exchange": "kraken",
        "symbol": symbol,
        "interval_minutes": 5,
        "interval_begin": base_time,
        "interval_end": interval_end,
        "as_of_time": as_of_time,
        "computed_at": computed_at,
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
        "momentum_3": momentum_3,
        "return_mean_12": 0.01,
        "return_std_12": 0.02,
        "realized_vol_12": realized_vol_12,
        "rsi_14": 55.0,
        "macd_line_12_26": macd_line_12_26,
        "volume_mean_12": 10.0,
        "volume_std_12": 2.0,
        "volume_zscore_12": 1.25,
        "close_zscore_12": 0.75,
        "lag_log_return_1": 0.005,
        "lag_log_return_2": 0.004,
        "lag_log_return_3": 0.003,
        "created_at": computed_at,
        "updated_at": computed_at,
    }


def _build_client(
    tmp_path: Path,
    *,
    prob_up: float = 0.7,
    database: FakeDatabase,
    reliability_store: FakeReliabilityStore | None = None,
    artifact_path: Path | None = None,
) -> TestClient:
    resolved_artifact_path = (
        _write_artifact(tmp_path, prob_up=prob_up)
        if artifact_path is None
        else artifact_path
    )
    artifact = load_model_artifact(str(resolved_artifact_path))
    service = InferenceService(
        _build_settings(str(resolved_artifact_path)),
        database=database,
        model_artifact=artifact,
        regime_runtime=_build_regime_runtime(tmp_path),
        reliability_store=reliability_store or FakeReliabilityStore(),
    )
    return TestClient(create_app(service))


def test_health_reports_success_and_dependency_failure(tmp_path: Path) -> None:
    """`/health` should reflect DB reachability while keeping the model loaded."""
    healthy_client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))
    healthy_response = healthy_client.get("/health")

    assert healthy_response.status_code == 200
    assert healthy_response.json()["status"] == "ok"
    assert healthy_response.json()["regime_loaded"] is True
    assert healthy_response.json()["regime_run_id"] == "20260320T120000Z"
    assert healthy_response.json()["health_overall_status"] == "HEALTHY"
    assert healthy_response.json()["freshness_status"] == "FRESH"

    unhealthy_client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row(), healthy=False),
    )
    unhealthy_response = unhealthy_client.get("/health")

    assert unhealthy_response.status_code == 503
    assert unhealthy_response.json()["database"] == "unavailable"
    assert unhealthy_response.json()["health_overall_status"] == "UNAVAILABLE"


def test_additive_runtime_metadata_is_exposed_on_health_metrics_and_reliability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    startup_report_path = tmp_path / "startup_report.json"
    startup_report_path.write_text(
        json.dumps(
            {
                "schema_version": "m16_startup_report_v1",
                "checked_at": "2026-03-22T10:00:00Z",
                "runtime_profile": "paper",
                "startup_validation_passed": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("STREAMALPHA_RUNTIME_PROFILE", "paper")
    monkeypatch.setenv("STREAMALPHA_STARTUP_REPORT_PATH", str(startup_report_path))

    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))

    health_payload = client.get("/health").json()
    metrics_payload = client.get("/metrics").json()
    reliability_payload = client.get("/reliability/system").json()

    assert health_payload["runtime_profile"] == "paper"
    assert health_payload["execution_mode"] == "paper"
    assert health_payload["startup_validation_passed"] is True
    assert health_payload["startup_report_path"] == str(startup_report_path.resolve())
    assert metrics_payload["runtime_profile"] == "paper"
    assert metrics_payload["execution_mode"] == "paper"
    assert metrics_payload["startup_validation_passed"] is True
    assert reliability_payload["runtime_profile"] == "paper"
    assert reliability_payload["execution_mode"] == "paper"
    assert reliability_payload["startup_validation_passed"] is True


def test_reliability_system_endpoint_returns_canonical_cross_service_summary(
    tmp_path: Path,
) -> None:
    """`/reliability/system` should aggregate heartbeats, lag, and breaker state."""
    now = utc_now().replace(microsecond=0)
    reliability_store = FakeReliabilityStore(
        heartbeats={
            (
                "producer",
                "producer",
                ): ServiceHeartbeat(
                    service_name="producer",
                    component_name="producer",
                    heartbeat_at=now,
                    health_overall_status="HEALTHY",
                    reason_code="SERVICE_HEARTBEAT_HEALTHY",
                    detail=(
                        '{"last_exchange_activity_at":"'
                        f'{to_rfc3339(now)}'
                        '"}'
                    ),
                ),
            (
                "features",
                "features",
            ): ServiceHeartbeat(
                service_name="features",
                component_name="features",
                heartbeat_at=now,
                health_overall_status="HEALTHY",
                reason_code="SERVICE_HEARTBEAT_HEALTHY",
                detail='{"lag_breach_active":false}',
            ),
            (
                "paper-trader",
                "trading_runner",
            ): ServiceHeartbeat(
                service_name="paper-trader",
                component_name="trading_runner",
                heartbeat_at=now,
                health_overall_status="HEALTHY",
                reason_code="SERVICE_HEARTBEAT_HEALTHY",
                detail="runner healthy",
            ),
        },
        reliability_states={
            (
                "paper-trader",
                "signal_client",
            ): ReliabilityState(
                service_name="paper-trader",
                component_name="signal_client",
                health_overall_status="DEGRADED",
                breaker_state="HALF_OPEN",
                failure_count=1,
                success_count=0,
                freshness_status="STALE",
                last_heartbeat_at=now,
                reason_code="SIGNAL_FETCH_FAILED",
                detail="Signal fetch failed once",
                updated_at=now,
            )
        },
        lag_states={
            (
                "features",
                "features",
            ): [
                FeatureLagSnapshot(
                    service_name="features",
                    component_name="features",
                    symbol="BTC/USD",
                    evaluated_at=now,
                    latest_raw_event_received_at=now,
                    latest_feature_interval_begin=now - timedelta(minutes=10),
                    latest_feature_as_of_time=now - timedelta(minutes=10),
                    time_lag_seconds=600.0,
                    processing_lag_seconds=600.0,
                    time_lag_reason_code="FEATURE_TIME_LAG_BREACH",
                    processing_lag_reason_code="FEATURE_PROCESSING_LAG_BREACH",
                    lag_breach=True,
                    health_overall_status="DEGRADED",
                    reason_code="FEATURE_LAG_BREACH",
                    detail="feature lag breach",
                )
            ]
        },
        latest_recovery_event=RecoveryEvent(
            service_name="features",
            component_name="BTC/USD",
            event_type="FEATURE_LAG_TRANSITION",
            event_time=now,
            reason_code="FEATURE_LAG_BREACH_DETECTED",
            health_overall_status="DEGRADED",
            freshness_status="STALE",
            detail="feature lag breach",
        ),
    )
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row(base_time=now - timedelta(minutes=5))),
        reliability_store=reliability_store,
    )

    response = client.get("/reliability/system")

    assert response.status_code == 200
    payload = response.json()
    assert payload["health_overall_status"] == "DEGRADED"
    assert payload["lag_breach_active"] is True
    assert payload["reason_codes"] == ["SIGNAL_FETCH_FAILED", "FEATURE_LAG_BREACH"]
    assert payload["services"][0]["component_name"] == "producer"
    assert payload["services"][0]["feed_freshness_status"] == "FRESH"
    assert payload["lag_by_symbol"][0]["symbol"] == "BTC/USD"
    assert payload["latest_recovery_event"]["reason_code"] == "FEATURE_LAG_BREACH_DETECTED"
    assert len(reliability_store.saved_system_snapshots) == 1


def test_predict_happy_path(tmp_path: Path) -> None:
    """`/predict` should return probabilities and class labels from the saved artifact."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))

    response = client.get("/predict", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "UP"
    assert payload["prob_up"] == 0.7
    assert payload["row_id"].startswith("BTC/USD|")
    assert payload["regime_label"] == "TREND_UP"
    assert payload["regime_run_id"] == "20260320T120000Z"
    assert payload["model_version"] == "model-0.70"
    assert payload["freshness_status"] == "FRESH"
    assert payload["health_overall_status"] == "HEALTHY"
    assert payload["prediction_explanation"]["available"] is True
    assert payload["prediction_explanation"]["method"] == "ONE_AT_A_TIME_REFERENCE_ABLATION"
    assert isinstance(payload["top_features"], list)


def test_predict_and_signal_include_m14_explainability_fields(tmp_path: Path) -> None:
    """`/predict` and `/signal` should expose additive M14 explanation fields."""
    row = _feature_row(
        realized_vol_12=0.03,
        momentum_3=0.08,
        macd_line_12_26=1.2,
    )
    database = FakeDatabase(
        row=row,
        reference_vector={
            "momentum_3": 0.02,
            "realized_vol_12": 0.04,
            "volume_zscore_12": 0.0,
        },
    )
    client = _build_client(
        tmp_path,
        database=database,
        artifact_path=_write_feature_aware_artifact(tmp_path),
    )

    predict_payload = client.get("/predict", params={"symbol": "BTC/USD"}).json()
    signal_payload = client.get("/signal", params={"symbol": "BTC/USD"}).json()

    assert predict_payload["model_version"] == "m3-20260321T120000Z"
    assert predict_payload["top_features"][0]["feature_name"] == "volume_zscore_12"
    assert predict_payload["prediction_explanation"]["available"] is True
    assert predict_payload["prediction_explanation"]["reference_vector_path"].endswith(
        "artifacts\\explainability\\m3-20260321T120000Z\\reference.json"
    )
    assert signal_payload["model_version"] == "m3-20260321T120000Z"
    assert signal_payload["top_features"][0]["feature_name"] == "volume_zscore_12"
    assert signal_payload["prediction_explanation"]["available"] is True
    assert signal_payload["threshold_snapshot"]["buy_prob_up"] == 0.54
    assert signal_payload["threshold_snapshot"]["allow_new_long_entries"] is True
    assert signal_payload["regime_reason"]["reason_code"] == "REGIME_TREND_UP"
    assert signal_payload["signal_explanation"]["decision_source"] == "model"
    assert signal_payload["signal_explanation"]["available"] is True


def test_signal_accepts_exact_interval_begin_selector(tmp_path: Path) -> None:
    """The additive M4 contract should allow M5 to request an exact finalized candle."""
    row = _feature_row(base_time=datetime(2026, 3, 19, 22, 0, tzinfo=timezone.utc))
    database = FakeDatabase(row=row)
    client = _build_client(tmp_path, prob_up=0.7, database=database)

    response = client.get(
        "/signal",
        params={
            "symbol": "BTC/USD",
            "interval_begin": row["interval_begin"].isoformat().replace("+00:00", "Z"),
        },
    )

    assert response.status_code == 200
    assert database.last_interval_begin == row["interval_begin"]


def test_signal_buy_sell_and_hold(tmp_path: Path) -> None:
    """`/signal` should map probabilities onto BUY, SELL, and HOLD thresholds."""
    buy_client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))
    sell_client = _build_client(tmp_path, prob_up=0.3, database=FakeDatabase(row=_feature_row()))
    hold_client = _build_client(tmp_path, prob_up=0.5, database=FakeDatabase(row=_feature_row()))

    buy_payload = buy_client.get("/signal", params={"symbol": "BTC/USD"}).json()
    sell_payload = sell_client.get("/signal", params={"symbol": "BTC/USD"}).json()
    hold_payload = hold_client.get("/signal", params={"symbol": "BTC/USD"}).json()

    assert buy_payload["signal"] == "BUY"
    assert buy_payload["trade_allowed"] is True
    assert buy_payload["decision_source"] == "model"
    assert buy_payload["signal_status"] == "MODEL_SIGNAL"
    assert sell_payload["signal"] == "SELL"
    assert sell_payload["trade_allowed"] is True
    assert sell_payload["decision_source"] == "model"
    assert hold_payload["signal"] == "HOLD"
    assert hold_payload["trade_allowed"] is False
    assert hold_payload["signal_status"] == "MODEL_HOLD"


def test_regime_endpoint_returns_exact_row_regime_and_policy(tmp_path: Path) -> None:
    """`/regime` should expose the exact-row M8 regime plus the active M9 policy."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))

    response = client.get("/regime", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["regime_label"] == "TREND_UP"
    assert payload["regime_run_id"] == "20260320T120000Z"
    assert payload["trade_allowed"] is True
    assert payload["buy_prob_up"] == 0.54
    assert payload["sell_prob_up"] == 0.44


def test_signal_blocks_new_buy_entries_in_high_vol_regime(tmp_path: Path) -> None:
    """High-volatility rows should block new BUY entries while preserving the regime label."""
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(
            row=_feature_row(
                realized_vol_12=0.08,
                momentum_3=0.03,
                macd_line_12_26=1.2,
            )
        ),
    )

    response = client.get("/signal", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["signal"] == "HOLD"
    assert payload["regime_label"] == "HIGH_VOL"
    assert payload["trade_allowed"] is False


def test_signal_sell_remains_allowed_in_high_vol_regime(tmp_path: Path) -> None:
    """No-trade regimes must still allow SELL because SELL reduces long-only risk."""
    client = _build_client(
        tmp_path,
        prob_up=0.3,
        database=FakeDatabase(
            row=_feature_row(
                realized_vol_12=0.08,
                momentum_3=0.03,
                macd_line_12_26=1.2,
            )
        ),
    )

    response = client.get("/signal", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["signal"] == "SELL"
    assert payload["regime_label"] == "HIGH_VOL"
    assert payload["trade_allowed"] is True


def test_invalid_symbol_and_missing_row_behaviour(tmp_path: Path) -> None:
    """Invalid symbols should 400 and `/signal` should degrade to a reliability HOLD."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=None))

    invalid_response = client.get("/predict", params={"symbol": "DOGE/USD"})
    missing_response = client.get("/predict", params={"symbol": "BTC/USD"})
    missing_signal_response = client.get(
        "/signal",
        params={
            "symbol": "BTC/USD",
            "interval_begin": "2026-03-21T12:00:00Z",
        },
    )

    assert invalid_response.status_code == 400
    assert missing_response.status_code == 404
    assert missing_signal_response.status_code == 200
    missing_signal_payload = missing_signal_response.json()
    assert missing_signal_payload["signal"] == "HOLD"
    assert missing_signal_payload["decision_source"] == "reliability"
    assert missing_signal_payload["reason_code"] == "RELIABILITY_HOLD_MISSING_FEATURE_ROW"
    assert missing_signal_payload["row_id"] == "BTC/USD|2026-03-21T12:00:00Z"
    assert missing_signal_payload["top_features"] == []
    assert missing_signal_payload["prediction_explanation"]["available"] is False
    assert missing_signal_payload["signal_explanation"]["decision_source"] == "reliability"
    assert missing_signal_payload["threshold_snapshot"]["regime_label"] is None


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
    assert payload["health_overall_status"] == "HEALTHY"
    assert payload["reason_code"] == "HEALTH_HEALTHY"
    assert payload["freshness_summary"]["BTC/USD"]["freshness_status"] == "FRESH"


def test_predict_returns_503_when_database_is_unavailable(tmp_path: Path) -> None:
    """DB failures should propagate as 503s from the inference endpoints."""
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(fetch_error=DatabaseUnavailableError("db down")),
    )

    response = client.get("/predict", params={"symbol": "BTC/USD"})

    assert response.status_code == 503


def test_signal_degrades_to_reliability_hold_when_feature_row_is_stale(tmp_path: Path) -> None:
    """`/signal` should downgrade to a reliability HOLD for stale exact-row inputs."""
    base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
        minutes=20
    )
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row(base_time=base_time)),
    )

    response = client.get("/signal", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["signal"] == "HOLD"
    assert payload["decision_source"] == "reliability"
    assert payload["signal_status"] == "RELIABILITY_HOLD"
    assert payload["freshness_status"] == "STALE"
    assert payload["reason_code"] == "RELIABILITY_HOLD_STALE_FEATURE_ROW"
    assert payload["top_features"] == []
    assert payload["prediction_explanation"]["available"] is False
    assert payload["threshold_snapshot"]["regime_label"] == "TREND_UP"
    assert payload["regime_reason"]["reason_code"] == "REGIME_TREND_UP"
    assert payload["signal_explanation"]["decision_source"] == "reliability"


def test_freshness_endpoint_returns_exact_row_status(tmp_path: Path) -> None:
    """`/freshness` should surface the exact-row freshness state for a live symbol."""
    row = _feature_row()
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=row))

    response = client.get("/freshness", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["health_overall_status"] == "HEALTHY"
    assert payload["feature_freshness_status"] == "FRESH"
    assert payload["regime_freshness_status"] == "FRESH"
    assert payload["row_id"].startswith("BTC/USD|")


def test_freshness_endpoint_reports_missing_exact_row(tmp_path: Path) -> None:
    """`/freshness` should mark the requested candle stale when no exact row exists."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=None))

    response = client.get(
        "/freshness",
        params={
            "symbol": "BTC/USD",
            "interval_begin": "2026-03-21T12:00:00Z",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["freshness_status"] == "STALE"
    assert payload["reason_code"] == "FEATURE_ROW_MISSING"
    assert payload["row_id"] == "BTC/USD|2026-03-21T12:00:00Z"
