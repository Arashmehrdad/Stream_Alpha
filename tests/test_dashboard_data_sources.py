"""Dashboard data-source tests for Stream Alpha M6."""

# pylint: disable=duplicate-code,missing-function-docstring,too-few-public-methods

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from app.common.config import Settings
from app.trading.config import ExecutionConfig, PaperTradingConfig, RiskConfig
from dashboards.data_sources import DashboardDataSources, shape_latest_feature_rows


def _settings() -> Settings:
    os.environ.setdefault("INFERENCE_MODEL_PATH", "placeholder-model.joblib")
    return Settings.from_env()


def _paper_config(*, execution_mode: str = "paper") -> PaperTradingConfig:
    return PaperTradingConfig(
        service_name="paper-trader",
        source_exchange="kraken",
        source_table="feature_ohlc",
        interval_minutes=5,
        symbols=("BTC/USD", "ETH/USD", "SOL/USD"),
        inference_base_url="http://127.0.0.1:8000",
        poll_interval_seconds=5.0,
        artifact_dir="artifacts/paper_trading",
        risk=RiskConfig(
            initial_cash=10_000.0,
            position_fraction=0.25,
            fee_bps=20.0,
            slippage_bps=5.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            cooldown_candles=1,
            max_open_positions=3,
            max_exposure_per_asset=0.25,
        ),
        execution=ExecutionConfig(mode=execution_mode, idempotency_key_version=1),
    )


def test_shape_latest_feature_rows_keeps_newest_row_per_symbol() -> None:
    """The dashboard should keep only the latest canonical row per configured asset."""
    early = datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc)
    later = datetime(2026, 3, 20, 10, 5, tzinfo=timezone.utc)
    rows = [
        {
            "symbol": "BTC/USD",
            "interval_begin": early,
            "as_of_time": early,
            "open_price": 1.0,
            "high_price": 1.0,
            "low_price": 1.0,
            "close_price": 1.0,
            "volume": 10.0,
            "log_return_1": 0.1,
            "log_return_3": 0.2,
            "rsi_14": 50.0,
            "macd_line_12_26": 0.3,
            "close_zscore_12": 0.4,
            "volume_zscore_12": 0.5,
        },
        {
            "symbol": "BTC/USD",
            "interval_begin": later,
            "as_of_time": later,
            "open_price": 2.0,
            "high_price": 2.0,
            "low_price": 2.0,
            "close_price": 2.0,
            "volume": 20.0,
            "log_return_1": 0.2,
            "log_return_3": 0.3,
            "rsi_14": 55.0,
            "macd_line_12_26": 0.4,
            "close_zscore_12": 0.5,
            "volume_zscore_12": 0.6,
        },
        {
            "symbol": "ETH/USD",
            "interval_begin": early,
            "as_of_time": early,
            "open_price": 3.0,
            "high_price": 3.0,
            "low_price": 3.0,
            "close_price": 3.0,
            "volume": 30.0,
            "log_return_1": 0.3,
            "log_return_3": 0.4,
            "rsi_14": 60.0,
            "macd_line_12_26": 0.5,
            "close_zscore_12": 0.6,
            "volume_zscore_12": 0.7,
        },
    ]

    shaped = shape_latest_feature_rows(symbols=("BTC/USD", "ETH/USD"), rows=rows)

    assert [row.symbol for row in shaped] == ["BTC/USD", "ETH/USD"]
    assert shaped[0].close_price == 2.0
    assert shaped[1].close_price == 3.0


class _FailingHttpClient:
    async def get(self, *_args, **_kwargs):
        """Raise a deterministic API error for degraded-state tests."""
        raise RuntimeError("api down")


class _Response:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _HealthyHttpClient:
    async def get(self, path: str, *_args, **_kwargs):
        if path == "/health":
            return _Response(
                200,
                {
                    "status": "ok",
                    "service": "inference",
                    "model_loaded": True,
                    "model_name": "logistic_regression",
                    "model_artifact_path": "artifacts/training/m3/model.joblib",
                    "regime_loaded": True,
                    "regime_run_id": "20260320T120000Z",
                    "regime_artifact_path": "artifacts/regime/m8/20260320T120000Z/thresholds.json",
                    "database": "healthy",
                    "started_at": "2026-03-20T12:00:00Z",
                },
            )
        return _Response(
            200,
            {
                "symbol": "BTC/USD",
                "signal": "BUY",
                "reason": "test",
                "prob_up": 0.7,
                "prob_down": 0.3,
                "confidence": 0.7,
                "predicted_class": "UP",
                "thresholds": {"buy_prob_up": 0.54, "sell_prob_up": 0.44},
                "row_id": "BTC/USD|2026-03-20T11:55:00Z",
                "as_of_time": "2026-03-20T12:00:00Z",
                "model_name": "logistic_regression",
                "regime_label": "TREND_UP",
                "regime_run_id": "20260320T120000Z",
                "trade_allowed": True,
            },
        )


async def _failing_db_connect(_dsn: str):
    raise RuntimeError("db down")


def test_dashboard_snapshot_reports_api_and_db_failures() -> None:
    """API and database failures should degrade cleanly instead of crashing the dashboard."""
    data_sources = DashboardDataSources(
        settings=_settings(),
        trading_config=_paper_config(),
        http_client=_FailingHttpClient(),
        db_connect=_failing_db_connect,
    )

    snapshot = asyncio.run(data_sources.load_snapshot())

    assert snapshot.api_health.available is False
    assert "api down" in (snapshot.api_health.error or "")
    assert snapshot.database.available is False
    assert "db down" in (snapshot.database.error or "")


def test_dashboard_snapshot_parses_regime_fields_from_api_payloads() -> None:
    """The dashboard should keep the additive M9 regime fields from `/health` and `/signal`."""
    data_sources = DashboardDataSources(
        settings=_settings(),
        trading_config=_paper_config(),
        http_client=_HealthyHttpClient(),
        db_connect=_failing_db_connect,
    )

    snapshot = asyncio.run(data_sources.load_snapshot())

    assert snapshot.api_health.regime_loaded is True
    assert snapshot.api_health.regime_run_id == "20260320T120000Z"
    assert snapshot.signals[0].regime_label == "TREND_UP"
    assert snapshot.signals[0].trade_allowed is True


class _RecordingConnection:
    def __init__(self) -> None:
        self.fetch_calls: list[tuple[str, tuple]] = []
        self.fetchval_calls: list[tuple[str, tuple]] = []
        self.fetchrow_calls: list[tuple[str, tuple]] = []

    async def fetch(self, query: str, *params):
        self.fetch_calls.append((query, params))
        return []

    async def fetchrow(self, query: str, *params):
        self.fetchrow_calls.append((query, params))
        if "execution_live_safety_state" in query:
            return None
        return None

    async def fetchval(self, query: str, *params):
        self.fetchval_calls.append((query, params))
        if "SELECT 1" in query:
            return 1
        return 0.0

    async def close(self) -> None:
        return None


def test_dashboard_snapshot_filters_database_queries_by_execution_mode() -> None:
    connection = _RecordingConnection()

    async def _db_connect(_dsn: str):
        return connection

    data_sources = DashboardDataSources(
        settings=_settings(),
        trading_config=_paper_config(execution_mode="shadow"),
        http_client=_HealthyHttpClient(),
        db_connect=_db_connect,
    )

    snapshot = asyncio.run(data_sources.load_snapshot())

    assert snapshot.database.available is True
    mode_params = [
        params[1]
        for _query, params in connection.fetch_calls
        if len(params) >= 2 and params[0] == "paper-trader"
    ]
    mode_params.extend(
        params[1]
        for _query, params in connection.fetchval_calls
        if len(params) >= 2 and params[0] == "paper-trader"
    )
    mode_params.extend(
        params[1]
        for _query, params in connection.fetchrow_calls
        if len(params) >= 2 and params[0] == "paper-trader"
    )
    assert mode_params
    assert all(mode == "shadow" for mode in mode_params)


def test_dashboard_snapshot_includes_live_safety_state_when_present() -> None:
    class _LiveConnection(_RecordingConnection):
        async def fetchrow(self, query: str, *params):
            self.fetchrow_calls.append((query, params))
            if "execution_live_safety_state" in query:
                return {
                    "service_name": "paper-trader",
                    "execution_mode": "live",
                    "broker_name": "alpaca",
                    "live_enabled": True,
                    "startup_checks_passed": True,
                    "startup_checks_passed_at": datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc),
                    "account_validated": True,
                    "account_id": "PA12345",
                    "environment_name": "paper",
                    "manual_disable_active": False,
                    "consecutive_live_failures": 0,
                    "failure_hard_stop_active": False,
                    "last_failure_reason": None,
                    "updated_at": datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc),
                }
            return None

    connection = _LiveConnection()

    async def _db_connect(_dsn: str):
        return connection

    data_sources = DashboardDataSources(
        settings=_settings(),
        trading_config=_paper_config(execution_mode="live"),
        http_client=_HealthyHttpClient(),
        db_connect=_db_connect,
    )

    snapshot = asyncio.run(data_sources.load_snapshot())

    assert snapshot.database.live_safety_state is not None
    assert snapshot.database.live_safety_state.broker_name == "alpaca"
    assert snapshot.database.live_safety_state.account_id == "PA12345"
