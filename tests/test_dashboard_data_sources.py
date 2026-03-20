"""Dashboard data-source tests for Stream Alpha M6."""

# pylint: disable=duplicate-code,too-few-public-methods

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from app.common.config import Settings
from app.trading.config import PaperTradingConfig, RiskConfig
from dashboards.data_sources import DashboardDataSources, shape_latest_feature_rows


def _settings() -> Settings:
    os.environ.setdefault("INFERENCE_MODEL_PATH", "placeholder-model.joblib")
    return Settings.from_env()


def _paper_config() -> PaperTradingConfig:
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
