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
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _HealthyHttpClient:
    async def get(  # pylint: disable=too-many-return-statements
        # pylint: disable=too-many-branches
        self,
        path: str,
        *_args,
        **_kwargs,
    ):
        if path == "/health":
            return _Response(
                200,
                {
                    "status": "ok",
                    "service": "inference",
                    "runtime_profile": "paper",
                    "active_alert_count": 1,
                    "max_alert_severity": "WARNING",
                    "startup_safety_status": "PASSED",
                    "startup_safety_reason_code": "STARTUP_SAFETY_PASSED",
                    "model_loaded": True,
                    "model_name": "logistic_regression",
                    "model_artifact_path": "artifacts/training/m3/model.joblib",
                    "regime_loaded": True,
                    "regime_run_id": "20260320T120000Z",
                    "regime_artifact_path": "artifacts/regime/m8/20260320T120000Z/thresholds.json",
                    "database": "healthy",
                    "started_at": "2026-03-20T12:00:00Z",
                    "health_overall_status": "HEALTHY",
                    "reason_code": "HEALTH_HEALTHY",
                    "freshness_status": "FRESH",
                    "active_continual_learning_profile_id": "cl-profile-1",
                    "continual_learning_status": "ACTIVE",
                    "continual_learning_drift_cap_status": "WATCH",
                },
            )
        if path == "/freshness":
            return _Response(
                200,
                {
                    "symbol": "BTC/USD",
                    "row_id": "BTC/USD|2026-03-20T11:55:00Z",
                    "interval_begin": "2026-03-20T11:55:00Z",
                    "as_of_time": "2026-03-20T12:00:00Z",
                    "health_overall_status": "HEALTHY",
                    "freshness_status": "FRESH",
                    "reason_code": "HEALTH_HEALTHY",
                    "feature_freshness_status": "FRESH",
                    "feature_reason_code": "FEATURE_FRESH",
                    "feature_age_seconds": 0.0,
                    "regime_freshness_status": "FRESH",
                    "regime_reason_code": "REGIME_FRESH",
                    "regime_age_seconds": 0.0,
                    "detail": "Exact-row regime resolution succeeded",
                },
            )
        if path == "/reliability/system":
            return _Response(
                200,
                {
                    "service_name": "streamalpha",
                    "checked_at": "2026-03-20T12:00:00Z",
                    "health_overall_status": "DEGRADED",
                    "reason_codes": [
                        "SIGNAL_FETCH_FAILED",
                        "FEATURE_LAG_BREACH",
                    ],
                    "lag_breach_active": True,
                    "services": [
                        {
                            "service_name": "producer",
                            "component_name": "producer",
                            "checked_at": "2026-03-20T12:00:00Z",
                            "heartbeat_at": "2026-03-20T12:00:00Z",
                            "heartbeat_age_seconds": 0.0,
                            "heartbeat_freshness_status": "FRESH",
                            "health_overall_status": "HEALTHY",
                            "reason_code": "SERVICE_HEARTBEAT_HEALTHY",
                            "detail": "producer healthy",
                            "feed_freshness_status": "FRESH",
                            "feed_reason_code": "FEED_FRESH",
                            "feed_age_seconds": 0.0,
                        }
                    ],
                    "lag_by_symbol": [
                        {
                            "service_name": "features",
                            "component_name": "features",
                            "symbol": "BTC/USD",
                            "evaluated_at": "2026-03-20T12:00:00Z",
                            "latest_raw_event_received_at": "2026-03-20T12:00:00Z",
                            "latest_feature_interval_begin": "2026-03-20T11:55:00Z",
                            "latest_feature_as_of_time": "2026-03-20T11:55:00Z",
                            "time_lag_seconds": 600.0,
                            "processing_lag_seconds": 600.0,
                            "time_lag_reason_code": "FEATURE_TIME_LAG_BREACH",
                            "processing_lag_reason_code": "FEATURE_PROCESSING_LAG_BREACH",
                            "lag_breach": True,
                            "health_overall_status": "DEGRADED",
                            "reason_code": "FEATURE_LAG_BREACH",
                            "detail": "lag breach",
                        }
                    ],
                    "latest_recovery_event": {
                        "service_name": "features",
                        "component_name": "BTC/USD",
                        "event_type": "FEATURE_LAG_TRANSITION",
                        "event_time": "2026-03-20T12:00:00Z",
                        "reason_code": "FEATURE_LAG_BREACH_DETECTED",
                        "health_overall_status": "DEGRADED",
                        "freshness_status": "STALE",
                        "breaker_state": None,
                        "detail": "lag breach",
                    },
                },
            )
        if path == "/alerts/active":
            return _Response(
                200,
                [
                    {
                        "fingerprint": "paper-trader|paper|FEED_STALE|producer|*",
                        "service_name": "paper-trader",
                        "execution_mode": "paper",
                        "category": "FEED_STALE",
                        "severity": "WARNING",
                        "reason_code": "FEED_STALE",
                        "source_component": "producer",
                        "is_active": True,
                        "opened_at": "2026-03-20T11:55:00Z",
                        "last_seen_at": "2026-03-20T12:00:00Z",
                        "occurrence_count": 2,
                    }
                ],
            )
        if path == "/alerts/timeline":
            return _Response(
                200,
                [
                    {
                        "id": 7,
                        "service_name": "paper-trader",
                        "execution_mode": "paper",
                        "category": "FEED_STALE",
                        "severity": "WARNING",
                        "event_state": "OPEN",
                        "reason_code": "FEED_STALE",
                        "source_component": "producer",
                        "symbol": None,
                        "fingerprint": "paper-trader|paper|FEED_STALE|producer|*",
                        "summary_text": "Producer feed is stale.",
                        "detail": "feed age exceeded max threshold",
                        "event_time": "2026-03-20T12:00:00Z",
                        "related_order_request_id": None,
                        "related_decision_trace_id": None,
                        "payload_json": {"feed_age_seconds": 120.0},
                        "created_at": "2026-03-20T12:00:00Z",
                    }
                ],
            )
        if path == "/operations/startup-safety":
            return _Response(
                200,
                {
                    "schema_version": "m17_startup_safety_report_v1",
                    "generated_at": "2026-03-20T12:00:00Z",
                    "service_name": "paper-trader",
                    "execution_mode": "paper",
                    "runtime_profile": "paper",
                    "startup_safety_passed": True,
                    "primary_reason_code": "STARTUP_SAFETY_PASSED",
                    "summary_text": "Startup safety is clear for this non-live runtime.",
                    "startup_validation": {
                        "report_path": "artifacts/runtime/startup_report.json",
                        "report_exists": True,
                        "startup_validation_passed": True,
                        "primary_reason_code": "STARTUP_SAFETY_PASSED",
                        "summary_text": "M16 startup validation passed.",
                        "payload": {},
                    },
                    "live_startup": {
                        "report_path": "artifacts/runtime/startup_report.json",
                        "report_exists": False,
                        "startup_validation_passed": True,
                        "primary_reason_code": "STARTUP_SAFETY_PASSED",
                        "summary_text": (
                            "M12 guarded-live startup checks are not required "
                            "for this mode."
                        ),
                        "payload": {},
                    },
                },
            )
        if path == "/operations/daily-summary":
            return _Response(
                200,
                {
                    "schema_version": "m17_daily_operations_summary_v1",
                    "generated_at": "2026-03-20T12:00:00Z",
                    "service_name": "paper-trader",
                    "execution_mode": "paper",
                    "runtime_profile": "paper",
                    "summary_date": "2026-03-20",
                    "counts_by_category": {"FEED_STALE": 1, "CONSUMER_LAG": 0},
                    "unresolved_count": 1,
                    "highest_severity": "WARNING",
                    "startup_safety_status": {
                        "startup_safety_passed": True,
                        "primary_reason_code": "STARTUP_SAFETY_PASSED",
                    },
                    "order_failure_counts": {
                        "rejected": 1,
                        "failed": 0,
                        "total_failures": 1,
                        "window_minutes": 15,
                    },
                    "drawdown_state": {"available": True, "breached": False},
                    "actionable_signal_counts": {
                        "buy_count": 2,
                        "sell_count": 1,
                        "total_actionable": 3,
                        "decision_trace_count": 5,
                    },
                    "silence_flood_episodes": {
                        "signal_silence_events": 0,
                        "signal_flood_events": 1,
                    },
                    "live_mode_activation_count": 0,
                },
            )
        if path == "/adaptation/summary":
            return _Response(
                200,
                {
                    "enabled": True,
                    "active_profile_count": 1,
                    "active_profile_id": "profile-test-1",
                    "adaptation_status": "ACTIVE",
                    "frozen_by_health_gate": False,
                    "latest_drift_status": "WATCH",
                    "latest_promotion_decision": "HOLD",
                    "reason_codes": ["ACTIVE_PROFILE_PRESENT"],
                },
            )
        if path == "/adaptation/drift":
            return _Response(
                200,
                {
                    "items": [
                        {
                            "symbol": "BTC/USD",
                            "regime_label": "ALL",
                            "detector_name": "psi",
                            "window_id": "drift-1",
                            "drift_score": 0.12,
                            "status": "WATCH",
                            "reason_code": "DRIFT_WATCH",
                            "updated_at": "2026-03-20T12:00:00Z",
                        }
                    ]
                },
            )
        if path == "/adaptation/performance":
            return _Response(
                200,
                {
                    "items": [
                        {
                            "execution_mode": "paper",
                            "symbol": "BTC/USD",
                            "regime_label": "ALL",
                            "window_id": "last_20_trades",
                            "window_type": "trade_count",
                            "trade_count": 20,
                            "net_pnl_after_costs": 0.02,
                            "max_drawdown": 0.01,
                            "profit_factor": 1.1,
                            "win_rate": 0.55,
                            "blocked_trade_rate": 0.1,
                            "shadow_divergence_rate": 0.05,
                        }
                    ]
                },
            )
        if path == "/adaptation/profiles":
            return _Response(
                200,
                {
                    "items": [
                        {
                            "profile_id": "profile-test-1",
                            "status": "ACTIVE",
                            "execution_mode_scope": "paper",
                            "symbol_scope": "ALL",
                            "regime_scope": "ALL",
                            "rollback_target_profile_id": None,
                            "activated_at": "2026-03-20T12:00:00Z",
                        }
                    ]
                },
            )
        if path == "/adaptation/promotions":
            return _Response(
                200,
                {
                    "items": [
                        {
                            "decision_id": "promotion-test-1",
                            "target_type": "PROFILE",
                            "target_id": "profile-test-1",
                            "decision": "HOLD",
                            "summary_text": "unit-test summary",
                            "decided_at": "2026-03-20T12:00:00Z",
                            "reason_codes": ["ACTIVE_PROFILE_PRESENT"],
                        }
                    ]
                },
            )
        if path == "/continual-learning/summary":
            return _Response(
                200,
                {
                    "enabled": True,
                    "active_profile_count": 1,
                    "active_profile_id": "cl-profile-1",
                    "continual_learning_status": "ACTIVE",
                    "active_candidate_type": "CALIBRATION_OVERLAY",
                    "latest_drift_cap_status": "WATCH",
                    "latest_promotion_decision": "HOLD",
                    "latest_event_type": "PROFILE_ACTIVE",
                    "reason_codes": [
                        "AGGREGATED_SCOPE_SUMMARY",
                        "ACTIVE_PROFILE_PRESENT",
                    ],
                },
            )
        if path == "/continual-learning/profiles":
            return _Response(
                200,
                {
                    "items": [
                        {
                            "profile_id": "cl-profile-1",
                            "candidate_type": "CALIBRATION_OVERLAY",
                            "status": "ACTIVE",
                            "execution_mode_scope": "paper",
                            "symbol_scope": "BTC/USD",
                            "regime_scope": "TREND_UP",
                            "baseline_target_type": "MODEL_VERSION",
                            "baseline_target_id": "m20-live",
                            "source_experiment_id": "cl-exp-1",
                            "promotion_stage": "LIVE_ELIGIBLE",
                            "live_eligible": True,
                            "activated_at": "2026-03-20T12:00:00Z",
                        }
                    ]
                },
            )
        if path == "/continual-learning/drift-caps":
            return _Response(
                200,
                {
                    "items": [
                        {
                            "cap_id": "cl-cap-1",
                            "execution_mode_scope": "paper",
                            "symbol_scope": "BTC/USD",
                            "regime_scope": "TREND_UP",
                            "candidate_type": "CALIBRATION_OVERLAY",
                            "status": "WATCH",
                            "observed_drift_score": 0.12,
                            "reason_code": "DRIFT_WATCH",
                            "updated_at": "2026-03-20T12:00:00Z",
                        }
                    ]
                },
            )
        if path == "/continual-learning/promotions":
            return _Response(
                200,
                {
                    "items": [
                        {
                            "decision_id": "cl-decision-1",
                            "target_type": "PROFILE",
                            "target_id": "cl-profile-1",
                            "decision": "HOLD",
                            "summary_text": "hold",
                            "decided_at": "2026-03-20T12:00:00Z",
                            "reason_codes": ["ACTIVE_PROFILE_PRESENT"],
                        }
                    ]
                },
            )
        if path == "/continual-learning/events":
            return _Response(
                200,
                {
                    "items": [
                        {
                            "event_id": "cl-event-1",
                            "event_type": "PROFILE_ACTIVE",
                            "profile_id": "cl-profile-1",
                            "experiment_id": "cl-exp-1",
                            "decision_id": "cl-decision-1",
                            "reason_code": "ACTIVE_PROFILE_PRESENT",
                            "created_at": "2026-03-20T12:00:00Z",
                        }
                    ]
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
                "signal_status": "MODEL_SIGNAL",
                "decision_source": "model",
                "reason_code": "HEALTH_HEALTHY",
                "freshness_status": "FRESH",
                "health_overall_status": "HEALTHY",
                "continual_learning_status": "ACTIVE",
                "continual_learning_profile_id": "cl-profile-1",
                "continual_learning_frozen": False,
                "continual_learning": {
                    "candidate_type": "CALIBRATION_OVERLAY",
                    "promotion_stage": "LIVE_ELIGIBLE",
                    "baseline_target_type": "MODEL_VERSION",
                    "baseline_target_id": "m20-live",
                    "drift_cap_status": "WATCH",
                    "reason_codes": ["ACTIVE_PROFILE_PRESENT"],
                },
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
    assert snapshot.system_reliability is not None
    assert snapshot.system_reliability.available is False
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
    assert snapshot.api_health.runtime_profile == "paper"


def test_dashboard_snapshot_loads_adaptation_surfaces() -> None:
    async def _db_connect(_dsn: str):
        return _RecordingConnection()

    snapshot = asyncio.run(
        DashboardDataSources(
            settings=_settings(),
            trading_config=_paper_config(),
            http_client=_HealthyHttpClient(),
            db_connect=_db_connect,
        ).load_snapshot()
    )

    assert snapshot.adaptation.summary.available is True
    assert snapshot.adaptation.summary.active_profile_id == "profile-test-1"
    assert snapshot.adaptation.drift[0].status == "WATCH"
    assert snapshot.adaptation.performance[0].window_id == "last_20_trades"
    assert snapshot.adaptation.profiles[0].status == "ACTIVE"
    assert snapshot.adaptation.promotions[0].decision == "HOLD"
    assert snapshot.continual_learning.summary.available is True
    assert snapshot.continual_learning.summary.active_profile_id == "cl-profile-1"
    assert snapshot.continual_learning.summary.aggregated_scope is True
    assert snapshot.continual_learning.profiles[0].status == "ACTIVE"
    assert snapshot.continual_learning.drift_caps[0].status == "WATCH"
    assert snapshot.continual_learning.promotions[0].decision == "HOLD"
    assert snapshot.continual_learning.events[0].event_type == "PROFILE_ACTIVE"
    assert snapshot.api_health.regime_run_id == "20260320T120000Z"
    assert snapshot.api_health.health_overall_status == "HEALTHY"
    assert snapshot.api_health.active_continual_learning_profile_id == "cl-profile-1"
    assert snapshot.api_health.continual_learning_status == "ACTIVE"
    assert snapshot.signals[0].continual_learning_status == "ACTIVE"
    assert snapshot.signals[0].continual_learning_profile_id == "cl-profile-1"
    assert snapshot.signals[0].continual_learning_candidate_type == "CALIBRATION_OVERLAY"
    assert snapshot.api_health.active_alert_count == 1
    assert snapshot.api_health.max_alert_severity == "WARNING"
    assert snapshot.system_reliability is not None
    assert snapshot.system_reliability.available is True
    assert snapshot.system_reliability.health_overall_status == "DEGRADED"
    assert snapshot.system_reliability.lag_breach_active is True
    assert snapshot.active_alerts.available is True
    assert snapshot.active_alerts.items[0].category == "FEED_STALE"
    assert snapshot.alert_timeline.available is True
    assert snapshot.alert_timeline.items[0].event_state == "OPEN"
    assert snapshot.startup_safety.available is True
    assert snapshot.startup_safety.startup_safety_passed is True
    assert snapshot.daily_operations_summary.available is True
    assert snapshot.daily_operations_summary.unresolved_count == 1
    assert snapshot.signals[0].regime_label == "TREND_UP"
    assert snapshot.signals[0].trade_allowed is True
    assert snapshot.signals[0].decision_source == "model"
    assert snapshot.freshness[0].freshness_status == "FRESH"


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
                    "system_health_status": "HEALTHY",
                    "system_health_reason_code": "SYSTEM_HEALTHY",
                    "system_health_checked_at": datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc),
                    "health_gate_status": "CLEAR",
                    "health_gate_reason_code": "HEALTH_GATE_CLEAR",
                    "health_gate_detail": "all clear",
                    "broker_cash": 1000.0,
                    "broker_equity": 1005.0,
                    "reconciliation_status": "CLEAR",
                    "reconciliation_reason_code": "RECONCILIATION_CLEAR",
                    "reconciliation_checked_at": datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc),
                    "unresolved_incident_count": 0,
                    "can_submit_live_now": True,
                    "primary_block_reason_code": None,
                    "block_detail": None,
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
    assert snapshot.database.live_safety_state.health_gate_status == "CLEAR"
    assert snapshot.database.live_safety_state.reconciliation_status == "CLEAR"
    assert snapshot.database.live_safety_state.can_submit_live_now is True
    assert snapshot.database.live_safety_state.primary_block_reason_code is None
    assert snapshot.database.live_safety_state.broker_cash == 1000.0


def test_dashboard_snapshot_includes_recent_decision_traces_and_latest_blocked_trade() -> None:
    class _TraceConnection(_RecordingConnection):
        async def fetch(self, query: str, *params):
            self.fetch_calls.append((query, params))
            if "FROM \"decision_traces\"" in query:
                return [
                    {
                        "id": 11,
                        "service_name": "paper-trader",
                        "execution_mode": "paper",
                        "symbol": "BTC/USD",
                        "signal": "BUY",
                        "signal_row_id": "BTC/USD|2026-03-21T12:00:00Z",
                        "signal_as_of_time": datetime(2026, 3, 21, 12, 5, tzinfo=timezone.utc),
                        "model_name": "logistic_regression",
                        "model_version": "m3-20260321T120000Z",
                        "risk_outcome": "MODIFIED",
                        "trace_payload": {
                            "schema_version": "m14_decision_trace_v1",
                            "service_name": "paper-trader",
                            "execution_mode": "paper",
                            "symbol": "BTC/USD",
                            "signal_row_id": "BTC/USD|2026-03-21T12:00:00Z",
                            "signal_interval_begin": "2026-03-21T12:00:00Z",
                            "signal_as_of_time": "2026-03-21T12:05:00Z",
                            "model_name": "logistic_regression",
                            "model_version": "m3-20260321T120000Z",
                            "prediction": {
                                "model_name": "logistic_regression",
                                "model_version": "m3-20260321T120000Z",
                                "prob_up": 0.71,
                                "prob_down": 0.29,
                                "confidence": 0.71,
                                "predicted_class": "UP",
                                "top_features": [],
                            },
                            "signal": {
                                "signal": "BUY",
                                "reason": "buy",
                            },
                            "risk": {
                                "outcome": "MODIFIED",
                                "primary_reason_code": "VOLATILITY_SIZE_ADJUSTED",
                                "reason_codes": ["VOLATILITY_SIZE_ADJUSTED"],
                                "reason_texts": ["volatility clamp"],
                                "requested_notional": 1000.0,
                                "approved_notional": 500.0,
                                "portfolio_context": {
                                    "available_cash": 10000.0,
                                    "open_position_count": 0,
                                    "current_equity": 10000.0,
                                    "total_open_exposure_notional": 0.0,
                                    "current_symbol_exposure_notional": 0.0,
                                },
                                "service_risk_state": {
                                    "trading_day": "2026-03-21",
                                    "realized_pnl_today": 0.0,
                                    "equity_high_watermark": 10000.0,
                                    "current_equity": 10000.0,
                                    "loss_streak_count": 0,
                                    "kill_switch_enabled": False,
                                },
                                "ordered_adjustments": [],
                            },
                        },
                        "json_report_path": "artifacts/rationale/paper-trader/paper/11.json",
                        "markdown_report_path": "artifacts/rationale/paper-trader/paper/11.md",
                        "created_at": datetime(2026, 3, 21, 12, 5, tzinfo=timezone.utc),
                        "updated_at": datetime(2026, 3, 21, 12, 5, tzinfo=timezone.utc),
                    }
                ]
            return []

        async def fetchrow(self, query: str, *params):
            self.fetchrow_calls.append((query, params))
            if "risk_outcome = 'BLOCKED'" in query:
                return {
                    "id": 12,
                    "service_name": "paper-trader",
                    "execution_mode": "paper",
                    "symbol": "ETH/USD",
                    "signal": "BUY",
                    "signal_row_id": "ETH/USD|2026-03-21T12:05:00Z",
                    "signal_as_of_time": datetime(2026, 3, 21, 12, 10, tzinfo=timezone.utc),
                    "model_name": "logistic_regression",
                    "model_version": "m3-20260321T120000Z",
                    "risk_outcome": "BLOCKED",
                    "trace_payload": {
                        "schema_version": "m14_decision_trace_v1",
                        "service_name": "paper-trader",
                        "execution_mode": "paper",
                        "symbol": "ETH/USD",
                        "signal_row_id": "ETH/USD|2026-03-21T12:05:00Z",
                        "signal_interval_begin": "2026-03-21T12:05:00Z",
                        "signal_as_of_time": "2026-03-21T12:10:00Z",
                        "model_name": "logistic_regression",
                        "model_version": "m3-20260321T120000Z",
                        "prediction": {
                            "model_name": "logistic_regression",
                            "model_version": "m3-20260321T120000Z",
                            "prob_up": 0.71,
                            "prob_down": 0.29,
                            "confidence": 0.71,
                            "predicted_class": "UP",
                            "top_features": [],
                        },
                        "signal": {
                            "signal": "BUY",
                            "reason": "buy",
                        },
                        "risk": {
                            "outcome": "BLOCKED",
                            "primary_reason_code": "TRADE_NOT_ALLOWED",
                            "reason_codes": ["TRADE_NOT_ALLOWED"],
                            "reason_texts": ["trade blocked"],
                            "requested_notional": 1000.0,
                            "approved_notional": 0.0,
                            "portfolio_context": {
                                "available_cash": 10000.0,
                                "open_position_count": 0,
                                "current_equity": 10000.0,
                                "total_open_exposure_notional": 0.0,
                                "current_symbol_exposure_notional": 0.0,
                            },
                            "service_risk_state": {
                                "trading_day": "2026-03-21",
                                "realized_pnl_today": 0.0,
                                "equity_high_watermark": 10000.0,
                                "current_equity": 10000.0,
                                "loss_streak_count": 0,
                                "kill_switch_enabled": False,
                            },
                            "ordered_adjustments": [],
                        },
                        "blocked_trade": {
                            "blocked_stage": "risk",
                            "reason_code": "TRADE_NOT_ALLOWED",
                            "reason_texts": ["trade blocked"],
                        },
                    },
                    "json_report_path": "artifacts/rationale/paper-trader/paper/12.json",
                    "markdown_report_path": "artifacts/rationale/paper-trader/paper/12.md",
                    "created_at": datetime(2026, 3, 21, 12, 10, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 3, 21, 12, 10, tzinfo=timezone.utc),
                }
            return None

    connection = _TraceConnection()

    async def _db_connect(_dsn: str):
        return connection

    data_sources = DashboardDataSources(
        settings=_settings(),
        trading_config=_paper_config(),
        http_client=_HealthyHttpClient(),
        db_connect=_db_connect,
    )

    snapshot = asyncio.run(data_sources.load_snapshot())

    assert snapshot.database.recent_decision_traces
    assert snapshot.database.recent_decision_traces[0].decision_trace_id == 11
    assert snapshot.database.recent_decision_traces[0].json_report_path.endswith("11.json")
    assert snapshot.database.recent_decision_traces[0].requested_notional == 1000.0
    assert snapshot.database.recent_decision_traces[0].approved_notional == 500.0
    assert snapshot.database.latest_blocked_trade is not None
    assert snapshot.database.latest_blocked_trade.blocked_stage == "risk"
    assert snapshot.database.latest_blocked_trade.primary_reason_code == "TRADE_NOT_ALLOWED"
