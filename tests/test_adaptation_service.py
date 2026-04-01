"""Focused tests for the Stream Alpha M19 adaptation layer."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from app.adaptation.calibration import apply_calibration, build_isotonic_calibration_profile
from app.adaptation.config import default_adaptation_config_path, load_adaptation_config
from app.adaptation.drift import classify_drift, population_stability_index
from app.adaptation.performance import build_rolling_performance_windows
from app.adaptation.promotion import decide_promotion
from app.adaptation.schemas import (
    AdaptiveDriftRecord,
    AdaptivePerformanceWindow,
    AdaptiveProfileRecord,
    CalibrationProfile,
    SizingPolicy,
    ThresholdPolicy,
)
from app.adaptation.service import AdaptationService
from app.adaptation.sizing import bounded_size_multiplier
from app.adaptation.thresholds import bounded_effective_thresholds
from app.trading.schemas import PaperPosition, TradeLedgerEntry


def test_population_stability_index_reports_zero_for_identical_distributions() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert population_stability_index(values, values) == 0.0


def test_classify_drift_returns_watch_and_breach() -> None:
    assert classify_drift(0.12, warning_threshold=0.10, breach_threshold=0.20) == (
        "WATCH",
        "DRIFT_WATCH",
    )
    assert classify_drift(0.21, warning_threshold=0.10, breach_threshold=0.20) == (
        "BREACHED",
        "DRIFT_BREACH",
    )


def test_build_rolling_performance_windows_tracks_trade_and_time_windows() -> None:
    now = datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)
    rows = [
        {
            "event_time": datetime(2026, 3, 21, 10, index, tzinfo=timezone.utc),
            "realized_pnl": 10.0 if index % 2 == 0 else -5.0,
            "slippage_bps": 3.0,
            "predicted_positive": True,
            "true_positive": index % 2 == 0,
            "blocked": index == 1,
            "shadow_diverged": index == 2,
            "health_context": {"health_overall_status": "HEALTHY"},
        }
        for index in range(4)
    ]
    windows = build_rolling_performance_windows(
        execution_mode="shadow",
        symbol="BTC/USD",
        regime_label="TREND_UP",
        rows=rows,
        trade_counts=(2,),
        day_windows=(7,),
        now=now,
    )

    assert [item.window_id for item in windows] == ["last_2_trades", "last_7d"]
    assert windows[0].trade_count == 2
    assert windows[1].blocked_trade_rate > 0.0


def test_bounded_thresholds_and_sizing_stay_inside_configured_limits() -> None:
    config = load_adaptation_config(default_adaptation_config_path())
    performance = AdaptivePerformanceWindow(
        execution_mode="paper",
        symbol="BTC/USD",
        regime_label="ALL",
        window_id="last_20_trades",
        window_type="trade_count",
        window_start=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        trade_count=20,
        net_pnl_after_costs=0.03,
        max_drawdown=0.01,
        profit_factor=1.2,
        expectancy=0.001,
        win_rate=0.55,
        precision=0.56,
        avg_slippage_bps=3.0,
        blocked_trade_rate=0.1,
        shadow_divergence_rate=0.05,
        health_context={},
    )

    thresholds = bounded_effective_thresholds(
        base_buy_prob_up=0.55,
        base_sell_prob_up=0.45,
        calibrated_confidence=0.70,
        performance=performance,
        configured_delta=0.50,
        bounds=config.threshold_bounds,
    )
    multiplier = bounded_size_multiplier(
        configured_multiplier=2.0,
        calibrated_confidence=0.70,
        performance=performance,
        bounds=config.sizing_bounds,
    )

    assert thresholds.buy_prob_up <= config.threshold_bounds.max_buy_prob_up
    assert thresholds.sell_prob_up >= config.threshold_bounds.min_sell_prob_up
    assert config.sizing_bounds.min_multiplier <= multiplier <= config.sizing_bounds.max_multiplier


def test_isotonic_calibration_profile_recalibrates_probability() -> None:
    profile = build_isotonic_calibration_profile(
        probabilities=[0.2, 0.4, 0.6, 0.8],
        outcomes=[0, 0, 1, 1],
        source_window="shadow:last_50",
    )

    calibrated = apply_calibration(profile, 0.7)

    assert profile.method == "isotonic"
    assert 0.0 <= calibrated <= 1.0


def test_decide_promotion_rejects_unstable_challenger() -> None:
    challenger = AdaptivePerformanceWindow(
        execution_mode="shadow",
        symbol="BTC/USD",
        regime_label="ALL",
        window_id="last_50_trades",
        window_type="trade_count",
        window_start=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        trade_count=50,
        net_pnl_after_costs=0.01,
        max_drawdown=0.10,
        profit_factor=0.8,
        expectancy=0.0,
        win_rate=0.40,
        precision=0.50,
        avg_slippage_bps=5.0,
        blocked_trade_rate=0.60,
        shadow_divergence_rate=0.30,
        health_context={},
    )
    incumbent = challenger.model_copy(
        update={
            "net_pnl_after_costs": 0.02,
            "max_drawdown": 0.02,
            "profit_factor": 1.1,
            "win_rate": 0.52,
        }
    )
    config = load_adaptation_config(default_adaptation_config_path())

    decision = decide_promotion(
        decision_id="decision-1",
        target_type="PROFILE",
        target_id="profile-2",
        incumbent_id="profile-1",
        challenger=challenger,
        incumbent=incumbent,
        thresholds=config.promotion_thresholds,
        reliability_healthy=True,
    )

    assert decision.decision == "REJECT"
    assert "DRAWDOWN_DEGRADED" in decision.reason_codes


class _FakeRepo:
    def __init__(self, *, profile, drift, performance) -> None:
        self._profile = profile
        self._drift = drift
        self._performance = performance
        self._profiles = {profile.profile_id: profile}
        self._promotion_decisions = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def load_active_adaptive_profile(self, **_kwargs):
        return self._profile

    async def load_latest_adaptive_drift_state(self, **_kwargs):
        return self._drift

    async def load_latest_adaptive_performance_window(self, **_kwargs):
        return self._performance

    async def load_adaptive_profiles(self, *, limit: int):
        del limit
        return list(self._profiles.values())

    async def load_adaptive_promotion_decisions(self, *, limit: int):
        del limit
        return list(self._promotion_decisions)

    async def load_adaptive_drift_states(self, **_kwargs):
        return [self._drift]

    async def load_adaptive_performance_windows(self, **_kwargs):
        return [self._performance]

    async def load_adaptive_profile(self, *, profile_id: str):
        return self._profiles.get(profile_id)

    async def rollback_adaptive_profile(
        self,
        *,
        active_profile_id: str,
        rollback_target_profile_id: str,
        changed_at,
    ) -> None:
        active_profile = self._profiles[active_profile_id]
        rollback_target = self._profiles[rollback_target_profile_id]
        self._profiles[active_profile_id] = active_profile.model_copy(
            update={"status": "ROLLED_BACK", "superseded_at": changed_at}
        )
        self._profiles[rollback_target_profile_id] = rollback_target.model_copy(
            update={"status": "ACTIVE", "activated_at": changed_at, "superseded_at": None}
        )
        self._profile = self._profiles[rollback_target_profile_id]

    async def save_adaptive_promotion_decision(self, decision):
        self._promotion_decisions.insert(0, decision)

    def add_profile(self, profile) -> None:
        self._profiles[profile.profile_id] = profile

    def profile_status(self, profile_id: str) -> str:
        return self._profiles[profile_id].status

    def latest_promotion_decision(self):
        return self._promotion_decisions[0]


class _RuntimeRepo:
    def __init__(self) -> None:
        base_time = datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)
        self.feature_rows = {
            "BTC/USD": [
                {
                    "symbol": "BTC/USD",
                    "interval_begin": base_time + timedelta(minutes=5 * index),
                    "as_of_time": base_time + timedelta(minutes=5 * index + 5),
                    "log_return_1": 0.01 if index < 3 else 0.08,
                    "realized_vol_12": 0.02 if index < 3 else 0.12,
                }
                for index in range(6)
            ],
            "ETH/USD": [
                {
                    "symbol": "ETH/USD",
                    "interval_begin": base_time + timedelta(minutes=5 * index),
                    "as_of_time": base_time + timedelta(minutes=5 * index + 5),
                    "log_return_1": 0.00 if index < 3 else 0.06,
                    "realized_vol_12": 0.03 if index < 3 else 0.10,
                }
                for index in range(6)
            ],
        }
        self.positions = [
            PaperPosition(
                service_name="paper-trader",
                symbol="BTC/USD",
                status="CLOSED",
                entry_signal_interval_begin=base_time,
                entry_signal_as_of_time=base_time + timedelta(minutes=5),
                entry_signal_row_id="BTC/USD|2026-03-22T12:00:00Z",
                entry_reason="buy",
                entry_model_name="logistic_regression",
                entry_prob_up=0.7,
                entry_confidence=0.7,
                entry_fill_interval_begin=base_time + timedelta(minutes=5),
                entry_fill_time=base_time + timedelta(minutes=5),
                entry_price=100.0,
                quantity=10.0,
                entry_notional=1000.0,
                entry_fee=2.0,
                stop_loss_price=98.0,
                take_profit_price=104.0,
                entry_regime_label="TREND_UP",
                entry_decision_trace_id=1,
                position_id=101,
                exit_reason="SIGNAL_SELL",
                exit_signal_interval_begin=base_time + timedelta(minutes=10),
                exit_signal_as_of_time=base_time + timedelta(minutes=15),
                exit_signal_row_id="BTC/USD|2026-03-22T12:10:00Z",
                exit_model_name="logistic_regression",
                exit_prob_up=0.4,
                exit_confidence=0.6,
                exit_fill_interval_begin=base_time + timedelta(minutes=15),
                exit_fill_time=base_time + timedelta(minutes=15),
                exit_price=103.0,
                exit_notional=1030.0,
                exit_fee=2.06,
                realized_pnl=25.94,
                realized_return=0.02594,
                exit_regime_label="TREND_UP",
                exit_decision_trace_id=2,
                closed_at=base_time + timedelta(minutes=15),
            ),
            PaperPosition(
                service_name="paper-trader",
                symbol="ETH/USD",
                status="CLOSED",
                entry_signal_interval_begin=base_time + timedelta(minutes=20),
                entry_signal_as_of_time=base_time + timedelta(minutes=25),
                entry_signal_row_id="ETH/USD|2026-03-22T12:20:00Z",
                entry_reason="buy",
                entry_model_name="logistic_regression",
                entry_prob_up=0.68,
                entry_confidence=0.68,
                entry_fill_interval_begin=base_time + timedelta(minutes=25),
                entry_fill_time=base_time + timedelta(minutes=25),
                entry_price=50.0,
                quantity=20.0,
                entry_notional=1000.0,
                entry_fee=2.0,
                stop_loss_price=49.0,
                take_profit_price=52.0,
                entry_regime_label="RANGE",
                entry_decision_trace_id=3,
                position_id=202,
                exit_reason="SIGNAL_SELL",
                exit_signal_interval_begin=base_time + timedelta(minutes=30),
                exit_signal_as_of_time=base_time + timedelta(minutes=35),
                exit_signal_row_id="ETH/USD|2026-03-22T12:30:00Z",
                exit_model_name="logistic_regression",
                exit_prob_up=0.42,
                exit_confidence=0.55,
                exit_fill_interval_begin=base_time + timedelta(minutes=35),
                exit_fill_time=base_time + timedelta(minutes=35),
                exit_price=48.0,
                exit_notional=960.0,
                exit_fee=1.92,
                realized_pnl=-43.92,
                realized_return=-0.04392,
                exit_regime_label="RANGE",
                exit_decision_trace_id=4,
                closed_at=base_time + timedelta(minutes=35),
            ),
        ]
        self.ledger_entries = [
            TradeLedgerEntry(
                service_name="paper-trader",
                symbol="BTC/USD",
                action="BUY",
                reason="buy",
                fill_interval_begin=base_time + timedelta(minutes=5),
                fill_time=base_time + timedelta(minutes=5),
                fill_price=100.0,
                quantity=10.0,
                notional=1000.0,
                fee=2.0,
                slippage_bps=4.0,
                cash_flow=-1002.0,
                position_id=101,
                decision_trace_id=1,
            ),
            TradeLedgerEntry(
                service_name="paper-trader",
                symbol="BTC/USD",
                action="SELL",
                reason="sell",
                fill_interval_begin=base_time + timedelta(minutes=15),
                fill_time=base_time + timedelta(minutes=15),
                fill_price=103.0,
                quantity=10.0,
                notional=1030.0,
                fee=2.06,
                slippage_bps=6.0,
                cash_flow=1027.94,
                position_id=101,
                decision_trace_id=2,
                realized_pnl=25.94,
            ),
            TradeLedgerEntry(
                service_name="paper-trader",
                symbol="ETH/USD",
                action="BUY",
                reason="buy",
                fill_interval_begin=base_time + timedelta(minutes=25),
                fill_time=base_time + timedelta(minutes=25),
                fill_price=50.0,
                quantity=20.0,
                notional=1000.0,
                fee=2.0,
                slippage_bps=5.0,
                cash_flow=-1002.0,
                position_id=202,
                decision_trace_id=3,
            ),
            TradeLedgerEntry(
                service_name="paper-trader",
                symbol="ETH/USD",
                action="SELL",
                reason="sell",
                fill_interval_begin=base_time + timedelta(minutes=35),
                fill_time=base_time + timedelta(minutes=35),
                fill_price=48.0,
                quantity=20.0,
                notional=960.0,
                fee=1.92,
                slippage_bps=7.0,
                cash_flow=958.08,
                position_id=202,
                decision_trace_id=4,
                realized_pnl=-43.92,
            ),
        ]
        self.decision_traces = [
            SimpleNamespace(
                decision_trace_id=1,
                signal="BUY",
                signal_as_of_time=base_time + timedelta(minutes=5),
                symbol="BTC/USD",
                risk_outcome="APPROVED",
                payload=SimpleNamespace(
                    regime_reason=SimpleNamespace(regime_label="TREND_UP"),
                    blocked_trade=None,
                ),
            ),
            SimpleNamespace(
                decision_trace_id=2,
                signal="SELL",
                signal_as_of_time=base_time + timedelta(minutes=15),
                symbol="BTC/USD",
                risk_outcome="APPROVED",
                payload=SimpleNamespace(
                    regime_reason=SimpleNamespace(regime_label="TREND_UP"),
                    blocked_trade=None,
                ),
            ),
            SimpleNamespace(
                decision_trace_id=3,
                signal="BUY",
                signal_as_of_time=base_time + timedelta(minutes=25),
                symbol="ETH/USD",
                risk_outcome="APPROVED",
                payload=SimpleNamespace(
                    regime_reason=SimpleNamespace(regime_label="RANGE"),
                    blocked_trade=None,
                ),
            ),
            SimpleNamespace(
                decision_trace_id=4,
                signal="SELL",
                signal_as_of_time=base_time + timedelta(minutes=35),
                symbol="ETH/USD",
                risk_outcome="APPROVED",
                payload=SimpleNamespace(
                    regime_reason=SimpleNamespace(regime_label="RANGE"),
                    blocked_trade=None,
                ),
            ),
            SimpleNamespace(
                decision_trace_id=5,
                signal="BUY",
                signal_as_of_time=base_time + timedelta(minutes=40),
                symbol="BTC/USD",
                risk_outcome="BLOCKED",
                payload=SimpleNamespace(
                    regime_reason=SimpleNamespace(regime_label="TREND_UP"),
                    blocked_trade=SimpleNamespace(blocked_stage="risk"),
                ),
            ),
        ]
        self.saved_drift_states = []
        self.saved_performance_windows = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def load_feature_rows_for_adaptation(  # pylint: disable=too-many-arguments
        self,
        *,
        symbol: str,
        source_exchange: str,
        interval_minutes: int,
        feature_columns,
        limit: int,
    ):
        del source_exchange, interval_minutes, feature_columns
        return list(self.feature_rows[symbol])[-limit:]

    async def save_adaptive_drift_state(self, record) -> None:
        self.saved_drift_states.append(record)

    async def load_positions(self, *, service_name: str, execution_mode: str):
        del service_name, execution_mode
        return list(self.positions)

    async def load_trade_ledger_entries(self, *, service_name: str, execution_mode: str, since):
        del service_name, execution_mode
        return [entry for entry in self.ledger_entries if entry.fill_time >= since]

    async def load_decision_traces_since(self, *, service_name: str, execution_mode: str, since):
        del service_name, execution_mode
        return [trace for trace in self.decision_traces if trace.signal_as_of_time >= since]

    async def save_adaptive_performance_window(self, record) -> None:
        self.saved_performance_windows.append(record)


def test_adaptation_service_freezes_when_reliability_is_degraded() -> None:
    applied = _resolve_adaptation_with_degraded_health()

    assert applied.profile_id == "profile-1"
    assert applied.frozen_by_health_gate is True
    assert applied.adaptive_size_multiplier == 1.0
    assert "ADAPTATION_FROZEN_BY_HEALTH_GATE" in applied.adaptation_reason_codes


def _resolve_adaptation_with_degraded_health():
    profile = AdaptiveProfileRecord(
        profile_id="profile-1",
        status="ACTIVE",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        threshold_policy_json=ThresholdPolicy(buy_threshold_delta=0.02),
        sizing_policy_json=SizingPolicy(size_multiplier=1.20),
        calibration_profile_json=CalibrationProfile(method="identity"),
        source_evidence_json={"source": "unit-test"},
    )
    drift = AdaptiveDriftRecord(
        symbol="BTC/USD",
        regime_label="TREND_UP",
        detector_name="psi",
        window_id="win-1",
        reference_window_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
        reference_window_end=datetime(2026, 3, 10, tzinfo=timezone.utc),
        live_window_start=datetime(2026, 3, 11, tzinfo=timezone.utc),
        live_window_end=datetime(2026, 3, 22, tzinfo=timezone.utc),
        drift_score=0.05,
        warning_threshold=0.10,
        breach_threshold=0.20,
        status="HEALTHY",
        reason_code="DRIFT_HEALTHY",
    )
    performance = AdaptivePerformanceWindow(
        execution_mode="paper",
        symbol="BTC/USD",
        regime_label="TREND_UP",
        window_id="last_20_trades",
        window_type="trade_count",
        window_start=datetime(2026, 3, 10, tzinfo=timezone.utc),
        window_end=datetime(2026, 3, 22, tzinfo=timezone.utc),
        trade_count=20,
        net_pnl_after_costs=0.02,
        max_drawdown=0.01,
        profit_factor=1.10,
        expectancy=0.001,
        win_rate=0.55,
        precision=0.55,
        avg_slippage_bps=3.0,
        blocked_trade_rate=0.10,
        shadow_divergence_rate=0.05,
        health_context={},
    )
    service = AdaptationService(
        repository=_FakeRepo(
            profile=profile,
            drift=drift,
            performance=performance,
        )
    )

    return asyncio.run(
        service.resolve_applied_adaptation(
            execution_mode="paper",
            symbol="BTC/USD",
            regime_label="TREND_UP",
            base_buy_prob_up=0.55,
            base_sell_prob_up=0.45,
            confidence=0.70,
            health_overall_status="DEGRADED",
            freshness_status="FRESH",
        )
    )


def test_adaptation_service_writes_configured_runtime_summary_artifacts() -> None:
    profile = AdaptiveProfileRecord(
        profile_id="profile-1",
        status="ACTIVE",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        threshold_policy_json=ThresholdPolicy(buy_threshold_delta=0.02),
        sizing_policy_json=SizingPolicy(size_multiplier=1.20),
        calibration_profile_json=CalibrationProfile(method="identity"),
        source_evidence_json={"source": "unit-test"},
    )
    drift = AdaptiveDriftRecord(
        symbol="BTC/USD",
        regime_label="TREND_UP",
        detector_name="psi",
        window_id="win-1",
        reference_window_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
        reference_window_end=datetime(2026, 3, 10, tzinfo=timezone.utc),
        live_window_start=datetime(2026, 3, 11, tzinfo=timezone.utc),
        live_window_end=datetime(2026, 3, 22, tzinfo=timezone.utc),
        drift_score=0.05,
        warning_threshold=0.10,
        breach_threshold=0.20,
        status="HEALTHY",
        reason_code="DRIFT_HEALTHY",
    )
    performance = AdaptivePerformanceWindow(
        execution_mode="paper",
        symbol="BTC/USD",
        regime_label="TREND_UP",
        window_id="last_20_trades",
        window_type="trade_count",
        window_start=datetime(2026, 3, 10, tzinfo=timezone.utc),
        window_end=datetime(2026, 3, 22, tzinfo=timezone.utc),
        trade_count=20,
        net_pnl_after_costs=0.02,
        max_drawdown=0.01,
        profit_factor=1.10,
        expectancy=0.001,
        win_rate=0.55,
        precision=0.55,
        avg_slippage_bps=3.0,
        blocked_trade_rate=0.10,
        shadow_divergence_rate=0.05,
        health_context={},
    )
    with TemporaryDirectory() as temp_dir:
        config = replace(
            load_adaptation_config(default_adaptation_config_path()),
            artifacts=replace(
                load_adaptation_config(default_adaptation_config_path()).artifacts,
                root_dir=temp_dir,
                drift_summary_path=str(Path(temp_dir) / "drift" / "latest_summary.json"),
                performance_summary_path=str(
                    Path(temp_dir) / "performance" / "latest_summary.json"
                ),
                current_profile_path=str(Path(temp_dir) / "profiles" / "current.json"),
                promotions_history_path=str(Path(temp_dir) / "promotions" / "history.jsonl"),
                reports_dir=str(Path(temp_dir) / "reports"),
                challengers_dir=str(Path(temp_dir) / "challengers"),
            ),
        )
        service = AdaptationService(
            repository=_FakeRepo(profile=profile, drift=drift, performance=performance),
            config=config,
        )

        asyncio.run(
            service.resolve_applied_adaptation(
                execution_mode="paper",
                symbol="BTC/USD",
                regime_label="TREND_UP",
                base_buy_prob_up=0.55,
                base_sell_prob_up=0.45,
                confidence=0.70,
                health_overall_status="HEALTHY",
                freshness_status="FRESH",
            )
        )

        assert Path(config.artifacts.drift_summary_path).exists()
        assert Path(config.artifacts.performance_summary_path).exists()


def test_adaptation_service_rolls_back_to_target_profile_and_persists_decision() -> None:
    active_profile = AdaptiveProfileRecord(
        profile_id="profile-current",
        status="ACTIVE",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        threshold_policy_json=ThresholdPolicy(buy_threshold_delta=0.02),
        sizing_policy_json=SizingPolicy(size_multiplier=1.20),
        calibration_profile_json=CalibrationProfile(method="identity"),
        source_evidence_json={"source": "unit-test"},
        rollback_target_profile_id="profile-rollback",
    )
    rollback_target = AdaptiveProfileRecord(
        profile_id="profile-rollback",
        status="SUPERSEDED",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        threshold_policy_json=ThresholdPolicy(buy_threshold_delta=0.01),
        sizing_policy_json=SizingPolicy(size_multiplier=1.05),
        calibration_profile_json=CalibrationProfile(method="identity"),
        source_evidence_json={"source": "unit-test"},
    )
    drift = AdaptiveDriftRecord(
        symbol="BTC/USD",
        regime_label="TREND_UP",
        detector_name="psi",
        window_id="win-1",
        reference_window_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
        reference_window_end=datetime(2026, 3, 10, tzinfo=timezone.utc),
        live_window_start=datetime(2026, 3, 11, tzinfo=timezone.utc),
        live_window_end=datetime(2026, 3, 22, tzinfo=timezone.utc),
        drift_score=0.05,
        warning_threshold=0.10,
        breach_threshold=0.20,
        status="HEALTHY",
        reason_code="DRIFT_HEALTHY",
    )
    performance = AdaptivePerformanceWindow(
        execution_mode="paper",
        symbol="BTC/USD",
        regime_label="TREND_UP",
        window_id="last_20_trades",
        window_type="trade_count",
        window_start=datetime(2026, 3, 10, tzinfo=timezone.utc),
        window_end=datetime(2026, 3, 22, tzinfo=timezone.utc),
        trade_count=20,
        net_pnl_after_costs=0.02,
        max_drawdown=0.01,
        profit_factor=1.10,
        expectancy=0.001,
        win_rate=0.55,
        precision=0.55,
        avg_slippage_bps=3.0,
        blocked_trade_rate=0.10,
        shadow_divergence_rate=0.05,
        health_context={},
    )
    repository = _FakeRepo(
        profile=active_profile,
        drift=drift,
        performance=performance,
    )
    repository.add_profile(rollback_target)
    with TemporaryDirectory() as temp_dir:
        config = replace(
            load_adaptation_config(default_adaptation_config_path()),
            artifacts=replace(
                load_adaptation_config(default_adaptation_config_path()).artifacts,
                root_dir=temp_dir,
                drift_summary_path=str(Path(temp_dir) / "drift" / "latest_summary.json"),
                performance_summary_path=str(
                    Path(temp_dir) / "performance" / "latest_summary.json"
                ),
                current_profile_path=str(Path(temp_dir) / "profiles" / "current.json"),
                promotions_history_path=str(Path(temp_dir) / "promotions" / "history.jsonl"),
                reports_dir=str(Path(temp_dir) / "reports"),
                challengers_dir=str(Path(temp_dir) / "challengers"),
            ),
        )
        service = AdaptationService(repository=repository, config=config)

        decision = asyncio.run(
            service.rollback_active_profile(
                execution_mode="paper",
                symbol="BTC/USD",
                regime_label="TREND_UP",
                decision_id="rollback-1",
                summary_text="Rollback to previous stable profile.",
            )
        )

        assert decision.decision == "ROLLBACK"
        assert repository.profile_status("profile-current") == "ROLLED_BACK"
        assert repository.profile_status("profile-rollback") == "ACTIVE"
        assert repository.latest_promotion_decision().decision == "ROLLBACK"
        assert Path(config.artifacts.current_profile_path).exists()


def test_write_runtime_persisted_truth_saves_m19_drift_and_performance_rows() -> None:
    base_config = load_adaptation_config(default_adaptation_config_path())
    config = replace(
        base_config,
        rolling_windows=replace(
            base_config.rolling_windows,
            trade_counts=(1, 2),
            day_windows=(7,),
        ),
        drift=replace(
            base_config.drift,
            minimum_reference_samples=3,
            minimum_live_samples=3,
            features=("log_return_1", "realized_vol_12"),
        ),
    )
    repository = _RuntimeRepo()
    service = AdaptationService(repository=repository, config=config)

    asyncio.run(
        service.write_runtime_persisted_truth(
            service_name="paper-trader",
            execution_mode="paper",
            source_exchange="kraken",
            interval_minutes=5,
            symbols=("BTC/USD", "ETH/USD"),
            system_reliability=SimpleNamespace(
                health_overall_status="HEALTHY",
                reason_codes=("HEALTH_HEALTHY",),
            ),
        )
    )

    drift_keys = {
        (item.symbol, item.regime_label)
        for item in repository.saved_drift_states
    }
    performance_keys = {
        (item.execution_mode, item.symbol, item.regime_label, item.window_id)
        for item in repository.saved_performance_windows
    }

    assert ("BTC/USD", "ALL") in drift_keys
    assert ("ALL", "ALL") in drift_keys
    assert (
        "paper",
        "ALL",
        "ALL",
        "last_2_trades",
    ) in performance_keys
    aggregate_window = next(
        item
        for item in repository.saved_performance_windows
        if item.execution_mode == "paper"
        and item.symbol == "ALL"
        and item.regime_label == "ALL"
        and item.window_id == "last_2_trades"
    )
    assert aggregate_window.avg_slippage_bps > 0.0
    assert aggregate_window.health_context["precision_comparable_count"] == 1
    assert aggregate_window.health_context["blocked_decision_count"] == 0
