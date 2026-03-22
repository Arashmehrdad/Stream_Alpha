"""Unit tests for the M14 decision-trace and risk-rationale builders."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from app.ensemble.schemas import EnsembleResult
from app.ensemble.service import ENSEMBLE_FALLBACK_NO_PROFILE, ENSEMBLE_FALLBACK_SINGLE_MODEL
from app.trading.config import PaperTradingConfig, RiskConfig
from app.trading.decision_trace import (
    build_initial_decision_trace,
    build_risk_section,
    enrich_decision_trace_with_risk,
    write_rationale_reports,
)
from app.trading.risk_engine import (
    MAX_ASSET_EXPOSURE_CLAMPED,
    TRADE_NOT_ALLOWED,
    VOLATILITY_SIZE_ADJUSTED,
    default_service_risk_state,
    evaluate_risk,
)
from app.trading.schemas import (
    FeatureCandle,
    OrderLifecycleEvent,
    OrderRequest,
    PaperEngineState,
    PortfolioContext,
    SignalDecision,
)


def _config() -> PaperTradingConfig:
    return PaperTradingConfig(
        service_name="paper-trader",
        source_exchange="kraken",
        source_table="feature_ohlc",
        interval_minutes=5,
        symbols=("BTC/USD",),
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
            max_open_positions=1,
            max_exposure_per_asset=0.05,
            max_total_exposure=0.60,
            max_daily_loss_amount=250.0,
            max_drawdown_pct=0.15,
            loss_streak_limit=3,
            loss_streak_cooldown_candles=3,
            kill_switch_enabled=False,
            min_trade_notional=50.0,
            volatility_target_realized_vol=0.03,
            min_volatility_size_multiplier=0.40,
            enable_confidence_weighted_sizing=False,
            min_confidence_size_multiplier=0.50,
            regime_position_fraction_caps={
                "TREND_UP": 0.25,
            },
        ),
    )


def _candle(*, realized_vol_12: float = 0.10) -> FeatureCandle:
    return FeatureCandle(
        id=1,
        source_exchange="kraken",
        symbol="BTC/USD",
        interval_minutes=5,
        interval_begin=datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc),
        interval_end=datetime(2026, 3, 21, 12, 5, tzinfo=timezone.utc),
        as_of_time=datetime(2026, 3, 21, 12, 5, tzinfo=timezone.utc),
        raw_event_id="evt-1",
        open_price=100.0,
        high_price=101.0,
        low_price=99.0,
        close_price=100.5,
        realized_vol_12=realized_vol_12,
    )


def _signal(*, trade_allowed: bool | None = True) -> SignalDecision:
    return SignalDecision(
        symbol="BTC/USD",
        signal="BUY",
        reason="buy",
        prob_up=0.71,
        prob_down=0.29,
        confidence=0.71,
        predicted_class="UP",
        row_id="BTC/USD|2026-03-21T12:00:00Z",
        as_of_time=datetime(2026, 3, 21, 12, 5, tzinfo=timezone.utc),
        model_name="logistic_regression",
        model_version="m3-20260321T120000Z",
        regime_label="TREND_UP",
        regime_run_id="20260321T120000Z",
        trade_allowed=trade_allowed,
    )


def test_initial_trace_includes_active_ensemble_context() -> None:
    """The canonical M14 trace should preserve active ensemble context from M4."""
    signal = replace(
        _signal(),
        ensemble=EnsembleResult(
            active=True,
            ensemble_profile_id="ens-profile-1",
            approval_stage="ACTIVATED",
            raw_ensemble_confidence=0.67,
            effective_confidence=0.62,
            agreement_band="MEDIUM",
            vote_agreement_ratio=1.0,
            probability_spread=0.14,
            agreement_multiplier=0.93,
            candidate_count=2,
            weighting_reason_codes=("ENSEMBLE_PROFILE_ACTIVE", "ENSEMBLE_WEIGHT_FROM_MATRIX"),
        ).to_context_payload(
            regime_label="TREND_UP",
            regime_run_id="20260321T120000Z",
        ),
    )

    trace = build_initial_decision_trace(
        service_name="paper-trader",
        execution_mode="paper",
        signal=signal,
    )

    assert trace.payload.ensemble is not None
    assert trace.payload.ensemble.ensemble_profile_id == "ens-profile-1"
    assert trace.payload.ensemble.effective_confidence == 0.62
    assert trace.payload.ensemble.candidate_count == 2


def test_initial_trace_includes_explicit_ensemble_fallback_context() -> None:
    """Fallback ensemble context should remain explicit in the canonical trace payload."""
    signal = replace(
        _signal(),
        ensemble=EnsembleResult(
            active=False,
            fallback_reason=ENSEMBLE_FALLBACK_NO_PROFILE,
            weighting_reason_codes=(
                ENSEMBLE_FALLBACK_SINGLE_MODEL,
                ENSEMBLE_FALLBACK_NO_PROFILE,
            ),
        ).to_context_payload(
            regime_label="TREND_UP",
            regime_run_id="20260321T120000Z",
        ),
    )

    trace = build_initial_decision_trace(
        service_name="paper-trader",
        execution_mode="paper",
        signal=signal,
    )

    assert trace.payload.ensemble is not None
    assert trace.payload.ensemble.ensemble_profile_id is None
    assert trace.payload.ensemble.weighting_reason_codes == [
        ENSEMBLE_FALLBACK_SINGLE_MODEL,
        ENSEMBLE_FALLBACK_NO_PROFILE,
    ]


def _portfolio() -> PortfolioContext:
    return PortfolioContext(
        available_cash=10_000.0,
        open_position_count=0,
        current_equity=10_000.0,
        total_open_exposure_notional=0.0,
        current_symbol_exposure_notional=0.0,
    )


def test_risk_rationale_builder_preserves_modified_adjustment_order() -> None:
    """Modified buys should keep their ordered M10 adjustment steps."""
    config = _config()
    signal = _signal()
    portfolio = _portfolio()
    service_risk_state = default_service_risk_state(
        service_name=config.service_name,
        trading_day=_candle().as_of_time.date(),
        initial_cash=config.risk.initial_cash,
        kill_switch_enabled=config.risk.kill_switch_enabled,
    )

    decision = evaluate_risk(
        config=config,
        candle=_candle(),
        signal=signal,
        engine_state=PaperEngineState(service_name=config.service_name, symbol=signal.symbol),
        open_position=None,
        portfolio=portfolio,
        service_risk_state=service_risk_state,
    )
    risk_section = build_risk_section(
        decision=decision,
        portfolio=portfolio,
        service_risk_state=service_risk_state,
    )

    assert decision.outcome == "MODIFIED"
    assert decision.primary_reason_code == VOLATILITY_SIZE_ADJUSTED
    assert [step.reason_code for step in decision.ordered_adjustments] == [
        VOLATILITY_SIZE_ADJUSTED,
        MAX_ASSET_EXPOSURE_CLAMPED,
    ]
    assert [step.reason_code for step in risk_section.ordered_adjustments] == [
        VOLATILITY_SIZE_ADJUSTED,
        MAX_ASSET_EXPOSURE_CLAMPED,
    ]
    assert risk_section.primary_reason_code == VOLATILITY_SIZE_ADJUSTED


def test_risk_rationale_builder_marks_blocked_buy_at_risk_stage() -> None:
    """Blocked buys should surface a clear risk-stage rationale in the trace."""
    config = _config()
    signal = _signal(trade_allowed=False)
    portfolio = _portfolio()
    service_risk_state = default_service_risk_state(
        service_name=config.service_name,
        trading_day=_candle().as_of_time.date(),
        initial_cash=config.risk.initial_cash,
        kill_switch_enabled=config.risk.kill_switch_enabled,
    )

    decision = evaluate_risk(
        config=config,
        candle=_candle(),
        signal=signal,
        engine_state=PaperEngineState(service_name=config.service_name, symbol=signal.symbol),
        open_position=None,
        portfolio=portfolio,
        service_risk_state=service_risk_state,
    )
    enriched_trace = enrich_decision_trace_with_risk(
        trace=build_initial_decision_trace(
            service_name=config.service_name,
            execution_mode="paper",
            signal=signal,
        ),
        decision=decision,
        portfolio=portfolio,
        service_risk_state=service_risk_state,
    )

    assert decision.outcome == "BLOCKED"
    assert decision.primary_reason_code == TRADE_NOT_ALLOWED
    assert enriched_trace.payload.risk is not None
    assert enriched_trace.payload.risk.primary_reason_code == TRADE_NOT_ALLOWED
    assert enriched_trace.payload.blocked_trade is not None
    assert enriched_trace.payload.blocked_trade.blocked_stage == "risk"
    assert enriched_trace.payload.blocked_trade.reason_texts


def test_rationale_reports_are_written_deterministically(tmp_path: Path) -> None:
    """One canonical trace should always render to the same JSON and Markdown paths."""
    portfolio = _portfolio()
    signal = _signal()
    config = _config()
    service_risk_state = default_service_risk_state(
        service_name=config.service_name,
        trading_day=_candle().as_of_time.date(),
        initial_cash=config.risk.initial_cash,
        kill_switch_enabled=config.risk.kill_switch_enabled,
    )
    decision = evaluate_risk(
        config=config,
        candle=_candle(),
        signal=signal,
        engine_state=PaperEngineState(service_name=config.service_name, symbol=signal.symbol),
        open_position=None,
        portfolio=portfolio,
        service_risk_state=service_risk_state,
    )
    trace = enrich_decision_trace_with_risk(
        trace=build_initial_decision_trace(
            service_name=config.service_name,
            execution_mode="paper",
            signal=signal,
        ),
        decision=decision,
        portfolio=portfolio,
        service_risk_state=service_risk_state,
    )
    trace = replace(trace, decision_trace_id=42)
    order_request = OrderRequest(
        service_name=config.service_name,
        execution_mode="paper",
        symbol=signal.symbol,
        action="BUY",
        signal_interval_begin=_candle().interval_begin,
        signal_as_of_time=_candle().as_of_time,
        signal_row_id=signal.row_id,
        target_fill_interval_begin=_candle().interval_end,
        requested_notional=decision.requested_notional,
        approved_notional=decision.approved_notional,
        idempotency_key="trace-42-buy",
        model_name=signal.model_name,
        model_version=signal.model_version,
        confidence=signal.confidence,
        regime_label=signal.regime_label,
        regime_run_id=signal.regime_run_id,
        risk_outcome=decision.outcome,
        risk_reason_codes=decision.reason_codes,
        decision_trace_id=42,
        order_request_id=7,
    )
    lifecycle_events = (
        OrderLifecycleEvent(
            order_request_id=7,
            service_name=config.service_name,
            execution_mode="paper",
            symbol=signal.symbol,
            action="BUY",
            lifecycle_state="CREATED",
            event_time=_candle().as_of_time,
            reason_code="ORDER_REQUEST_CREATED",
            decision_trace_id=42,
            event_id=1,
        ),
        OrderLifecycleEvent(
            order_request_id=7,
            service_name=config.service_name,
            execution_mode="paper",
            symbol=signal.symbol,
            action="BUY",
            lifecycle_state="FILLED",
            event_time=_candle().interval_end,
            reason_code="PAPER_ORDER_FILLED",
            decision_trace_id=42,
            event_id=2,
        ),
    )

    updated_trace = write_rationale_reports(
        trace=trace,
        artifact_root=tmp_path / "artifacts" / "rationale",
        order_request=order_request,
        lifecycle_events=lifecycle_events,
    )

    assert updated_trace.json_report_path == (
        tmp_path / "artifacts" / "rationale" / "paper-trader" / "paper" / "42.json"
    ).as_posix()
    assert updated_trace.markdown_report_path == (
        tmp_path / "artifacts" / "rationale" / "paper-trader" / "paper" / "42.md"
    ).as_posix()
    json_path = Path(updated_trace.json_report_path)
    markdown_path = Path(updated_trace.markdown_report_path)
    assert json_path.exists()
    assert markdown_path.exists()
    assert '"decision_trace_id": 42' in json_path.read_text(encoding="utf-8")
    assert '"execution_intent"' in json_path.read_text(encoding="utf-8")
    assert "# Decision Trace 42" in markdown_path.read_text(encoding="utf-8")
    assert "## Lifecycle Events" in markdown_path.read_text(encoding="utf-8")
