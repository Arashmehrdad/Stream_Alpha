"""Pure M10 risk-engine helpers for Stream Alpha paper trading."""

from __future__ import annotations

from app.common.time import parse_rfc3339, utc_now
from app.trading.config import PaperTradingConfig
from app.trading.risk import calculate_fee, next_cooldown_boundary
from app.trading.schemas import (
    FeatureCandle,
    PaperEngineState,
    PaperPosition,
    PendingSignalState,
    PortfolioContext,
    RiskAdjustmentStep,
    RiskDecision,
    RiskDecisionLogEntry,
    ServiceRiskState,
    SignalDecision,
)


BUY_APPROVED = "BUY_APPROVED"
SELL_EXIT_APPROVED = "SELL_EXIT_APPROVED"
SELL_NO_OPEN_POSITION = "SELL_NO_OPEN_POSITION"
HOLD_NO_OP = "HOLD_NO_OP"
TRADE_NOT_ALLOWED = "TRADE_NOT_ALLOWED"
KILL_SWITCH_ENABLED = "KILL_SWITCH_ENABLED"
MAX_DAILY_LOSS_BREACHED = "MAX_DAILY_LOSS_BREACHED"
MAX_DRAWDOWN_BREACHED = "MAX_DRAWDOWN_BREACHED"
LOSS_STREAK_COOLDOWN_ACTIVE = "LOSS_STREAK_COOLDOWN_ACTIVE"
OPEN_POSITION_EXISTS = "OPEN_POSITION_EXISTS"
ENTRY_COOLDOWN_ACTIVE = "ENTRY_COOLDOWN_ACTIVE"
MAX_OPEN_POSITIONS_REACHED = "MAX_OPEN_POSITIONS_REACHED"
INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
REGIME_POSITION_CAP_CLAMPED = "REGIME_POSITION_CAP_CLAMPED"
VOLATILITY_SIZE_ADJUSTED = "VOLATILITY_SIZE_ADJUSTED"
CONFIDENCE_SIZE_ADJUSTED = "CONFIDENCE_SIZE_ADJUSTED"
MAX_ASSET_EXPOSURE_CLAMPED = "MAX_ASSET_EXPOSURE_CLAMPED"
MAX_TOTAL_EXPOSURE_CLAMPED = "MAX_TOTAL_EXPOSURE_CLAMPED"
MIN_TRADE_NOTIONAL_BREACHED = "MIN_TRADE_NOTIONAL_BREACHED"
RISK_BLOCKED_STAGE = "risk"

RISK_REASON_TEXTS: dict[str, str] = {
    BUY_APPROVED: "Buy passed all configured M10 risk checks with no size adjustments.",
    SELL_EXIT_APPROVED: "Sell was approved to reduce or close the current open position.",
    SELL_NO_OPEN_POSITION: (
        "Sell was approved as a zero-notional exit because no open position existed."
    ),
    HOLD_NO_OP: "Hold produces no trade and requires no risk adjustment.",
    TRADE_NOT_ALLOWED: (
        "The resolved regime or M4 signal policy does not allow opening a new long trade."
    ),
    KILL_SWITCH_ENABLED: (
        "The configured kill switch is enabled, so new risk approvals are blocked."
    ),
    MAX_DAILY_LOSS_BREACHED: (
        "Realized daily loss has breached the configured maximum daily loss limit."
    ),
    MAX_DRAWDOWN_BREACHED: (
        "Portfolio drawdown has breached the configured maximum drawdown limit."
    ),
    LOSS_STREAK_COOLDOWN_ACTIVE: "Loss-streak cooldown is still active for the current candle.",
    OPEN_POSITION_EXISTS: "A long position is already open for this symbol.",
    ENTRY_COOLDOWN_ACTIVE: "The per-symbol entry cooldown is still active for this candle.",
    MAX_OPEN_POSITIONS_REACHED: "The maximum number of open positions is already reached.",
    INSUFFICIENT_CASH: "Available cash is not sufficient to request a new entry.",
    REGIME_POSITION_CAP_CLAMPED: (
        "Approved notional was reduced to the configured regime position-fraction cap."
    ),
    VOLATILITY_SIZE_ADJUSTED: (
        "Approved notional was reduced by the realized-volatility sizing multiplier."
    ),
    CONFIDENCE_SIZE_ADJUSTED: (
        "Approved notional was reduced by the confidence-weighted sizing multiplier."
    ),
    MAX_ASSET_EXPOSURE_CLAMPED: (
        "Approved notional was reduced to stay within the maximum per-asset exposure."
    ),
    MAX_TOTAL_EXPOSURE_CLAMPED: (
        "Approved notional was reduced to stay within the maximum total exposure."
    ),
    MIN_TRADE_NOTIONAL_BREACHED: (
        "Adjusted approved notional fell below the configured minimum trade notional."
    ),
}


def default_service_risk_state(
    *,
    service_name: str,
    trading_day,
    initial_cash: float,
    kill_switch_enabled: bool,
    execution_mode: str = "paper",
) -> ServiceRiskState:
    """Return the default restart-safe risk state for one service."""
    return ServiceRiskState(
        service_name=service_name,
        trading_day=trading_day,
        realized_pnl_today=0.0,
        equity_high_watermark=initial_cash,
        current_equity=initial_cash,
        loss_streak_count=0,
        execution_mode=execution_mode,
        kill_switch_enabled=kill_switch_enabled,
    )


def mark_to_market_portfolio_context(
    *,
    symbol: str,
    available_cash: float,
    open_positions: dict[str, PaperPosition],
    latest_mark_prices: dict[str, float],
    fee_bps: float,
) -> PortfolioContext:
    """Build the richer portfolio context required by the M10 risk engine."""
    total_open_exposure_notional = 0.0
    current_symbol_exposure_notional = 0.0
    liquidation_value = 0.0
    for open_symbol, position in open_positions.items():
        mark_price = latest_mark_prices.get(open_symbol, position.entry_price)
        mark_notional = mark_price * position.quantity
        exit_fee = calculate_fee(mark_notional, fee_bps)
        total_open_exposure_notional += mark_notional
        liquidation_value += mark_notional - exit_fee
        if open_symbol == symbol:
            current_symbol_exposure_notional = mark_notional
    return PortfolioContext(
        available_cash=available_cash,
        open_position_count=len(open_positions),
        current_equity=available_cash + liquidation_value,
        total_open_exposure_notional=total_open_exposure_notional,
        current_symbol_exposure_notional=current_symbol_exposure_notional,
    )


def advance_service_risk_state(
    *,
    config: PaperTradingConfig,
    state: ServiceRiskState,
    candle: FeatureCandle,
    portfolio: PortfolioContext,
    closed_position: PaperPosition | None,
) -> ServiceRiskState:
    """Advance the service-level risk state after one processed candle."""
    trading_day = candle.as_of_time.date()
    realized_pnl_today = state.realized_pnl_today
    if state.trading_day != trading_day:
        realized_pnl_today = 0.0

    loss_streak_count = state.loss_streak_count
    cooldown_until = state.loss_streak_cooldown_until_interval_begin
    if cooldown_until is not None and candle.interval_begin > cooldown_until:
        cooldown_until = None

    if closed_position is not None and closed_position.realized_pnl is not None:
        realized_pnl_today += closed_position.realized_pnl
        if closed_position.realized_pnl < 0.0:
            loss_streak_count += 1
            if (
                config.risk.loss_streak_limit > 0
                and loss_streak_count >= config.risk.loss_streak_limit
                and config.risk.loss_streak_cooldown_candles > 0
            ):
                cooldown_until = next_cooldown_boundary(
                    candle,
                    config.risk.loss_streak_cooldown_candles,
                )
        elif closed_position.realized_pnl > 0.0:
            loss_streak_count = 0
            cooldown_until = None

    current_equity = portfolio.current_equity
    return ServiceRiskState(
        service_name=state.service_name,
        trading_day=trading_day,
        realized_pnl_today=realized_pnl_today,
        equity_high_watermark=max(state.equity_high_watermark, current_equity),
        current_equity=current_equity,
        loss_streak_count=loss_streak_count,
        execution_mode=state.execution_mode,
        loss_streak_cooldown_until_interval_begin=cooldown_until,
        kill_switch_enabled=config.risk.kill_switch_enabled,
        updated_at=utc_now(),
    )


def evaluate_risk(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    *,
    config: PaperTradingConfig,
    candle: FeatureCandle,
    signal: SignalDecision,
    engine_state: PaperEngineState,
    open_position: PaperPosition | None,
    portfolio: PortfolioContext,
    service_risk_state: ServiceRiskState,
) -> RiskDecision:
    """Evaluate one fetched M4 signal against the explicit M10 rules."""
    if signal.signal == "HOLD":
        return _build_risk_decision(
            service_name=config.service_name,
            symbol=signal.symbol,
            signal=signal.signal,
            outcome="APPROVED",
            approved_notional=0.0,
            requested_notional=0.0,
            reason_codes=(HOLD_NO_OP,),
            regime_label=signal.regime_label,
            regime_run_id=signal.regime_run_id,
            trade_allowed=signal.trade_allowed,
        )

    if signal.signal == "SELL":
        approved_notional = portfolio.current_symbol_exposure_notional
        reason_code = SELL_EXIT_APPROVED if open_position is not None else SELL_NO_OPEN_POSITION
        return _build_risk_decision(
            service_name=config.service_name,
            symbol=signal.symbol,
            signal=signal.signal,
            outcome="APPROVED",
            approved_notional=approved_notional,
            requested_notional=approved_notional,
            reason_codes=(reason_code,),
            regime_label=signal.regime_label,
            regime_run_id=signal.regime_run_id,
            trade_allowed=signal.trade_allowed,
        )

    requested_notional = _requested_buy_notional(config, portfolio)
    block_code = _first_buy_blocker(
        config=config,
        candle=candle,
        signal=signal,
        engine_state=engine_state,
        open_position=open_position,
        portfolio=portfolio,
        service_risk_state=service_risk_state,
    )
    if block_code is not None:
        return _build_risk_decision(
            service_name=config.service_name,
            symbol=signal.symbol,
            signal=signal.signal,
            outcome="BLOCKED",
            approved_notional=0.0,
            requested_notional=requested_notional,
            reason_codes=(block_code,),
            primary_reason_code=block_code,
            blocked_stage=RISK_BLOCKED_STAGE,
            regime_label=signal.regime_label,
            regime_run_id=signal.regime_run_id,
            trade_allowed=signal.trade_allowed,
        )

    approved_notional = requested_notional
    reason_codes: list[str] = []
    ordered_adjustments: list[RiskAdjustmentStep] = []
    step_index = 1

    regime_cap_fraction = config.risk.regime_position_fraction_caps.get(
        signal.regime_label or "",
        config.risk.position_fraction,
    )
    regime_cap_notional = portfolio.current_equity * regime_cap_fraction
    previous_notional = approved_notional
    approved_notional, changed = _clamp_notional(approved_notional, regime_cap_notional)
    if changed:
        reason_codes.append(REGIME_POSITION_CAP_CLAMPED)
        ordered_adjustments.append(
            _build_adjustment_step(
                step_index=step_index,
                reason_code=REGIME_POSITION_CAP_CLAMPED,
                before_notional=previous_notional,
                after_notional=approved_notional,
            )
        )
        step_index += 1

    volatility_multiplier = _volatility_size_multiplier(
        realized_vol_12=candle.realized_vol_12,
        target_realized_vol=config.risk.volatility_target_realized_vol,
        min_multiplier=config.risk.min_volatility_size_multiplier,
    )
    previous_notional = approved_notional
    volatility_adjusted_notional = approved_notional * volatility_multiplier
    if volatility_adjusted_notional < approved_notional:
        approved_notional = volatility_adjusted_notional
        reason_codes.append(VOLATILITY_SIZE_ADJUSTED)
        ordered_adjustments.append(
            _build_adjustment_step(
                step_index=step_index,
                reason_code=VOLATILITY_SIZE_ADJUSTED,
                before_notional=previous_notional,
                after_notional=approved_notional,
            )
        )
        step_index += 1

    if config.risk.enable_confidence_weighted_sizing:
        confidence_multiplier = _confidence_size_multiplier(
            confidence=signal.confidence,
            min_multiplier=config.risk.min_confidence_size_multiplier,
        )
        previous_notional = approved_notional
        confidence_adjusted_notional = approved_notional * confidence_multiplier
        if confidence_adjusted_notional < approved_notional:
            approved_notional = confidence_adjusted_notional
            reason_codes.append(CONFIDENCE_SIZE_ADJUSTED)
            ordered_adjustments.append(
                _build_adjustment_step(
                    step_index=step_index,
                    reason_code=CONFIDENCE_SIZE_ADJUSTED,
                    before_notional=previous_notional,
                    after_notional=approved_notional,
                )
            )
            step_index += 1

    max_asset_notional = max(
        0.0,
        (portfolio.current_equity * config.risk.max_exposure_per_asset)
        - portfolio.current_symbol_exposure_notional,
    )
    previous_notional = approved_notional
    approved_notional, changed = _clamp_notional(approved_notional, max_asset_notional)
    if changed:
        reason_codes.append(MAX_ASSET_EXPOSURE_CLAMPED)
        ordered_adjustments.append(
            _build_adjustment_step(
                step_index=step_index,
                reason_code=MAX_ASSET_EXPOSURE_CLAMPED,
                before_notional=previous_notional,
                after_notional=approved_notional,
            )
        )
        step_index += 1

    max_total_notional = max(
        0.0,
        (portfolio.current_equity * config.risk.max_total_exposure)
        - portfolio.total_open_exposure_notional,
    )
    previous_notional = approved_notional
    approved_notional, changed = _clamp_notional(approved_notional, max_total_notional)
    if changed:
        reason_codes.append(MAX_TOTAL_EXPOSURE_CLAMPED)
        ordered_adjustments.append(
            _build_adjustment_step(
                step_index=step_index,
                reason_code=MAX_TOTAL_EXPOSURE_CLAMPED,
                before_notional=previous_notional,
                after_notional=approved_notional,
            )
        )
        step_index += 1

    if approved_notional < config.risk.min_trade_notional:
        return _build_risk_decision(
            service_name=config.service_name,
            symbol=signal.symbol,
            signal=signal.signal,
            outcome="BLOCKED",
            approved_notional=0.0,
            requested_notional=requested_notional,
            reason_codes=tuple([*reason_codes, MIN_TRADE_NOTIONAL_BREACHED]),
            primary_reason_code=MIN_TRADE_NOTIONAL_BREACHED,
            ordered_adjustments=tuple(ordered_adjustments),
            blocked_stage=RISK_BLOCKED_STAGE,
            regime_label=signal.regime_label,
            regime_run_id=signal.regime_run_id,
            trade_allowed=signal.trade_allowed,
        )

    outcome = "APPROVED" if not reason_codes else "MODIFIED"
    final_reason_codes = (BUY_APPROVED,) if not reason_codes else tuple(reason_codes)
    return _build_risk_decision(
        service_name=config.service_name,
        symbol=signal.symbol,
        signal=signal.signal,
        outcome=outcome,
        approved_notional=approved_notional,
        requested_notional=requested_notional,
        reason_codes=final_reason_codes,
        ordered_adjustments=tuple(ordered_adjustments),
        regime_label=signal.regime_label,
        regime_run_id=signal.regime_run_id,
        trade_allowed=signal.trade_allowed,
    )


def build_pending_signal_state(
    *,
    signal: SignalDecision,
    decision: RiskDecision,
) -> PendingSignalState | None:
    """Build the next pending signal state from the risk-engine outcome."""
    if signal.signal == "BUY" and decision.outcome in {"APPROVED", "MODIFIED"}:
        return PendingSignalState(
            signal=signal.signal,
            signal_interval_begin=_parse_row_id_interval_begin(signal.row_id),
            signal_as_of_time=signal.as_of_time,
            row_id=signal.row_id,
            reason=signal.reason,
            prob_up=signal.prob_up,
            prob_down=signal.prob_down,
            confidence=signal.confidence,
            predicted_class=signal.predicted_class,
            model_name=signal.model_name,
            regime_label=signal.regime_label,
            regime_run_id=signal.regime_run_id,
            approved_notional=decision.approved_notional,
            risk_outcome=decision.outcome,
            risk_reason_codes=decision.reason_codes,
        )
    if signal.signal == "SELL" and decision.approved_notional > 0.0:
        return PendingSignalState(
            signal=signal.signal,
            signal_interval_begin=_parse_row_id_interval_begin(signal.row_id),
            signal_as_of_time=signal.as_of_time,
            row_id=signal.row_id,
            reason=signal.reason,
            prob_up=signal.prob_up,
            prob_down=signal.prob_down,
            confidence=signal.confidence,
            predicted_class=signal.predicted_class,
            model_name=signal.model_name,
            regime_label=signal.regime_label,
            regime_run_id=signal.regime_run_id,
            approved_notional=decision.approved_notional,
            risk_outcome=decision.outcome,
            risk_reason_codes=decision.reason_codes,
        )
    return None


def build_risk_decision_log_entry(  # pylint: disable=too-many-arguments
    *,
    service_name: str,
    execution_mode: str,
    candle: FeatureCandle,
    signal: SignalDecision,
    decision: RiskDecision,
    portfolio: PortfolioContext,
    decision_trace_id: int | None = None,
) -> RiskDecisionLogEntry:
    """Build the persisted M10 audit row for one processed signal."""
    return RiskDecisionLogEntry(
        service_name=service_name,
        symbol=signal.symbol,
        signal=signal.signal,
        signal_interval_begin=candle.interval_begin,
        signal_as_of_time=signal.as_of_time,
        signal_row_id=signal.row_id,
        outcome=decision.outcome,
        reason_codes=decision.reason_codes,
        requested_notional=decision.requested_notional,
        approved_notional=decision.approved_notional,
        available_cash=portfolio.available_cash,
        current_equity=portfolio.current_equity,
        current_symbol_exposure_notional=portfolio.current_symbol_exposure_notional,
        total_open_exposure_notional=portfolio.total_open_exposure_notional,
        realized_vol_12=candle.realized_vol_12,
        confidence=signal.confidence,
        execution_mode=execution_mode,
        regime_label=signal.regime_label,
        regime_run_id=signal.regime_run_id,
        trade_allowed=signal.trade_allowed,
        decision_trace_id=decision_trace_id,
        model_version=signal.model_version,
    )


def current_drawdown_pct(state: ServiceRiskState) -> float:
    """Return the current drawdown percentage from the stored high watermark."""
    if state.equity_high_watermark <= 0:
        return 0.0
    return max(
        0.0,
        (state.equity_high_watermark - state.current_equity) / state.equity_high_watermark,
    )


def _requested_buy_notional(
    config: PaperTradingConfig,
    portfolio: PortfolioContext,
) -> float:
    fee_rate = config.risk.fee_bps / 10_000.0
    base_notional_pre_fee = min(
        portfolio.available_cash,
        portfolio.current_equity * config.risk.position_fraction,
    )
    if base_notional_pre_fee <= 0.0:
        return 0.0
    return base_notional_pre_fee / (1.0 + fee_rate)


def _first_buy_blocker(  # pylint: disable=too-many-arguments,too-many-return-statements
    *,
    config: PaperTradingConfig,
    candle: FeatureCandle,
    signal: SignalDecision,
    engine_state: PaperEngineState,
    open_position: PaperPosition | None,
    portfolio: PortfolioContext,
    service_risk_state: ServiceRiskState,
) -> str | None:
    if signal.trade_allowed is False:
        return TRADE_NOT_ALLOWED
    if config.risk.kill_switch_enabled or service_risk_state.kill_switch_enabled:
        return KILL_SWITCH_ENABLED
    if (
        config.risk.max_daily_loss_amount > 0.0
        and service_risk_state.realized_pnl_today <= -config.risk.max_daily_loss_amount
    ):
        return MAX_DAILY_LOSS_BREACHED
    if (
        config.risk.max_drawdown_pct > 0.0
        and current_drawdown_pct(service_risk_state) >= config.risk.max_drawdown_pct
    ):
        return MAX_DRAWDOWN_BREACHED
    if (
        service_risk_state.loss_streak_cooldown_until_interval_begin is not None
        and candle.interval_begin <= service_risk_state.loss_streak_cooldown_until_interval_begin
    ):
        return LOSS_STREAK_COOLDOWN_ACTIVE
    if open_position is not None and open_position.status == "OPEN":
        return OPEN_POSITION_EXISTS
    if (
        engine_state.cooldown_until_interval_begin is not None
        and candle.interval_begin <= engine_state.cooldown_until_interval_begin
    ):
        return ENTRY_COOLDOWN_ACTIVE
    if portfolio.open_position_count >= config.risk.max_open_positions:
        return MAX_OPEN_POSITIONS_REACHED
    if portfolio.available_cash <= 0.0:
        return INSUFFICIENT_CASH
    return None


def _volatility_size_multiplier(
    *,
    realized_vol_12: float,
    target_realized_vol: float,
    min_multiplier: float,
) -> float:
    if realized_vol_12 <= 0.0 or target_realized_vol <= 0.0:
        return 1.0
    multiplier = min(1.0, target_realized_vol / realized_vol_12)
    return max(min_multiplier, multiplier)


def _confidence_size_multiplier(*, confidence: float, min_multiplier: float) -> float:
    return max(min_multiplier, min(1.0, confidence))


def _clamp_notional(current_notional: float, cap: float) -> tuple[float, bool]:
    next_notional = min(current_notional, max(0.0, cap))
    return next_notional, next_notional < current_notional


def _build_risk_decision(  # pylint: disable=too-many-arguments
    *,
    service_name: str,
    symbol: str,
    signal: str,
    outcome: str,
    approved_notional: float,
    requested_notional: float,
    reason_codes: tuple[str, ...],
    primary_reason_code: str | None = None,
    ordered_adjustments: tuple[RiskAdjustmentStep, ...] = (),
    blocked_stage: str | None = None,
    regime_label: str | None,
    regime_run_id: str | None,
    trade_allowed: bool | None,
) -> RiskDecision:
    resolved_primary_reason = (
        primary_reason_code
        if primary_reason_code is not None
        else (None if not reason_codes else reason_codes[0])
    )
    return RiskDecision(
        service_name=service_name,
        symbol=symbol,
        signal=signal,
        outcome=outcome,
        approved_notional=approved_notional,
        requested_notional=requested_notional,
        reason_codes=reason_codes,
        primary_reason_code=resolved_primary_reason,
        reason_texts=tuple(_reason_text(reason_code) for reason_code in reason_codes),
        ordered_adjustments=ordered_adjustments,
        blocked_stage=blocked_stage,
        regime_label=regime_label,
        regime_run_id=regime_run_id,
        trade_allowed=trade_allowed,
    )


def _build_adjustment_step(
    *,
    step_index: int,
    reason_code: str,
    before_notional: float,
    after_notional: float,
) -> RiskAdjustmentStep:
    return RiskAdjustmentStep(
        step_index=step_index,
        reason_code=reason_code,
        reason_text=_reason_text(reason_code),
        before_notional=before_notional,
        after_notional=after_notional,
    )


def _reason_text(reason_code: str) -> str:
    return RISK_REASON_TEXTS.get(
        reason_code,
        f"No explicit rationale text is registered for {reason_code}.",
    )


def _parse_row_id_interval_begin(row_id: str):
    _, _, timestamp = row_id.partition("|")
    return parse_rfc3339(timestamp)
