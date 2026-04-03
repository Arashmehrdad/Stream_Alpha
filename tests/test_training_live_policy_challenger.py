"""Focused tests for research-only live paper challenger scoring."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.trading.config import ExecutionConfig, PaperTradingConfig, RiskConfig
from app.trading.schemas import FeatureCandle, RiskDecision, SignalDecision
from app.training.live_policy_challenger import (
    LivePolicyChallengerConfig,
    LivePolicyChallengerTracker,
    build_live_policy_challenger_summary,
    score_policy_candidate,
)
from app.training.policy_candidates import find_policy_candidate


def _trading_config(tmp_path: Path) -> PaperTradingConfig:
    return PaperTradingConfig(
        service_name="paper-trader",
        source_exchange="kraken",
        source_table="feature_ohlc",
        interval_minutes=5,
        symbols=("BTC/USD",),
        inference_base_url="http://127.0.0.1:8000",
        poll_interval_seconds=5.0,
        artifact_dir=str(tmp_path / "artifacts"),
        risk=RiskConfig(
            initial_cash=10_000.0,
            position_fraction=0.25,
            fee_bps=20.0,
            slippage_bps=5.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            cooldown_candles=1,
            max_open_positions=1,
            max_exposure_per_asset=0.25,
        ),
        execution=ExecutionConfig(mode="paper"),
    )


def _challenger_config(*candidate_names: str) -> LivePolicyChallengerConfig:
    return LivePolicyChallengerConfig(
        training_config_path="configs/training.m7.json",
        evaluation_horizon_candles=3,
        fee_rate=0.002,
        candidates=tuple(find_policy_candidate(name) for name in candidate_names),
    )


def _candle(index: int, *, close_price: float | None = None, symbol: str = "BTC/USD") -> FeatureCandle:
    interval_begin = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * index)
    resolved_close = 100.0 + index if close_price is None else close_price
    return FeatureCandle(
        id=index + 1,
        source_exchange="kraken",
        symbol=symbol,
        interval_minutes=5,
        interval_begin=interval_begin,
        interval_end=interval_begin + timedelta(minutes=5),
        as_of_time=interval_begin + timedelta(minutes=5),
        raw_event_id=f"evt-{index}",
        open_price=resolved_close - 0.5,
        high_price=resolved_close + 0.5,
        low_price=resolved_close - 1.0,
        close_price=resolved_close,
    )


def _signal(
    candle: FeatureCandle,
    *,
    prob_up: float,
    signal: str = "BUY",
    regime_label: str = "RANGE",
) -> SignalDecision:
    return SignalDecision(
        symbol=candle.symbol,
        signal=signal,
        reason=signal.lower(),
        prob_up=prob_up,
        prob_down=1.0 - prob_up,
        confidence=prob_up,
        predicted_class="UP" if prob_up >= 0.5 else "DOWN",
        row_id=candle.row_id,
        as_of_time=candle.as_of_time,
        model_name="autogluon_tabular",
        model_version="m7-test",
        regime_label=regime_label,
    )


def _risk_decision(*, regime_label: str = "RANGE") -> RiskDecision:
    return RiskDecision(
        service_name="paper-trader",
        symbol="BTC/USD",
        signal="BUY",
        outcome="APPROVED",
        approved_notional=1000.0,
        requested_notional=1000.0,
        reason_codes=("BUY_APPROVED",),
        primary_reason_code="BUY_APPROVED",
        regime_label=regime_label,
        trade_allowed=True,
    )


def test_score_policy_candidate_blocks_regimes_and_respects_thresholds() -> None:
    blocked = score_policy_candidate(
        candidate=find_policy_candidate("no_long_in_trend_down_high_vol_080"),
        prob_up=0.95,
        regime_label="TREND_DOWN",
    )
    below_threshold = score_policy_candidate(
        candidate=find_policy_candidate("range_or_trend_up_080"),
        prob_up=0.79,
        regime_label="TREND_UP",
    )
    per_regime = score_policy_candidate(
        candidate=find_policy_candidate("per_regime_thresholds_v1"),
        prob_up=0.71,
        regime_label="TREND_UP",
    )

    assert blocked.would_trade is False
    assert blocked.reason_code == "BLOCKED_REGIME"
    assert below_threshold.would_trade is False
    assert below_threshold.reason_code == "PROB_UP_BELOW_THRESHOLD"
    assert below_threshold.effective_prob_up_threshold == pytest.approx(0.80)
    assert per_regime.would_trade is True
    assert per_regime.effective_prob_up_threshold == pytest.approx(0.70)


def test_tracker_writes_observations_settlements_and_scoreboard(tmp_path: Path) -> None:
    tracker = LivePolicyChallengerTracker(
        trading_config=_trading_config(tmp_path),
        challenger_config=_challenger_config(
            "m7_research_long_only_v1",
            "range_or_trend_up_080",
        ),
    )
    first_candle = _candle(0, close_price=100.0)
    later_candles = (
        _candle(1, close_price=100.5),
        _candle(2, close_price=101.0),
        _candle(3, close_price=103.0),
    )

    tracker.observe_signal(
        candle=first_candle,
        signal=_signal(first_candle, prob_up=0.82, regime_label="RANGE"),
        risk_decision=_risk_decision(regime_label="RANGE"),
        production_trade_taken=False,
    )
    tracker.observe_signal(
        candle=later_candles[0],
        signal=_signal(later_candles[0], prob_up=0.72, signal="HOLD", regime_label="TREND_UP"),
        risk_decision=_risk_decision(regime_label="TREND_UP"),
        production_trade_taken=False,
    )
    tracker.observe_signal(
        candle=later_candles[1],
        signal=_signal(later_candles[1], prob_up=0.65, signal="HOLD", regime_label="HIGH_VOL"),
        risk_decision=_risk_decision(regime_label="HIGH_VOL"),
        production_trade_taken=False,
    )
    tracker.observe_signal(
        candle=later_candles[2],
        signal=_signal(later_candles[2], prob_up=0.60, signal="HOLD", regime_label="TREND_DOWN"),
        risk_decision=_risk_decision(regime_label="TREND_DOWN"),
        production_trade_taken=False,
    )

    summary = tracker.write_latest_scoreboard()
    assert summary is not None

    observations = tracker.observations_path.read_text(encoding="utf-8").strip().splitlines()
    settlements = tracker.settlements_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(observations) == 4
    assert len(settlements) == 1
    assert tracker.latest_scoreboard_json_path.exists()
    assert tracker.latest_scoreboard_csv_path.exists()
    assert tracker.summary_md_path.exists()

    best_candidate = summary["best_candidate"]
    assert best_candidate["candidate_name"] == "m7_research_long_only_v1"
    assert best_candidate["hypothetical_trade_count"] == 1
    assert best_candidate["cumulative_net_proxy"] == pytest.approx(0.028)


def test_summary_flags_sparse_and_never_trades_trend_up(tmp_path: Path) -> None:
    tracker = LivePolicyChallengerTracker(
        trading_config=_trading_config(tmp_path),
        challenger_config=_challenger_config("range_only_080"),
    )
    first_candle = _candle(0, close_price=100.0)
    second_candle = _candle(1, close_price=100.0)
    third_candle = _candle(2, close_price=100.0)
    settlement_candle = _candle(3, close_price=101.5)

    tracker.observe_signal(
        candle=first_candle,
        signal=_signal(first_candle, prob_up=0.84, regime_label="RANGE"),
        risk_decision=_risk_decision(regime_label="RANGE"),
        production_trade_taken=False,
    )
    tracker.observe_signal(
        candle=second_candle,
        signal=_signal(second_candle, prob_up=0.90, regime_label="TREND_UP"),
        risk_decision=_risk_decision(regime_label="TREND_UP"),
        production_trade_taken=False,
    )
    tracker.observe_signal(
        candle=third_candle,
        signal=_signal(third_candle, prob_up=0.55, signal="HOLD", regime_label="TREND_DOWN"),
        risk_decision=_risk_decision(regime_label="TREND_DOWN"),
        production_trade_taken=False,
    )
    tracker.observe_signal(
        candle=settlement_candle,
        signal=_signal(settlement_candle, prob_up=0.55, signal="HOLD", regime_label="HIGH_VOL"),
        risk_decision=_risk_decision(regime_label="HIGH_VOL"),
        production_trade_taken=False,
    )

    summary = build_live_policy_challenger_summary(
        artifact_dir=tracker.artifact_dir,
        challenger_config=tracker.challenger_config,
    )
    result = summary["candidate_results"][0]

    assert result["positive_but_sparse"] is True
    assert result["never_trades_trend_up"] is True
    assert "Trade count remains below 20." in result["warnings"]
    assert "Candidate never trades TREND_UP." in result["warnings"]

