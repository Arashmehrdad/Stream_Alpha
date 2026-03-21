"""Typed configuration loading for the Stream Alpha M5 paper trader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class RiskConfig:  # pylint: disable=too-many-instance-attributes
    """Explicit risk and execution settings for the M5 engine."""

    initial_cash: float
    position_fraction: float
    fee_bps: float
    slippage_bps: float
    stop_loss_pct: float
    take_profit_pct: float
    cooldown_candles: int
    max_open_positions: int
    max_exposure_per_asset: float
    max_total_exposure: float = 1.0
    max_daily_loss_amount: float = 1_000_000_000.0
    max_drawdown_pct: float = 1.0
    loss_streak_limit: int = 1_000_000_000
    loss_streak_cooldown_candles: int = 0
    kill_switch_enabled: bool = False
    min_trade_notional: float = 0.0
    volatility_target_realized_vol: float = 1.0
    min_volatility_size_multiplier: float = 1.0
    enable_confidence_weighted_sizing: bool = False
    min_confidence_size_multiplier: float = 1.0
    regime_position_fraction_caps: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Explicit M11 execution settings for paper and shadow modes."""

    mode: str = "paper"
    idempotency_key_version: int = 1


@dataclass(frozen=True, slots=True)
class PaperTradingConfig:  # pylint: disable=too-many-instance-attributes
    """Checked-in paper-trading configuration."""

    service_name: str
    source_exchange: str
    source_table: str
    interval_minutes: int
    symbols: tuple[str, ...]
    inference_base_url: str
    poll_interval_seconds: float
    artifact_dir: str
    risk: RiskConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


def load_paper_trading_config(config_path: Path) -> PaperTradingConfig:
    """Load the checked-in YAML config for M5."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    risk_payload = dict(payload["risk"])
    execution_payload = dict(payload.get("execution", {}))
    symbols = tuple(str(symbol) for symbol in payload["symbols"])
    if not symbols:
        raise ValueError("Paper trading config must contain at least one symbol")

    config = PaperTradingConfig(
        service_name=str(payload["service_name"]),
        source_exchange=str(payload["source_exchange"]),
        source_table=str(payload["source_table"]),
        interval_minutes=int(payload["interval_minutes"]),
        symbols=symbols,
        inference_base_url=str(payload["inference_base_url"]).rstrip("/"),
        poll_interval_seconds=float(payload["poll_interval_seconds"]),
        artifact_dir=str(payload["artifact_dir"]),
        risk=RiskConfig(
            initial_cash=float(risk_payload["initial_cash"]),
            position_fraction=float(risk_payload["position_fraction"]),
            fee_bps=float(risk_payload["fee_bps"]),
            slippage_bps=float(risk_payload["slippage_bps"]),
            stop_loss_pct=float(risk_payload["stop_loss_pct"]),
            take_profit_pct=float(risk_payload["take_profit_pct"]),
            cooldown_candles=int(risk_payload["cooldown_candles"]),
            max_open_positions=int(risk_payload["max_open_positions"]),
            max_exposure_per_asset=float(risk_payload["max_exposure_per_asset"]),
            max_total_exposure=float(risk_payload["max_total_exposure"]),
            max_daily_loss_amount=float(risk_payload["max_daily_loss_amount"]),
            max_drawdown_pct=float(risk_payload["max_drawdown_pct"]),
            loss_streak_limit=int(risk_payload["loss_streak_limit"]),
            loss_streak_cooldown_candles=int(risk_payload["loss_streak_cooldown_candles"]),
            kill_switch_enabled=bool(risk_payload["kill_switch_enabled"]),
            min_trade_notional=float(risk_payload["min_trade_notional"]),
            volatility_target_realized_vol=float(risk_payload["volatility_target_realized_vol"]),
            min_volatility_size_multiplier=float(risk_payload["min_volatility_size_multiplier"]),
            enable_confidence_weighted_sizing=bool(
                risk_payload["enable_confidence_weighted_sizing"]
            ),
            min_confidence_size_multiplier=float(
                risk_payload["min_confidence_size_multiplier"]
            ),
            regime_position_fraction_caps={
                str(label): float(value)
                for label, value in dict(risk_payload["regime_position_fraction_caps"]).items()
            },
        ),
        execution=ExecutionConfig(
            mode=str(execution_payload.get("mode", "paper")),
            idempotency_key_version=int(
                execution_payload.get("idempotency_key_version", 1)
            ),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: PaperTradingConfig) -> None:
    if config.interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    if config.poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    if config.risk.initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if not 0 < config.risk.position_fraction <= 1:
        raise ValueError("position_fraction must be in (0, 1]")
    if not 0 < config.risk.max_exposure_per_asset <= 1:
        raise ValueError("max_exposure_per_asset must be in (0, 1]")
    if not 0 < config.risk.max_total_exposure <= 1:
        raise ValueError("max_total_exposure must be in (0, 1]")
    if config.risk.max_total_exposure < config.risk.max_exposure_per_asset:
        raise ValueError("max_total_exposure cannot be less than max_exposure_per_asset")
    if config.execution.mode not in {"paper", "shadow"}:
        raise ValueError("execution.mode must be 'paper' or 'shadow'")
    if config.execution.idempotency_key_version <= 0:
        raise ValueError("execution.idempotency_key_version must be positive")
    if config.risk.max_open_positions <= 0:
        raise ValueError("max_open_positions must be positive")
    if config.risk.cooldown_candles < 0:
        raise ValueError("cooldown_candles cannot be negative")
    if config.risk.stop_loss_pct < 0:
        raise ValueError("stop_loss_pct cannot be negative")
    if config.risk.take_profit_pct < 0:
        raise ValueError("take_profit_pct cannot be negative")
    if config.risk.fee_bps < 0 or config.risk.slippage_bps < 0:
        raise ValueError("fee_bps and slippage_bps cannot be negative")
    if config.risk.max_daily_loss_amount < 0:
        raise ValueError("max_daily_loss_amount cannot be negative")
    if not 0 <= config.risk.max_drawdown_pct <= 1:
        raise ValueError("max_drawdown_pct must be in [0, 1]")
    if config.risk.loss_streak_limit < 0:
        raise ValueError("loss_streak_limit cannot be negative")
    if config.risk.loss_streak_cooldown_candles < 0:
        raise ValueError("loss_streak_cooldown_candles cannot be negative")
    if config.risk.min_trade_notional < 0:
        raise ValueError("min_trade_notional cannot be negative")
    if config.risk.volatility_target_realized_vol < 0:
        raise ValueError("volatility_target_realized_vol cannot be negative")
    if not 0 < config.risk.min_volatility_size_multiplier <= 1:
        raise ValueError("min_volatility_size_multiplier must be in (0, 1]")
    if not 0 < config.risk.min_confidence_size_multiplier <= 1:
        raise ValueError("min_confidence_size_multiplier must be in (0, 1]")
    for regime_label, cap in config.risk.regime_position_fraction_caps.items():
        if not 0 < cap <= 1:
            raise ValueError(
                f"regime_position_fraction_caps[{regime_label}] must be in (0, 1]"
            )
