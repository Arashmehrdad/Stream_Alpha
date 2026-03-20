"""Runner orchestration and summary writing for Stream Alpha M5."""

# pylint: disable=duplicate-code

from __future__ import annotations

import asyncio
import csv
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339
from app.trading.config import PaperTradingConfig
from app.trading.engine import process_candle
from app.trading.metrics import build_summary
from app.trading.repository import TradingRepository
from app.trading.schemas import FeatureCandle, PaperPosition, PortfolioContext
from app.trading.signal_client import SignalClient


class PaperTradingRunner:
    """Poll canonical feature rows, fetch M4 signals, and persist M5 paper trades."""

    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        repository: TradingRepository,
        signal_client: SignalClient,
    ) -> None:
        self.config = config
        self.repository = repository
        self.signal_client = signal_client
        self.logger = logging.getLogger(f"{config.service_name}.runner")

    async def startup(self) -> None:
        """Connect the repository before polling."""
        await self.repository.connect()

    async def shutdown(self) -> None:
        """Close the signal client and repository."""
        await self.signal_client.close()
        await self.repository.close()

    async def run_forever(self) -> None:
        """Poll forever at the configured interval."""
        while True:
            await self.run_once()
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def run_once(self) -> None:
        """Process all newly observed finalized candles exactly once."""
        states = await self.repository.load_engine_states(
            service_name=self.config.service_name,
            symbols=self.config.symbols,
        )
        open_positions = await self.repository.load_open_positions(self.config.service_name)
        available_cash = await self.repository.load_cash_balance(
            service_name=self.config.service_name,
            initial_cash=self.config.risk.initial_cash,
        )
        candles = await self._load_pending_candles(states)

        for candle in candles:
            signal = await self.signal_client.fetch_signal(
                symbol=candle.symbol,
                interval_begin=candle.interval_begin,
            )
            state = states[candle.symbol]
            open_position = open_positions.get(candle.symbol)
            portfolio = PortfolioContext(
                available_cash=available_cash,
                open_position_count=len(open_positions),
            )
            result = process_candle(
                config=self.config,
                candle=candle,
                state=state,
                open_position=open_position,
                signal=signal,
                portfolio=portfolio,
            )
            persisted_open, persisted_closed, ledger_entries = await self._persist_result(result)
            available_cash += result.cash_delta
            if persisted_open is None:
                open_positions.pop(candle.symbol, None)
            else:
                open_positions[candle.symbol] = persisted_open
            if persisted_closed is not None:
                open_positions.pop(candle.symbol, None)
            states[candle.symbol] = result.state
            await self.repository.save_engine_state(result.state)
            self.logger.info(
                "Processed paper-trading candle",
                extra={
                    "symbol": candle.symbol,
                    "interval_begin": to_rfc3339(candle.interval_begin),
                    "signal": signal.signal,
                    "ledger_entries": len(ledger_entries),
                    "cash_balance": round(available_cash, 6),
                },
            )

        await self._write_summaries(available_cash)

    async def _load_pending_candles(
        self,
        states: dict[str, Any],
    ) -> list[FeatureCandle]:
        candles: list[FeatureCandle] = []
        for symbol in self.config.symbols:
            state = states[symbol]
            candles.extend(
                await self.repository.fetch_new_feature_rows(
                    symbol=symbol,
                    source_exchange=self.config.source_exchange,
                    interval_minutes=self.config.interval_minutes,
                    last_processed_interval_begin=state.last_processed_interval_begin,
                )
            )
        return sorted(candles, key=lambda row: (row.as_of_time, row.symbol, row.interval_begin))

    async def _persist_result(
        self,
        result,
    ) -> tuple[PaperPosition | None, PaperPosition | None, tuple]:
        created_position = result.created_position
        open_position = result.open_position
        closed_position = result.closed_position
        ledger_entries = result.ledger_entries

        if created_position is not None:
            position_id = await self.repository.insert_position(created_position)
            created_position = replace(created_position, position_id=position_id)
            if open_position is not None and open_position.position_id is None:
                open_position = replace(open_position, position_id=position_id)
            if closed_position is not None and closed_position.position_id is None:
                closed_position = replace(closed_position, position_id=position_id)
            ledger_entries = tuple(
                replace(entry, position_id=position_id)
                if entry.position_id is None
                else entry
                for entry in ledger_entries
            )

        if closed_position is not None:
            await self.repository.close_position(closed_position)
            open_position = None

        for entry in ledger_entries:
            await self.repository.insert_ledger_entry(entry)

        return open_position, closed_position, ledger_entries

    async def _write_summaries(self, cash_balance: float) -> None:
        positions = await self.repository.load_positions(service_name=self.config.service_name)
        latest_prices = await self.repository.load_latest_prices(
            source_exchange=self.config.source_exchange,
            interval_minutes=self.config.interval_minutes,
            symbols=self.config.symbols,
        )
        summary = build_summary(
            config=self.config,
            positions=positions,
            latest_prices=latest_prices,
            cash_balance=cash_balance,
        )
        artifact_dir = Path(self.config.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json(artifact_dir / "latest_summary.json", summary["overall"])
        _write_csv(artifact_dir / "by_asset_summary.csv", summary["by_asset"])
        _write_csv(
            artifact_dir / "open_positions.csv",
            _positions_to_rows([row for row in positions if row.status == "OPEN"]),
        )
        _write_csv(
            artifact_dir / "closed_positions.csv",
            _positions_to_rows([row for row in positions if row.status == "CLOSED"]),
        )


def _positions_to_rows(positions: list[PaperPosition]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in positions:
        rows.append(
            {
                "position_id": position.position_id,
                "service_name": position.service_name,
                "symbol": position.symbol,
                "status": position.status,
                "entry_signal_interval_begin": to_rfc3339(position.entry_signal_interval_begin),
                "entry_signal_as_of_time": to_rfc3339(position.entry_signal_as_of_time),
                "entry_signal_row_id": position.entry_signal_row_id,
                "entry_reason": position.entry_reason,
                "entry_model_name": position.entry_model_name,
                "entry_prob_up": position.entry_prob_up,
                "entry_confidence": position.entry_confidence,
                "entry_fill_interval_begin": to_rfc3339(position.entry_fill_interval_begin),
                "entry_fill_time": to_rfc3339(position.entry_fill_time),
                "entry_price": position.entry_price,
                "quantity": position.quantity,
                "entry_notional": position.entry_notional,
                "entry_fee": position.entry_fee,
                "stop_loss_price": position.stop_loss_price,
                "take_profit_price": position.take_profit_price,
                "exit_reason": position.exit_reason,
                "exit_signal_interval_begin": None
                if position.exit_signal_interval_begin is None
                else to_rfc3339(position.exit_signal_interval_begin),
                "exit_signal_as_of_time": None
                if position.exit_signal_as_of_time is None
                else to_rfc3339(position.exit_signal_as_of_time),
                "exit_signal_row_id": position.exit_signal_row_id,
                "exit_model_name": position.exit_model_name,
                "exit_prob_up": position.exit_prob_up,
                "exit_confidence": position.exit_confidence,
                "exit_fill_interval_begin": None
                if position.exit_fill_interval_begin is None
                else to_rfc3339(position.exit_fill_interval_begin),
                "exit_fill_time": None
                if position.exit_fill_time is None
                else to_rfc3339(position.exit_fill_time),
                "exit_price": position.exit_price,
                "exit_notional": position.exit_notional,
                "exit_fee": position.exit_fee,
                "realized_pnl": position.realized_pnl,
                "realized_return": position.realized_return,
                "opened_at": None if position.opened_at is None else to_rfc3339(position.opened_at),
                "closed_at": None if position.closed_at is None else to_rfc3339(position.closed_at),
            }
        )
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(make_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    field_names = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
