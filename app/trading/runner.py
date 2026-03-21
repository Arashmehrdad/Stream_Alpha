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
from app.common.time import to_rfc3339, utc_now
from app.trading.config import PaperTradingConfig
from app.trading.execution import (
    build_created_event,
    build_execution_adapter,
    build_order_request,
    build_pending_order_request,
)
from app.trading.live import (
    assert_live_startup_passed,
    validate_live_startup,
    write_live_status_artifact,
    write_startup_checklist_artifact,
)
from app.trading.metrics import build_summary
from app.trading.repository import TradingRepository
from app.trading.risk_engine import (
    advance_service_risk_state,
    build_pending_signal_state,
    build_risk_decision_log_entry,
    default_service_risk_state,
    evaluate_risk,
    mark_to_market_portfolio_context,
)
from app.trading.schemas import FeatureCandle, OrderRequest, PaperPosition
from app.trading.signal_client import SignalClient


class PaperTradingRunner:
    """Poll canonical feature rows, fetch M4 signals, and persist M5 paper trades."""

    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        repository: TradingRepository,
        signal_client: SignalClient,
        broker_client=None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.signal_client = signal_client
        self.execution_adapter = build_execution_adapter(
            config.execution.mode,
            broker_client=broker_client,
        )
        self.live_safety_state = None
        self.logger = logging.getLogger(f"{config.service_name}.runner")

    async def startup(self) -> None:
        """Connect the repository before polling."""
        await self.repository.connect()
        if self.config.execution.mode != "live":
            return

        existing_live_state = await self.repository.load_live_safety_state(
            service_name=self.config.service_name,
            execution_mode=self.config.execution.mode,
        )
        checklist, live_safety_state, resolved_broker_client = await validate_live_startup(
            config=self.config,
            broker_client=getattr(self.execution_adapter, "broker_client", None),
        )
        if resolved_broker_client is not None and hasattr(
            self.execution_adapter,
            "set_broker_client",
        ):
            self.execution_adapter.set_broker_client(resolved_broker_client)
        if existing_live_state is not None:
            live_safety_state = replace(
                live_safety_state,
                consecutive_live_failures=existing_live_state.consecutive_live_failures,
                failure_hard_stop_active=existing_live_state.failure_hard_stop_active,
                last_failure_reason=existing_live_state.last_failure_reason,
            )
        self.live_safety_state = live_safety_state
        await self.repository.save_live_safety_state(live_safety_state)
        write_startup_checklist_artifact(
            checklist=checklist,
            artifact_path=self.config.execution.live.startup_checklist_path,
        )
        write_live_status_artifact(
            state=live_safety_state,
            config=self.config,
        )
        assert_live_startup_passed(checklist)

    async def shutdown(self) -> None:
        """Close the signal client and repository."""
        await self.execution_adapter.close()
        await self.signal_client.close()
        await self.repository.close()

    async def run_forever(self) -> None:
        """Poll forever at the configured interval."""
        while True:
            await self.run_once()
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def run_once(self) -> None:  # pylint: disable=too-many-locals
        """Process all newly observed finalized candles exactly once."""
        if hasattr(self.execution_adapter, "begin_run"):
            self.execution_adapter.begin_run()
        states = await self.repository.load_engine_states(
            service_name=self.config.service_name,
            execution_mode=self.config.execution.mode,
            symbols=self.config.symbols,
        )
        open_positions = await self.repository.load_open_positions(
            self.config.service_name,
            execution_mode=self.config.execution.mode,
        )
        available_cash = await self.repository.load_cash_balance(
            service_name=self.config.service_name,
            execution_mode=self.config.execution.mode,
            initial_cash=self.config.risk.initial_cash,
        )
        latest_mark_prices = {
            symbol: position.entry_price
            for symbol, position in open_positions.items()
        }
        service_risk_state = await self.repository.load_service_risk_state(
            service_name=self.config.service_name,
            execution_mode=self.config.execution.mode,
        )
        candles = await self._load_pending_candles(states)
        if service_risk_state is None:
            service_risk_state = default_service_risk_state(
                service_name=self.config.service_name,
                trading_day=(candles[0].as_of_time.date() if candles else utc_now().date()),
                initial_cash=self.config.risk.initial_cash,
                kill_switch_enabled=self.config.risk.kill_switch_enabled,
                execution_mode=self.config.execution.mode,
            )

        for candle in candles:
            signal = await self.signal_client.fetch_signal(
                symbol=candle.symbol,
                interval_begin=candle.interval_begin,
            )
            state = states[candle.symbol]
            open_position = open_positions.get(candle.symbol)
            pre_execution_portfolio = mark_to_market_portfolio_context(
                symbol=candle.symbol,
                available_cash=available_cash,
                open_positions=open_positions,
                latest_mark_prices=latest_mark_prices,
                fee_bps=self.config.risk.fee_bps,
            )
            state, due_order_request = await self._hydrate_due_order_request(
                candle=candle,
                state=state,
            )
            execution_result = await self.execution_adapter.execute_candle(
                config=self.config,
                candle=candle,
                state=state,
                open_position=open_position,
                signal=signal,
                portfolio=pre_execution_portfolio,
                order_request=due_order_request,
                live_safety_state=self.live_safety_state,
            )
            (
                persisted_open,
                persisted_closed,
                ledger_entries,
            ) = await self._persist_execution_result(execution_result)
            if execution_result.live_safety_state is not None:
                self.live_safety_state = execution_result.live_safety_state
                await self.repository.save_live_safety_state(self.live_safety_state)
                write_live_status_artifact(
                    state=self.live_safety_state,
                    config=self.config,
                )
            available_cash += execution_result.cash_delta
            if persisted_open is None:
                open_positions.pop(candle.symbol, None)
            else:
                open_positions[candle.symbol] = persisted_open
            if persisted_closed is not None:
                open_positions.pop(candle.symbol, None)
            latest_mark_prices[candle.symbol] = candle.close_price
            post_execution_portfolio = mark_to_market_portfolio_context(
                symbol=candle.symbol,
                available_cash=available_cash,
                open_positions=open_positions,
                latest_mark_prices=latest_mark_prices,
                fee_bps=self.config.risk.fee_bps,
            )
            service_risk_state = advance_service_risk_state(
                config=self.config,
                state=service_risk_state,
                candle=candle,
                portfolio=post_execution_portfolio,
                closed_position=persisted_closed,
            )
            risk_decision = evaluate_risk(
                config=self.config,
                candle=candle,
                signal=signal,
                engine_state=execution_result.state,
                open_position=open_positions.get(candle.symbol),
                portfolio=post_execution_portfolio,
                service_risk_state=service_risk_state,
            )
            next_pending_signal = build_pending_signal_state(
                signal=signal,
                decision=risk_decision,
            )
            created_order_request = await self._create_next_order_request(
                candle=candle,
                signal=signal,
                decision=risk_decision,
            )
            if next_pending_signal is not None and created_order_request is not None:
                next_pending_signal = replace(
                    next_pending_signal,
                    order_request_id=created_order_request.order_request_id,
                    order_request_idempotency_key=created_order_request.idempotency_key,
                )
            next_state = replace(
                execution_result.state,
                pending_signal=next_pending_signal,
            )
            states[candle.symbol] = next_state
            await self.repository.save_engine_state(next_state)
            await self.repository.save_service_risk_state(service_risk_state)
            await self.repository.insert_risk_decision(
                build_risk_decision_log_entry(
                    service_name=self.config.service_name,
                    execution_mode=self.config.execution.mode,
                    candle=candle,
                    signal=signal,
                    decision=risk_decision,
                    portfolio=post_execution_portfolio,
                )
            )
            self.logger.info(
                "Processed paper-trading candle",
                extra={
                    "symbol": candle.symbol,
                    "interval_begin": to_rfc3339(candle.interval_begin),
                    "signal": signal.signal,
                    "execution_mode": self.config.execution.mode,
                    "risk_outcome": risk_decision.outcome,
                    "risk_reason_codes": list(risk_decision.reason_codes),
                    "ledger_entries": len(ledger_entries),
                    "order_request_created": created_order_request is not None,
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

    async def _persist_execution_result(
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
        for event in result.lifecycle_events:
            await self.repository.insert_order_event_if_absent(event)

        return open_position, closed_position, ledger_entries

    async def _hydrate_due_order_request(
        self,
        *,
        candle: FeatureCandle,
        state,
    ) -> tuple[Any, OrderRequest | None]:
        pending_signal = state.pending_signal
        if pending_signal is None:
            return state, None

        order_request = None
        if pending_signal.order_request_idempotency_key is not None:
            order_request = await self.repository.load_order_request_by_idempotency_key(
                idempotency_key=pending_signal.order_request_idempotency_key,
            )

        if order_request is None:
            seeded_request = build_pending_order_request(
                config=self.config,
                candle=candle,
                pending_signal=pending_signal,
            )
            order_request = await self.repository.ensure_order_request(seeded_request)
            await self.repository.insert_order_event_if_absent(
                build_created_event(
                    order_request=order_request,
                    event_time=pending_signal.signal_as_of_time,
                )
            )

        updated_state = state
        if (
            pending_signal.order_request_id != order_request.order_request_id
            or pending_signal.order_request_idempotency_key != order_request.idempotency_key
        ):
            updated_state = replace(
                state,
                pending_signal=replace(
                    pending_signal,
                    order_request_id=order_request.order_request_id,
                    order_request_idempotency_key=order_request.idempotency_key,
                ),
            )
        return updated_state, order_request

    async def _create_next_order_request(
        self,
        *,
        candle: FeatureCandle,
        signal,
        decision,
    ) -> OrderRequest | None:
        order_request = build_order_request(
            config=self.config,
            candle=candle,
            signal=signal,
            decision=decision,
        )
        if order_request is None:
            return None
        stored_request = await self.repository.ensure_order_request(order_request)
        await self.repository.insert_order_event_if_absent(
            build_created_event(
                order_request=stored_request,
                event_time=candle.as_of_time,
            )
        )
        return stored_request

    async def _write_summaries(self, cash_balance: float) -> None:
        positions = await self.repository.load_positions(
            service_name=self.config.service_name,
            execution_mode=self.config.execution.mode,
        )
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
        _write_csv(artifact_dir / "by_regime_summary.csv", summary["by_regime"])
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
                "execution_mode": position.execution_mode,
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
                "entry_order_request_id": position.entry_order_request_id,
                "entry_regime_label": position.entry_regime_label,
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
                "exit_regime_label": position.exit_regime_label,
                "exit_order_request_id": position.exit_order_request_id,
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
