"""Focused Alpaca client tests for the M12 guarded live foundation."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from app.trading.alpaca import (
    ALPACA_CRYPTO_INTEGER_QTY_REQUIRED,
    ALPACA_CRYPTO_MIN_ORDER_VALUE_REQUIRED,
    AlpacaOrderConstraintError,
    AlpacaTradingClient,
)
from app.trading.schemas import FeatureCandle, OrderRequest


def test_alpaca_client_builds_account_url_from_root_base_url() -> None:
    captured_requests: list[httpx.Request] = []

    async def _exercise() -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "acct-id",
                    "account_number": "PA12345",
                    "status": "ACTIVE",
                },
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        client = AlpacaTradingClient(
            api_key_id="test-key",
            api_secret_key="test-secret",
            base_url="https://paper-api.alpaca.markets",
            http_client=http_client,
        )
        await client.validate_account()
        await http_client.aclose()

    asyncio.run(_exercise())

    assert len(captured_requests) == 1
    assert str(captured_requests[0].url) == "https://paper-api.alpaca.markets/v2/account"


def test_alpaca_client_uses_required_apca_headers() -> None:
    captured_requests: list[httpx.Request] = []

    async def _exercise() -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "acct-id",
                    "account_number": "PA12345",
                    "status": "ACTIVE",
                },
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        client = AlpacaTradingClient(
            api_key_id="test-key",
            api_secret_key="test-secret",
            base_url="https://paper-api.alpaca.markets",
            http_client=http_client,
        )
        await client.validate_account()
        await http_client.aclose()

    asyncio.run(_exercise())

    assert len(captured_requests) == 1
    assert captured_requests[0].headers["APCA-API-KEY-ID"] == "test-key"
    assert captured_requests[0].headers["APCA-API-SECRET-KEY"] == "test-secret"


def test_alpaca_paper_crypto_buy_rejects_fractional_quantity_before_submit() -> None:
    captured_requests: list[httpx.Request] = []

    async def _exercise() -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "order-id",
                    "status": "accepted",
                    "account_id": "acct-id",
                },
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        client = AlpacaTradingClient(
            api_key_id="test-key",
            api_secret_key="test-secret",
            base_url="https://paper-api.alpaca.markets",
            http_client=http_client,
        )
        with pytest.raises(AlpacaOrderConstraintError) as error:
            await client.submit_order(
                order_request=OrderRequest(
                    service_name="svc",
                    execution_mode="live",
                    symbol="BTC/USD",
                    action="BUY",
                    signal_interval_begin=datetime(2026, 3, 21, 15, 15, tzinfo=timezone.utc),
                    signal_as_of_time=datetime(2026, 3, 21, 15, 15, tzinfo=timezone.utc),
                    signal_row_id="BTC/USD|2026-03-21T15:15:00Z",
                    target_fill_interval_begin=datetime(
                        2026,
                        3,
                        21,
                        15,
                        20,
                        tzinfo=timezone.utc,
                    ),
                    requested_notional=20.0,
                    approved_notional=19.96007984,
                    idempotency_key="idempotency-key",
                ),
                open_position=None,
                candle=FeatureCandle(
                    id=1,
                    source_exchange="kraken",
                    symbol="BTC/USD",
                    interval_minutes=5,
                    interval_begin=datetime(2026, 3, 21, 15, 20, tzinfo=timezone.utc),
                    interval_end=datetime(2026, 3, 21, 15, 25, tzinfo=timezone.utc),
                    as_of_time=datetime(2026, 3, 21, 15, 20, tzinfo=timezone.utc),
                    raw_event_id="raw-1",
                    open_price=2000.0,
                    high_price=2001.0,
                    low_price=1999.0,
                    close_price=2000.5,
                    realized_vol_12=0.01,
                ),
            )
        await http_client.aclose()
        assert error.value.reason_code == ALPACA_CRYPTO_INTEGER_QTY_REQUIRED

    asyncio.run(_exercise())

    assert not captured_requests


def test_alpaca_paper_crypto_buy_uses_integer_qty_when_constraints_are_met() -> None:
    captured_requests: list[httpx.Request] = []

    async def _exercise() -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "order-id",
                    "status": "accepted",
                    "account_id": "acct-id",
                },
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        client = AlpacaTradingClient(
            api_key_id="test-key",
            api_secret_key="test-secret",
            base_url="https://paper-api.alpaca.markets",
            http_client=http_client,
        )
        await client.submit_order(
            order_request=OrderRequest(
                service_name="svc",
                execution_mode="live",
                symbol="DOGE/USD",
                action="BUY",
                signal_interval_begin=datetime(2026, 3, 21, 15, 15, tzinfo=timezone.utc),
                signal_as_of_time=datetime(2026, 3, 21, 15, 15, tzinfo=timezone.utc),
                signal_row_id="DOGE/USD|2026-03-21T15:15:00Z",
                target_fill_interval_begin=datetime(
                    2026,
                    3,
                    21,
                    15,
                    20,
                    tzinfo=timezone.utc,
                ),
                requested_notional=10.0,
                approved_notional=10.0,
                idempotency_key="idempotency-key",
            ),
            open_position=None,
            candle=FeatureCandle(
                id=1,
                source_exchange="kraken",
                symbol="DOGE/USD",
                interval_minutes=5,
                interval_begin=datetime(2026, 3, 21, 15, 20, tzinfo=timezone.utc),
                interval_end=datetime(2026, 3, 21, 15, 25, tzinfo=timezone.utc),
                as_of_time=datetime(2026, 3, 21, 15, 20, tzinfo=timezone.utc),
                raw_event_id="raw-1",
                open_price=0.02,
                high_price=0.021,
                low_price=0.019,
                close_price=0.0205,
                realized_vol_12=0.01,
            ),
        )
        await http_client.aclose()

    asyncio.run(_exercise())

    assert len(captured_requests) == 1
    assert str(captured_requests[0].url) == "https://paper-api.alpaca.markets/v2/orders"
    assert captured_requests[0].read().decode("utf-8") == (
        '{"symbol":"DOGE/USD","side":"buy","type":"market",'
        '"time_in_force":"gtc","qty":"500"}'
    )


def test_alpaca_paper_crypto_buy_rejects_sub_minimum_order_value_before_submit() -> None:
    async def _exercise() -> None:
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"id": "unused", "status": "accepted"})
            )
        )
        client = AlpacaTradingClient(
            api_key_id="test-key",
            api_secret_key="test-secret",
            base_url="https://paper-api.alpaca.markets",
            http_client=http_client,
        )
        with pytest.raises(AlpacaOrderConstraintError) as error:
            await client.submit_order(
                order_request=OrderRequest(
                    service_name="svc",
                    execution_mode="live",
                    symbol="DOGE/USD",
                    action="BUY",
                    signal_interval_begin=datetime(2026, 3, 21, 15, 15, tzinfo=timezone.utc),
                    signal_as_of_time=datetime(2026, 3, 21, 15, 15, tzinfo=timezone.utc),
                    signal_row_id="DOGE/USD|2026-03-21T15:15:00Z",
                    target_fill_interval_begin=datetime(
                        2026,
                        3,
                        21,
                        15,
                        20,
                        tzinfo=timezone.utc,
                    ),
                    requested_notional=5.0,
                    approved_notional=5.0,
                    idempotency_key="idempotency-key",
                ),
                open_position=None,
                candle=FeatureCandle(
                    id=1,
                    source_exchange="kraken",
                    symbol="DOGE/USD",
                    interval_minutes=5,
                    interval_begin=datetime(2026, 3, 21, 15, 20, tzinfo=timezone.utc),
                    interval_end=datetime(2026, 3, 21, 15, 25, tzinfo=timezone.utc),
                    as_of_time=datetime(2026, 3, 21, 15, 20, tzinfo=timezone.utc),
                    raw_event_id="raw-1",
                    open_price=0.02,
                    high_price=0.021,
                    low_price=0.019,
                    close_price=0.0205,
                    realized_vol_12=0.01,
                ),
            )
        await http_client.aclose()
        assert error.value.reason_code == ALPACA_CRYPTO_MIN_ORDER_VALUE_REQUIRED

    asyncio.run(_exercise())
