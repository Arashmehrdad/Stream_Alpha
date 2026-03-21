"""Minimal Alpaca Trading API client for Stream Alpha M12."""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse

import httpx

from app.trading.schemas import (
    BrokerAccount,
    BrokerSubmitResult,
    FeatureCandle,
    OrderRequest,
    PaperPosition,
)


class AlpacaClientError(RuntimeError):
    """Base error for guarded Alpaca client failures."""


class AlpacaAuthError(AlpacaClientError):
    """Raised when Alpaca authentication fails."""


class AlpacaNetworkError(AlpacaClientError):
    """Raised when Alpaca requests fail at the network layer."""


class AlpacaResponseError(AlpacaClientError):
    """Raised when Alpaca returns an invalid or unsuccessful response."""


class AlpacaOrderConstraintError(AlpacaClientError):
    """Raised when an order violates an explicit Alpaca order constraint."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


ALPACA_CRYPTO_INTEGER_QTY_REQUIRED = "ALPACA_CRYPTO_INTEGER_QTY_REQUIRED"
ALPACA_CRYPTO_MIN_ORDER_VALUE_REQUIRED = "ALPACA_CRYPTO_MIN_ORDER_VALUE_REQUIRED"
ALPACA_PAPER_CRYPTO_MIN_ORDER_VALUE = 10.0


def infer_alpaca_environment(base_url: str) -> str:
    """Infer the configured Alpaca environment from the root base URL."""
    host = urlparse(base_url).netloc.lower()
    if "paper-api.alpaca.markets" in host:
        return "paper"
    if "api.alpaca.markets" in host:
        return "live"
    raise ValueError(
        "ALPACA_BASE_URL must contain either paper-api.alpaca.markets "
        "or api.alpaca.markets"
    )


def normalize_alpaca_symbol(symbol: str) -> str:
    """Preserve Alpaca's slash-delimited crypto pair format."""
    return symbol.strip().upper()


class AlpacaTradingClient:
    """Small typed Alpaca client used by the guarded live execution adapter."""

    broker_name = "alpaca"

    def __init__(
        self,
        *,
        api_key_id: str,
        api_secret_key: str,
        base_url: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key_id.strip():
            raise ValueError("APCA_API_KEY_ID must not be empty")
        if not api_secret_key.strip():
            raise ValueError("APCA_API_SECRET_KEY must not be empty")
        self.base_url = _normalize_alpaca_base_url(base_url)
        self.api_key_id = api_key_id.strip()
        self.api_secret_key = api_secret_key.strip()
        self._http_client = http_client
        self._owns_client = http_client is None

    @classmethod
    def from_env(
        cls,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> "AlpacaTradingClient":
        """Build the guarded Alpaca client from environment variables only."""
        return cls(
            api_key_id=os.getenv("APCA_API_KEY_ID", ""),
            api_secret_key=os.getenv("APCA_API_SECRET_KEY", ""),
            base_url=os.getenv("ALPACA_BASE_URL", ""),
            http_client=http_client,
        )

    async def close(self) -> None:
        """Close the owned HTTP client."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
        self._http_client = None

    async def validate_account(self) -> BrokerAccount:
        """Validate Alpaca account credentials with GET /v2/account."""
        response = await self._request("GET", "/v2/account")
        payload = _decode_json(response, context="validate account")
        account_id = payload.get("account_number") or payload.get("id")
        if account_id in {None, ""}:
            raise AlpacaResponseError(
                "Alpaca account response did not contain account_number or id"
            )
        status = None if payload.get("status") is None else str(payload["status"])
        return BrokerAccount(
            broker_name=self.broker_name,
            account_id=str(account_id),
            environment_name=infer_alpaca_environment(self.base_url),
            status=status,
        )

    async def submit_order(
        self,
        *,
        order_request: OrderRequest,
        open_position: PaperPosition | None,
        candle: FeatureCandle,
    ) -> BrokerSubmitResult:
        """Submit one minimal market order to Alpaca."""
        environment_name = infer_alpaca_environment(self.base_url)
        payload = {
            "symbol": normalize_alpaca_symbol(order_request.symbol),
            "side": "buy" if order_request.action == "BUY" else "sell",
            "type": "market",
            "time_in_force": "gtc",
        }
        if order_request.action == "BUY":
            if candle.open_price <= 0.0:
                raise AlpacaResponseError("BUY submission requires a positive candle open_price")
            payload["qty"] = _build_buy_quantity(
                approved_notional=order_request.approved_notional,
                reference_price=candle.open_price,
                environment_name=environment_name,
            )
        else:
            if open_position is None:
                raise AlpacaResponseError(
                    "SELL submission requires an open position quantity"
                )
            payload["qty"] = _build_sell_quantity(
                quantity=open_position.quantity,
                reference_price=candle.open_price,
                environment_name=environment_name,
            )

        response = await self._request("POST", "/v2/orders", json_payload=payload)
        response_payload = _decode_json(response, context="submit order")
        external_order_id = response_payload.get("id")
        external_status = response_payload.get("status")
        if external_order_id in {None, ""} or external_status in {None, ""}:
            raise AlpacaResponseError(
                "Alpaca order response did not contain id and status"
            )

        redacted_details = json.dumps(
            {
                "symbol": payload["symbol"],
                "side": payload["side"],
                "type": payload["type"],
                "time_in_force": payload["time_in_force"],
                "qty": payload.get("qty"),
                "approved_notional": _round_decimal(order_request.approved_notional),
                "reference_price": _round_decimal(candle.open_price),
            },
            sort_keys=True,
        )
        return BrokerSubmitResult(
            broker_name=self.broker_name,
            external_order_id=str(external_order_id),
            external_status=str(external_status),
            account_id=str(
                response_payload.get("account_number")
                or response_payload.get("account_id")
                or "<unknown>"
            ),
            environment_name=environment_name,
            details=redacted_details,
        )

    async def _request(
        self,
        method: str,
        endpoint_path: str,
        *,
        json_payload: dict[str, object] | None = None,
    ) -> httpx.Response:
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
            self._http_client = client
            self._owns_client = True

        url = f"{self.base_url}{endpoint_path}"
        try:
            response = await client.request(
                method,
                url,
                headers={
                    "APCA-API-KEY-ID": self.api_key_id,
                    "APCA-API-SECRET-KEY": self.api_secret_key,
                },
                json=json_payload,
            )
        except httpx.HTTPError as error:
            raise AlpacaNetworkError(str(error)) from error

        if response.status_code in {401, 403}:
            raise AlpacaAuthError(
                f"Alpaca authentication failed with HTTP {response.status_code}"
            )
        if response.status_code == 403 and "minimal amount of order 10" in response.text:
            raise AlpacaOrderConstraintError(
                ALPACA_CRYPTO_MIN_ORDER_VALUE_REQUIRED,
                "Alpaca PAPER crypto orders require a minimum order value of 10",
            )
        if response.status_code == 422 and "qty must be integer" in response.text:
            raise AlpacaOrderConstraintError(
                ALPACA_CRYPTO_INTEGER_QTY_REQUIRED,
                "Alpaca PAPER crypto orders require integer quantity",
            )
        if response.status_code == 422 and "qty is required" in response.text:
            raise AlpacaOrderConstraintError(
                ALPACA_CRYPTO_INTEGER_QTY_REQUIRED,
                "Alpaca PAPER crypto orders require explicit quantity",
            )
        if response.status_code < 200 or response.status_code >= 300:
            raise AlpacaResponseError(
                f"Alpaca request failed with HTTP {response.status_code}: "
                f"{response.text.strip()}"
            )
        return response


def _normalize_alpaca_base_url(base_url: str) -> str:
    candidate = base_url.strip().rstrip("/")
    if not candidate:
        raise ValueError("ALPACA_BASE_URL must not be empty")
    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("ALPACA_BASE_URL must include scheme and host")
    if parsed.path not in {"", "/"}:
        raise ValueError(
            "ALPACA_BASE_URL must be the root domain only and must not include /v2"
        )
    infer_alpaca_environment(candidate)
    return candidate


def _decode_json(response: httpx.Response, *, context: str) -> dict[str, object]:
    try:
        payload = response.json()
    except ValueError as error:
        raise AlpacaResponseError(
            f"Alpaca returned malformed JSON for {context}"
        ) from error
    if not isinstance(payload, dict):
        raise AlpacaResponseError(
            f"Alpaca returned a non-object JSON payload for {context}"
        )
    return payload


def _round_decimal(value: float) -> float:
    return round(value, 8)


def _format_decimal(value: float) -> str:
    return f"{_round_decimal(value):.8f}"


def _build_buy_quantity(
    *,
    approved_notional: float,
    reference_price: float,
    environment_name: str,
) -> str:
    implied_quantity = approved_notional / reference_price
    if environment_name != "paper":
        return _format_decimal(implied_quantity)
    integer_quantity = _require_integer_quantity(implied_quantity)
    if integer_quantity * reference_price < ALPACA_PAPER_CRYPTO_MIN_ORDER_VALUE:
        raise AlpacaOrderConstraintError(
            ALPACA_CRYPTO_MIN_ORDER_VALUE_REQUIRED,
            "Alpaca PAPER crypto orders require a minimum order value of 10",
        )
    return str(integer_quantity)


def _build_sell_quantity(
    *,
    quantity: float,
    reference_price: float,
    environment_name: str,
) -> str:
    if environment_name != "paper":
        return _format_decimal(quantity)
    integer_quantity = _require_integer_quantity(quantity)
    if integer_quantity * reference_price < ALPACA_PAPER_CRYPTO_MIN_ORDER_VALUE:
        raise AlpacaOrderConstraintError(
            ALPACA_CRYPTO_MIN_ORDER_VALUE_REQUIRED,
            "Alpaca PAPER crypto orders require a minimum order value of 10",
        )
    return str(integer_quantity)


def _require_integer_quantity(value: float) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-9 or rounded <= 0:
        raise AlpacaOrderConstraintError(
            ALPACA_CRYPTO_INTEGER_QTY_REQUIRED,
            "Alpaca PAPER crypto orders require integer quantity",
        )
    return int(rounded)
