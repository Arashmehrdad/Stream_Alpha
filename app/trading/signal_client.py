"""Thin M4 signal client for the Stream Alpha M5 paper trader."""

from __future__ import annotations

from datetime import datetime

import httpx

from app.common.time import parse_rfc3339, to_rfc3339
from app.trading.schemas import SignalDecision


class SignalClientError(RuntimeError):
    """Raised when the paper trader cannot safely use the M4 signal response."""


class SignalClient:
    """Small HTTP client that treats the M4 `/signal` endpoint as authoritative."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def fetch_signal(self, *, symbol: str, interval_begin: datetime) -> SignalDecision:
        """Fetch the authoritative signal for one exact finalized candle."""
        response = await self._client.get(
            "/signal",
            params={
                "symbol": symbol,
                "interval_begin": to_rfc3339(interval_begin),
            },
        )
        response.raise_for_status()
        payload = response.json()
        row_id = str(payload["row_id"])
        expected_row_id = f"{symbol}|{to_rfc3339(interval_begin)}"
        if row_id != expected_row_id:
            raise SignalClientError(
                f"M4 returned row_id {row_id} but expected {expected_row_id}",
            )
        return SignalDecision(
            symbol=str(payload["symbol"]),
            signal=str(payload["signal"]),
            reason=str(payload["reason"]),
            prob_up=float(payload["prob_up"]),
            prob_down=float(payload["prob_down"]),
            confidence=float(payload["confidence"]),
            predicted_class=str(payload["predicted_class"]),
            row_id=row_id,
            as_of_time=parse_rfc3339(str(payload["as_of_time"])),
            model_name=str(payload["model_name"]),
        )
