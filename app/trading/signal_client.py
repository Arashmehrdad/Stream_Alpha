"""Thin M4 signal client for the Stream Alpha M5 paper trader."""

from __future__ import annotations

from datetime import datetime

import httpx

from app.common.time import parse_rfc3339, to_rfc3339
from app.explainability.schemas import (
    PredictionExplanation,
    RegimeReason,
    SignalExplanation,
    ThresholdSnapshot,
    TopFeatureContribution,
)
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
            model_version=(
                None
                if payload.get("model_version") is None
                else str(payload["model_version"])
            ),
            regime_label=(
                None
                if payload.get("regime_label") is None
                else str(payload["regime_label"])
            ),
            regime_run_id=(
                None
                if payload.get("regime_run_id") is None
                else str(payload["regime_run_id"])
            ),
            trade_allowed=(
                None
                if payload.get("trade_allowed") is None
                else bool(payload["trade_allowed"])
            ),
            signal_status=(
                None
                if payload.get("signal_status") is None
                else str(payload["signal_status"])
            ),
            decision_source=(
                None
                if payload.get("decision_source") is None
                else str(payload["decision_source"])
            ),
            reason_code=(
                None
                if payload.get("reason_code") is None
                else str(payload["reason_code"])
            ),
            freshness_status=(
                None
                if payload.get("freshness_status") is None
                else str(payload["freshness_status"])
            ),
            health_overall_status=(
                None
                if payload.get("health_overall_status") is None
                else str(payload["health_overall_status"])
            ),
            top_features=tuple(
                TopFeatureContribution.model_validate(item)
                for item in payload.get("top_features", [])
            ),
            prediction_explanation=(
                None
                if payload.get("prediction_explanation") is None
                else PredictionExplanation.model_validate(payload["prediction_explanation"])
            ),
            threshold_snapshot=(
                None
                if payload.get("threshold_snapshot") is None
                else ThresholdSnapshot.model_validate(payload["threshold_snapshot"])
            ),
            regime_reason=(
                None
                if payload.get("regime_reason") is None
                else RegimeReason.model_validate(payload["regime_reason"])
            ),
            signal_explanation=(
                None
                if payload.get("signal_explanation") is None
                else SignalExplanation.model_validate(payload["signal_explanation"])
            ),
        )
