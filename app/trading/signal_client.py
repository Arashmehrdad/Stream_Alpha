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
from app.trading.schemas import (
    CanonicalFeatureLag,
    CanonicalRecoveryEvent,
    CanonicalServiceHealth,
    CanonicalSystemReliability,
)


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

    async def fetch_system_reliability(self) -> CanonicalSystemReliability:
        """Fetch the canonical M13 cross-service reliability summary."""
        response = await self._client.get("/reliability/system")
        response.raise_for_status()
        payload = response.json()
        return CanonicalSystemReliability(
            service_name=str(payload["service_name"]),
            checked_at=parse_rfc3339(str(payload["checked_at"])),
            health_overall_status=str(payload["health_overall_status"]),
            reason_codes=tuple(str(code) for code in payload.get("reason_codes", [])),
            lag_breach_active=bool(payload.get("lag_breach_active", False)),
            services=tuple(
                CanonicalServiceHealth(
                    service_name=str(item["service_name"]),
                    component_name=str(item["component_name"]),
                    checked_at=parse_rfc3339(str(item["checked_at"])),
                    heartbeat_at=(
                        None
                        if item.get("heartbeat_at") is None
                        else parse_rfc3339(str(item["heartbeat_at"]))
                    ),
                    heartbeat_age_seconds=(
                        None
                        if item.get("heartbeat_age_seconds") is None
                        else float(item["heartbeat_age_seconds"])
                    ),
                    heartbeat_freshness_status=str(item["heartbeat_freshness_status"]),
                    health_overall_status=str(item["health_overall_status"]),
                    reason_code=str(item["reason_code"]),
                    detail=(
                        None if item.get("detail") is None else str(item["detail"])
                    ),
                    feed_freshness_status=(
                        None
                        if item.get("feed_freshness_status") is None
                        else str(item["feed_freshness_status"])
                    ),
                    feed_reason_code=(
                        None
                        if item.get("feed_reason_code") is None
                        else str(item["feed_reason_code"])
                    ),
                    feed_age_seconds=(
                        None
                        if item.get("feed_age_seconds") is None
                        else float(item["feed_age_seconds"])
                    ),
                )
                for item in payload.get("services", [])
            ),
            lag_by_symbol=tuple(
                CanonicalFeatureLag(
                    service_name=str(item["service_name"]),
                    component_name=str(item["component_name"]),
                    symbol=str(item["symbol"]),
                    evaluated_at=parse_rfc3339(str(item["evaluated_at"])),
                    latest_raw_event_received_at=(
                        None
                        if item.get("latest_raw_event_received_at") is None
                        else parse_rfc3339(str(item["latest_raw_event_received_at"]))
                    ),
                    latest_feature_interval_begin=(
                        None
                        if item.get("latest_feature_interval_begin") is None
                        else parse_rfc3339(str(item["latest_feature_interval_begin"]))
                    ),
                    latest_feature_as_of_time=(
                        None
                        if item.get("latest_feature_as_of_time") is None
                        else parse_rfc3339(str(item["latest_feature_as_of_time"]))
                    ),
                    time_lag_seconds=(
                        None
                        if item.get("time_lag_seconds") is None
                        else float(item["time_lag_seconds"])
                    ),
                    processing_lag_seconds=(
                        None
                        if item.get("processing_lag_seconds") is None
                        else float(item["processing_lag_seconds"])
                    ),
                    time_lag_reason_code=str(item["time_lag_reason_code"]),
                    processing_lag_reason_code=str(item["processing_lag_reason_code"]),
                    lag_breach=bool(item["lag_breach"]),
                    health_overall_status=str(item["health_overall_status"]),
                    reason_code=str(item["reason_code"]),
                    detail=(
                        None if item.get("detail") is None else str(item["detail"])
                    ),
                )
                for item in payload.get("lag_by_symbol", [])
            ),
            latest_recovery_event=(
                None
                if payload.get("latest_recovery_event") is None
                else CanonicalRecoveryEvent(
                    service_name=str(payload["latest_recovery_event"]["service_name"]),
                    component_name=str(
                        payload["latest_recovery_event"]["component_name"]
                    ),
                    event_type=str(payload["latest_recovery_event"]["event_type"]),
                    event_time=parse_rfc3339(
                        str(payload["latest_recovery_event"]["event_time"])
                    ),
                    reason_code=str(payload["latest_recovery_event"]["reason_code"]),
                    health_overall_status=(
                        None
                        if payload["latest_recovery_event"].get(
                            "health_overall_status"
                        )
                        is None
                        else str(
                            payload["latest_recovery_event"]["health_overall_status"]
                        )
                    ),
                    freshness_status=(
                        None
                        if payload["latest_recovery_event"].get("freshness_status")
                        is None
                        else str(payload["latest_recovery_event"]["freshness_status"])
                    ),
                    breaker_state=(
                        None
                        if payload["latest_recovery_event"].get("breaker_state")
                        is None
                        else str(payload["latest_recovery_event"]["breaker_state"])
                    ),
                    detail=(
                        None
                        if payload["latest_recovery_event"].get("detail") is None
                        else str(payload["latest_recovery_event"]["detail"])
                    ),
                )
            ),
        )
