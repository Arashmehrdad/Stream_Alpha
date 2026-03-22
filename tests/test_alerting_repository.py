"""Focused PostgreSQL round-trip tests for the M17 alerting repository."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
import os
from datetime import date, datetime, timezone
from uuid import uuid4

import asyncpg
import pytest

from app.alerting.repository import OperationalAlertRepository
from app.alerting.schemas import OperationalAlertEvent, OperationalAlertState
from app.alerting.service import build_alert_fingerprint


def _postgres_dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if host == "postgres":
        host = "127.0.0.1"
    port = int(os.getenv("POSTGRES_PORT", "5432").strip())
    database = os.getenv("POSTGRES_DB", "streamalpha").strip() or "streamalpha"
    user = os.getenv("POSTGRES_USER", "streamalpha").strip() or "streamalpha"
    password = os.getenv("POSTGRES_PASSWORD", "change-me-local-only").strip()
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def test_operational_alert_repository_round_trip_supports_event_and_state() -> None:
    service_name = f"alerting-test-{uuid4().hex[:10]}"
    event_time = datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)
    fingerprint = build_alert_fingerprint(
        service_name=service_name,
        execution_mode="paper",
        category="FEED_STALE",
        source_component="reliability",
    )
    event = OperationalAlertEvent(
        service_name=service_name,
        execution_mode="paper",
        category="FEED_STALE",
        severity="WARNING",
        event_state="OPEN",
        reason_code="FEED_STALE",
        source_component="reliability",
        fingerprint=fingerprint,
        summary_text="Producer feed freshness is stale.",
        detail="feed_age_seconds=90.0",
        event_time=event_time,
        payload_json={"feed_age_seconds": 90.0},
    )

    asyncio.run(_run_operational_alert_repository_round_trip_test(event))


async def _run_operational_alert_repository_round_trip_test(
    event: OperationalAlertEvent,
) -> None:
    repository = OperationalAlertRepository(_postgres_dsn())
    try:
        await repository.connect()
    except (OSError, asyncpg.CannotConnectNowError) as error:
        pytest.skip(f"PostgreSQL not reachable for alerting repository test: {error}")
        return

    stored_event = None
    try:
        stored_event = await repository.insert_event(event)
        state = OperationalAlertState(
            fingerprint=event.fingerprint,
            service_name=event.service_name,
            execution_mode=event.execution_mode,
            category=event.category,
            symbol=event.symbol,
            source_component=event.source_component,
            is_active=True,
            severity=event.severity,
            reason_code=event.reason_code,
            opened_at=event.event_time,
            last_seen_at=event.event_time,
            last_event_id=stored_event.event_id,
            occurrence_count=1,
        )
        await repository.save_state(state)

        loaded_state = await repository.load_state(fingerprint=event.fingerprint)
        active_states = await repository.load_active_states(
            service_name=event.service_name,
            execution_mode=event.execution_mode,
        )
        daily_events = await repository.load_events_for_day(
            service_name=event.service_name,
            execution_mode=event.execution_mode,
            summary_date=date(2026, 3, 22),
        )

        assert stored_event.event_id is not None
        assert stored_event.detail == "feed_age_seconds=90.0"
        assert loaded_state is not None
        assert loaded_state.is_active is True
        assert loaded_state.last_event_id == stored_event.event_id
        assert len(active_states) == 1
        assert active_states[0].fingerprint == event.fingerprint
        assert len(daily_events) == 1
        assert daily_events[0].reason_code == "FEED_STALE"
        assert daily_events[0].payload_json["feed_age_seconds"] == 90.0
    finally:
        pool = repository._require_pool()  # pylint: disable=protected-access
        await pool.execute(
            """
            DELETE FROM operational_alert_state
            WHERE service_name = $1 AND execution_mode = $2
            """,
            event.service_name,
            event.execution_mode,
        )
        await pool.execute(
            """
            DELETE FROM operational_alert_events
            WHERE service_name = $1 AND execution_mode = $2
            """,
            event.service_name,
            event.execution_mode,
        )
        await repository.close()
