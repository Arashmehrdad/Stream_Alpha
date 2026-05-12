"""Tests for the canonical M9 RegimeContext contract."""

from __future__ import annotations

from types import SimpleNamespace

from app.regime.context import (
    REGIME_CONTEXT_RUNTIME_HOLD,
    context_from_resolved_regime,
    missing_regime_context,
)


def test_context_from_resolved_regime_is_usable_when_fresh() -> None:
    """Fresh exact-row resolution should produce a usable context."""
    resolved = SimpleNamespace(
        regime_label="TREND_UP",
        regime_run_id="20260320T165813Z",
        row_id="BTC/USD|2026-05-01T00:00:00Z",
        interval_begin="2026-05-01T00:00:00Z",
        as_of_time="2026-05-01T00:05:00Z",
        regime_artifact_path="artifacts/regime/m8/run/thresholds.json",
    )

    context = context_from_resolved_regime(resolved)

    assert context.usable is True
    assert context.regime_label == "TREND_UP"
    assert context.fallback_behavior == "ALLOW_REGIME_POLICY"
    assert context.to_dict()["m20_research_authority"] is False
    assert context.to_dict()["runtime_effect"] == "NO_RUNTIME_EFFECT"


def test_context_from_resolved_regime_fails_closed_when_stale() -> None:
    """Stale regime context should be deterministic and fail closed."""
    resolved = SimpleNamespace(
        regime_label="RANGE",
        regime_run_id="20260320T165813Z",
        row_id="ETH/USD|2026-05-01T00:00:00Z",
        interval_begin="2026-05-01T00:00:00Z",
        as_of_time="2026-05-01T00:05:00Z",
        regime_artifact_path="artifacts/regime/m8/run/thresholds.json",
    )

    context = context_from_resolved_regime(
        resolved,
        freshness_status="STALE",
        health_overall_status="DEGRADED",
        reason_code="REGIME_STALE",
    )

    assert context.usable is False
    assert context.fallback_behavior == REGIME_CONTEXT_RUNTIME_HOLD
    assert context.reason_code == "REGIME_STALE"


def test_missing_regime_context_has_no_policy_authority() -> None:
    """Missing regime truth should never create a usable runtime context."""
    context = missing_regime_context(
        row_id="SOL/USD|MISSING",
        interval_begin="2026-05-01T00:00:00Z",
    )

    payload = context.to_dict()
    assert context.usable is False
    assert payload["regime_label"] is None
    assert payload["fallback_behavior"] == REGIME_CONTEXT_RUNTIME_HOLD
    assert payload["m20_research_authority"] is False
