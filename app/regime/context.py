"""Canonical M9 regime context contract."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REGIME_CONTEXT_SCHEMA_VERSION = "m9_regime_context_v1"
REGIME_SOURCE_M8_THRESHOLDS = "M8_FIXED_THRESHOLDS"
REGIME_CONTEXT_FRESH = "FRESH"
REGIME_CONTEXT_STALE = "STALE"
REGIME_CONTEXT_UNKNOWN = "UNKNOWN"
REGIME_CONTEXT_MISSING = "REGIME_CONTEXT_MISSING"
REGIME_CONTEXT_RUNTIME_HOLD = "FAIL_CLOSED_HOLD"


@dataclass(frozen=True, slots=True)
class RegimeContext:
    """Canonical regime context shared by M9 read surfaces and downstream gates."""

    regime_label: str | None
    regime_run_id: str | None
    row_id: str | None
    interval_begin: str | None
    as_of_time: str | None
    source: str
    source_version: str
    artifact_path: str | None
    freshness_status: str
    health_overall_status: str
    reason_code: str
    fallback_behavior: str
    runtime_effect: str = "NO_RUNTIME_EFFECT"
    m20_research_authority: bool = False

    @property
    def usable(self) -> bool:
        """Return whether the context can support normal regime-aware routing."""
        return (
            self.regime_label is not None
            and self.freshness_status == REGIME_CONTEXT_FRESH
            and self.health_overall_status == "HEALTHY"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe contract payload."""
        return {
            "schema_version": REGIME_CONTEXT_SCHEMA_VERSION,
            "regime_label": self.regime_label,
            "regime_run_id": self.regime_run_id,
            "row_id": self.row_id,
            "interval_begin": self.interval_begin,
            "as_of_time": self.as_of_time,
            "source": self.source,
            "source_version": self.source_version,
            "artifact_path": self.artifact_path,
            "freshness_status": self.freshness_status,
            "health_overall_status": self.health_overall_status,
            "reason_code": self.reason_code,
            "fallback_behavior": self.fallback_behavior,
            "runtime_effect": self.runtime_effect,
            "m20_research_authority": self.m20_research_authority,
            "usable": self.usable,
        }


def context_from_resolved_regime(
    resolved_regime: Any,
    *,
    freshness_status: str = REGIME_CONTEXT_FRESH,
    health_overall_status: str = "HEALTHY",
    reason_code: str = "REGIME_FRESH",
) -> RegimeContext:
    """Build canonical context from the existing exact-row runtime resolution."""
    return RegimeContext(
        regime_label=str(resolved_regime.regime_label),
        regime_run_id=str(resolved_regime.regime_run_id),
        row_id=str(resolved_regime.row_id),
        interval_begin=str(resolved_regime.interval_begin),
        as_of_time=str(resolved_regime.as_of_time),
        source=REGIME_SOURCE_M8_THRESHOLDS,
        source_version=REGIME_CONTEXT_SCHEMA_VERSION,
        artifact_path=str(resolved_regime.regime_artifact_path),
        freshness_status=freshness_status,
        health_overall_status=health_overall_status,
        reason_code=reason_code,
        fallback_behavior=(
            "ALLOW_REGIME_POLICY"
            if freshness_status == REGIME_CONTEXT_FRESH
            and health_overall_status == "HEALTHY"
            else REGIME_CONTEXT_RUNTIME_HOLD
        ),
    )


def missing_regime_context(
    *,
    row_id: str | None = None,
    interval_begin: str | None = None,
    reason_code: str = REGIME_CONTEXT_MISSING,
    health_overall_status: str = "DEGRADED",
) -> RegimeContext:
    """Build deterministic fail-closed context when regime truth is unavailable."""
    return RegimeContext(
        regime_label=None,
        regime_run_id=None,
        row_id=row_id,
        interval_begin=interval_begin,
        as_of_time=None,
        source=REGIME_SOURCE_M8_THRESHOLDS,
        source_version=REGIME_CONTEXT_SCHEMA_VERSION,
        artifact_path=None,
        freshness_status=REGIME_CONTEXT_UNKNOWN,
        health_overall_status=health_overall_status,
        reason_code=reason_code,
        fallback_behavior=REGIME_CONTEXT_RUNTIME_HOLD,
    )
