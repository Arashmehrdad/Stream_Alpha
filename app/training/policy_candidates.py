"""Named research-only long-only policy candidates for completed M7 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


LOW_TRADE_COUNT_CAUTION_THRESHOLD = 20


@dataclass(frozen=True, slots=True)
class LongOnlyPolicyCandidate:
    """Explicit research-only long-only policy candidate for completed-run evaluation."""

    name: str
    description: str
    prob_up_min: float
    blocked_regimes: frozenset[str] = frozenset()
    allowed_regimes: frozenset[str] | None = None
    per_regime_thresholds: Mapping[str, float] | None = None

    def threshold_for_regime(self, regime_label: str) -> float | None:
        """Return the effective prob_up threshold, or None when longs are blocked."""
        if self.allowed_regimes is not None and regime_label not in self.allowed_regimes:
            return None
        if regime_label in self.blocked_regimes:
            return None
        if self.per_regime_thresholds is not None and regime_label in self.per_regime_thresholds:
            return float(self.per_regime_thresholds[regime_label])
        return float(self.prob_up_min)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe structured policy definition."""
        return {
            "name": self.name,
            "description": self.description,
            "prob_up_min": float(self.prob_up_min),
            "blocked_regimes": sorted(self.blocked_regimes),
            "allowed_regimes": (
                sorted(self.allowed_regimes)
                if self.allowed_regimes is not None
                else None
            ),
            "per_regime_thresholds": (
                {
                    regime_label: float(threshold)
                    for regime_label, threshold in sorted(self.per_regime_thresholds.items())
                }
                if self.per_regime_thresholds is not None
                else None
            ),
        }


def build_default_policy_candidates() -> tuple[LongOnlyPolicyCandidate, ...]:
    """Return the bounded research-only candidate set for completed M7 runs."""
    return (
        LongOnlyPolicyCandidate(
            name="default_long_only_050",
            description="Default long-only policy using prob_up >= 0.50 with no regime blocks",
            prob_up_min=0.50,
        ),
        LongOnlyPolicyCandidate(
            name="m7_research_long_only_v1",
            description=(
                "Research-only challenger: prob_up >= 0.80 with TREND_DOWN and HIGH_VOL blocked"
            ),
            prob_up_min=0.80,
            blocked_regimes=frozenset({"TREND_DOWN", "HIGH_VOL"}),
        ),
    )


def find_policy_candidate(
    candidate_name: str,
    *,
    candidates: tuple[LongOnlyPolicyCandidate, ...] | None = None,
) -> LongOnlyPolicyCandidate:
    """Resolve one named candidate from the default bounded candidate set."""
    resolved_candidates = candidates or build_default_policy_candidates()
    for candidate in resolved_candidates:
        if candidate.name == candidate_name:
            return candidate
    raise ValueError(f"Unknown M7 research policy candidate: {candidate_name}")


def low_trade_count_caution(
    trade_count: int,
    *,
    threshold: int = LOW_TRADE_COUNT_CAUTION_THRESHOLD,
) -> str | None:
    """Return a brief caution when the policy is too sparse to count as robust evidence."""
    if int(trade_count) < threshold:
        return "Positive but too sparse to count as robust promotion evidence."
    return None
