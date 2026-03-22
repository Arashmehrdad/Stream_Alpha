"""Focused M18 schema and taxonomy tests."""

# pylint: disable=missing-function-docstring

from app.evaluation.schemas import (
    COMPARISON_FAMILIES,
    DECISION_LAYERS,
    DIVERGENCE_REASON_CODES,
    DIVERGENCE_STAGES,
)


def test_m18_comparison_families_are_stable() -> None:
    assert COMPARISON_FAMILIES == (
        "paper_vs_shadow",
        "shadow_vs_tiny_live",
        "paper_to_tiny_live",
    )


def test_m18_layers_and_reason_codes_are_stable() -> None:
    assert DECISION_LAYERS == (
        "model_only",
        "regime_aware",
        "risk_gated",
        "executed",
    )
    assert DIVERGENCE_STAGES == (
        "coverage",
        "decision",
        "risk",
        "order_intent",
        "order_lifecycle",
        "fill_quality",
        "safety_gate",
        "reliability",
    )
    assert DIVERGENCE_REASON_CODES == (
        "CONFIG_MISMATCH",
        "COVERAGE_GAP",
        "MISSING_COUNTERPART",
        "SIGNAL_ACTION_MISMATCH",
        "RISK_OUTCOME_MISMATCH",
        "APPROVED_NOTIONAL_MISMATCH",
        "ORDER_REQUEST_MISSING",
        "ORDER_TERMINAL_STATE_MISMATCH",
        "ORDER_REJECTION_MISMATCH",
        "SAFETY_BLOCK_MISMATCH",
        "STALE_INPUT_BLOCK",
        "FILL_PRICE_DRIFT",
        "SLIPPAGE_DRIFT",
        "LATENCY_DRIFT",
        "FAILURE_INTERRUPTION",
        "RECOVERY_DELAY",
        "DOWNTIME_IMPACT",
    )
