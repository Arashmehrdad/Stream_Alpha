"""Typed M18 evaluation schemas and taxonomy."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


ComparisonFamily = Literal[
    "paper_vs_shadow",
    "shadow_vs_tiny_live",
    "paper_to_tiny_live",
]
DecisionLayer = Literal["model_only", "regime_aware", "risk_gated", "executed"]
DivergenceStage = Literal[
    "coverage",
    "decision",
    "risk",
    "order_intent",
    "order_lifecycle",
    "fill_quality",
    "safety_gate",
    "reliability",
]
DivergenceReasonCode = Literal[
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
]
EvaluationTruthStatus = Literal["OBSERVED", "SIMULATED", "NOT_APPLICABLE", "MISSING"]


COMPARISON_FAMILIES: tuple[ComparisonFamily, ...] = (
    "paper_vs_shadow",
    "shadow_vs_tiny_live",
    "paper_to_tiny_live",
)
COMPARISON_FAMILY_MODE_PAIRS: dict[ComparisonFamily, tuple[str, str]] = {
    "paper_vs_shadow": ("paper", "shadow"),
    "shadow_vs_tiny_live": ("shadow", "live"),
    "paper_to_tiny_live": ("paper", "live"),
}
DECISION_LAYERS: tuple[DecisionLayer, ...] = (
    "model_only",
    "regime_aware",
    "risk_gated",
    "executed",
)
DIVERGENCE_STAGES: tuple[DivergenceStage, ...] = (
    "coverage",
    "decision",
    "risk",
    "order_intent",
    "order_lifecycle",
    "fill_quality",
    "safety_gate",
    "reliability",
)
DIVERGENCE_REASON_CODES: tuple[DivergenceReasonCode, ...] = (
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

EVALUATION_MANIFEST_SCHEMA_VERSION = "m18_evaluation_manifest_v1"
EVALUATION_REPORT_SCHEMA_VERSION = "m18_evaluation_report_v1"
UPTIME_FAILURES_SCHEMA_VERSION = "m18_uptime_failures_v1"
LAYER_COMPARISON_SCHEMA_VERSION = "m18_layer_comparison_v1"
PAPER_TO_LIVE_DEGRADATION_SCHEMA_VERSION = "m18_paper_to_live_degradation_v1"
EVALUATION_INDEX_SCHEMA_VERSION = "m18_evaluation_index_v1"
EXPERIMENT_INDEX_SCHEMA_VERSION = "m18_experiment_index_v1"
PROMOTION_INDEX_SCHEMA_VERSION = "m18_promotion_index_v1"


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """One explicit M18 evaluation run request."""

    service_name: str
    source_exchange: str
    interval_minutes: int
    symbols: tuple[str, ...]
    execution_modes: tuple[str, ...]
    comparison_families: tuple[ComparisonFamily, ...]
    window_start: datetime
    window_end: datetime
    trading_config_path: str
    evaluation_run_id: str
    generated_at: datetime


@dataclass(frozen=True, slots=True)
class OrderLifecycleSummary:
    """Normalized order-intent and lifecycle summary for one decision trace."""

    truth_status: EvaluationTruthStatus
    order_request_id: int | None = None
    created_at: datetime | None = None
    first_response_at: datetime | None = None
    terminal_at: datetime | None = None
    terminal_state: str | None = None
    terminal_reason_code: str | None = None
    lifecycle_states: tuple[str, ...] = field(default_factory=tuple)
    broker_name: str | None = None
    account_id: str | None = None
    environment_name: str | None = None


@dataclass(frozen=True, slots=True)
class FillSummary:
    """Normalized fill and slippage truth for one decision trace."""

    truth_status: EvaluationTruthStatus
    action: str | None = None
    fill_time: datetime | None = None
    fill_price: float | None = None
    notional: float | None = None
    fee: float | None = None
    slippage_bps: float | None = None


@dataclass(frozen=True, slots=True)
class PositionOutcomeSummary:
    """Normalized position outcome linked to one decision trace."""

    position_id: int | None = None
    position_status: str | None = None
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    realized_pnl: float | None = None
    realized_return: float | None = None
    entry_decision_trace_id: int | None = None
    exit_decision_trace_id: int | None = None


@dataclass(frozen=True, slots=True)
class DecisionOpportunity:
    """Canonical M18 decision-opportunity row built from persisted truth."""

    service_name: str
    execution_mode: str
    symbol: str
    signal_row_id: str
    decision_trace_id: int
    signal_interval_begin: datetime
    signal_as_of_time: datetime
    model_name: str
    model_version: str
    regime_label: str | None
    regime_run_id: str | None
    signal_action: str
    decision_source: str | None
    signal_reason_code: str | None
    freshness_status: str | None
    health_overall_status: str | None
    buy_prob_up: float | None
    sell_prob_up: float | None
    allow_new_long_entries: bool | None
    model_only_action: str
    regime_aware_action: str
    risk_gated_action: str
    executed_action: str
    risk_outcome: str | None
    risk_primary_reason_code: str | None
    requested_notional: float | None
    approved_notional: float | None
    risk_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    safety_blocked: bool = False
    reliability_blocked: bool = False
    order: OrderLifecycleSummary = field(
        default_factory=lambda: OrderLifecycleSummary(truth_status="MISSING")
    )
    fill: FillSummary = field(default_factory=lambda: FillSummary(truth_status="MISSING"))
    position: PositionOutcomeSummary = field(default_factory=PositionOutcomeSummary)


@dataclass(frozen=True, slots=True)
class ComparisonWindow:
    """Overlap window used for one comparison family."""

    comparison_family: ComparisonFamily
    left_mode: str
    right_mode: str
    overlap_start: datetime | None
    overlap_end: datetime | None
    left_count: int
    right_count: int
    matched_count: int


@dataclass(frozen=True, slots=True)
class DivergenceEvent:
    """Canonical M18 divergence event between two comparison modes."""

    comparison_family: ComparisonFamily
    divergence_stage: DivergenceStage
    reason_code: DivergenceReasonCode
    left_mode: str
    right_mode: str
    event_time: datetime
    summary_text: str
    detail: str | None = None
    symbol: str | None = None
    signal_row_id: str | None = None
    left_decision_trace_id: int | None = None
    right_decision_trace_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PerformanceBreakdownRow:
    """Performance summary row grouped by asset or regime."""

    execution_mode: str
    breakdown_type: Literal["asset", "regime"]
    key: str
    position_count: int
    closed_position_count: int
    open_position_count: int
    realized_pnl: float
    average_realized_return: float | None
    win_rate: float | None
    comparable_buy_count: int
    positive_after_cost_buy_count: int
    total_entry_notional: float
    total_fees: float
    total_slippage_bps: float | None
    cost_aware_precision: float | None


@dataclass(frozen=True, slots=True)
class DistributionRow:
    """One distribution summary row for latency or slippage."""

    execution_mode: str
    metric_name: str
    truth_status: EvaluationTruthStatus
    count: int
    min_value: float | None
    mean_value: float | None
    p50_value: float | None
    p90_value: float | None
    p95_value: float | None
    max_value: float | None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class UptimeComponentSummary:
    """Heartbeat-derived uptime summary for one runtime component."""

    component_name: str
    total_heartbeats: int
    healthy_heartbeats: int
    degraded_heartbeats: int
    unavailable_heartbeats: int
    healthy_ratio: float | None


@dataclass(frozen=True, slots=True)
class UptimeAndFailuresReport:
    """Canonical uptime, failure, and recovery summary."""

    schema_version: str
    service_name: str
    window_start: str
    window_end: str
    heartbeat_components: list[UptimeComponentSummary]
    reliability_events_total: int
    reliability_events_by_component: dict[str, int]
    reliability_events_by_reason_code: dict[str, int]
    reliability_events_by_type: dict[str, int]
    recovery_event_count: int
    order_failure_counts_by_mode: dict[str, dict[str, int]]
    divergence_failure_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class LayerTransitionMetrics:
    """One mode's layer-transition counts."""

    action_counts: dict[str, dict[str, int]]
    transition_changes: dict[str, int]
    blocked_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class LayerComparisonReport:
    """Canonical layer-comparison summary for M18."""

    schema_version: str
    modes: dict[str, LayerTransitionMetrics]


@dataclass(frozen=True, slots=True)
class CostAwarePrecisionSummary:
    """Comparable BUY counts and precision for one mode or grouping."""

    comparable_buy_count: int
    positive_after_cost_buy_count: int
    cost_aware_precision: float | None


@dataclass(frozen=True, slots=True)
class DegradationFamilySummary:
    """Canonical degradation summary for one comparison family."""

    comparison_family: ComparisonFamily
    left_mode: str
    right_mode: str
    coverage_counts: dict[str, int]
    matched_count: int
    comparable_counts: dict[str, int]
    divergence_counts_by_stage: dict[str, int]
    divergence_counts_by_reason_code: dict[str, int]
    blockers: list[str]
    notes: list[str]


@dataclass(frozen=True, slots=True)
class PaperToLiveDegradationReport:
    """Canonical degradation summary across the M18 comparison families."""

    schema_version: str
    families: dict[str, DegradationFamilySummary]


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    """Canonical M18 evaluation manifest."""

    schema_version: str
    evaluation_run_id: str
    generated_at: str
    service_name: str
    source_exchange: str
    interval_minutes: int
    symbols: list[str]
    execution_modes_requested: list[str]
    execution_modes_available: list[str]
    comparison_families: list[str]
    window_start: str
    window_end: str
    trading_config_path: str
    current_registry_entry: dict[str, Any] | None
    known_limitations: list[str]
    artifact_paths: dict[str, str]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Canonical M18 evaluation report summary."""

    schema_version: str
    evaluation_run_id: str
    generated_at: str
    service_name: str
    window_start: str
    window_end: str
    opportunity_counts_by_mode: dict[str, int]
    matched_counts_by_family: dict[str, int]
    divergence_counts_by_family: dict[str, int]
    divergence_counts_by_reason_code: dict[str, int]
    cost_aware_precision_by_mode: dict[str, float | None]
    cost_aware_precision_counts_by_mode: dict[str, CostAwarePrecisionSummary]
    slippage_availability_by_mode: dict[str, str]
    latency_availability_by_mode: dict[str, str]
    degradation_summary: PaperToLiveDegradationReport
    threshold_context: dict[str, Any]
    registry_context: dict[str, Any]
    known_limitations: list[str]
    artifact_paths: dict[str, str]
