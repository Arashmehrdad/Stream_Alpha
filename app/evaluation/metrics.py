"""Aggregate metric computation for M18."""

# pylint: disable=too-many-arguments,too-many-locals

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from app.evaluation.schemas import (
    DECISION_LAYERS,
    DistributionRow,
    DivergenceEvent,
    LayerComparisonReport,
    LayerTransitionMetrics,
    PaperToLiveDegradationReport,
    PerformanceBreakdownRow,
    UptimeAndFailuresReport,
    UptimeComponentSummary,
)
from app.reliability.schemas import RecoveryEvent, ServiceHeartbeat
from app.trading.schemas import PaperPosition, TradeLedgerEntry


def compute_performance_rows(
    *,
    positions: list[PaperPosition],
    in_window_trace_ids_by_mode: dict[str, set[int]],
) -> tuple[list[PerformanceBreakdownRow], list[PerformanceBreakdownRow]]:
    """Compute performance rows by asset and by regime from linked position truth."""
    by_asset: dict[tuple[str, str], list[PaperPosition]] = defaultdict(list)
    by_regime: dict[tuple[str, str], list[PaperPosition]] = defaultdict(list)
    for position in positions:
        trace_ids = in_window_trace_ids_by_mode.get(position.execution_mode, set())
        if position.entry_decision_trace_id not in trace_ids:
            continue
        by_asset[(position.execution_mode, position.symbol)].append(position)
        regime_key = position.entry_regime_label or "UNKNOWN"
        by_regime[(position.execution_mode, regime_key)].append(position)
    return (
        _build_performance_rows(grouped_rows=by_asset, breakdown_type="asset"),
        _build_performance_rows(grouped_rows=by_regime, breakdown_type="regime"),
    )


def compute_latency_distribution(opportunities) -> list[DistributionRow]:
    """Compute latency distribution rows from normalized opportunities."""
    grouped: dict[str, list[float]] = defaultdict(list)
    truth_status_by_mode: dict[str, str] = {}
    for row in opportunities:
        truth_status_by_mode[row.execution_mode] = row.order.truth_status
        if row.order.created_at is None or row.order.first_response_at is None:
            continue
        latency_ms = (
            row.order.first_response_at - row.order.created_at
        ).total_seconds() * 1000.0
        grouped[row.execution_mode].append(latency_ms)
    rows: list[DistributionRow] = []
    for mode in sorted(truth_status_by_mode):
        rows.append(
            _distribution_row(
                execution_mode=mode,
                metric_name="order_first_response_latency_ms",
                truth_status=truth_status_by_mode[mode],
                values=grouped.get(mode, []),
                note=(
                    "Shadow and paper latency is local-path timing, not broker-network latency."
                    if mode in {"paper", "shadow"}
                    else None
                ),
            )
        )
    return rows


def compute_slippage_distribution(ledger_entries: list[TradeLedgerEntry]) -> list[DistributionRow]:
    """Compute slippage distribution rows with honest truth-status handling."""
    grouped: dict[str, list[float]] = defaultdict(list)
    truth_status_by_mode: dict[str, str] = {
        "paper": "SIMULATED",
        "shadow": "NOT_APPLICABLE",
        "live": "MISSING",
    }
    for entry in ledger_entries:
        if entry.execution_mode == "shadow":
            continue
        grouped[entry.execution_mode].append(entry.slippage_bps)
        if entry.execution_mode == "live":
            truth_status_by_mode["live"] = "OBSERVED"
    return [
        _distribution_row(
            execution_mode=mode,
            metric_name="slippage_bps",
            truth_status=truth_status_by_mode[mode],
            values=grouped.get(mode, []),
            note=(
                "Shadow slippage is intentionally not applicable without validated fill truth."
                if mode == "shadow"
                else (
                    "Tiny-live slippage remains unavailable until broker fill truth is persisted."
                    if mode == "live" and truth_status_by_mode[mode] != "OBSERVED"
                    else None
                )
            ),
        )
        for mode in ("paper", "shadow", "live")
    ]


def compute_uptime_and_failures(
    *,
    service_name: str,
    window_start: str,
    window_end: str,
    heartbeats: list[ServiceHeartbeat],
    reliability_events: list[RecoveryEvent],
    divergence_events: list[DivergenceEvent],
    order_failure_counts_by_mode: dict[str, dict[str, int]],
) -> UptimeAndFailuresReport:
    """Compute uptime, failure, and recovery summaries from M13 truth."""
    components: dict[str, list[ServiceHeartbeat]] = defaultdict(list)
    for heartbeat in heartbeats:
        components[heartbeat.component_name].append(heartbeat)
    component_rows: list[UptimeComponentSummary] = []
    for component_name in sorted(components):
        rows = components[component_name]
        total = len(rows)
        healthy = len([row for row in rows if row.health_overall_status == "HEALTHY"])
        degraded = len([row for row in rows if row.health_overall_status == "DEGRADED"])
        unavailable = len([row for row in rows if row.health_overall_status == "UNAVAILABLE"])
        component_rows.append(
            UptimeComponentSummary(
                component_name=component_name,
                total_heartbeats=total,
                healthy_heartbeats=healthy,
                degraded_heartbeats=degraded,
                unavailable_heartbeats=unavailable,
                healthy_ratio=(None if total == 0 else healthy / total),
            )
        )
    events_by_component = Counter(event.component_name for event in reliability_events)
    events_by_reason = Counter(event.reason_code for event in reliability_events)
    events_by_type = Counter(event.event_type for event in reliability_events)
    recovery_event_count = len(
        [
            event
            for event in reliability_events
            if "RECOVERY" in event.event_type or "CLEAR" in event.reason_code
        ]
    )
    divergence_failure_counts = Counter(
        event.reason_code
        for event in divergence_events
        if event.reason_code
        in {"FAILURE_INTERRUPTION", "RECOVERY_DELAY", "DOWNTIME_IMPACT"}
    )
    return UptimeAndFailuresReport(
        schema_version="m18_uptime_failures_v1",
        service_name=service_name,
        window_start=window_start,
        window_end=window_end,
        heartbeat_components=component_rows,
        reliability_events_total=len(reliability_events),
        reliability_events_by_component=dict(sorted(events_by_component.items())),
        reliability_events_by_reason_code=dict(sorted(events_by_reason.items())),
        reliability_events_by_type=dict(sorted(events_by_type.items())),
        recovery_event_count=recovery_event_count,
        order_failure_counts_by_mode=order_failure_counts_by_mode,
        divergence_failure_counts=dict(sorted(divergence_failure_counts.items())),
    )


def compute_layer_comparison(opportunities) -> LayerComparisonReport:
    """Compute layer transition counts from normalized opportunities."""
    by_mode = defaultdict(list)
    for row in opportunities:
        by_mode[row.execution_mode].append(row)
    mode_reports: dict[str, LayerTransitionMetrics] = {}
    for mode in sorted(by_mode):
        rows = by_mode[mode]
        action_counts = {
            layer: Counter(getattr(row, f"{layer}_action") for row in rows)
            for layer in DECISION_LAYERS
        }
        transition_changes = {
            "model_only_to_regime_aware_changed": len(
                [row for row in rows if row.model_only_action != row.regime_aware_action]
            ),
            "regime_aware_to_risk_gated_changed": len(
                [row for row in rows if row.regime_aware_action != row.risk_gated_action]
            ),
            "risk_gated_to_executed_changed": len(
                [row for row in rows if row.risk_gated_action != row.executed_action]
            ),
        }
        blocked_counts = {
            "reliability_blocked": len([row for row in rows if row.reliability_blocked]),
            "safety_blocked": len([row for row in rows if row.safety_blocked]),
            "risk_blocked": len([row for row in rows if row.risk_outcome == "BLOCKED"]),
        }
        mode_reports[mode] = LayerTransitionMetrics(
            action_counts={
                layer: dict(sorted(counts.items()))
                for layer, counts in action_counts.items()
            },
            transition_changes=transition_changes,
            blocked_counts=blocked_counts,
        )
    return LayerComparisonReport(
        schema_version="m18_layer_comparison_v1",
        modes=mode_reports,
    )


def compute_paper_to_live_degradation(
    *,
    opportunities,
    comparison_windows,
    divergence_events: list[DivergenceEvent],
) -> PaperToLiveDegradationReport:
    """Build the canonical paper-to-tiny-live degradation summary."""
    family = "paper_to_tiny_live"
    window = next(
        (
            item
            for item in comparison_windows
            if item.comparison_family == family
        ),
        None,
    )
    family_events = [
        event for event in divergence_events if event.comparison_family == family
    ]
    stage_counts = Counter(event.divergence_stage for event in family_events)
    reason_counts = Counter(event.reason_code for event in family_events)
    paper_rows = [row for row in opportunities if row.execution_mode == "paper"]
    live_rows = [row for row in opportunities if row.execution_mode == "live"]
    observed_live_slippage = len(
        [
            row for row in live_rows
            if row.fill.truth_status == "OBSERVED" and row.fill.slippage_bps is not None
        ]
    )
    observed_live_fills = len(
        [row for row in live_rows if row.fill.truth_status == "OBSERVED"]
    )
    blockers: list[str] = []
    if not live_rows:
        blockers.append("No tiny-live decision traces were available in the evaluation window.")
    if observed_live_fills == 0:
        blockers.append(
            "Tiny-live fill comparison is unavailable because broker fill "
            "truth is not persisted."
        )
    if observed_live_slippage == 0:
        blockers.append(
            "Tiny-live slippage comparison is unavailable because observed "
            "broker slippage is absent."
        )
    return PaperToLiveDegradationReport(
        schema_version="m18_paper_to_live_degradation_v1",
        comparison_family=family,
        coverage={
            "paper_opportunities": len(paper_rows),
            "tiny_live_opportunities": len(live_rows),
            "matched_count": 0 if window is None else window.matched_count,
        },
        divergence_counts_by_stage=dict(sorted(stage_counts.items())),
        divergence_counts_by_reason_code=dict(sorted(reason_counts.items())),
        comparable_counts={
            "paper_fill_rows": len(
                [
                    row
                    for row in paper_rows
                    if row.fill.truth_status in {"SIMULATED", "OBSERVED"}
                ]
            ),
            "tiny_live_fill_rows": observed_live_fills,
            "tiny_live_slippage_rows": observed_live_slippage,
        },
        blockers=blockers,
        notes=[
            "Shadow and paper comparisons can use simulated execution truth; "
            "tiny-live comparisons remain explicit about missing broker fill "
            "truth.",
        ],
    )


def _build_performance_rows(
    *,
    grouped_rows: dict[tuple[str, str], list[PaperPosition]],
    breakdown_type: str,
) -> list[PerformanceBreakdownRow]:
    rows: list[PerformanceBreakdownRow] = []
    for (execution_mode, key), positions in sorted(grouped_rows.items()):
        closed = [position for position in positions if position.status == "CLOSED"]
        open_rows = [position for position in positions if position.status == "OPEN"]
        realized_returns = [
            float(position.realized_return)
            for position in closed
            if position.realized_return is not None
        ]
        winning_trades = [
            position
            for position in closed
            if (position.realized_pnl or 0.0) > 0.0
        ]
        rows.append(
            PerformanceBreakdownRow(
                execution_mode=execution_mode,
                breakdown_type=breakdown_type,
                key=key,
                position_count=len(positions),
                closed_position_count=len(closed),
                open_position_count=len(open_rows),
                realized_pnl=sum((position.realized_pnl or 0.0) for position in closed),
                average_realized_return=None if not realized_returns else mean(realized_returns),
                win_rate=None if not closed else len(winning_trades) / len(closed),
                total_entry_notional=sum(position.entry_notional for position in positions),
                total_fees=sum(
                    position.entry_fee + (0.0 if position.exit_fee is None else position.exit_fee)
                    for position in positions
                ),
                total_slippage_bps=None,
                cost_aware_precision=None if not closed else len(winning_trades) / len(closed),
            )
        )
    return rows


def _distribution_row(
    *,
    execution_mode: str,
    metric_name: str,
    truth_status: str,
    values: list[float],
    note: str | None,
) -> DistributionRow:
    ordered = sorted(values)
    return DistributionRow(
        execution_mode=execution_mode,
        metric_name=metric_name,
        truth_status=truth_status,
        count=len(ordered),
        min_value=None if not ordered else ordered[0],
        mean_value=None if not ordered else mean(ordered),
        p50_value=_percentile(ordered, 0.50),
        p90_value=_percentile(ordered, 0.90),
        p95_value=_percentile(ordered, 0.95),
        max_value=None if not ordered else ordered[-1],
        note=note,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    index = int(round((len(values) - 1) * percentile))
    return values[index]
