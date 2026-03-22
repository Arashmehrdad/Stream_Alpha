"""Aggregate metric computation for M18."""

# pylint: disable=too-many-arguments,too-many-locals

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from app.evaluation.schemas import (
    COMPARISON_FAMILIES,
    COMPARISON_FAMILY_MODE_PAIRS,
    DECISION_LAYERS,
    CostAwarePrecisionSummary,
    DegradationFamilySummary,
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
    opportunities,
    positions: list[PaperPosition],
    in_window_trace_ids_by_mode: dict[str, set[int]],
) -> tuple[list[PerformanceBreakdownRow], list[PerformanceBreakdownRow]]:
    """Compute performance rows by asset and by regime from linked position truth."""
    by_asset: dict[tuple[str, str], list[PaperPosition]] = defaultdict(list)
    by_regime: dict[tuple[str, str], list[PaperPosition]] = defaultdict(list)
    cost_aware_by_asset = compute_cost_aware_precision_by_group(
        opportunities=opportunities,
        key_builder=lambda row: (row.execution_mode, row.symbol),
    )
    cost_aware_by_regime = compute_cost_aware_precision_by_group(
        opportunities=opportunities,
        key_builder=lambda row: (row.execution_mode, row.regime_label or "UNKNOWN"),
    )
    for position in positions:
        trace_ids = in_window_trace_ids_by_mode.get(position.execution_mode, set())
        if position.entry_decision_trace_id not in trace_ids:
            continue
        by_asset[(position.execution_mode, position.symbol)].append(position)
        regime_key = position.entry_regime_label or "UNKNOWN"
        by_regime[(position.execution_mode, regime_key)].append(position)
    return (
        _build_performance_rows(
            grouped_rows=by_asset,
            breakdown_type="asset",
            cost_aware_summaries=cost_aware_by_asset,
        ),
        _build_performance_rows(
            grouped_rows=by_regime,
            breakdown_type="regime",
            cost_aware_summaries=cost_aware_by_regime,
        ),
    )


def compute_cost_aware_precision_by_mode(
    *,
    opportunities,
    execution_modes: tuple[str, ...],
) -> dict[str, CostAwarePrecisionSummary]:
    """Compute mode-level actioned BUY precision directly from opportunities."""
    summaries = _group_cost_aware_precision(
        opportunities=opportunities,
        key_builder=lambda row: row.execution_mode,
    )
    return {
        mode: summaries.get(
            mode,
            CostAwarePrecisionSummary(
                comparable_buy_count=0,
                positive_after_cost_buy_count=0,
                cost_aware_precision=None,
            ),
        )
        for mode in execution_modes
    }


def compute_cost_aware_precision_by_group(
    *,
    opportunities,
    key_builder,
) -> dict[tuple[str, str], CostAwarePrecisionSummary]:
    """Compute grouped actioned BUY precision summaries from opportunities."""
    return _group_cost_aware_precision(
        opportunities=opportunities,
        key_builder=key_builder,
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
    """Build the canonical degradation summary across all comparison families."""
    windows_by_family = {
        window.comparison_family: window for window in comparison_windows
    }
    opportunities_by_mode = _opportunities_by_mode(opportunities)
    families: dict[str, DegradationFamilySummary] = {}
    for family in COMPARISON_FAMILIES:
        left_mode, right_mode = COMPARISON_FAMILY_MODE_PAIRS[family]
        left_rows = opportunities_by_mode[left_mode]
        right_rows = opportunities_by_mode[right_mode]
        window = windows_by_family.get(family)
        matched_pairs = _matched_pairs_for_family(
            left_rows=left_rows,
            right_rows=right_rows,
            window=window,
        )
        family_events = [
            event for event in divergence_events if event.comparison_family == family
        ]
        stage_counts = Counter(event.divergence_stage for event in family_events)
        reason_counts = Counter(event.reason_code for event in family_events)
        comparable_fill_pair_count = len(
            [
                1
                for left_row, right_row in matched_pairs
                if _has_comparable_fill_truth(left_row, right_row)
            ]
        )
        comparable_slippage_pair_count = len(
            [
                1
                for left_row, right_row in matched_pairs
                if _has_comparable_slippage_truth(left_row, right_row)
            ]
        )
        families[family] = DegradationFamilySummary(
            comparison_family=family,
            left_mode=left_mode,
            right_mode=right_mode,
            coverage_counts={
                _coverage_label(left_mode): len(left_rows),
                _coverage_label(right_mode): len(right_rows),
            },
            matched_count=0 if window is None else window.matched_count,
            comparable_counts={
                "comparable_fill_pair_count": comparable_fill_pair_count,
                "comparable_slippage_pair_count": comparable_slippage_pair_count,
            },
            divergence_counts_by_stage=dict(sorted(stage_counts.items())),
            divergence_counts_by_reason_code=dict(sorted(reason_counts.items())),
            blockers=_family_blockers(
                family=family,
                left_rows=left_rows,
                right_rows=right_rows,
                matched_count=0 if window is None else window.matched_count,
                comparable_fill_pair_count=comparable_fill_pair_count,
                comparable_slippage_pair_count=comparable_slippage_pair_count,
            ),
            notes=_family_notes(family),
        )
    return PaperToLiveDegradationReport(
        schema_version="m18_paper_to_live_degradation_v2",
        families=families,
    )


def _build_performance_rows(
    *,
    grouped_rows: dict[tuple[str, str], list[PaperPosition]],
    breakdown_type: str,
    cost_aware_summaries: dict[tuple[str, str], CostAwarePrecisionSummary],
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
                comparable_buy_count=cost_aware_summaries.get(
                    (execution_mode, key),
                    CostAwarePrecisionSummary(
                        comparable_buy_count=0,
                        positive_after_cost_buy_count=0,
                        cost_aware_precision=None,
                    ),
                ).comparable_buy_count,
                positive_after_cost_buy_count=cost_aware_summaries.get(
                    (execution_mode, key),
                    CostAwarePrecisionSummary(
                        comparable_buy_count=0,
                        positive_after_cost_buy_count=0,
                        cost_aware_precision=None,
                    ),
                ).positive_after_cost_buy_count,
                total_entry_notional=sum(position.entry_notional for position in positions),
                total_fees=sum(
                    position.entry_fee + (0.0 if position.exit_fee is None else position.exit_fee)
                    for position in positions
                ),
                total_slippage_bps=None,
                cost_aware_precision=cost_aware_summaries.get(
                    (execution_mode, key),
                    CostAwarePrecisionSummary(
                        comparable_buy_count=0,
                        positive_after_cost_buy_count=0,
                        cost_aware_precision=None,
                    ),
                ).cost_aware_precision,
            )
        )
    return rows


def _group_cost_aware_precision(
    *,
    opportunities,
    key_builder,
) -> dict[object, CostAwarePrecisionSummary]:
    counts: dict[object, list[int]] = defaultdict(lambda: [0, 0])
    for row in opportunities:
        if row.executed_action != "BUY":
            continue
        comparable_positive = _cost_aware_positive_outcome(row)
        if comparable_positive is None:
            continue
        key = key_builder(row)
        counts[key][0] += 1
        if comparable_positive:
            counts[key][1] += 1
    summaries: dict[object, CostAwarePrecisionSummary] = {}
    for key, (comparable_buy_count, positive_buy_count) in counts.items():
        summaries[key] = CostAwarePrecisionSummary(
            comparable_buy_count=comparable_buy_count,
            positive_after_cost_buy_count=positive_buy_count,
            cost_aware_precision=(
                None
                if comparable_buy_count == 0
                else positive_buy_count / comparable_buy_count
            ),
        )
    return summaries


def _cost_aware_positive_outcome(row) -> bool | None:
    if row.position.position_status != "CLOSED":
        return None
    if row.position.realized_pnl is None:
        return None
    return row.position.realized_pnl > 0.0


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


def _opportunities_by_mode(opportunities) -> dict[str, list]:
    grouped: dict[str, list] = {"paper": [], "shadow": [], "live": []}
    for row in opportunities:
        grouped.setdefault(row.execution_mode, [])
        grouped[row.execution_mode].append(row)
    return grouped


def _matched_pairs_for_family(
    *,
    left_rows: list,
    right_rows: list,
    window,
) -> list[tuple[object, object]]:
    if window is None or window.overlap_start is None or window.overlap_end is None:
        return []
    left_index = {
        (row.symbol, row.signal_row_id): row
        for row in left_rows
        if window.overlap_start <= row.signal_as_of_time <= window.overlap_end
    }
    right_index = {
        (row.symbol, row.signal_row_id): row
        for row in right_rows
        if window.overlap_start <= row.signal_as_of_time <= window.overlap_end
    }
    return [
        (left_index[key], right_index[key])
        for key in sorted(set(left_index) & set(right_index))
    ]


def _has_comparable_fill_truth(left_row, right_row) -> bool:
    return bool(
        left_row.fill.truth_status in {"SIMULATED", "OBSERVED"}
        and right_row.fill.truth_status in {"SIMULATED", "OBSERVED"}
        and left_row.fill.fill_price is not None
        and right_row.fill.fill_price is not None
    )


def _has_comparable_slippage_truth(left_row, right_row) -> bool:
    return bool(
        left_row.fill.truth_status in {"SIMULATED", "OBSERVED"}
        and right_row.fill.truth_status in {"SIMULATED", "OBSERVED"}
        and left_row.fill.slippage_bps is not None
        and right_row.fill.slippage_bps is not None
    )


def _coverage_label(mode: str) -> str:
    return "tiny_live_opportunities" if mode == "live" else f"{mode}_opportunities"


def _family_blockers(
    *,
    family: str,
    left_rows: list,
    right_rows: list,
    matched_count: int,
    comparable_fill_pair_count: int,
    comparable_slippage_pair_count: int,
) -> list[str]:
    blockers: list[str] = []
    left_mode, right_mode = COMPARISON_FAMILY_MODE_PAIRS[family]
    if not left_rows:
        blockers.append(f"No {_display_mode_name(left_mode)} decision traces were available.")
    if not right_rows:
        blockers.append(f"No {_display_mode_name(right_mode)} decision traces were available.")
    if matched_count == 0:
        blockers.append(f"{family} had no matched overlap opportunities.")
    if "shadow" in COMPARISON_FAMILY_MODE_PAIRS[family]:
        if comparable_fill_pair_count == 0:
            blockers.append(
                "Shadow fill comparison remains unavailable because shadow does "
                "not persist validated fill truth."
            )
        if comparable_slippage_pair_count == 0:
            blockers.append(
                "Shadow slippage comparison remains not_applicable without "
                "validated fill truth."
            )
    if "live" in COMPARISON_FAMILY_MODE_PAIRS[family]:
        if comparable_fill_pair_count == 0:
            blockers.append(
                "Tiny-live fill comparison is unavailable because broker fill "
                "truth is not persisted."
            )
        if comparable_slippage_pair_count == 0:
            blockers.append(
                "Tiny-live slippage comparison is unavailable because observed "
                "broker slippage is absent."
            )
    return blockers


def _family_notes(family: str) -> list[str]:
    notes: list[str] = []
    if family == "paper_vs_shadow":
        notes.append(
            "Paper uses simulated fills while shadow intentionally keeps fill "
            "quality not_applicable until validated truth exists."
        )
    if family == "shadow_vs_tiny_live":
        notes.append(
            "Shadow versus tiny-live comparisons remain honest about unequal "
            "execution truth and broker fill coverage."
        )
    if family == "paper_to_tiny_live":
        notes.append(
            "Paper-to-tiny-live degradation stays explicit about guarded-live "
            "coverage gaps and missing broker fill truth."
        )
    return notes


def _display_mode_name(mode: str) -> str:
    return "tiny-live" if mode == "live" else mode
