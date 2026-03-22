"""M18 evaluation orchestration and canonical report generation."""

# pylint: disable=too-many-locals

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339, utc_now
from app.evaluation.artifacts import (
    append_index_entry,
    evaluation_root,
    markdown_report,
    write_evaluation_artifacts,
    write_json,
    write_text,
)
from app.evaluation.config import (
    EvaluationConfig,
    default_evaluation_config_path,
    load_evaluation_config,
)
from app.evaluation.matching import build_comparison_windows, summarize_divergence_counts
from app.evaluation.metrics import (
    compute_cost_aware_precision_by_mode,
    compute_latency_distribution,
    compute_layer_comparison,
    compute_paper_to_live_degradation,
    compute_performance_rows,
    compute_slippage_distribution,
    compute_uptime_and_failures,
)
from app.evaluation.normalize import build_decision_opportunities
from app.evaluation.schemas import (
    EVALUATION_INDEX_SCHEMA_VERSION,
    EVALUATION_MANIFEST_SCHEMA_VERSION,
    EVALUATION_REPORT_SCHEMA_VERSION,
    EXPERIMENT_INDEX_SCHEMA_VERSION,
    EvaluationManifest,
    EvaluationReport,
    EvaluationRequest,
    PROMOTION_INDEX_SCHEMA_VERSION,
)
from app.training.registry import (
    current_registry_path,
    history_registry_path,
    load_current_registry_entry,
    repo_root as registry_repo_root,
)


class EvaluationDataRepository(Protocol):  # pylint: disable=too-few-public-methods
    """Repository contract for M18 evaluation loading."""

    async def load_decision_traces(
        self,
        *,
        service_name: str,
        execution_modes: tuple[str, ...],
        start: datetime,
        end: datetime,
    ):
        """Load decision traces for the evaluation window."""

    async def load_order_events_for_trace_ids(
        self,
        *,
        service_name: str,
        execution_modes: tuple[str, ...],
        decision_trace_ids: tuple[int, ...],
    ):
        """Load order events linked to the decision traces."""

    async def load_positions_for_trace_ids(
        self,
        *,
        service_name: str,
        execution_modes: tuple[str, ...],
        decision_trace_ids: tuple[int, ...],
    ):
        """Load positions linked to the decision traces."""

    async def load_ledger_entries_for_trace_ids(
        self,
        *,
        service_name: str,
        execution_modes: tuple[str, ...],
        decision_trace_ids: tuple[int, ...],
    ):
        """Load ledger rows linked to the decision traces."""

    async def load_service_heartbeats(
        self,
        *,
        service_name: str,
        start: datetime,
        end: datetime,
    ):
        """Load service heartbeats in the evaluation window."""

    async def load_reliability_events(
        self,
        *,
        service_name: str,
        start: datetime,
        end: datetime,
    ):
        """Load reliability events in the evaluation window."""


class EvaluationService:
    """Generate one canonical M18 evaluation run."""

    # Packet 1 deliberately exposes one public orchestration method so the CLI
    # and future API surfaces share a single canonical evaluation path.
    # pylint: disable=too-few-public-methods,too-many-arguments

    def __init__(
        self,
        *,
        repository: EvaluationDataRepository,
        repo_root: Path | None = None,
        registry_root: Path | None = None,
        evaluation_config_path: Path | None = None,
        evaluation_config: EvaluationConfig | None = None,
    ) -> None:
        self.repository = repository
        self.repo_root = registry_repo_root() if repo_root is None else Path(repo_root)
        self.registry_root = (
            self.repo_root / "artifacts" / "registry"
            if registry_root is None
            else Path(registry_root)
        )
        self.evaluation_config_path = (
            default_evaluation_config_path()
            if evaluation_config_path is None
            else Path(evaluation_config_path)
        )
        self.evaluation_config = (
            load_evaluation_config(self.evaluation_config_path)
            if evaluation_config is None
            else evaluation_config
        )

    async def generate_run(self, request: EvaluationRequest) -> dict[str, Any]:
        """Generate one canonical M18 evaluation run."""
        decision_traces = await self.repository.load_decision_traces(
            service_name=request.service_name,
            execution_modes=request.execution_modes,
            start=request.window_start,
            end=request.window_end,
        )
        decision_trace_ids = tuple(
            trace.decision_trace_id
            for trace in decision_traces
            if trace.decision_trace_id is not None
        )
        order_events = await self.repository.load_order_events_for_trace_ids(
            service_name=request.service_name,
            execution_modes=request.execution_modes,
            decision_trace_ids=decision_trace_ids,
        )
        positions = await self.repository.load_positions_for_trace_ids(
            service_name=request.service_name,
            execution_modes=request.execution_modes,
            decision_trace_ids=decision_trace_ids,
        )
        ledger_entries = await self.repository.load_ledger_entries_for_trace_ids(
            service_name=request.service_name,
            execution_modes=request.execution_modes,
            decision_trace_ids=decision_trace_ids,
        )
        heartbeats = await self.repository.load_service_heartbeats(
            service_name=request.service_name,
            start=request.window_start,
            end=request.window_end,
        )
        reliability_events = await self.repository.load_reliability_events(
            service_name=request.service_name,
            start=request.window_start,
            end=request.window_end,
        )
        opportunities = build_decision_opportunities(
            decision_traces=decision_traces,
            order_events=order_events,
            ledger_entries=ledger_entries,
            positions=positions,
        )
        comparison_windows, divergence_events = build_comparison_windows(
            opportunities=opportunities,
            comparison_families=request.comparison_families,
            evaluation_config=self.evaluation_config,
        )
        trace_ids_by_mode = {
            mode: {
                row.decision_trace_id
                for row in opportunities
                if row.execution_mode == mode
            }
            for mode in request.execution_modes
        }
        performance_by_asset, performance_by_regime = compute_performance_rows(
            opportunities=opportunities,
            positions=positions,
            in_window_trace_ids_by_mode=trace_ids_by_mode,
        )
        cost_aware_precision_by_mode = compute_cost_aware_precision_by_mode(
            opportunities=opportunities,
            execution_modes=request.execution_modes,
        )
        latency_distribution = compute_latency_distribution(opportunities)
        slippage_distribution = compute_slippage_distribution(ledger_entries)
        order_failure_counts_by_mode = _order_failure_counts_by_mode(opportunities)
        uptime_and_failures = compute_uptime_and_failures(
            service_name=request.service_name,
            window_start=to_rfc3339(request.window_start),
            window_end=to_rfc3339(request.window_end),
            heartbeats=heartbeats,
            reliability_events=reliability_events,
            divergence_events=divergence_events,
            order_failure_counts_by_mode=order_failure_counts_by_mode,
        )
        layer_comparison = compute_layer_comparison(opportunities)
        paper_to_live_degradation = compute_paper_to_live_degradation(
            opportunities=opportunities,
            comparison_windows=comparison_windows,
            divergence_events=divergence_events,
        )
        (
            divergence_counts_by_family,
            divergence_counts_by_reason,
        ) = summarize_divergence_counts(divergence_events)
        current_registry_entry = load_current_registry_entry(self.registry_root)
        known_limitations = _known_limitations(
            opportunities=opportunities,
            slippage_distribution=slippage_distribution,
            comparison_windows=comparison_windows,
        )
        manifest = EvaluationManifest(
            schema_version=EVALUATION_MANIFEST_SCHEMA_VERSION,
            evaluation_run_id=request.evaluation_run_id,
            generated_at=to_rfc3339(request.generated_at),
            service_name=request.service_name,
            source_exchange=request.source_exchange,
            interval_minutes=request.interval_minutes,
            symbols=list(request.symbols),
            execution_modes_requested=list(request.execution_modes),
            execution_modes_available=sorted({row.execution_mode for row in opportunities}),
            comparison_families=list(request.comparison_families),
            window_start=to_rfc3339(request.window_start),
            window_end=to_rfc3339(request.window_end),
            trading_config_path=request.trading_config_path,
            current_registry_entry=current_registry_entry,
            known_limitations=known_limitations,
            artifact_paths={},
        )
        report = EvaluationReport(
            schema_version=EVALUATION_REPORT_SCHEMA_VERSION,
            evaluation_run_id=request.evaluation_run_id,
            generated_at=to_rfc3339(request.generated_at),
            service_name=request.service_name,
            window_start=to_rfc3339(request.window_start),
            window_end=to_rfc3339(request.window_end),
            opportunity_counts_by_mode=_opportunity_counts(opportunities),
            matched_counts_by_family={
                window.comparison_family: window.matched_count
                for window in comparison_windows
            },
            divergence_counts_by_family=divergence_counts_by_family,
            divergence_counts_by_reason_code=divergence_counts_by_reason,
            cost_aware_precision_by_mode={
                mode: summary.cost_aware_precision
                for mode, summary in cost_aware_precision_by_mode.items()
            },
            cost_aware_precision_counts_by_mode=cost_aware_precision_by_mode,
            slippage_availability_by_mode=_distribution_truth_status(slippage_distribution),
            latency_availability_by_mode=_distribution_truth_status(latency_distribution),
            degradation_summary=paper_to_live_degradation,
            threshold_context={
                "config_path": str(self.evaluation_config_path.resolve()),
                "latency_drift_ms_threshold": self.evaluation_config.latency_drift_ms_threshold,
                "fill_price_drift_bps_threshold": (
                    self.evaluation_config.fill_price_drift_bps_threshold
                ),
                "slippage_drift_bps_threshold": (
                    self.evaluation_config.slippage_drift_bps_threshold
                ),
                "cost_aware_precision_horizon_notes": (
                    self.evaluation_config.cost_aware_precision_horizon_notes
                ),
                "minimum_comparable_count_notes": (
                    self.evaluation_config.minimum_comparable_count_notes
                ),
            },
            registry_context=_registry_context(current_registry_entry),
            known_limitations=known_limitations,
            artifact_paths={},
        )
        artifact_paths = write_evaluation_artifacts(
            repo_root=self.repo_root,
            manifest=manifest,
            report=report,
            decision_opportunity_rows=_decision_opportunity_rows(opportunities),
            performance_by_asset_rows=_performance_rows(performance_by_asset),
            performance_by_regime_rows=_performance_rows(performance_by_regime),
            divergence_rows=_divergence_rows(divergence_events),
            latency_rows=_distribution_rows(latency_distribution),
            slippage_rows=_distribution_rows(slippage_distribution),
            uptime_failures_payload=make_json_safe(asdict(uptime_and_failures)),
            layer_comparison_payload=make_json_safe(asdict(layer_comparison)),
            paper_to_live_payload=make_json_safe(asdict(paper_to_live_degradation)),
        )
        manifest = EvaluationManifest(**{**asdict(manifest), "artifact_paths": artifact_paths})
        report = EvaluationReport(**{**asdict(report), "artifact_paths": artifact_paths})
        write_json(Path(artifact_paths["evaluation_manifest"]), manifest)
        write_json(Path(artifact_paths["evaluation_report_json"]), report)
        write_text(
            Path(artifact_paths["evaluation_report_markdown"]),
            markdown_report(manifest=manifest, report=report),
        )
        _append_indexes(
            repo_root=self.repo_root,
            registry_root=self.registry_root,
            manifest=manifest,
            report=report,
            current_registry_entry=current_registry_entry,
        )
        return {
            "manifest": manifest,
            "report": report,
            "comparison_windows": comparison_windows,
            "divergence_events": divergence_events,
            "artifact_paths": artifact_paths,
            "known_limitations": known_limitations,
        }


def default_evaluation_run_id(*, generated_at: datetime | None = None) -> str:
    """Return the default deterministic run identifier."""
    observed_at = utc_now() if generated_at is None else generated_at.astimezone(timezone.utc)
    return observed_at.strftime("%Y%m%dT%H%M%SZ")


def _append_indexes(
    *,
    repo_root: Path,
    registry_root: Path,
    manifest: EvaluationManifest,
    report: EvaluationReport,
    current_registry_entry: dict[str, Any] | None,
) -> None:
    root = evaluation_root(repo_root)
    append_index_entry(
        root / "m18" / "index.jsonl",
        {
            "schema_version": EVALUATION_INDEX_SCHEMA_VERSION,
            "generated_at": manifest.generated_at,
            "evaluation_run_id": manifest.evaluation_run_id,
            "service_name": manifest.service_name,
            "window_start": manifest.window_start,
            "window_end": manifest.window_end,
            "execution_modes_requested": manifest.execution_modes_requested,
            "execution_modes_available": manifest.execution_modes_available,
            "comparison_families": manifest.comparison_families,
            "opportunity_counts_by_mode": report.opportunity_counts_by_mode,
            "divergence_counts_by_family": report.divergence_counts_by_family,
            "artifact_paths": manifest.artifact_paths,
        },
    )
    append_index_entry(
        root / "experiments" / "index.jsonl",
        {
            "schema_version": EXPERIMENT_INDEX_SCHEMA_VERSION,
            "recorded_at": manifest.generated_at,
            "evaluation_run_id": manifest.evaluation_run_id,
            "service_name": manifest.service_name,
            "model_version": (
                None
                if current_registry_entry is None
                else current_registry_entry.get("model_version")
            ),
            "model_name": (
                None
                if current_registry_entry is None
                else current_registry_entry.get("model_name")
            ),
            "comparison_path": (
                None
                if current_registry_entry is None
                else current_registry_entry.get("comparison_path")
            ),
            "run_manifest_path": (
                None
                if current_registry_entry is None
                else current_registry_entry.get("run_manifest_path")
            ),
            "window_start": manifest.window_start,
            "window_end": manifest.window_end,
            "execution_modes": manifest.execution_modes_available,
            "evaluation_report_path": manifest.artifact_paths["evaluation_report_json"],
        },
    )
    append_index_entry(
        root / "promotions" / "index.jsonl",
        {
            "schema_version": PROMOTION_INDEX_SCHEMA_VERSION,
            "recorded_at": manifest.generated_at,
            "evaluation_run_id": manifest.evaluation_run_id,
            "current_registry_path": str(current_registry_path(registry_root)),
            "registry_history_path": str(history_registry_path(registry_root)),
            "current_registry_entry": current_registry_entry,
            "evaluation_report_path": manifest.artifact_paths["evaluation_report_json"],
            "paper_to_live_degradation_path": manifest.artifact_paths[
                "paper_to_live_degradation_json"
            ],
        },
    )


def _decision_opportunity_rows(opportunities) -> list[dict[str, object]]:
    return [
        {
            "service_name": row.service_name,
            "execution_mode": row.execution_mode,
            "symbol": row.symbol,
            "signal_row_id": row.signal_row_id,
            "decision_trace_id": row.decision_trace_id,
            "signal_interval_begin": row.signal_interval_begin,
            "signal_as_of_time": row.signal_as_of_time,
            "model_name": row.model_name,
            "model_version": row.model_version,
            "regime_label": row.regime_label,
            "regime_run_id": row.regime_run_id,
            "signal_action": row.signal_action,
            "decision_source": row.decision_source,
            "signal_reason_code": row.signal_reason_code,
            "freshness_status": row.freshness_status,
            "health_overall_status": row.health_overall_status,
            "model_only_action": row.model_only_action,
            "regime_aware_action": row.regime_aware_action,
            "risk_gated_action": row.risk_gated_action,
            "executed_action": row.executed_action,
            "risk_outcome": row.risk_outcome,
            "risk_primary_reason_code": row.risk_primary_reason_code,
            "requested_notional": row.requested_notional,
            "approved_notional": row.approved_notional,
            "risk_reason_codes": row.risk_reason_codes,
            "safety_blocked": row.safety_blocked,
            "reliability_blocked": row.reliability_blocked,
            "order_truth_status": row.order.truth_status,
            "order_request_id": row.order.order_request_id,
            "order_created_at": row.order.created_at,
            "order_first_response_at": row.order.first_response_at,
            "order_terminal_at": row.order.terminal_at,
            "order_terminal_state": row.order.terminal_state,
            "order_terminal_reason_code": row.order.terminal_reason_code,
            "order_lifecycle_states": row.order.lifecycle_states,
            "broker_name": row.order.broker_name,
            "account_id": row.order.account_id,
            "environment_name": row.order.environment_name,
            "fill_truth_status": row.fill.truth_status,
            "fill_action": row.fill.action,
            "fill_time": row.fill.fill_time,
            "fill_price": row.fill.fill_price,
            "fill_notional": row.fill.notional,
            "fill_fee": row.fill.fee,
            "fill_slippage_bps": row.fill.slippage_bps,
            "position_id": row.position.position_id,
            "position_status": row.position.position_status,
            "position_opened_at": row.position.opened_at,
            "position_closed_at": row.position.closed_at,
            "realized_pnl": row.position.realized_pnl,
            "realized_return": row.position.realized_return,
        }
        for row in opportunities
    ]


def _performance_rows(rows) -> list[dict[str, object]]:
    return [make_json_safe(asdict(row)) for row in rows]


def _divergence_rows(rows) -> list[dict[str, object]]:
    return [make_json_safe(asdict(row)) for row in rows]


def _distribution_rows(rows) -> list[dict[str, object]]:
    return [make_json_safe(asdict(row)) for row in rows]


def _known_limitations(*, opportunities, slippage_distribution, comparison_windows) -> list[str]:
    limitations: list[str] = []
    if not [row for row in opportunities if row.execution_mode == "live"]:
        limitations.append("No tiny-live decision traces were available in the evaluation window.")
    live_slippage = next(
        (row for row in slippage_distribution if row.execution_mode == "live"),
        None,
    )
    if live_slippage is not None and live_slippage.truth_status != "OBSERVED":
        limitations.append(
            "Tiny-live slippage remains unavailable without observed broker fill truth."
        )
    shadow_slippage = next(
        (row for row in slippage_distribution if row.execution_mode == "shadow"),
        None,
    )
    if shadow_slippage is not None and shadow_slippage.truth_status == "NOT_APPLICABLE":
        limitations.append(
            "Shadow fill and slippage quality remain not_applicable because "
            "shadow does not carry validated fill truth."
        )
    for window in comparison_windows:
        if window.matched_count == 0:
            limitations.append(f"{window.comparison_family} had no matched overlap opportunities.")
    return limitations


def _opportunity_counts(opportunities) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in opportunities:
        counts[row.execution_mode] = counts.get(row.execution_mode, 0) + 1
    return dict(sorted(counts.items()))


def _distribution_truth_status(rows) -> dict[str, str]:
    return {row.execution_mode: row.truth_status for row in rows}


def _registry_context(current_registry_entry: dict[str, Any] | None) -> dict[str, Any]:
    if current_registry_entry is None:
        return {
            "current_registry_available": False,
            "model_version": None,
            "comparison_path": None,
        }
    return {
        "current_registry_available": True,
        "model_version": current_registry_entry.get("model_version"),
        "model_name": current_registry_entry.get("model_name"),
        "comparison_path": current_registry_entry.get("comparison_path"),
        "run_manifest_path": current_registry_entry.get("run_manifest_path"),
    }


def _order_failure_counts_by_mode(opportunities) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for mode in sorted({row.execution_mode for row in opportunities}):
        mode_rows = [row for row in opportunities if row.execution_mode == mode]
        rejected = len([row for row in mode_rows if row.order.terminal_state == "REJECTED"])
        failed = len([row for row in mode_rows if row.order.terminal_state == "FAILED"])
        rows[mode] = {
            "rejected": rejected,
            "failed": failed,
            "total_failures": rejected + failed,
        }
    return rows
