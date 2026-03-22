"""Focused M18 service and artifact generation tests."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.schemas import COMPARISON_FAMILIES, EvaluationRequest
from app.evaluation.service import EvaluationService
from app.reliability.schemas import RecoveryEvent, ServiceHeartbeat
from app.trading.schemas import OrderLifecycleEvent, TradeLedgerEntry

from tests.test_evaluation_normalize import _trace, _ts


class FakeEvaluationRepository:
    """Small in-memory repository for deterministic M18 service tests."""

    def __init__(self) -> None:
        self.traces = [
            _trace(trace_id=201, execution_mode="paper", signal="BUY"),
            _trace(trace_id=202, execution_mode="shadow", signal="BUY"),
        ]
        self.order_events = [
            OrderLifecycleEvent(
                order_request_id=1,
                service_name="paper-trader",
                execution_mode="paper",
                symbol="BTC/USD",
                action="BUY",
                lifecycle_state="CREATED",
                event_time=_ts(12, 5),
                decision_trace_id=201,
            ),
            OrderLifecycleEvent(
                order_request_id=1,
                service_name="paper-trader",
                execution_mode="paper",
                symbol="BTC/USD",
                action="BUY",
                lifecycle_state="FILLED",
                event_time=_ts(12, 10),
                decision_trace_id=201,
            ),
            OrderLifecycleEvent(
                order_request_id=2,
                service_name="paper-trader",
                execution_mode="shadow",
                symbol="BTC/USD",
                action="BUY",
                lifecycle_state="CREATED",
                event_time=_ts(12, 5),
                decision_trace_id=202,
            ),
            OrderLifecycleEvent(
                order_request_id=2,
                service_name="paper-trader",
                execution_mode="shadow",
                symbol="BTC/USD",
                action="BUY",
                lifecycle_state="FILLED",
                event_time=_ts(12, 10),
                decision_trace_id=202,
            ),
        ]
        self.ledger_entries = [
            TradeLedgerEntry(
                service_name="paper-trader",
                execution_mode="paper",
                symbol="BTC/USD",
                action="BUY",
                reason="entry",
                fill_interval_begin=_ts(12, 10),
                fill_time=_ts(12, 10),
                fill_price=100.0,
                quantity=10.0,
                notional=1000.0,
                fee=1.0,
                slippage_bps=5.0,
                cash_flow=-1001.0,
                decision_trace_id=201,
            ),
        ]
        self.positions = []
        self.heartbeats = [
            ServiceHeartbeat(
                service_name="paper-trader",
                component_name="trading_runner",
                heartbeat_at=_ts(12, 5),
                health_overall_status="HEALTHY",
                reason_code="SERVICE_HEARTBEAT_HEALTHY",
            ),
        ]
        self.reliability_events = [
            RecoveryEvent(
                service_name="paper-trader",
                component_name="signal_client",
                event_type="RECOVERY_EVENT",
                event_time=_ts(12, 6),
                reason_code="SIGNAL_FETCH_FAILED",
            ),
        ]

    async def load_decision_traces(self, **_kwargs):
        return self.traces

    async def load_order_events_for_trace_ids(self, **_kwargs):
        return self.order_events

    async def load_positions_for_trace_ids(self, **_kwargs):
        return self.positions

    async def load_ledger_entries_for_trace_ids(self, **_kwargs):
        return self.ledger_entries

    async def load_service_heartbeats(self, **_kwargs):
        return self.heartbeats

    async def load_reliability_events(self, **_kwargs):
        return self.reliability_events


def test_generate_run_writes_required_artifacts_deterministically(tmp_path: Path) -> None:
    request = EvaluationRequest(
        service_name="paper-trader",
        source_exchange="kraken",
        interval_minutes=5,
        symbols=("BTC/USD",),
        execution_modes=("paper", "shadow", "live"),
        comparison_families=COMPARISON_FAMILIES,
        window_start=_ts(12, 0),
        window_end=_ts(12, 15),
        trading_config_path=str((tmp_path / "configs" / "paper_trading.paper.yaml").resolve()),
        evaluation_run_id="20260322T120000Z",
        generated_at=datetime(2026, 3, 22, 12, 20, tzinfo=timezone.utc),
    )
    service = EvaluationService(
        repository=FakeEvaluationRepository(),
        repo_root=tmp_path,
        registry_root=tmp_path / "artifacts" / "registry",
    )

    result = asyncio.run(service.generate_run(request))
    report_path = Path(result["artifact_paths"]["evaluation_report_json"])
    manifest_path = Path(result["artifact_paths"]["evaluation_manifest"])
    opportunities_csv = Path(result["artifact_paths"]["decision_opportunities_csv"])
    latency_csv = Path(result["artifact_paths"]["latency_distribution_csv"])
    index_path = tmp_path / "artifacts" / "evaluation" / "m18" / "index.jsonl"
    experiment_index = tmp_path / "artifacts" / "evaluation" / "experiments" / "index.jsonl"
    promotion_index = tmp_path / "artifacts" / "evaluation" / "promotions" / "index.jsonl"

    assert report_path.is_file()
    assert manifest_path.is_file()
    assert opportunities_csv.is_file()
    assert latency_csv.is_file()
    assert index_path.is_file()
    assert experiment_index.is_file()
    assert promotion_index.is_file()

    first_report = report_path.read_text(encoding="utf-8")
    repeat = asyncio.run(service.generate_run(request))
    assert report_path.read_text(encoding="utf-8") == first_report
    assert repeat["artifact_paths"]["evaluation_report_json"] == str(report_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["evaluation_run_id"] == "20260322T120000Z"
    assert payload["artifact_paths"]["paper_to_live_degradation_json"].endswith(
        "paper_to_live_degradation.json"
    )
