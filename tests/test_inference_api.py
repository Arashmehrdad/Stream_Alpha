"""API tests for the Stream Alpha M4 inference service."""

# pylint: disable=duplicate-code,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-lines

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import joblib
from fastapi.testclient import TestClient
import pytest

from app.adaptation.schemas import (
    AdaptationDriftResponse,
    AdaptationPerformanceResponse,
    AdaptationProfilesResponse,
    AdaptationPromotionsResponse,
    AdaptationSummaryResponse,
    AdaptiveDriftRecord,
    AdaptivePerformanceWindow,
    AdaptiveProfileRecord,
    AdaptivePromotionDecisionRecord,
    AdaptiveRecentPerformanceSummary,
    CalibrationProfile,
    EffectiveThresholds,
    SizingPolicy,
    ThresholdPolicy,
)
from app.adaptation.service import AppliedAdaptation
from app.alerting.config import (
    AlertingArtifactConfig,
    AlertingConfig,
    OrderFailureSpikeConfig,
    SignalAlertConfig,
)
from app.alerting.schemas import OperationalAlertEvent, OperationalAlertState
from app.common.time import to_rfc3339, utc_now
from app.common.config import (
    FeatureSettings,
    InferenceSettings,
    KafkaSettings,
    KrakenSettings,
    PostgresSettings,
    RetrySettings,
    Settings,
    TableSettings,
    TopicSettings,
)
from app.continual_learning.schemas import (
    ContinualLearningContextPayload,
    ContinualLearningDriftCapRecord,
    ContinualLearningDriftCapsResponse,
    ContinualLearningEventRecord,
    ContinualLearningEventsResponse,
    ContinualLearningExperimentsResponse,
    ContinualLearningProfileRecord,
    ContinualLearningProfilesResponse,
    ContinualLearningPromoteProfileRequest,
    ContinualLearningPromotionDecisionRecord,
    ContinualLearningPromotionsResponse,
    ContinualLearningRollbackRequest,
    ContinualLearningSummaryResponse,
    ContinualLearningWorkflowResponse,
)
from app.ensemble.config import AgreementPolicyConfig, EnsembleConfig, load_ensemble_config
from app.ensemble.schemas import EnsembleProfileRecord, EnsembleResult
from app.ensemble.service import EnsembleService
from app.inference.db import DatabaseUnavailableError
from app.inference.main import create_app
from app.inference.service import InferenceService, load_model_artifact
from app.reliability.schemas import (
    FeatureLagSnapshot,
    RecoveryEvent,
    ReliabilityState,
    ServiceHeartbeat,
)
from app.regime.live import load_live_regime_runtime


class SerializableProbabilityModel:
    """Serializable classifier stub for API tests."""

    def __init__(self, prob_up: float) -> None:
        self._prob_up = prob_up

    def predict_proba(self, rows: list[dict]) -> list[list[float]]:
        """Return a fixed probability for each requested row."""
        return [[1.0 - self._prob_up, self._prob_up] for _ in rows]


class SerializableFeatureAwareModel:
    """Serializable model stub whose probabilities move with numeric features."""

    def predict_proba(self, rows: list[dict]) -> list[list[float]]:
        """Return deterministic binary probabilities from a few explainable inputs."""
        payload: list[list[float]] = []
        for row in rows:
            prob_up = (
                0.50
                + (0.50 * float(row["momentum_3"]))
                - (0.20 * float(row["realized_vol_12"]))
                + (0.05 * float(row["volume_zscore_12"]))
            )
            payload.append([1.0 - prob_up, prob_up])
        return payload


class FakeDatabase:
    """Minimal async database stub for service and API tests."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        row: dict | None = None,
        history_rows: list[dict] | None = None,
        reference_vector: dict[str, float] | None = None,
        healthy: bool = True,
        fetch_error: Exception | None = None,
    ) -> None:
        self.row = row
        self.history_rows = [] if history_rows is None else list(history_rows)
        self.reference_vector = {} if reference_vector is None else reference_vector
        self.healthy = healthy
        self.fetch_error = fetch_error
        self.last_interval_begin = None

    async def connect(self) -> None:
        """Open the fake connection."""
        return None

    async def close(self) -> None:
        """Close the fake connection."""
        return None

    async def is_healthy(self) -> bool:
        """Return the configured fake health state."""
        return self.healthy

    async def fetch_latest_feature_row(
        self,
        *,
        symbol: str,
        interval_minutes: int,
        interval_begin=None,
    ) -> dict | None:
        """Return the configured row or raise the configured error."""
        del symbol, interval_minutes
        self.last_interval_begin = interval_begin
        if self.fetch_error is not None:
            raise self.fetch_error
        if (
            interval_begin is not None
            and self.row is not None
            and self.row["interval_begin"] != interval_begin
        ):
            return None
        return self.row

    async def fetch_feature_reference_vector(
        self,
        *,
        feature_names: tuple[str, ...],
        interval_minutes: int,
    ) -> dict[str, float]:
        """Return configured or row-derived reference values for explainability tests."""
        del interval_minutes
        if self.fetch_error is not None:
            raise self.fetch_error
        if self.reference_vector:
            return {
                feature_name: float(self.reference_vector[feature_name])
                for feature_name in feature_names
                if feature_name in self.reference_vector
            }
        if self.row is None:
            return {}
        return {
            feature_name: float(self.row[feature_name])
            for feature_name in feature_names
            if feature_name in self.row
            and isinstance(self.row[feature_name], (int, float))
            and not isinstance(self.row[feature_name], bool)
        }

    async def fetch_feature_history_rows(
        self,
        *,
        symbol: str,
        interval_minutes: int,
        end_as_of_time,
        limit: int,
    ) -> list[dict]:
        """Return ordered same-symbol history rows up to the requested scoring cutoff."""
        del interval_minutes
        if self.fetch_error is not None:
            raise self.fetch_error
        candidate_rows = (
            list(self.history_rows)
            if self.history_rows
            else ([] if self.row is None else [self.row])
        )
        ordered_rows = sorted(
            [
                row
                for row in candidate_rows
                if row["symbol"] == symbol and row["as_of_time"] <= end_as_of_time
            ],
            key=lambda row: (row["as_of_time"], row["interval_begin"]),
        )
        if limit <= 0:
            return []
        return ordered_rows[-limit:]


class FakeReliabilityStore:
    """Minimal reliability-store stub for inference API tests."""

    def __init__(
        self,
        *,
        heartbeats: dict[tuple[str, str], ServiceHeartbeat] | None = None,
        reliability_states: dict[tuple[str, str], ReliabilityState] | None = None,
        lag_states: dict[tuple[str, str], list[FeatureLagSnapshot]] | None = None,
        latest_recovery_event: RecoveryEvent | None = None,
    ) -> None:
        self.heartbeats = {} if heartbeats is None else heartbeats
        self.reliability_states = (
            {} if reliability_states is None else reliability_states
        )
        self.lag_states = {} if lag_states is None else lag_states
        self.latest_recovery_event = latest_recovery_event
        self.saved_system_snapshots = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def save_service_heartbeat(self, heartbeat: ServiceHeartbeat) -> None:
        self.heartbeats[(heartbeat.service_name, heartbeat.component_name)] = heartbeat

    async def load_latest_service_heartbeat(
        self,
        *,
        service_name: str,
        component_name: str,
    ) -> ServiceHeartbeat | None:
        return self.heartbeats.get((service_name, component_name))

    async def load_reliability_state(
        self,
        *,
        service_name: str,
        component_name: str,
    ) -> ReliabilityState | None:
        return self.reliability_states.get((service_name, component_name))

    async def load_feature_lag_states(
        self,
        *,
        service_name: str,
        component_name: str,
    ) -> list[FeatureLagSnapshot]:
        return list(self.lag_states.get((service_name, component_name), []))

    async def load_latest_recovery_event(self) -> RecoveryEvent | None:
        return self.latest_recovery_event

    async def save_system_reliability_state(self, snapshot) -> None:
        self.saved_system_snapshots.append(snapshot)


class FakeAlertRepository:
    """Minimal alert repository stub for M17 inference API tests."""

    def __init__(
        self,
        *,
        active_states: list[OperationalAlertState] | None = None,
        timeline_events: list[OperationalAlertEvent] | None = None,
    ) -> None:
        self.active_states = [] if active_states is None else list(active_states)
        self.timeline_events = [] if timeline_events is None else list(timeline_events)

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def load_active_states(
        self,
        *,
        service_name: str,
        execution_mode: str,
    ) -> list[OperationalAlertState]:
        return [
            state
            for state in self.active_states
            if state.service_name == service_name and state.execution_mode == execution_mode
        ]

    async def load_timeline_events(  # pylint: disable=too-many-arguments
        self,
        *,
        service_name: str,
        execution_mode: str,
        limit: int,
        category: str | None = None,
        severity: str | None = None,
        symbol: str | None = None,
        active_only: bool = False,
    ) -> list[OperationalAlertEvent]:
        active_fingerprints = {
            state.fingerprint
            for state in self.active_states
            if state.is_active
        }
        events = [
            event
            for event in self.timeline_events
            if event.service_name == service_name
            and event.execution_mode == execution_mode
            and (category is None or event.category == category)
            and (severity is None or event.severity == severity)
            and (symbol is None or event.symbol == symbol)
            and (not active_only or event.fingerprint in active_fingerprints)
        ]
        return list(
            sorted(
                events,
                key=lambda event: (
                    event.event_time,
                    -1 if event.event_id is None else event.event_id,
                ),
                reverse=True,
            )
        )[:limit]


class FakeAdaptationService:
    """Minimal adaptation-service stub for M19 API tests."""

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def resolve_applied_adaptation(self, **_kwargs) -> AppliedAdaptation:
        return AppliedAdaptation(
            profile_id="profile-test-1",
            calibrated_confidence=0.74,
            effective_thresholds=EffectiveThresholds(buy_prob_up=0.53, sell_prob_up=0.47),
            adaptive_size_multiplier=1.10,
            drift_status="WATCH",
            recent_performance_summary=AdaptiveRecentPerformanceSummary(
                window_id="last_20_trades",
                window_type="trade_count",
                trade_count=20,
                net_pnl_after_costs=0.02,
                max_drawdown=0.01,
                profit_factor=1.10,
                win_rate=0.55,
                blocked_trade_rate=0.10,
                shadow_divergence_rate=0.05,
            ),
            adaptation_reason_codes=("ADAPTATION_PROFILE_ACTIVE",),
            frozen_by_health_gate=False,
        )

    async def summary(self, **_kwargs) -> AdaptationSummaryResponse:
        return AdaptationSummaryResponse(
            enabled=True,
            active_profile_count=1,
            active_profile_id="profile-test-1",
            adaptation_status="ACTIVE",
            evidence_backed=True,
            latest_drift_status="WATCH",
            latest_promotion_decision="HOLD",
            reason_codes=["ACTIVE_PROFILE_PRESENT"],
        )

    async def drift(self, **_kwargs) -> AdaptationDriftResponse:
        return AdaptationDriftResponse(
            items=[
                AdaptiveDriftRecord(
                    symbol="BTC/USD",
                    regime_label="ALL",
                    detector_name="psi",
                    window_id="drift-1",
                    reference_window_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                    reference_window_end=datetime(2026, 3, 10, tzinfo=timezone.utc),
                    live_window_start=datetime(2026, 3, 11, tzinfo=timezone.utc),
                    live_window_end=datetime(2026, 3, 22, tzinfo=timezone.utc),
                    drift_score=0.12,
                    warning_threshold=0.10,
                    breach_threshold=0.20,
                    status="WATCH",
                    reason_code="DRIFT_WATCH",
                )
            ]
        )

    async def performance(self, **_kwargs) -> AdaptationPerformanceResponse:
        return AdaptationPerformanceResponse(
            items=[
                AdaptivePerformanceWindow(
                    execution_mode="paper",
                    symbol="BTC/USD",
                    regime_label="ALL",
                    window_id="last_20_trades",
                    window_type="trade_count",
                    window_start=datetime(2026, 3, 10, tzinfo=timezone.utc),
                    window_end=datetime(2026, 3, 22, tzinfo=timezone.utc),
                    trade_count=20,
                    net_pnl_after_costs=0.02,
                    max_drawdown=0.01,
                    profit_factor=1.1,
                    expectancy=0.001,
                    win_rate=0.55,
                    precision=0.56,
                    avg_slippage_bps=3.0,
                    blocked_trade_rate=0.10,
                    shadow_divergence_rate=0.05,
                    health_context={"health_overall_status": "HEALTHY"},
                )
            ]
        )

    async def profiles(self, **_kwargs) -> AdaptationProfilesResponse:
        return AdaptationProfilesResponse(
            items=[
                AdaptiveProfileRecord(
                    profile_id="profile-test-1",
                    status="ACTIVE",
                    execution_mode_scope="paper",
                    symbol_scope="ALL",
                    regime_scope="ALL",
                    threshold_policy_json=ThresholdPolicy(buy_threshold_delta=0.02),
                    sizing_policy_json=SizingPolicy(size_multiplier=1.1),
                    calibration_profile_json=CalibrationProfile(method="identity"),
                    source_evidence_json={"source": "unit-test"},
                )
            ]
        )

    async def promotions(self, **_kwargs) -> AdaptationPromotionsResponse:
        return AdaptationPromotionsResponse(
            items=[
                AdaptivePromotionDecisionRecord(
                    decision_id="promotion-test-1",
                    target_type="PROFILE",
                    target_id="profile-test-1",
                    incumbent_id=None,
                    decision="HOLD",
                    metrics_delta_json={"net_pnl_after_costs": 0.02},
                    safety_checks_json={"reliability_healthy": True},
                    research_integrity_json={"trade_count": 20},
                    reason_codes=["ACTIVE_PROFILE_PRESENT"],
                    summary_text="unit-test summary",
                    decided_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
                )
            ]
        )


class RecordingAdaptationService(FakeAdaptationService):
    """Adaptation stub that records the last confidence input it received."""

    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    async def resolve_applied_adaptation(self, **kwargs) -> AppliedAdaptation:
        self.last_kwargs = dict(kwargs)
        return await super().resolve_applied_adaptation(**kwargs)


class NullAdaptationService:
    """No-op adaptation stub for tests that are not exercising M19 behavior."""

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def resolve_applied_adaptation(self, **_kwargs) -> AppliedAdaptation:
        return AppliedAdaptation(
            adaptation_reason_codes=("ADAPTATION_REPOSITORY_UNAVAILABLE",)
        )

    async def summary(self, **_kwargs) -> AdaptationSummaryResponse:
        return AdaptationSummaryResponse(
            enabled=True,
            active_profile_count=0,
            adaptation_status="UNAVAILABLE",
            evidence_backed=False,
            reason_codes=["ADAPTATION_REPOSITORY_UNAVAILABLE"],
        )

    async def drift(self, **_kwargs) -> AdaptationDriftResponse:
        return AdaptationDriftResponse()

    async def performance(self, **_kwargs) -> AdaptationPerformanceResponse:
        return AdaptationPerformanceResponse()

    async def profiles(self, **_kwargs) -> AdaptationProfilesResponse:
        return AdaptationProfilesResponse()

    async def promotions(self, **_kwargs) -> AdaptationPromotionsResponse:
        return AdaptationPromotionsResponse()


class NullEnsembleService:
    """No-op ensemble stub for tests that are not exercising M20 behavior."""

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def resolve_ensemble(self, **_kwargs):
        return EnsembleResult(
            active=False,
            fallback_reason="ENSEMBLE_FALLBACK_SINGLE_MODEL",
            weighting_reason_codes=("ENSEMBLE_FALLBACK_SINGLE_MODEL",),
        )

    @property
    def config(self):
        return EnsembleConfig(
            enabled=False,
            candidate_roles=("GENERALIST",),
            regime_weight_matrix={"TREND_UP": {"GENERALIST": 1.00}},
            agreement_policy=AgreementPolicyConfig(
                high_ratio_min=0.80,
                high_spread_max=0.12,
                medium_ratio_min=0.67,
                medium_spread_max=0.20,
                high_multiplier=1.00,
                medium_multiplier=0.93,
                low_multiplier=0.85,
            ),
            artifact_root="artifacts/ensemble",
        )


class NullContinualLearningService:
    """No-op continual-learning stub for tests not exercising M21 read surfaces."""

    def __init__(self) -> None:
        self.config = type("Cfg", (), {"enabled": True})()

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def resolve_runtime_context(self, **_kwargs) -> ContinualLearningContextPayload:
        return ContinualLearningContextPayload(
            enabled=True,
            active_profile_id=None,
            live_eligible=False,
            reason_codes=["NO_ACTIVE_CONTINUAL_LEARNING_PROFILE"],
        )

    async def summary(self, **_kwargs) -> ContinualLearningSummaryResponse:
        return ContinualLearningSummaryResponse(
            enabled=True,
            active_profile_count=0,
            active_profile_id=None,
            continual_learning_status="IDLE",
            evidence_backed=False,
            latest_drift_cap_status=None,
            latest_promotion_decision=None,
            reason_codes=["NO_ACTIVE_CONTINUAL_LEARNING_PROFILE"],
        )

    async def experiments(self, **_kwargs) -> ContinualLearningExperimentsResponse:
        return ContinualLearningExperimentsResponse(items=[])

    async def profiles(self, **_kwargs) -> ContinualLearningProfilesResponse:
        return ContinualLearningProfilesResponse(
            items=[
                ContinualLearningProfileRecord(
                    profile_id="cl-profile-1",
                    candidate_type="CALIBRATION_OVERLAY",
                    status="ACTIVE",
                    execution_mode_scope="paper",
                    symbol_scope="BTC/USD",
                    regime_scope="TREND_UP",
                    baseline_target_type="MODEL_VERSION",
                    baseline_target_id="m20-live",
                    source_experiment_id="cl-exp-1",
                    promotion_stage="LIVE_ELIGIBLE",
                    live_eligible=True,
                )
            ]
        )

    async def drift_caps(self, **_kwargs) -> ContinualLearningDriftCapsResponse:
        return ContinualLearningDriftCapsResponse(
            items=[
                ContinualLearningDriftCapRecord(
                    cap_id="cl-cap-1",
                    execution_mode_scope="paper",
                    symbol_scope="BTC/USD",
                    regime_scope="TREND_UP",
                    candidate_type="CALIBRATION_OVERLAY",
                    status="WATCH",
                    observed_drift_score=0.12,
                    warning_threshold=0.10,
                    breach_threshold=0.20,
                    reason_code="DRIFT_WATCH",
                )
            ]
        )

    async def promotions(self, **_kwargs) -> ContinualLearningPromotionsResponse:
        return ContinualLearningPromotionsResponse(
            items=[
                ContinualLearningPromotionDecisionRecord(
                    decision_id="cl-decision-1",
                    target_type="PROFILE",
                    target_id="cl-profile-1",
                    candidate_type="CALIBRATION_OVERLAY",
                    decision="HOLD",
                    summary_text="hold",
                    decided_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
                )
            ]
        )

    async def events(self, **_kwargs) -> ContinualLearningEventsResponse:
        return ContinualLearningEventsResponse(
            items=[
                ContinualLearningEventRecord(
                    event_id="cl-event-1",
                    event_type="PROFILE_ACTIVE",
                    profile_id="cl-profile-1",
                    reason_code="ACTIVE_PROFILE_PRESENT",
                )
            ]
        )

    async def load_profile(self, *, profile_id: str) -> ContinualLearningProfileRecord | None:
        if profile_id != "cl-profile-1":
            return None
        return ContinualLearningProfileRecord(
            profile_id="cl-profile-1",
            candidate_type="CALIBRATION_OVERLAY",
            status="APPROVED",
            execution_mode_scope="paper",
            symbol_scope="BTC/USD",
            regime_scope="TREND_UP",
            baseline_target_type="MODEL_VERSION",
            baseline_target_id="m20-live",
            source_experiment_id="cl-exp-1",
            promotion_stage="PAPER_APPROVED",
            live_eligible=False,
            rollback_target_profile_id="cl-profile-prev-1",
        )

    async def promote_profile(
        self,
        request: ContinualLearningPromoteProfileRequest,
        health_overall_status: str | None = None,
        freshness_status: str | None = None,
    ) -> ContinualLearningWorkflowResponse:
        return ContinualLearningWorkflowResponse(
            success=True,
            blocked=False,
            decision_id=request.decision_id,
            decision="PROMOTE",
            target_profile_id=request.profile_id,
            incumbent_profile_id="cl-profile-prev-1",
            promotion_stage_after=request.requested_promotion_stage,
            live_eligible_after=(request.requested_promotion_stage == "LIVE_ELIGIBLE"),
            drift_cap_status="WATCH",
            health_overall_status=health_overall_status,
            freshness_status=freshness_status,
            event_id=f"event:{request.decision_id}",
            reason_codes=["CONTINUAL_LEARNING_PROMOTION_APPLIED"],
            summary_text=request.summary_text,
        )

    async def rollback_profile(
        self,
        request: ContinualLearningRollbackRequest,
        health_overall_status: str | None = None,
        freshness_status: str | None = None,
    ) -> ContinualLearningWorkflowResponse:
        return ContinualLearningWorkflowResponse(
            success=True,
            blocked=False,
            decision_id=request.decision_id,
            decision="ROLLBACK",
            target_profile_id="cl-profile-prev-1",
            incumbent_profile_id="cl-profile-1",
            promotion_stage_after="LIVE_ELIGIBLE",
            live_eligible_after=True,
            drift_cap_status="WATCH",
            health_overall_status=health_overall_status,
            freshness_status=freshness_status,
            event_id=f"event:{request.decision_id}",
            reason_codes=["CONTINUAL_LEARNING_ROLLBACK_APPLIED"],
            summary_text=request.summary_text,
        )


class WorkflowContinualLearningService(NullContinualLearningService):
    """Continual-learning stub that records workflow endpoint inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.promote_requests: list[dict] = []
        self.rollback_requests: list[dict] = []

    async def promote_profile(
        self,
        request: ContinualLearningPromoteProfileRequest,
        health_overall_status: str | None = None,
        freshness_status: str | None = None,
    ) -> ContinualLearningWorkflowResponse:
        self.promote_requests.append(
            {
                "request": request,
                "health_overall_status": health_overall_status,
                "freshness_status": freshness_status,
            }
        )
        return await super().promote_profile(
            request,
            health_overall_status=health_overall_status,
            freshness_status=freshness_status,
        )

    async def rollback_profile(
        self,
        request: ContinualLearningRollbackRequest,
        health_overall_status: str | None = None,
        freshness_status: str | None = None,
    ) -> ContinualLearningWorkflowResponse:
        self.rollback_requests.append(
            {
                "request": request,
                "health_overall_status": health_overall_status,
                "freshness_status": freshness_status,
            }
        )
        return await super().rollback_profile(
            request,
            health_overall_status=health_overall_status,
            freshness_status=freshness_status,
        )


def _build_settings(model_path: str, *, operator_api_key: str = "") -> Settings:
    return Settings(
        app_name="streamalpha",
        log_level="INFO",
        service_name="producer",
        heartbeat_interval_seconds=15,
        kraken=KrakenSettings(
            ws_url="wss://ws.kraken.com/v2",
            rest_ohlc_url="https://api.kraken.com/0/public/OHLC",
            symbols=("BTC/USD", "ETH/USD", "SOL/USD"),
            ohlc_interval_minutes=5,
        ),
        kafka=KafkaSettings(
            bootstrap_servers="redpanda:9092",
            client_id="streamalpha-producer",
        ),
        postgres=PostgresSettings(
            host="127.0.0.1",
            port=5432,
            database="streamalpha",
            user="streamalpha",
            password="change-me-local-only",
        ),
        topics=TopicSettings(
            raw_trades="raw.trades",
            raw_ohlc="raw.ohlc",
            raw_health="raw.health",
        ),
        tables=TableSettings(
            raw_trades="raw_trades",
            raw_ohlc="raw_ohlc",
            feature_ohlc="feature_ohlc",
            producer_heartbeat="producer_heartbeat",
        ),
        retry=RetrySettings(
            initial_delay_seconds=1.0,
            max_delay_seconds=30.0,
            multiplier=2.0,
            jitter_seconds=0.5,
        ),
        features=FeatureSettings(
            consumer_group_id="streamalpha-feature-consumer",
            service_name="features",
            finalization_grace_seconds=30,
            bootstrap_candles=64,
        ),
        inference=InferenceSettings(
            model_path=model_path,
            service_name="inference",
            signal_buy_prob_up=0.55,
            signal_sell_prob_up=0.45,
            operator_api_key=operator_api_key,
        ),
    )


def _write_thresholds_artifact(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "thresholds.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "m8_thresholds_v1",
                "run_id": "20260320T120000Z",
                "source_table": "feature_ohlc",
                "source_exchange": "kraken",
                "interval_minutes": 5,
                "required_inputs": [
                    "realized_vol_12",
                    "momentum_3",
                    "macd_line_12_26",
                ],
                "regime_labels": [
                    "TREND_UP",
                    "TREND_DOWN",
                    "RANGE",
                    "HIGH_VOL",
                ],
                "thresholds_by_symbol": {
                    "BTC/USD": {
                        "symbol": "BTC/USD",
                        "fitted_row_count": 100,
                        "high_vol_threshold": 0.05,
                        "trend_abs_threshold": 0.02,
                    },
                    "ETH/USD": {
                        "symbol": "ETH/USD",
                        "fitted_row_count": 100,
                        "high_vol_threshold": 0.06,
                        "trend_abs_threshold": 0.03,
                    },
                    "SOL/USD": {
                        "symbol": "SOL/USD",
                        "fitted_row_count": 100,
                        "high_vol_threshold": 0.07,
                        "trend_abs_threshold": 0.04,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def _write_signal_policy(tmp_path: Path) -> Path:
    policy_path = tmp_path / "regime_signal_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "m9_regime_signal_policy_v1",
                "policies": {
                    "TREND_UP": {
                        "buy_prob_up": 0.54,
                        "sell_prob_up": 0.44,
                        "allow_new_long_entries": True,
                    },
                    "RANGE": {
                        "buy_prob_up": 0.58,
                        "sell_prob_up": 0.42,
                        "allow_new_long_entries": True,
                    },
                    "TREND_DOWN": {
                        "buy_prob_up": 0.60,
                        "sell_prob_up": 0.46,
                        "allow_new_long_entries": False,
                    },
                    "HIGH_VOL": {
                        "buy_prob_up": 0.62,
                        "sell_prob_up": 0.48,
                        "allow_new_long_entries": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return policy_path


def _build_regime_runtime(tmp_path: Path):
    return load_live_regime_runtime(
        thresholds_path=str(_write_thresholds_artifact(tmp_path)),
        signal_policy_path=str(_write_signal_policy(tmp_path)),
    )


def _build_alerting_config(tmp_path: Path) -> AlertingConfig:
    operations_dir = tmp_path / "artifacts" / "operations"
    return AlertingConfig(
        schema_version="m17_alerting_v1",
        order_failure_spike=OrderFailureSpikeConfig(
            window_minutes=15,
            warning_count=2,
            critical_count=4,
        ),
        signals=SignalAlertConfig(
            silence_window_intervals=3,
            flood_window_intervals=3,
            flood_warning_count=2,
            flood_critical_count=4,
        ),
        artifacts=AlertingArtifactConfig(
            daily_summary_dir=str((operations_dir / "daily").resolve()),
            startup_safety_path=str((operations_dir / "startup_safety.json").resolve()),
        ),
    )


def _write_startup_safety_artifact(tmp_path: Path) -> Path:
    alerting_config = _build_alerting_config(tmp_path)
    artifact_path = Path(alerting_config.artifacts.startup_safety_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "m17_startup_safety_report_v1",
                "generated_at": "2026-03-22T10:05:00Z",
                "service_name": "paper-trader",
                "execution_mode": "paper",
                "runtime_profile": "paper",
                "startup_safety_passed": True,
                "primary_reason_code": "STARTUP_SAFETY_PASSED",
                "summary_text": "Startup safety is clear for this non-live runtime.",
                "startup_validation": {
                    "report_path": str((tmp_path / "startup_report.json").resolve()),
                    "report_exists": True,
                    "startup_validation_passed": True,
                    "primary_reason_code": "STARTUP_SAFETY_PASSED",
                    "summary_text": "M16 startup validation passed.",
                    "payload": {
                        "startup_validation_passed": True,
                    },
                },
                "live_startup": {
                    "report_path": str((tmp_path / "startup_report.json").resolve()),
                    "report_exists": False,
                    "startup_validation_passed": True,
                    "primary_reason_code": "STARTUP_SAFETY_PASSED",
                    "summary_text": (
                        "M12 guarded-live startup checks are not required for this mode."
                    ),
                    "payload": {},
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def _write_daily_summary_artifact(tmp_path: Path) -> Path:
    alerting_config = _build_alerting_config(tmp_path)
    artifact_path = (
        Path(alerting_config.artifacts.daily_summary_dir) / "2026-03-22.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "m17_daily_operations_summary_v1",
                "generated_at": "2026-03-22T10:10:00Z",
                "service_name": "paper-trader",
                "execution_mode": "paper",
                "runtime_profile": "paper",
                "summary_date": "2026-03-22",
                "counts_by_category": {
                    "FEED_STALE": 1,
                    "CONSUMER_LAG": 0,
                },
                "unresolved_count": 1,
                "highest_severity": "WARNING",
                "startup_safety_status": {
                    "startup_safety_passed": True,
                    "primary_reason_code": "STARTUP_SAFETY_PASSED",
                    "summary_text": "Startup safety is clear for this non-live runtime.",
                    "startup_report_path": str((tmp_path / "startup_report.json").resolve()),
                },
                "order_failure_counts": {
                    "rejected": 1,
                    "failed": 0,
                    "total_failures": 1,
                    "window_minutes": 15,
                },
                "drawdown_state": {
                    "available": True,
                    "breached": False,
                },
                "actionable_signal_counts": {
                    "buy_count": 2,
                    "sell_count": 1,
                    "total_actionable": 3,
                    "decision_trace_count": 5,
                },
                "silence_flood_episodes": {
                    "signal_silence_events": 0,
                    "signal_flood_events": 1,
                },
                "live_mode_activation_count": 0,
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def _write_artifact(tmp_path: Path, *, prob_up: float) -> Path:
    artifact_path = tmp_path / f"model-{prob_up:.2f}.joblib"
    joblib.dump(
        {
            "model_name": "runtime_candidate_fixture",
            "trained_at": "2026-03-19T22:30:02Z",
            "feature_columns": ["symbol", "close_price"],
            "expanded_feature_names": ["symbol=BTC/USD", "close_price"],
            "model": SerializableProbabilityModel(prob_up),
        },
        artifact_path,
    )
    return artifact_path


def _write_registry_model_artifact(
    tmp_path: Path,
    *,
    model_version: str,
    prob_up: float,
    model_name: str,
) -> Path:
    model_dir = tmp_path / "artifacts" / "registry" / "models" / model_version
    model_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = model_dir / "model.joblib"
    joblib.dump(
        {
            "model_name": model_name,
            "trained_at": "2026-03-19T22:30:02Z",
            "feature_columns": ["symbol", "close_price"],
            "expanded_feature_names": ["symbol=BTC/USD", "close_price"],
            "model": SerializableProbabilityModel(prob_up),
        },
        artifact_path,
    )
    (model_dir / "registry_entry.json").write_text(
        json.dumps(
            {
                "model_version": model_version,
                "model_name": model_name,
                "model_artifact_path": str(artifact_path.resolve()),
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def _build_active_ensemble_service(tmp_path: Path) -> EnsembleService:
    _write_registry_model_artifact(
        tmp_path,
        model_version="ensemble-generalist-v1",
        prob_up=0.60,
        model_name="ensemble_generalist",
    )
    _write_registry_model_artifact(
        tmp_path,
        model_version="ensemble-trend-v1",
        prob_up=0.70,
        model_name="ensemble_trend_specialist",
    )
    profile = EnsembleProfileRecord(
        profile_id="ens-profile-active-1",
        status="ACTIVE",
        approval_stage="ACTIVATED",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_roster_json=[
            {
                "candidate_id": "generalist-1",
                "candidate_role": "GENERALIST",
                "model_version": "ensemble-generalist-v1",
                "scope_regimes": ["TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"],
                "enabled": True,
                "expected_model_name": "ensemble_generalist",
            },
            {
                "candidate_id": "trend-1",
                "candidate_role": "TREND_SPECIALIST",
                "model_version": "ensemble-trend-v1",
                "scope_regimes": ["TREND_UP", "TREND_DOWN"],
                "enabled": True,
                "expected_model_name": "ensemble_trend_specialist",
            },
        ],
        evidence_summary_json={
            "runtime_truth": {
                "current_truth": {
                    "roster_status": "ACTIVE_WEAK",
                    "reason_codes": [
                        "GENERALIST_ROLE_PRESENT",
                        "RANGE_SPECIALIST_ROLE_MISSING",
                        "STRONGER_SPECIALIST_ROSTER_UNSUPPORTED",
                    ],
                }
            }
        },
    )

    async def _load_active_profile(**_kwargs) -> EnsembleProfileRecord | None:
        return profile

    return EnsembleService(
        config=load_ensemble_config(Path("configs/ensemble.yaml")),
        profile_loader=_load_active_profile,
    )


def _write_feature_aware_artifact(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "artifacts" / "training" / "m3" / "20260321T120000Z" / "model.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=False)
    joblib.dump(
        {
            "model_name": "runtime_candidate_fixture",
            "trained_at": "2026-03-21T12:00:00Z",
            "feature_columns": [
                "symbol",
                "momentum_3",
                "realized_vol_12",
                "volume_zscore_12",
            ],
            "expanded_feature_names": [
                "symbol=BTC/USD",
                "momentum_3",
                "realized_vol_12",
                "volume_zscore_12",
            ],
            "model": SerializableFeatureAwareModel(),
        },
        artifact_path,
    )
    return artifact_path


def _feature_row(
    symbol: str = "BTC/USD",
    *,
    realized_vol_12: float = 0.03,
    momentum_3: float = 0.03,
    macd_line_12_26: float = 1.2,
    base_time: datetime | None = None,
) -> dict:
    if base_time is None:
        base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
            minutes=5
        )
    interval_end = base_time + timedelta(minutes=5)
    as_of_time = interval_end
    computed_at = interval_end + timedelta(minutes=1)
    return {
        "id": 1,
        "source_exchange": "kraken",
        "symbol": symbol,
        "interval_minutes": 5,
        "interval_begin": base_time,
        "interval_end": interval_end,
        "as_of_time": as_of_time,
        "computed_at": computed_at,
        "raw_event_id": "evt-1",
        "open_price": 70000.0,
        "high_price": 70100.0,
        "low_price": 69900.0,
        "close_price": 70050.0,
        "vwap": 70020.0,
        "trade_count": 120,
        "volume": 12.5,
        "log_return_1": 0.01,
        "log_return_3": 0.02,
        "momentum_3": momentum_3,
        "return_mean_12": 0.01,
        "return_std_12": 0.02,
        "realized_vol_12": realized_vol_12,
        "rsi_14": 55.0,
        "macd_line_12_26": macd_line_12_26,
        "volume_mean_12": 10.0,
        "volume_std_12": 2.0,
        "volume_zscore_12": 1.25,
        "close_zscore_12": 0.75,
        "lag_log_return_1": 0.005,
        "lag_log_return_2": 0.004,
        "lag_log_return_3": 0.003,
        "created_at": computed_at,
        "updated_at": computed_at,
    }


def _build_client(  # pylint: disable=too-many-arguments
    tmp_path: Path,
    *,
    prob_up: float = 0.7,
    database: FakeDatabase,
    reliability_store: FakeReliabilityStore | None = None,
    alert_repository: FakeAlertRepository | None = None,
    artifact_path: Path | None = None,
    adaptation_service: FakeAdaptationService | NullAdaptationService | None = None,
    ensemble_service: EnsembleService | NullEnsembleService | None = None,
    continual_learning_service: NullContinualLearningService | None = None,
    operator_api_key: str = "",
) -> TestClient:
    resolved_artifact_path = (
        _write_artifact(tmp_path, prob_up=prob_up)
        if artifact_path is None
        else artifact_path
    )
    artifact = load_model_artifact(str(resolved_artifact_path))
    service = InferenceService(
        _build_settings(
            str(resolved_artifact_path),
            operator_api_key=operator_api_key,
        ),
        database=database,
        model_artifact=artifact,
        regime_runtime=_build_regime_runtime(tmp_path),
        reliability_store=reliability_store or FakeReliabilityStore(),
        alerting_config=_build_alerting_config(tmp_path),
        alert_repository=alert_repository or FakeAlertRepository(),
        adaptation_service=adaptation_service or NullAdaptationService(),
        ensemble_service=ensemble_service or NullEnsembleService(),
        continual_learning_service=continual_learning_service or NullContinualLearningService(),
    )
    return TestClient(create_app(service))


def test_health_reports_success_and_dependency_failure(tmp_path: Path) -> None:
    """`/health` should reflect DB reachability while keeping the model loaded."""
    healthy_client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))
    healthy_response = healthy_client.get("/health")

    assert healthy_response.status_code == 200
    assert healthy_response.json()["status"] == "ok"
    assert healthy_response.json()["regime_loaded"] is True
    assert healthy_response.json()["regime_run_id"] == "20260320T120000Z"
    assert healthy_response.json()["health_overall_status"] == "HEALTHY"
    assert healthy_response.json()["freshness_status"] == "FRESH"

    unhealthy_client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row(), healthy=False),
    )
    unhealthy_response = unhealthy_client.get("/health")

    assert unhealthy_response.status_code == 503
    assert unhealthy_response.json()["database"] == "unavailable"
    assert unhealthy_response.json()["health_overall_status"] == "UNAVAILABLE"


def test_additive_runtime_metadata_is_exposed_on_health_metrics_and_reliability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    startup_report_path = tmp_path / "startup_report.json"
    startup_report_path.write_text(
        json.dumps(
            {
                "schema_version": "m16_startup_report_v1",
                "checked_at": "2026-03-22T10:00:00Z",
                "runtime_profile": "paper",
                "startup_validation_passed": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("STREAMALPHA_RUNTIME_PROFILE", "paper")
    monkeypatch.setenv("STREAMALPHA_STARTUP_REPORT_PATH", str(startup_report_path))

    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))

    health_payload = client.get("/health").json()
    metrics_payload = client.get("/metrics").json()
    reliability_payload = client.get("/reliability/system").json()

    assert health_payload["runtime_profile"] == "paper"
    assert health_payload["execution_mode"] == "paper"
    assert health_payload["startup_validation_passed"] is True
    assert health_payload["startup_report_path"] == str(startup_report_path.resolve())
    assert metrics_payload["runtime_profile"] == "paper"
    assert metrics_payload["execution_mode"] == "paper"
    assert metrics_payload["startup_validation_passed"] is True
    assert reliability_payload["runtime_profile"] == "paper"
    assert reliability_payload["execution_mode"] == "paper"
    assert reliability_payload["startup_validation_passed"] is True


def test_m19_additive_adaptation_fields_and_read_only_endpoints_are_exposed(
    tmp_path: Path,
) -> None:
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row()),
        adaptation_service=FakeAdaptationService(),
    )

    health_payload = client.get("/health").json()
    predict_payload = client.get("/predict", params={"symbol": "BTC/USD"}).json()
    signal_payload = client.get("/signal", params={"symbol": "BTC/USD"}).json()
    summary_payload = client.get("/adaptation/summary").json()
    drift_payload = client.get("/adaptation/drift").json()

    assert health_payload["active_adaptation_count"] == 1
    assert health_payload["adaptation_status"] == "ACTIVE"
    assert health_payload["adaptation_evidence_backed"] is True
    assert predict_payload["adaptation_profile_id"] == "profile-test-1"
    assert predict_payload["calibrated_confidence"] == 0.74
    assert signal_payload["adaptation_profile_id"] == "profile-test-1"
    assert signal_payload["effective_thresholds"]["buy_prob_up"] == 0.53
    assert signal_payload["adaptive_size_multiplier"] == 1.1
    assert signal_payload["drift_status"] == "WATCH"
    assert summary_payload["active_profile_id"] == "profile-test-1"
    assert drift_payload["items"][0]["status"] == "WATCH"


def test_m17_alert_and_operations_endpoints_are_exposed_read_only(
    tmp_path: Path,
) -> None:
    checked_at = datetime(2026, 3, 22, 10, 15, tzinfo=timezone.utc)
    _write_startup_safety_artifact(tmp_path)
    _write_daily_summary_artifact(tmp_path)
    active_state = OperationalAlertState(
        fingerprint="paper-trader|paper|FEED_STALE|producer|*",
        service_name="paper-trader",
        execution_mode="paper",
        category="FEED_STALE",
        symbol=None,
        source_component="producer",
        is_active=True,
        severity="WARNING",
        reason_code="FEED_STALE",
        opened_at=checked_at - timedelta(minutes=5),
        last_seen_at=checked_at,
        last_event_id=7,
        occurrence_count=2,
    )
    timeline_event = OperationalAlertEvent(
        service_name="paper-trader",
        execution_mode="paper",
        category="FEED_STALE",
        severity="WARNING",
        event_state="OPEN",
        reason_code="FEED_STALE",
        source_component="producer",
        symbol=None,
        fingerprint=active_state.fingerprint,
        summary_text="Producer feed is stale.",
        detail="feed age exceeded max threshold",
        event_time=checked_at,
        payload_json={"feed_age_seconds": 120.0},
        event_id=7,
        created_at=checked_at,
    )
    client = _build_client(
        tmp_path,
        database=FakeDatabase(row=_feature_row(base_time=checked_at - timedelta(minutes=5))),
        alert_repository=FakeAlertRepository(
            active_states=[active_state],
            timeline_events=[timeline_event],
        ),
    )

    health_payload = client.get("/health").json()
    active_payload = client.get("/alerts/active").json()
    timeline_payload = client.get(
        "/alerts/timeline",
        params={"limit": 10, "category": "FEED_STALE", "active_only": True},
    ).json()
    daily_payload = client.get(
        "/operations/daily-summary",
        params={"date": "2026-03-22"},
    ).json()
    startup_payload = client.get("/operations/startup-safety").json()

    assert health_payload["active_alert_count"] == 1
    assert health_payload["max_alert_severity"] == "WARNING"
    assert health_payload["startup_safety_status"] == "PASSED"
    assert health_payload["startup_safety_reason_code"] == "STARTUP_SAFETY_PASSED"
    assert active_payload[0]["category"] == "FEED_STALE"
    assert active_payload[0]["reason_code"] == "FEED_STALE"
    assert timeline_payload[0]["event_state"] == "OPEN"
    assert timeline_payload[0]["summary_text"] == "Producer feed is stale."
    assert daily_payload["summary_date"] == "2026-03-22"
    assert daily_payload["unresolved_count"] == 1
    assert startup_payload["startup_safety_passed"] is True
    assert startup_payload["primary_reason_code"] == "STARTUP_SAFETY_PASSED"


def test_reliability_system_endpoint_returns_canonical_cross_service_summary(
    tmp_path: Path,
) -> None:
    """`/reliability/system` should aggregate heartbeats, lag, and breaker state."""
    now = utc_now().replace(microsecond=0)
    reliability_store = FakeReliabilityStore(
        heartbeats={
            (
                "producer",
                "producer",
                ): ServiceHeartbeat(
                    service_name="producer",
                    component_name="producer",
                    heartbeat_at=now,
                    health_overall_status="HEALTHY",
                    reason_code="SERVICE_HEARTBEAT_HEALTHY",
                    detail=(
                        '{"last_exchange_activity_at":"'
                        f'{to_rfc3339(now)}'
                        '"}'
                    ),
                ),
            (
                "features",
                "features",
            ): ServiceHeartbeat(
                service_name="features",
                component_name="features",
                heartbeat_at=now,
                health_overall_status="HEALTHY",
                reason_code="SERVICE_HEARTBEAT_HEALTHY",
                detail='{"lag_breach_active":false}',
            ),
            (
                "paper-trader",
                "trading_runner",
            ): ServiceHeartbeat(
                service_name="paper-trader",
                component_name="trading_runner",
                heartbeat_at=now,
                health_overall_status="HEALTHY",
                reason_code="SERVICE_HEARTBEAT_HEALTHY",
                detail="runner healthy",
            ),
        },
        reliability_states={
            (
                "paper-trader",
                "signal_client",
            ): ReliabilityState(
                service_name="paper-trader",
                component_name="signal_client",
                health_overall_status="DEGRADED",
                breaker_state="HALF_OPEN",
                failure_count=1,
                success_count=0,
                freshness_status="STALE",
                last_heartbeat_at=now,
                reason_code="SIGNAL_FETCH_FAILED",
                detail="Signal fetch failed once",
                updated_at=now,
            )
        },
        lag_states={
            (
                "features",
                "features",
            ): [
                FeatureLagSnapshot(
                    service_name="features",
                    component_name="features",
                    symbol="BTC/USD",
                    evaluated_at=now,
                    latest_raw_event_received_at=now,
                    latest_feature_interval_begin=now - timedelta(minutes=10),
                    latest_feature_as_of_time=now - timedelta(minutes=10),
                    time_lag_seconds=600.0,
                    processing_lag_seconds=600.0,
                    time_lag_reason_code="FEATURE_TIME_LAG_BREACH",
                    processing_lag_reason_code="FEATURE_PROCESSING_LAG_BREACH",
                    lag_breach=True,
                    health_overall_status="DEGRADED",
                    reason_code="FEATURE_LAG_BREACH",
                    detail="feature lag breach",
                )
            ]
        },
        latest_recovery_event=RecoveryEvent(
            service_name="features",
            component_name="BTC/USD",
            event_type="FEATURE_LAG_TRANSITION",
            event_time=now,
            reason_code="FEATURE_LAG_BREACH_DETECTED",
            health_overall_status="DEGRADED",
            freshness_status="STALE",
            detail="feature lag breach",
        ),
    )
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row(base_time=now - timedelta(minutes=5))),
        reliability_store=reliability_store,
    )

    response = client.get("/reliability/system")

    assert response.status_code == 200
    payload = response.json()
    assert payload["health_overall_status"] == "DEGRADED"
    assert payload["lag_breach_active"] is True
    assert payload["reason_codes"] == ["SIGNAL_FETCH_FAILED", "FEATURE_LAG_BREACH"]
    assert payload["services"][0]["component_name"] == "producer"
    assert payload["services"][0]["feed_freshness_status"] == "FRESH"
    assert payload["lag_by_symbol"][0]["symbol"] == "BTC/USD"
    assert payload["latest_recovery_event"]["reason_code"] == "FEATURE_LAG_BREACH_DETECTED"
    assert len(reliability_store.saved_system_snapshots) == 1


def test_predict_happy_path(tmp_path: Path) -> None:
    """`/predict` should return probabilities and class labels from the saved artifact."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))

    response = client.get("/predict", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "UP"
    assert payload["prob_up"] == 0.7
    assert payload["row_id"].startswith("BTC/USD|")
    assert payload["regime_label"] == "TREND_UP"
    assert payload["regime_run_id"] == "20260320T120000Z"
    assert payload["model_version"] == "model-0.70"
    assert payload["freshness_status"] == "FRESH"
    assert payload["health_overall_status"] == "HEALTHY"
    assert payload["prediction_explanation"]["available"] is True
    assert payload["prediction_explanation"]["method"] == "ONE_AT_A_TIME_REFERENCE_ABLATION"
    assert isinstance(payload["top_features"], list)


def test_predict_and_signal_include_m14_explainability_fields(tmp_path: Path) -> None:
    """`/predict` and `/signal` should expose additive M14 explanation fields."""
    row = _feature_row(
        realized_vol_12=0.03,
        momentum_3=0.08,
        macd_line_12_26=1.2,
    )
    database = FakeDatabase(
        row=row,
        reference_vector={
            "momentum_3": 0.02,
            "realized_vol_12": 0.04,
            "volume_zscore_12": 0.0,
        },
    )
    client = _build_client(
        tmp_path,
        database=database,
        artifact_path=_write_feature_aware_artifact(tmp_path),
    )

    predict_payload = client.get("/predict", params={"symbol": "BTC/USD"}).json()
    signal_payload = client.get("/signal", params={"symbol": "BTC/USD"}).json()

    assert predict_payload["model_version"] == "m3-20260321T120000Z"
    assert predict_payload["top_features"][0]["feature_name"] == "volume_zscore_12"
    assert predict_payload["prediction_explanation"]["available"] is True
    assert Path(
        predict_payload["prediction_explanation"]["reference_vector_path"]
    ).as_posix().endswith(
        "artifacts/explainability/m3-20260321T120000Z/reference.json"
    )
    assert signal_payload["model_version"] == "m3-20260321T120000Z"
    assert signal_payload["top_features"][0]["feature_name"] == "volume_zscore_12"
    assert signal_payload["prediction_explanation"]["available"] is True
    assert signal_payload["threshold_snapshot"]["buy_prob_up"] == 0.54
    assert signal_payload["threshold_snapshot"]["allow_new_long_entries"] is True
    assert signal_payload["regime_reason"]["reason_code"] == "REGIME_TREND_UP"
    assert signal_payload["signal_explanation"]["decision_source"] == "model"
    assert signal_payload["signal_explanation"]["available"] is True


def test_signal_accepts_exact_interval_begin_selector(tmp_path: Path) -> None:
    """The additive M4 contract should allow M5 to request an exact finalized candle."""
    row = _feature_row(base_time=datetime(2026, 3, 19, 22, 0, tzinfo=timezone.utc))
    database = FakeDatabase(row=row)
    client = _build_client(tmp_path, prob_up=0.7, database=database)

    response = client.get(
        "/signal",
        params={
            "symbol": "BTC/USD",
            "interval_begin": row["interval_begin"].isoformat().replace("+00:00", "Z"),
        },
    )

    assert response.status_code == 200
    assert database.last_interval_begin == row["interval_begin"]


def test_signal_buy_sell_and_hold(tmp_path: Path) -> None:
    """`/signal` should map probabilities onto BUY, SELL, and HOLD thresholds."""
    buy_client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))
    sell_client = _build_client(tmp_path, prob_up=0.3, database=FakeDatabase(row=_feature_row()))
    hold_client = _build_client(tmp_path, prob_up=0.5, database=FakeDatabase(row=_feature_row()))

    buy_payload = buy_client.get("/signal", params={"symbol": "BTC/USD"}).json()
    sell_payload = sell_client.get("/signal", params={"symbol": "BTC/USD"}).json()
    hold_payload = hold_client.get("/signal", params={"symbol": "BTC/USD"}).json()

    assert buy_payload["signal"] == "BUY"
    assert buy_payload["trade_allowed"] is True
    assert buy_payload["decision_source"] == "model"
    assert buy_payload["signal_status"] == "MODEL_SIGNAL"
    assert sell_payload["signal"] == "SELL"
    assert sell_payload["trade_allowed"] is True
    assert sell_payload["decision_source"] == "model"
    assert hold_payload["signal"] == "HOLD"
    assert hold_payload["trade_allowed"] is False
    assert hold_payload["signal_status"] == "MODEL_HOLD"


def test_regime_endpoint_returns_exact_row_regime_and_policy(tmp_path: Path) -> None:
    """`/regime` should expose the exact-row M8 regime plus the active M9 policy."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))

    response = client.get("/regime", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["regime_label"] == "TREND_UP"
    assert payload["regime_run_id"] == "20260320T120000Z"
    assert payload["trade_allowed"] is True
    assert payload["buy_prob_up"] == 0.54
    assert payload["sell_prob_up"] == 0.44


def test_signal_blocks_new_buy_entries_in_high_vol_regime(tmp_path: Path) -> None:
    """High-volatility rows should block new BUY entries while preserving the regime label."""
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(
            row=_feature_row(
                realized_vol_12=0.08,
                momentum_3=0.03,
                macd_line_12_26=1.2,
            )
        ),
    )

    response = client.get("/signal", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["signal"] == "HOLD"
    assert payload["regime_label"] == "HIGH_VOL"
    assert payload["trade_allowed"] is False


def test_signal_sell_remains_allowed_in_high_vol_regime(tmp_path: Path) -> None:
    """No-trade regimes must still allow SELL because SELL reduces long-only risk."""
    client = _build_client(
        tmp_path,
        prob_up=0.3,
        database=FakeDatabase(
            row=_feature_row(
                realized_vol_12=0.08,
                momentum_3=0.03,
                macd_line_12_26=1.2,
            )
        ),
    )

    response = client.get("/signal", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["signal"] == "SELL"
    assert payload["regime_label"] == "HIGH_VOL"
    assert payload["trade_allowed"] is True


def test_invalid_symbol_and_missing_row_behaviour(tmp_path: Path) -> None:
    """Invalid symbols should 400 and `/signal` should degrade to a reliability HOLD."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=None))

    invalid_response = client.get("/predict", params={"symbol": "DOGE/USD"})
    missing_response = client.get("/predict", params={"symbol": "BTC/USD"})
    missing_signal_response = client.get(
        "/signal",
        params={
            "symbol": "BTC/USD",
            "interval_begin": "2026-03-21T12:00:00Z",
        },
    )

    assert invalid_response.status_code == 400
    assert missing_response.status_code == 404
    assert missing_signal_response.status_code == 200
    missing_signal_payload = missing_signal_response.json()
    assert missing_signal_payload["signal"] == "HOLD"
    assert missing_signal_payload["decision_source"] == "reliability"
    assert missing_signal_payload["reason_code"] == "RELIABILITY_HOLD_MISSING_FEATURE_ROW"
    assert missing_signal_payload["row_id"] == "BTC/USD|2026-03-21T12:00:00Z"
    assert missing_signal_payload["top_features"] == []
    assert missing_signal_payload["prediction_explanation"]["available"] is False
    assert missing_signal_payload["signal_explanation"]["decision_source"] == "reliability"
    assert missing_signal_payload["threshold_snapshot"]["regime_label"] is None


def test_metrics_increment_after_requests(tmp_path: Path) -> None:
    """The in-memory metrics endpoint should count completed requests since startup."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))

    client.get("/health")
    client.get("/predict", params={"symbol": "BTC/USD"})
    metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200
    payload = metrics_response.json()
    assert payload["requests_total"] == 2
    assert payload["endpoint_counts"]["/health"] == 1
    assert payload["endpoint_counts"]["/predict"] == 1
    assert payload["health_overall_status"] == "HEALTHY"
    assert payload["reason_code"] == "HEALTH_HEALTHY"
    assert payload["freshness_summary"]["BTC/USD"]["freshness_status"] == "FRESH"


def test_predict_returns_503_when_database_is_unavailable(tmp_path: Path) -> None:
    """DB failures should propagate as 503s from the inference endpoints."""
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(fetch_error=DatabaseUnavailableError("db down")),
    )

    response = client.get("/predict", params={"symbol": "BTC/USD"})

    assert response.status_code == 503


def test_signal_degrades_to_reliability_hold_when_feature_row_is_stale(tmp_path: Path) -> None:
    """`/signal` should downgrade to a reliability HOLD for stale exact-row inputs."""
    base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
        minutes=20
    )
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row(base_time=base_time)),
    )

    response = client.get("/signal", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["signal"] == "HOLD"
    assert payload["decision_source"] == "reliability"
    assert payload["signal_status"] == "RELIABILITY_HOLD"
    assert payload["freshness_status"] == "STALE"
    assert payload["reason_code"] == "RELIABILITY_HOLD_STALE_FEATURE_ROW"
    assert payload["top_features"] == []
    assert payload["prediction_explanation"]["available"] is False
    assert payload["threshold_snapshot"]["regime_label"] == "TREND_UP"
    assert payload["regime_reason"]["reason_code"] == "REGIME_TREND_UP"
    assert payload["signal_explanation"]["decision_source"] == "reliability"


def test_freshness_endpoint_returns_exact_row_status(tmp_path: Path) -> None:
    """`/freshness` should surface the exact-row freshness state for a live symbol."""
    row = _feature_row()
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=row))

    response = client.get("/freshness", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["health_overall_status"] == "HEALTHY"
    assert payload["feature_freshness_status"] == "FRESH"
    assert payload["regime_freshness_status"] == "FRESH"
    assert payload["row_id"].startswith("BTC/USD|")


def test_freshness_endpoint_reports_missing_exact_row(tmp_path: Path) -> None:
    """`/freshness` should mark the requested candle stale when no exact row exists."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=None))

    response = client.get(
        "/freshness",
        params={
            "symbol": "BTC/USD",
            "interval_begin": "2026-03-21T12:00:00Z",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["freshness_status"] == "STALE"
    assert payload["reason_code"] == "FEATURE_ROW_MISSING"
    assert payload["row_id"] == "BTC/USD|2026-03-21T12:00:00Z"


# ---------------------------------------------------------------------------
# M20 ensemble fields on API endpoints
# ---------------------------------------------------------------------------


def test_health_includes_ensemble_fields(tmp_path: Path) -> None:
    """/health should include M20 ensemble status fields."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "ensemble_status" in payload
    assert "ensemble_candidate_count" in payload
    assert "ensemble_roster_status" in payload
    assert "ensemble_roster_reason_codes" in payload
    assert payload["ensemble_status"] == "DISABLED"
    assert payload["ensemble_candidate_count"] == 0


def test_predict_includes_ensemble_fields(tmp_path: Path) -> None:
    """/predict should include M20 ensemble fields in fallback mode."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))
    response = client.get("/predict", params={"symbol": "BTC/USD"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ensemble_active"] is False
    assert payload["ensemble_candidate_count"] == 0
    assert payload["ensemble_fallback_reason"] == "ENSEMBLE_FALLBACK_DISABLED"
    assert payload["ensemble_profile_id"] is None
    assert payload["ensemble_roster_status"] is None


def test_signal_includes_ensemble_fields(tmp_path: Path) -> None:
    """/signal should include M20 ensemble fields in fallback mode."""
    client = _build_client(tmp_path, prob_up=0.7, database=FakeDatabase(row=_feature_row()))
    response = client.get("/signal", params={"symbol": "BTC/USD"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ensemble_active"] is False
    assert payload["ensemble_candidate_count"] == 0
    assert payload["ensemble_fallback_reason"] == "ENSEMBLE_FALLBACK_DISABLED"
    assert payload["ensemble_profile_id"] is None
    assert payload["ensemble_roster_status"] is None


def test_health_reports_real_active_ensemble_runtime(tmp_path: Path, monkeypatch) -> None:
    """/health should report ACTIVE and the real usable candidate count when ensemble runs."""
    monkeypatch.setattr(
        "app.training.registry.default_registry_root",
        lambda: tmp_path / "artifacts" / "registry",
    )
    client = _build_client(
        tmp_path,
        prob_up=0.4,
        database=FakeDatabase(row=_feature_row()),
        ensemble_service=_build_active_ensemble_service(tmp_path),
    )

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ensemble_status"] == "ACTIVE"
    assert payload["ensemble_profile_id"] == "ens-profile-active-1"
    assert payload["ensemble_candidate_count"] == 2
    assert payload["ensemble_roster_status"] == "ACTIVE_WEAK"
    assert "RANGE_SPECIALIST_ROLE_MISSING" in payload["ensemble_roster_reason_codes"]
    assert payload["model_name"] == "dynamic_ensemble"
    assert payload["model_artifact_path"] is None


def test_predict_returns_ensemble_backed_output_when_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """/predict should return ensemble probabilities, truthful explainability, and context."""
    monkeypatch.setattr(
        "app.training.registry.default_registry_root",
        lambda: tmp_path / "artifacts" / "registry",
    )
    client = _build_client(
        tmp_path,
        prob_up=0.4,
        database=FakeDatabase(row=_feature_row()),
        ensemble_service=_build_active_ensemble_service(tmp_path),
    )

    response = client.get("/predict", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ensemble_active"] is True
    assert payload["ensemble_profile_id"] == "ens-profile-active-1"
    assert payload["ensemble_candidate_count"] == 2
    assert payload["model_name"] == "dynamic_ensemble"
    assert payload["model_version"] == "ensemble_profile:ens-profile-active-1"
    assert payload["prob_up"] == pytest.approx(0.675)
    assert payload["prob_down"] == pytest.approx(0.325)
    assert payload["confidence"] == pytest.approx(0.675)
    assert payload["ensemble_effective_confidence"] == pytest.approx(0.675)
    assert payload["ensemble_roster_status"] == "ACTIVE_WEAK"
    assert "RANGE_SPECIALIST_ROLE_MISSING" in payload["ensemble_roster_reason_codes"]
    assert payload["prediction_explanation"]["available"] is False
    assert payload["prediction_explanation"]["method"] == "ensemble_pending"
    assert (
        payload["prediction_explanation"]["reason_code"]
        == "ENSEMBLE_EXPLAINABILITY_UNAVAILABLE"
    )
    assert payload["top_features"] == []
    assert payload["ensemble"]["ensemble_profile_id"] == "ens-profile-active-1"
    assert payload["ensemble"]["candidate_count"] == 2
    assert payload["ensemble"]["roster_status"] == "ACTIVE_WEAK"


def test_signal_uses_ensemble_effective_confidence_when_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """/signal should feed ensemble effective_confidence into M19 adaptation."""
    monkeypatch.setattr(
        "app.training.registry.default_registry_root",
        lambda: tmp_path / "artifacts" / "registry",
    )
    adaptation_service = RecordingAdaptationService()
    client = _build_client(
        tmp_path,
        prob_up=0.4,
        database=FakeDatabase(row=_feature_row()),
        adaptation_service=adaptation_service,
        ensemble_service=_build_active_ensemble_service(tmp_path),
    )

    response = client.get("/signal", params={"symbol": "BTC/USD"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ensemble_active"] is True
    assert payload["ensemble_profile_id"] == "ens-profile-active-1"
    assert payload["ensemble_candidate_count"] == 2
    assert payload["model_name"] == "dynamic_ensemble"
    assert payload["model_version"] == "ensemble_profile:ens-profile-active-1"
    assert payload["ensemble_effective_confidence"] == pytest.approx(0.675)
    assert payload["ensemble_roster_status"] == "ACTIVE_WEAK"
    assert adaptation_service.last_kwargs is not None
    assert adaptation_service.last_kwargs["confidence"] == pytest.approx(0.675)
    assert payload["ensemble"]["agreement_band"] == "HIGH"


# ---------------------------------------------------------------------------
# M21 continual-learning read surfaces on API endpoints
# ---------------------------------------------------------------------------


class ActiveFrozenContinualLearningService(NullContinualLearningService):
    """Continual-learning stub with an active profile and health-gate freeze behavior."""

    async def resolve_runtime_context(self, **kwargs) -> ContinualLearningContextPayload:
        frozen = (
            kwargs.get("health_overall_status") not in {None, "HEALTHY"}
            or kwargs.get("freshness_status") not in {None, "FRESH"}
        )
        reason_codes = ["ACTIVE_PROFILE_PRESENT"]
        if frozen:
            reason_codes.append("CONTINUAL_LEARNING_FROZEN_BY_HEALTH_GATE")
        return ContinualLearningContextPayload(
            enabled=True,
            active_profile_id="cl-profile-1",
            candidate_type="CALIBRATION_OVERLAY",
            promotion_stage="LIVE_ELIGIBLE",
            live_eligible=True,
            baseline_target_type="MODEL_VERSION",
            baseline_target_id="m20-live",
            source_experiment_id="cl-exp-1",
            drift_cap_status="WATCH",
            latest_promotion_decision="HOLD",
            frozen_by_health_gate=frozen,
            reason_codes=reason_codes,
        )

    async def summary(self, **_kwargs) -> ContinualLearningSummaryResponse:
        return ContinualLearningSummaryResponse(
            enabled=True,
            active_profile_count=1,
            active_profile_id="cl-profile-1",
            continual_learning_status="ACTIVE",
            evidence_backed=True,
            active_candidate_type="CALIBRATION_OVERLAY",
            latest_drift_cap_status="WATCH",
            latest_promotion_decision="HOLD",
            reason_codes=["ACTIVE_PROFILE_PRESENT", "DRIFT_CAP_EVIDENCE_PRESENT"],
        )


def test_health_includes_continual_learning_fields(tmp_path: Path) -> None:
    """/health should expose additive M21 continual-learning summary fields."""
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row()),
        continual_learning_service=ActiveFrozenContinualLearningService(),
    )
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["active_continual_learning_profile_id"] == "cl-profile-1"
    assert payload["continual_learning_status"] == "ACTIVE"
    assert payload["continual_learning_drift_cap_status"] == "WATCH"
    assert payload["continual_learning_evidence_backed"] is True


def test_predict_includes_continual_learning_fields_without_value_changes(tmp_path: Path) -> None:
    """/predict should add M21 context fields while preserving probability outputs."""
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row()),
        continual_learning_service=ActiveFrozenContinualLearningService(),
    )
    response = client.get("/predict", params={"symbol": "BTC/USD"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["prob_up"] == pytest.approx(0.7)
    assert payload["prob_down"] == pytest.approx(0.3)
    assert payload["predicted_class"] == "UP"
    assert payload["continual_learning_profile_id"] == "cl-profile-1"
    assert payload["continual_learning_status"] == "ACTIVE"
    assert payload["continual_learning_evidence_backed"] is True
    assert payload["continual_learning_frozen"] is False
    assert payload["continual_learning"]["baseline_target_id"] == "m20-live"
    assert payload["continual_learning"]["promotion_stage"] == "LIVE_ELIGIBLE"


def test_signal_marks_continual_learning_frozen_under_degraded_health(tmp_path: Path) -> None:
    """/signal should expose frozen continual-learning context when freshness is degraded."""
    base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
        minutes=20
    )
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row(base_time=base_time)),
        continual_learning_service=ActiveFrozenContinualLearningService(),
    )

    response = client.get("/signal", params={"symbol": "BTC/USD"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["signal"] == "HOLD"
    assert payload["decision_source"] == "reliability"
    assert payload["continual_learning_profile_id"] == "cl-profile-1"
    assert payload["continual_learning_status"] == "ACTIVE"
    assert payload["continual_learning_evidence_backed"] is True
    assert payload["continual_learning_frozen"] is True
    assert "CONTINUAL_LEARNING_FROZEN_BY_HEALTH_GATE" in payload["continual_learning"][
        "reason_codes"
    ]


def test_continual_learning_read_only_endpoints_return_payloads(tmp_path: Path) -> None:
    """/continual-learning/* endpoints should return additive read-only payloads."""
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row()),
        continual_learning_service=NullContinualLearningService(),
    )

    summary = client.get("/continual-learning/summary")
    experiments = client.get("/continual-learning/experiments", params={"limit": 10})
    profiles = client.get("/continual-learning/profiles", params={"limit": 10})
    drift_caps = client.get(
        "/continual-learning/drift-caps",
        params={"execution_mode": "paper", "symbol": "BTC/USD", "regime_label": "TREND_UP"},
    )
    promotions = client.get("/continual-learning/promotions", params={"limit": 10})
    events = client.get("/continual-learning/events", params={"limit": 10})

    assert summary.status_code == 200
    assert experiments.status_code == 200
    assert profiles.status_code == 200
    assert drift_caps.status_code == 200
    assert promotions.status_code == 200
    assert events.status_code == 200
    assert summary.json()["continual_learning_status"] == "IDLE"
    assert summary.json()["evidence_backed"] is False
    assert profiles.json()["items"][0]["profile_id"] == "cl-profile-1"
    assert drift_caps.json()["items"][0]["status"] == "WATCH"
    assert promotions.json()["items"][0]["decision"] == "HOLD"
    assert events.json()["items"][0]["event_type"] == "PROFILE_ACTIVE"


def test_continual_learning_workflow_post_endpoints_return_guarded_results(
    tmp_path: Path,
) -> None:
    operator_key = "test-operator-key"
    workflow_service = WorkflowContinualLearningService()
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row()),
        continual_learning_service=workflow_service,
        operator_api_key=operator_key,
    )

    promote_response = client.post(
        "/continual-learning/promotions/promote-profile",
        headers={"X-StreamAlpha-Operator-Key": operator_key},
        json={
            "decision_id": "decision-promote-1",
            "profile_id": "cl-profile-1",
            "requested_promotion_stage": "PAPER_APPROVED",
            "summary_text": "promote after review",
            "reason_codes": ["OPERATOR_REVIEWED_EVIDENCE"],
            "operator_confirmed": True,
        },
    )
    rollback_response = client.post(
        "/continual-learning/promotions/rollback-active-profile",
        headers={"X-StreamAlpha-Operator-Key": operator_key},
        json={
            "decision_id": "decision-rollback-1",
            "execution_mode": "paper",
            "symbol": "BTC/USD",
            "regime_label": "TREND_UP",
            "summary_text": "rollback after review",
            "operator_confirmed": True,
        },
    )

    assert promote_response.status_code == 200
    assert rollback_response.status_code == 200
    assert promote_response.json()["decision"] == "PROMOTE"
    assert promote_response.json()["blocked"] is False
    assert rollback_response.json()["decision"] == "ROLLBACK"
    assert rollback_response.json()["blocked"] is False
    assert workflow_service.promote_requests[0]["health_overall_status"] == "HEALTHY"
    assert workflow_service.promote_requests[0]["freshness_status"] == "FRESH"
    assert workflow_service.rollback_requests[0]["health_overall_status"] == "HEALTHY"
    assert workflow_service.rollback_requests[0]["freshness_status"] == "FRESH"


def test_continual_learning_workflow_post_endpoint_denies_missing_operator_key(
    tmp_path: Path,
) -> None:
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row()),
        continual_learning_service=WorkflowContinualLearningService(),
        operator_api_key="test-operator-key",
    )

    response = client.post(
        "/continual-learning/promotions/promote-profile",
        json={
            "decision_id": "decision-promote-1",
            "profile_id": "cl-profile-1",
            "requested_promotion_stage": "PAPER_APPROVED",
            "summary_text": "promote after review",
            "reason_codes": ["OPERATOR_REVIEWED_EVIDENCE"],
            "operator_confirmed": True,
        },
    )

    assert response.status_code == 403


def test_continual_learning_workflow_post_endpoint_denies_wrong_operator_key(
    tmp_path: Path,
) -> None:
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row()),
        continual_learning_service=WorkflowContinualLearningService(),
        operator_api_key="test-operator-key",
    )

    response = client.post(
        "/continual-learning/promotions/promote-profile",
        headers={"X-StreamAlpha-Operator-Key": "wrong-key"},
        json={
            "decision_id": "decision-promote-1",
            "profile_id": "cl-profile-1",
            "requested_promotion_stage": "PAPER_APPROVED",
            "summary_text": "promote after review",
            "reason_codes": ["OPERATOR_REVIEWED_EVIDENCE"],
            "operator_confirmed": True,
        },
    )

    assert response.status_code == 403


def test_continual_learning_workflow_post_endpoint_denies_when_key_unset(
    tmp_path: Path,
) -> None:
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row()),
        continual_learning_service=WorkflowContinualLearningService(),
        operator_api_key="",
    )

    response = client.post(
        "/continual-learning/promotions/promote-profile",
        headers={"X-StreamAlpha-Operator-Key": "test-operator-key"},
        json={
            "decision_id": "decision-promote-1",
            "profile_id": "cl-profile-1",
            "requested_promotion_stage": "PAPER_APPROVED",
            "summary_text": "promote after review",
            "reason_codes": ["OPERATOR_REVIEWED_EVIDENCE"],
            "operator_confirmed": True,
        },
    )

    assert response.status_code == 403


def test_health_remains_available_without_operator_key(tmp_path: Path) -> None:
    client = _build_client(
        tmp_path,
        prob_up=0.7,
        database=FakeDatabase(row=_feature_row()),
        operator_api_key="",
    )

    response = client.get("/health")

    assert response.status_code == 200
