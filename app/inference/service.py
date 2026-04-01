"""Core inference service logic for the Stream Alpha M4 API."""

# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib

from app.adaptation.config import default_adaptation_config_path, load_adaptation_config
from app.adaptation.service import AdaptationService
from app.alerting.config import (
    AlertingConfig,
    default_alerting_config_path,
    load_alerting_config,
)
from app.alerting.repository import OperationalAlertRepository
from app.common.config import Settings
from app.common.time import parse_rfc3339, to_rfc3339, utc_now
from app.continual_learning.service import ContinualLearningService
from app.continual_learning.schemas import (
    ContinualLearningContextPayload,
    ContinualLearningDriftCapsResponse,
    ContinualLearningEventsResponse,
    ContinualLearningExperimentsResponse,
    ContinualLearningProfilesResponse,
    ContinualLearningPromoteProfileRequest,
    ContinualLearningPromotionsResponse,
    ContinualLearningRollbackRequest,
    ContinualLearningSummaryResponse,
    ContinualLearningWorkflowResponse,
)
from app.ensemble.config import default_ensemble_config_path, load_ensemble_config
from app.ensemble.schemas import EnsembleResult, ParticipatingCandidate
from app.ensemble.service import (
    ENSEMBLE_FALLBACK_ALL_SCORE_FAILED,
    ENSEMBLE_FALLBACK_DISABLED,
    ENSEMBLE_FALLBACK_INVALID_PROFILE,
    ENSEMBLE_FALLBACK_NO_CANDIDATES,
    ENSEMBLE_FALLBACK_NO_PROFILE,
    EnsembleService,
    build_ensemble_fallback_result,
)
from app.explainability.config import (
    default_explainability_config_path,
    load_explainability_config,
)
from app.explainability.schemas import PredictionExplanation, TopFeatureContribution
from app.explainability.service import ExplainabilityService, build_regime_reason
from app.inference.db import DatabaseUnavailableError, InferenceDatabase
from app.inference.schemas import (
    DailyOperationsSummaryResponse,
    FreshnessResponse,
    HealthResponse,
    LatencyStatsResponse,
    MetricsResponse,
    OperationalAlertEventResponse,
    OperationalAlertStateResponse,
    PredictionResponse,
    RegimeResponse,
    SignalResponse,
    StartupSafetyReportResponse,
    ThresholdsResponse,
)
from app.reliability.artifacts import write_json_artifact
from app.reliability.config import default_reliability_config_path, load_reliability_config
from app.reliability.schemas import (
    FreshnessStatus,
    ReliabilityHealthSnapshot,
    ServiceHeartbeat,
    SystemReliabilitySnapshot,
    SymbolFreshnessSnapshot,
)
from app.reliability.service import (
    FEATURE_ROW_MISSING,
    HEALTH_HEALTHY,
    REGIME_ROW_INCOMPATIBLE,
    RELIABILITY_HOLD_INPUTS_MISSING,
    RELIABILITY_HOLD_MISSING_FEATURE_ROW,
    RELIABILITY_HOLD_STALE_FEATURE_ROW,
    SERVICE_HEARTBEAT_DEGRADED,
    SERVICE_HEARTBEAT_HEALTHY,
    aggregate_health_status,
    aggregate_system_reliability,
    build_service_health_snapshot,
    build_signal_client_health_snapshot,
    evaluate_feature_freshness,
    evaluate_regime_freshness,
)
from app.reliability.store import ReliabilityStore
from app.regime.live import LiveRegimeRuntime, ResolvedRegime, load_live_regime_runtime
from app.runtime.config import build_runtime_metadata, resolve_trading_config_path
from app.trading.config import load_paper_trading_config
from app.trading.repository import TradingRepository
from app.training.registry import load_registry_entry, resolve_inference_model_metadata


class InvalidSymbolError(ValueError):
    """Raised when a request uses a symbol outside the configured Kraken set."""


class ArtifactSchemaMismatchError(RuntimeError):
    """Raised when the saved model artifact cannot score the DB row safely."""


@dataclass(frozen=True, slots=True)
class LoadedModelArtifact:  # pylint: disable=too-many-instance-attributes
    """Validated saved M3 model artifact for direct online reuse."""

    model_name: str
    trained_at: str
    model_version: str
    model_version_source: str
    feature_columns: tuple[str, ...]
    expanded_feature_names: tuple[str, ...]
    model_artifact_path: str
    model: Any


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Internal prediction build bundle that preserves resolved regime details."""

    prediction: PredictionResponse
    resolved_regime: ResolvedRegime
    ensemble_result: EnsembleResult = field(default_factory=EnsembleResult)


@dataclass(frozen=True, slots=True)
class EnsembleRuntimeState:
    """Resolved ensemble runtime state for inference and health surfaces."""

    result: EnsembleResult = field(default_factory=EnsembleResult)
    status: str = "FALLBACK"


@dataclass(frozen=True, slots=True)
class FreshnessEvaluation:  # pylint: disable=too-many-instance-attributes
    """Exact-row freshness evaluation used across M4 endpoints."""

    symbol: str
    row_id: str | None
    interval_begin: str | None
    as_of_time: str | None
    health_overall_status: str
    freshness_status: str
    reason_code: str
    feature_freshness: FreshnessStatus
    regime_freshness: FreshnessStatus
    regime_label: str | None = None
    regime_run_id: str | None = None
    detail: str | None = None


@dataclass(slots=True)
class LatencyStats:
    """Simple in-memory latency counters."""

    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def record(self, latency_ms: float) -> None:
        """Add one observed request latency."""
        self.count += 1
        self.total_ms += latency_ms
        self.max_ms = max(self.max_ms, latency_ms)

    def as_response(self) -> LatencyStatsResponse:
        """Return the JSON schema payload."""
        average = 0.0 if self.count == 0 else self.total_ms / self.count
        return LatencyStatsResponse(
            count=self.count,
            avg=average,
            max=self.max_ms,
        )


@dataclass(slots=True)
class MetricsState:
    """Request counters maintained since service startup."""

    started_at: Any = field(default_factory=utc_now)
    requests_total: int = 0
    errors_total: int = 0
    endpoint_counts: dict[str, int] = field(default_factory=dict)
    latency: LatencyStats = field(default_factory=LatencyStats)

    def record(self, *, path: str, status_code: int, latency_ms: float) -> None:
        """Record one completed request."""
        self.requests_total += 1
        self.endpoint_counts[path] = self.endpoint_counts.get(path, 0) + 1
        if status_code >= 400:
            self.errors_total += 1
        self.latency.record(latency_ms)


def load_model_artifact(
    model_path: str,
    *,
    registry_root: Path | None = None,
) -> LoadedModelArtifact:
    """Load and validate the saved M3 model artifact."""
    metadata = resolve_inference_model_metadata(
        model_path,
        registry_root=registry_root,
    )
    artifact_path = Path(metadata["model_artifact_path"]).resolve()

    payload = joblib.load(artifact_path)
    if not isinstance(payload, dict):
        raise ValueError("Model artifact must deserialize into a dictionary payload")

    required_keys = {
        "model_name",
        "trained_at",
        "feature_columns",
        "expanded_feature_names",
        "model",
    }
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        raise ValueError(f"Model artifact is missing required keys: {missing_keys}")

    feature_columns = tuple(str(column) for column in payload["feature_columns"])
    expanded_feature_names = tuple(str(name) for name in payload["expanded_feature_names"])
    model = payload["model"]
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model artifact must expose predict_proba")

    return LoadedModelArtifact(
        model_name=str(payload["model_name"]),
        trained_at=str(payload["trained_at"]),
        model_version=metadata["model_version"],
        model_version_source=metadata["model_version_source"],
        feature_columns=feature_columns,
        expanded_feature_names=expanded_feature_names,
        model_artifact_path=str(artifact_path),
        model=model,
    )


class InferenceService:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Serve online predictions from the latest canonical feature row."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        settings: Settings,
        *,
        database: InferenceDatabase | None = None,
        model_artifact: LoadedModelArtifact | None = None,
        regime_runtime: LiveRegimeRuntime | None = None,
        reliability_store: ReliabilityStore | None = None,
        alerting_config: AlertingConfig | None = None,
        alert_repository: OperationalAlertRepository | Any | None = None,
        adaptation_service: AdaptationService | None = None,
        ensemble_service: EnsembleService | None = None,
        continual_learning_service: ContinualLearningService | None = None,
    ) -> None:
        self.settings = settings
        self.started_at = utc_now()
        self.metrics = MetricsState(started_at=self.started_at)
        self.reliability_config = load_reliability_config(default_reliability_config_path())
        self.alerting_config = alerting_config or load_alerting_config(
            default_alerting_config_path()
        )
        self.explainability_config = load_explainability_config(
            default_explainability_config_path()
        )
        self.database = database or InferenceDatabase(
            settings.postgres.dsn,
            settings.tables.feature_ohlc,
        )
        self.reliability_store = reliability_store or ReliabilityStore(
            settings.postgres.dsn
        )
        self.alert_repository = alert_repository or OperationalAlertRepository(
            settings.postgres.dsn
        )
        self.model_artifact = model_artifact or load_model_artifact(
            settings.inference.model_path,
        )
        self.regime_runtime = regime_runtime or load_live_regime_runtime(
            thresholds_path=settings.inference.regime_thresholds_path,
            signal_policy_path=settings.inference.regime_signal_policy_path,
        )
        self.explainability_service = ExplainabilityService(self.explainability_config)
        self.trading_config = load_paper_trading_config(resolve_trading_config_path())
        self.adaptation_service = adaptation_service or AdaptationService(
            repository=TradingRepository(settings.postgres.dsn, settings.tables.feature_ohlc),
            config=load_adaptation_config(default_adaptation_config_path()),
        )
        self.ensemble_service = ensemble_service or EnsembleService(
            config=load_ensemble_config(default_ensemble_config_path()),
            repository=TradingRepository(settings.postgres.dsn, settings.tables.feature_ohlc),
        )
        self.continual_learning_service = (
            continual_learning_service
            or ContinualLearningService(
                repository=TradingRepository(
                    settings.postgres.dsn,
                    settings.tables.feature_ohlc,
                )
            )
        )
        self._symbols = set(settings.kraken.symbols)
        self._last_heartbeat_at: datetime | None = None
        self._validate_thresholds()
        self.regime_runtime.validate_runtime_compatibility(
            source_table=settings.tables.feature_ohlc,
            source_exchange="kraken",
            interval_minutes=settings.kraken.ohlc_interval_minutes,
            symbols=settings.kraken.symbols,
        )
    # pylint: enable=too-many-arguments

    async def startup(self) -> None:
        """Open the read-only database pool for serving."""
        try:
            await self.database.connect()
        except Exception:  # pylint: disable=broad-exception-caught
            return
        try:
            await self.reliability_store.connect()
        except Exception:  # pylint: disable=broad-exception-caught
            return
        try:
            await self.alert_repository.connect()
        except Exception:  # pylint: disable=broad-exception-caught
            return
        try:
            await self.adaptation_service.startup()
        except Exception:  # pylint: disable=broad-exception-caught
            return
        try:
            await self.ensemble_service.startup()
        except Exception:  # pylint: disable=broad-exception-caught
            return
        try:
            await self.continual_learning_service.startup()
        except Exception:  # pylint: disable=broad-exception-caught
            return

    async def shutdown(self) -> None:
        """Close the database pool."""
        await self.database.close()
        try:
            await self.reliability_store.close()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        try:
            await self.alert_repository.close()
        except Exception:  # pylint: disable=broad-exception-caught
            return
        try:
            await self.adaptation_service.shutdown()
        except Exception:  # pylint: disable=broad-exception-caught
            return
        try:
            await self.ensemble_service.shutdown()
        except Exception:  # pylint: disable=broad-exception-caught
            return
        try:
            await self.continual_learning_service.shutdown()
        except Exception:  # pylint: disable=broad-exception-caught
            return

    async def _resolve_runtime_ensemble_state(  # pylint: disable=too-many-return-statements,too-many-locals
        self,
        *,
        row: dict[str, Any],
        resolved_regime: ResolvedRegime,
    ) -> EnsembleRuntimeState:
        """Load the active profile, score real candidates, and resolve ensemble runtime."""
        if not self.ensemble_service.config.enabled:
            return EnsembleRuntimeState(
                result=build_ensemble_fallback_result(ENSEMBLE_FALLBACK_DISABLED),
                status="DISABLED",
            )

        try:
            active_profile = await self.ensemble_service.load_active_profile(
                execution_mode=self.trading_config.execution.mode,
                symbol=str(row["symbol"]),
                regime_label=resolved_regime.regime_label,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return EnsembleRuntimeState(
                result=build_ensemble_fallback_result(ENSEMBLE_FALLBACK_NO_PROFILE),
                status="UNAVAILABLE",
            )

        if active_profile is None:
            return EnsembleRuntimeState(
                result=build_ensemble_fallback_result(ENSEMBLE_FALLBACK_NO_PROFILE),
                status="FALLBACK",
            )

        try:
            roster = self.ensemble_service.parse_candidate_roster(active_profile)
        except Exception:  # pylint: disable=broad-exception-caught
            return EnsembleRuntimeState(
                result=build_ensemble_fallback_result(ENSEMBLE_FALLBACK_INVALID_PROFILE),
                status="FALLBACK",
            )

        candidate_scores: list[dict[str, Any]] = []
        failed_candidates: list[ParticipatingCandidate] = []

        for roster_entry in roster:
            if not roster_entry.enabled:
                continue
            if (
                roster_entry.scope_regimes
                and resolved_regime.regime_label not in roster_entry.scope_regimes
            ):
                continue
            try:
                registry_entry = load_registry_entry(roster_entry.model_version)
                loaded_artifact = load_model_artifact(
                    str(registry_entry["model_artifact_path"]),
                )
                if (
                    roster_entry.expected_model_name is not None
                    and loaded_artifact.model_name != roster_entry.expected_model_name
                ):
                    raise ValueError(
                        "Candidate model_name did not match expected_model_name"
                    )
                candidate_feature_input = self._build_feature_input(
                    row,
                    model_artifact=loaded_artifact,
                )
                candidate_prob_down, candidate_prob_up = self._score_loaded_model_artifact(
                    model_artifact=loaded_artifact,
                    feature_input=candidate_feature_input,
                )
                candidate_scores.append(
                    {
                        "candidate_id": roster_entry.candidate_id,
                        "candidate_role": roster_entry.candidate_role,
                        "model_name": loaded_artifact.model_name,
                        "model_version": loaded_artifact.model_version,
                        "scope_regimes": list(roster_entry.scope_regimes),
                        "prob_up": candidate_prob_up,
                        "prob_down": candidate_prob_down,
                        "predicted_class": (
                            "UP" if candidate_prob_up >= candidate_prob_down else "DOWN"
                        ),
                    }
                )
            except Exception:  # pylint: disable=broad-exception-caught
                failed_candidates.append(
                    ParticipatingCandidate(
                        candidate_id=roster_entry.candidate_id,
                        candidate_role=roster_entry.candidate_role,
                        model_name=(
                            roster_entry.expected_model_name
                            if roster_entry.expected_model_name is not None
                            else roster_entry.candidate_id
                        ),
                        model_version=roster_entry.model_version,
                        participation_status="SCORE_FAILED",
                        scope_regimes=list(roster_entry.scope_regimes),
                    )
                )

        if not candidate_scores and failed_candidates:
            return EnsembleRuntimeState(
                result=build_ensemble_fallback_result(
                    ENSEMBLE_FALLBACK_ALL_SCORE_FAILED,
                    participating_candidates=tuple(failed_candidates),
                ),
                status="FALLBACK",
            )
        if not candidate_scores:
            return EnsembleRuntimeState(
                result=build_ensemble_fallback_result(ENSEMBLE_FALLBACK_NO_CANDIDATES),
                status="FALLBACK",
            )

        result = await self.ensemble_service.resolve_ensemble(
            regime_label=resolved_regime.regime_label,
            candidate_scores=candidate_scores,
            active_profile=active_profile,
            failed_candidates=failed_candidates,
        )
        roster_status, roster_reason_codes = self._resolve_ensemble_roster_truth(
            active_profile
        )
        return EnsembleRuntimeState(
            result=EnsembleResult(
                active=result.active,
                ensemble_profile_id=result.ensemble_profile_id,
                approval_stage=result.approval_stage,
                ensemble_prob_up=result.ensemble_prob_up,
                ensemble_prob_down=result.ensemble_prob_down,
                ensemble_predicted_class=result.ensemble_predicted_class,
                raw_ensemble_confidence=result.raw_ensemble_confidence,
                effective_confidence=result.effective_confidence,
                agreement_band=result.agreement_band,
                vote_agreement_ratio=result.vote_agreement_ratio,
                probability_spread=result.probability_spread,
                agreement_multiplier=result.agreement_multiplier,
                candidate_count=result.candidate_count,
                roster_status=roster_status,
                roster_reason_codes=roster_reason_codes,
                participating_candidates=result.participating_candidates,
                weighting_reason_codes=result.weighting_reason_codes,
                fallback_reason=result.fallback_reason,
            ),
            status=(
                "ACTIVE" if result.active and result.candidate_count > 0 else "FALLBACK"
            ),
        )

    async def _resolve_health_ensemble_state(  # pylint: disable=too-many-return-statements
        self,
    ) -> EnsembleRuntimeState:
        """Resolve truthful ensemble health from runtime rows and active profile state."""
        if not self.ensemble_service.config.enabled:
            return EnsembleRuntimeState(
                result=build_ensemble_fallback_result(ENSEMBLE_FALLBACK_DISABLED),
                status="DISABLED",
            )

        fallback_state: EnsembleRuntimeState | None = None
        for symbol in self.settings.kraken.symbols:
            row = await self.latest_feature_row(symbol)
            if row is None:
                continue
            resolved_regime = self._try_resolve_regime(row)
            if resolved_regime is None:
                continue
            runtime_state = await self._resolve_runtime_ensemble_state(
                row=row,
                resolved_regime=resolved_regime,
            )
            if runtime_state.status == "ACTIVE":
                return runtime_state
            if fallback_state is None:
                fallback_state = runtime_state

        if fallback_state is not None:
            return fallback_state

        try:
            active_profile = await self.ensemble_service.load_active_profile(
                execution_mode=self.trading_config.execution.mode,
                symbol="ALL",
                regime_label="ALL",
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return EnsembleRuntimeState(
                result=build_ensemble_fallback_result(ENSEMBLE_FALLBACK_NO_PROFILE),
                status="UNAVAILABLE",
            )

        if active_profile is None:
            return EnsembleRuntimeState(
                result=build_ensemble_fallback_result(ENSEMBLE_FALLBACK_NO_PROFILE),
                status="FALLBACK",
            )

        try:
            roster = self.ensemble_service.parse_candidate_roster(active_profile)
        except Exception:  # pylint: disable=broad-exception-caught
            return EnsembleRuntimeState(
                result=build_ensemble_fallback_result(ENSEMBLE_FALLBACK_INVALID_PROFILE),
                status="FALLBACK",
            )

        candidate_count = sum(1 for item in roster if item.enabled)
        roster_status, roster_reason_codes = self._resolve_ensemble_roster_truth(
            active_profile
        )
        return EnsembleRuntimeState(
            result=EnsembleResult(
                active=candidate_count > 0,
                ensemble_profile_id=active_profile.profile_id,
                approval_stage=active_profile.approval_stage,
                candidate_count=candidate_count,
                roster_status=roster_status,
                roster_reason_codes=roster_reason_codes,
            ),
            status="ACTIVE" if candidate_count > 0 else "FALLBACK",
        )

    def _score_loaded_model_artifact(
        self,
        *,
        model_artifact: LoadedModelArtifact,
        feature_input: dict[str, Any],
    ) -> tuple[float, float]:
        """Score one loaded binary classifier artifact against one feature row."""
        probabilities = model_artifact.model.predict_proba([feature_input])
        if len(probabilities) != 1 or len(probabilities[0]) != 2:
            raise ArtifactSchemaMismatchError(
                "Model predict_proba must return binary probabilities",
            )
        return float(probabilities[0][0]), float(probabilities[0][1])

    def _build_ensemble_pending_explainability(self) -> PredictionExplanation:
        """Return truthful Packet 1 explainability when ensemble is active."""
        return PredictionExplanation(
            method="ensemble_pending",
            available=False,
            reason_code="ENSEMBLE_EXPLAINABILITY_UNAVAILABLE",
            summary_text=(
                "Ensemble prediction is active. Aggregate candidate-level explainability "
                "is not yet implemented in Packet 1."
            ),
            explainable_feature_count=0,
            top_feature_count=0,
        )

    def _resolve_top_level_model_identity(
        self,
        *,
        ensemble_result: EnsembleResult,
    ) -> tuple[str, str]:
        """Return the truthful top-level model identity for API and trace surfaces."""
        if ensemble_result.active and ensemble_result.ensemble_profile_id is not None:
            return (
                "dynamic_ensemble",
                f"ensemble_profile:{ensemble_result.ensemble_profile_id}",
            )
        return self.model_artifact.model_name, self.model_artifact.model_version

    def _resolve_health_model_fields(
        self,
        *,
        ensemble_result: EnsembleResult,
    ) -> tuple[str | None, str | None]:
        """Return truthful top-level health identity fields for model name and path."""
        if self.model_artifact is None:
            return None, None
        if ensemble_result.active and ensemble_result.ensemble_profile_id is not None:
            return "dynamic_ensemble", None
        return self.model_artifact.model_name, self.model_artifact.model_artifact_path

    def _resolve_ensemble_roster_truth(
        self,
        active_profile,
    ) -> tuple[str | None, tuple[str, ...]]:
        if active_profile is None:
            return None, ()
        runtime_truth = active_profile.evidence_summary_json.get("runtime_truth", {})
        current_truth = runtime_truth.get("current_truth", {})
        roster_status = current_truth.get("roster_status")
        reason_codes = current_truth.get("reason_codes", [])
        if isinstance(roster_status, str) and isinstance(reason_codes, list):
            return roster_status, tuple(str(code) for code in reason_codes)
        enabled_entries = [
            item
            for item in active_profile.candidate_roster_json
            if bool(item.get("enabled", True))
        ]
        if len(enabled_entries) < 3:
            return "ACTIVE_WEAK", ("ACTIVE_PROFILE_ROSTER_INCOMPLETE",)
        return "ACTIVE", ("ACTIVE_PROFILE_PRESENT",)

    def record_request(self, *, path: str, status_code: int, latency_ms: float) -> None:
        """Record one completed HTTP request."""
        self.metrics.record(path=path, status_code=status_code, latency_ms=latency_ms)

    async def health(self) -> tuple[int, HealthResponse]:  # pylint: disable=too-many-locals
        """Return the current dependency health payload and status code."""
        runtime_metadata = self.runtime_metadata_fields()
        alert_health = await self._health_alert_fields()
        adaptation_summary = await self.adaptation_service.summary(
            execution_mode=self.trading_config.execution.mode,
            symbol="ALL",
            regime_label="ALL",
        )
        database_healthy = await self.database.is_healthy()
        model_loaded = self.model_artifact is not None
        ensemble_health = (
            await self._resolve_health_ensemble_state()
            if database_healthy and model_loaded
            else EnsembleRuntimeState(status="UNAVAILABLE")
        )
        continual_learning_profile_id = None
        continual_learning_status = "UNAVAILABLE"
        continual_learning_drift_cap_status = None
        if database_healthy and model_loaded:
            freshness_rows = await self._freshness_summary_rows()
            (
                continual_learning_profile_id,
                continual_learning_status,
                continual_learning_drift_cap_status,
            ) = await self._resolve_health_continual_learning_state(
                freshness_rows=freshness_rows,
            )
            health_snapshot = self._build_health_snapshot(freshness_rows)
            await self._maybe_write_service_heartbeat(
                health_overall_status=health_snapshot.health_overall_status,
                reason_code=(
                    SERVICE_HEARTBEAT_HEALTHY
                    if health_snapshot.health_overall_status == "HEALTHY"
                    else SERVICE_HEARTBEAT_DEGRADED
                ),
                detail=health_snapshot.reason_code,
                observed_at=health_snapshot.checked_at,
            )
            self._write_health_artifacts(health_snapshot)
            status_code = 200
            status_text = (
                "ok"
                if health_snapshot.health_overall_status == "HEALTHY"
                else "degraded"
            )
            health_overall_status = health_snapshot.health_overall_status
            reason_code = health_snapshot.reason_code
            freshness_status = health_snapshot.freshness_status
        else:
            status_code = 503
            status_text = "unavailable"
            health_overall_status = "UNAVAILABLE"
            reason_code = "DATABASE_UNAVAILABLE" if not database_healthy else "MODEL_UNAVAILABLE"
            freshness_status = "UNKNOWN"
            write_json_artifact(
                self.reliability_config.artifacts.health_snapshot_path,
                {
                    "service_name": self.settings.inference.service_name,
                    "checked_at": to_rfc3339(utc_now()),
                    "health_overall_status": health_overall_status,
                    "reason_code": reason_code,
                    "freshness_status": freshness_status,
                    "symbols": [],
                },
            )
        health_model_name, health_model_artifact_path = self._resolve_health_model_fields(
            ensemble_result=ensemble_health.result,
        )
        return (
            status_code,
            HealthResponse(
                status=status_text,
                service=self.settings.inference.service_name,
                runtime_profile=runtime_metadata["runtime_profile"],
                execution_mode=runtime_metadata["execution_mode"],
                startup_validation_passed=runtime_metadata["startup_validation_passed"],
                startup_report_path=runtime_metadata["startup_report_path"],
                model_loaded=model_loaded,
                model_name=health_model_name if model_loaded else None,
                model_artifact_path=health_model_artifact_path if model_loaded else None,
                regime_loaded=self.regime_runtime is not None,
                regime_run_id=self.regime_runtime.run_id,
                regime_artifact_path=self.regime_runtime.artifact_path,
                database="healthy" if database_healthy else "unavailable",
                started_at=self.started_at,
                health_overall_status=health_overall_status,
                reason_code=reason_code,
                freshness_status=freshness_status,
                active_alert_count=alert_health["active_alert_count"],
                max_alert_severity=alert_health["max_alert_severity"],
                startup_safety_status=alert_health["startup_safety_status"],
                startup_safety_reason_code=alert_health["startup_safety_reason_code"],
                active_adaptation_count=adaptation_summary.active_profile_count,
                adaptation_status=adaptation_summary.adaptation_status,
                adaptation_evidence_backed=adaptation_summary.evidence_backed,
                ensemble_profile_id=ensemble_health.result.ensemble_profile_id,
                ensemble_status=ensemble_health.status,
                ensemble_candidate_count=ensemble_health.result.candidate_count,
                ensemble_roster_status=ensemble_health.result.roster_status,
                ensemble_roster_reason_codes=list(
                    ensemble_health.result.roster_reason_codes
                ),
                active_continual_learning_profile_id=continual_learning_profile_id,
                continual_learning_status=continual_learning_status,
                continual_learning_drift_cap_status=continual_learning_drift_cap_status,
                continual_learning_evidence_backed=(
                    False
                    if not database_healthy or not model_loaded
                    else await self._resolve_health_continual_learning_evidence_backed()
                ),
            ),
        )

    async def _resolve_health_continual_learning_state(
        self,
        *,
        freshness_rows: list[FreshnessEvaluation],
    ) -> tuple[str | None, str, str | None]:
        """Resolve explicit M21 health summary fields from healthy runtime scopes."""
        if not self.continual_learning_service.config.enabled:
            return None, "DISABLED", None

        latest_drift_cap_status = None
        checked_any_scope = False
        for row in freshness_rows:
            if row.health_overall_status != "HEALTHY" or row.regime_label is None:
                continue
            context = await self._resolve_runtime_continual_learning_context(
                symbol=row.symbol,
                regime_label=row.regime_label,
                health_overall_status=row.health_overall_status,
                freshness_status=row.freshness_status,
            )
            if "CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE" in context.reason_codes:
                return None, "UNAVAILABLE", None
            checked_any_scope = True
            if context.active_profile_id is not None:
                return context.active_profile_id, "ACTIVE", context.drift_cap_status
            if context.drift_cap_status is not None and latest_drift_cap_status is None:
                latest_drift_cap_status = context.drift_cap_status

        if checked_any_scope:
            return None, "IDLE", latest_drift_cap_status

        fallback_context = await self._resolve_runtime_continual_learning_context(
            symbol="ALL",
            regime_label="ALL",
        )
        if "CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE" in fallback_context.reason_codes:
            return None, "UNAVAILABLE", None
        return (
            fallback_context.active_profile_id,
            "ACTIVE" if fallback_context.active_profile_id is not None else "IDLE",
            fallback_context.drift_cap_status,
        )

    async def _resolve_runtime_continual_learning_context(
        self,
        *,
        symbol: str,
        regime_label: str,
        health_overall_status: str | None = None,
        freshness_status: str | None = None,
    ) -> ContinualLearningContextPayload:
        """Resolve one read-only runtime continual-learning context payload."""
        return await self.continual_learning_service.resolve_runtime_context(
            execution_mode=self.trading_config.execution.mode,
            symbol=symbol,
            regime_label=regime_label,
            health_overall_status=health_overall_status,
            freshness_status=freshness_status,
        )

    async def _resolve_health_continual_learning_evidence_backed(self) -> bool:
        summary = await self.continual_learning_service.summary(
            execution_mode=self.trading_config.execution.mode,
            symbol="ALL",
            regime_label="ALL",
        )
        return summary.evidence_backed

    def validate_symbol(self, symbol: str) -> None:
        """Validate that the request symbol is part of the configured Kraken set."""
        if symbol not in self._symbols:
            raise InvalidSymbolError(f"Unsupported symbol: {symbol}")

    async def latest_feature_row(
        self,
        symbol: str,
        interval_begin: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Fetch the latest canonical feature row for one valid symbol."""
        self.validate_symbol(symbol)
        return await self.database.fetch_latest_feature_row(
            symbol=symbol,
            interval_minutes=self.settings.kraken.ohlc_interval_minutes,
            interval_begin=interval_begin,
        )

    async def predict_from_row(
        self,
        row: dict[str, Any],
        *,
        freshness: FreshnessEvaluation | None = None,
    ) -> PredictionResponse:
        """Run predict_proba against one canonical feature row."""
        prediction_context = await self._build_prediction_context(
            row,
            freshness=freshness,
        )
        return prediction_context.prediction

    async def _build_prediction_context(  # pylint: disable=too-many-locals
        self,
        row: dict[str, Any],
        *,
        freshness: FreshnessEvaluation | None = None,
    ) -> PredictionContext:
        """Build the prediction payload plus resolved regime details for reuse."""
        feature_input = self._build_feature_input(row)
        resolved_regime = self._resolve_regime(row)
        prob_down, prob_up = self._score_loaded_model_artifact(
            model_artifact=self.model_artifact,
            feature_input=feature_input,
        )
        predicted_class = "UP" if prob_up >= prob_down else "DOWN"
        base_buy_threshold, base_sell_threshold = self._default_thresholds(
            resolved_regime.regime_label
        )
        ensemble_state = await self._resolve_runtime_ensemble_state(
            row=row,
            resolved_regime=resolved_regime,
        )
        ensemble_result = ensemble_state.result
        if ensemble_result.active:
            response_prob_up = float(ensemble_result.ensemble_prob_up)
            response_prob_down = float(ensemble_result.ensemble_prob_down)
            response_predicted_class = str(ensemble_result.ensemble_predicted_class)
            response_confidence = float(ensemble_result.raw_ensemble_confidence)
            top_features: list[TopFeatureContribution] = []
            prediction_explanation = self._build_ensemble_pending_explainability()
        else:
            response_prob_up = prob_up
            response_prob_down = prob_down
            response_predicted_class = predicted_class
            response_confidence = max(prob_up, prob_down)
            top_features, prediction_explanation = await self._build_prediction_explainability(
                feature_input=feature_input,
                prob_up=prob_up,
            )
        # If ensemble is active, pass effective_confidence to adaptation
        confidence_for_adaptation = (
            ensemble_result.effective_confidence
            if ensemble_result.active and ensemble_result.effective_confidence is not None
            else response_confidence
        )
        applied_adaptation = await self.adaptation_service.resolve_applied_adaptation(
            execution_mode=self.trading_config.execution.mode,
            symbol=str(row["symbol"]),
            regime_label=resolved_regime.regime_label,
            base_buy_prob_up=base_buy_threshold,
            base_sell_prob_up=base_sell_threshold,
            confidence=confidence_for_adaptation,
            health_overall_status=(
                None if freshness is None else freshness.health_overall_status
            ),
            freshness_status=(None if freshness is None else freshness.freshness_status),
        )
        continual_learning_context = await self._resolve_runtime_continual_learning_context(
            symbol=str(row["symbol"]),
            regime_label=resolved_regime.regime_label,
            health_overall_status=(
                None if freshness is None else freshness.health_overall_status
            ),
            freshness_status=(None if freshness is None else freshness.freshness_status),
        )
        top_level_model_name, top_level_model_version = self._resolve_top_level_model_identity(
            ensemble_result=ensemble_result,
        )
        return PredictionContext(
            prediction=PredictionResponse(
                symbol=str(row["symbol"]),
                model_name=top_level_model_name,
                model_trained_at=self.model_artifact.trained_at,
                model_artifact_path=self.model_artifact.model_artifact_path,
                model_version=top_level_model_version,
                row_id=f"{row['symbol']}|{to_rfc3339(row['interval_begin'])}",
                interval_begin=to_rfc3339(row["interval_begin"]),
                as_of_time=to_rfc3339(row["as_of_time"]),
                prob_up=response_prob_up,
                prob_down=response_prob_down,
                predicted_class=response_predicted_class,
                confidence=response_confidence,
                regime_label=resolved_regime.regime_label,
                regime_run_id=resolved_regime.regime_run_id,
                decision_source="model",
                reason_code=None if freshness is None else freshness.reason_code,
                freshness_status=None if freshness is None else freshness.freshness_status,
                health_overall_status=(
                    None if freshness is None else freshness.health_overall_status
                ),
                top_features=top_features,
                prediction_explanation=prediction_explanation,
                adaptation_profile_id=applied_adaptation.profile_id,
                calibrated_confidence=applied_adaptation.calibrated_confidence,
                adaptation_reason_codes=list(applied_adaptation.adaptation_reason_codes),
                ensemble_profile_id=ensemble_result.ensemble_profile_id,
                ensemble_active=ensemble_result.active,
                ensemble_agreement_band=ensemble_result.agreement_band,
                ensemble_effective_confidence=ensemble_result.effective_confidence,
                ensemble_candidate_count=ensemble_result.candidate_count,
                ensemble_fallback_reason=ensemble_result.fallback_reason,
                ensemble_roster_status=ensemble_result.roster_status,
                ensemble_roster_reason_codes=list(ensemble_result.roster_reason_codes),
                ensemble=ensemble_result.to_context_payload(
                    regime_label=resolved_regime.regime_label,
                    regime_run_id=resolved_regime.regime_run_id,
                ),
                continual_learning_profile_id=continual_learning_context.active_profile_id,
                continual_learning_status=(
                    "ACTIVE"
                    if continual_learning_context.active_profile_id is not None
                    else (
                        "UNAVAILABLE"
                        if "CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE"
                        in continual_learning_context.reason_codes
                        else "IDLE"
                    )
                ),
                continual_learning_evidence_backed=(
                    continual_learning_context.drift_cap_status is not None
                ),
                continual_learning_frozen=continual_learning_context.frozen_by_health_gate,
                continual_learning=continual_learning_context,
            ),
            resolved_regime=resolved_regime,
            ensemble_result=ensemble_result,
        )

    async def signal_from_prediction(  # pylint: disable=too-many-locals
        self,
        prediction: PredictionResponse,
        *,
        resolved_regime: ResolvedRegime,
        freshness: FreshnessEvaluation | None = None,
        ensemble_result: EnsembleResult | None = None,
    ) -> SignalResponse:
        """Convert one prediction into BUY, SELL, or HOLD."""
        if ensemble_result is None:
            ensemble_result = EnsembleResult()
        policy = self.regime_runtime.policy_for(prediction.regime_label)
        # When ensemble is active, adaptation receives agreement-adjusted confidence
        if ensemble_result.active and ensemble_result.effective_confidence is not None:
            signal_confidence_for_adaptation = ensemble_result.effective_confidence
        elif prediction.calibrated_confidence is not None:
            signal_confidence_for_adaptation = prediction.calibrated_confidence
        else:
            signal_confidence_for_adaptation = prediction.confidence
        applied_adaptation = await self.adaptation_service.resolve_applied_adaptation(
            execution_mode=self.trading_config.execution.mode,
            symbol=prediction.symbol,
            regime_label=prediction.regime_label,
            base_buy_prob_up=policy.buy_prob_up,
            base_sell_prob_up=policy.sell_prob_up,
            confidence=signal_confidence_for_adaptation,
            health_overall_status=(
                None if freshness is None else freshness.health_overall_status
            ),
            freshness_status=(None if freshness is None else freshness.freshness_status),
        )
        buy_threshold = (
            policy.buy_prob_up
            if applied_adaptation.effective_thresholds is None
            else applied_adaptation.effective_thresholds.buy_prob_up
        )
        sell_threshold = (
            policy.sell_prob_up
            if applied_adaptation.effective_thresholds is None
            else applied_adaptation.effective_thresholds.sell_prob_up
        )
        decision_source = "model"
        reason_code = None if freshness is None else freshness.reason_code
        if prediction.prob_up >= buy_threshold:
            if policy.allow_new_long_entries:
                signal = "BUY"
                reason = (
                    f"prob_up {prediction.prob_up:.4f} >= buy threshold "
                    f"{buy_threshold:.2f}"
                )
                signal_status = "MODEL_SIGNAL"
            else:
                signal = "HOLD"
                reason = (
                    f"prob_up {prediction.prob_up:.4f} >= buy threshold "
                    f"{buy_threshold:.2f} but new long entries are disabled in "
                    f"{prediction.regime_label}"
                )
                signal_status = "MODEL_HOLD"
        elif prediction.prob_up <= sell_threshold:
            signal = "SELL"
            reason = f"prob_up {prediction.prob_up:.4f} <= sell threshold {sell_threshold:.2f}"
            signal_status = "MODEL_SIGNAL"
        else:
            signal = "HOLD"
            reason = (
                f"prob_up {prediction.prob_up:.4f} is between "
                f"{sell_threshold:.2f} and {buy_threshold:.2f}"
            )
            signal_status = "MODEL_HOLD"
        trade_allowed = signal in {"BUY", "SELL"}
        threshold_snapshot = self.explainability_service.build_threshold_snapshot(
            buy_prob_up=buy_threshold,
            sell_prob_up=sell_threshold,
            allow_new_long_entries=policy.allow_new_long_entries,
            resolved_regime=resolved_regime,
        )
        regime_reason = build_regime_reason(
            resolved_regime=resolved_regime,
            trade_allowed=policy.allow_new_long_entries,
        )
        signal_explanation = self.explainability_service.build_signal_explanation(
            signal=signal,
            decision_source=decision_source,
            reason=reason,
            trade_allowed=trade_allowed,
            regime_reason=regime_reason,
        )
        continual_learning_context = await self._resolve_runtime_continual_learning_context(
            symbol=prediction.symbol,
            regime_label=prediction.regime_label,
            health_overall_status=(
                None if freshness is None else freshness.health_overall_status
            ),
            freshness_status=(None if freshness is None else freshness.freshness_status),
        )

        return SignalResponse(
            symbol=prediction.symbol,
            signal=signal,
            reason=reason,
            prob_up=prediction.prob_up,
            prob_down=prediction.prob_down,
            confidence=prediction.confidence,
            predicted_class=prediction.predicted_class,
            thresholds=ThresholdsResponse(
                buy_prob_up=buy_threshold,
                sell_prob_up=sell_threshold,
            ),
            row_id=prediction.row_id,
            as_of_time=prediction.as_of_time,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
            regime_label=prediction.regime_label,
            regime_run_id=prediction.regime_run_id,
            trade_allowed=trade_allowed,
            signal_status=signal_status,
            decision_source=decision_source,
            reason_code=reason_code,
            freshness_status=None if freshness is None else freshness.freshness_status,
            health_overall_status=(
                None if freshness is None else freshness.health_overall_status
            ),
            top_features=prediction.top_features,
            prediction_explanation=prediction.prediction_explanation,
            threshold_snapshot=threshold_snapshot,
            regime_reason=regime_reason,
            signal_explanation=signal_explanation,
            adaptation_profile_id=applied_adaptation.profile_id,
            calibrated_confidence=applied_adaptation.calibrated_confidence,
            effective_thresholds=ThresholdsResponse(
                buy_prob_up=buy_threshold,
                sell_prob_up=sell_threshold,
            ),
            adaptation_reason_codes=list(applied_adaptation.adaptation_reason_codes),
            adaptive_size_multiplier=applied_adaptation.adaptive_size_multiplier,
            drift_status=applied_adaptation.drift_status,
            recent_performance_summary=applied_adaptation.recent_performance_summary,
            frozen_by_health_gate=applied_adaptation.frozen_by_health_gate,
            ensemble_profile_id=ensemble_result.ensemble_profile_id,
            ensemble_active=ensemble_result.active,
            ensemble_agreement_band=ensemble_result.agreement_band,
            ensemble_effective_confidence=ensemble_result.effective_confidence,
            ensemble_candidate_count=ensemble_result.candidate_count,
            ensemble_fallback_reason=ensemble_result.fallback_reason,
            ensemble_roster_status=ensemble_result.roster_status,
            ensemble_roster_reason_codes=list(ensemble_result.roster_reason_codes),
            ensemble=ensemble_result.to_context_payload(
                regime_label=prediction.regime_label,
                regime_run_id=prediction.regime_run_id,
            ),
            continual_learning_profile_id=continual_learning_context.active_profile_id,
            continual_learning_status=(
                "ACTIVE"
                if continual_learning_context.active_profile_id is not None
                else (
                    "UNAVAILABLE"
                    if "CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE"
                    in continual_learning_context.reason_codes
                    else "IDLE"
                )
            ),
            continual_learning_evidence_backed=(
                continual_learning_context.drift_cap_status is not None
            ),
            continual_learning_frozen=continual_learning_context.frozen_by_health_gate,
            continual_learning=continual_learning_context,
        )

    async def _build_prediction_explainability(
        self,
        *,
        feature_input: dict[str, Any],
        prob_up: float,
    ) -> tuple[list[TopFeatureContribution], PredictionExplanation]:
        """Build additive M14 prediction explainability without weakening M4 serving."""
        try:
            return await self.explainability_service.build_prediction_details(
                model_artifact=self.model_artifact,
                database=self.database,
                feature_input=feature_input,
                interval_minutes=self.settings.kraken.ohlc_interval_minutes,
                source_table=self.settings.tables.feature_ohlc,
                prob_up=prob_up,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            return (
                [],
                self.explainability_service.build_prediction_unavailable(
                    summary_text=(
                        "Explainability was unavailable for this prediction: "
                        f"{error}"
                    )
                ),
            )

    def regime_from_row(
        self,
        row: dict[str, Any],
        *,
        freshness: FreshnessEvaluation | None = None,
    ) -> RegimeResponse:
        """Resolve the exact-row regime payload used by M4."""
        resolved_regime = self._resolve_regime(row)
        policy = self.regime_runtime.policy_for(resolved_regime.regime_label)
        return RegimeResponse(
            symbol=resolved_regime.symbol,
            row_id=resolved_regime.row_id,
            interval_begin=resolved_regime.interval_begin,
            as_of_time=resolved_regime.as_of_time,
            regime_label=resolved_regime.regime_label,
            regime_run_id=resolved_regime.regime_run_id,
            regime_artifact_path=resolved_regime.regime_artifact_path,
            realized_vol_12=resolved_regime.realized_vol_12,
            momentum_3=resolved_regime.momentum_3,
            macd_line_12_26=resolved_regime.macd_line_12_26,
            high_vol_threshold=resolved_regime.high_vol_threshold,
            trend_abs_threshold=resolved_regime.trend_abs_threshold,
            trade_allowed=policy.allow_new_long_entries,
            buy_prob_up=policy.buy_prob_up,
            sell_prob_up=policy.sell_prob_up,
            freshness_status=None if freshness is None else freshness.freshness_status,
            health_overall_status=(
                None if freshness is None else freshness.health_overall_status
            ),
        )

    async def metrics_snapshot(self) -> MetricsResponse:
        """Return the current in-memory metrics payload."""
        runtime_metadata = self.runtime_metadata_fields()
        uptime_seconds = max(0.0, (utc_now() - self.started_at).total_seconds())
        try:
            freshness_rows = await self._freshness_summary_rows()
            health_snapshot = self._build_health_snapshot(freshness_rows)
            await self._maybe_write_service_heartbeat(
                health_overall_status=health_snapshot.health_overall_status,
                reason_code=(
                    SERVICE_HEARTBEAT_HEALTHY
                    if health_snapshot.health_overall_status == "HEALTHY"
                    else SERVICE_HEARTBEAT_DEGRADED
                ),
                detail=health_snapshot.reason_code,
                observed_at=health_snapshot.checked_at,
            )
            self._write_health_artifacts(health_snapshot)
            freshness_summary = {
                row.symbol: {
                    "health_overall_status": row.health_overall_status,
                    "freshness_status": row.freshness_status,
                    "reason_code": row.reason_code,
                    "feature_freshness_status": row.feature_freshness.freshness_status,
                    "regime_freshness_status": row.regime_freshness.freshness_status,
                }
                for row in freshness_rows
            }
            health_overall_status = health_snapshot.health_overall_status
            reason_code = health_snapshot.reason_code
        except Exception:  # pylint: disable=broad-exception-caught
            freshness_summary = None
            health_overall_status = "UNAVAILABLE"
            reason_code = "DATABASE_UNAVAILABLE"
        return MetricsResponse(
            requests_total=self.metrics.requests_total,
            errors_total=self.metrics.errors_total,
            endpoint_counts=dict(self.metrics.endpoint_counts),
            latency_ms=self.metrics.latency.as_response(),
            service=self.settings.inference.service_name,
            runtime_profile=runtime_metadata["runtime_profile"],
            execution_mode=runtime_metadata["execution_mode"],
            startup_validation_passed=runtime_metadata["startup_validation_passed"],
            startup_report_path=runtime_metadata["startup_report_path"],
            started_at=self.started_at,
            uptime_seconds=uptime_seconds,
            model_name=self.model_artifact.model_name,
            health_overall_status=health_overall_status,
            reason_code=reason_code,
            freshness_summary=freshness_summary,
        )

    async def adaptation_summary(
        self,
        *,
        symbol: str,
        regime_label: str,
    ):
        """Return the M19 read-only adaptation summary."""
        return await self.adaptation_service.summary(
            execution_mode=self.trading_config.execution.mode,
            symbol=symbol,
            regime_label=regime_label,
        )

    async def adaptation_drift(
        self,
        *,
        symbol: str,
        regime_label: str,
        limit: int,
    ):
        """Return the M19 read-only drift collection."""
        return await self.adaptation_service.drift(
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )

    async def adaptation_performance(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        limit: int,
    ):
        """Return the M19 read-only performance collection."""
        return await self.adaptation_service.performance(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )

    async def adaptation_profiles(self, *, limit: int):
        """Return the M19 read-only adaptive profile collection."""
        return await self.adaptation_service.profiles(limit=limit)

    async def adaptation_promotions(self, *, limit: int):
        """Return the M19 read-only adaptive promotion collection."""
        return await self.adaptation_service.promotions(limit=limit)

    async def continual_learning_summary(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
    ) -> ContinualLearningSummaryResponse:
        """Return the M21 read-only continual-learning summary."""
        return await self.continual_learning_service.summary(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )

    async def continual_learning_experiments(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        limit: int,
    ) -> ContinualLearningExperimentsResponse:
        """Return M21 read-only continual-learning experiments."""
        return await self.continual_learning_service.experiments(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )

    async def continual_learning_profiles(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        limit: int,
    ) -> ContinualLearningProfilesResponse:
        """Return M21 read-only continual-learning profiles."""
        return await self.continual_learning_service.profiles(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )

    async def continual_learning_drift_caps(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        limit: int,
    ) -> ContinualLearningDriftCapsResponse:
        """Return M21 read-only continual-learning drift-caps."""
        return await self.continual_learning_service.drift_caps(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )

    async def continual_learning_promotions(
        self,
        *,
        limit: int,
    ) -> ContinualLearningPromotionsResponse:
        """Return M21 read-only continual-learning promotions."""
        return await self.continual_learning_service.promotions(limit=limit)

    async def continual_learning_events(
        self,
        *,
        limit: int,
    ) -> ContinualLearningEventsResponse:
        """Return M21 read-only continual-learning events."""
        return await self.continual_learning_service.events(limit=limit)

    async def continual_learning_promote_profile(
        self,
        request: ContinualLearningPromoteProfileRequest,
    ) -> ContinualLearningWorkflowResponse:
        """Apply one guarded M21 profile promotion from existing operator inputs."""
        profile = await self.continual_learning_service.load_profile(
            profile_id=request.profile_id,
        )
        health_overall_status = await self._workflow_health_overall_status()
        freshness_status = await self._workflow_freshness_status(
            None if profile is None else profile.symbol_scope,
        )
        return await self.continual_learning_service.promote_profile(
            request,
            health_overall_status=health_overall_status,
            freshness_status=freshness_status,
        )

    async def continual_learning_rollback_profile(
        self,
        request: ContinualLearningRollbackRequest,
    ) -> ContinualLearningWorkflowResponse:
        """Apply one guarded M21 rollback from existing operator inputs."""
        health_overall_status = await self._workflow_health_overall_status()
        freshness_status = await self._workflow_freshness_status(request.symbol)
        return await self.continual_learning_service.rollback_profile(
            request,
            health_overall_status=health_overall_status,
            freshness_status=freshness_status,
        )

    async def _workflow_health_overall_status(self) -> str | None:
        """Resolve canonical M13 health for guarded M21 workflow calls when available."""
        try:
            _status_code, payload = await self.health()
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        return payload.health_overall_status

    async def _workflow_freshness_status(self, symbol: str | None) -> str | None:
        """Resolve exact-row freshness for one workflow symbol when it is simply available."""
        if symbol is None or symbol == "ALL":
            return None
        try:
            freshness = await self.freshness_evaluation(symbol=symbol)
        except (InvalidSymbolError, DatabaseUnavailableError):
            return None
        return freshness.freshness_status

    async def system_reliability_snapshot(
        self,
    ) -> tuple[int, SystemReliabilitySnapshot]:
        """Return the canonical cross-service reliability summary."""
        evaluated_at = utc_now()
        try:
            await self._refresh_inference_health_snapshot(evaluated_at)
            producer_heartbeat = await self.reliability_store.load_latest_service_heartbeat(
                service_name=self.settings.service_name,
                component_name="producer",
            )
            feature_heartbeat = await self.reliability_store.load_latest_service_heartbeat(
                service_name=self.settings.features.service_name,
                component_name="features",
            )
            inference_heartbeat = await self.reliability_store.load_latest_service_heartbeat(
                service_name=self.settings.inference.service_name,
                component_name="inference",
            )
            trading_runner_heartbeat = (
                await self.reliability_store.load_latest_service_heartbeat(
                    service_name=self.trading_config.service_name,
                    component_name="trading_runner",
                )
            )
            signal_client_state = await self.reliability_store.load_reliability_state(
                service_name=self.trading_config.service_name,
                component_name="signal_client",
            )
            lag_by_symbol = await self.reliability_store.load_feature_lag_states(
                service_name=self.settings.features.service_name,
                component_name="features",
            )
            latest_recovery_event = await self.reliability_store.load_latest_recovery_event()
        except Exception:  # pylint: disable=broad-exception-caught
            snapshot = SystemReliabilitySnapshot(
                service_name=self.settings.app_name,
                checked_at=evaluated_at,
                health_overall_status="UNAVAILABLE",
                reason_codes=("DATABASE_UNAVAILABLE",),
                lag_breach_active=False,
                services=(),
                lag_by_symbol=(),
                latest_recovery_event=None,
            )
            self._write_system_reliability_artifact(snapshot)
            return 503, snapshot

        snapshot = aggregate_system_reliability(
            service_name=self.settings.app_name,
            evaluated_at=evaluated_at,
            services=(
                build_service_health_snapshot(
                    service_name=self.settings.service_name,
                    component_name="producer",
                    heartbeat=producer_heartbeat,
                    evaluated_at=evaluated_at,
                    heartbeat_stale_after_seconds=(
                        self.reliability_config.heartbeat.stale_after_seconds
                    ),
                    feed_max_age_seconds=self.reliability_config.freshness.feed_max_age_seconds,
                ),
                build_service_health_snapshot(
                    service_name=self.settings.features.service_name,
                    component_name="features",
                    heartbeat=feature_heartbeat,
                    evaluated_at=evaluated_at,
                    heartbeat_stale_after_seconds=(
                        self.reliability_config.heartbeat.stale_after_seconds
                    ),
                ),
                build_service_health_snapshot(
                    service_name=self.settings.inference.service_name,
                    component_name="inference",
                    heartbeat=inference_heartbeat,
                    evaluated_at=evaluated_at,
                    heartbeat_stale_after_seconds=(
                        self.reliability_config.heartbeat.stale_after_seconds
                    ),
                ),
                build_service_health_snapshot(
                    service_name=self.trading_config.service_name,
                    component_name="trading_runner",
                    heartbeat=trading_runner_heartbeat,
                    evaluated_at=evaluated_at,
                    heartbeat_stale_after_seconds=(
                        self.reliability_config.heartbeat.stale_after_seconds
                    ),
                ),
                build_signal_client_health_snapshot(
                    service_name=self.trading_config.service_name,
                    component_name="signal_client",
                    state=signal_client_state,
                    evaluated_at=evaluated_at,
                    heartbeat_stale_after_seconds=(
                        self.reliability_config.heartbeat.stale_after_seconds
                    ),
                    idle_healthy_after_seconds=(
                        (self.trading_config.interval_minutes * 60)
                        + self.reliability_config.heartbeat.stale_after_seconds
                    ),
                ),
            ),
            lag_by_symbol=lag_by_symbol,
            latest_recovery_event=latest_recovery_event,
        )
        try:
            await self.reliability_store.save_system_reliability_state(snapshot)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        self._write_system_reliability_artifact(snapshot)
        return 200, snapshot

    async def active_alerts(self) -> list[OperationalAlertStateResponse]:
        """Return current active M17 alert states for the trading runtime."""
        states = await self.alert_repository.load_active_states(
            service_name=self.trading_config.service_name,
            execution_mode=self.trading_config.execution.mode,
        )
        return [
            OperationalAlertStateResponse(
                fingerprint=state.fingerprint,
                service_name=state.service_name,
                execution_mode=state.execution_mode,
                category=state.category,
                symbol=state.symbol,
                source_component=state.source_component,
                is_active=state.is_active,
                severity=state.severity,
                reason_code=state.reason_code,
                opened_at=state.opened_at,
                last_seen_at=state.last_seen_at,
                last_event_id=state.last_event_id,
                occurrence_count=state.occurrence_count,
            )
            for state in states
        ]

    async def alert_timeline(  # pylint: disable=too-many-arguments
        self,
        *,
        limit: int,
        category: str | None = None,
        severity: str | None = None,
        symbol: str | None = None,
        active_only: bool = False,
    ) -> list[OperationalAlertEventResponse]:
        """Return recent canonical alert timeline events for the trading runtime."""
        events = await self.alert_repository.load_timeline_events(
            service_name=self.trading_config.service_name,
            execution_mode=self.trading_config.execution.mode,
            limit=limit,
            category=category,
            severity=severity,
            symbol=symbol,
            active_only=active_only,
        )
        return [
            OperationalAlertEventResponse(
                id=event.event_id,
                service_name=event.service_name,
                execution_mode=event.execution_mode,
                category=event.category,
                severity=event.severity,
                event_state=event.event_state,
                reason_code=event.reason_code,
                source_component=event.source_component,
                symbol=event.symbol,
                fingerprint=event.fingerprint,
                summary_text=event.summary_text,
                detail=event.detail,
                event_time=event.event_time,
                related_order_request_id=event.related_order_request_id,
                related_decision_trace_id=event.related_decision_trace_id,
                payload_json=event.payload_json,
                created_at=event.created_at,
            )
            for event in events
        ]

    async def daily_operations_summary(
        self,
        *,
        summary_date: date | None = None,
    ) -> DailyOperationsSummaryResponse:
        """Read the canonical M17 daily operations summary artifact."""
        resolved_date = utc_now().date() if summary_date is None else summary_date
        artifact_path = (
            Path(self.alerting_config.artifacts.daily_summary_dir)
            / f"{resolved_date.isoformat()}.json"
        )
        return DailyOperationsSummaryResponse.model_validate(
            self._load_required_json_artifact(artifact_path)
        )

    async def startup_safety_report(self) -> StartupSafetyReportResponse:
        """Read the canonical M17 startup-safety artifact."""
        return self._load_startup_safety_report_model()

    def runtime_metadata_fields(self) -> dict[str, str | bool | None]:
        """Return additive runtime metadata exposed by M16 APIs."""
        metadata = build_runtime_metadata(
            execution_mode=self.trading_config.execution.mode,
        )
        return {
            "runtime_profile": metadata.runtime_profile,
            "execution_mode": metadata.execution_mode,
            "startup_validation_passed": metadata.startup_validation_passed,
            "startup_report_path": metadata.startup_report_path,
        }

    async def _health_alert_fields(self) -> dict[str, int | str | None]:
        """Return additive M17 health-summary fields without breaking `/health`."""
        try:
            active_states = await self.alert_repository.load_active_states(
                service_name=self.trading_config.service_name,
                execution_mode=self.trading_config.execution.mode,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            active_states = None

        try:
            startup_safety = self._load_startup_safety_report_model()
        except Exception:  # pylint: disable=broad-exception-caught
            startup_safety = None

        return {
            "active_alert_count": (
                None if active_states is None else len(active_states)
            ),
            "max_alert_severity": _max_alert_severity(active_states),
            "startup_safety_status": (
                None
                if startup_safety is None
                else (
                    "PASSED"
                    if startup_safety.startup_safety_passed
                    else "FAILED"
                )
            ),
            "startup_safety_reason_code": (
                None if startup_safety is None else startup_safety.primary_reason_code
            ),
        }

    def _load_startup_safety_report_model(self) -> StartupSafetyReportResponse:
        return StartupSafetyReportResponse.model_validate(
            self._load_required_json_artifact(
                Path(self.alerting_config.artifacts.startup_safety_path)
            )
        )

    def _load_required_json_artifact(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Artifact not found: {path}")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"Artifact must deserialize into a mapping: {path}")
        return parsed

    async def _refresh_inference_health_snapshot(
        self,
        evaluated_at: datetime,
    ) -> ReliabilityHealthSnapshot:
        """Refresh the inference heartbeat before cross-service aggregation."""
        freshness_rows = await self._freshness_summary_rows(evaluated_at=evaluated_at)
        health_snapshot = self._build_health_snapshot(
            freshness_rows,
            checked_at=evaluated_at,
        )
        await self._maybe_write_service_heartbeat(
            health_overall_status=health_snapshot.health_overall_status,
            reason_code=(
                SERVICE_HEARTBEAT_HEALTHY
                if health_snapshot.health_overall_status == "HEALTHY"
                else SERVICE_HEARTBEAT_DEGRADED
            ),
            detail=health_snapshot.reason_code,
            observed_at=health_snapshot.checked_at,
        )
        self._write_health_artifacts(health_snapshot)
        return health_snapshot

    async def freshness_response(
        self,
        *,
        symbol: str,
        interval_begin: datetime | None = None,
    ) -> FreshnessResponse:
        """Return explicit exact-row freshness for one symbol."""
        freshness = await self.freshness_evaluation(
            symbol=symbol,
            interval_begin=interval_begin,
        )
        await self._maybe_write_service_heartbeat(
            health_overall_status=freshness.health_overall_status,
            reason_code=(
                SERVICE_HEARTBEAT_HEALTHY
                if freshness.health_overall_status == "HEALTHY"
                else SERVICE_HEARTBEAT_DEGRADED
            ),
            detail=freshness.reason_code,
            observed_at=utc_now(),
        )
        self._write_freshness_artifact([freshness])
        return FreshnessResponse(
            symbol=freshness.symbol,
            row_id=freshness.row_id,
            interval_begin=freshness.interval_begin,
            as_of_time=freshness.as_of_time,
            health_overall_status=freshness.health_overall_status,
            freshness_status=freshness.freshness_status,
            reason_code=freshness.reason_code,
            feature_freshness_status=freshness.feature_freshness.freshness_status,
            feature_reason_code=freshness.feature_freshness.reason_code,
            feature_age_seconds=freshness.feature_freshness.age_seconds,
            regime_freshness_status=freshness.regime_freshness.freshness_status,
            regime_reason_code=freshness.regime_freshness.reason_code,
            regime_age_seconds=freshness.regime_freshness.age_seconds,
            detail=freshness.detail,
        )

    async def freshness_evaluation(
        self,
        *,
        symbol: str,
        interval_begin: datetime | None = None,
    ) -> FreshnessEvaluation:
        """Return the internal exact-row freshness evaluation for one symbol."""
        row = await self.latest_feature_row(symbol, interval_begin=interval_begin)
        return self._evaluate_freshness_for_row(
            symbol=symbol,
            row=row,
            interval_begin=interval_begin,
            evaluated_at=utc_now(),
        )

    async def signal_for_request(
        self,
        *,
        symbol: str,
        interval_begin: datetime | None = None,
    ) -> SignalResponse:
        """Return the authoritative signal or an explicit reliability HOLD."""
        row = await self.latest_feature_row(symbol, interval_begin=interval_begin)
        freshness = self._evaluate_freshness_for_row(
            symbol=symbol,
            row=row,
            interval_begin=interval_begin,
            evaluated_at=utc_now(),
        )
        await self._maybe_write_service_heartbeat(
            health_overall_status=freshness.health_overall_status,
            reason_code=(
                SERVICE_HEARTBEAT_HEALTHY
                if freshness.health_overall_status == "HEALTHY"
                else SERVICE_HEARTBEAT_DEGRADED
            ),
            detail=freshness.reason_code,
            observed_at=utc_now(),
        )

        if row is None:
            return await self._build_reliability_hold(
                symbol=symbol,
                as_of_time=None,
                row_id=freshness.row_id or _fallback_row_id(symbol, interval_begin),
                resolved_regime=None,
                freshness=freshness,
                reason_code=RELIABILITY_HOLD_MISSING_FEATURE_ROW,
                reason="No canonical feature row was found for the requested candle",
            )

        if freshness.feature_freshness.freshness_status != "FRESH":
            resolved_regime = self._try_resolve_regime(row)
            return await self._build_reliability_hold(
                symbol=symbol,
                as_of_time=row["as_of_time"],
                row_id=f"{row['symbol']}|{to_rfc3339(row['interval_begin'])}",
                resolved_regime=resolved_regime,
                freshness=freshness,
                reason_code=RELIABILITY_HOLD_STALE_FEATURE_ROW,
                reason=(
                    "Canonical feature inputs are stale for the requested signal decision"
                ),
            )

        try:
            prediction_context = await self._build_prediction_context(
                row,
                freshness=freshness,
            )
            return await self.signal_from_prediction(
                prediction_context.prediction,
                resolved_regime=prediction_context.resolved_regime,
                freshness=freshness,
                ensemble_result=prediction_context.ensemble_result,
            )
        except ArtifactSchemaMismatchError as error:
            return await self._build_reliability_hold(
                symbol=symbol,
                as_of_time=row["as_of_time"],
                row_id=f"{row['symbol']}|{to_rfc3339(row['interval_begin'])}",
                resolved_regime=self._try_resolve_regime(row),
                freshness=freshness,
                reason_code=(
                    REGIME_ROW_INCOMPATIBLE
                    if freshness.regime_freshness.freshness_status != "FRESH"
                    else RELIABILITY_HOLD_INPUTS_MISSING
                ),
                reason=str(error),
            )

    def _build_feature_input(
        self,
        row: dict[str, Any],
        *,
        model_artifact: LoadedModelArtifact | None = None,
    ) -> dict[str, Any]:
        resolved_artifact = self.model_artifact if model_artifact is None else model_artifact
        missing_columns = [
            column
            for column in resolved_artifact.feature_columns
            if column not in row or row[column] is None
        ]
        if missing_columns:
            raise ArtifactSchemaMismatchError(
                f"Latest feature row is missing required model columns: {missing_columns}",
            )
        return {
            column: row[column]
            for column in resolved_artifact.feature_columns
        }

    def _validate_thresholds(self) -> None:
        buy_threshold = self.settings.inference.signal_buy_prob_up
        sell_threshold = self.settings.inference.signal_sell_prob_up
        if not 0.0 <= sell_threshold <= 1.0:
            raise ValueError("INFERENCE_SIGNAL_SELL_PROB_UP must be between 0 and 1")
        if not 0.0 <= buy_threshold <= 1.0:
            raise ValueError("INFERENCE_SIGNAL_BUY_PROB_UP must be between 0 and 1")
        if sell_threshold > buy_threshold:
            raise ValueError("SELL threshold cannot be greater than BUY threshold")

    def _resolve_regime(self, row: dict[str, Any]) -> ResolvedRegime:
        try:
            return self.regime_runtime.resolve_feature_row_regime(row)
        except ValueError as error:
            raise ArtifactSchemaMismatchError(str(error)) from error

    def _try_resolve_regime(self, row: dict[str, Any]) -> ResolvedRegime | None:
        try:
            return self._resolve_regime(row)
        except ArtifactSchemaMismatchError:
            return None

    async def _freshness_summary_rows(
        self,
        evaluated_at: datetime | None = None,
    ) -> list[FreshnessEvaluation]:
        rows: list[FreshnessEvaluation] = []
        effective_evaluated_at = utc_now() if evaluated_at is None else evaluated_at
        for symbol in self.settings.kraken.symbols:
            row = await self.latest_feature_row(symbol)
            rows.append(
                self._evaluate_freshness_for_row(
                    symbol=symbol,
                    row=row,
                    interval_begin=None,
                    evaluated_at=effective_evaluated_at,
                )
            )
        return rows

    # pylint: disable=too-many-locals
    def _evaluate_freshness_for_row(
        self,
        *,
        symbol: str,
        row: dict[str, Any] | None,
        interval_begin: datetime | None,
        evaluated_at: datetime,
    ) -> FreshnessEvaluation:
        if row is None:
            feature_freshness = FreshnessStatus(
                component_name="feature_ohlc",
                freshness_status="STALE",
                evaluated_at=evaluated_at,
                max_age_seconds=self.reliability_config.freshness.feature_max_age_seconds,
                reason_code=FEATURE_ROW_MISSING,
                observed_at=None,
                age_seconds=None,
                detail="No exact canonical feature row was found",
            )
            regime_freshness = evaluate_regime_freshness(
                observed_at=None,
                evaluated_at=evaluated_at,
                max_age_seconds=self.reliability_config.freshness.regime_max_age_seconds,
                exact_row_resolved=False,
                detail="Cannot resolve regime without an exact canonical feature row",
            )
            health = aggregate_health_status(
                freshness_statuses=(feature_freshness, regime_freshness)
            )
            row_id = _fallback_row_id(symbol, interval_begin)
            interval_begin_text = (
                None if interval_begin is None else to_rfc3339(interval_begin)
            )
            return FreshnessEvaluation(
                symbol=symbol,
                row_id=row_id,
                interval_begin=interval_begin_text,
                as_of_time=None,
                health_overall_status=health.health_overall_status,
                freshness_status=_overall_freshness_status(
                    feature_freshness,
                    regime_freshness,
                ),
                reason_code=FEATURE_ROW_MISSING,
                feature_freshness=feature_freshness,
                regime_freshness=regime_freshness,
                detail=feature_freshness.detail,
            )

        interval_begin_text = to_rfc3339(row["interval_begin"])
        as_of_time_text = to_rfc3339(row["as_of_time"])
        feature_freshness = evaluate_feature_freshness(
            observed_at=row["as_of_time"],
            evaluated_at=evaluated_at,
            max_age_seconds=self.reliability_config.freshness.feature_max_age_seconds,
        )
        resolved_regime = None
        regime_error = None
        try:
            resolved_regime = self.regime_runtime.resolve_feature_row_regime(row)
        except ValueError as error:
            regime_error = str(error)
        regime_freshness = evaluate_regime_freshness(
            observed_at=row["as_of_time"],
            evaluated_at=evaluated_at,
            max_age_seconds=self.reliability_config.freshness.regime_max_age_seconds,
            exact_row_resolved=resolved_regime is not None,
            detail=(
                None
                if resolved_regime is not None
                else regime_error or "Exact-row regime resolution failed"
            ),
        )
        health = aggregate_health_status(
            freshness_statuses=(feature_freshness, regime_freshness)
        )
        if feature_freshness.freshness_status != "FRESH":
            reason_code = feature_freshness.reason_code
        elif resolved_regime is None:
            reason_code = REGIME_ROW_INCOMPATIBLE
        else:
            reason_code = HEALTH_HEALTHY
        detail = (
            regime_freshness.detail
            if regime_freshness.freshness_status != "FRESH"
            else feature_freshness.detail
        )
        return FreshnessEvaluation(
            symbol=symbol,
            row_id=f"{row['symbol']}|{interval_begin_text}",
            interval_begin=interval_begin_text,
            as_of_time=as_of_time_text,
            health_overall_status=health.health_overall_status,
            freshness_status=_overall_freshness_status(
                feature_freshness,
                regime_freshness,
            ),
            reason_code=reason_code,
            feature_freshness=feature_freshness,
            regime_freshness=regime_freshness,
            regime_label=None if resolved_regime is None else resolved_regime.regime_label,
            regime_run_id=None if resolved_regime is None else resolved_regime.regime_run_id,
            detail=detail,
        )
    # pylint: enable=too-many-locals

    def _build_health_snapshot(
        self,
        freshness_rows: list[FreshnessEvaluation],
        *,
        checked_at: datetime | None = None,
    ) -> ReliabilityHealthSnapshot:
        overall_status = (
            "HEALTHY"
            if all(row.health_overall_status == "HEALTHY" for row in freshness_rows)
            else "DEGRADED"
        )
        reason_code = (
            HEALTH_HEALTHY
            if overall_status == "HEALTHY"
            else next(
                row.reason_code
                for row in freshness_rows
                if row.health_overall_status != "HEALTHY"
            )
        )
        freshness_status = (
            "FRESH"
            if all(row.freshness_status == "FRESH" for row in freshness_rows)
            else next(
                row.freshness_status
                for row in freshness_rows
                if row.freshness_status != "FRESH"
            )
        )
        return ReliabilityHealthSnapshot(
            service_name=self.settings.inference.service_name,
            checked_at=utc_now() if checked_at is None else checked_at,
            health_overall_status=overall_status,
            reason_code=reason_code,
            freshness_status=freshness_status,
            symbols=tuple(
                SymbolFreshnessSnapshot(
                    symbol=row.symbol,
                    row_id=row.row_id,
                    interval_begin=(
                        None
                        if row.interval_begin is None
                        else _parse_cached_rfc3339(row.interval_begin)
                    ),
                    as_of_time=(
                        None
                        if row.as_of_time is None
                        else _parse_cached_rfc3339(row.as_of_time)
                    ),
                    health_overall_status=row.health_overall_status,
                    freshness_status=row.freshness_status,
                    reason_code=row.reason_code,
                    feature_freshness=row.feature_freshness,
                    regime_freshness=row.regime_freshness,
                )
                for row in freshness_rows
            ),
        )

    async def _maybe_write_service_heartbeat(
        self,
        *,
        health_overall_status: str,
        reason_code: str,
        detail: str | None,
        observed_at: datetime,
    ) -> None:
        if (
            self._last_heartbeat_at is not None
            and (observed_at - self._last_heartbeat_at).total_seconds()
            < self.reliability_config.heartbeat.write_interval_seconds
        ):
            return
        self._last_heartbeat_at = observed_at
        try:
            await self.reliability_store.save_service_heartbeat(
                ServiceHeartbeat(
                    service_name=self.settings.inference.service_name,
                    component_name="inference",
                    heartbeat_at=observed_at,
                    health_overall_status=health_overall_status,
                    reason_code=reason_code,
                    detail=detail,
                )
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return

    def _write_health_artifacts(
        self,
        health_snapshot: ReliabilityHealthSnapshot,
    ) -> None:
        payload = {
            "service_name": health_snapshot.service_name,
            "checked_at": to_rfc3339(health_snapshot.checked_at),
            "health_overall_status": health_snapshot.health_overall_status,
            "reason_code": health_snapshot.reason_code,
            "freshness_status": health_snapshot.freshness_status,
            "symbols": [
                {
                    "symbol": symbol.symbol,
                    "row_id": symbol.row_id,
                    "interval_begin": (
                        None
                        if symbol.interval_begin is None
                        else to_rfc3339(symbol.interval_begin)
                    ),
                    "as_of_time": (
                        None if symbol.as_of_time is None else to_rfc3339(symbol.as_of_time)
                    ),
                    "health_overall_status": symbol.health_overall_status,
                    "freshness_status": symbol.freshness_status,
                    "reason_code": symbol.reason_code,
                    "feature_freshness_status": symbol.feature_freshness.freshness_status,
                    "regime_freshness_status": symbol.regime_freshness.freshness_status,
                }
                for symbol in health_snapshot.symbols
            ],
        }
        write_json_artifact(
            self.reliability_config.artifacts.health_snapshot_path,
            payload,
        )
        self._write_freshness_artifact(
            [
                FreshnessEvaluation(
                    symbol=symbol.symbol,
                    row_id=symbol.row_id,
                    interval_begin=(
                        None
                        if symbol.interval_begin is None
                        else to_rfc3339(symbol.interval_begin)
                    ),
                    as_of_time=(
                        None
                        if symbol.as_of_time is None
                        else to_rfc3339(symbol.as_of_time)
                    ),
                    health_overall_status=symbol.health_overall_status,
                    freshness_status=symbol.freshness_status,
                    reason_code=symbol.reason_code,
                    feature_freshness=symbol.feature_freshness,
                    regime_freshness=symbol.regime_freshness,
                )
                for symbol in health_snapshot.symbols
            ]
        )

    def _write_system_reliability_artifact(
        self,
        system_snapshot: SystemReliabilitySnapshot,
    ) -> None:
        write_json_artifact(
            self.reliability_config.artifacts.system_health_path,
            asdict(system_snapshot),
        )

    def _write_freshness_artifact(
        self,
        freshness_rows: list[FreshnessEvaluation],
    ) -> None:
        write_json_artifact(
            self.reliability_config.artifacts.freshness_summary_path,
            {
                "generated_at": to_rfc3339(utc_now()),
                "service_name": self.settings.inference.service_name,
                "symbols": [
                    {
                        "symbol": row.symbol,
                        "row_id": row.row_id,
                        "interval_begin": row.interval_begin,
                        "as_of_time": row.as_of_time,
                        "health_overall_status": row.health_overall_status,
                        "freshness_status": row.freshness_status,
                        "reason_code": row.reason_code,
                        "feature_freshness_status": row.feature_freshness.freshness_status,
                        "feature_reason_code": row.feature_freshness.reason_code,
                        "feature_age_seconds": row.feature_freshness.age_seconds,
                        "regime_freshness_status": row.regime_freshness.freshness_status,
                        "regime_reason_code": row.regime_freshness.reason_code,
                        "regime_age_seconds": row.regime_freshness.age_seconds,
                    }
                    for row in freshness_rows
                ],
            },
        )

    # pylint: disable=too-many-arguments
    async def _build_reliability_hold(  # pylint: disable=too-many-locals
        self,
        *,
        symbol: str,
        as_of_time: datetime | None,
        row_id: str,
        resolved_regime: ResolvedRegime | None,
        freshness: FreshnessEvaluation,
        reason_code: str,
        reason: str,
    ) -> SignalResponse:
        regime_label = (
            None if resolved_regime is None else resolved_regime.regime_label
        )
        regime_run_id = (
            None if resolved_regime is None else resolved_regime.regime_run_id
        )
        buy_threshold, sell_threshold = self._default_thresholds(regime_label)
        allow_new_long_entries = (
            False
            if regime_label is None
            else self.regime_runtime.policy_for(regime_label).allow_new_long_entries
        )
        threshold_snapshot = self.explainability_service.build_threshold_snapshot(
            buy_prob_up=buy_threshold,
            sell_prob_up=sell_threshold,
            allow_new_long_entries=allow_new_long_entries,
            resolved_regime=resolved_regime,
        )
        regime_reason = (
            None
            if resolved_regime is None
            else build_regime_reason(
                resolved_regime=resolved_regime,
                trade_allowed=allow_new_long_entries,
            )
        )
        signal_explanation = self.explainability_service.build_signal_explanation(
            signal="HOLD",
            decision_source="reliability",
            reason=reason,
            trade_allowed=False,
            regime_reason=regime_reason,
        )
        continual_learning_context = await self._resolve_runtime_continual_learning_context(
            symbol=symbol,
            regime_label=("ALL" if regime_label is None else regime_label),
            health_overall_status=freshness.health_overall_status,
            freshness_status=freshness.freshness_status,
        )
        effective_as_of_time = (
            utc_now()
            if as_of_time is None
            else as_of_time
        )
        return SignalResponse(
            symbol=symbol,
            signal="HOLD",
            reason=reason,
            prob_up=0.5,
            prob_down=0.5,
            confidence=0.0,
            predicted_class="UNKNOWN",
            thresholds=ThresholdsResponse(
                buy_prob_up=buy_threshold,
                sell_prob_up=sell_threshold,
            ),
            row_id=row_id,
            as_of_time=to_rfc3339(effective_as_of_time),
            model_name=self.model_artifact.model_name,
            model_version=self.model_artifact.model_version,
            regime_label=regime_label,
            regime_run_id=regime_run_id,
            trade_allowed=False,
            signal_status="RELIABILITY_HOLD",
            decision_source="reliability",
            reason_code=reason_code,
            freshness_status=freshness.freshness_status,
            health_overall_status=freshness.health_overall_status,
            top_features=[],
            prediction_explanation=self.explainability_service.build_prediction_unavailable(
                summary_text=reason,
            ),
            threshold_snapshot=threshold_snapshot,
            regime_reason=regime_reason,
            signal_explanation=signal_explanation,
            continual_learning_profile_id=continual_learning_context.active_profile_id,
            continual_learning_status=(
                "ACTIVE"
                if continual_learning_context.active_profile_id is not None
                else (
                    "UNAVAILABLE"
                    if "CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE"
                    in continual_learning_context.reason_codes
                    else "IDLE"
                )
            ),
            continual_learning_evidence_backed=(
                continual_learning_context.drift_cap_status is not None
            ),
            continual_learning_frozen=continual_learning_context.frozen_by_health_gate,
            continual_learning=continual_learning_context,
        )
    # pylint: enable=too-many-arguments

    def _default_thresholds(self, regime_label: str | None) -> tuple[float, float]:
        if regime_label is not None:
            policy = self.regime_runtime.policy_for(regime_label)
            return policy.buy_prob_up, policy.sell_prob_up
        return (
            self.settings.inference.signal_buy_prob_up,
            self.settings.inference.signal_sell_prob_up,
        )


def request_latency_ms(started_at: float) -> float:
    """Return elapsed request time in milliseconds."""
    return (perf_counter() - started_at) * 1000.0


def _overall_freshness_status(*statuses: FreshnessStatus) -> str:
    if any(status.freshness_status == "STALE" for status in statuses):
        return "STALE"
    if any(status.freshness_status == "UNKNOWN" for status in statuses):
        return "UNKNOWN"
    return "FRESH"


def _fallback_row_id(symbol: str, interval_begin: datetime | None) -> str:
    if interval_begin is None:
        return f"{symbol}|MISSING"
    return f"{symbol}|{to_rfc3339(interval_begin)}"


def _parse_cached_rfc3339(value: str) -> datetime:
    return parse_rfc3339(value)


def _max_alert_severity(active_states: list[Any] | None) -> str | None:
    if active_states is None or not active_states:
        return None
    severity_order = {
        "INFO": 0,
        "WARNING": 1,
        "CRITICAL": 2,
    }
    return max(
        (str(state.severity) for state in active_states),
        key=lambda value: severity_order.get(value, -1),
    )
