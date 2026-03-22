"""Core inference service logic for the Stream Alpha M4 API."""

# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib

from app.common.config import Settings
from app.common.time import parse_rfc3339, to_rfc3339, utc_now
from app.explainability.config import (
    default_explainability_config_path,
    load_explainability_config,
)
from app.explainability.schemas import PredictionExplanation, TopFeatureContribution
from app.explainability.service import ExplainabilityService, build_regime_reason
from app.inference.db import InferenceDatabase
from app.inference.schemas import (
    FreshnessResponse,
    HealthResponse,
    LatencyStatsResponse,
    MetricsResponse,
    PredictionResponse,
    RegimeResponse,
    SignalResponse,
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
from app.training.registry import resolve_inference_model_metadata


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


class InferenceService:  # pylint: disable=too-many-instance-attributes
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
    ) -> None:
        self.settings = settings
        self.started_at = utc_now()
        self.metrics = MetricsState(started_at=self.started_at)
        self.reliability_config = load_reliability_config(default_reliability_config_path())
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
        self.model_artifact = model_artifact or load_model_artifact(
            settings.inference.model_path,
        )
        self.regime_runtime = regime_runtime or load_live_regime_runtime(
            thresholds_path=settings.inference.regime_thresholds_path,
            signal_policy_path=settings.inference.regime_signal_policy_path,
        )
        self.explainability_service = ExplainabilityService(self.explainability_config)
        self.trading_config = load_paper_trading_config(resolve_trading_config_path())
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

    async def shutdown(self) -> None:
        """Close the database pool."""
        await self.database.close()
        try:
            await self.reliability_store.close()
        except Exception:  # pylint: disable=broad-exception-caught
            return

    def record_request(self, *, path: str, status_code: int, latency_ms: float) -> None:
        """Record one completed HTTP request."""
        self.metrics.record(path=path, status_code=status_code, latency_ms=latency_ms)

    async def health(self) -> tuple[int, HealthResponse]:
        """Return the current dependency health payload and status code."""
        runtime_metadata = self.runtime_metadata_fields()
        database_healthy = await self.database.is_healthy()
        model_loaded = self.model_artifact is not None
        if database_healthy and model_loaded:
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
                model_name=self.model_artifact.model_name if model_loaded else None,
                model_artifact_path=(
                    self.model_artifact.model_artifact_path if model_loaded else None
                ),
                regime_loaded=self.regime_runtime is not None,
                regime_run_id=self.regime_runtime.run_id,
                regime_artifact_path=self.regime_runtime.artifact_path,
                database="healthy" if database_healthy else "unavailable",
                started_at=self.started_at,
                health_overall_status=health_overall_status,
                reason_code=reason_code,
                freshness_status=freshness_status,
            ),
        )

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

    async def _build_prediction_context(
        self,
        row: dict[str, Any],
        *,
        freshness: FreshnessEvaluation | None = None,
    ) -> PredictionContext:
        """Build the prediction payload plus resolved regime details for reuse."""
        feature_input = self._build_feature_input(row)
        resolved_regime = self._resolve_regime(row)
        probabilities = self.model_artifact.model.predict_proba([feature_input])
        if len(probabilities) != 1 or len(probabilities[0]) != 2:
            raise ArtifactSchemaMismatchError(
                "Model predict_proba must return binary probabilities",
            )

        prob_down = float(probabilities[0][0])
        prob_up = float(probabilities[0][1])
        predicted_class = "UP" if prob_up >= prob_down else "DOWN"
        top_features, prediction_explanation = await self._build_prediction_explainability(
            feature_input=feature_input,
            prob_up=prob_up,
        )
        return PredictionContext(
            prediction=PredictionResponse(
                symbol=str(row["symbol"]),
                model_name=self.model_artifact.model_name,
                model_trained_at=self.model_artifact.trained_at,
                model_artifact_path=self.model_artifact.model_artifact_path,
                model_version=self.model_artifact.model_version,
                row_id=f"{row['symbol']}|{to_rfc3339(row['interval_begin'])}",
                interval_begin=to_rfc3339(row["interval_begin"]),
                as_of_time=to_rfc3339(row["as_of_time"]),
                prob_up=prob_up,
                prob_down=prob_down,
                predicted_class=predicted_class,
                confidence=max(prob_up, prob_down),
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
            ),
            resolved_regime=resolved_regime,
        )

    def signal_from_prediction(  # pylint: disable=too-many-locals
        self,
        prediction: PredictionResponse,
        *,
        resolved_regime: ResolvedRegime,
        freshness: FreshnessEvaluation | None = None,
    ) -> SignalResponse:
        """Convert one prediction into BUY, SELL, or HOLD."""
        policy = self.regime_runtime.policy_for(prediction.regime_label)
        buy_threshold = policy.buy_prob_up
        sell_threshold = policy.sell_prob_up
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
            return self._build_reliability_hold(
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
            return self._build_reliability_hold(
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
            return self.signal_from_prediction(
                prediction_context.prediction,
                resolved_regime=prediction_context.resolved_regime,
                freshness=freshness,
            )
        except ArtifactSchemaMismatchError as error:
            return self._build_reliability_hold(
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

    def _build_feature_input(self, row: dict[str, Any]) -> dict[str, Any]:
        missing_columns = [
            column
            for column in self.model_artifact.feature_columns
            if column not in row or row[column] is None
        ]
        if missing_columns:
            raise ArtifactSchemaMismatchError(
                f"Latest feature row is missing required model columns: {missing_columns}",
            )
        return {
            column: row[column]
            for column in self.model_artifact.feature_columns
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
    def _build_reliability_hold(  # pylint: disable=too-many-locals
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
