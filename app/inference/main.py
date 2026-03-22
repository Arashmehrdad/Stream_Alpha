"""FastAPI app factory for the Stream Alpha M4 inference service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
import logging
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from app.adaptation.schemas import (
    AdaptationDriftResponse,
    AdaptationPerformanceResponse,
    AdaptationProfilesResponse,
    AdaptationPromotionsResponse,
    AdaptationSummaryResponse,
)
from app.continual_learning.schemas import (
    ContinualLearningDriftCapsResponse,
    ContinualLearningEventsResponse,
    ContinualLearningExperimentsResponse,
    ContinualLearningProfilesResponse,
    ContinualLearningPromotionsResponse,
    ContinualLearningSummaryResponse,
)
from app.common.config import Settings
from app.common.logging import configure_logging
from app.common.time import parse_rfc3339
from app.inference.db import DatabaseUnavailableError
from app.inference.schemas import (
    DailyOperationsSummaryResponse,
    FeatureRowResponse,
    FreshnessResponse,
    HealthResponse,
    MetricsResponse,
    OperationalAlertEventResponse,
    OperationalAlertStateResponse,
    StartupSafetyReportResponse,
    SystemReliabilityResponse,
)
from app.inference.schemas import PredictionResponse, RegimeResponse, SignalResponse
from app.inference.service import (
    ArtifactSchemaMismatchError,
    InferenceService,
    InvalidSymbolError,
    request_latency_ms,
)

# pylint: disable=too-many-statements,too-many-locals
def create_app(
    service: InferenceService | None = None,
) -> FastAPI:
    """Create the M4 FastAPI app with strict model loading."""
    if service is None:
        settings = Settings.from_env()
        configure_logging(settings.log_level)
        service = InferenceService(settings)

    logger = logging.getLogger(f"{service.settings.app_name}.inference")

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        await service.startup()
        try:
            yield
        finally:
            await service.shutdown()

    app = FastAPI(title="Stream Alpha Inference API", version="m14", lifespan=_lifespan)
    app.state.service = service

    @app.middleware("http")
    async def _logging_middleware(request: Request, call_next):
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # pylint: disable=broad-exception-caught
            latency_ms = request_latency_ms(started_at)
            service.record_request(path=request.url.path, status_code=500, latency_ms=latency_ms)
            logger.exception(
                "Inference request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "status_code": 500,
                    "latency_ms": round(latency_ms, 3),
                },
            )
            raise

        latency_ms = request_latency_ms(started_at)
        service.record_request(
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        logger.info(
            "Inference request complete",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 3),
            },
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    async def health() -> JSONResponse:
        status_code, payload = await service.health()
        return JSONResponse(
            status_code=status_code,
            content=payload.model_dump(mode="json"),
        )

    @app.get("/latest-features", response_model=FeatureRowResponse)
    async def latest_features(
        symbol: str,
        interval_begin: str | None = None,
    ) -> FeatureRowResponse:
        row = await _latest_row_or_error(
            service,
            symbol,
            interval_begin=_parse_interval_begin(interval_begin),
        )
        return FeatureRowResponse.model_validate(row)

    @app.get("/predict", response_model=PredictionResponse)
    async def predict(
        symbol: str,
        interval_begin: str | None = None,
    ) -> PredictionResponse:
        parsed_interval_begin = _parse_interval_begin(interval_begin)
        row = await _latest_row_or_error(
            service,
            symbol,
            interval_begin=parsed_interval_begin,
        )
        try:
            freshness = await service.freshness_evaluation(
                symbol=symbol,
                interval_begin=parsed_interval_begin,
            )
            return await service.predict_from_row(row, freshness=freshness)
        except ArtifactSchemaMismatchError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/regime", response_model=RegimeResponse)
    async def regime(
        symbol: str,
        interval_begin: str | None = None,
    ) -> RegimeResponse:
        parsed_interval_begin = _parse_interval_begin(interval_begin)
        row = await _latest_row_or_error(
            service,
            symbol,
            interval_begin=parsed_interval_begin,
        )
        try:
            freshness = await service.freshness_evaluation(
                symbol=symbol,
                interval_begin=parsed_interval_begin,
            )
            return service.regime_from_row(
                row,
                freshness=freshness,
            )
        except ArtifactSchemaMismatchError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/signal", response_model=SignalResponse)
    async def signal(
        symbol: str,
        interval_begin: str | None = None,
    ) -> SignalResponse:
        parsed_interval_begin = _parse_interval_begin(interval_begin)
        try:
            return await service.signal_for_request(
                symbol=symbol,
                interval_begin=parsed_interval_begin,
            )
        except InvalidSymbolError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except DatabaseUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ArtifactSchemaMismatchError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/freshness", response_model=FreshnessResponse)
    async def freshness(
        symbol: str,
        interval_begin: str | None = None,
    ) -> FreshnessResponse:
        parsed_interval_begin = _parse_interval_begin(interval_begin)
        try:
            return await service.freshness_response(
                symbol=symbol,
                interval_begin=parsed_interval_begin,
            )
        except InvalidSymbolError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except DatabaseUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        return await service.metrics_snapshot()

    @app.get("/adaptation/summary", response_model=AdaptationSummaryResponse)
    async def adaptation_summary(
        symbol: str = Query(default="ALL"),
        regime_label: str = Query(default="ALL"),
    ) -> AdaptationSummaryResponse:
        return await service.adaptation_summary(symbol=symbol, regime_label=regime_label)

    @app.get("/adaptation/drift", response_model=AdaptationDriftResponse)
    async def adaptation_drift(
        symbol: str = Query(default="ALL"),
        regime_label: str = Query(default="ALL"),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> AdaptationDriftResponse:
        return await service.adaptation_drift(
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )

    @app.get("/adaptation/performance", response_model=AdaptationPerformanceResponse)
    async def adaptation_performance(
        execution_mode: str = Query(default="ALL"),
        symbol: str = Query(default="ALL"),
        regime_label: str = Query(default="ALL"),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> AdaptationPerformanceResponse:
        return await service.adaptation_performance(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )

    @app.get("/adaptation/profiles", response_model=AdaptationProfilesResponse)
    async def adaptation_profiles(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> AdaptationProfilesResponse:
        return await service.adaptation_profiles(limit=limit)

    @app.get("/adaptation/promotions", response_model=AdaptationPromotionsResponse)
    async def adaptation_promotions(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> AdaptationPromotionsResponse:
        return await service.adaptation_promotions(limit=limit)

    @app.get(
        "/continual-learning/summary",
        response_model=ContinualLearningSummaryResponse,
    )
    async def continual_learning_summary(
        execution_mode: str = Query(default="ALL"),
        symbol: str = Query(default="ALL"),
        regime_label: str = Query(default="ALL"),
    ) -> ContinualLearningSummaryResponse:
        return await service.continual_learning_summary(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )

    @app.get(
        "/continual-learning/experiments",
        response_model=ContinualLearningExperimentsResponse,
    )
    async def continual_learning_experiments(
        execution_mode: str = Query(default="ALL"),
        symbol: str = Query(default="ALL"),
        regime_label: str = Query(default="ALL"),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> ContinualLearningExperimentsResponse:
        return await service.continual_learning_experiments(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )

    @app.get(
        "/continual-learning/profiles",
        response_model=ContinualLearningProfilesResponse,
    )
    async def continual_learning_profiles(
        execution_mode: str = Query(default="ALL"),
        symbol: str = Query(default="ALL"),
        regime_label: str = Query(default="ALL"),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> ContinualLearningProfilesResponse:
        return await service.continual_learning_profiles(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )

    @app.get(
        "/continual-learning/drift-caps",
        response_model=ContinualLearningDriftCapsResponse,
    )
    async def continual_learning_drift_caps(
        execution_mode: str = Query(default="ALL"),
        symbol: str = Query(default="ALL"),
        regime_label: str = Query(default="ALL"),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> ContinualLearningDriftCapsResponse:
        return await service.continual_learning_drift_caps(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )

    @app.get(
        "/continual-learning/promotions",
        response_model=ContinualLearningPromotionsResponse,
    )
    async def continual_learning_promotions(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> ContinualLearningPromotionsResponse:
        return await service.continual_learning_promotions(limit=limit)

    @app.get(
        "/continual-learning/events",
        response_model=ContinualLearningEventsResponse,
    )
    async def continual_learning_events(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> ContinualLearningEventsResponse:
        return await service.continual_learning_events(limit=limit)

    @app.get("/reliability/system", response_model=SystemReliabilityResponse)
    async def reliability_system() -> JSONResponse:
        status_code, snapshot = await service.system_reliability_snapshot()
        payload = SystemReliabilityResponse.model_validate(
            {
                **asdict(snapshot),
                **service.runtime_metadata_fields(),
            }
        )
        return JSONResponse(
            status_code=status_code,
            content=payload.model_dump(mode="json"),
        )

    @app.get("/alerts/active", response_model=list[OperationalAlertStateResponse])
    async def alerts_active() -> list[OperationalAlertStateResponse]:
        try:
            return await service.active_alerts()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/alerts/timeline", response_model=list[OperationalAlertEventResponse])
    async def alerts_timeline(
        limit: int = Query(default=50, ge=1, le=500),
        category: str | None = None,
        severity: str | None = None,
        symbol: str | None = None,
        active_only: bool = False,
    ) -> list[OperationalAlertEventResponse]:
        try:
            return await service.alert_timeline(
                limit=limit,
                category=category,
                severity=severity,
                symbol=symbol,
                active_only=active_only,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get(
        "/operations/daily-summary",
        response_model=DailyOperationsSummaryResponse,
    )
    async def operations_daily_summary(
        summary_date: date | None = Query(default=None, alias="date"),
    ) -> DailyOperationsSummaryResponse:
        try:
            return await service.daily_operations_summary(summary_date=summary_date)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get(
        "/operations/startup-safety",
        response_model=StartupSafetyReportResponse,
    )
    async def operations_startup_safety() -> StartupSafetyReportResponse:
        try:
            return await service.startup_safety_report()
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return app
# pylint: enable=too-many-statements


async def _latest_row_or_error(
    service: InferenceService,
    symbol: str,
    *,
    interval_begin: Any | None = None,
) -> dict[str, Any]:
    try:
        row = await service.latest_feature_row(symbol, interval_begin=interval_begin)
    except InvalidSymbolError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except DatabaseUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No feature row found for symbol {symbol}",
        )
    return row


def _parse_interval_begin(value: str | None) -> Any | None:
    """Parse the optional exact-candle selector for M4 lookups."""
    if value is None:
        return None
    try:
        return parse_rfc3339(value)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
