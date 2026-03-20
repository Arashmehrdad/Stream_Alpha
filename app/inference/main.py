"""FastAPI app factory for the Stream Alpha M4 inference service."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.common.config import Settings
from app.common.logging import configure_logging
from app.common.time import parse_rfc3339
from app.inference.db import DatabaseUnavailableError
from app.inference.schemas import FeatureRowResponse, HealthResponse, MetricsResponse
from app.inference.schemas import PredictionResponse, SignalResponse
from app.inference.service import (
    ArtifactSchemaMismatchError,
    InferenceService,
    InvalidSymbolError,
    request_latency_ms,
)


def create_app(service: InferenceService | None = None) -> FastAPI:
    """Create the M4 FastAPI app with strict model loading."""
    if service is None:
        settings = Settings.from_env()
        configure_logging(settings.log_level)
        service = InferenceService(settings)

    logger = logging.getLogger(f"{service.settings.app_name}.inference")
    app = FastAPI(title="Stream Alpha Inference API", version="m4")
    app.state.service = service

    @app.on_event("startup")
    async def _startup() -> None:
        await service.startup()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await service.shutdown()

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
        row = await _latest_row_or_error(
            service,
            symbol,
            interval_begin=_parse_interval_begin(interval_begin),
        )
        try:
            return service.predict_from_row(row)
        except ArtifactSchemaMismatchError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/signal", response_model=SignalResponse)
    async def signal(
        symbol: str,
        interval_begin: str | None = None,
    ) -> SignalResponse:
        row = await _latest_row_or_error(
            service,
            symbol,
            interval_begin=_parse_interval_begin(interval_begin),
        )
        try:
            prediction = service.predict_from_row(row)
            return service.signal_from_prediction(prediction)
        except ArtifactSchemaMismatchError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        return service.metrics_snapshot()

    return app


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
