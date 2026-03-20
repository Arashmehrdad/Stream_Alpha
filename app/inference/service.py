"""Core inference service logic for the Stream Alpha M4 API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib

from app.common.config import Settings
from app.common.time import to_rfc3339, utc_now
from app.inference.db import InferenceDatabase
from app.inference.schemas import HealthResponse, MetricsResponse, RegimeResponse
from app.inference.schemas import ThresholdsResponse
from app.inference.schemas import LatencyStatsResponse, PredictionResponse, SignalResponse
from app.regime.live import LiveRegimeRuntime, load_live_regime_runtime
from app.training.registry import resolve_inference_model_path


class InvalidSymbolError(ValueError):
    """Raised when a request uses a symbol outside the configured Kraken set."""


class ArtifactSchemaMismatchError(RuntimeError):
    """Raised when the saved model artifact cannot score the DB row safely."""


@dataclass(frozen=True, slots=True)
class LoadedModelArtifact:
    """Validated saved M3 model artifact for direct online reuse."""

    model_name: str
    trained_at: str
    feature_columns: tuple[str, ...]
    expanded_feature_names: tuple[str, ...]
    model_artifact_path: str
    model: Any


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


def load_model_artifact(model_path: str) -> LoadedModelArtifact:
    """Load and validate the saved M3 model artifact."""
    artifact_path = Path(resolve_inference_model_path(model_path)).resolve()

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
        feature_columns=feature_columns,
        expanded_feature_names=expanded_feature_names,
        model_artifact_path=str(artifact_path),
        model=model,
    )


class InferenceService:
    """Serve online predictions from the latest canonical feature row."""

    def __init__(
        self,
        settings: Settings,
        *,
        database: InferenceDatabase | None = None,
        model_artifact: LoadedModelArtifact | None = None,
        regime_runtime: LiveRegimeRuntime | None = None,
    ) -> None:
        self.settings = settings
        self.started_at = utc_now()
        self.metrics = MetricsState(started_at=self.started_at)
        self.database = database or InferenceDatabase(
            settings.postgres.dsn,
            settings.tables.feature_ohlc,
        )
        self.model_artifact = model_artifact or load_model_artifact(
            settings.inference.model_path,
        )
        self.regime_runtime = regime_runtime or load_live_regime_runtime(
            thresholds_path=settings.inference.regime_thresholds_path,
            signal_policy_path=settings.inference.regime_signal_policy_path,
        )
        self._symbols = set(settings.kraken.symbols)
        self._validate_thresholds()
        self.regime_runtime.validate_runtime_compatibility(
            source_table=settings.tables.feature_ohlc,
            source_exchange="kraken",
            interval_minutes=settings.kraken.ohlc_interval_minutes,
            symbols=settings.kraken.symbols,
        )

    async def startup(self) -> None:
        """Open the read-only database pool for serving."""
        try:
            await self.database.connect()
        except Exception:  # pylint: disable=broad-exception-caught
            return

    async def shutdown(self) -> None:
        """Close the database pool."""
        await self.database.close()

    def record_request(self, *, path: str, status_code: int, latency_ms: float) -> None:
        """Record one completed HTTP request."""
        self.metrics.record(path=path, status_code=status_code, latency_ms=latency_ms)

    async def health(self) -> tuple[int, HealthResponse]:
        """Return the current dependency health payload and status code."""
        database_healthy = await self.database.is_healthy()
        model_loaded = self.model_artifact is not None
        status_code = 200 if database_healthy and model_loaded else 503
        status_text = "ok" if status_code == 200 else "unavailable"
        return (
            status_code,
            HealthResponse(
                status=status_text,
                service=self.settings.inference.service_name,
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

    def predict_from_row(self, row: dict[str, Any]) -> PredictionResponse:
        """Run predict_proba against one canonical feature row."""
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
        return PredictionResponse(
            symbol=str(row["symbol"]),
            model_name=self.model_artifact.model_name,
            model_trained_at=self.model_artifact.trained_at,
            model_artifact_path=self.model_artifact.model_artifact_path,
            row_id=f"{row['symbol']}|{to_rfc3339(row['interval_begin'])}",
            interval_begin=to_rfc3339(row["interval_begin"]),
            as_of_time=to_rfc3339(row["as_of_time"]),
            prob_up=prob_up,
            prob_down=prob_down,
            predicted_class=predicted_class,
            confidence=max(prob_up, prob_down),
            regime_label=resolved_regime.regime_label,
            regime_run_id=resolved_regime.regime_run_id,
        )

    def signal_from_prediction(self, prediction: PredictionResponse) -> SignalResponse:
        """Convert one prediction into BUY, SELL, or HOLD."""
        policy = self.regime_runtime.policy_for(prediction.regime_label)
        buy_threshold = policy.buy_prob_up
        sell_threshold = policy.sell_prob_up
        if prediction.prob_up >= buy_threshold:
            if policy.allow_new_long_entries:
                signal = "BUY"
                reason = (
                    f"prob_up {prediction.prob_up:.4f} >= buy threshold "
                    f"{buy_threshold:.2f}"
                )
            else:
                signal = "HOLD"
                reason = (
                    f"prob_up {prediction.prob_up:.4f} >= buy threshold "
                    f"{buy_threshold:.2f} but new long entries are disabled in "
                    f"{prediction.regime_label}"
                )
        elif prediction.prob_up <= sell_threshold:
            signal = "SELL"
            reason = f"prob_up {prediction.prob_up:.4f} <= sell threshold {sell_threshold:.2f}"
        else:
            signal = "HOLD"
            reason = (
                f"prob_up {prediction.prob_up:.4f} is between "
                f"{sell_threshold:.2f} and {buy_threshold:.2f}"
            )
        trade_allowed = signal in {"BUY", "SELL"}

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
            regime_label=prediction.regime_label,
            regime_run_id=prediction.regime_run_id,
            trade_allowed=trade_allowed,
        )

    def regime_from_row(self, row: dict[str, Any]) -> RegimeResponse:
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
        )

    def metrics_snapshot(self) -> MetricsResponse:
        """Return the current in-memory metrics payload."""
        uptime_seconds = max(0.0, (utc_now() - self.started_at).total_seconds())
        return MetricsResponse(
            requests_total=self.metrics.requests_total,
            errors_total=self.metrics.errors_total,
            endpoint_counts=dict(self.metrics.endpoint_counts),
            latency_ms=self.metrics.latency.as_response(),
            service=self.settings.inference.service_name,
            started_at=self.started_at,
            uptime_seconds=uptime_seconds,
            model_name=self.model_artifact.model_name,
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

    def _resolve_regime(self, row: dict[str, Any]):
        try:
            return self.regime_runtime.resolve_feature_row_regime(row)
        except ValueError as error:
            raise ArtifactSchemaMismatchError(str(error)) from error


def request_latency_ms(started_at: float) -> float:
    """Return elapsed request time in milliseconds."""
    return (perf_counter() - started_at) * 1000.0
