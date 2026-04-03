"""Shared pretrained forecaster contracts, metadata helpers, and Chronos-2 wrapper."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.training.dataset import (
    SEQUENCE_CONTEXT_KEY,
    DatasetSample,
    SourceFeatureRow,
    build_sequence_context_rows,
    future_target_timestamp,
    infer_source_frequency_minutes,
)


ProgressCallback = Callable[[dict[str, Any]], None]

GENERALIST = "GENERALIST"
TREND_SPECIALIST = "TREND_SPECIALIST"
RANGE_SPECIALIST = "RANGE_SPECIALIST"

KNOWN_CANDIDATE_ROLES = frozenset({GENERALIST, TREND_SPECIALIST, RANGE_SPECIALIST})

DEFAULT_SCOPE_REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL")
TREND_SCOPE_REGIMES = ("TREND_UP", "TREND_DOWN")
RANGE_SCOPE_REGIMES = ("RANGE",)

MODEL_FAMILY_AUTOGLUON = "AUTOGLUON"
MODEL_FAMILY_REGISTRY_CHAMPION_BASELINE = "REGISTRY_CHAMPION_BASELINE"
MODEL_FAMILY_NEURALFORECAST_NHITS = "NEURALFORECAST_NHITS"
MODEL_FAMILY_NEURALFORECAST_NBEATSX = "NEURALFORECAST_NBEATSX"
MODEL_FAMILY_NEURALFORECAST_TFT = "NEURALFORECAST_TFT"
MODEL_FAMILY_NEURALFORECAST_PATCHTST = "NEURALFORECAST_PATCHTST"
MODEL_FAMILY_AMAZON_CHRONOS_2 = "AMAZON_CHRONOS_2"
MODEL_FAMILY_GOOGLE_TIMESFM_2_0_500M_PYTORCH = "GOOGLE_TIMESFM_2_0_500M_PYTORCH"
MODEL_FAMILY_MOIRAI_SMALL = "MOIRAI_SMALL"
MODEL_FAMILY_MOIRAI_BASE = "MOIRAI_BASE"

DEFAULT_PRETRAINED_ARTIFACT_FORMAT = "SELF_CONTAINED_JOBLIB"
HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT = "HF_REFERENCE_JOBLIB"
DEFAULT_PRETRAINED_CALIBRATION_METHOD = "RAW_SCORE_TO_PROB_UP"

DEFAULT_CHRONOS_2_PRETRAINED_SOURCE = "amazon/chronos-2"
DEFAULT_CHRONOS_2_LICENSE_NAME = "Apache-2.0"
_DEFAULT_CHRONOS_2_LOOKBACK_CANDLES = 96
_DEFAULT_CHRONOS_2_CALIBRATION_WINDOWS = 256
_DEFAULT_CHRONOS_2_BATCH_SIZE = 32
_DEFAULT_CHRONOS_2_QUANTILE_LEVELS = (0.1, 0.5, 0.9)
DEFAULT_TIMESFM_2_0_500M_PYTORCH_PRETRAINED_SOURCE = "google/timesfm-2.0-500m-pytorch"
DEFAULT_TIMESFM_2_0_LICENSE_NAME = "Apache-2.0"
_DEFAULT_TIMESFM_LOOKBACK_CANDLES = 256
_DEFAULT_TIMESFM_MAX_CONTEXT_LEN = 2048
_DEFAULT_TIMESFM_CALIBRATION_WINDOWS = 256
_DEFAULT_TIMESFM_BATCH_SIZE = 32
_DEFAULT_TIMESFM_INPUT_PATCH_LEN = 32
_DEFAULT_TIMESFM_OUTPUT_PATCH_LEN = 128
_DEFAULT_TIMESFM_NUM_LAYERS = 50
_DEFAULT_TIMESFM_MODEL_DIMS = 1280
_DEFAULT_TIMESFM_POINT_FORECAST_MODE = "median"
# We intentionally target the sktime-hosted Apache-2.0 checkpoint snapshot here
# instead of the Salesforce CC-BY-NC checkpoint family, so license constraints
# stay explicit and permissive in saved registry metadata.
DEFAULT_MOIRAI_1_0_R_BASE_PRETRAINED_SOURCE = "sktime/moirai-1.0-R-base"
DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NAME = "Apache-2.0"
DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NOTES = (
    "Using the sktime Apache-2.0 snapshot checkpoint; not the Salesforce "
    "CC-BY-NC-4.0 checkpoint family."
)
_DEFAULT_MOIRAI_LOOKBACK_CANDLES = 200
_DEFAULT_MOIRAI_PATCH_SIZE = 32
_DEFAULT_MOIRAI_NUM_SAMPLES = 100
_DEFAULT_MOIRAI_BATCH_SIZE = 32

_COMMON_SPECIALIST_MODEL_FAMILIES = frozenset(
    {
        MODEL_FAMILY_NEURALFORECAST_NHITS,
        MODEL_FAMILY_NEURALFORECAST_NBEATSX,
        MODEL_FAMILY_NEURALFORECAST_TFT,
        MODEL_FAMILY_NEURALFORECAST_PATCHTST,
    }
)

PACKET2_GENERALIST_MODEL_FAMILIES = frozenset(
    {
        MODEL_FAMILY_AUTOGLUON,
        MODEL_FAMILY_AMAZON_CHRONOS_2,
        MODEL_FAMILY_REGISTRY_CHAMPION_BASELINE,
    }
)

PACKET2_TREND_SPECIALIST_MODEL_FAMILIES = frozenset(
    {
        *_COMMON_SPECIALIST_MODEL_FAMILIES,
        MODEL_FAMILY_GOOGLE_TIMESFM_2_0_500M_PYTORCH,
    }
)

PACKET2_RANGE_SPECIALIST_MODEL_FAMILIES = frozenset(
    {
        *_COMMON_SPECIALIST_MODEL_FAMILIES,
        MODEL_FAMILY_MOIRAI_SMALL,
        MODEL_FAMILY_MOIRAI_BASE,
    }
)


@dataclass(frozen=True, slots=True)
class PretrainedForecasterArtifactMetadata:
    """Stable metadata carried through training artifacts and registry entries."""

    model_family: str
    candidate_role: str
    scope_regimes: tuple[str, ...]
    pretrained_source: str | None = None
    artifact_format: str = DEFAULT_PRETRAINED_ARTIFACT_FORMAT
    calibration_method: str = DEFAULT_PRETRAINED_CALIBRATION_METHOD
    license_name: str | None = None
    license_notes: str | None = None

    def to_registry_metadata(self) -> dict[str, Any]:
        """Return the candidate-discovery metadata persisted with this artifact."""
        metadata = {
            "model_family": str(self.model_family),
            "candidate_role": str(self.candidate_role),
            "scope_regimes": [str(regime) for regime in self.scope_regimes],
            "artifact_format": str(self.artifact_format),
            "calibration_method": str(self.calibration_method),
        }
        if self.pretrained_source is not None:
            metadata["pretrained_source"] = str(self.pretrained_source)
        if self.license_name is not None:
            metadata["license_name"] = str(self.license_name)
        if self.license_notes is not None:
            metadata["license_notes"] = str(self.license_notes)
        return metadata

    def to_training_config(self) -> dict[str, Any]:
        """Return the stable metadata fields that also belong in training config truth."""
        return dict(self.to_registry_metadata())


@dataclass(frozen=True, slots=True)
class ValidatedPretrainedForecasterContract:
    """Validated shared contract for pretrained challenger wrappers."""

    feature_columns: tuple[str, ...]
    expanded_feature_names: tuple[str, ...]
    training_config: dict[str, Any]
    registry_metadata: dict[str, Any]
    requires_sequence_context: bool
    sequence_lookback_candles: int | None


class PretrainedForecaster(Protocol):
    """Generic pretrained forecaster contract for staged challengers."""

    def fit_samples(
        self,
        samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        dataset_export_root: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """Fit any calibration or local adaptation needed for this wrapper."""

    def predict(self, rows: list[dict[str, Any]]) -> list[int]:
        """Predict binary labels from scoring rows."""

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        """Predict binary probabilities from scoring rows."""

    def predict_samples(
        self,
        test_samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> list[int]:
        """Predict binary labels from dataset samples."""

    def predict_proba_samples(
        self,
        test_samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Predict binary probabilities from dataset samples."""

    def requires_sequence_context(self) -> bool:
        """Return whether this artifact needs ordered lookback history at scoring time."""

    def get_sequence_lookback_candles(self) -> int:
        """Return the exact lookback required for runtime scoring."""

    def get_feature_columns(self) -> list[str]:
        """Return the saved feature schema."""

    def get_expanded_feature_names(self) -> list[str]:
        """Return the saved expanded feature schema."""

    def get_training_config(self) -> dict[str, Any]:
        """Return audit-stable training config metadata."""

    def get_registry_metadata(
        self,
    ) -> dict[str, Any] | PretrainedForecasterArtifactMetadata:
        """Return registry discovery metadata."""


class ForecastToProbabilityCalibrator:
    """Serializable calibrated bridge from raw forecast score to P(UP)."""

    def __init__(self) -> None:
        self._mode = "constant"
        self._constant_prob_up = 0.5
        self._model: LogisticRegression | None = None

    def fit(self, raw_scores: list[float], labels: list[int]) -> "ForecastToProbabilityCalibrator":
        """Fit a bounded probability bridge on raw forecasted-return scores."""
        if not labels:
            self._mode = "constant"
            self._constant_prob_up = 0.5
            self._model = None
            return self

        clipped_labels = [int(label) for label in labels]
        empirical_prob_up = float(sum(clipped_labels)) / float(len(clipped_labels))
        if len(set(clipped_labels)) < 2 or len(set(raw_scores)) < 2:
            self._mode = "constant"
            self._constant_prob_up = empirical_prob_up
            self._model = None
            return self

        logistic_model = LogisticRegression(random_state=0, solver="lbfgs")
        features = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        logistic_model.fit(features, np.asarray(clipped_labels, dtype=int))
        self._mode = "logistic"
        self._constant_prob_up = empirical_prob_up
        self._model = logistic_model
        return self

    def predict_proba(self, raw_scores: list[float]) -> list[list[float]]:
        """Return binary probabilities in the repository's saved-artifact contract."""
        if self._mode == "constant" or self._model is None:
            return [
                [1.0 - self._constant_prob_up, self._constant_prob_up]
                for _ in raw_scores
            ]

        features = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        probabilities = self._model.predict_proba(features)
        return [
            [float(row[0]), float(row[1])]
            for row in probabilities.tolist()
        ]


def is_pretrained_forecaster_model(model: Any) -> bool:
    """Return whether a model exposes the staged pretrained forecaster hooks."""
    return all(
        hasattr(model, attribute_name)
        for attribute_name in (
            "fit_samples",
            "predict_proba_samples",
        )
    )


def validate_pretrained_forecaster_contract(
    model: PretrainedForecaster,
) -> ValidatedPretrainedForecasterContract:
    """Validate the shared pretrained wrapper contract before saving or loading."""
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Pretrained forecaster artifacts must expose predict_proba",
        )
    feature_columns = tuple(str(column) for column in model.get_feature_columns())
    if not feature_columns:
        raise ValueError("Pretrained forecaster artifacts must expose feature columns")

    expanded_feature_names = tuple(
        str(name) for name in model.get_expanded_feature_names()
    )
    if not expanded_feature_names:
        raise ValueError(
            "Pretrained forecaster artifacts must expose expanded feature names",
        )

    training_config = model.get_training_config()
    if not isinstance(training_config, dict):
        raise ValueError(
            "Pretrained forecaster artifacts must expose dictionary training config",
        )
    resolved_training_config = dict(training_config)

    registry_metadata = normalize_pretrained_registry_metadata(model.get_registry_metadata())
    for key, value in registry_metadata.items():
        resolved_training_config.setdefault(key, value)

    requires_sequence_context = bool(model.requires_sequence_context())
    sequence_lookback_candles: int | None = None
    if requires_sequence_context:
        sequence_lookback_candles = int(model.get_sequence_lookback_candles())
        if sequence_lookback_candles <= 0:
            raise ValueError(
                "Pretrained forecaster sequence lookback candles must be positive",
            )

    return ValidatedPretrainedForecasterContract(
        feature_columns=feature_columns,
        expanded_feature_names=expanded_feature_names,
        training_config=resolved_training_config,
        registry_metadata=registry_metadata,
        requires_sequence_context=requires_sequence_context,
        sequence_lookback_candles=sequence_lookback_candles,
    )


def normalize_pretrained_registry_metadata(
    metadata: dict[str, Any] | PretrainedForecasterArtifactMetadata,
) -> dict[str, Any]:
    """Normalize saved registry metadata into the stable Packet 2 discovery shape."""
    if isinstance(metadata, PretrainedForecasterArtifactMetadata):
        payload = metadata.to_registry_metadata()
    elif isinstance(metadata, dict):
        payload = dict(metadata)
    else:
        raise ValueError(
            "Pretrained forecaster registry metadata must be a dictionary payload",
        )

    model_family = str(payload.get("model_family", "")).strip()
    if not model_family:
        raise ValueError(
            "Pretrained forecaster registry metadata must include model_family",
        )

    candidate_role = str(payload.get("candidate_role", "")).strip()
    if candidate_role not in KNOWN_CANDIDATE_ROLES:
        raise ValueError(
            "Pretrained forecaster registry metadata must include a supported "
            f"candidate_role, got {candidate_role!r}",
        )

    scope_regimes_payload = payload.get("scope_regimes")
    if scope_regimes_payload is None:
        scope_regimes = _default_scope_regimes(candidate_role)
    else:
        scope_regimes = tuple(str(regime) for regime in scope_regimes_payload)
    if not scope_regimes:
        raise ValueError(
            "Pretrained forecaster registry metadata must include scope_regimes",
        )

    normalized = {
        "model_family": model_family,
        "candidate_role": candidate_role,
        "scope_regimes": list(scope_regimes),
        "artifact_format": str(
            payload.get("artifact_format", DEFAULT_PRETRAINED_ARTIFACT_FORMAT)
        ),
        "calibration_method": str(
            payload.get(
                "calibration_method",
                DEFAULT_PRETRAINED_CALIBRATION_METHOD,
            )
        ),
    }
    for optional_key in ("pretrained_source", "license_name", "license_notes"):
        optional_value = payload.get(optional_key)
        if optional_value is None:
            continue
        rendered = str(optional_value).strip()
        if rendered:
            normalized[optional_key] = rendered
    return normalized


def _default_scope_regimes(candidate_role: str) -> tuple[str, ...]:
    if candidate_role == GENERALIST:
        return DEFAULT_SCOPE_REGIMES
    if candidate_role == TREND_SPECIALIST:
        return TREND_SCOPE_REGIMES
    if candidate_role == RANGE_SPECIALIST:
        return RANGE_SCOPE_REGIMES
    raise ValueError(f"Unsupported candidate role for scope resolution: {candidate_role}")


class Chronos2Forecaster:  # pylint: disable=too-many-instance-attributes
    """Amazon Chronos-2 challenger wrapper with calibrated P(UP) output."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        pretrained_source: str = DEFAULT_CHRONOS_2_PRETRAINED_SOURCE,
        candidate_role: str = GENERALIST,
        scope_regimes: tuple[str, ...] = DEFAULT_SCOPE_REGIMES,
        horizon_candles: int = 3,
        context_lookback_candles: int = _DEFAULT_CHRONOS_2_LOOKBACK_CANDLES,
        calibration_windows: int = _DEFAULT_CHRONOS_2_CALIBRATION_WINDOWS,
        prediction_batch_size: int = _DEFAULT_CHRONOS_2_BATCH_SIZE,
        quantile_levels: tuple[float, ...] = _DEFAULT_CHRONOS_2_QUANTILE_LEVELS,
        covariate_columns: tuple[str, ...] | None = None,
        device_map: str = "auto",
        torch_dtype: str | None = "auto",
        cache_dir: str | None = None,
    ) -> None:
        self.model_family = MODEL_FAMILY_AMAZON_CHRONOS_2
        self.pretrained_source = str(pretrained_source)
        self.candidate_role = str(candidate_role)
        self.scope_regimes = tuple(str(regime) for regime in scope_regimes)
        self.horizon_candles = int(horizon_candles)
        self.context_lookback_candles = int(context_lookback_candles)
        self.calibration_windows = int(calibration_windows)
        self.prediction_batch_size = int(prediction_batch_size)
        self.quantile_levels = tuple(float(level) for level in quantile_levels)
        self.covariate_columns = (
            None
            if covariate_columns is None
            else tuple(str(column) for column in covariate_columns)
        )
        self.device_map = str(device_map)
        self.torch_dtype = None if torch_dtype is None else str(torch_dtype)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self._feature_columns: tuple[str, ...] = ()
        self._resolved_covariate_columns: tuple[str, ...] = ()
        self._frequency_minutes: int | None = None
        self._calibrator: ForecastToProbabilityCalibrator | None = None
        self._pipeline: Any | None = None

    def fit(
        self,
        rows: list[dict[str, Any]],
        labels: list[int],
    ) -> "Chronos2Forecaster":
        """Reject flat-row fitting so Chronos stays sequence-honest."""
        del rows, labels
        raise ValueError("Chronos-2 requires fit_samples(...) with ordered source history")

    def fit_samples(
        self,
        samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        dataset_export_root: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> "Chronos2Forecaster":
        """Load the pretrained runtime and fit the forecast-to-probability bridge."""
        del dataset_export_root
        if not samples:
            raise ValueError("Chronos-2 cannot fit without labeled samples")
        if not source_rows:
            raise ValueError("Chronos-2 cannot fit without ordered source feature rows")

        self._feature_columns = _resolve_feature_columns(source_rows)
        self._resolved_covariate_columns = _resolve_covariate_columns(
            source_rows,
            feature_columns=self._feature_columns,
            configured_covariate_columns=self.covariate_columns,
        )
        self._frequency_minutes = infer_source_frequency_minutes(source_rows)
        self._ensure_pipeline()
        self._calibrator = self._fit_calibrator(
            samples=samples,
            source_rows=source_rows,
            progress_callback=progress_callback,
        )
        return self

    def predict(self, rows: list[dict[str, Any]]) -> list[int]:
        """Return binary labels from calibrated Chronos-2 probabilities."""
        probabilities = self.predict_proba(rows)
        return [1 if row[1] >= 0.5 else 0 for row in probabilities]

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        """Return calibrated probabilities from explicit sequence contexts."""
        raw_scores = self._predict_raw_scores_from_context_rows(rows)
        return self._require_calibrator().predict_proba(raw_scores)

    def predict_samples(
        self,
        test_samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> list[int]:
        """Return binary labels from calibrated Chronos-2 probabilities."""
        probabilities = self.predict_proba_samples(
            test_samples,
            source_rows=source_rows,
            progress_callback=progress_callback,
        )
        return [1 if row[1] >= 0.5 else 0 for row in probabilities]

    def predict_proba_samples(
        self,
        test_samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Return calibrated probabilities from ordered training samples."""
        if not test_samples:
            return []
        context_rows = build_sequence_context_rows(
            target_samples=test_samples,
            source_rows=source_rows,
            feature_columns=self._require_feature_columns(),
            lookback_candles=self.context_lookback_candles,
        )
        raw_scores = self._predict_raw_scores_from_context_rows(
            context_rows,
            progress_callback=progress_callback,
        )
        return self._require_calibrator().predict_proba(raw_scores)

    def requires_sequence_context(self) -> bool:
        """Chronos-2 needs ordered lookback rows to score honestly."""
        return True

    def get_sequence_lookback_candles(self) -> int:
        """Return the saved runtime lookback contract."""
        return self.context_lookback_candles

    def get_feature_columns(self) -> list[str]:
        """Return the saved feature schema."""
        return list(self._feature_columns)

    def get_expanded_feature_names(self) -> list[str]:
        """Return the saved expanded feature schema."""
        return list(self._feature_columns)

    def get_training_config(self) -> dict[str, Any]:
        """Return audit-stable Chronos-2 wrapper metadata."""
        return {
            "model_family": self.model_family,
            "candidate_role": self.candidate_role,
            "scope_regimes": list(self.scope_regimes),
            "pretrained_source": self.pretrained_source,
            "license_name": DEFAULT_CHRONOS_2_LICENSE_NAME,
            "artifact_format": HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT,
            "calibration_method": DEFAULT_PRETRAINED_CALIBRATION_METHOD,
            "horizon_candles": self.horizon_candles,
            "context_lookback_candles": self.context_lookback_candles,
            "calibration_windows": self.calibration_windows,
            "prediction_batch_size": self.prediction_batch_size,
            "quantile_levels": list(self.quantile_levels),
            "covariate_columns": list(self.covariate_columns or ()),
            "resolved_covariate_columns": list(self._resolved_covariate_columns),
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "cache_dir": self.cache_dir,
        }

    def get_registry_metadata(self) -> PretrainedForecasterArtifactMetadata:
        """Return truthful Packet 2 discovery metadata for Chronos-2."""
        return PretrainedForecasterArtifactMetadata(
            model_family=self.model_family,
            candidate_role=self.candidate_role,
            scope_regimes=self.scope_regimes,
            pretrained_source=self.pretrained_source,
            artifact_format=HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT,
            calibration_method=DEFAULT_PRETRAINED_CALIBRATION_METHOD,
            license_name=DEFAULT_CHRONOS_2_LICENSE_NAME,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Persist the smallest honest compatible artifact form."""
        return {
            "model_family": self.model_family,
            "pretrained_source": self.pretrained_source,
            "candidate_role": self.candidate_role,
            "scope_regimes": list(self.scope_regimes),
            "horizon_candles": self.horizon_candles,
            "context_lookback_candles": self.context_lookback_candles,
            "calibration_windows": self.calibration_windows,
            "prediction_batch_size": self.prediction_batch_size,
            "quantile_levels": list(self.quantile_levels),
            "covariate_columns": list(self.covariate_columns or ()),
            "resolved_covariate_columns": list(self._resolved_covariate_columns),
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "cache_dir": self.cache_dir,
            "feature_columns": list(self._feature_columns),
            "frequency_minutes": self._frequency_minutes,
            "calibrator": self._calibrator,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the lazy-load wrapper without embedding model weights."""
        self.model_family = str(state["model_family"])
        self.pretrained_source = str(state["pretrained_source"])
        self.candidate_role = str(state["candidate_role"])
        self.scope_regimes = tuple(str(regime) for regime in state["scope_regimes"])
        self.horizon_candles = int(state["horizon_candles"])
        self.context_lookback_candles = int(state["context_lookback_candles"])
        self.calibration_windows = int(state["calibration_windows"])
        self.prediction_batch_size = int(state["prediction_batch_size"])
        self.quantile_levels = tuple(float(level) for level in state["quantile_levels"])
        configured_covariates = state.get("covariate_columns") or []
        self.covariate_columns = tuple(str(column) for column in configured_covariates)
        self._resolved_covariate_columns = tuple(
            str(column)
            for column in state.get("resolved_covariate_columns", [])
        )
        self.device_map = str(state["device_map"])
        self.torch_dtype = (
            None if state.get("torch_dtype") is None else str(state["torch_dtype"])
        )
        self.cache_dir = None if state.get("cache_dir") is None else str(state["cache_dir"])
        self._feature_columns = tuple(str(column) for column in state["feature_columns"])
        self._frequency_minutes = (
            None
            if state.get("frequency_minutes") is None
            else int(state["frequency_minutes"])
        )
        self._calibrator = state["calibrator"]
        self._pipeline = None

    def _fit_calibrator(
        self,
        *,
        samples: list[DatasetSample],
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> ForecastToProbabilityCalibrator:
        calibration_samples = self._select_calibration_samples(
            samples=samples,
            source_rows=source_rows,
        )
        labels = [int(sample.label) for sample in calibration_samples]
        if not calibration_samples:
            fallback_labels = [int(sample.label) for sample in samples]
            return ForecastToProbabilityCalibrator().fit(
                [0.0 for _ in fallback_labels],
                fallback_labels,
            )

        context_rows = build_sequence_context_rows(
            target_samples=calibration_samples,
            source_rows=source_rows,
            feature_columns=self._require_feature_columns(),
            lookback_candles=self.context_lookback_candles,
        )
        raw_scores = self._predict_raw_scores_from_context_rows(
            context_rows,
            progress_callback=progress_callback,
        )
        return ForecastToProbabilityCalibrator().fit(raw_scores, labels)

    def _select_calibration_samples(
        self,
        *,
        samples: list[DatasetSample],
        source_rows: list[SourceFeatureRow],
    ) -> list[DatasetSample]:
        if self.calibration_windows <= 0:
            return []
        eligible_samples: list[DatasetSample] = []
        for sample in samples:
            history_count = sum(
                1
                for source_row in source_rows
                if source_row.symbol == sample.symbol
                and source_row.as_of_time <= sample.as_of_time
            )
            if history_count >= self.context_lookback_candles:
                eligible_samples.append(sample)
        if not eligible_samples:
            return []
        return eligible_samples[-min(len(eligible_samples), self.calibration_windows) :]

    def _predict_raw_scores_from_context_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[float]:
        if not rows:
            return []
        pipeline = self._ensure_pipeline()
        batch_size = max(1, self.prediction_batch_size)
        total_rows = len(rows)
        total_batches = (total_rows + batch_size - 1) // batch_size
        raw_scores: list[float] = []

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sequence_scoring_start",
                    "row_count": total_rows,
                    "batch_count": total_batches,
                    "batch_size": batch_size,
                }
            )

        for batch_index, batch_start in enumerate(range(0, total_rows, batch_size), start=1):
            batch_rows = rows[batch_start : batch_start + batch_size]
            context_frame, expected_targets = self._build_context_frame(
                batch_rows,
                batch_offset=batch_start,
            )
            prediction_frame = pipeline.predict_df(
                context_frame,
                prediction_length=self.horizon_candles,
                quantile_levels=list(self.quantile_levels),
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
            raw_scores.extend(
                self._extract_raw_scores_from_prediction_frame(
                    prediction_frame,
                    expected_targets=expected_targets,
                )
            )
            if progress_callback is not None:
                completed_rows = min(total_rows, batch_start + len(batch_rows))
                progress_callback(
                    {
                        "event": "sequence_scoring_progress",
                        "row_count": total_rows,
                        "completed_rows": completed_rows,
                        "batch_count": total_batches,
                        "completed_batches": batch_index,
                        "progress": batch_index / float(total_batches),
                    }
                )

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sequence_scoring_complete",
                    "row_count": total_rows,
                    "batch_count": total_batches,
                }
            )
        return raw_scores

    def _build_context_frame(
        self,
        rows: list[dict[str, Any]],
        *,
        batch_offset: int,
    ) -> tuple[Any, list[tuple[str, Any, float]]]:
        pandas = _import_pandas()
        records: list[dict[str, Any]] = []
        expected_targets: list[tuple[str, Any, float]] = []
        covariate_columns = self._resolved_covariate_columns
        for row_index, row in enumerate(rows):
            context_rows = row.get(SEQUENCE_CONTEXT_KEY)
            if not isinstance(context_rows, list) or not context_rows:
                raise ValueError(
                    "Chronos-2 scoring rows must include non-empty "
                    f"{SEQUENCE_CONTEXT_KEY} lookback data",
                )
            synthetic_id = f"ctx_{batch_offset + row_index:06d}"
            current_close = float(context_rows[-1]["close_price"])
            for context_row in context_rows:
                record = {
                    "id": synthetic_id,
                    "timestamp": context_row["as_of_time"],
                    "target": float(context_row["close_price"]),
                }
                for column in covariate_columns:
                    record[column] = float(context_row[column])
                records.append(record)
            expected_targets.append(
                (
                    synthetic_id,
                    pandas.Timestamp(
                        future_target_timestamp(
                            as_of_time=context_rows[-1]["as_of_time"],
                            horizon_candles=self.horizon_candles,
                            frequency_minutes=self._require_frequency_minutes(),
                        )
                    ),
                    current_close,
                )
            )
        return pandas.DataFrame.from_records(records), expected_targets

    def _extract_raw_scores_from_prediction_frame(
        self,
        prediction_frame: Any,
        *,
        expected_targets: list[tuple[str, Any, float]],
    ) -> list[float]:
        pandas = _import_pandas()
        frame = (
            prediction_frame
            if isinstance(prediction_frame, pandas.DataFrame)
            else pandas.DataFrame(prediction_frame)
        )
        normalized = self._normalize_prediction_frame(frame)
        prediction_column = self._resolve_prediction_column(normalized)
        lookup = {
            (str(row["id"]), pandas.Timestamp(row["timestamp"])): float(row[prediction_column])
            for row in normalized.to_dict("records")
        }
        raw_scores: list[float] = []
        for series_id, target_timestamp, current_close in expected_targets:
            forecast_value = lookup.get((series_id, pandas.Timestamp(target_timestamp)))
            if forecast_value is None:
                raise ValueError(
                    "Chronos-2 forecast output is missing the expected horizon row for "
                    f"{series_id} at {pandas.Timestamp(target_timestamp).isoformat()}"
                )
            raw_scores.append((forecast_value / current_close) - 1.0)
        return raw_scores

    def _normalize_prediction_frame(self, frame: Any) -> Any:
        pandas = _import_pandas()
        normalized = frame.copy()
        rename_map: dict[str, str] = {}
        if "id" not in normalized.columns:
            for candidate in ("unique_id", "series_id"):
                if candidate in normalized.columns:
                    rename_map[candidate] = "id"
                    break
        if "timestamp" not in normalized.columns:
            for candidate in ("ds", "time"):
                if candidate in normalized.columns:
                    rename_map[candidate] = "timestamp"
                    break
        if rename_map:
            normalized = normalized.rename(columns=rename_map)
        if "id" not in normalized.columns or "timestamp" not in normalized.columns:
            normalized = normalized.reset_index()
            if "id" not in normalized.columns and "unique_id" in normalized.columns:
                normalized = normalized.rename(columns={"unique_id": "id"})
            if "timestamp" not in normalized.columns and "ds" in normalized.columns:
                normalized = normalized.rename(columns={"ds": "timestamp"})
        if "id" not in normalized.columns or "timestamp" not in normalized.columns:
            raise ValueError(
                "Chronos-2 forecast output is missing id/timestamp columns after normalization",
            )
        normalized["id"] = normalized["id"].astype(str)
        normalized["timestamp"] = pandas.to_datetime(normalized["timestamp"])
        return normalized

    def _resolve_prediction_column(self, frame: Any) -> Any:
        for candidate in ("predictions", "0.5", 0.5, "median"):
            if candidate in frame.columns:
                return candidate
        excluded_columns = {"id", "timestamp"}
        quantile_columns = [
            column
            for column in frame.columns
            if str(column) not in excluded_columns
        ]
        if len(quantile_columns) == 1:
            return quantile_columns[0]
        raise ValueError(
            "Chronos-2 forecast output is missing a recoverable prediction column",
        )

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            pipeline_class = _import_chronos_runtime()
            load_kwargs: dict[str, Any] = {
                "device_map": self.device_map,
            }
            resolved_torch_dtype = _resolve_torch_dtype(self.torch_dtype)
            if resolved_torch_dtype is not None:
                load_kwargs["torch_dtype"] = resolved_torch_dtype
            if self.cache_dir is not None:
                load_kwargs["cache_dir"] = self.cache_dir
            self._pipeline = pipeline_class.from_pretrained(
                self.pretrained_source,
                **load_kwargs,
            )
        return self._pipeline

    def _require_calibrator(self) -> ForecastToProbabilityCalibrator:
        if self._calibrator is None:
            raise ValueError("Chronos-2 artifact has not been fit/calibrated yet")
        return self._calibrator

    def _require_feature_columns(self) -> tuple[str, ...]:
        if not self._feature_columns:
            raise ValueError("Chronos-2 artifact is missing feature columns")
        return self._feature_columns

    def _require_frequency_minutes(self) -> int:
        if self._frequency_minutes is None:
            raise ValueError("Chronos-2 artifact is missing source frequency metadata")
        return self._frequency_minutes


class TimesFm2Forecaster:  # pylint: disable=too-many-instance-attributes
    """Google TimesFM 2.0 500M PyTorch trend-specialist wrapper."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        pretrained_source: str = DEFAULT_TIMESFM_2_0_500M_PYTORCH_PRETRAINED_SOURCE,
        candidate_role: str = TREND_SPECIALIST,
        scope_regimes: tuple[str, ...] = TREND_SCOPE_REGIMES,
        horizon_candles: int = 3,
        context_lookback_candles: int = _DEFAULT_TIMESFM_LOOKBACK_CANDLES,
        max_context_len: int = _DEFAULT_TIMESFM_MAX_CONTEXT_LEN,
        calibration_windows: int = _DEFAULT_TIMESFM_CALIBRATION_WINDOWS,
        prediction_batch_size: int = _DEFAULT_TIMESFM_BATCH_SIZE,
        backend: str = "auto",
        per_core_batch_size: int = _DEFAULT_TIMESFM_BATCH_SIZE,
        input_patch_len: int = _DEFAULT_TIMESFM_INPUT_PATCH_LEN,
        output_patch_len: int = _DEFAULT_TIMESFM_OUTPUT_PATCH_LEN,
        num_layers: int = _DEFAULT_TIMESFM_NUM_LAYERS,
        model_dims: int = _DEFAULT_TIMESFM_MODEL_DIMS,
        use_positional_embedding: bool = False,
        point_forecast_mode: str = _DEFAULT_TIMESFM_POINT_FORECAST_MODE,
        normalize_inputs: bool = True,
    ) -> None:
        self.model_family = MODEL_FAMILY_GOOGLE_TIMESFM_2_0_500M_PYTORCH
        self.pretrained_source = str(pretrained_source)
        self.candidate_role = str(candidate_role)
        self.scope_regimes = tuple(str(regime) for regime in scope_regimes)
        self.horizon_candles = int(horizon_candles)
        self.context_lookback_candles = int(context_lookback_candles)
        self.max_context_len = int(max_context_len)
        self.calibration_windows = int(calibration_windows)
        self.prediction_batch_size = int(prediction_batch_size)
        self.backend = str(backend)
        self.per_core_batch_size = int(per_core_batch_size)
        self.input_patch_len = int(input_patch_len)
        self.output_patch_len = int(output_patch_len)
        self.num_layers = int(num_layers)
        self.model_dims = int(model_dims)
        self.use_positional_embedding = bool(use_positional_embedding)
        self.point_forecast_mode = str(point_forecast_mode)
        self.normalize_inputs = bool(normalize_inputs)
        self._feature_columns: tuple[str, ...] = ("symbol", "close_price")
        self._frequency_minutes: int | None = None
        self._frequency_indicator: int | None = None
        self._resolved_backend: str | None = None
        self._calibrator: ForecastToProbabilityCalibrator | None = None
        self._model: Any | None = None

    def fit(
        self,
        rows: list[dict[str, Any]],
        labels: list[int],
    ) -> "TimesFm2Forecaster":
        """Reject flat-row fitting so TimesFM stays sequence-honest."""
        del rows, labels
        raise ValueError("TimesFM requires fit_samples(...) with ordered source history")

    def fit_samples(
        self,
        samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        dataset_export_root: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> "TimesFm2Forecaster":
        """Load the pretrained TimesFM runtime and fit the P(UP) bridge."""
        del dataset_export_root
        if not samples:
            raise ValueError("TimesFM cannot fit without labeled samples")
        if not source_rows:
            raise ValueError("TimesFM cannot fit without ordered source feature rows")

        self._frequency_minutes = infer_source_frequency_minutes(source_rows)
        self._frequency_indicator = _resolve_timesfm_frequency_indicator(
            self._frequency_minutes,
        )
        self._ensure_model()
        self._calibrator = self._fit_calibrator(
            samples=samples,
            source_rows=source_rows,
            progress_callback=progress_callback,
        )
        return self

    def predict(self, rows: list[dict[str, Any]]) -> list[int]:
        """Return binary labels from calibrated TimesFM probabilities."""
        probabilities = self.predict_proba(rows)
        return [1 if row[1] >= 0.5 else 0 for row in probabilities]

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        """Return calibrated probabilities from explicit sequence contexts."""
        raw_scores = self._predict_raw_scores_from_context_rows(rows)
        return self._require_calibrator().predict_proba(raw_scores)

    def predict_samples(
        self,
        test_samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> list[int]:
        """Return binary labels from calibrated TimesFM probabilities."""
        probabilities = self.predict_proba_samples(
            test_samples,
            source_rows=source_rows,
            progress_callback=progress_callback,
        )
        return [1 if row[1] >= 0.5 else 0 for row in probabilities]

    def predict_proba_samples(
        self,
        test_samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Return calibrated probabilities from ordered training samples."""
        if not test_samples:
            return []
        context_rows = build_sequence_context_rows(
            target_samples=test_samples,
            source_rows=source_rows,
            feature_columns=self._feature_columns,
            lookback_candles=self.context_lookback_candles,
        )
        raw_scores = self._predict_raw_scores_from_context_rows(
            context_rows,
            progress_callback=progress_callback,
        )
        return self._require_calibrator().predict_proba(raw_scores)

    def requires_sequence_context(self) -> bool:
        """TimesFM needs ordered lookback rows to score honestly."""
        return True

    def get_sequence_lookback_candles(self) -> int:
        """Return the saved runtime lookback contract."""
        return self.context_lookback_candles

    def get_feature_columns(self) -> list[str]:
        """Return the saved feature schema."""
        return list(self._feature_columns)

    def get_expanded_feature_names(self) -> list[str]:
        """Return the saved expanded feature schema."""
        return list(self._feature_columns)

    def get_training_config(self) -> dict[str, Any]:
        """Return audit-stable TimesFM wrapper metadata."""
        return {
            "model_family": self.model_family,
            "candidate_role": self.candidate_role,
            "scope_regimes": list(self.scope_regimes),
            "pretrained_source": self.pretrained_source,
            "license_name": DEFAULT_TIMESFM_2_0_LICENSE_NAME,
            "artifact_format": HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT,
            "calibration_method": DEFAULT_PRETRAINED_CALIBRATION_METHOD,
            "input_mapping": "UNIVARIATE_CLOSE_PRICE_ONLY",
            "horizon_candles": self.horizon_candles,
            "context_lookback_candles": self.context_lookback_candles,
            "max_context_len": self.max_context_len,
            "calibration_windows": self.calibration_windows,
            "prediction_batch_size": self.prediction_batch_size,
            "backend": self.backend,
            "resolved_backend": self._resolved_backend,
            "per_core_batch_size": self.per_core_batch_size,
            "input_patch_len": self.input_patch_len,
            "output_patch_len": self.output_patch_len,
            "num_layers": self.num_layers,
            "model_dims": self.model_dims,
            "use_positional_embedding": self.use_positional_embedding,
            "point_forecast_mode": self.point_forecast_mode,
            "normalize_inputs": self.normalize_inputs,
            "frequency_minutes": self._frequency_minutes,
            "frequency_indicator": self._frequency_indicator,
        }

    def get_registry_metadata(self) -> PretrainedForecasterArtifactMetadata:
        """Return truthful Packet 2 discovery metadata for TimesFM 2.0."""
        return PretrainedForecasterArtifactMetadata(
            model_family=self.model_family,
            candidate_role=self.candidate_role,
            scope_regimes=self.scope_regimes,
            pretrained_source=self.pretrained_source,
            artifact_format=HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT,
            calibration_method=DEFAULT_PRETRAINED_CALIBRATION_METHOD,
            license_name=DEFAULT_TIMESFM_2_0_LICENSE_NAME,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Persist the smallest honest compatible artifact form."""
        return {
            "model_family": self.model_family,
            "pretrained_source": self.pretrained_source,
            "candidate_role": self.candidate_role,
            "scope_regimes": list(self.scope_regimes),
            "horizon_candles": self.horizon_candles,
            "context_lookback_candles": self.context_lookback_candles,
            "max_context_len": self.max_context_len,
            "calibration_windows": self.calibration_windows,
            "prediction_batch_size": self.prediction_batch_size,
            "backend": self.backend,
            "resolved_backend": self._resolved_backend,
            "per_core_batch_size": self.per_core_batch_size,
            "input_patch_len": self.input_patch_len,
            "output_patch_len": self.output_patch_len,
            "num_layers": self.num_layers,
            "model_dims": self.model_dims,
            "use_positional_embedding": self.use_positional_embedding,
            "point_forecast_mode": self.point_forecast_mode,
            "normalize_inputs": self.normalize_inputs,
            "feature_columns": list(self._feature_columns),
            "frequency_minutes": self._frequency_minutes,
            "frequency_indicator": self._frequency_indicator,
            "calibrator": self._calibrator,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the lazy-load wrapper without embedding TimesFM weights."""
        self.model_family = str(state["model_family"])
        self.pretrained_source = str(state["pretrained_source"])
        self.candidate_role = str(state["candidate_role"])
        self.scope_regimes = tuple(str(regime) for regime in state["scope_regimes"])
        self.horizon_candles = int(state["horizon_candles"])
        self.context_lookback_candles = int(state["context_lookback_candles"])
        self.max_context_len = int(state["max_context_len"])
        self.calibration_windows = int(state["calibration_windows"])
        self.prediction_batch_size = int(state["prediction_batch_size"])
        self.backend = str(state["backend"])
        self._resolved_backend = (
            None if state.get("resolved_backend") is None else str(state["resolved_backend"])
        )
        self.per_core_batch_size = int(state["per_core_batch_size"])
        self.input_patch_len = int(state["input_patch_len"])
        self.output_patch_len = int(state["output_patch_len"])
        self.num_layers = int(state["num_layers"])
        self.model_dims = int(state["model_dims"])
        self.use_positional_embedding = bool(state["use_positional_embedding"])
        self.point_forecast_mode = str(state["point_forecast_mode"])
        self.normalize_inputs = bool(state["normalize_inputs"])
        self._feature_columns = tuple(str(column) for column in state["feature_columns"])
        self._frequency_minutes = (
            None
            if state.get("frequency_minutes") is None
            else int(state["frequency_minutes"])
        )
        self._frequency_indicator = (
            None
            if state.get("frequency_indicator") is None
            else int(state["frequency_indicator"])
        )
        self._calibrator = state["calibrator"]
        self._model = None

    def _fit_calibrator(
        self,
        *,
        samples: list[DatasetSample],
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> ForecastToProbabilityCalibrator:
        calibration_samples = self._select_calibration_samples(
            samples=samples,
            source_rows=source_rows,
        )
        labels = [int(sample.label) for sample in calibration_samples]
        if not calibration_samples:
            fallback_labels = [int(sample.label) for sample in samples]
            return ForecastToProbabilityCalibrator().fit(
                [0.0 for _ in fallback_labels],
                fallback_labels,
            )

        context_rows = build_sequence_context_rows(
            target_samples=calibration_samples,
            source_rows=source_rows,
            feature_columns=self._feature_columns,
            lookback_candles=self.context_lookback_candles,
        )
        raw_scores = self._predict_raw_scores_from_context_rows(
            context_rows,
            progress_callback=progress_callback,
        )
        return ForecastToProbabilityCalibrator().fit(raw_scores, labels)

    def _select_calibration_samples(
        self,
        *,
        samples: list[DatasetSample],
        source_rows: list[SourceFeatureRow],
    ) -> list[DatasetSample]:
        if self.calibration_windows <= 0:
            return []
        eligible_samples: list[DatasetSample] = []
        for sample in samples:
            history_count = sum(
                1
                for source_row in source_rows
                if source_row.symbol == sample.symbol
                and source_row.as_of_time <= sample.as_of_time
            )
            if history_count >= self.context_lookback_candles:
                eligible_samples.append(sample)
        if not eligible_samples:
            return []
        return eligible_samples[-min(len(eligible_samples), self.calibration_windows) :]

    def _predict_raw_scores_from_context_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[float]:
        if not rows:
            return []
        timesfm_model = self._ensure_model()
        batch_size = max(1, self.prediction_batch_size)
        total_rows = len(rows)
        total_batches = (total_rows + batch_size - 1) // batch_size
        raw_scores: list[float] = []
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sequence_scoring_start",
                    "row_count": total_rows,
                    "batch_count": total_batches,
                    "batch_size": batch_size,
                }
            )

        for batch_index, batch_start in enumerate(range(0, total_rows, batch_size), start=1):
            batch_rows = rows[batch_start : batch_start + batch_size]
            batch_contexts, batch_freq, current_closes = self._build_forecast_inputs(batch_rows)
            mean_forecasts, _ = timesfm_model.forecast(
                inputs=batch_contexts,
                freq=batch_freq,
                forecast_context_len=min(self.context_lookback_candles, self.max_context_len),
                normalize=self.normalize_inputs,
            )
            for forecast_row, current_close in zip(
                np.asarray(mean_forecasts),
                current_closes,
                strict=True,
            ):
                forecast_value = float(forecast_row[self.horizon_candles - 1])
                raw_scores.append((forecast_value / current_close) - 1.0)
            if progress_callback is not None:
                completed_rows = min(total_rows, batch_start + len(batch_rows))
                progress_callback(
                    {
                        "event": "sequence_scoring_progress",
                        "row_count": total_rows,
                        "completed_rows": completed_rows,
                        "batch_count": total_batches,
                        "completed_batches": batch_index,
                        "progress": batch_index / float(total_batches),
                    }
                )

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sequence_scoring_complete",
                    "row_count": total_rows,
                    "batch_count": total_batches,
                }
            )
        return raw_scores

    def _build_forecast_inputs(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[list[np.ndarray], list[int], list[float]]:
        frequency_indicator = self._require_frequency_indicator()
        contexts: list[np.ndarray] = []
        frequency_inputs: list[int] = []
        current_closes: list[float] = []
        for row in rows:
            context_rows = row.get(SEQUENCE_CONTEXT_KEY)
            if not isinstance(context_rows, list) or not context_rows:
                raise ValueError(
                    "TimesFM scoring rows must include non-empty "
                    f"{SEQUENCE_CONTEXT_KEY} lookback data",
                )
            context_values = np.asarray(
                [float(context_row["close_price"]) for context_row in context_rows],
                dtype=float,
            )
            contexts.append(context_values[-min(len(context_values), self.max_context_len) :])
            current_closes.append(float(context_values[-1]))
            frequency_inputs.append(frequency_indicator)
        return contexts, frequency_inputs, current_closes

    def _ensure_model(self) -> Any:
        if self._model is None:
            timesfm_module = _import_timesfm_runtime()
            resolved_backend = _resolve_timesfm_backend(self.backend)
            hparams = timesfm_module.TimesFmHparams(
                backend=resolved_backend,
                per_core_batch_size=self.per_core_batch_size,
                horizon_len=self.horizon_candles,
                context_len=self.max_context_len,
                input_patch_len=self.input_patch_len,
                output_patch_len=self.output_patch_len,
                num_layers=self.num_layers,
                model_dims=self.model_dims,
                use_positional_embedding=self.use_positional_embedding,
                point_forecast_mode=self.point_forecast_mode,
            )
            checkpoint = timesfm_module.TimesFmCheckpoint(
                huggingface_repo_id=self.pretrained_source,
            )
            self._model = timesfm_module.TimesFm(
                hparams=hparams,
                checkpoint=checkpoint,
            )
            self._resolved_backend = resolved_backend
        return self._model

    def _require_calibrator(self) -> ForecastToProbabilityCalibrator:
        if self._calibrator is None:
            raise ValueError("TimesFM artifact has not been fit/calibrated yet")
        return self._calibrator

    def _require_frequency_indicator(self) -> int:
        if self._frequency_indicator is None:
            raise ValueError("TimesFM artifact is missing frequency indicator metadata")
        return self._frequency_indicator


class Moirai1RBaseForecaster:  # pylint: disable=too-many-instance-attributes
    """Apache-2.0 sktime snapshot wrapper for Moirai-1.0-R-base."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        pretrained_source: str = DEFAULT_MOIRAI_1_0_R_BASE_PRETRAINED_SOURCE,
        candidate_role: str = RANGE_SPECIALIST,
        scope_regimes: tuple[str, ...] = RANGE_SCOPE_REGIMES,
        horizon_candles: int = 3,
        context_lookback_candles: int = _DEFAULT_MOIRAI_LOOKBACK_CANDLES,
        patch_size: int = _DEFAULT_MOIRAI_PATCH_SIZE,
        num_samples: int = _DEFAULT_MOIRAI_NUM_SAMPLES,
        batch_size: int = _DEFAULT_MOIRAI_BATCH_SIZE,
        deterministic: bool = True,
        map_location: str = "auto",
        use_source_package: bool = False,
        calibration_windows: int = _DEFAULT_TIMESFM_CALIBRATION_WINDOWS,
    ) -> None:
        self.model_family = MODEL_FAMILY_MOIRAI_BASE
        self.pretrained_source = str(pretrained_source)
        self.candidate_role = str(candidate_role)
        self.scope_regimes = tuple(str(regime) for regime in scope_regimes)
        self.horizon_candles = int(horizon_candles)
        self.context_lookback_candles = int(context_lookback_candles)
        self.patch_size = int(patch_size)
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.deterministic = bool(deterministic)
        self.map_location = str(map_location)
        self.use_source_package = bool(use_source_package)
        self.calibration_windows = int(calibration_windows)
        self._feature_columns: tuple[str, ...] = ("symbol", "close_price")
        self._frequency_minutes: int | None = None
        self._resolved_map_location: str | None = None
        self._bootstrap_panel_records: list[dict[str, Any]] = []
        self._calibrator: ForecastToProbabilityCalibrator | None = None
        self._forecaster: Any | None = None

    def fit(
        self,
        rows: list[dict[str, Any]],
        labels: list[int],
    ) -> "Moirai1RBaseForecaster":
        """Reject flat-row fitting so Moirai stays sequence-honest."""
        del rows, labels
        raise ValueError("Moirai requires fit_samples(...) with ordered source history")

    def fit_samples(
        self,
        samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        dataset_export_root: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> "Moirai1RBaseForecaster":
        """Load the pretrained Moirai checkpoint and fit the P(UP) bridge."""
        del dataset_export_root
        if not samples:
            raise ValueError("Moirai cannot fit without labeled samples")
        if not source_rows:
            raise ValueError("Moirai cannot fit without ordered source feature rows")

        self._frequency_minutes = infer_source_frequency_minutes(source_rows)
        self._bootstrap_panel_records = self._build_bootstrap_panel_records(source_rows)
        self._ensure_forecaster()
        self._calibrator = self._fit_calibrator(
            samples=samples,
            source_rows=source_rows,
            progress_callback=progress_callback,
        )
        return self

    def predict(self, rows: list[dict[str, Any]]) -> list[int]:
        """Return binary labels from calibrated Moirai probabilities."""
        probabilities = self.predict_proba(rows)
        return [1 if row[1] >= 0.5 else 0 for row in probabilities]

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        """Return calibrated probabilities from explicit sequence contexts."""
        raw_scores = self._predict_raw_scores_from_context_rows(rows)
        return self._require_calibrator().predict_proba(raw_scores)

    def predict_samples(
        self,
        test_samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> list[int]:
        """Return binary labels from calibrated Moirai probabilities."""
        probabilities = self.predict_proba_samples(
            test_samples,
            source_rows=source_rows,
            progress_callback=progress_callback,
        )
        return [1 if row[1] >= 0.5 else 0 for row in probabilities]

    def predict_proba_samples(
        self,
        test_samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Return calibrated probabilities from ordered training samples."""
        if not test_samples:
            return []
        context_rows = build_sequence_context_rows(
            target_samples=test_samples,
            source_rows=source_rows,
            feature_columns=self._feature_columns,
            lookback_candles=self.context_lookback_candles,
        )
        raw_scores = self._predict_raw_scores_from_context_rows(
            context_rows,
            progress_callback=progress_callback,
        )
        return self._require_calibrator().predict_proba(raw_scores)

    def requires_sequence_context(self) -> bool:
        """Moirai needs ordered lookback rows to score honestly."""
        return True

    def get_sequence_lookback_candles(self) -> int:
        """Return the saved runtime lookback contract."""
        return self.context_lookback_candles

    def get_feature_columns(self) -> list[str]:
        """Return the saved feature schema."""
        return list(self._feature_columns)

    def get_expanded_feature_names(self) -> list[str]:
        """Return the saved expanded feature schema."""
        return list(self._feature_columns)

    def get_training_config(self) -> dict[str, Any]:
        """Return audit-stable Moirai wrapper metadata."""
        return {
            "model_family": self.model_family,
            "candidate_role": self.candidate_role,
            "scope_regimes": list(self.scope_regimes),
            "pretrained_source": self.pretrained_source,
            "license_name": DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NAME,
            "license_notes": DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NOTES,
            "artifact_format": HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT,
            "calibration_method": DEFAULT_PRETRAINED_CALIBRATION_METHOD,
            "input_mapping": "UNIVARIATE_CLOSE_PRICE_ONLY",
            "checkpoint_provider": "sktime_snapshot",
            "horizon_candles": self.horizon_candles,
            "context_lookback_candles": self.context_lookback_candles,
            "patch_size": self.patch_size,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "deterministic": self.deterministic,
            "map_location": self.map_location,
            "resolved_map_location": self._resolved_map_location,
            "use_source_package": self.use_source_package,
            "calibration_windows": self.calibration_windows,
            "frequency_minutes": self._frequency_minutes,
        }

    def get_registry_metadata(self) -> PretrainedForecasterArtifactMetadata:
        """Return truthful Packet 2 discovery metadata for Moirai-1.0-R-base."""
        return PretrainedForecasterArtifactMetadata(
            model_family=self.model_family,
            candidate_role=self.candidate_role,
            scope_regimes=self.scope_regimes,
            pretrained_source=self.pretrained_source,
            artifact_format=HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT,
            calibration_method=DEFAULT_PRETRAINED_CALIBRATION_METHOD,
            license_name=DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NAME,
            license_notes=DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NOTES,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Persist the smallest honest compatible artifact form."""
        return {
            "model_family": self.model_family,
            "pretrained_source": self.pretrained_source,
            "candidate_role": self.candidate_role,
            "scope_regimes": list(self.scope_regimes),
            "horizon_candles": self.horizon_candles,
            "context_lookback_candles": self.context_lookback_candles,
            "patch_size": self.patch_size,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "deterministic": self.deterministic,
            "map_location": self.map_location,
            "resolved_map_location": self._resolved_map_location,
            "use_source_package": self.use_source_package,
            "calibration_windows": self.calibration_windows,
            "feature_columns": list(self._feature_columns),
            "frequency_minutes": self._frequency_minutes,
            "bootstrap_panel_records": list(self._bootstrap_panel_records),
            "calibrator": self._calibrator,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the lazy-load wrapper without embedding Moirai weights."""
        self.model_family = str(state["model_family"])
        self.pretrained_source = str(state["pretrained_source"])
        self.candidate_role = str(state["candidate_role"])
        self.scope_regimes = tuple(str(regime) for regime in state["scope_regimes"])
        self.horizon_candles = int(state["horizon_candles"])
        self.context_lookback_candles = int(state["context_lookback_candles"])
        self.patch_size = int(state["patch_size"])
        self.num_samples = int(state["num_samples"])
        self.batch_size = int(state["batch_size"])
        self.deterministic = bool(state["deterministic"])
        self.map_location = str(state["map_location"])
        self._resolved_map_location = (
            None
            if state.get("resolved_map_location") is None
            else str(state["resolved_map_location"])
        )
        self.use_source_package = bool(state["use_source_package"])
        self.calibration_windows = int(state["calibration_windows"])
        self._feature_columns = tuple(str(column) for column in state["feature_columns"])
        self._frequency_minutes = (
            None
            if state.get("frequency_minutes") is None
            else int(state["frequency_minutes"])
        )
        self._bootstrap_panel_records = [
            dict(record) for record in state.get("bootstrap_panel_records", [])
        ]
        self._calibrator = state["calibrator"]
        self._forecaster = None

    def _fit_calibrator(
        self,
        *,
        samples: list[DatasetSample],
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> ForecastToProbabilityCalibrator:
        calibration_samples = self._select_calibration_samples(
            samples=samples,
            source_rows=source_rows,
        )
        labels = [int(sample.label) for sample in calibration_samples]
        if not calibration_samples:
            fallback_labels = [int(sample.label) for sample in samples]
            return ForecastToProbabilityCalibrator().fit(
                [0.0 for _ in fallback_labels],
                fallback_labels,
            )

        context_rows = build_sequence_context_rows(
            target_samples=calibration_samples,
            source_rows=source_rows,
            feature_columns=self._feature_columns,
            lookback_candles=self.context_lookback_candles,
        )
        raw_scores = self._predict_raw_scores_from_context_rows(
            context_rows,
            progress_callback=progress_callback,
        )
        return ForecastToProbabilityCalibrator().fit(raw_scores, labels)

    def _select_calibration_samples(
        self,
        *,
        samples: list[DatasetSample],
        source_rows: list[SourceFeatureRow],
    ) -> list[DatasetSample]:
        if self.calibration_windows <= 0:
            return []
        eligible_samples: list[DatasetSample] = []
        for sample in samples:
            history_count = sum(
                1
                for source_row in source_rows
                if source_row.symbol == sample.symbol
                and source_row.as_of_time <= sample.as_of_time
            )
            if history_count >= self.context_lookback_candles:
                eligible_samples.append(sample)
        if not eligible_samples:
            return []
        return eligible_samples[-min(len(eligible_samples), self.calibration_windows) :]

    def _predict_raw_scores_from_context_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[float]:
        if not rows:
            return []
        forecaster = self._ensure_forecaster()
        pandas = _import_pandas()
        forecasting_horizon = list(range(1, self.horizon_candles + 1))
        total_rows = len(rows)
        total_batches = total_rows
        raw_scores: list[float] = []
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sequence_scoring_start",
                    "row_count": total_rows,
                    "batch_count": total_batches,
                    "batch_size": 1,
                }
            )

        for row_index, row in enumerate(rows, start=1):
            context_panel = self._build_context_panel_frame(row)
            prediction = forecaster.predict(
                fh=forecasting_horizon,
                y=context_panel,
            )
            forecast_value = self._extract_terminal_forecast_value(
                prediction,
                pandas_module=pandas,
            )
            current_close = float(context_panel.iloc[-1, 0])
            raw_scores.append((forecast_value / current_close) - 1.0)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "sequence_scoring_progress",
                        "row_count": total_rows,
                        "completed_rows": row_index,
                        "batch_count": total_batches,
                        "completed_batches": row_index,
                        "progress": row_index / float(total_batches),
                    }
                )

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sequence_scoring_complete",
                    "row_count": total_rows,
                    "batch_count": total_batches,
                }
            )
        return raw_scores

    def _build_bootstrap_panel_records(
        self,
        source_rows: list[SourceFeatureRow],
    ) -> list[dict[str, Any]]:
        by_symbol: dict[str, list[SourceFeatureRow]] = {}
        for source_row in source_rows:
            by_symbol.setdefault(source_row.symbol, []).append(source_row)
        for symbol, symbol_rows in sorted(by_symbol.items()):
            ordered_rows = sorted(symbol_rows, key=lambda row: row.as_of_time)
            if len(ordered_rows) >= self.context_lookback_candles:
                selected_rows = ordered_rows[-self.context_lookback_candles :]
                return [
                    {
                        "symbol": symbol,
                        "timestamp": selected_row.as_of_time.isoformat(),
                        "close_price": float(selected_row.close_price),
                    }
                    for selected_row in selected_rows
                ]
        raise ValueError(
            "Moirai requires at least one symbol with enough lookback rows to bootstrap the runtime",
        )

    def _build_context_panel_frame(self, row: dict[str, Any]) -> Any:
        pandas = _import_pandas()
        context_rows = row.get(SEQUENCE_CONTEXT_KEY)
        if not isinstance(context_rows, list) or not context_rows:
            raise ValueError(
                "Moirai scoring rows must include non-empty "
                f"{SEQUENCE_CONTEXT_KEY} lookback data",
            )
        context_records = [
            {
                "instance": str(context_row["symbol"]),
                "timestamp": context_row["as_of_time"],
                "close_price": float(context_row["close_price"]),
            }
            for context_row in context_rows
        ]
        panel = pandas.DataFrame.from_records(context_records)
        panel["timestamp"] = pandas.to_datetime(panel["timestamp"])
        panel = panel.set_index(["instance", "timestamp"]).sort_index()
        return panel[["close_price"]]

    def _bootstrap_panel_frame(self) -> Any:
        pandas = _import_pandas()
        if not self._bootstrap_panel_records:
            raise ValueError("Moirai artifact is missing bootstrap panel records")
        frame = pandas.DataFrame.from_records(self._bootstrap_panel_records)
        frame["timestamp"] = pandas.to_datetime(frame["timestamp"])
        frame = frame.set_index(["symbol", "timestamp"]).sort_index()
        frame.index = frame.index.set_names(["instance", "timestamp"])
        return frame[["close_price"]]

    def _extract_terminal_forecast_value(
        self,
        prediction: Any,
        *,
        pandas_module: Any,
    ) -> float:
        if isinstance(prediction, pandas_module.Series):
            return float(prediction.iloc[-1])
        if isinstance(prediction, pandas_module.DataFrame):
            if prediction.empty:
                raise ValueError("Moirai forecast output is empty")
            return float(prediction.iloc[-1, 0])
        frame = pandas_module.DataFrame(prediction)
        if frame.empty:
            raise ValueError("Moirai forecast output is empty")
        return float(frame.iloc[-1, 0])

    def _ensure_forecaster(self) -> Any:
        if self._forecaster is None:
            forecaster_class = _import_sktime_moirai_runtime()
            resolved_map_location = _resolve_torch_map_location(self.map_location)
            self._resolved_map_location = resolved_map_location
            self._forecaster = forecaster_class(
                checkpoint_path=self.pretrained_source,
                context_length=self.context_lookback_candles,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                map_location=resolved_map_location,
                target_dim=1,
                deterministic=self.deterministic,
                batch_size=self.batch_size,
                use_source_package=self.use_source_package,
            )
            bootstrap_panel = self._bootstrap_panel_frame()
            self._forecaster.fit(
                bootstrap_panel,
                fh=list(range(1, self.horizon_candles + 1)),
            )
        return self._forecaster

    def _require_calibrator(self) -> ForecastToProbabilityCalibrator:
        if self._calibrator is None:
            raise ValueError("Moirai artifact has not been fit/calibrated yet")
        return self._calibrator


def build_chronos2_classifier(model_config: dict[str, Any]) -> Chronos2Forecaster:
    """Build the real Chronos-2 generalist challenger wrapper."""
    raw_torch_dtype = model_config.get("torch_dtype", "auto")
    return Chronos2Forecaster(
        pretrained_source=str(
            model_config.get("pretrained_source", DEFAULT_CHRONOS_2_PRETRAINED_SOURCE)
        ),
        candidate_role=str(model_config.get("candidate_role", GENERALIST)),
        scope_regimes=_resolve_scope_regimes(
            model_config,
            default_scope_regimes=DEFAULT_SCOPE_REGIMES,
        ),
        horizon_candles=int(model_config.get("horizon_candles", 3)),
        context_lookback_candles=int(
            model_config.get(
                "context_lookback_candles",
                _DEFAULT_CHRONOS_2_LOOKBACK_CANDLES,
            )
        ),
        calibration_windows=int(
            model_config.get("calibration_windows", _DEFAULT_CHRONOS_2_CALIBRATION_WINDOWS)
        ),
        prediction_batch_size=int(
            model_config.get("prediction_batch_size", _DEFAULT_CHRONOS_2_BATCH_SIZE)
        ),
        quantile_levels=tuple(
            float(level)
            for level in model_config.get(
                "quantile_levels",
                list(_DEFAULT_CHRONOS_2_QUANTILE_LEVELS),
            )
        ),
        covariate_columns=_resolve_optional_columns(model_config.get("covariate_columns")),
        device_map=str(model_config.get("device_map", "auto")),
        torch_dtype=None if raw_torch_dtype is None else str(raw_torch_dtype),
        cache_dir=(
            None
            if model_config.get("cache_dir") is None
            else str(model_config.get("cache_dir"))
        ),
    )


def build_timesfm_2_0_500m_pytorch_classifier(
    model_config: dict[str, Any],
) -> TimesFm2Forecaster:
    """Build the real TimesFM 2.0 500M PyTorch trend-specialist wrapper."""
    return TimesFm2Forecaster(
        pretrained_source=str(
            model_config.get(
                "pretrained_source",
                DEFAULT_TIMESFM_2_0_500M_PYTORCH_PRETRAINED_SOURCE,
            )
        ),
        candidate_role=str(model_config.get("candidate_role", TREND_SPECIALIST)),
        scope_regimes=_resolve_scope_regimes(
            model_config,
            default_scope_regimes=TREND_SCOPE_REGIMES,
        ),
        horizon_candles=int(model_config.get("horizon_candles", 3)),
        context_lookback_candles=int(
            model_config.get(
                "context_lookback_candles",
                _DEFAULT_TIMESFM_LOOKBACK_CANDLES,
            )
        ),
        max_context_len=int(
            model_config.get("max_context_len", _DEFAULT_TIMESFM_MAX_CONTEXT_LEN)
        ),
        calibration_windows=int(
            model_config.get("calibration_windows", _DEFAULT_TIMESFM_CALIBRATION_WINDOWS)
        ),
        prediction_batch_size=int(
            model_config.get("prediction_batch_size", _DEFAULT_TIMESFM_BATCH_SIZE)
        ),
        backend=str(model_config.get("backend", "auto")),
        per_core_batch_size=int(
            model_config.get("per_core_batch_size", _DEFAULT_TIMESFM_BATCH_SIZE)
        ),
        input_patch_len=int(
            model_config.get("input_patch_len", _DEFAULT_TIMESFM_INPUT_PATCH_LEN)
        ),
        output_patch_len=int(
            model_config.get("output_patch_len", _DEFAULT_TIMESFM_OUTPUT_PATCH_LEN)
        ),
        num_layers=int(model_config.get("num_layers", _DEFAULT_TIMESFM_NUM_LAYERS)),
        model_dims=int(model_config.get("model_dims", _DEFAULT_TIMESFM_MODEL_DIMS)),
        use_positional_embedding=bool(
            model_config.get("use_positional_embedding", False)
        ),
        point_forecast_mode=str(
            model_config.get("point_forecast_mode", _DEFAULT_TIMESFM_POINT_FORECAST_MODE)
        ),
        normalize_inputs=bool(model_config.get("normalize_inputs", True)),
    )


def build_moirai_1_0_r_base_range_classifier(
    model_config: dict[str, Any],
) -> Moirai1RBaseForecaster:
    """Build the Apache-2.0 sktime snapshot Moirai range-specialist wrapper."""
    return Moirai1RBaseForecaster(
        pretrained_source=str(
            model_config.get(
                "pretrained_source",
                DEFAULT_MOIRAI_1_0_R_BASE_PRETRAINED_SOURCE,
            )
        ),
        candidate_role=str(model_config.get("candidate_role", RANGE_SPECIALIST)),
        scope_regimes=_resolve_scope_regimes(
            model_config,
            default_scope_regimes=RANGE_SCOPE_REGIMES,
        ),
        horizon_candles=int(model_config.get("horizon_candles", 3)),
        context_lookback_candles=int(
            model_config.get(
                "context_lookback_candles",
                _DEFAULT_MOIRAI_LOOKBACK_CANDLES,
            )
        ),
        patch_size=int(model_config.get("patch_size", _DEFAULT_MOIRAI_PATCH_SIZE)),
        num_samples=int(model_config.get("num_samples", _DEFAULT_MOIRAI_NUM_SAMPLES)),
        batch_size=int(model_config.get("batch_size", _DEFAULT_MOIRAI_BATCH_SIZE)),
        deterministic=bool(model_config.get("deterministic", True)),
        map_location=str(model_config.get("map_location", "auto")),
        use_source_package=bool(model_config.get("use_source_package", False)),
        calibration_windows=int(
            model_config.get("calibration_windows", _DEFAULT_TIMESFM_CALIBRATION_WINDOWS)
        ),
    )


def _resolve_feature_columns(
    source_rows: list[SourceFeatureRow] | tuple[SourceFeatureRow, ...],
) -> tuple[str, ...]:
    if not source_rows:
        raise ValueError("Pretrained forecaster wrappers require source feature rows")
    first_row = source_rows[0]
    ordered_columns = tuple(str(column) for column in first_row.features)
    if not ordered_columns:
        raise ValueError("Pretrained forecaster wrappers require configured feature columns")
    return ordered_columns


def _resolve_covariate_columns(
    source_rows: list[SourceFeatureRow] | tuple[SourceFeatureRow, ...],
    *,
    feature_columns: tuple[str, ...],
    configured_covariate_columns: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if configured_covariate_columns is not None:
        return tuple(
            str(column)
            for column in configured_covariate_columns
            if str(column) not in {"symbol", "close_price"}
        )
    if not source_rows:
        return ()
    first_row = source_rows[0]
    covariate_columns: list[str] = []
    for column in feature_columns:
        if column in {"symbol", "close_price"}:
            continue
        value = first_row.features.get(column)
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value,
            bool,
        ):
            covariate_columns.append(str(column))
    return tuple(covariate_columns)


def _resolve_optional_columns(columns: Any) -> tuple[str, ...] | None:
    if columns is None:
        return None
    return tuple(str(column) for column in columns)


def _resolve_scope_regimes(
    model_config: dict[str, Any],
    *,
    default_scope_regimes: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        str(regime)
        for regime in model_config.get("scope_regimes", list(default_scope_regimes))
    )


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - exercised through unit doubles
        raise ValueError(
            "Chronos-2 challenger support requires pandas to be installed.",
        ) from error
    return pd


def _import_chronos_runtime() -> Any:
    try:
        from chronos import Chronos2Pipeline
    except ImportError as error:  # pragma: no cover - exercised through unit doubles
        raise ValueError(
            "Chronos-2 challenger support requires the optional "
            "chronos-forecasting dependency to be installed.",
        ) from error
    return Chronos2Pipeline


def _import_timesfm_runtime() -> Any:
    try:
        import timesfm
    except ImportError as error:  # pragma: no cover - exercised through unit doubles
        raise ValueError(
            "TimesFM challenger support requires the optional "
            "timesfm dependency to be installed.",
        ) from error
    return timesfm


def _import_sktime_moirai_runtime() -> Any:
    try:
        from sktime.forecasting.moirai_forecaster import MOIRAIForecaster
    except ImportError as error:  # pragma: no cover - exercised through unit doubles
        raise ValueError(
            "Moirai challenger support requires the optional sktime "
            "dependency to be installed.",
        ) from error
    return MOIRAIForecaster


def _resolve_torch_dtype(torch_dtype: str | None) -> Any | None:
    if torch_dtype is None:
        return None
    normalized = str(torch_dtype).strip().lower()
    if not normalized or normalized == "auto":
        return "auto"
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised through unit doubles
        raise ValueError(
            "Chronos-2 torch_dtype overrides require torch to be installed.",
        ) from error
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported Chronos-2 torch_dtype override: {torch_dtype!r}")
    return aliases[normalized]


def _resolve_timesfm_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized == "auto":
        return "gpu" if _torch_cuda_is_available() else "cpu"
    if normalized not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported TimesFM backend override: {backend!r}")
    if normalized == "gpu" and not _torch_cuda_is_available():
        return "cpu"
    return normalized


def _resolve_timesfm_frequency_indicator(frequency_minutes: int) -> int:
    if frequency_minutes <= 24 * 60:
        return 0
    if frequency_minutes <= 31 * 24 * 60:
        return 1
    return 2


def _resolve_torch_map_location(map_location: str) -> str:
    normalized = str(map_location).strip().lower()
    if normalized == "auto":
        return "cuda" if _torch_cuda_is_available() else "cpu"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported torch map_location override: {map_location!r}")
    if normalized == "cuda" and not _torch_cuda_is_available():
        return "cpu"
    return normalized


def _torch_cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())
