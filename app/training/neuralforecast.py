"""NeuralForecast specialist wrappers for Stream Alpha M20 challenger training."""

from __future__ import annotations

import json
import copy
import io
import shutil
import zipfile
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import quote

import pandas as pd

from app.training.dataset import (
    SEQUENCE_CONTEXT_KEY,
    DatasetSample,
    SourceFeatureRow,
    build_sequence_context_rows,
    future_target_timestamp,
    infer_source_frequency_minutes,
)
from app.training.pretrained_forecasters import (
    ForecastToProbabilityCalibrator,
    MODEL_FAMILY_NEURALFORECAST_NHITS,
    MODEL_FAMILY_NEURALFORECAST_PATCHTST,
    ProgressCallback,
    RANGE_SPECIALIST,
    TREND_SPECIALIST,
)
from app.training.workdirs import create_local_training_work_dir


_DEFAULT_CALIBRATION_WINDOWS = 256
_PREDICT_CONTEXT_BATCH_SIZE = 256


class _BaseNeuralForecastClassifier:  # pylint: disable=too-many-instance-attributes
    """Shared sequence-model wrapper that keeps the saved artifact self-contained."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        model_family: str,
        model_class_name: str,
        model_alias: str,
        candidate_role: str,
        scope_regimes: tuple[str, ...],
        horizon_candles: int,
        input_size_candles: int,
        hist_exog_columns: tuple[str, ...] | None,
        calibration_windows: int,
        dataset_mode: str,
        max_steps: int,
        learning_rate: float,
        batch_size: int,
        scaler_type: str,
        early_stop_patience_steps: int,
        val_check_steps: int,
        random_seed: int,
        enable_progress_bar: bool,
        logger: bool,
        refit_each_window: bool,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_family = str(model_family)
        self.model_class_name = str(model_class_name)
        self.model_alias = str(model_alias)
        self.candidate_role = str(candidate_role)
        self.scope_regimes = tuple(str(regime) for regime in scope_regimes)
        self.horizon_candles = int(horizon_candles)
        self.input_size_candles = int(input_size_candles)
        self.hist_exog_columns = (
            None
            if hist_exog_columns is None
            else tuple(str(column) for column in hist_exog_columns)
        )
        self.calibration_windows = int(calibration_windows)
        self.dataset_mode = str(dataset_mode)
        self.max_steps = int(max_steps)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.scaler_type = str(scaler_type)
        self.early_stop_patience_steps = int(early_stop_patience_steps)
        self.val_check_steps = int(val_check_steps)
        self.random_seed = int(random_seed)
        self.enable_progress_bar = bool(enable_progress_bar)
        self.logger = bool(logger)
        self.refit_each_window = bool(refit_each_window)
        self.model_kwargs = (
            {}
            if model_kwargs is None
            else copy.deepcopy(dict(model_kwargs))
        )
        self._feature_columns: tuple[str, ...] = ()
        self._hist_exog_columns: tuple[str, ...] = ()
        self._series_id_by_symbol: dict[str, str] = {}
        self._frequency_minutes: int | None = None
        self._backend_archive: bytes | None = None
        self._runtime_dir: Path | None = None
        self._backend: Any | None = None
        self._calibrator: ForecastToProbabilityCalibrator | None = None

    def fit(
        self,
        rows: list[dict[str, Any]],
        labels: list[int],
    ) -> "_BaseNeuralForecastClassifier":
        """Reject flat-row training so these models stay sequence-honest."""
        del rows, labels
        raise ValueError(
            f"{self.model_class_name} requires fit_samples(...) with ordered source history",
        )

    def fit_samples(
        self,
        samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        dataset_export_root: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> "_BaseNeuralForecastClassifier":
        """Train the final sequence model plus an explicit forecast-to-probability bridge."""
        if not samples:
            raise ValueError(f"{self.model_class_name} cannot fit without labeled samples")
        if not source_rows:
            raise ValueError(
                f"{self.model_class_name} cannot fit without ordered source feature rows",
            )

        self._feature_columns = _resolve_feature_columns(source_rows)
        self._hist_exog_columns = _resolve_hist_exog_columns(
            source_rows,
            configured_columns=self.hist_exog_columns,
        )
        self._series_id_by_symbol = _build_series_id_map(source_rows)
        self._frequency_minutes = infer_source_frequency_minutes(source_rows)

        fit_root = create_local_training_work_dir(
            prefix=f"streamalpha-{self.model_alias.lower()}-fit-",
        )
        save_dir = fit_root / "backend"
        try:
            backend = self._fit_backend(
                source_rows=source_rows,
                dataset_export_root=(
                    None
                    if dataset_export_root is None
                    else Path(dataset_export_root) / "full_history_fit"
                ),
            )
            self._backend = backend
            self._calibrator = self._fit_calibrator(
                samples=samples,
                source_rows=source_rows,
                fitted_backend=backend,
            )
            self._save_backend_archive(
                backend=backend,
                save_dir=save_dir,
                progress_callback=progress_callback,
            )
            self._backend_archive = _archive_backend_dir(save_dir)
        finally:
            shutil.rmtree(fit_root, ignore_errors=True)
        self._cleanup_runtime_dir()
        return self

    def predict(self, rows: list[dict[str, Any]]) -> list[int]:
        """Return binary class predictions from calibrated sequence probabilities."""
        probabilities = self.predict_proba(rows)
        return [1 if row[1] >= 0.5 else 0 for row in probabilities]

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        """Return calibrated probabilities from explicit sequence contexts."""
        raw_scores = self._predict_raw_scores_from_context_rows(rows)
        calibrator = self._require_calibrator()
        return calibrator.predict_proba(raw_scores)

    def predict_samples(
        self,
        test_samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> list[int]:
        """Return fold-evaluation labels from out-of-sample rolling sequence forecasts."""
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
        """Return fold-evaluation probabilities via rolling out-of-sample forecasts."""
        if not test_samples:
            return []
        if not source_rows:
            raise ValueError(
                f"{self.model_class_name} requires source_rows for sequence evaluation",
            )
        raw_scores = self._cross_validated_raw_scores(
            target_samples=test_samples,
            source_rows=source_rows,
            progress_callback=progress_callback,
        )
        calibrator = self._require_calibrator()
        return calibrator.predict_proba(raw_scores)

    def requires_sequence_context(self) -> bool:
        """Return whether runtime scoring must provide ordered lookback rows."""
        return True

    def get_sequence_lookback_candles(self) -> int:
        """Return the exact lookback length required for honest runtime scoring."""
        return self.input_size_candles

    def get_expanded_feature_names(self) -> list[str]:
        """Return the stable feature schema expected by future runtime lookback scoring."""
        return list(self._feature_columns)

    def get_feature_columns(self) -> list[str]:
        """Return the stable base feature schema required by the saved artifact."""
        return list(self._feature_columns)

    def get_training_config(self) -> dict[str, Any]:
        """Return the effective saved training config for auditability."""
        return {
            "model_family": self.model_family,
            "model_class_name": self.model_class_name,
            "model_alias": self.model_alias,
            "candidate_role": self.candidate_role,
            "scope_regimes": list(self.scope_regimes),
            "horizon_candles": self.horizon_candles,
            "input_size_candles": self.input_size_candles,
            "hist_exog_columns": list(self._hist_exog_columns),
            "calibration_windows": self.calibration_windows,
            "dataset_mode": self.dataset_mode,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "scaler_type": self.scaler_type,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "random_seed": self.random_seed,
            "enable_progress_bar": self.enable_progress_bar,
            "logger": self.logger,
            "refit_each_window": self.refit_each_window,
            "model_kwargs": copy.deepcopy(self.model_kwargs),
        }

    def get_registry_metadata(self) -> dict[str, Any]:
        """Return the metadata needed for later honest M20 discovery."""
        return {
            "model_family": self.model_family,
            "candidate_role": self.candidate_role,
            "scope_regimes": list(self.scope_regimes),
        }

    def __getstate__(self) -> dict[str, Any]:
        """Serialize only stable config, calibration, and the saved backend bundle."""
        return {
            "model_family": self.model_family,
            "model_class_name": self.model_class_name,
            "model_alias": self.model_alias,
            "candidate_role": self.candidate_role,
            "scope_regimes": list(self.scope_regimes),
            "horizon_candles": self.horizon_candles,
            "input_size_candles": self.input_size_candles,
            "hist_exog_columns": list(self.hist_exog_columns or ()),
            "resolved_hist_exog_columns": list(self._hist_exog_columns),
            "calibration_windows": self.calibration_windows,
            "dataset_mode": self.dataset_mode,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "scaler_type": self.scaler_type,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "val_check_steps": self.val_check_steps,
            "random_seed": self.random_seed,
            "enable_progress_bar": self.enable_progress_bar,
            "logger": self.logger,
            "refit_each_window": self.refit_each_window,
            "model_kwargs": copy.deepcopy(self.model_kwargs),
            "feature_columns": list(self._feature_columns),
            "series_id_by_symbol": dict(self._series_id_by_symbol),
            "frequency_minutes": self._frequency_minutes,
            "backend_archive": self._backend_archive,
            "calibrator": self._calibrator,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the minimal self-contained state and lazily reload the backend."""
        self.model_family = str(state["model_family"])
        self.model_class_name = str(state["model_class_name"])
        self.model_alias = str(state["model_alias"])
        self.candidate_role = str(state["candidate_role"])
        self.scope_regimes = tuple(str(regime) for regime in state["scope_regimes"])
        self.horizon_candles = int(state["horizon_candles"])
        self.input_size_candles = int(state["input_size_candles"])
        configured_hist_exog_columns = state.get("hist_exog_columns") or []
        self.hist_exog_columns = tuple(str(column) for column in configured_hist_exog_columns)
        self._hist_exog_columns = tuple(
            str(column) for column in state.get("resolved_hist_exog_columns", [])
        )
        self.calibration_windows = int(state["calibration_windows"])
        self.dataset_mode = str(state.get("dataset_mode", "local_files_partitioned"))
        self.max_steps = int(state["max_steps"])
        self.learning_rate = float(state["learning_rate"])
        self.batch_size = int(state["batch_size"])
        self.scaler_type = str(state["scaler_type"])
        self.early_stop_patience_steps = int(state["early_stop_patience_steps"])
        self.val_check_steps = int(state["val_check_steps"])
        self.random_seed = int(state["random_seed"])
        self.enable_progress_bar = bool(state["enable_progress_bar"])
        self.logger = bool(state["logger"])
        self.refit_each_window = bool(state["refit_each_window"])
        self.model_kwargs = copy.deepcopy(dict(state["model_kwargs"]))
        self._feature_columns = tuple(str(column) for column in state["feature_columns"])
        self._series_id_by_symbol = {
            str(symbol): str(series_id)
            for symbol, series_id in dict(state.get("series_id_by_symbol", {})).items()
        }
        self._frequency_minutes = (
            None
            if state.get("frequency_minutes") is None
            else int(state["frequency_minutes"])
        )
        self._backend_archive = state["backend_archive"]
        self._calibrator = state["calibrator"]
        self._runtime_dir = None
        self._backend = None

    def __del__(self) -> None:
        self._cleanup_runtime_dir()

    def _fit_calibrator(
        self,
        *,
        samples: list[DatasetSample],
        source_rows: list[SourceFeatureRow],
        fitted_backend: Any,
    ) -> ForecastToProbabilityCalibrator:
        calibration_samples = self._select_calibration_holdout(
            samples=samples,
            source_rows=source_rows,
        )
        if not calibration_samples:
            return ForecastToProbabilityCalibrator().fit([], [sample.label for sample in samples])

        calibration_rows = self._build_sequence_context_inputs(
            target_samples=calibration_samples,
            source_rows=source_rows,
        )
        raw_scores = self._predict_raw_scores_from_context_rows(
            calibration_rows,
            backend=fitted_backend,
        )
        labels = [int(sample.label) for sample in calibration_samples]
        return ForecastToProbabilityCalibrator().fit(raw_scores, labels)

    def _cross_validated_raw_scores(
        self,
        *,
        target_samples: list[DatasetSample],
        source_rows: list[SourceFeatureRow],
        progress_callback: ProgressCallback | None = None,
    ) -> list[float]:
        scoring_rows = self._build_sequence_context_inputs(
            target_samples=target_samples,
            source_rows=source_rows,
        )
        return self._predict_raw_scores_from_context_rows(
            scoring_rows,
            progress_callback=progress_callback,
        )

    def _cross_validate_rows(
        self,
        *,
        source_rows: list[SourceFeatureRow],
        n_windows: int,
    ) -> list[dict[str, Any]]:
        if n_windows <= 0:
            return []
        forecast_backend = self._build_unfitted_backend(source_rows=source_rows)
        panel_frame = self._build_panel_frame(source_rows)
        cross_validated = forecast_backend.cross_validation(
            df=panel_frame,
            n_windows=n_windows,
            step_size=1,
            verbose=False,
            refit=(True if self.refit_each_window else False),
            use_init_models=True,
        )
        return self._extract_rolling_forecast_rows(
            forecast_frame=cross_validated,
            source_rows=source_rows,
        )

    def _predict_raw_scores_from_context_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        backend: Any | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[float]:
        scoring_backend = self._ensure_backend() if backend is None else backend
        raw_scores: list[float] = []
        total_rows = len(rows)
        total_batches = (
            0
            if total_rows == 0
            else ((total_rows + _PREDICT_CONTEXT_BATCH_SIZE - 1) // _PREDICT_CONTEXT_BATCH_SIZE)
        )
        started_at = perf_counter()
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sequence_scoring_start",
                    "row_count": total_rows,
                    "batch_count": total_batches,
                    "batch_size": _PREDICT_CONTEXT_BATCH_SIZE,
                }
            )

        for batch_start in range(0, total_rows, _PREDICT_CONTEXT_BATCH_SIZE):
            batch_rows = rows[batch_start : batch_start + _PREDICT_CONTEXT_BATCH_SIZE]
            raw_scores.extend(
                self._predict_raw_scores_from_context_batch(
                    batch_rows,
                    backend=scoring_backend,
                )
            )
            if progress_callback is not None and total_batches > 0:
                completed_batches = (batch_start // _PREDICT_CONTEXT_BATCH_SIZE) + 1
                completed_rows = min(total_rows, batch_start + len(batch_rows))
                elapsed_seconds = perf_counter() - started_at
                eta_seconds = 0.0
                if completed_batches < total_batches and elapsed_seconds > 0.0:
                    eta_seconds = (
                        elapsed_seconds / float(completed_batches)
                    ) * float(total_batches - completed_batches)
                progress_callback(
                    {
                        "event": "sequence_scoring_progress",
                        "row_count": total_rows,
                        "completed_rows": completed_rows,
                        "batch_count": total_batches,
                        "completed_batches": completed_batches,
                        "progress": completed_batches / float(total_batches),
                        "elapsed_seconds": elapsed_seconds,
                        "eta_seconds": eta_seconds,
                    }
                )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sequence_scoring_complete",
                    "row_count": total_rows,
                    "batch_count": total_batches,
                    "elapsed_seconds": perf_counter() - started_at,
                }
            )
        return raw_scores

    def _predict_raw_scores_from_context_batch(
        self,
        rows: list[dict[str, Any]],
        *,
        backend: Any,
    ) -> list[float]:
        if not rows:
            return []

        batch_records: list[dict[str, Any]] = []
        expected_targets: list[tuple[str, pd.Timestamp, float]] = []
        hist_exog_columns = self._hist_exog_columns
        for row_index, row in enumerate(rows):
            context = row.get(SEQUENCE_CONTEXT_KEY)
            if not isinstance(context, list) or not context:
                raise ValueError(
                    f"{self.model_class_name} rows must include non-empty {SEQUENCE_CONTEXT_KEY}",
                )
            current_close = float(context[-1]["close_price"])
            synthetic_series_id = f"ctx_{row_index:06d}"
            for context_row in context:
                record = {
                    "unique_id": synthetic_series_id,
                    "ds": context_row["as_of_time"],
                    "y": float(context_row["close_price"]),
                }
                for column in hist_exog_columns:
                    record[column] = (
                        float(context_row["close_price"])
                        if column == "close_price"
                        else context_row[column]
                    )
                batch_records.append(record)
            target_timestamp = future_target_timestamp(
                as_of_time=pd.Timestamp(context[-1]["as_of_time"]).to_pydatetime(),
                horizon_candles=self.horizon_candles,
                frequency_minutes=self._require_frequency_minutes(),
            )
            expected_targets.append(
                (
                    synthetic_series_id,
                    pd.Timestamp(target_timestamp),
                    current_close,
                )
            )

        batch_frame = pd.DataFrame.from_records(batch_records)
        predictions = backend.predict(df=batch_frame)
        normalized_predictions = self._normalize_forecast_prediction_frame(predictions).copy()
        normalized_predictions["unique_id"] = normalized_predictions["unique_id"].astype(str)
        normalized_predictions["ds"] = pd.to_datetime(normalized_predictions["ds"])
        forecast_lookup = {
            (str(row["unique_id"]), pd.Timestamp(row["ds"])): float(row[self.model_alias])
            for row in normalized_predictions.to_dict("records")
        }

        raw_scores: list[float] = []
        for synthetic_series_id, target_timestamp, current_close in expected_targets:
            forecast_value = forecast_lookup.get((synthetic_series_id, target_timestamp))
            if forecast_value is None:
                raise ValueError(
                    f"{self.model_class_name} missing forecast for {synthetic_series_id} "
                    f"at {target_timestamp.isoformat()}",
                )
            raw_scores.append((forecast_value / current_close) - 1.0)
        return raw_scores

    def _build_sequence_context_inputs(
        self,
        *,
        target_samples: list[DatasetSample],
        source_rows: list[SourceFeatureRow],
    ) -> list[dict[str, Any]]:
        return build_sequence_context_rows(
            target_samples=target_samples,
            source_rows=source_rows,
            feature_columns=tuple(self.get_feature_columns()),
            lookback_candles=self.input_size_candles,
        )

    def _select_calibration_holdout(
        self,
        *,
        samples: list[DatasetSample],
        source_rows: list[SourceFeatureRow],
    ) -> list[DatasetSample]:
        if len(samples) < 2:
            return []

        unique_times = sorted({sample.as_of_time for sample in samples})
        if len(unique_times) < 2:
            return []

        required_symbols = sorted({source_row.symbol for source_row in source_rows})
        minimum_fit_rows_per_symbol = self.input_size_candles + self.horizon_candles + 1
        max_holdout_timestamps = min(self.calibration_windows, len(unique_times) - 1)

        for holdout_timestamp_count in range(max_holdout_timestamps, 0, -1):
            holdout_start = unique_times[-holdout_timestamp_count]
            holdout_times = set(unique_times[-holdout_timestamp_count:])
            calibration_samples = [
                sample
                for sample in samples
                if sample.as_of_time in holdout_times
            ]
            fit_source_rows = [
                source_row
                for source_row in source_rows
                if source_row.as_of_time < holdout_start
            ]
            if not calibration_samples or not fit_source_rows:
                continue
            grouped_fit_rows = _group_source_rows_by_symbol(fit_source_rows)
            if all(
                len(grouped_fit_rows.get(symbol, [])) >= minimum_fit_rows_per_symbol
                for symbol in required_symbols
            ):
                return calibration_samples

        return []

    def _fit_backend(
        self,
        *,
        source_rows: list[SourceFeatureRow],
        dataset_export_root: Path | None,
    ) -> Any:
        backend = self._build_unfitted_backend(source_rows=source_rows)
        val_size = self._resolve_fit_val_size(source_rows=source_rows)
        if self.dataset_mode == "local_files_partitioned":
            export_directories, export_root_path, cleanup_export_root = (
                self._export_partitioned_training_dataset(
                    source_rows=source_rows,
                    export_root=dataset_export_root,
                )
            )
            try:
                backend.fit(
                    df=export_directories,
                    val_size=val_size,
                    sort_df=False,
                )
            finally:
                if cleanup_export_root:
                    shutil.rmtree(export_root_path, ignore_errors=True)
        else:
            backend.fit(
                df=self._build_panel_frame(source_rows),
                val_size=val_size,
            )
        return backend

    def _resolve_fit_val_size(
        self,
        *,
        source_rows: list[SourceFeatureRow],
    ) -> int:
        if self.early_stop_patience_steps <= 0:
            return 0
        grouped_rows = _group_source_rows_by_symbol(source_rows)
        if not grouped_rows:
            return 0
        max_supported_val_size = min(
            max(len(symbol_rows) - self.input_size_candles - self.horizon_candles, 0)
            for symbol_rows in grouped_rows.values()
        )
        if max_supported_val_size <= 0:
            return 0
        return max(1, min(self.horizon_candles, max_supported_val_size))

    def _build_unfitted_backend(
        self,
        *,
        source_rows: list[SourceFeatureRow],
    ) -> Any:
        neuralforecast_core, model_class = _import_neuralforecast_runtime(
            self.model_class_name
        )
        resolved_model_kwargs = copy.deepcopy(self.model_kwargs)
        if (
            str(resolved_model_kwargs.get("precision", "")).lower() == "16-mixed"
            and not _torch_cuda_is_available()
        ):
            resolved_model_kwargs["precision"] = "32-true"
        model = model_class(
            h=self.horizon_candles,
            input_size=self.input_size_candles,
            hist_exog_list=list(self._resolved_hist_exog_columns(source_rows)),
            scaler_type=self.scaler_type,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            early_stop_patience_steps=self.early_stop_patience_steps,
            val_check_steps=self.val_check_steps,
            random_seed=self.random_seed,
            alias=self.model_alias,
            enable_checkpointing=True,
            enable_progress_bar=self.enable_progress_bar,
            logger=self.logger,
            **resolved_model_kwargs,
        )
        return neuralforecast_core(
            models=[model],
            freq=f"{infer_source_frequency_minutes(source_rows)}min",
        )

    def _export_partitioned_training_dataset(
        self,
        *,
        source_rows: list[SourceFeatureRow],
        export_root: Path | None,
    ) -> tuple[list[str], Path, bool]:
        export_root_path = (
            create_local_training_work_dir(
                prefix=f"streamalpha-{self.model_alias.lower()}-local-files-",
            )
            if export_root is None
            else Path(export_root)
        )
        cleanup_export_root = export_root is None
        if export_root is not None:
            shutil.rmtree(export_root_path, ignore_errors=True)
            export_root_path.mkdir(parents=True, exist_ok=True)

        hist_exog_columns = self._resolved_hist_exog_columns(source_rows)
        grouped_rows = _group_source_rows_by_symbol(source_rows)
        directories: list[str] = []
        manifest_payload: dict[str, Any] = {
            "dataset_mode": self.dataset_mode,
            "model_alias": self.model_alias,
            "source_row_count": len(source_rows),
            "symbols": {},
        }
        for symbol, symbol_rows in sorted(grouped_rows.items()):
            series_id = self._resolve_series_id(symbol)
            symbol_directory = export_root_path / f"unique_id={series_id}"
            symbol_directory.mkdir(parents=True, exist_ok=True)
            parquet_path = symbol_directory / "part-00000.parquet"
            frame = self._build_partition_frame(
                symbol_rows=symbol_rows,
                hist_exog_columns=hist_exog_columns,
            )
            frame.to_parquet(parquet_path, index=False)
            directories.append(str(symbol_directory))
            manifest_payload["symbols"][symbol] = {
                "series_id": series_id,
                "row_count": len(symbol_rows),
                "earliest_as_of_time": symbol_rows[0].as_of_time.isoformat(),
                "latest_as_of_time": symbol_rows[-1].as_of_time.isoformat(),
                "directory": str(symbol_directory),
                "parquet_path": str(parquet_path),
            }

        (export_root_path / "dataset_export.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            "[training] "
            f"{self.model_alias} specialist dataset mode: {self.dataset_mode} "
            f"({len(source_rows)} source rows -> {len(directories)} parquet partitions at {export_root_path})"
        )
        return directories, export_root_path, cleanup_export_root

    def _build_partition_frame(
        self,
        *,
        symbol_rows: list[SourceFeatureRow],
        hist_exog_columns: tuple[str, ...],
    ) -> pd.DataFrame:
        symbol = symbol_rows[0].symbol
        series_id = self._resolve_series_id(symbol)
        records: list[dict[str, Any]] = []
        for source_row in symbol_rows:
            record = {
                "unique_id": series_id,
                "ds": source_row.as_of_time,
                "y": source_row.close_price,
            }
            for column in hist_exog_columns:
                record[column] = (
                    source_row.close_price
                    if column == "close_price"
                    else source_row.features[column]
                )
            records.append(record)
        return pd.DataFrame.from_records(records)

    def _build_panel_frame(
        self,
        source_rows: list[SourceFeatureRow],
    ) -> pd.DataFrame:
        hist_exog_columns = self._resolved_hist_exog_columns(source_rows)
        records: list[dict[str, Any]] = []
        for source_row in sorted(source_rows, key=lambda row: (row.symbol, row.as_of_time)):
            record = {
                "unique_id": self._resolve_series_id(source_row.symbol),
                "ds": source_row.as_of_time,
                "y": source_row.close_price,
            }
            for column in hist_exog_columns:
                record[column] = (
                    source_row.close_price
                    if column == "close_price"
                    else source_row.features[column]
                )
            records.append(record)
        return pd.DataFrame.from_records(records)

    def _build_context_panel_frame(
        self,
        context_rows: list[dict[str, Any]],
    ) -> pd.DataFrame:
        hist_exog_columns = self._hist_exog_columns
        records: list[dict[str, Any]] = []
        for context_row in context_rows:
            record = {
                "unique_id": self._resolve_series_id(str(context_row["symbol"])),
                "ds": context_row["as_of_time"],
                "y": float(context_row["close_price"]),
            }
            for column in hist_exog_columns:
                record[column] = (
                    float(context_row["close_price"])
                    if column == "close_price"
                    else context_row[column]
                )
            records.append(record)
        return pd.DataFrame.from_records(records)

    def _extract_rolling_forecast_rows(
        self,
        *,
        forecast_frame: Any,
        source_rows: list[SourceFeatureRow],
    ) -> list[dict[str, Any]]:
        frame = (
            forecast_frame
            if isinstance(forecast_frame, pd.DataFrame)
            else pd.DataFrame(forecast_frame)
        )
        frequency_minutes = infer_source_frequency_minutes(source_rows)
        close_by_key = {
            (
                self._resolve_series_id(source_row.symbol),
                pd.Timestamp(source_row.as_of_time),
            ): float(source_row.close_price)
            for source_row in source_rows
        }
        rows: list[dict[str, Any]] = []
        for record in frame.to_dict("records"):
            cutoff_timestamp = pd.Timestamp(record["cutoff"])
            forecast_timestamp = pd.Timestamp(record["ds"])
            expected_timestamp = pd.Timestamp(
                future_target_timestamp(
                    as_of_time=cutoff_timestamp.to_pydatetime(),
                    horizon_candles=self.horizon_candles,
                    frequency_minutes=frequency_minutes,
                )
            )
            if forecast_timestamp != expected_timestamp:
                continue
            symbol = str(record["unique_id"])
            current_close = close_by_key.get((symbol, cutoff_timestamp))
            if current_close is None:
                continue
            forecast_value = float(record[self.model_alias])
            rows.append(
                {
                    "symbol": self._reverse_series_id(symbol),
                    "cutoff": cutoff_timestamp.to_pydatetime(),
                    "target_timestamp": forecast_timestamp.to_pydatetime(),
                    "raw_score": (forecast_value / current_close) - 1.0,
                }
            )
        return rows

    def _extract_forecast_value(
        self,
        *,
        forecast_frame: Any,
        symbol: str,
        target_timestamp: Any,
    ) -> float:
        frame = self._normalize_forecast_prediction_frame(forecast_frame)
        target = pd.Timestamp(target_timestamp)
        series_id = self._resolve_series_id(symbol)
        matched = frame[
            (frame["unique_id"].astype(str) == series_id)
            & (pd.to_datetime(frame["ds"]) == target)
        ]
        if matched.empty:
            raise ValueError(
                f"{self.model_class_name} missing forecast for {symbol} at {target.isoformat()}",
            )
        return float(matched.iloc[0][self.model_alias])

    def _normalize_forecast_prediction_frame(
        self,
        forecast_frame: Any,
    ) -> pd.DataFrame:
        frame = (
            forecast_frame
            if isinstance(forecast_frame, pd.DataFrame)
            else pd.DataFrame(forecast_frame)
        )
        if "unique_id" in frame.columns and "ds" in frame.columns:
            return frame

        normalized = frame.reset_index()
        if "unique_id" in normalized.columns and "ds" in normalized.columns:
            return normalized
        series_id_column = self._resolve_forecast_series_id_column(normalized)
        timestamp_column = self._resolve_forecast_timestamp_column(
            normalized,
            excluded_columns={series_id_column},
        )
        rename_map: dict[str, str] = {}
        if series_id_column != "unique_id":
            rename_map[series_id_column] = "unique_id"
        if timestamp_column != "ds":
            rename_map[timestamp_column] = "ds"
        return normalized.rename(columns=rename_map)

    def _resolve_forecast_series_id_column(
        self,
        frame: pd.DataFrame,
    ) -> str:
        expected_series_ids = set(self._series_id_by_symbol.values())
        preferred_columns = ("unique_id", "series_id", "id", "index", "level_0")
        for column_name in preferred_columns:
            if column_name not in frame.columns:
                continue
            if not expected_series_ids:
                return column_name
            values = {
                str(value)
                for value in frame[column_name].dropna().astype(str).tolist()
            }
            if values & expected_series_ids:
                return column_name

        for column_name in frame.columns:
            values = {
                str(value)
                for value in frame[column_name].dropna().astype(str).tolist()
            }
            if values & expected_series_ids:
                return str(column_name)

        raise ValueError(
            f"{self.model_class_name} forecast output is missing a recoverable series-id field. "
            f"Columns after reset_index(): {list(frame.columns)}"
        )

    def _resolve_forecast_timestamp_column(
        self,
        frame: pd.DataFrame,
        *,
        excluded_columns: set[str],
    ) -> str:
        preferred_columns = ("ds", "timestamp", "time", "index", "level_1")
        for column_name in preferred_columns:
            if column_name in excluded_columns or column_name not in frame.columns:
                continue
            if _column_has_timestamp_values(frame[column_name]):
                return column_name

        for column_name in frame.columns:
            if str(column_name) in excluded_columns:
                continue
            if _column_has_timestamp_values(frame[column_name]):
                return str(column_name)

        raise ValueError(
            f"{self.model_class_name} forecast output is missing a recoverable timestamp field. "
            f"Columns after reset_index(): {list(frame.columns)}"
        )

    def _ensure_backend(self) -> Any:
        if self._backend is None:
            if self._backend_archive is None:
                raise ValueError(f"{self.model_class_name} backend archive is missing")
            self._cleanup_runtime_dir()
            runtime_root = create_local_training_work_dir(
                prefix=f"streamalpha-{self.model_alias.lower()}-runtime-",
            )
            backend_dir = runtime_root / "backend"
            try:
                _restore_backend_dir(self._backend_archive, backend_dir)
                neuralforecast_core, _ = _import_neuralforecast_runtime(self.model_class_name)
                self._backend = neuralforecast_core.load(path=str(backend_dir))
                self._runtime_dir = runtime_root
            except Exception:
                shutil.rmtree(runtime_root, ignore_errors=True)
                raise
        return self._backend

    def _cleanup_runtime_dir(self) -> None:
        if self._runtime_dir is None:
            return
        shutil.rmtree(self._runtime_dir, ignore_errors=True)
        self._runtime_dir = None

    def _require_calibrator(self) -> ForecastToProbabilityCalibrator:
        if self._calibrator is None:
            raise ValueError(f"{self.model_class_name} calibrator is missing from the artifact")
        return self._calibrator

    def _require_frequency_minutes(self) -> int:
        if self._frequency_minutes is None:
            raise ValueError(f"{self.model_class_name} frequency is missing from the artifact")
        return self._frequency_minutes

    def _save_backend_archive(
        self,
        *,
        backend: Any,
        save_dir: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        try:
            backend.save(
                path=str(save_dir),
                overwrite=True,
                save_dataset=True,
            )
            return
        except Exception as error:  # pragma: no cover - exercised via focused test double
            if not _is_neuralforecast_dataset_save_compatibility_error(error):
                raise
            message = (
                f"{self.model_alias} backend archive: saving without embedded dataset "
                "because the current NeuralForecast backend shape does not expose "
                "full in-memory dataset state."
            )
            print(f"[training] {message}")
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "backend_archive_dataset_fallback",
                        "message": message,
                    }
                )
            backend.save(
                path=str(save_dir),
                overwrite=True,
                save_dataset=False,
            )

    def _resolved_hist_exog_columns(
        self,
        source_rows: list[SourceFeatureRow],
    ) -> tuple[str, ...]:
        if self._hist_exog_columns:
            return self._hist_exog_columns
        resolved = _resolve_hist_exog_columns(
            source_rows,
            configured_columns=self.hist_exog_columns,
        )
        self._hist_exog_columns = resolved
        return resolved

    def _resolve_series_id(self, symbol: str) -> str:
        if symbol in self._series_id_by_symbol:
            return self._series_id_by_symbol[symbol]
        resolved_series_id = _encode_series_id(symbol)
        self._series_id_by_symbol[symbol] = resolved_series_id
        return resolved_series_id

    def _reverse_series_id(self, series_id: str) -> str:
        for symbol, mapped_series_id in self._series_id_by_symbol.items():
            if mapped_series_id == series_id:
                return symbol
        return series_id


class NeuralForecastNHITSClassifier(_BaseNeuralForecastClassifier):
    """Real NHITS specialist wrapper for offline Stream Alpha research runs."""


class NeuralForecastPatchTSTClassifier(_BaseNeuralForecastClassifier):
    """Real PatchTST specialist wrapper for offline Stream Alpha research runs."""


def build_neuralforecast_nhits_classifier(
    model_config: dict[str, Any],
) -> NeuralForecastNHITSClassifier:
    """Build the real NHITS specialist challenger wrapper."""
    return NeuralForecastNHITSClassifier(
        model_family=MODEL_FAMILY_NEURALFORECAST_NHITS,
        model_class_name="NHITS",
        model_alias="NHITS",
        candidate_role=str(model_config.get("candidate_role", TREND_SPECIALIST)),
        scope_regimes=_resolve_scope_regimes(
            model_config,
            default_scope_regimes=("TREND_UP", "TREND_DOWN"),
        ),
        horizon_candles=int(model_config.get("horizon_candles", 3)),
        input_size_candles=int(model_config.get("input_size_candles", 96)),
        hist_exog_columns=_resolve_optional_columns(model_config.get("hist_exog_columns")),
        calibration_windows=int(
            model_config.get("calibration_windows", _DEFAULT_CALIBRATION_WINDOWS)
        ),
        dataset_mode=str(model_config.get("dataset_mode", "local_files_partitioned")),
        max_steps=int(model_config.get("max_steps", 200)),
        learning_rate=float(model_config.get("learning_rate", 1e-3)),
        batch_size=int(model_config.get("batch_size", 32)),
        scaler_type=str(model_config.get("scaler_type", "robust")),
        early_stop_patience_steps=int(model_config.get("early_stop_patience_steps", 20)),
        val_check_steps=int(model_config.get("val_check_steps", 50)),
        random_seed=int(model_config.get("random_seed", 1)),
        enable_progress_bar=bool(model_config.get("enable_progress_bar", False)),
        logger=bool(model_config.get("logger", False)),
        refit_each_window=bool(model_config.get("refit_each_window", False)),
        model_kwargs=dict(model_config.get("model_kwargs", {})),
    )


def build_neuralforecast_patchtst_classifier(
    model_config: dict[str, Any],
) -> NeuralForecastPatchTSTClassifier:
    """Build the real PatchTST specialist challenger wrapper."""
    return NeuralForecastPatchTSTClassifier(
        model_family=MODEL_FAMILY_NEURALFORECAST_PATCHTST,
        model_class_name="PatchTST",
        model_alias="PatchTST",
        candidate_role=str(model_config.get("candidate_role", RANGE_SPECIALIST)),
        scope_regimes=_resolve_scope_regimes(
            model_config,
            default_scope_regimes=("RANGE",),
        ),
        horizon_candles=int(model_config.get("horizon_candles", 3)),
        input_size_candles=int(model_config.get("input_size_candles", 96)),
        hist_exog_columns=_resolve_optional_columns(model_config.get("hist_exog_columns")),
        calibration_windows=int(
            model_config.get("calibration_windows", _DEFAULT_CALIBRATION_WINDOWS)
        ),
        dataset_mode=str(model_config.get("dataset_mode", "local_files_partitioned")),
        max_steps=int(model_config.get("max_steps", 200)),
        learning_rate=float(model_config.get("learning_rate", 1e-3)),
        batch_size=int(model_config.get("batch_size", 32)),
        scaler_type=str(model_config.get("scaler_type", "robust")),
        early_stop_patience_steps=int(model_config.get("early_stop_patience_steps", 20)),
        val_check_steps=int(model_config.get("val_check_steps", 50)),
        random_seed=int(model_config.get("random_seed", 1)),
        enable_progress_bar=bool(model_config.get("enable_progress_bar", False)),
        logger=bool(model_config.get("logger", False)),
        refit_each_window=bool(model_config.get("refit_each_window", False)),
        model_kwargs=dict(model_config.get("model_kwargs", {})),
    )


def _resolve_optional_columns(columns: Any) -> tuple[str, ...] | None:
    if columns is None:
        return None
    return tuple(str(column) for column in columns)


def _resolve_scope_regimes(
    model_config: dict[str, Any],
    *,
    default_scope_regimes: tuple[str, ...],
) -> tuple[str, ...]:
    configured = model_config.get("scope_regimes")
    if configured is None:
        return default_scope_regimes
    return tuple(str(regime) for regime in configured)


def _resolve_feature_columns(
    source_rows: list[SourceFeatureRow],
) -> tuple[str, ...]:
    if not source_rows:
        raise ValueError("Sequence models require source feature rows")
    first_row = source_rows[0]
    feature_columns = ["symbol"]
    ordered_feature_names = []
    for column in first_row.features:
        if column == "symbol":
            continue
        ordered_feature_names.append(str(column))
    return tuple(feature_columns + ordered_feature_names)


def _resolve_hist_exog_columns(
    source_rows: list[SourceFeatureRow],
    *,
    configured_columns: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if configured_columns is not None:
        return tuple(str(column) for column in configured_columns)
    if not source_rows:
        raise ValueError("Sequence models require source feature rows to resolve exogenous inputs")
    candidate_columns: list[str] = []
    for column, value in source_rows[0].features.items():
        if column in {"symbol", "close_price"}:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            candidate_columns.append(str(column))
    return tuple(candidate_columns)


def _group_source_rows_by_symbol(
    source_rows: list[SourceFeatureRow],
) -> dict[str, list[SourceFeatureRow]]:
    grouped: dict[str, list[SourceFeatureRow]] = {}
    for source_row in source_rows:
        grouped.setdefault(source_row.symbol, []).append(source_row)
    for symbol, symbol_rows in grouped.items():
        grouped[symbol] = sorted(symbol_rows, key=lambda row: row.as_of_time)
    return grouped


def _build_series_id_map(
    source_rows: list[SourceFeatureRow],
) -> dict[str, str]:
    return {
        symbol: _encode_series_id(symbol)
        for symbol in sorted({source_row.symbol for source_row in source_rows})
    }


def _encode_series_id(symbol: str) -> str:
    return quote(symbol, safe="")


def _column_has_timestamp_values(values: Any) -> bool:
    series = pd.Series(values)
    non_null = series.dropna()
    if non_null.empty:
        return False
    converted = pd.to_datetime(non_null, errors="coerce")
    return bool(converted.notna().any())


def _is_neuralforecast_dataset_save_compatibility_error(error: Exception) -> bool:
    message = str(error)
    compatibility_markers = (
        "object has no attribute 'ds'",
        "Cannot save distributed dataset",
        "You need to have a stored dataset to save it",
    )
    return any(marker in message for marker in compatibility_markers)


def _archive_backend_dir(backend_dir: Path) -> bytes:
    """Zip one NeuralForecast save directory into the single artifact payload."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(backend_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, arcname=path.relative_to(backend_dir).as_posix())
    return buffer.getvalue()


def _restore_backend_dir(archive_bytes: bytes, backend_dir: Path) -> None:
    """Restore one NeuralForecast save directory from the archived artifact payload."""
    backend_dir.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
        archive.extractall(backend_dir)


def _import_neuralforecast_runtime(model_class_name: str) -> tuple[Any, Any]:
    """Import the bounded NeuralForecast runtime only when the wrappers are exercised."""
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS, PatchTST
    except ImportError as error:  # pragma: no cover - exercised through unit doubles
        raise ValueError(
            "NeuralForecast specialist models require the optional neuralforecast "
            "and lightning dependencies to be installed.",
        ) from error
    model_classes = {
        "NHITS": NHITS,
        "PatchTST": PatchTST,
    }
    if model_class_name not in model_classes:
        raise ValueError(f"Unsupported NeuralForecast model class: {model_class_name}")
    return NeuralForecast, model_classes[model_class_name]


def _release_torch_cuda_cache() -> None:
    """Free temporary CUDA allocations after bounded calibration fits when possible."""
    try:
        import gc

        gc.collect()

        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _torch_cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())
