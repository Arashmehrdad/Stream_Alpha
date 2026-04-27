"""Offline training orchestration for Stream Alpha M3."""

from __future__ import annotations

import csv
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import joblib
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
)

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339, utc_now
from app.regime.config import load_regime_config
from app.regime.dataset import RegimeSourceRow
from app.regime.service import (
    REGIME_LABELS,
    SymbolThresholds,
    classify_row,
    compute_percentile,
)
from app.training.baselines import PersistenceBaseline, build_dummy_classifier
from app.training.autogluon import build_autogluon_tabular_classifier
from app.training.data_readiness import assert_training_data_ready
from app.training.dataset import (
    DatasetSample,
    LEGACY_ARCHIVED_MODEL_NAMES,
    SourceFeatureRow,
    TrainingConfig,
    load_training_config,
    load_training_dataset,
)
from app.training.incumbent_scoring import (
    load_incumbent_model as _load_incumbent_model,
    score_incumbent_on_recent_samples as _score_incumbent_on_recent_samples_impl,
)
from app.training.neuralforecast import (
    build_neuralforecast_nhits_classifier,
    build_neuralforecast_patchtst_classifier,
)
from app.training.pretrained_forecasters import (
    build_chronos2_classifier,
    build_moirai_1_0_r_base_range_classifier,
    build_timesfm_2_0_500m_pytorch_classifier,
    is_pretrained_forecaster_model,
    validate_pretrained_forecaster_contract,
)
from app.training.specialist_verdicts import (
    build_specialist_verdicts as _build_specialist_verdicts_impl,
    compute_max_drawdown as _compute_max_drawdown,
    filter_recent_predictions as _filter_recent_predictions,
)
from app.training.splits import WalkForwardFold, build_walk_forward_splits
from app.training.splits import minimum_required_unique_timestamps


_REGIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "regime.m8.json"
_REQUIRED_PROMOTION_BASELINES = ("persistence_3", "dummy_most_frequent")
_AUTHORITATIVE_MODEL_BUILDERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "autogluon_tabular": build_autogluon_tabular_classifier,
    "chronos2_generalist": build_chronos2_classifier,
    "moirai_1_0_r_base_range": build_moirai_1_0_r_base_range_classifier,
    "timesfm_2_0_500m_pytorch_trend": build_timesfm_2_0_500m_pytorch_classifier,
    "neuralforecast_nhits": build_neuralforecast_nhits_classifier,
    "neuralforecast_patchtst": build_neuralforecast_patchtst_classifier,
}


@dataclass(frozen=True, slots=True)
class PredictionRecord:  # pylint: disable=too-many-instance-attributes
    """One out-of-fold prediction emitted by a baseline or learned model."""

    model_name: str
    fold_index: int
    row_id: str
    symbol: str
    interval_begin: str
    as_of_time: str
    y_true: int
    y_pred: int
    prob_up: float
    confidence: float
    regime_label: str
    long_trade_taken: int
    future_return_3: float
    long_only_gross_value_proxy: float
    long_only_net_value_proxy: float

    def to_csv_row(self) -> dict[str, Any]:
        """Return a CSV-friendly representation for artifact persistence."""
        return {
            "model_name": self.model_name,
            "fold_index": self.fold_index,
            "row_id": self.row_id,
            "symbol": self.symbol,
            "interval_begin": self.interval_begin,
            "as_of_time": self.as_of_time,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "prob_up": self.prob_up,
            "confidence": self.confidence,
            "regime_label": self.regime_label,
            "long_trade_taken": self.long_trade_taken,
            "future_return_3": self.future_return_3,
            "long_only_gross_value_proxy": self.long_only_gross_value_proxy,
            "long_only_net_value_proxy": self.long_only_net_value_proxy,
        }


@dataclass(frozen=True, slots=True)
class TrainingRegimeContext:
    """Deterministic M8-style regime labels used for training economics slices."""

    config_path: str
    high_vol_percentile: float
    trend_abs_momentum_percentile: float
    thresholds_by_symbol: dict[str, SymbolThresholds]
    labels_by_row_id: dict[str, str]


@dataclass(frozen=True, slots=True)
class SavedCandidateArtifact:
    """One full-fit learned-model artifact bundle persisted for later challenger use."""

    model_name: str
    model_path: Path
    fitted_model: Any
    feature_columns: tuple[str, ...]
    expanded_feature_names: list[str]
    training_model_config: dict[str, Any] | None
    registry_metadata: dict[str, Any] | None


class _TrainingProgressRecorder:
    """Artifact-backed progress trail for long-running offline training jobs."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.log_path = artifact_dir / "progress.log"
        self.status_path = artifact_dir / "progress_status.json"
        self._last_sequence_log_batches: dict[str, int] = {}

    def record(
        self,
        *,
        stage: str,
        message: str,
        state: str = "running",
        **metadata: Any,
    ) -> None:
        timestamp = to_rfc3339(utc_now())
        entry = {
            "timestamp": timestamp,
            "state": state,
            "stage": stage,
            "message": message,
            **metadata,
        }
        self._append_log_line(
            f"{timestamp} [{stage}] {message}{_format_log_metadata(metadata)}"
        )
        _write_json(self.status_path, entry)

    def record_sequence_event(
        self,
        *,
        model_name: str,
        payload: dict[str, Any],
        fold_index: int | None = None,
        total_folds: int | None = None,
    ) -> None:
        timestamp = to_rfc3339(utc_now())
        event = str(payload.get("event", "sequence_event"))
        entry = {
            "timestamp": timestamp,
            "state": "running",
            "stage": "sequence_scoring",
            "event": event,
            "model_name": model_name,
            **payload,
        }
        if fold_index is not None:
            entry["fold_index"] = fold_index + 1
        if total_folds is not None:
            entry["total_folds"] = total_folds
        _write_json(self.status_path, entry)

        fold_label = (
            ""
            if fold_index is None
            else f" fold {fold_index + 1}/{total_folds}"
            if total_folds is not None
            else f" fold {fold_index + 1}"
        )
        prefix = f"{timestamp} [sequence_scoring]{fold_label} [{model_name}] "

        if event == "sequence_scoring_start":
            self._append_log_line(
                prefix
                + "started"
                + _format_log_metadata(
                    {
                        "rows": payload.get("row_count"),
                        "batches": payload.get("batch_count"),
                        "batch_size": payload.get("batch_size"),
                    }
                )
            )
            return

        if event == "sequence_scoring_progress":
            completed_batches = int(payload.get("completed_batches", 0))
            total_batches = max(1, int(payload.get("batch_count", 1)))
            sequence_key = f"{fold_index}:{model_name}" if fold_index is not None else model_name
            log_interval = max(1, total_batches // 20)
            last_logged_batch = self._last_sequence_log_batches.get(sequence_key, 0)
            should_log = (
                completed_batches == 1
                or completed_batches == total_batches
                or (completed_batches - last_logged_batch) >= log_interval
            )
            if not should_log:
                return
            self._last_sequence_log_batches[sequence_key] = completed_batches
            bar = _render_progress_bar(float(payload.get("progress", 0.0)))
            self._append_log_line(
                prefix
                + f"{bar} batches={completed_batches}/{total_batches}"
                + _format_log_metadata(
                    {
                        "rows": (
                            f"{int(payload.get('completed_rows', 0))}/"
                            f"{int(payload.get('row_count', 0))}"
                        ),
                        "elapsed": _format_eta_seconds(payload.get("elapsed_seconds")),
                        "eta": _format_eta_seconds(payload.get("eta_seconds")),
                    }
                )
            )
            return

        if event == "sequence_scoring_complete":
            self._append_log_line(
                prefix
                + "completed"
                + _format_log_metadata(
                    {
                        "rows": payload.get("row_count"),
                        "batches": payload.get("batch_count"),
                        "elapsed": _format_eta_seconds(payload.get("elapsed_seconds")),
                    }
                )
            )
            return

        if event == "backend_archive_dataset_fallback":
            self._append_log_line(prefix + str(payload.get("message", "archive fallback")))
            return

        self._append_log_line(
            prefix
            + event
            + _format_log_metadata(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "event"
                }
            )
        )

    def _append_log_line(self, line: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as output_file:
            output_file.write(f"{line}\n")


class _ConsoleProgress:
    """Live console ETA progress display for the walk-forward evaluation loop."""

    def __init__(self, total_folds: int, total_models: int) -> None:
        self.total_folds = total_folds
        self.total_models = total_models
        self.total_units = total_folds * total_models
        self.completed_units = 0
        self.start_time = time.monotonic()

    def tick(self, *, fold_index: int, model_name: str) -> None:
        self.completed_units += 1
        elapsed = time.monotonic() - self.start_time
        remaining_units = self.total_units - self.completed_units
        if self.completed_units > 0:
            per_unit = elapsed / self.completed_units
            eta = per_unit * remaining_units
        else:
            eta = 0.0
        pct = self.completed_units / self.total_units
        bar = _render_progress_bar(pct, width=30)
        eta_str = _format_eta_seconds(eta)
        elapsed_str = _format_eta_seconds(elapsed)
        print(
            f"\r[training] {bar}  "
            f"fold {fold_index + 1}/{self.total_folds} | "
            f"{model_name} done  "
            f"elapsed={elapsed_str} eta={eta_str}    ",
            end="",
            flush=True,
        )
        if self.completed_units == self.total_units:
            print()

    def model_start(self, *, fold_index: int, model_name: str) -> None:
        elapsed = time.monotonic() - self.start_time
        elapsed_str = _format_eta_seconds(elapsed)
        pct = self.completed_units / self.total_units
        bar = _render_progress_bar(pct, width=30)
        print(
            f"\r[training] {bar}  "
            f"fold {fold_index + 1}/{self.total_folds} | "
            f"scoring {model_name}...  "
            f"elapsed={elapsed_str}       ",
            end="",
            flush=True,
        )


def _save_fold_checkpoint(
    artifact_dir: Path,
    *,
    fold_metric_rows: list[dict[str, Any]],
    all_prediction_rows: list[PredictionRecord],
    completed_fold_indices: list[int],
    partial_fold_index: int | None = None,
    completed_models_in_partial_fold: list[str] | None = None,
) -> None:
    """Persist checkpoint so the run can be resumed after interruption.

    Supports both fold-level and model-level granularity.
    """
    checkpoint: dict[str, Any] = {
        "completed_fold_indices": completed_fold_indices,
        "fold_metric_rows": fold_metric_rows,
        "all_prediction_rows": [row.to_csv_row() for row in all_prediction_rows],
    }
    if partial_fold_index is not None:
        checkpoint["partial_fold_index"] = partial_fold_index
        checkpoint["completed_models_in_partial_fold"] = (
            completed_models_in_partial_fold or []
        )
    _write_json(artifact_dir / "checkpoint.json", checkpoint)


@dataclass
class _CheckpointState:
    """Loaded checkpoint state for resume."""

    completed_fold_indices: list[int]
    fold_metric_rows: list[dict[str, Any]]
    all_prediction_rows: list[PredictionRecord]
    partial_fold_index: int | None = None
    completed_models_in_partial_fold: list[str] | None = None


def _load_fold_checkpoint(artifact_dir: Path) -> _CheckpointState | None:
    """Load a saved checkpoint. Returns None if no checkpoint exists."""
    checkpoint_path = artifact_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        return None
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return _CheckpointState(
        completed_fold_indices=raw["completed_fold_indices"],
        fold_metric_rows=raw["fold_metric_rows"],
        all_prediction_rows=[
            PredictionRecord(**row) for row in raw["all_prediction_rows"]
        ],
        partial_fold_index=raw.get("partial_fold_index"),
        completed_models_in_partial_fold=raw.get(
            "completed_models_in_partial_fold"
        ),
    )


def _remove_checkpoint(artifact_dir: Path) -> None:
    """Remove the checkpoint file after successful completion."""
    checkpoint_path = artifact_dir / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def run_training(
    config_path: Path,
    *,
    resume_artifact_dir: Path | None = None,
    parquet_dir: Path | None = None,
    fit_only: bool = False,
    score_only_dir: Path | None = None,
) -> Path:
    """Run the full offline M3 training flow and save artifacts to disk.

    If *resume_artifact_dir* is given, the run continues from the last
    completed fold checkpoint found inside that directory.

    If *parquet_dir* is given, load training data from exported parquet files
    instead of PostgreSQL.

    If *fit_only* is True, fit models on each fold and on the full dataset,
    save the fitted estimators to ``fitted_models/`` inside the artifact dir,
    then return without scoring.

    If *score_only_dir* is given, load pre-fitted estimators from that
    directory and run scoring only (skip fitting).
    """
    config = load_training_config(config_path)
    _validate_authoritative_model_stack(config)
    if parquet_dir is not None:
        print(f"[training] loading dataset from parquet: {parquet_dir}")
    else:
        print(
            "[training] readiness gate: probing "
            f"{config.source_table} for {list(config.symbols)}"
        )
        assert_training_data_ready(config, config_path=config_path)
        print("[training] readiness gate passed")
        print(f"[training] loading full offline dataset from {config.source_table}")
    dataset = load_training_dataset(config, parquet_dir=parquet_dir)
    print(
        "[training] dataset loaded: "
        f"source_rows={len(dataset.source_rows)}, "
        f"labeled_rows={len(dataset.samples)}, "
        f"unique_timestamps={int(dataset.manifest['unique_timestamps'])}"
    )
    _validate_split_readiness(dataset, config)
    regime_context = _build_training_regime_context(dataset.samples, config)
    folds = build_walk_forward_splits(
        dataset.timestamps,
        first_train_fraction=config.first_train_fraction,
        test_fraction=config.test_fraction,
        test_folds=config.test_folds,
        purge_gap_candles=config.purge_gap_candles,
    )

    # --- resume vs. fresh run ---
    completed_fold_indices: list[int] = []
    resume_partial_fold_index: int | None = None
    resume_completed_models: list[str] | None = None
    if resume_artifact_dir is not None:
        artifact_dir = resume_artifact_dir
        loaded = _load_fold_checkpoint(artifact_dir)
        if loaded is not None:
            completed_fold_indices = loaded.completed_fold_indices
            fold_metric_rows = loaded.fold_metric_rows
            all_prediction_rows = loaded.all_prediction_rows
            resume_partial_fold_index = loaded.partial_fold_index
            resume_completed_models = loaded.completed_models_in_partial_fold
            partial_msg = ""
            if resume_partial_fold_index is not None:
                partial_msg = (
                    f" (fold {resume_partial_fold_index + 1} partially done: "
                    f"{len(resume_completed_models or [])} models)"
                )
            print(
                f"[training] resuming from checkpoint — "
                f"{len(completed_fold_indices)}/{len(folds)} folds done{partial_msg}"
            )
        else:
            raise ValueError(
                f"--resume specified but no checkpoint.json found in {artifact_dir}"
            )
    else:
        artifact_dir = _create_artifact_dir(config)
        fold_metric_rows = []
        all_prediction_rows = []

    execution_mode = (
        "score_only"
        if score_only_dir is not None
        else "fit_only"
        if fit_only
        else "full_training"
    )
    run_config_payload = {
        **config.to_dict(),
        "execution_mode": execution_mode,
        **(
            {"score_only_dir": str(score_only_dir)}
            if score_only_dir is not None
            else {}
        ),
    }
    _write_json(artifact_dir / "run_config.json", run_config_payload)
    _write_json(
        artifact_dir / "dataset_manifest.json",
        {
            **dataset.manifest,
            "source_schema": list(dataset.source_schema),
            "feature_columns": list(dataset.feature_columns),
        },
    )

    model_factories = _build_model_factories(config)
    progress_recorder = _TrainingProgressRecorder(artifact_dir)
    progress_recorder.record(
        stage="setup",
        message=(
            "resumed from checkpoint"
            if resume_artifact_dir is not None
            else "training artifact directory created"
        ),
        artifact_dir=str(artifact_dir),
        config_path=str(config_path),
        progress_log_path=str(progress_recorder.log_path),
        progress_status_path=str(progress_recorder.status_path),
    )
    progress_recorder.record(
        stage="dataset",
        message="dataset manifest captured for offline training",
        source_rows=len(dataset.source_rows),
        labeled_rows=len(dataset.samples),
        unique_timestamps=int(dataset.manifest["unique_timestamps"]),
    )

    # --- pause flag: Ctrl+C sets this, loop checks between folds ---
    pause_requested = False
    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_pause(signum: int, frame: Any) -> None:  # noqa: ARG001
        nonlocal pause_requested
        pause_requested = True
        print(
            "\n[training] pause requested — "
            "will checkpoint after the current fold finishes. "
            "Press Ctrl+C again to force-kill."
        )
        signal.signal(signal.SIGINT, original_sigint)

    signal.signal(signal.SIGINT, _handle_pause)

    console_progress = _ConsoleProgress(
        total_folds=len(folds),
        total_models=len(model_factories),
    )
    # advance the counter for already-completed folds on resume
    console_progress.completed_units = (
        len(completed_fold_indices) * len(model_factories)
    )
    if resume_completed_models:
        console_progress.completed_units += len(resume_completed_models)

    # Initialized before the try block so crash recovery never masks an
    # earlier setup/preload failure with an unbound local.
    _completed_models_this_fold: list[str] = []
    _fold_metrics_this_fold: list[dict[str, Any]] = []
    _fold_predictions_this_fold: list[PredictionRecord] = []
    _current_fold_index = -1
    _preloaded_estimators: dict[tuple[int, str], Any] = {}

    try:
        remaining_folds = [
            fold for fold in folds
            if fold.fold_index not in completed_fold_indices
        ]
        print(
            "[training] evaluating "
            f"{len(remaining_folds)} remaining walk-forward folds "
            f"across {len(model_factories)} models"
        )
        progress_recorder.record(
            stage="evaluation",
            message="starting walk-forward evaluation",
            total_folds=len(folds),
            remaining_folds=len(remaining_folds),
            total_models=len(model_factories),
        )

        # --- pre-load all score-only models once (avoid repeated joblib.load) ---
        if score_only_dir is not None:
            for fold in remaining_folds:
                for model_name in model_factories:
                    if model_name in _REQUIRED_PROMOTION_BASELINES:
                        continue
                    load_path = (
                        score_only_dir / f"fold{fold.fold_index}"
                        / f"{model_name}.joblib"
                    )
                    if load_path.exists():
                        _preloaded_estimators[(fold.fold_index, model_name)] = (
                            joblib.load(load_path)
                        )
            if _preloaded_estimators:
                print(
                    f"[training] pre-loaded {len(_preloaded_estimators)} "
                    f"fitted models from {score_only_dir.name}/",
                    flush=True,
                )

        for fold in remaining_folds:
            if pause_requested:
                _save_fold_checkpoint(
                    artifact_dir,
                    fold_metric_rows=fold_metric_rows,
                    all_prediction_rows=all_prediction_rows,
                    completed_fold_indices=completed_fold_indices,
                )
                progress_recorder.record(
                    stage="paused",
                    message="training paused by user",
                    state="paused",
                    completed_folds=len(completed_fold_indices),
                    total_folds=len(folds),
                    checkpoint_path=str(artifact_dir / "checkpoint.json"),
                )
                print(
                    f"\n[training] paused after {len(completed_fold_indices)}/{len(folds)} folds. "
                    f"Resume with: python -m app.training --config {config_path} "
                    f"--resume {artifact_dir}"
                )
                signal.signal(signal.SIGINT, original_sigint)
                return artifact_dir

            train_samples, test_samples = _partition_samples(dataset.samples, fold)
            train_source_rows, evaluation_source_rows = _partition_source_rows(
                dataset.source_rows,
                fold=fold,
                horizon_candles=config.label_horizon_candles,
                frequency_minutes=dataset.frequency_minutes,
            )
            print(
                f"\n[training] fold "
                f"{fold.fold_index + 1}/{len(folds)}: "
                f"train_rows={len(train_samples)} test_rows={len(test_samples)}"
            )
            progress_recorder.record(
                stage="fold",
                message="starting walk-forward fold",
                fold_index=fold.fold_index + 1,
                total_folds=len(folds),
                train_rows=len(train_samples),
                test_rows=len(test_samples),
            )

            # determine which models to skip on resume
            skip_models: set[str] = set()
            if (
                resume_partial_fold_index is not None
                and fold.fold_index == resume_partial_fold_index
                and resume_completed_models
            ):
                skip_models = set(resume_completed_models)
                print(
                    f"[training] resuming fold {fold.fold_index + 1} — "
                    f"skipping {len(skip_models)} already-scored models: "
                    f"{sorted(skip_models)}"
                )

            # track which models finish for mid-fold checkpoint state
            _current_fold_index = fold.fold_index
            _completed_models_this_fold = list(skip_models)
            _fold_metrics_this_fold = []
            _fold_predictions_this_fold = []

            def _model_checkpoint_callback(
                model_name: str,
                model_metrics: list[dict[str, Any]],
                model_predictions: list[PredictionRecord],
            ) -> None:
                _completed_models_this_fold.append(model_name)
                _fold_metrics_this_fold.extend(model_metrics)
                _fold_predictions_this_fold.extend(model_predictions)
                _save_fold_checkpoint(
                    artifact_dir,
                    fold_metric_rows=fold_metric_rows + _fold_metrics_this_fold,
                    all_prediction_rows=(
                        all_prediction_rows + _fold_predictions_this_fold
                    ),
                    completed_fold_indices=completed_fold_indices,
                    partial_fold_index=fold.fold_index,
                    completed_models_in_partial_fold=list(
                        _completed_models_this_fold
                    ),
                )

            fold_metrics, fold_predictions = _evaluate_fold(
                train_samples=train_samples,
                test_samples=test_samples,
                train_source_rows=train_source_rows,
                evaluation_source_rows=evaluation_source_rows,
                artifact_dir=artifact_dir,
                fold=fold,
                model_factories=model_factories,
                fee_rate=config.round_trip_fee_rate,
                regime_labels_by_row_id=regime_context.labels_by_row_id,
                progress_recorder=progress_recorder,
                total_folds=len(folds),
                console_progress=console_progress,
                skip_models=skip_models,
                model_checkpoint_callback=_model_checkpoint_callback,
                fit_only=fit_only,
                fitted_models_dir=score_only_dir,
                preloaded_estimators=_preloaded_estimators,
            )
            fold_metric_rows.extend(fold_metrics)
            all_prediction_rows.extend(fold_predictions)
            completed_fold_indices.append(fold.fold_index)
            # clear partial-fold resume state after first use
            resume_partial_fold_index = None
            resume_completed_models = None
            progress_recorder.record(
                stage="fold",
                message="completed walk-forward fold",
                fold_index=fold.fold_index + 1,
                total_folds=len(folds),
                prediction_rows=len(fold_predictions),
            )
            # checkpoint after every fold so we can resume
            _save_fold_checkpoint(
                artifact_dir,
                fold_metric_rows=fold_metric_rows,
                all_prediction_rows=all_prediction_rows,
                completed_fold_indices=completed_fold_indices,
            )

        # ---- fit-only: do full-fit, save, and return early ----
        if fit_only:
            print("[training] fit-only: fitting full-dataset models for later scoring")
            _fit_only_full_dataset(
                model_factories=model_factories,
                samples=dataset.samples,
                source_rows=dataset.source_rows,
                artifact_dir=artifact_dir,
                progress_recorder=progress_recorder,
            )
            _write_json(
                artifact_dir / "fitted_models" / "manifest.json",
                {
                    "mode": "fit_only",
                    "config_path": str(config_path),
                    "folds": len(folds),
                    "models": [
                        m for m in model_factories
                        if m not in _REQUIRED_PROMOTION_BASELINES
                    ],
                },
            )
            progress_recorder.record(
                stage="complete",
                message="fit-only completed — fitted models saved",
                state="completed",
                fitted_models_dir=str(artifact_dir / "fitted_models"),
            )
            print(
                f"\n[training] fit-only complete. Fitted models at:\n"
                f"  {artifact_dir / 'fitted_models'}\n"
                f"Score locally with:\n"
                f"  python -m app.training --config <local_config> "
                f"--score-only {artifact_dir / 'fitted_models'}"
            )
            return artifact_dir

        _write_csv(artifact_dir / "fold_metrics.csv", fold_metric_rows)
        _write_csv(
            artifact_dir / "oof_predictions.csv",
            [record.to_csv_row() for record in all_prediction_rows],
        )

        aggregate_summary = _build_aggregate_summary(
            all_prediction_rows=all_prediction_rows,
            fee_rate=config.round_trip_fee_rate,
        )
        winner_name = _select_winner(aggregate_summary)
        print(f"[training] winner by offline selection rule: {winner_name}")
        progress_recorder.record(
            stage="selection",
            message="winner selected by offline rule",
            winner_name=winner_name,
        )
        if score_only_dir is not None:
            print("[training] score-only: loading pre-fitted full-dataset models")
            learned_candidate_artifacts = _load_full_fit_models(
                model_factories=model_factories,
                fitted_models_dir=score_only_dir,
                artifact_dir=artifact_dir,
                configured_feature_columns=dataset.feature_columns,
            )
        else:
            print("[training] fitting full-dataset challenger artifacts")
            progress_recorder.record(
                stage="full_fit",
                message="starting full-dataset challenger fits",
                learned_models=len(
                    [
                        model_name
                        for model_name in model_factories
                        if model_name not in _REQUIRED_PROMOTION_BASELINES
                    ]
                ),
            )
            learned_candidate_artifacts = _fit_learned_models_on_full_dataset(
                model_factories=model_factories,
                samples=dataset.samples,
                source_rows=dataset.source_rows,
                artifact_dir=artifact_dir,
                configured_feature_columns=dataset.feature_columns,
                progress_recorder=progress_recorder,
            )
        winner_candidate_artifact = learned_candidate_artifacts[winner_name]
        _save_model_artifact(
            model_path=artifact_dir / "model.joblib",
            model_name=winner_candidate_artifact.model_name,
            fitted_model=winner_candidate_artifact.fitted_model,
            feature_columns=winner_candidate_artifact.feature_columns,
            expanded_feature_names=winner_candidate_artifact.expanded_feature_names,
            training_model_config=winner_candidate_artifact.training_model_config,
            registry_metadata=winner_candidate_artifact.registry_metadata,
        )
        _write_json(
            artifact_dir / "feature_columns.json",
            {
                "configured_feature_columns": list(winner_candidate_artifact.feature_columns),
                "categorical_feature_columns": list(dataset.categorical_feature_columns),
                "numeric_feature_columns": list(dataset.numeric_feature_columns),
                "expanded_feature_names": winner_candidate_artifact.expanded_feature_names,
            },
        )
        _write_json(
            artifact_dir / "candidate_artifacts.json",
            {
                model_name: {
                    "model_path": str(candidate_artifact.model_path),
                    "feature_columns": list(candidate_artifact.feature_columns),
                    "expanded_feature_names": list(candidate_artifact.expanded_feature_names),
                    "training_config": candidate_artifact.training_model_config,
                    "registry_metadata": candidate_artifact.registry_metadata,
                }
                for model_name, candidate_artifact in sorted(
                    learned_candidate_artifacts.items()
                )
            },
        )

        # --- recent-window scoring for specialist promotion verdicts ---
        recent_aggregate_summary = None
        specialist_verdicts = None
        recent_window_meta = None
        incumbent_model_version: str | None = None
        if config.recent_scoring_window_days is not None:
            recent_preds, recent_window_meta = _filter_recent_predictions(
                all_prediction_rows, config.recent_scoring_window_days,
            )
            if recent_preds:
                recent_aggregate_summary = _build_aggregate_summary(
                    all_prediction_rows=recent_preds,
                    fee_rate=config.round_trip_fee_rate,
                )

                # --- load and score incumbent for comparison ---
                incumbent_predictions: list[PredictionRecord] | None = None
                incumbent_result = _load_incumbent_model()
                if incumbent_result is not None:
                    inc_model, inc_version, inc_name = incumbent_result
                    incumbent_model_version = inc_version
                    print(
                        f"[training] scoring incumbent {inc_name} "
                        f"({inc_version}) on recent window...",
                        end="", flush=True,
                    )
                    inc_start = time.monotonic()
                    incumbent_predictions = (
                        _score_incumbent_on_recent_samples(
                            incumbent_model=inc_model,
                            incumbent_model_name=inc_name,
                            samples=dataset.samples,
                            recent_cutoff_rfc3339=recent_window_meta["cutoff"],
                            fee_rate=config.round_trip_fee_rate,
                            regime_labels_by_row_id=(
                                regime_context.labels_by_row_id
                            ),
                        )
                    )
                    inc_elapsed = time.monotonic() - inc_start
                    print(
                        f" done ({len(incumbent_predictions)} rows, "
                        f"{_format_eta_seconds(inc_elapsed)})",
                    )
                else:
                    print(
                        "[training] no registry incumbent found — "
                        "specialist verdicts will use baseline-only comparison"
                    )

                specialist_verdicts = _build_specialist_verdicts(
                    recent_predictions=recent_preds,
                    model_configs=config.models,
                    incumbent_predictions=incumbent_predictions,
                    incumbent_model_version=incumbent_model_version,
                    max_drawdown_tolerance=config.max_drawdown_tolerance,
                )
                print(
                    f"[training] recent-window scoring: "
                    f"{recent_window_meta['eligible_rows']}/{recent_window_meta['total_oof_rows']} "
                    f"rows in last {config.recent_scoring_window_days} days"
                )
                for sn, sv in specialist_verdicts.items():
                    basis = sv.get("verdict_basis", "?")
                    print(
                        f"[training]   {sn} ({sv['candidate_role']}): "
                        f"verdict={sv['verdict']} basis={basis}"
                    )

        summary = _build_summary_payload(
            config=config,
            dataset_manifest=dataset.manifest,
            aggregate_summary=aggregate_summary,
            regime_context=regime_context,
            winner_name=winner_name,
            model_path=artifact_dir / "model.joblib",
            winner_training_config=winner_candidate_artifact.training_model_config,
            winner_registry_metadata=winner_candidate_artifact.registry_metadata,
            candidate_artifacts=learned_candidate_artifacts,
            recent_aggregate_summary=recent_aggregate_summary,
            recent_window_meta=recent_window_meta,
            specialist_verdicts=specialist_verdicts,
            incumbent_model_version=incumbent_model_version,
        )
        _write_json(artifact_dir / "summary.json", summary)
        _remove_checkpoint(artifact_dir)
        progress_recorder.record(
            stage="complete",
            message="training completed",
            state="completed",
            winner_name=winner_name,
            summary_path=str(artifact_dir / "summary.json"),
        )
    except Exception as error:
        # on crash, save a checkpoint (model-level callback may have already
        # saved mid-fold state, but re-save to be safe with full accumulator)
        _has_progress = completed_fold_indices or _completed_models_this_fold
        if _has_progress:
            _save_fold_checkpoint(
                artifact_dir,
                fold_metric_rows=fold_metric_rows + _fold_metrics_this_fold,
                all_prediction_rows=(
                    all_prediction_rows + _fold_predictions_this_fold
                ),
                completed_fold_indices=completed_fold_indices,
                partial_fold_index=(
                    _current_fold_index
                    if _completed_models_this_fold
                    else None
                ),
                completed_models_in_partial_fold=(
                    list(_completed_models_this_fold)
                    if _completed_models_this_fold
                    else None
                ),
            )
            print(
                f"\n[training] crashed after {len(completed_fold_indices)} full folds"
                + (
                    f" + {len(_completed_models_this_fold)} models in fold "
                    f"{_current_fold_index + 1}"
                    if _completed_models_this_fold
                    else ""
                )
                + f"/{len(folds)} folds. "
                f"Checkpoint saved. Resume with:\n"
                f"  python -m app.training --config {config_path} "
                f"--resume {artifact_dir}",
                file=sys.stderr,
            )
        progress_recorder.record(
            stage="error",
            message="training failed",
            state="failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        signal.signal(signal.SIGINT, original_sigint)
    return artifact_dir


def _validate_split_readiness(dataset: Any, config: TrainingConfig) -> None:
    """Fail early with a clear message when live data is not yet sufficient."""
    actual_unique_timestamps = int(dataset.manifest["unique_timestamps"])
    required_unique_timestamps = minimum_required_unique_timestamps(
        first_train_fraction=config.first_train_fraction,
        test_fraction=config.test_fraction,
        test_folds=config.test_folds,
        purge_gap_candles=config.purge_gap_candles,
    )
    if actual_unique_timestamps >= required_unique_timestamps:
        return
    raise ValueError(
        "Not enough unique eligible timestamps for the configured walk-forward split. "
        f"Required at least {required_unique_timestamps}, found {actual_unique_timestamps}. "
        f"Eligible labeled rows: {dataset.manifest['eligible_rows']}."
    )


def _create_artifact_dir(config: TrainingConfig) -> Path:
    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = Path(config.artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def _build_model_factories(
    config: TrainingConfig,
) -> dict[str, Callable[[], Any]]:
    model_factories: dict[str, Callable[[], Any]] = {
        "persistence_3": _build_persistence_baseline,
        "dummy_most_frequent": build_dummy_classifier,
    }
    for model_name, model_config in config.models.items():
        builder = _AUTHORITATIVE_MODEL_BUILDERS.get(model_name)
        if builder is None:
            raise ValueError(
                "No authoritative training builder exists yet for configured model(s): "
                f"{sorted(config.models)}"
            )
        resolved_model_config = dict(model_config)
        if model_name.startswith(("neuralforecast_", "chronos2_", "timesfm_", "moirai_")):
            resolved_model_config.setdefault(
                "horizon_candles",
                config.label_horizon_candles,
            )
        model_factories[model_name] = (
            lambda builder=builder, model_config=dict(resolved_model_config): builder(model_config)
        )
    return model_factories


def _build_persistence_baseline() -> PersistenceBaseline:
    return PersistenceBaseline()


def _validate_authoritative_model_stack(config: TrainingConfig) -> None:
    configured_models = tuple(sorted(config.models))
    if not configured_models:
        raise ValueError(
            "No active authoritative trainable models are configured. "
            "Legacy sklearn models have been removed from the main training path "
            "and the intended primary stack is not implemented yet."
        )
    legacy_models = [
        model_name
        for model_name in configured_models
        if model_name in LEGACY_ARCHIVED_MODEL_NAMES
    ]
    if legacy_models:
        raise ValueError(
            "Legacy archived sklearn models are no longer allowed in the "
            f"authoritative training path: {legacy_models}"
        )
    unsupported_models = [
        model_name
        for model_name in configured_models
        if model_name not in _AUTHORITATIVE_MODEL_BUILDERS
    ]
    if unsupported_models:
        raise ValueError(
            "No authoritative training builder exists yet for configured model(s): "
            f"{unsupported_models}"
        )


def _partition_samples(
    samples: tuple[DatasetSample, ...],
    fold: WalkForwardFold,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    train_times = set(fold.train_timestamps)
    test_times = set(fold.test_timestamps)
    train_samples = [sample for sample in samples if sample.as_of_time in train_times]
    test_samples = [sample for sample in samples if sample.as_of_time in test_times]
    if not train_samples:
        raise ValueError(f"Fold {fold.fold_index} has no training samples")
    if not test_samples:
        raise ValueError(f"Fold {fold.fold_index} has no test samples")
    return train_samples, test_samples


def _partition_source_rows(
    source_rows: tuple[SourceFeatureRow, ...],
    *,
    fold: WalkForwardFold,
    horizon_candles: int,
    frequency_minutes: int,
) -> tuple[list[SourceFeatureRow], list[SourceFeatureRow]]:
    """Return the ordered source rows needed for sequence-model fold fit and scoring."""
    train_end = max(fold.train_timestamps)
    test_end = max(fold.test_timestamps)
    evaluation_end = test_end + timedelta(minutes=horizon_candles * frequency_minutes)
    train_source_rows = [
        source_row
        for source_row in source_rows
        if source_row.as_of_time <= train_end
    ]
    evaluation_source_rows = [
        source_row
        for source_row in source_rows
        if source_row.as_of_time <= evaluation_end
    ]
    if not train_source_rows:
        raise ValueError(f"Fold {fold.fold_index} has no sequence training source rows")
    if not evaluation_source_rows:
        raise ValueError(f"Fold {fold.fold_index} has no sequence evaluation source rows")
    return train_source_rows, evaluation_source_rows


# pylint: disable=too-many-arguments
def _evaluate_fold(
    *,
    train_samples: list[DatasetSample],
    test_samples: list[DatasetSample],
    train_source_rows: list[SourceFeatureRow],
    evaluation_source_rows: list[SourceFeatureRow],
    artifact_dir: Path,
    fold: WalkForwardFold,
    model_factories: dict[str, Callable[[], Any]],
    fee_rate: float,
    regime_labels_by_row_id: dict[str, str],
    progress_recorder: _TrainingProgressRecorder | None = None,
    total_folds: int | None = None,
    console_progress: _ConsoleProgress | None = None,
    skip_models: set[str] | None = None,
    model_checkpoint_callback: Callable[
        [str, list[dict[str, Any]], list[PredictionRecord]], None
    ] | None = None,
    fit_only: bool = False,
    fitted_models_dir: Path | None = None,
    preloaded_estimators: dict[tuple[int, str], Any] | None = None,
) -> tuple[list[dict[str, Any]], list[PredictionRecord]]:
    # The explicit inputs keep the fold evaluation contract inspectable for M3/M7.
    fold_metrics: list[dict[str, Any]] = []
    prediction_rows: list[PredictionRecord] = []
    _skip = skip_models or set()
    for model_name, factory in model_factories.items():
        if model_name in _skip:
            if console_progress is not None:
                console_progress.tick(
                    fold_index=fold.fold_index, model_name=f"{model_name} (cached)",
                )
            continue
        if console_progress is not None:
            console_progress.model_start(
                fold_index=fold.fold_index, model_name=model_name,
            )
        if progress_recorder is not None:
            progress_recorder.record(
                stage="fold_model",
                message="starting model evaluation",
                fold_index=fold.fold_index + 1,
                total_folds=total_folds,
                model_name=model_name,
                train_rows=len(train_samples),
                test_rows=len(test_samples),
            )

        def _progress_callback(payload: dict[str, Any]) -> None:
            if progress_recorder is None:
                return
            progress_recorder.record_sequence_event(
                fold_index=fold.fold_index,
                total_folds=total_folds,
                model_name=model_name,
                payload=payload,
            )

        try:
            # resolve fit-only / score-only paths for this model
            _fit_save = (
                artifact_dir / "fitted_models" / f"fold{fold.fold_index}"
                / f"{model_name}.joblib"
            ) if fit_only and model_name not in _REQUIRED_PROMOTION_BASELINES else None
            _score_load = (
                fitted_models_dir / f"fold{fold.fold_index}" / f"{model_name}.joblib"
                if fitted_models_dir is not None
                and model_name not in _REQUIRED_PROMOTION_BASELINES
                else None
            )

            # use pre-loaded estimator when available (avoids repeated joblib.load)
            _preloaded = (
                preloaded_estimators.get((fold.fold_index, model_name))
                if preloaded_estimators else None
            )

            predicted_labels, probabilities = _predict_for_model(
                model_name=model_name,
                factory=factory,
                train_samples=train_samples,
                test_samples=test_samples,
                train_source_rows=train_source_rows,
                evaluation_source_rows=evaluation_source_rows,
                sequence_export_root=(
                    artifact_dir
                    / "specialist_local_files"
                    / f"fold_{fold.fold_index + 1:02d}"
                    / model_name
                ),
                progress_callback=(
                    _progress_callback if progress_recorder is not None else None
                ),
                scoring_checkpoint_path=(
                    artifact_dir
                    / "scoring_cache"
                    / f"fold{fold.fold_index}_{model_name}.json"
                ),
                fit_only_save_path=_fit_save,
                score_only_load_path=_score_load,
                preloaded_estimator=_preloaded,
            )
        except Exception as error:
            if progress_recorder is not None:
                progress_recorder.record(
                    stage="fold_model_error",
                    message="model evaluation failed",
                    fold_index=fold.fold_index + 1,
                    total_folds=total_folds,
                    model_name=model_name,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            raise
        model_predictions = _build_prediction_records(
            model_name=model_name,
            fold_index=fold.fold_index,
            test_samples=test_samples,
            predicted_labels=predicted_labels,
            probabilities=probabilities,
            fee_rate=fee_rate,
            regime_labels_by_row_id=regime_labels_by_row_id,
        )
        metrics = _compute_metrics(model_predictions)
        model_metric_row = {
            "model_name": model_name,
            "fold_index": fold.fold_index,
            "train_rows": len(train_samples),
            "test_rows": len(test_samples),
            **metrics["metrics"],
        }
        if progress_recorder is not None:
            progress_recorder.record(
                stage="fold_model",
                message="completed model evaluation",
                fold_index=fold.fold_index + 1,
                total_folds=total_folds,
                model_name=model_name,
                prediction_rows=len(model_predictions),
                directional_accuracy=metrics["metrics"]["directional_accuracy"],
                mean_long_only_net_value_proxy=(
                    metrics["metrics"]["mean_long_only_net_value_proxy"]
                ),
            )
        if console_progress is not None:
            console_progress.tick(
                fold_index=fold.fold_index, model_name=model_name,
            )
        fold_metrics.append(model_metric_row)
        prediction_rows.extend(model_predictions)
        if model_checkpoint_callback is not None:
            model_checkpoint_callback(
                model_name, [model_metric_row], model_predictions,
            )
    return fold_metrics, prediction_rows


def _predict_for_model(
    *,
    model_name: str,
    factory: Callable[[], Any],
    train_samples: list[DatasetSample],
    test_samples: list[DatasetSample],
    train_source_rows: list[SourceFeatureRow],
    evaluation_source_rows: list[SourceFeatureRow],
    sequence_export_root: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    scoring_checkpoint_path: Path | None = None,
    fit_only_save_path: Path | None = None,
    score_only_load_path: Path | None = None,
    preloaded_estimator: Any | None = None,
) -> tuple[list[int], list[float]]:
    if model_name == "persistence_3":
        baseline = factory().fit(train_samples)
        if fit_only_save_path is not None:
            return [0] * len(test_samples), [0.5] * len(test_samples)
        labels = baseline.predict(test_samples)
        probabilities = [row[1] for row in baseline.predict_proba(test_samples)]
        return labels, probabilities

    train_labels = [sample.label for sample in train_samples]
    if model_name not in _REQUIRED_PROMOTION_BASELINES and len(
        set(train_labels)
    ) < 2:
        constant_label = int(train_labels[0])
        return [constant_label] * len(test_samples), [float(constant_label)] * len(test_samples)

    # --- score-only: use pre-loaded or load from disk ---
    if preloaded_estimator is not None:
        estimator = preloaded_estimator
        print(
            f"\n[training]   {model_name}: using pre-loaded fitted model",
            flush=True,
        )
    elif score_only_load_path is not None:
        estimator = joblib.load(score_only_load_path)
        print(
            f"\n[training]   {model_name}: loaded pre-fitted model from "
            f"{score_only_load_path.name}",
            flush=True,
        )
    else:
        estimator = factory()

    if is_pretrained_forecaster_model(estimator):
        # --- fit phase (skip when score-only) ---
        if score_only_load_path is None:
            fit_kwargs: dict[str, Any] = {
                "source_rows": train_source_rows,
                "dataset_export_root": sequence_export_root,
            }
            if progress_callback is not None:
                fit_kwargs["progress_callback"] = progress_callback
            print(
                f"\n[training]   {model_name}: fitting on "
                f"{len(train_samples)} samples...",
                end="", flush=True,
            )
            fit_start = time.monotonic()
            estimator.fit_samples(train_samples, **fit_kwargs)
            fit_elapsed = time.monotonic() - fit_start
            print(
                f" done ({_format_eta_seconds(fit_elapsed)})",
                flush=True,
            )

        # --- fit-only: save and return dummy results ---
        if fit_only_save_path is not None:
            fit_only_save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(estimator, fit_only_save_path)
            print(
                f"[training]   {model_name}: saved fitted model → "
                f"{fit_only_save_path.name}",
                flush=True,
            )
            return [0] * len(test_samples), [0.5] * len(test_samples)

        # --- score phase ---
        predict_kwargs: dict[str, Any] = {
            "source_rows": evaluation_source_rows,
        }
        if scoring_checkpoint_path is not None:
            predict_kwargs["scoring_checkpoint_path"] = scoring_checkpoint_path
        _scoring_line_prefix = (
            f"[training]   {model_name}: scoring "
            f"{len(test_samples)} test samples"
        )

        def _scoring_progress(payload: dict[str, Any]) -> None:
            event = payload.get("event", "")
            if event == "sequence_scoring_start":
                print(
                    f"\r{_scoring_line_prefix}  "
                    f"[  0%]",
                    end="", flush=True,
                )
            elif event == "sequence_scoring_progress":
                pct = int(payload.get("progress", 0) * 100)
                done_rows = int(payload.get("completed_rows", 0))
                total_rows = int(payload.get("row_count", 0))
                eta = _format_eta_seconds(payload.get("eta_seconds", 0))
                elapsed = _format_eta_seconds(
                    payload.get("elapsed_seconds", 0),
                )
                done_k = f"{done_rows // 1000}K" if done_rows >= 1000 else str(done_rows)
                total_k = f"{total_rows // 1000}K" if total_rows >= 1000 else str(total_rows)
                print(
                    f"\r{_scoring_line_prefix}  "
                    f"[{pct:3d}%  {done_k}/{total_k}  "
                    f"elapsed {elapsed}  ETA {eta}]",
                    end="", flush=True,
                )
            if progress_callback is not None:
                progress_callback(payload)

        predict_kwargs["progress_callback"] = _scoring_progress
        print(f"\r{_scoring_line_prefix}...", end="", flush=True)
        score_start = time.monotonic()
        predicted_probabilities = estimator.predict_proba_samples(test_samples, **predict_kwargs)
        score_elapsed = time.monotonic() - score_start
        print(
            f"\r{_scoring_line_prefix}  "
            f"done ({_format_eta_seconds(score_elapsed)})"
            + " " * 30,
            flush=True,
        )
        probabilities = [float(probability[1]) for probability in predicted_probabilities]
        predicted_labels = [1 if probability >= 0.5 else 0 for probability in probabilities]
        return predicted_labels, probabilities

    # --- tabular model path ---
    if score_only_load_path is None:
        print(
            f"\n[training]   {model_name}: fitting on "
            f"{len(train_samples)} samples...",
            end="", flush=True,
        )
        fit_start = time.monotonic()
        estimator.fit([sample.features for sample in train_samples], train_labels)
        fit_elapsed = time.monotonic() - fit_start
        print(f" done ({_format_eta_seconds(fit_elapsed)})", flush=True)

    if fit_only_save_path is not None:
        fit_only_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, fit_only_save_path)
        print(
            f"[training]   {model_name}: saved fitted model → "
            f"{fit_only_save_path.name}",
            flush=True,
        )
        return [0] * len(test_samples), [0.5] * len(test_samples)

    test_features = [sample.features for sample in test_samples]
    predicted_labels = [int(label) for label in estimator.predict(test_features)]
    predicted_probabilities = estimator.predict_proba(test_features)
    probabilities = [float(probability[1]) for probability in predicted_probabilities]
    return predicted_labels, probabilities

# pylint: disable=too-many-arguments
def _build_prediction_records(
    *,
    model_name: str,
    fold_index: int,
    test_samples: list[DatasetSample],
    predicted_labels: list[int],
    probabilities: list[float],
    fee_rate: float,
    regime_labels_by_row_id: dict[str, str],
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for sample, label, prob_up in zip(test_samples, predicted_labels, probabilities, strict=True):
        long_trade_taken = 1 if label == 1 else 0
        long_only_gross_value_proxy = sample.future_return_3 if long_trade_taken else 0.0
        long_only_net_value_proxy = (
            long_only_gross_value_proxy - fee_rate if long_trade_taken else 0.0
        )
        records.append(
            PredictionRecord(
                model_name=model_name,
                fold_index=fold_index,
                row_id=sample.row_id,
                symbol=sample.symbol,
                interval_begin=to_rfc3339(sample.interval_begin),
                as_of_time=to_rfc3339(sample.as_of_time),
                y_true=sample.label,
                y_pred=label,
                prob_up=prob_up,
                confidence=max(prob_up, 1.0 - prob_up),
                regime_label=regime_labels_by_row_id[sample.row_id],
                long_trade_taken=long_trade_taken,
                future_return_3=sample.future_return_3,
                long_only_gross_value_proxy=long_only_gross_value_proxy,
                long_only_net_value_proxy=long_only_net_value_proxy,
            )
        )
    return records


def _compute_metrics(predictions: list[PredictionRecord]) -> dict[str, Any]:
    y_true = [prediction.y_true for prediction in predictions]
    y_pred = [prediction.y_pred for prediction in predictions]
    prob_up = [prediction.prob_up for prediction in predictions]
    directional_accuracy = accuracy_score(y_true, y_pred)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()
    trade_count = sum(prediction.long_trade_taken for prediction in predictions)
    trade_rate = trade_count / len(predictions)
    mean_long_only_gross_value_proxy = (
        sum(prediction.long_only_gross_value_proxy for prediction in predictions)
        / len(predictions)
    )
    mean_long_only_net_value_proxy = (
        sum(prediction.long_only_net_value_proxy for prediction in predictions)
        / len(predictions)
    )
    return {
        "metrics": {
            "directional_accuracy": directional_accuracy,
            "precision_class_0": float(precision[0]),
            "precision_class_1": float(precision[1]),
            "recall_class_0": float(recall[0]),
            "recall_class_1": float(recall[1]),
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
            "brier_score": brier_score_loss(y_true, prob_up),
            "trade_count": trade_count,
            "trade_rate": trade_rate,
            "mean_long_only_gross_value_proxy": mean_long_only_gross_value_proxy,
            "mean_long_only_net_value_proxy": mean_long_only_net_value_proxy,
        },
        "confidence_analysis": _confidence_analysis(predictions),
    }


def _confidence_analysis(predictions: list[PredictionRecord]) -> list[dict[str, Any]]:
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    analysis: list[dict[str, Any]] = []
    for lower_bound, upper_bound in bins:
        bin_rows = [
            prediction
            for prediction in predictions
            if lower_bound <= prediction.confidence < upper_bound
        ]
        if not bin_rows:
            continue
        accuracy = sum(
            int(prediction.y_true == prediction.y_pred) for prediction in bin_rows
        ) / len(bin_rows)
        analysis.append(
            {
                "bin": f"{lower_bound:.2f}-{min(upper_bound, 1.0):.2f}",
                "count": len(bin_rows),
                "accuracy": accuracy,
                "mean_confidence": sum(row.confidence for row in bin_rows) / len(bin_rows),
                "mean_prob_up": sum(row.prob_up for row in bin_rows) / len(bin_rows),
            }
        )
    return analysis


def _build_aggregate_summary(
    *,
    all_prediction_rows: list[PredictionRecord],
    fee_rate: float,
) -> dict[str, dict[str, Any]]:
    del fee_rate
    by_model: dict[str, list[PredictionRecord]] = {}
    for prediction in all_prediction_rows:
        by_model.setdefault(prediction.model_name, []).append(prediction)

    aggregate_summary: dict[str, dict[str, Any]] = {}
    for model_name, rows in by_model.items():
        metrics = _compute_metrics(rows)
        aggregate_summary[model_name] = {
            "prediction_count": len(rows),
            **metrics["metrics"],
            "confidence_analysis": metrics["confidence_analysis"],
            "economics_by_regime": _build_regime_economics(rows),
        }
    return aggregate_summary


def _select_winner(aggregate_summary: dict[str, dict[str, Any]]) -> str:
    learned_models = {
        name: metrics
        for name, metrics in aggregate_summary.items()
        if name not in _REQUIRED_PROMOTION_BASELINES
    }
    if not learned_models:
        raise ValueError("No learned models were available for winner selection")
    return sorted(
        learned_models.items(),
        key=lambda item: (
            -item[1]["mean_long_only_net_value_proxy"],
            -item[1]["directional_accuracy"],
            item[1]["brier_score"],
        ),
    )[0][0]


def _fit_only_full_dataset(
    *,
    model_factories: dict[str, Callable[[], Any]],
    samples: tuple[DatasetSample, ...],
    source_rows: tuple[SourceFeatureRow, ...],
    artifact_dir: Path,
    progress_recorder: _TrainingProgressRecorder | None = None,
) -> None:
    """Fit every learned model on the full dataset and save to fitted_models/full_fit/."""
    train_labels = [sample.label for sample in samples]
    if len(set(train_labels)) < 2:
        raise ValueError("Full-fit requires at least two distinct labels")

    full_fit_dir = artifact_dir / "fitted_models" / "full_fit"
    full_fit_dir.mkdir(parents=True, exist_ok=True)

    for model_name, factory in model_factories.items():
        if model_name in _REQUIRED_PROMOTION_BASELINES:
            continue
        print(f"[training]   full-fit {model_name}...", end="", flush=True)
        fit_start = time.monotonic()
        fitted_model = factory()
        if is_pretrained_forecaster_model(fitted_model):
            fit_kwargs: dict[str, Any] = {"source_rows": list(source_rows)}
            if progress_recorder is not None:
                def _cb(
                    payload: dict[str, Any],
                    *,
                    _mn: str = model_name,
                ) -> None:
                    progress_recorder.record_sequence_event(
                        model_name=_mn, payload=payload,
                    )
                fit_kwargs["progress_callback"] = _cb
            fitted_model.fit_samples(list(samples), **fit_kwargs)
        else:
            fitted_model.fit(
                [sample.features for sample in samples], train_labels,
            )
        save_path = full_fit_dir / f"{model_name}.joblib"
        joblib.dump(fitted_model, save_path)
        elapsed = time.monotonic() - fit_start
        print(f" done ({_format_eta_seconds(elapsed)}) → {save_path.name}", flush=True)


def _load_full_fit_models(
    *,
    model_factories: dict[str, Callable[[], Any]],
    fitted_models_dir: Path,
    artifact_dir: Path,
    configured_feature_columns: tuple[str, ...],
) -> dict[str, SavedCandidateArtifact]:
    """Load pre-fitted full-dataset models from a fitted_models directory."""
    saved_candidates: dict[str, SavedCandidateArtifact] = {}
    candidate_root = artifact_dir / "candidate_artifacts"
    candidate_root.mkdir(parents=True, exist_ok=True)

    for model_name in model_factories:
        if model_name in _REQUIRED_PROMOTION_BASELINES:
            continue
        load_path = fitted_models_dir / "full_fit" / f"{model_name}.joblib"
        if not load_path.exists():
            raise ValueError(
                f"Pre-fitted model not found: {load_path}. "
                f"Run --fit-only first to generate fitted models."
            )
        fitted_model = joblib.load(load_path)
        print(f"[training]   loaded full-fit {model_name} from {load_path.name}")

        validated_pretrained_contract = None
        if is_pretrained_forecaster_model(fitted_model):
            validated_pretrained_contract = validate_pretrained_forecaster_contract(
                fitted_model,
            )

        if validated_pretrained_contract is None:
            feature_columns = _resolve_model_feature_columns(
                fitted_model=fitted_model,
                configured_feature_columns=configured_feature_columns,
            )
            expanded_feature_names = _resolve_expanded_feature_names(
                fitted_model=fitted_model,
            )
            training_model_config = _extract_training_model_config(
                fitted_model=fitted_model,
            )
            registry_metadata = _extract_registry_metadata(
                fitted_model=fitted_model,
            )
        else:
            feature_columns = validated_pretrained_contract.feature_columns
            expanded_feature_names = list(
                validated_pretrained_contract.expanded_feature_names,
            )
            training_model_config = validated_pretrained_contract.training_config
            registry_metadata = validated_pretrained_contract.registry_metadata

        model_dir = candidate_root / model_name
        model_dir.mkdir(parents=True, exist_ok=False)
        model_path = model_dir / "model.joblib"
        _save_model_artifact(
            model_path=model_path,
            model_name=model_name,
            fitted_model=fitted_model,
            feature_columns=feature_columns,
            expanded_feature_names=expanded_feature_names,
            training_model_config=training_model_config,
            registry_metadata=registry_metadata,
        )
        saved_candidates[model_name] = SavedCandidateArtifact(
            model_name=model_name,
            model_path=model_path,
            fitted_model=fitted_model,
            feature_columns=feature_columns,
            expanded_feature_names=expanded_feature_names,
            training_model_config=training_model_config,
            registry_metadata=registry_metadata,
        )
    return saved_candidates


def _fit_learned_models_on_full_dataset(
    *,
    model_factories: dict[str, Callable[[], Any]],
    samples: tuple[DatasetSample, ...],
    source_rows: tuple[SourceFeatureRow, ...],
    artifact_dir: Path,
    configured_feature_columns: tuple[str, ...],
    progress_recorder: _TrainingProgressRecorder | None = None,
) -> dict[str, SavedCandidateArtifact]:
    """Full-fit and persist every learned candidate model for later challenger use."""
    train_labels = [sample.label for sample in samples]
    if len(set(train_labels)) < 2:
        raise ValueError("Learned models cannot be fit because the full dataset has only one class")

    saved_candidates: dict[str, SavedCandidateArtifact] = {}
    candidate_root = artifact_dir / "candidate_artifacts"
    candidate_root.mkdir(parents=True, exist_ok=True)

    for model_name, factory in model_factories.items():
        if model_name in _REQUIRED_PROMOTION_BASELINES:
            continue
        if progress_recorder is not None:
            progress_recorder.record(
                stage="full_fit_model",
                message="starting full-dataset challenger fit",
                model_name=model_name,
                sample_count=len(samples),
            )
        fitted_model = factory()
        validated_pretrained_contract = None
        if is_pretrained_forecaster_model(fitted_model):
            fit_kwargs: dict[str, Any] = {
                "source_rows": list(source_rows),
                "dataset_export_root": (
                    artifact_dir / "specialist_local_files" / "full_fit" / model_name
                ),
            }
            if progress_recorder is not None:
                def _progress_callback(
                    payload: dict[str, Any],
                    *,
                    _model_name: str = model_name,
                ) -> None:
                    progress_recorder.record_sequence_event(
                        model_name=_model_name,
                        payload=payload,
                    )

                fit_kwargs["progress_callback"] = _progress_callback
            fitted_model.fit_samples(list(samples), **fit_kwargs)
            validated_pretrained_contract = validate_pretrained_forecaster_contract(
                fitted_model,
            )
        else:
            fitted_model.fit([sample.features for sample in samples], train_labels)
        if validated_pretrained_contract is None:
            feature_columns = _resolve_model_feature_columns(
                fitted_model=fitted_model,
                configured_feature_columns=configured_feature_columns,
            )
            expanded_feature_names = _resolve_expanded_feature_names(
                fitted_model=fitted_model,
            )
            training_model_config = _extract_training_model_config(
                fitted_model=fitted_model,
            )
            registry_metadata = _extract_registry_metadata(
                fitted_model=fitted_model,
            )
        else:
            feature_columns = validated_pretrained_contract.feature_columns
            expanded_feature_names = list(
                validated_pretrained_contract.expanded_feature_names
            )
            training_model_config = validated_pretrained_contract.training_config
            registry_metadata = validated_pretrained_contract.registry_metadata
        model_dir = candidate_root / model_name
        model_dir.mkdir(parents=True, exist_ok=False)
        model_path = model_dir / "model.joblib"
        _save_model_artifact(
            model_path=model_path,
            model_name=model_name,
            fitted_model=fitted_model,
            feature_columns=feature_columns,
            expanded_feature_names=expanded_feature_names,
            training_model_config=training_model_config,
            registry_metadata=registry_metadata,
        )
        saved_candidates[model_name] = SavedCandidateArtifact(
            model_name=model_name,
            model_path=model_path,
            fitted_model=fitted_model,
            feature_columns=feature_columns,
            expanded_feature_names=expanded_feature_names,
            training_model_config=training_model_config,
            registry_metadata=registry_metadata,
        )
        if progress_recorder is not None:
            progress_recorder.record(
                stage="full_fit_model",
                message="completed full-dataset challenger fit",
                model_name=model_name,
                model_path=str(model_path),
            )
    return saved_candidates


def _resolve_model_feature_columns(
    *,
    fitted_model: Any,
    configured_feature_columns: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve the stable artifact feature schema across model families."""
    if hasattr(fitted_model, "get_feature_columns"):
        feature_columns = fitted_model.get_feature_columns()
        return tuple(str(column) for column in feature_columns)
    return tuple(str(column) for column in configured_feature_columns)


def _resolve_expanded_feature_names(
    *,
    fitted_model: Any,
) -> list[str]:
    """Resolve stable expanded feature names across model families."""
    if hasattr(fitted_model, "get_expanded_feature_names"):
        names = fitted_model.get_expanded_feature_names()
        return [str(name) for name in names]
    vectorizer = fitted_model.named_steps["vectorizer"]
    return [str(name) for name in vectorizer.get_feature_names_out()]


def _save_model_artifact(
    *,
    model_path: Path,
    model_name: str,
    fitted_model: Any,
    feature_columns: tuple[str, ...],
    expanded_feature_names: list[str],
    training_model_config: dict[str, Any] | None,
    registry_metadata: dict[str, Any] | None,
) -> None:
    payload = {
        "model_name": model_name,
        "trained_at": to_rfc3339(utc_now()),
        "feature_columns": list(feature_columns),
        "expanded_feature_names": expanded_feature_names,
        "training_model_config": training_model_config,
        "registry_metadata": registry_metadata,
        "model": fitted_model,
    }
    joblib.dump(payload, model_path)
    reloaded = joblib.load(model_path)
    if reloaded["model_name"] != model_name:
        raise ValueError("Saved model artifact failed reload validation")


def _build_acceptance_block(
    *,
    winner_metrics: dict[str, Any],
    learned_models_positive_after_costs: list[str],
    learned_models_beating_persistence: list[str],
    learned_models_beating_dummy: list[str],
    learned_models_beating_all_baselines: list[str],
    full_history_meets_acceptance: bool,
    specialist_verdicts: dict[str, dict[str, Any]] | None,
    recent_window_meta: dict[str, Any] | None,
    incumbent_model_version: str | None = None,
) -> dict[str, Any]:
    """Build the acceptance block — recent-window specialist verdicts when available."""
    full_history = {
        "scope": "full_history",
        "note": "Diagnostic only — not used for promotion decisions"
        if specialist_verdicts
        else "Gating (no recent scoring window configured)",
        "winner_after_cost_positive": winner_metrics["mean_long_only_net_value_proxy"]
        > 0.0,
        "learned_models_positive_after_costs": learned_models_positive_after_costs,
        "learned_models_beating_persistence_after_costs": (
            learned_models_beating_persistence
        ),
        "learned_models_beating_dummy_after_costs": learned_models_beating_dummy,
        "learned_models_beating_all_baselines_after_costs": (
            learned_models_beating_all_baselines
        ),
        "meets_acceptance_target": full_history_meets_acceptance,
    }
    if specialist_verdicts is None:
        # No recent window — fall back to the full-history acceptance
        full_history.pop("note")
        return full_history

    any_accepted = any(
        v.get("verdict") == "accepted" for v in specialist_verdicts.values()
    )
    all_conclusive = all(
        v.get("verdict") != "inconclusive" for v in specialist_verdicts.values()
    )
    # Determine verdict basis from the verdicts themselves
    verdict_bases = {
        v.get("verdict_basis", "baseline_only")
        for v in specialist_verdicts.values()
        if v.get("verdict") != "inconclusive"
    }
    verdict_basis = (
        "incumbent_comparison" if "incumbent_comparison" in verdict_bases
        else "baseline_only"
    )
    block: dict[str, Any] = {
        "scope": "recent_window",
        "verdict_basis": verdict_basis,
        "window_days": recent_window_meta.get("window_days") if recent_window_meta else None,
        "meets_acceptance_target": any_accepted,
        "all_verdicts_conclusive": all_conclusive,
        "specialist_verdicts_summary": {
            model_name: v.get("verdict", "unknown")
            for model_name, v in specialist_verdicts.items()
        },
        "diagnostic_full_history": full_history,
    }
    if incumbent_model_version is not None:
        block["incumbent_model_version"] = incumbent_model_version
    return block


def _build_summary_payload(
    *,
    config: TrainingConfig,
    dataset_manifest: dict[str, Any],
    aggregate_summary: dict[str, dict[str, Any]],
    regime_context: TrainingRegimeContext,
    winner_name: str,
    model_path: Path,
    winner_training_config: dict[str, Any] | None = None,
    winner_registry_metadata: dict[str, Any] | None = None,
    candidate_artifacts: dict[str, SavedCandidateArtifact] | None = None,
    recent_aggregate_summary: dict[str, dict[str, Any]] | None = None,
    recent_window_meta: dict[str, Any] | None = None,
    specialist_verdicts: dict[str, dict[str, Any]] | None = None,
    incumbent_model_version: str | None = None,
) -> dict[str, Any]:
    winner_metrics = aggregate_summary[winner_name]
    learned_model_names = tuple(
        model_name
        for model_name in aggregate_summary
        if model_name not in _REQUIRED_PROMOTION_BASELINES
    )
    learned_models_positive_after_costs = [
        model_name
        for model_name in learned_model_names
        if aggregate_summary[model_name]["mean_long_only_net_value_proxy"] > 0.0
    ]
    learned_models_beating_persistence = _learned_models_beating_after_costs(
        aggregate_summary,
        baseline_name="persistence_3",
        learned_model_names=learned_model_names,
    )
    learned_models_beating_dummy = _learned_models_beating_after_costs(
        aggregate_summary,
        baseline_name="dummy_most_frequent",
        learned_model_names=learned_model_names,
    )
    learned_models_beating_all_baselines = [
        model_name
        for model_name in learned_model_names
        if all(
            aggregate_summary[model_name]["mean_long_only_net_value_proxy"]
            > aggregate_summary[baseline_name]["mean_long_only_net_value_proxy"]
            for baseline_name in _REQUIRED_PROMOTION_BASELINES
        )
    ]
    meets_acceptance_target = (
        winner_metrics["mean_long_only_net_value_proxy"] > 0.0
        and winner_name in learned_models_beating_all_baselines
    )
    return {
        "generated_at": to_rfc3339(utc_now()),
        "source_table": config.source_table,
        "dataset_manifest": dataset_manifest,
        "economics_contract": {
            "name": "LONG_ONLY_AFTER_COST_PROXY",
            "description": (
                "Predicted UP enters one long for the 3-candle horizon; "
                "predicted DOWN stays flat; the configured round-trip fee is "
                "only charged when a long trade is taken."
            ),
            "primary_metric": "mean_long_only_net_value_proxy",
            "fee_rate": config.round_trip_fee_rate,
        },
        "regime_economics": {
            "source": "M8_PERCENTILE_RULES_REFIT_ON_TRAINING_SOURCE",
            "config_path": regime_context.config_path,
            "high_vol_percentile": regime_context.high_vol_percentile,
            "trend_abs_momentum_percentile": (
                regime_context.trend_abs_momentum_percentile
            ),
            "thresholds_by_symbol": {
                symbol: threshold.to_dict()
                for symbol, threshold in sorted(regime_context.thresholds_by_symbol.items())
            },
        },
        "models": aggregate_summary,
        "official_baseline": "persistence_3",
        "promotion_baselines": list(_REQUIRED_PROMOTION_BASELINES),
        "winner": {
            "model_name": winner_name,
            "model_path": str(model_path),
            "training_config": winner_training_config,
            "metadata": winner_registry_metadata,
            "selection_rule": {
                "primary": "mean_long_only_net_value_proxy",
                "tie_break_1": "directional_accuracy",
                "tie_break_2": "lower_brier_score",
            },
        },
        "candidate_artifacts": (
            {}
            if candidate_artifacts is None
            else {
                model_name: {
                    "model_path": str(candidate_artifact.model_path),
                    "training_config": candidate_artifact.training_model_config,
                    "metadata": candidate_artifact.registry_metadata,
                }
                for model_name, candidate_artifact in sorted(candidate_artifacts.items())
            }
        ),
        "acceptance": _build_acceptance_block(
            winner_metrics=winner_metrics,
            learned_models_positive_after_costs=learned_models_positive_after_costs,
            learned_models_beating_persistence=learned_models_beating_persistence,
            learned_models_beating_dummy=learned_models_beating_dummy,
            learned_models_beating_all_baselines=learned_models_beating_all_baselines,
            full_history_meets_acceptance=meets_acceptance_target,
            specialist_verdicts=specialist_verdicts,
            recent_window_meta=recent_window_meta,
            incumbent_model_version=incumbent_model_version,
        ),
        **(
            {"recent_scoring_window": {
                **recent_window_meta,
                "models": recent_aggregate_summary,
            }}
            if recent_aggregate_summary is not None and recent_window_meta is not None
            else {}
        ),
        **(
            {"specialist_verdicts": specialist_verdicts}
            if specialist_verdicts is not None
            else {}
        ),
    }


def _extract_training_model_config(
    *,
    fitted_model: Any,
) -> dict[str, Any] | None:
    """Return stable winner training config metadata when the model exposes it."""
    if not hasattr(fitted_model, "get_training_config"):
        return None
    training_config = fitted_model.get_training_config()
    if not isinstance(training_config, dict):
        raise ValueError(
            "Authoritative model artifacts must expose a dictionary "
            "training config for auditability",
        )
    return dict(training_config)


def _extract_registry_metadata(
    *,
    fitted_model: Any,
) -> dict[str, Any] | None:
    """Return optional registry discovery metadata when the model exposes it."""
    if not hasattr(fitted_model, "get_registry_metadata"):
        return None
    metadata = fitted_model.get_registry_metadata()
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError(
            "Authoritative model artifacts must expose dictionary registry metadata",
        )
    return dict(metadata)


def _format_log_metadata(metadata: dict[str, Any]) -> str:
    """Render compact metadata for one human-readable training log line."""
    rendered_items: list[str] = []
    for key, value in metadata.items():
        if value is None:
            continue
        safe_value = make_json_safe(value)
        if isinstance(safe_value, (dict, list, tuple)):
            rendered_value = json.dumps(safe_value, sort_keys=True)
        else:
            rendered_value = str(safe_value)
        rendered_items.append(f"{key}={rendered_value}")
    if not rendered_items:
        return ""
    return " | " + ", ".join(rendered_items)


def _render_progress_bar(progress: float, width: int = 20) -> str:
    """Render a fixed-width textual progress bar for artifact-side logs."""
    bounded_progress = max(0.0, min(1.0, float(progress)))
    filled_width = min(width, max(0, int(round(bounded_progress * width))))
    return (
        "[" + ("#" * filled_width) + ("-" * (width - filled_width)) + "] "
        f"{bounded_progress * 100.0:.1f}%"
    )


def _format_eta_seconds(value: Any) -> str:
    """Render seconds as mm:ss or hh:mm:ss for log readability."""
    try:
        total_seconds = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(total_seconds) or total_seconds < 0.0:
        return "n/a"
    rounded_seconds = int(round(total_seconds))
    minutes, seconds = divmod(rounded_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _build_training_regime_context(
    samples: tuple[DatasetSample, ...],
    config: TrainingConfig,
) -> TrainingRegimeContext:
    regime_config = load_regime_config(_REGIME_CONFIG_PATH)
    missing_symbols = sorted(set(config.symbols) - set(regime_config.symbols))
    if missing_symbols:
        raise ValueError(
            "M8 regime config is missing training symbols required for regime-sliced "
            f"economics: {missing_symbols}"
        )
    regime_rows = [_sample_to_regime_row(sample) for sample in samples]
    thresholds_by_symbol = _fit_training_regime_thresholds(
        regime_rows,
        symbols=config.symbols,
        high_vol_percentile=regime_config.thresholds.high_vol_percentile,
        trend_abs_momentum_percentile=(
            regime_config.thresholds.trend_abs_momentum_percentile
        ),
    )
    labels_by_row_id = {
        sample.row_id: classify_row(
            _sample_to_regime_row(sample),
            thresholds_by_symbol,
        )
        for sample in samples
    }
    return TrainingRegimeContext(
        config_path=str(_REGIME_CONFIG_PATH.resolve()),
        high_vol_percentile=regime_config.thresholds.high_vol_percentile,
        trend_abs_momentum_percentile=(
            regime_config.thresholds.trend_abs_momentum_percentile
        ),
        thresholds_by_symbol=thresholds_by_symbol,
        labels_by_row_id=labels_by_row_id,
    )


def _sample_to_regime_row(sample: DatasetSample) -> RegimeSourceRow:
    missing_inputs = [
        column
        for column in ("realized_vol_12", "momentum_3", "macd_line_12_26")
        if column not in sample.features or sample.features[column] is None
    ]
    if missing_inputs:
        raise ValueError(
            "Training samples are missing required regime inputs for regime-sliced "
            f"economics: {missing_inputs}"
        )
    return RegimeSourceRow(
        symbol=sample.symbol,
        interval_begin=sample.interval_begin,
        as_of_time=sample.as_of_time,
        realized_vol_12=float(sample.features["realized_vol_12"]),
        momentum_3=float(sample.features["momentum_3"]),
        macd_line_12_26=float(sample.features["macd_line_12_26"]),
    )


def _build_regime_economics(
    predictions: list[PredictionRecord],
) -> dict[str, dict[str, Any]]:
    rows_by_regime: dict[str, list[PredictionRecord]] = {}
    for prediction in predictions:
        rows_by_regime.setdefault(prediction.regime_label, []).append(prediction)
    ordered_regimes = list(REGIME_LABELS) + sorted(set(rows_by_regime) - set(REGIME_LABELS))
    return {
        regime_label: _build_regime_economics_row(rows_by_regime[regime_label])
        for regime_label in ordered_regimes
        if regime_label in rows_by_regime
    }


def _build_regime_economics_row(predictions: list[PredictionRecord]) -> dict[str, Any]:
    prediction_count = len(predictions)
    trade_count = sum(prediction.long_trade_taken for prediction in predictions)
    mean_long_only_gross_value_proxy = (
        sum(prediction.long_only_gross_value_proxy for prediction in predictions)
        / prediction_count
    )
    mean_long_only_net_value_proxy = (
        sum(prediction.long_only_net_value_proxy for prediction in predictions)
        / prediction_count
    )
    return {
        "prediction_count": prediction_count,
        "trade_count": trade_count,
        "trade_rate": trade_count / prediction_count,
        "mean_long_only_gross_value_proxy": mean_long_only_gross_value_proxy,
        "mean_long_only_net_value_proxy": mean_long_only_net_value_proxy,
        "after_cost_positive": mean_long_only_net_value_proxy > 0.0,
    }


def _fit_training_regime_thresholds(
    rows: list[RegimeSourceRow],
    *,
    symbols: tuple[str, ...],
    high_vol_percentile: float,
    trend_abs_momentum_percentile: float,
) -> dict[str, SymbolThresholds]:
    rows_by_symbol: dict[str, list[RegimeSourceRow]] = {symbol: [] for symbol in symbols}
    for row in rows:
        rows_by_symbol.setdefault(row.symbol, []).append(row)

    thresholds_by_symbol: dict[str, SymbolThresholds] = {}
    for symbol in symbols:
        symbol_rows = sorted(
            rows_by_symbol.get(symbol, []),
            key=lambda row: (row.interval_begin, row.as_of_time),
        )
        if not symbol_rows:
            raise ValueError(
                "Training data does not contain rows for regime-sliced economics "
                f"symbol {symbol}"
            )
        thresholds_by_symbol[symbol] = SymbolThresholds(
            symbol=symbol,
            fitted_row_count=len(symbol_rows),
            high_vol_threshold=compute_percentile(
                [row.realized_vol_12 for row in symbol_rows],
                high_vol_percentile,
            ),
            trend_abs_threshold=compute_percentile(
                [abs(row.momentum_3) for row in symbol_rows],
                trend_abs_momentum_percentile,
            ),
        )
    return thresholds_by_symbol


def _learned_models_beating_after_costs(
    aggregate_summary: dict[str, dict[str, Any]],
    *,
    baseline_name: str,
    learned_model_names: tuple[str, ...],
) -> list[str]:
    baseline_metric = aggregate_summary[baseline_name]["mean_long_only_net_value_proxy"]
    return [
        model_name
        for model_name in learned_model_names
        if aggregate_summary[model_name]["mean_long_only_net_value_proxy"] > baseline_metric
    ]


def _build_specialist_verdicts(
    *,
    recent_predictions: list[PredictionRecord],
    model_configs: dict[str, dict[str, Any]],
    incumbent_predictions: list[PredictionRecord] | None = None,
    incumbent_model_version: str | None = None,
    max_drawdown_tolerance: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Compatibility wrapper for M20 specialist verdict construction."""
    return _build_specialist_verdicts_impl(
        recent_predictions=recent_predictions,
        model_configs=model_configs,
        required_baselines=_REQUIRED_PROMOTION_BASELINES,
        incumbent_predictions=incumbent_predictions,
        incumbent_model_version=incumbent_model_version,
        max_drawdown_tolerance=max_drawdown_tolerance,
    )


def _score_incumbent_on_recent_samples(
    *,
    incumbent_model: Any,
    incumbent_model_name: str,
    samples: tuple[DatasetSample, ...],
    recent_cutoff_rfc3339: str,
    fee_rate: float,
    regime_labels_by_row_id: dict[str, str],
) -> list[PredictionRecord]:
    """Compatibility wrapper for incumbent recent-window scoring."""
    return _score_incumbent_on_recent_samples_impl(
        incumbent_model=incumbent_model,
        incumbent_model_name=incumbent_model_name,
        samples=samples,
        recent_cutoff_rfc3339=recent_cutoff_rfc3339,
        fee_rate=fee_rate,
        regime_labels_by_row_id=regime_labels_by_row_id,
        build_prediction_records=_build_prediction_records,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(make_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    field_names = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
