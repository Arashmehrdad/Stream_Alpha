"""AutoGluon Tabular wrapper for the authoritative Stream Alpha training path."""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor


_DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "GBM": {},
    "CAT": {},
    "XGB": {},
}


class AutoGluonTabularClassifier:  # pylint: disable=too-many-instance-attributes
    """Serializable AutoGluon binary classifier with a self-contained artifact bundle."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        presets: str = "medium_quality",
        time_limit: int | None = 120,
        eval_metric: str = "log_loss",
        hyperparameters: dict[str, Any] | None = None,
        fit_weighted_ensemble: bool = True,
        num_bag_folds: int = 0,
        num_stack_levels: int = 0,
        verbosity: int = 0,
    ) -> None:
        self.presets = str(presets)
        self.time_limit = None if time_limit is None else int(time_limit)
        self.eval_metric = str(eval_metric)
        self.hyperparameters = (
            dict(_DEFAULT_HYPERPARAMETERS)
            if hyperparameters is None
            else dict(hyperparameters)
        )
        self.fit_weighted_ensemble = bool(fit_weighted_ensemble)
        self.num_bag_folds = int(num_bag_folds)
        self.num_stack_levels = int(num_stack_levels)
        self.verbosity = int(verbosity)
        self._feature_columns: tuple[str, ...] = ()
        self._predictor_archive: bytes | None = None
        self._runtime_dir: Path | None = None
        self._predictor: TabularPredictor | None = None

    def fit(
        self,
        rows: list[dict[str, Any]],
        labels: list[int],
    ) -> "AutoGluonTabularClassifier":
        """Train one self-contained AutoGluon predictor."""
        training_frame = pd.DataFrame(rows)
        training_frame["label"] = [int(label) for label in labels]
        self._feature_columns = tuple(
            str(column)
            for column in training_frame.columns
            if column != "label"
        )

        fit_root = Path(tempfile.mkdtemp(prefix="streamalpha-autogluon-fit-"))
        predictor_dir = fit_root / "predictor"
        predictor = TabularPredictor(
            label="label",
            problem_type="binary",
            eval_metric=self.eval_metric,
            path=str(predictor_dir),
        )
        fit_kwargs: dict[str, Any] = {
            "train_data": training_frame,
            "presets": self.presets,
            "hyperparameters": self.hyperparameters,
            "fit_weighted_ensemble": self.fit_weighted_ensemble,
            "num_bag_folds": self.num_bag_folds,
            "num_stack_levels": self.num_stack_levels,
            "verbosity": self.verbosity,
        }
        if self.time_limit is not None:
            fit_kwargs["time_limit"] = self.time_limit
        predictor.fit(**fit_kwargs)
        self._predictor_archive = _archive_predictor_dir(predictor_dir)
        shutil.rmtree(fit_root, ignore_errors=True)
        self._cleanup_runtime_dir()
        self._predictor = None
        self._ensure_predictor()
        return self

    def predict(self, rows: list[dict[str, Any]]) -> list[int]:
        """Return binary class predictions for the requested feature rows."""
        predictor = self._ensure_predictor()
        prediction_frame = predictor.predict(pd.DataFrame(rows))
        values = (
            prediction_frame.tolist()
            if hasattr(prediction_frame, "tolist")
            else list(prediction_frame)
        )
        return [int(value) for value in values]

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        """Return binary probabilities in the existing Stream Alpha artifact contract."""
        predictor = self._ensure_predictor()
        probabilities = predictor.predict_proba(
            pd.DataFrame(rows),
            as_multiclass=True,
        )
        return _coerce_binary_probabilities(probabilities)

    def get_expanded_feature_names(self) -> list[str]:
        """Return stable feature names for artifact metadata and explainability contracts."""
        return list(self._feature_columns)

    def __getstate__(self) -> dict[str, Any]:
        """Serialize only the self-contained predictor bundle and stable config."""
        return {
            "presets": self.presets,
            "time_limit": self.time_limit,
            "eval_metric": self.eval_metric,
            "hyperparameters": self.hyperparameters,
            "fit_weighted_ensemble": self.fit_weighted_ensemble,
            "num_bag_folds": self.num_bag_folds,
            "num_stack_levels": self.num_stack_levels,
            "verbosity": self.verbosity,
            "feature_columns": list(self._feature_columns),
            "predictor_archive": self._predictor_archive,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the minimal self-contained state and lazily reload the predictor."""
        self.presets = str(state["presets"])
        self.time_limit = state["time_limit"]
        self.eval_metric = str(state["eval_metric"])
        self.hyperparameters = dict(state["hyperparameters"])
        self.fit_weighted_ensemble = bool(state["fit_weighted_ensemble"])
        self.num_bag_folds = int(state["num_bag_folds"])
        self.num_stack_levels = int(state["num_stack_levels"])
        self.verbosity = int(state["verbosity"])
        self._feature_columns = tuple(str(column) for column in state["feature_columns"])
        self._predictor_archive = state["predictor_archive"]
        self._runtime_dir = None
        self._predictor = None

    def __del__(self) -> None:
        self._cleanup_runtime_dir()

    def _ensure_predictor(self) -> TabularPredictor:
        """Restore the predictor from the bundled archive on first use."""
        if self._predictor is None:
            if self._predictor_archive is None:
                raise ValueError(
                    "AutoGluon predictor archive is missing from the artifact"
                )
            self._cleanup_runtime_dir()
            runtime_root = Path(tempfile.mkdtemp(prefix="streamalpha-autogluon-runtime-"))
            predictor_dir = runtime_root / "predictor"
            _restore_predictor_dir(self._predictor_archive, predictor_dir)
            self._predictor = TabularPredictor.load(str(predictor_dir))
            self._runtime_dir = runtime_root
        return self._predictor

    def _cleanup_runtime_dir(self) -> None:
        """Remove any temporary runtime directory created from the bundled archive."""
        if self._runtime_dir is None:
            return
        shutil.rmtree(self._runtime_dir, ignore_errors=True)
        self._runtime_dir = None


def build_autogluon_tabular_classifier(
    model_config: dict[str, Any],
) -> AutoGluonTabularClassifier:
    """Build the authoritative AutoGluon tabular classifier from checked-in config."""
    return AutoGluonTabularClassifier(
        presets=str(model_config.get("presets", "medium_quality")),
        time_limit=(
            None
            if model_config.get("time_limit") is None
            else int(model_config["time_limit"])
        ),
        eval_metric=str(model_config.get("eval_metric", "log_loss")),
        hyperparameters=dict(
            model_config.get("hyperparameters", _DEFAULT_HYPERPARAMETERS)
        ),
        fit_weighted_ensemble=bool(model_config.get("fit_weighted_ensemble", True)),
        num_bag_folds=int(model_config.get("num_bag_folds", 0)),
        num_stack_levels=int(model_config.get("num_stack_levels", 0)),
        verbosity=int(model_config.get("verbosity", 0)),
    )


def _archive_predictor_dir(predictor_dir: Path) -> bytes:
    """Zip one AutoGluon predictor directory into the single artifact payload."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(predictor_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, arcname=path.relative_to(predictor_dir).as_posix())
    return buffer.getvalue()


def _restore_predictor_dir(archive_bytes: bytes, predictor_dir: Path) -> None:
    """Restore one predictor directory from the archived artifact payload."""
    predictor_dir.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
        archive.extractall(predictor_dir)


def _coerce_binary_probabilities(probabilities: Any) -> list[list[float]]:
    """Normalize AutoGluon output into the repository's binary probability contract."""
    if hasattr(probabilities, "to_dict"):
        records = probabilities.to_dict("records")
        return [
            [
                float(_lookup_probability(record, label=0)),
                float(_lookup_probability(record, label=1)),
            ]
            for record in records
        ]
    if hasattr(probabilities, "tolist"):
        rows = probabilities.tolist()
    else:
        rows = list(probabilities)
    return [[float(row[0]), float(row[1])] for row in rows]


def _lookup_probability(record: dict[Any, Any], *, label: int) -> float:
    """Read one binary class probability from an AutoGluon probability row."""
    candidate_keys = (label, str(label), bool(label))
    for key in candidate_keys:
        if key in record:
            return float(record[key])
    raise ValueError(f"AutoGluon probability output is missing class {label}")
