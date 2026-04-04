"""AutoGluon Tabular wrapper for the authoritative Stream Alpha training path."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import io
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor

from app.training.workdirs import create_local_training_work_dir


class AutoGluonTabularClassifier:  # pylint: disable=too-many-instance-attributes
    """Serializable AutoGluon binary classifier with a self-contained artifact bundle."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        presets: str = "medium_quality",
        time_limit: int | None = 120,
        eval_metric: str = "log_loss",
        hyperparameters: Any | None = None,
        fit_weighted_ensemble: bool = True,
        num_bag_folds: int = 0,
        num_stack_levels: int = 0,
        num_bag_sets: int | None = 1,
        fold_fitting_strategy: str | None = None,
        dynamic_stacking: Any | None = None,
        calibrate_decision_threshold: bool = False,
        verbosity: int = 0,
        num_gpus: int = 0,
    ) -> None:
        self.presets = str(presets)
        self.time_limit = None if time_limit is None else int(time_limit)
        self.eval_metric = str(eval_metric)
        self.num_gpus = int(num_gpus)
        self.hyperparameters = _copy_optional_value(hyperparameters)
        self.fit_weighted_ensemble = bool(fit_weighted_ensemble)
        self.num_bag_folds = int(num_bag_folds)
        self.num_stack_levels = int(num_stack_levels)
        self.num_bag_sets = None if num_bag_sets is None else int(num_bag_sets)
        self.fold_fitting_strategy = (
            None
            if fold_fitting_strategy is None
            else str(fold_fitting_strategy)
        )
        self.dynamic_stacking = _copy_optional_value(dynamic_stacking)
        self.calibrate_decision_threshold = bool(calibrate_decision_threshold)
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

        fit_root = create_local_training_work_dir(
            prefix="streamalpha-autogluon-fit-",
        )
        predictor_dir = fit_root / "predictor"
        try:
            predictor = TabularPredictor(
                label="label",
                problem_type="binary",
                eval_metric=self.eval_metric,
                path=str(predictor_dir),
            )
            fit_kwargs: dict[str, Any] = {
                "train_data": training_frame,
                "presets": self.presets,
                "fit_weighted_ensemble": self.fit_weighted_ensemble,
                "num_bag_folds": self.num_bag_folds,
                "num_stack_levels": self.num_stack_levels,
                "calibrate_decision_threshold": self.calibrate_decision_threshold,
                "verbosity": self.verbosity,
            }
            if self.time_limit is not None:
                fit_kwargs["time_limit"] = self.time_limit
            if self.hyperparameters is not None:
                fit_kwargs["hyperparameters"] = _copy_optional_value(
                    self.hyperparameters
                )
            if self.num_bag_sets is not None and self.num_bag_folds > 0:
                fit_kwargs["num_bag_sets"] = self.num_bag_sets
            if self.fold_fitting_strategy is not None and self.num_bag_folds > 0:
                fit_kwargs["ag_args_ensemble"] = {
                    "fold_fitting_strategy": self.fold_fitting_strategy
                }
            if self.dynamic_stacking is not None:
                fit_kwargs["dynamic_stacking"] = _copy_optional_value(
                    self.dynamic_stacking
                )
            if self.num_gpus > 0:
                fit_kwargs["ag_args_fit"] = {"num_gpus": self.num_gpus}
            predictor.fit(**fit_kwargs)
            self._predictor_archive = _archive_predictor_dir(predictor_dir)
        finally:
            shutil.rmtree(fit_root, ignore_errors=True)
        self._cleanup_runtime_dir()
        self._predictor = None
        self._ensure_predictor()
        return self

    def predict(self, rows: list[dict[str, Any]]) -> list[int]:
        """Return binary class predictions for the requested feature rows."""
        predictor = self._ensure_predictor()
        with _numpy_random_pickle_compat():
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
        with _numpy_random_pickle_compat():
            probabilities = predictor.predict_proba(
                pd.DataFrame(rows),
                as_multiclass=True,
            )
        return _coerce_binary_probabilities(probabilities)

    def get_expanded_feature_names(self) -> list[str]:
        """Return stable feature names for artifact metadata and explainability contracts."""
        return list(self._feature_columns)

    def get_training_config(self) -> dict[str, Any]:
        """Return the effective AutoGluon fit config stored in this artifact."""
        return {
            "presets": self.presets,
            "time_limit": self.time_limit,
            "eval_metric": self.eval_metric,
            "hyperparameters": _copy_optional_value(self.hyperparameters),
            "fit_weighted_ensemble": self.fit_weighted_ensemble,
            "num_bag_folds": self.num_bag_folds,
            "num_stack_levels": self.num_stack_levels,
            "num_bag_sets": self.num_bag_sets,
            "fold_fitting_strategy": self.fold_fitting_strategy,
            "dynamic_stacking": _copy_optional_value(self.dynamic_stacking),
            "calibrate_decision_threshold": self.calibrate_decision_threshold,
            "verbosity": self.verbosity,
        }

    def __getstate__(self) -> dict[str, Any]:
        """Serialize only the self-contained predictor bundle and stable config."""
        return {
            "presets": self.presets,
            "time_limit": self.time_limit,
            "eval_metric": self.eval_metric,
            "hyperparameters": _copy_optional_value(self.hyperparameters),
            "fit_weighted_ensemble": self.fit_weighted_ensemble,
            "num_bag_folds": self.num_bag_folds,
            "num_stack_levels": self.num_stack_levels,
            "num_bag_sets": self.num_bag_sets,
            "fold_fitting_strategy": self.fold_fitting_strategy,
            "dynamic_stacking": _copy_optional_value(self.dynamic_stacking),
            "calibrate_decision_threshold": self.calibrate_decision_threshold,
            "verbosity": self.verbosity,
            "feature_columns": list(self._feature_columns),
            "predictor_archive": self._predictor_archive,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the minimal self-contained state and lazily reload the predictor."""
        self.presets = str(state["presets"])
        self.time_limit = state["time_limit"]
        self.eval_metric = str(state["eval_metric"])
        self.hyperparameters = _copy_optional_value(state.get("hyperparameters"))
        self.fit_weighted_ensemble = bool(state["fit_weighted_ensemble"])
        self.num_bag_folds = int(state["num_bag_folds"])
        self.num_stack_levels = int(state["num_stack_levels"])
        if "num_bag_sets" not in state:
            self.num_bag_sets = 1
        else:
            self.num_bag_sets = (
                None if state["num_bag_sets"] is None else int(state["num_bag_sets"])
            )
        self.fold_fitting_strategy = (
            None
            if state.get("fold_fitting_strategy") is None
            else str(state["fold_fitting_strategy"])
        )
        self.dynamic_stacking = _copy_optional_value(state.get("dynamic_stacking"))
        self.calibrate_decision_threshold = bool(
            state.get("calibrate_decision_threshold", False)
        )
        self.verbosity = int(state["verbosity"])
        self._feature_columns = tuple(str(column) for column in state["feature_columns"])
        self._predictor_archive = state["predictor_archive"]
        self._runtime_dir = None
        self._predictor = None

    def __del__(self) -> None:
        try:
            self._cleanup_runtime_dir()
        except Exception:  # pragma: no cover - interpreter shutdown best effort only
            pass

    def _ensure_predictor(self) -> TabularPredictor:
        """Restore the predictor from the bundled archive on first use."""
        if self._predictor is None:
            if self._predictor_archive is None:
                raise ValueError(
                    "AutoGluon predictor archive is missing from the artifact"
                )
            self._cleanup_runtime_dir()
            runtime_root = create_local_training_work_dir(
                prefix="streamalpha-autogluon-runtime-",
            )
            predictor_dir = runtime_root / "predictor"
            try:
                _restore_predictor_dir(self._predictor_archive, predictor_dir)
                # Local-first runs can train on Windows and score inside Linux containers.
                with _numpy_random_pickle_compat():
                    self._predictor = TabularPredictor.load(
                        str(predictor_dir),
                        require_py_version_match=False,
                    )
                self._runtime_dir = runtime_root
            except Exception:
                shutil.rmtree(runtime_root, ignore_errors=True)
                raise
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
    hyperparameters = (
        None
        if "hyperparameters" not in model_config
        or model_config.get("hyperparameters") is None
        else _copy_optional_value(model_config["hyperparameters"])
    )
    if "num_bag_sets" not in model_config:
        num_bag_sets = 1
    else:
        num_bag_sets = (
            None
            if model_config.get("num_bag_sets") is None
            else int(model_config["num_bag_sets"])
        )
    fold_fitting_strategy = (
        None
        if model_config.get("fold_fitting_strategy") is None
        else str(model_config["fold_fitting_strategy"])
    )
    dynamic_stacking = (
        None
        if "dynamic_stacking" not in model_config
        else _copy_optional_value(model_config.get("dynamic_stacking"))
    )
    return AutoGluonTabularClassifier(
        presets=str(model_config.get("presets", "medium_quality")),
        time_limit=(
            None
            if model_config.get("time_limit") is None
            else int(model_config["time_limit"])
        ),
        eval_metric=str(model_config.get("eval_metric", "log_loss")),
        hyperparameters=hyperparameters,
        fit_weighted_ensemble=bool(model_config.get("fit_weighted_ensemble", True)),
        num_bag_folds=int(model_config.get("num_bag_folds", 0)),
        num_stack_levels=int(model_config.get("num_stack_levels", 0)),
        num_bag_sets=num_bag_sets,
        fold_fitting_strategy=fold_fitting_strategy,
        dynamic_stacking=dynamic_stacking,
        calibrate_decision_threshold=bool(
            model_config.get("calibrate_decision_threshold", False)
        ),
        verbosity=int(model_config.get("verbosity", 0)),
        num_gpus=int(model_config.get("num_gpus", 0)),
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


@contextmanager
def _numpy_random_pickle_compat():
    """Temporarily normalize legacy NumPy MT19937 pickles during artifact restore/score."""
    try:
        import numpy.random._pickle as numpy_random_pickle  # pylint: disable=import-outside-toplevel
        from numpy.random.mtrand import RandomState  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - numpy is always present in runtime tests
        yield
        return

    original_bit_generator_ctor = getattr(
        numpy_random_pickle, "__bit_generator_ctor", None
    )
    original_randomstate_ctor = getattr(numpy_random_pickle, "__randomstate_ctor", None)
    if (
        original_bit_generator_ctor is None
        or original_randomstate_ctor is None
        or (
            _bit_generator_ctor_accepts_legacy_types(original_bit_generator_ctor)
            and _bit_generator_accepts_legacy_state_tuples()
        )
    ):
        yield
        return

    class _LegacyBitGeneratorProxy:
        """Bridge legacy pickle state into the current immutable NumPy bit-generator API."""

        def __init__(self, bit_generator: Any) -> None:
            self._bit_generator = bit_generator

        def __getattr__(self, name: str) -> Any:
            return getattr(self._bit_generator, name)

        def __setstate__(self, state: Any) -> None:
            normalized_state = _normalize_legacy_bit_generator_state(state)
            self._bit_generator.__setstate__(normalized_state)

        def unwrap(self) -> Any:
            return self._bit_generator

    def _compat_bit_generator_ctor(bit_generator_name: Any = "MT19937"):
        normalized_name = _normalize_legacy_bit_generator_name(bit_generator_name)
        bit_generator = original_bit_generator_ctor(normalized_name)
        return _LegacyBitGeneratorProxy(bit_generator)

    def _compat_randomstate_ctor(
        bit_generator_name: Any = "MT19937",
        bit_generator_ctor: Any = _compat_bit_generator_ctor,
    ):
        if isinstance(bit_generator_name, _LegacyBitGeneratorProxy):
            return RandomState(bit_generator_name.unwrap())
        bit_generator = bit_generator_ctor(bit_generator_name)
        if isinstance(bit_generator, _LegacyBitGeneratorProxy):
            bit_generator = bit_generator.unwrap()
        return RandomState(bit_generator)

    numpy_random_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
    numpy_random_pickle.__randomstate_ctor = _compat_randomstate_ctor
    try:
        yield
    finally:
        numpy_random_pickle.__bit_generator_ctor = original_bit_generator_ctor
        numpy_random_pickle.__randomstate_ctor = original_randomstate_ctor


def _bit_generator_ctor_accepts_legacy_types(bit_generator_ctor: Any) -> bool:
    """Return whether the active NumPy pickle helper already accepts class objects."""
    try:
        from numpy.random._mt19937 import MT19937  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - numpy is always present in runtime tests
        return True

    try:
        bit_generator_ctor(MT19937)
    except ValueError as error:
        return "BitGenerator module" not in str(error)
    except Exception:
        return True
    return True


def _bit_generator_accepts_legacy_state_tuples() -> bool:
    """Return whether the active NumPy MT19937 runtime still accepts tuple legacy state."""
    try:
        from numpy.random._mt19937 import MT19937  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - numpy is always present in runtime tests
        return True

    bit_generator = MT19937()
    legacy_state = (bit_generator.__getstate__(), None)
    try:
        bit_generator.__setstate__(legacy_state)
    except ValueError as error:
        return "legacy MT19937 state" not in str(error)
    except Exception:
        return True
    return True


def _normalize_legacy_bit_generator_name(bit_generator_name: Any) -> Any:
    """Map legacy pickled NumPy bit-generator class objects back to stable names."""
    if isinstance(bit_generator_name, str):
        return bit_generator_name
    module_name = str(getattr(bit_generator_name, "__module__", ""))
    name = getattr(bit_generator_name, "__name__", None)
    if name is not None and module_name.startswith("numpy.random"):
        return str(name)
    return bit_generator_name


def _normalize_legacy_bit_generator_state(state: Any) -> Any:
    """Strip the deprecated gaussian-cache tuple wrapper from legacy MT19937 state."""
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
        return state[0]
    return state


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


def _copy_optional_value(value: Any | None) -> Any | None:
    """Return a defensive copy for explicit config values while preserving None."""
    return None if value is None else copy.deepcopy(value)
