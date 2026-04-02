"""Small progress helpers for local M7 AutoGluon operator runs."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339


@dataclass(frozen=True, slots=True)
class AutoGluonFitProgressSnapshot:
    """Inspectable progress snapshot for one local AutoGluon fit work directory."""

    fit_dir: str
    predictor_dir: str | None
    models_dir: str | None
    trainer_path: str | None
    current_best_model: str | None
    total_model_count: int | None
    discovered_model_count: int
    latest_model_name: str | None
    latest_model_updated_at: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe payload for PowerShell operator scripts."""
        return make_json_safe(asdict(self))


def build_autogluon_fit_progress_snapshot(
    fit_dir: Path,
) -> AutoGluonFitProgressSnapshot:
    """Inspect one local AutoGluon fit directory without mutating it."""
    resolved_fit_dir = Path(fit_dir).resolve()
    predictor_dir = resolved_fit_dir / "predictor"
    models_dir = predictor_dir / "models"
    if not models_dir.exists():
        return AutoGluonFitProgressSnapshot(
            fit_dir=str(resolved_fit_dir),
            predictor_dir=None if not predictor_dir.exists() else str(predictor_dir),
            models_dir=None,
            trainer_path=None,
            current_best_model=None,
            total_model_count=None,
            discovered_model_count=0,
            latest_model_name=None,
            latest_model_updated_at=None,
        )

    model_dirs = sorted(
        (
            path
            for path in models_dir.iterdir()
            if path.is_dir() and path.name != "utils"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    latest_model_dir = model_dirs[0] if model_dirs else None
    trainer_path = models_dir / "trainer.pkl"
    current_best_model: str | None = None
    total_model_count: int | None = None
    if trainer_path.exists():
        trainer = _load_pickle(trainer_path)
        current_best_model = _coerce_optional_string(getattr(trainer, "model_best", None))
        model_graph = getattr(trainer, "model_graph", None)
        if model_graph is not None and hasattr(model_graph, "nodes"):
            total_model_count = len(list(model_graph.nodes))

    return AutoGluonFitProgressSnapshot(
        fit_dir=str(resolved_fit_dir),
        predictor_dir=str(predictor_dir),
        models_dir=str(models_dir),
        trainer_path=str(trainer_path) if trainer_path.exists() else None,
        current_best_model=current_best_model,
        total_model_count=total_model_count,
        discovered_model_count=len(model_dirs),
        latest_model_name=None if latest_model_dir is None else latest_model_dir.name,
        latest_model_updated_at=(
            None
            if latest_model_dir is None
            else to_rfc3339(_to_utc_datetime(latest_model_dir.stat().st_mtime))
        ),
    )


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as input_file:
        return pickle.load(input_file)


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _to_utc_datetime(timestamp: float):
    from datetime import datetime, UTC

    return datetime.fromtimestamp(timestamp, tz=UTC)


def main() -> None:
    """Emit one JSON progress snapshot for an AutoGluon fit directory."""
    parser = argparse.ArgumentParser(description="Inspect local AutoGluon fit progress")
    parser.add_argument("--fit-dir", required=True, help="Path to the fit work directory")
    arguments = parser.parse_args()
    snapshot = build_autogluon_fit_progress_snapshot(Path(arguments.fit_dir))
    print(json.dumps(snapshot.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
