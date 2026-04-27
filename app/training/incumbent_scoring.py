"""Registry incumbent loading and recent-window scoring for M20 verdicts."""
# pylint: disable=too-many-arguments

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import joblib

from app.training.registry import load_current_registry_entry


def load_incumbent_model() -> tuple[Any, str, str] | None:
    """Load the current registry champion for comparison scoring."""
    entry = load_current_registry_entry()
    if entry is None:
        return None
    model_path = Path(str(entry["model_artifact_path"]))
    if not model_path.is_file():
        print(
            "[training] warning: incumbent model artifact not found at "
            f"{model_path}; skipping incumbent comparison",
        )
        return None
    payload = joblib.load(model_path)
    return (
        payload["model"],
        str(entry["model_version"]),
        str(entry.get("model_name", payload.get("model_name", "unknown"))),
    )


def score_incumbent_on_recent_samples(
    *,
    incumbent_model: Any,
    incumbent_model_name: str,
    samples: tuple[Any, ...],
    recent_cutoff_rfc3339: str,
    fee_rate: float,
    regime_labels_by_row_id: dict[str, str],
    build_prediction_records: Callable[..., list[Any]],
) -> list[Any]:
    """Score the incumbent model on recent-window samples."""
    cutoff = _parse_time(recent_cutoff_rfc3339)
    recent_samples = [
        sample
        for sample in samples
        if sample.as_of_time >= cutoff and sample.row_id in regime_labels_by_row_id
    ]
    if not recent_samples:
        return []

    test_features = [sample.features for sample in recent_samples]
    predicted_proba = incumbent_model.predict_proba(test_features)
    probabilities = [float(row[1]) for row in predicted_proba]
    predicted_labels = [
        1 if probability >= 0.5 else 0
        for probability in probabilities
    ]
    return build_prediction_records(
        model_name=incumbent_model_name,
        fold_index=-1,
        test_samples=recent_samples,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        fee_rate=fee_rate,
        regime_labels_by_row_id=regime_labels_by_row_id,
    )


def _parse_time(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
