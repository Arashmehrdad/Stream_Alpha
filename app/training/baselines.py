"""Baseline models used for M3 offline training comparison."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

from app.training.dataset import DatasetSample


class PersistenceBaseline:
    """Official naive benchmark based on the most recent realized 3-candle direction."""

    name = "persistence_3"

    def fit(self, _: Sequence[DatasetSample]) -> "PersistenceBaseline":
        """Return self because the persistence baseline is parameter-free."""
        return self

    def predict(self, samples: Sequence[DatasetSample]) -> list[int]:
        """Predict the direction using only the symbol-local realized 3-candle direction."""
        return [sample.persistence_prediction for sample in samples]

    def predict_proba(self, samples: Sequence[DatasetSample]) -> list[list[float]]:
        """Emit deterministic probabilities consistent with the persistence prediction."""
        return [
            [1.0, 0.0] if sample.persistence_prediction == 0 else [0.0, 1.0]
            for sample in samples
        ]


def build_dummy_classifier() -> Pipeline:
    """Return the sanity-floor most-frequent dummy classifier pipeline."""
    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )
