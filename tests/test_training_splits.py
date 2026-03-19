"""Tests for M3 purged expanding walk-forward split integrity."""

# pylint: disable=duplicate-code

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.training.splits import build_walk_forward_splits


def test_walk_forward_splits_are_time_ordered_and_purged() -> None:
    """Each fold should expand the train window and keep a 3-candle purge gap."""
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = tuple(base_time + timedelta(minutes=5 * index) for index in range(100))

    folds = build_walk_forward_splits(
        timestamps,
        first_train_fraction=0.5,
        test_fraction=0.1,
        test_folds=5,
        purge_gap_candles=3,
    )

    assert len(folds) == 5
    assert len(folds[0].train_timestamps) == 47
    assert len(folds[0].purged_timestamps) == 3
    assert len(folds[0].test_timestamps) == 10

    previous_train_size = 0
    previous_test_end = None
    for fold in folds:
        assert set(fold.train_timestamps).isdisjoint(fold.purged_timestamps)
        assert set(fold.train_timestamps).isdisjoint(fold.test_timestamps)
        assert set(fold.purged_timestamps).isdisjoint(fold.test_timestamps)
        assert len(fold.train_timestamps) > previous_train_size
        assert fold.train_timestamps[-1] < fold.test_timestamps[0]
        assert len(fold.purged_timestamps) == 3
        if previous_test_end is not None:
            assert previous_test_end < fold.test_timestamps[0]
        previous_train_size = len(fold.train_timestamps)
        previous_test_end = fold.test_timestamps[-1]
