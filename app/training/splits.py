"""Purged expanding walk-forward splits for M3 offline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One expanding train / purged gap / test time split."""

    fold_index: int
    train_timestamps: tuple[datetime, ...]
    purged_timestamps: tuple[datetime, ...]
    test_timestamps: tuple[datetime, ...]


def build_walk_forward_splits(
    timestamps: tuple[datetime, ...],
    *,
    first_train_fraction: float,
    test_fraction: float,
    test_folds: int,
    purge_gap_candles: int,
) -> tuple[WalkForwardFold, ...]:
    """Build global time-ordered purged walk-forward folds from unique timestamps."""
    unique_timestamps = tuple(sorted(set(timestamps)))
    total_timestamps = len(unique_timestamps)
    if total_timestamps == 0:
        raise ValueError("At least one timestamp is required to build walk-forward splits")

    folds: list[WalkForwardFold] = []
    for fold_index in range(test_folds):
        test_start = int(total_timestamps * (first_train_fraction + (fold_index * test_fraction)))
        test_end = int(
            total_timestamps * (first_train_fraction + ((fold_index + 1) * test_fraction))
        )
        train_end = max(0, test_start - purge_gap_candles)

        train_timestamps = unique_timestamps[:train_end]
        purged_timestamps = unique_timestamps[train_end:test_start]
        test_timestamps = unique_timestamps[test_start:test_end]

        if not train_timestamps:
            raise ValueError(f"Fold {fold_index} has no training timestamps")
        if not test_timestamps:
            raise ValueError(f"Fold {fold_index} has no test timestamps")

        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                train_timestamps=train_timestamps,
                purged_timestamps=purged_timestamps,
                test_timestamps=test_timestamps,
            )
        )

    return tuple(folds)


def minimum_required_unique_timestamps(
    *,
    first_train_fraction: float,
    test_fraction: float,
    test_folds: int,
    purge_gap_candles: int,
) -> int:
    """Return the smallest timestamp count that can satisfy the configured split."""
    base_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    for total_timestamps in range(1, 10_001):
        synthetic_timestamps = tuple(
            base_time + timedelta(minutes=index) for index in range(total_timestamps)
        )
        try:
            build_walk_forward_splits(
                synthetic_timestamps,
                first_train_fraction=first_train_fraction,
                test_fraction=test_fraction,
                test_folds=test_folds,
                purge_gap_candles=purge_gap_candles,
            )
            return total_timestamps
        except ValueError:
            continue
    raise ValueError("Could not determine a feasible minimum timestamp count")
