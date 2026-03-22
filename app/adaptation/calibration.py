"""Simple inspectable confidence recalibration helpers for M19."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.isotonic import IsotonicRegression

from app.adaptation.schemas import CalibrationProfile


def build_isotonic_calibration_profile(
    *,
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    source_window: str,
) -> CalibrationProfile:
    """Train a local isotonic calibration profile on recent offline or shadow data."""
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal length")
    if len(probabilities) < 2:
        return CalibrationProfile(method="identity", trained_sample_count=len(probabilities))
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(list(probabilities), list(outcomes))
    return CalibrationProfile(
        method="isotonic",
        x_points=[float(value) for value in model.X_thresholds_],
        y_points=[float(value) for value in model.y_thresholds_],
        trained_sample_count=len(probabilities),
        source_window=source_window,
    )


def apply_calibration(profile: CalibrationProfile, probability: float) -> float:
    """Apply one stored local calibration profile."""
    if profile.method != "isotonic" or not profile.x_points or not profile.y_points:
        return probability
    if probability <= profile.x_points[0]:
        return profile.y_points[0]
    if probability >= profile.x_points[-1]:
        return profile.y_points[-1]
    for index in range(1, len(profile.x_points)):
        left_x = profile.x_points[index - 1]
        right_x = profile.x_points[index]
        if left_x <= probability <= right_x:
            left_y = profile.y_points[index - 1]
            right_y = profile.y_points[index]
            if right_x == left_x:
                return right_y
            ratio = (probability - left_x) / (right_x - left_x)
            return left_y + ((right_y - left_y) * ratio)
    return probability
