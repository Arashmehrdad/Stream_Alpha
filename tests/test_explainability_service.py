"""Unit tests for the M14 explainability foundation."""

# pylint: disable=too-few-public-methods

from __future__ import annotations

from app.explainability.service import (
    REGIME_HIGH_VOL,
    REGIME_RANGE,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
    build_regime_reason,
    compute_top_feature_contributions,
)
from app.regime.live import ResolvedRegime


class LinearContributionModel:
    """Small deterministic model stub for one-at-a-time ablation tests."""

    def predict_proba(self, rows: list[dict]) -> list[list[float]]:
        """Return one stable binary probability pair per row."""
        payload: list[list[float]] = []
        for row in rows:
            prob_up = (
                0.5
                + (0.50 * float(row["momentum_3"]))
                - (0.20 * float(row["realized_vol_12"]))
                + (0.05 * float(row["volume_zscore_12"]))
            )
            payload.append([1.0 - prob_up, prob_up])
        return payload


def test_top_feature_contributions_are_ranked_by_absolute_signed_effect() -> None:
    """Reference ablation should return stable top features in descending impact order."""
    top_features = compute_top_feature_contributions(
        model=LinearContributionModel(),
        feature_input={
            "momentum_3": 0.08,
            "realized_vol_12": 0.10,
            "volume_zscore_12": 2.0,
        },
        base_prob_up=0.62,
        reference_values={
            "momentum_3": 0.02,
            "realized_vol_12": 0.04,
            "volume_zscore_12": 0.0,
        },
        explainable_feature_names=(
            "momentum_3",
            "realized_vol_12",
            "volume_zscore_12",
        ),
        top_feature_count=3,
    )

    assert [feature.feature_name for feature in top_features] == [
        "volume_zscore_12",
        "momentum_3",
        "realized_vol_12",
    ]
    assert top_features[0].direction == "UP"
    assert top_features[1].direction == "UP"
    assert top_features[2].direction == "DOWN"
    assert abs(top_features[0].signed_contribution) >= abs(
        top_features[1].signed_contribution
    )
    assert abs(top_features[1].signed_contribution) >= abs(
        top_features[2].signed_contribution
    )


def test_regime_reason_generation_uses_explicit_reason_codes() -> None:
    """Each supported regime should map onto one explicit reason code and text."""
    cases = (
        (
            "HIGH_VOL",
            0.08,
            0.01,
            0.50,
            REGIME_HIGH_VOL,
        ),
        (
            "TREND_UP",
            0.03,
            0.03,
            0.80,
            REGIME_TREND_UP,
        ),
        (
            "TREND_DOWN",
            0.03,
            -0.03,
            -0.80,
            REGIME_TREND_DOWN,
        ),
        (
            "RANGE",
            0.03,
            0.005,
            0.10,
            REGIME_RANGE,
        ),
    )

    for regime_label, realized_vol_12, momentum_3, macd_line_12_26, expected_code in cases:
        regime_reason = build_regime_reason(
            resolved_regime=ResolvedRegime(
                symbol="BTC/USD",
                interval_begin="2026-03-21T12:00:00Z",
                as_of_time="2026-03-21T12:05:00Z",
                row_id="BTC/USD|2026-03-21T12:00:00Z",
                regime_label=regime_label,
                regime_run_id="20260321T120000Z",
                regime_artifact_path="artifacts/regime/m8/20260321T120000Z/thresholds.json",
                high_vol_threshold=0.05,
                trend_abs_threshold=0.02,
                realized_vol_12=realized_vol_12,
                momentum_3=momentum_3,
                macd_line_12_26=macd_line_12_26,
            ),
            trade_allowed=regime_label in {"TREND_UP", "RANGE"},
        )

        assert regime_reason.reason_code == expected_code
        assert regime_reason.regime_label == regime_label
        assert regime_reason.reason_text
