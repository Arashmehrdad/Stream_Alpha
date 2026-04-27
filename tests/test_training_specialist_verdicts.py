"""Tests for M20 specialist verdict construction."""
# pylint: disable=missing-function-docstring

from app.training.service import PredictionRecord
from app.training.specialist_verdicts import build_specialist_verdicts


_REQUIRED_VERDICT_BASELINES = ("persistence_3", "dummy_most_frequent")


def _pred(
    *,
    model_name: str,
    regime: str,
    net_value: float,
    row_id: str = "r1",
) -> PredictionRecord:
    return PredictionRecord(
        model_name=model_name,
        fold_index=0,
        row_id=row_id,
        symbol="BTC/USD",
        interval_begin="2025-06-01T00:00:00Z",
        as_of_time="2025-06-01T00:00:00Z",
        y_true=1,
        y_pred=1,
        prob_up=0.7,
        confidence=0.7,
        regime_label=regime,
        long_trade_taken=1,
        future_return_3=net_value + 0.001,
        long_only_gross_value_proxy=net_value + 0.001,
        long_only_net_value_proxy=net_value,
    )


def test_specialist_verdicts_accepted_when_beats_incumbent_and_positive() -> None:
    model_configs = {
        "nhits": {
            "candidate_role": "TREND_SPECIALIST",
            "scope_regimes": ["TREND_UP"],
        }
    }
    challenger_preds = [
        _pred(
            model_name="nhits",
            regime="TREND_UP",
            net_value=0.005,
            row_id=f"r{i}",
        )
        for i in range(120)
    ]
    baseline_preds = [
        _pred(
            model_name="persistence_3",
            regime="TREND_UP",
            net_value=0.001,
            row_id=f"r{i}",
        )
        for i in range(120)
    ] + [
        _pred(
            model_name="dummy_most_frequent",
            regime="TREND_UP",
            net_value=0.0005,
            row_id=f"r{i}",
        )
        for i in range(120)
    ]
    incumbent_preds = [
        _pred(
            model_name="autogluon_tabular",
            regime="TREND_UP",
            net_value=0.002,
            row_id=f"r{i}",
        )
        for i in range(120)
    ]
    verdicts = build_specialist_verdicts(
        recent_predictions=challenger_preds + baseline_preds,
        model_configs=model_configs,
        required_baselines=_REQUIRED_VERDICT_BASELINES,
        incumbent_predictions=incumbent_preds,
        incumbent_model_version="m7-test",
        max_drawdown_tolerance=0.01,
    )

    verdict = verdicts["nhits"]
    assert verdict["verdict"] == "accepted"
    assert verdict["verdict_basis"] == "incumbent_comparison"
    assert verdict["incumbent_comparison"]["beats_incumbent"] is True
    assert (
        verdict["incumbent_comparison"]["incumbent_model_version"]
        == "m7-test"
    )


def test_specialist_verdicts_rejected_when_does_not_beat_incumbent() -> None:
    model_configs = {
        "nhits": {
            "candidate_role": "TREND_SPECIALIST",
            "scope_regimes": ["TREND_UP"],
        }
    }
    challenger_preds = [
        _pred(
            model_name="nhits",
            regime="TREND_UP",
            net_value=0.001,
            row_id=f"r{i}",
        )
        for i in range(120)
    ]
    baseline_preds = [
        _pred(
            model_name="persistence_3",
            regime="TREND_UP",
            net_value=0.0005,
            row_id=f"r{i}",
        )
        for i in range(120)
    ] + [
        _pred(
            model_name="dummy_most_frequent",
            regime="TREND_UP",
            net_value=0.0003,
            row_id=f"r{i}",
        )
        for i in range(120)
    ]
    incumbent_preds = [
        _pred(
            model_name="autogluon_tabular",
            regime="TREND_UP",
            net_value=0.005,
            row_id=f"r{i}",
        )
        for i in range(120)
    ]
    verdicts = build_specialist_verdicts(
        recent_predictions=challenger_preds + baseline_preds,
        model_configs=model_configs,
        required_baselines=_REQUIRED_VERDICT_BASELINES,
        incumbent_predictions=incumbent_preds,
        incumbent_model_version="m7-test",
    )

    verdict = verdicts["nhits"]
    assert verdict["verdict"] == "rejected"
    assert verdict["verdict_basis"] == "incumbent_comparison"
    assert "not beat incumbent" in verdict["reason"]


def test_specialist_verdicts_baseline_only_when_no_incumbent() -> None:
    model_configs = {
        "nhits": {
            "candidate_role": "TREND_SPECIALIST",
            "scope_regimes": ["TREND_UP"],
        }
    }
    challenger_preds = [
        _pred(
            model_name="nhits",
            regime="TREND_UP",
            net_value=0.005,
            row_id=f"r{i}",
        )
        for i in range(120)
    ]
    baseline_preds = [
        _pred(
            model_name="persistence_3",
            regime="TREND_UP",
            net_value=0.001,
            row_id=f"r{i}",
        )
        for i in range(120)
    ] + [
        _pred(
            model_name="dummy_most_frequent",
            regime="TREND_UP",
            net_value=0.0005,
            row_id=f"r{i}",
        )
        for i in range(120)
    ]
    verdicts = build_specialist_verdicts(
        recent_predictions=challenger_preds + baseline_preds,
        model_configs=model_configs,
        required_baselines=_REQUIRED_VERDICT_BASELINES,
        incumbent_predictions=None,
    )

    verdict = verdicts["nhits"]
    assert verdict["verdict"] == "accepted"
    assert verdict["verdict_basis"] == "baseline_only"
    assert "incumbent_comparison" not in verdict


def test_specialist_verdicts_rejected_when_drawdown_exceeds_tolerance() -> None:
    model_configs = {
        "nhits": {
            "candidate_role": "TREND_SPECIALIST",
            "scope_regimes": ["TREND_UP"],
        }
    }
    challenger_preds = []
    for index in range(120):
        net_value = 0.01 if index % 2 == 0 else -0.008
        challenger_preds.append(
            _pred(
                model_name="nhits",
                regime="TREND_UP",
                net_value=net_value,
                row_id=f"r{index}",
            )
        )
    baseline_preds = [
        _pred(
            model_name="persistence_3",
            regime="TREND_UP",
            net_value=0.0001,
            row_id=f"r{i}",
        )
        for i in range(120)
    ] + [
        _pred(
            model_name="dummy_most_frequent",
            regime="TREND_UP",
            net_value=0.00005,
            row_id=f"r{i}",
        )
        for i in range(120)
    ]
    incumbent_preds = [
        _pred(
            model_name="autogluon_tabular",
            regime="TREND_UP",
            net_value=0.0005,
            row_id=f"r{i}",
        )
        for i in range(120)
    ]
    verdicts = build_specialist_verdicts(
        recent_predictions=challenger_preds + baseline_preds,
        model_configs=model_configs,
        required_baselines=_REQUIRED_VERDICT_BASELINES,
        incumbent_predictions=incumbent_preds,
        incumbent_model_version="m7-test",
        max_drawdown_tolerance=0.0001,
    )

    verdict = verdicts["nhits"]
    assert verdict["verdict"] == "rejected"
    assert (
        "drawdown" in verdict["reason"].lower()
        or "not beat" in verdict["reason"].lower()
    )
