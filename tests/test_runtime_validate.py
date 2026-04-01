"""Focused M16 runtime validation and profile-resolution tests."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path

import joblib

from app.runtime.config import resolve_runtime_profile
from app.runtime.validate import (
    build_startup_validation_report,
    write_startup_validation_report,
)
from dashboards.streamlit_app import resolve_dashboard_trading_config_path
from scripts.run_paper_trader import resolve_trader_config_path


class _SerializableProbabilityModel:  # pylint: disable=too-few-public-methods
    def predict_proba(self, rows: list[dict]) -> list[list[float]]:
        return [[0.4, 0.6] for _ in rows]


def _write_model_artifact(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model_name": "runtime_candidate_fixture",
            "trained_at": "2026-03-22T10:00:00Z",
            "feature_columns": ["symbol", "close_price"],
            "expanded_feature_names": ["symbol=BTC/USD", "close_price"],
            "model": _SerializableProbabilityModel(),
        },
        artifact_path,
    )
    return artifact_path


def _write_thresholds_artifact(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "thresholds.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "m8_thresholds_v1",
                "run_id": "20260322T100000Z",
                "source_table": "feature_ohlc",
                "source_exchange": "kraken",
                "interval_minutes": 5,
                "required_inputs": [
                    "realized_vol_12",
                    "momentum_3",
                    "macd_line_12_26",
                ],
                "regime_labels": [
                    "TREND_UP",
                    "TREND_DOWN",
                    "RANGE",
                    "HIGH_VOL",
                ],
                "thresholds_by_symbol": {
                    "BTC/USD": {
                        "symbol": "BTC/USD",
                        "fitted_row_count": 10,
                        "high_vol_threshold": 0.05,
                        "trend_abs_threshold": 0.02,
                    },
                    "ETH/USD": {
                        "symbol": "ETH/USD",
                        "fitted_row_count": 10,
                        "high_vol_threshold": 0.06,
                        "trend_abs_threshold": 0.03,
                    },
                    "SOL/USD": {
                        "symbol": "SOL/USD",
                        "fitted_row_count": 10,
                        "high_vol_threshold": 0.07,
                        "trend_abs_threshold": 0.04,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def _write_signal_policy(tmp_path: Path) -> Path:
    policy_path = tmp_path / "regime_signal_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "m9_regime_signal_policy_v1",
                "policies": {
                    "TREND_UP": {
                        "buy_prob_up": 0.54,
                        "sell_prob_up": 0.44,
                        "allow_new_long_entries": True,
                    },
                    "TREND_DOWN": {
                        "buy_prob_up": 0.60,
                        "sell_prob_up": 0.46,
                        "allow_new_long_entries": False,
                    },
                    "RANGE": {
                        "buy_prob_up": 0.58,
                        "sell_prob_up": 0.42,
                        "allow_new_long_entries": True,
                    },
                    "HIGH_VOL": {
                        "buy_prob_up": 0.62,
                        "sell_prob_up": 0.48,
                        "allow_new_long_entries": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return policy_path


def _set_common_runtime_env(
    monkeypatch,
    *,
    tmp_path: Path,
    runtime_profile: str,
    trading_config_name: str,
) -> Path:
    report_path = tmp_path / "startup_report.json"
    monkeypatch.setenv("STREAMALPHA_RUNTIME_PROFILE", runtime_profile)
    monkeypatch.setenv("STREAMALPHA_STARTUP_REPORT_PATH", str(report_path))
    monkeypatch.setenv(
        "STREAMALPHA_TRADING_CONFIG_PATH",
        str(Path("configs") / trading_config_name),
    )
    return report_path


def test_runtime_profile_resolution_accepts_only_supported_profiles() -> None:
    assert resolve_runtime_profile("dev", default=None) == "dev"
    assert resolve_runtime_profile("paper", default=None) == "paper"
    try:
        resolve_runtime_profile("staging", default=None)
    except ValueError as error:
        assert "must be one of" in str(error)
    else:
        raise AssertionError("Unsupported runtime profile should fail validation")


def test_startup_validation_passes_for_paper_with_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = _set_common_runtime_env(
        monkeypatch,
        tmp_path=tmp_path,
        runtime_profile="paper",
        trading_config_name="paper_trading.paper.yaml",
    )
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(_write_model_artifact(tmp_path)))
    monkeypatch.setenv(
        "INFERENCE_REGIME_THRESHOLDS_PATH",
        str(_write_thresholds_artifact(tmp_path)),
    )
    monkeypatch.setenv(
        "INFERENCE_REGIME_SIGNAL_POLICY_PATH",
        str(_write_signal_policy(tmp_path)),
    )

    report = build_startup_validation_report()
    written_path = write_startup_validation_report(report)

    assert report.startup_validation_passed is True
    assert report.runtime_profile == "paper"
    assert report.execution_mode == "paper"
    assert report.model_version_source == "MODEL_OVERRIDE_PATH"
    assert written_path == report_path.resolve()


def test_startup_validation_passes_for_dev_without_trading_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Dev startup should not require a paper/shadow/live trading config path."""
    report_path = tmp_path / "startup_report.json"
    monkeypatch.setenv("STREAMALPHA_RUNTIME_PROFILE", "dev")
    monkeypatch.setenv("STREAMALPHA_STARTUP_REPORT_PATH", str(report_path))
    monkeypatch.delenv("STREAMALPHA_TRADING_CONFIG_PATH", raising=False)

    report = build_startup_validation_report()
    written_path = write_startup_validation_report(report)

    assert report.startup_validation_passed is True
    assert report.runtime_profile == "dev"
    assert report.trading_config_path is None
    assert written_path == report_path.resolve()


def test_startup_validation_fails_fast_when_required_artifacts_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_common_runtime_env(
        monkeypatch,
        tmp_path=tmp_path,
        runtime_profile="shadow",
        trading_config_name="paper_trading.shadow.yaml",
    )
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    monkeypatch.setenv(
        "INFERENCE_REGIME_THRESHOLDS_PATH",
        str(tmp_path / "missing-thresholds.json"),
    )
    monkeypatch.setenv(
        "INFERENCE_REGIME_SIGNAL_POLICY_PATH",
        str(tmp_path / "missing-policy.json"),
    )

    report = build_startup_validation_report()

    assert report.startup_validation_passed is False
    assert any("INFERENCE_MODEL_PATH does not exist" in error for error in report.errors)
    assert any(
        "Regime thresholds artifact does not exist" in error for error in report.errors
    )


def test_live_profile_requires_secrets_and_explicit_arming(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_common_runtime_env(
        monkeypatch,
        tmp_path=tmp_path,
        runtime_profile="live",
        trading_config_name="paper_trading.live.yaml",
    )
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(_write_model_artifact(tmp_path)))
    monkeypatch.setenv(
        "INFERENCE_REGIME_THRESHOLDS_PATH",
        str(_write_thresholds_artifact(tmp_path)),
    )
    monkeypatch.setenv(
        "INFERENCE_REGIME_SIGNAL_POLICY_PATH",
        str(_write_signal_policy(tmp_path)),
    )
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    monkeypatch.delenv("STREAMALPHA_ENABLE_LIVE", raising=False)
    monkeypatch.delenv("STREAMALPHA_LIVE_CONFIRM", raising=False)

    report = build_startup_validation_report()

    assert report.startup_validation_passed is False
    assert any("APCA_API_KEY_ID must be set" in error for error in report.errors)
    assert any("STREAMALPHA_ENABLE_LIVE must be true" in error for error in report.errors)
    assert any(
        "STREAMALPHA_LIVE_CONFIRM must match" in error for error in report.errors
    )


def test_startup_report_redacts_secret_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = _set_common_runtime_env(
        monkeypatch,
        tmp_path=tmp_path,
        runtime_profile="live",
        trading_config_name="paper_trading.live.yaml",
    )
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(_write_model_artifact(tmp_path)))
    monkeypatch.setenv(
        "INFERENCE_REGIME_THRESHOLDS_PATH",
        str(_write_thresholds_artifact(tmp_path)),
    )
    monkeypatch.setenv(
        "INFERENCE_REGIME_SIGNAL_POLICY_PATH",
        str(_write_signal_policy(tmp_path)),
    )
    monkeypatch.setenv("APCA_API_KEY_ID", "secret-key-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "super-secret-value")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("STREAMALPHA_ENABLE_LIVE", "true")
    monkeypatch.setenv(
        "STREAMALPHA_LIVE_CONFIRM",
        "I UNDERSTAND STREAM ALPHA LIVE TRADING IS ENABLED",
    )

    report = build_startup_validation_report()
    write_startup_validation_report(report)
    text = report_path.read_text(encoding="utf-8")

    assert "secret-key-id" not in text
    assert "super-secret-value" not in text


def test_trader_and_dashboard_honor_runtime_trading_config_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "custom-paper.yaml"
    config_path.write_text(
        "service_name: test\n"
        "source_exchange: kraken\n"
        "source_table: feature_ohlc\n"
        "interval_minutes: 5\n"
        "symbols:\n"
        "  - BTC/USD\n"
        "inference_base_url: http://127.0.0.1:8000\n"
        "poll_interval_seconds: 5\n"
        "artifact_dir: artifacts/test\n"
        "execution:\n"
        "  mode: paper\n"
        "  idempotency_key_version: 1\n"
        "risk:\n"
        "  initial_cash: 10000\n"
        "  position_fraction: 0.25\n"
        "  fee_bps: 20\n"
        "  slippage_bps: 5\n"
        "  stop_loss_pct: 0.02\n"
        "  take_profit_pct: 0.04\n"
        "  cooldown_candles: 1\n"
        "  max_open_positions: 1\n"
        "  max_exposure_per_asset: 0.25\n"
        "  max_total_exposure: 0.5\n"
        "  max_daily_loss_amount: 100\n"
        "  max_drawdown_pct: 0.2\n"
        "  loss_streak_limit: 3\n"
        "  loss_streak_cooldown_candles: 1\n"
        "  kill_switch_enabled: false\n"
        "  min_trade_notional: 10\n"
        "  volatility_target_realized_vol: 0.03\n"
        "  min_volatility_size_multiplier: 0.5\n"
        "  enable_confidence_weighted_sizing: false\n"
        "  min_confidence_size_multiplier: 0.5\n"
        "  regime_position_fraction_caps:\n"
        "    TREND_UP: 0.25\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STREAMALPHA_TRADING_CONFIG_PATH", str(config_path))

    assert resolve_trader_config_path("") == config_path.resolve()
    assert resolve_dashboard_trading_config_path() == config_path.resolve()
