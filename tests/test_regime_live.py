"""Tests for the M9 live regime runtime loader and exact-row resolver."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.regime.live import (
    load_live_regime_runtime,
    load_thresholds_artifact,
    resolve_thresholds_artifact_path,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _threshold_payload(*, schema_version: str = "m8_thresholds_v1") -> dict:
    return {
        "schema_version": schema_version,
        "run_id": "20260320T120000Z",
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
                "fitted_row_count": 100,
                "high_vol_threshold": 0.05,
                "trend_abs_threshold": 0.02,
            },
            "ETH/USD": {
                "symbol": "ETH/USD",
                "fitted_row_count": 100,
                "high_vol_threshold": 0.06,
                "trend_abs_threshold": 0.03,
            },
            "SOL/USD": {
                "symbol": "SOL/USD",
                "fitted_row_count": 100,
                "high_vol_threshold": 0.07,
                "trend_abs_threshold": 0.04,
            },
        },
    }


def _policy_payload() -> dict:
    return {
        "schema_version": "m9_regime_signal_policy_v1",
        "policies": {
            "TREND_UP": {
                "buy_prob_up": 0.54,
                "sell_prob_up": 0.44,
                "allow_new_long_entries": True,
            },
            "RANGE": {
                "buy_prob_up": 0.58,
                "sell_prob_up": 0.42,
                "allow_new_long_entries": True,
            },
            "TREND_DOWN": {
                "buy_prob_up": 0.60,
                "sell_prob_up": 0.46,
                "allow_new_long_entries": False,
            },
            "HIGH_VOL": {
                "buy_prob_up": 0.62,
                "sell_prob_up": 0.48,
                "allow_new_long_entries": False,
            },
        },
    }


def test_resolve_thresholds_artifact_path_picks_latest_run(tmp_path: Path) -> None:
    """When no explicit path is set, the newest M8 thresholds artifact should win."""
    root = tmp_path / "regime-live"
    _write_json(root / "20260319T120000Z" / "thresholds.json", _threshold_payload())
    expected = _write_json(root / "20260320T120000Z" / "thresholds.json", _threshold_payload())

    resolved = resolve_thresholds_artifact_path("", thresholds_root=root)

    assert resolved == expected.resolve()


def test_load_live_regime_runtime_resolves_exact_row_regime(tmp_path: Path) -> None:
    """The live runtime should reuse the saved M8 thresholds for exact-row classification."""
    root = tmp_path / "regime-live-runtime"
    thresholds_path = _write_json(root / "thresholds.json", _threshold_payload())
    policy_path = _write_json(root / "regime_signal_policy.json", _policy_payload())

    runtime = load_live_regime_runtime(
        thresholds_path=str(thresholds_path),
        signal_policy_path=str(policy_path),
    )
    runtime.validate_runtime_compatibility(
        source_table="feature_ohlc",
        source_exchange="kraken",
        interval_minutes=5,
        symbols=("BTC/USD", "ETH/USD", "SOL/USD"),
    )

    resolved = runtime.resolve_feature_row_regime(
        {
            "symbol": "BTC/USD",
            "interval_begin": datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
            "as_of_time": datetime(2026, 3, 20, 12, 5, tzinfo=timezone.utc),
            "realized_vol_12": 0.08,
            "momentum_3": 0.03,
            "macd_line_12_26": 1.2,
        }
    )

    assert resolved.regime_label == "HIGH_VOL"
    assert resolved.regime_run_id == "20260320T120000Z"
    assert resolved.high_vol_threshold == 0.05
    assert runtime.policy_for("HIGH_VOL").allow_new_long_entries is False


def test_load_thresholds_artifact_rejects_bad_schema_version(tmp_path: Path) -> None:
    """Unsupported threshold artifact schema versions should fail clearly."""
    artifact_path = _write_json(
        tmp_path / "thresholds.json",
        _threshold_payload(schema_version="m8_thresholds_v0"),
    )

    with pytest.raises(ValueError, match="schema_version is not supported"):
        load_thresholds_artifact(str(artifact_path))


def test_runtime_compatibility_rejects_mismatched_source_table(tmp_path: Path) -> None:
    """Live inference must stop if the loaded thresholds were fit from another source."""
    thresholds_path = _write_json(tmp_path / "thresholds.json", _threshold_payload())
    policy_path = _write_json(tmp_path / "regime_signal_policy.json", _policy_payload())
    runtime = load_live_regime_runtime(
        thresholds_path=str(thresholds_path),
        signal_policy_path=str(policy_path),
    )

    with pytest.raises(ValueError, match="source_table does not match the live runtime"):
        runtime.validate_runtime_compatibility(
            source_table="another_table",
            source_exchange="kraken",
            interval_minutes=5,
            symbols=("BTC/USD", "ETH/USD", "SOL/USD"),
        )
