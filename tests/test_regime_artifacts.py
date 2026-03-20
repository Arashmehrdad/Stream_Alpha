"""Tests for the M8 regime artifact helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.regime.artifacts import (
    load_thresholds_artifact,
    required_run_artifact_paths,
    write_csv,
    write_json_atomic,
    write_thresholds_artifact,
)
from app.regime.config import RegimeConfig, ThresholdConfig
from app.regime.service import SymbolThresholds, build_thresholds_payload


def _make_config() -> RegimeConfig:
    return RegimeConfig(
        source_table="feature_ohlc",
        source_exchange="kraken",
        interval_minutes=5,
        symbols=("BTC/USD", "ETH/USD"),
        artifact_dir="artifacts/regime/m8",
        min_rows_per_symbol=5,
        thresholds=ThresholdConfig(
            high_vol_percentile=75.0,
            trend_abs_momentum_percentile=60.0,
        ),
    )


def test_thresholds_artifact_roundtrip_preserves_serving_payload(tmp_path: Path) -> None:
    """Threshold artifacts should save and reload without losing serving metadata."""
    config = _make_config()
    payload = build_thresholds_payload(
        run_id="20260320T120000Z",
        created_at=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        config=config,
        thresholds_by_symbol={
            "BTC/USD": SymbolThresholds(
                symbol="BTC/USD",
                fitted_row_count=5,
                high_vol_threshold=4.0,
                trend_abs_threshold=3.4,
            ),
            "ETH/USD": SymbolThresholds(
                symbol="ETH/USD",
                fitted_row_count=5,
                high_vol_threshold=40.0,
                trend_abs_threshold=8.8,
            ),
        },
    )
    path = tmp_path / "thresholds.json"

    write_thresholds_artifact(path, payload)
    reloaded = load_thresholds_artifact(path)

    assert reloaded == payload


def test_required_run_artifact_paths_accepts_complete_regime_run_directory(
    tmp_path: Path,
) -> None:
    """The manifest helper should validate the explicit M8 artifact set."""
    write_json_atomic(tmp_path / "thresholds.json", {"ok": True})
    write_csv(
        tmp_path / "regime_predictions.csv",
        [{"symbol": "BTC/USD", "regime": "RANGE"}],
    )
    write_csv(
        tmp_path / "by_symbol_summary.csv",
        [{"symbol": "BTC/USD", "predicted_row_count": 1}],
    )
    write_json_atomic(tmp_path / "overall_summary.json", {"ok": True})
    write_json_atomic(tmp_path / "run_config.json", {"ok": True})
    write_json_atomic(tmp_path / "run_manifest.json", {"ok": True})

    artifact_paths = required_run_artifact_paths(tmp_path)

    assert sorted(artifact_paths) == [
        "by_symbol_summary.csv",
        "overall_summary.json",
        "regime_predictions.csv",
        "run_config.json",
        "run_manifest.json",
        "thresholds.json",
    ]
