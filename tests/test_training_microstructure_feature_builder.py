"""Tests for DU11 microstructure feature builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.training.microstructure_feature_builder import (
    build_features_from_replay,
    write_microstructure_features,
)
from app.training.microstructure_replay import ReplayRow


def test_feature_builder_computes_spread_from_replay() -> None:
    """Feature builder should compute top-of-book spread from replay rows."""
    rows = [
        ReplayRow(
            source_exchange="kraken",
            symbol="BTC/USD",
            event_time="2026-01-01T00:00:00+00:00",
            received_at="2026-01-01T00:00:01+00:00",
            replay_sequence=1,
            sequence_or_checksum="1",
            best_bid=100.0,
            best_ask=101.0,
            bid_level_count=1,
            ask_level_count=1,
            book_gap_flag=False,
        )
    ]

    features = build_features_from_replay(rows)

    assert features[0].mid_price == 100.5
    assert features[0].top_of_book_spread == 1.0
    assert features[0].relative_spread == pytest.approx(1.0 / 100.5)


def test_feature_builder_preserves_missing_spread() -> None:
    """Missing book side should not invent spread values."""
    rows = [
        ReplayRow("kraken", "BTC/USD", "t", "r", 1, "1", 100.0, None, 1, 0, False)
    ]

    features = build_features_from_replay(rows)

    assert features[0].top_of_book_spread is None
    assert features[0].relative_spread is None


def test_feature_builder_writes_artifacts(tmp_path: Path) -> None:
    """Feature builder should persist deterministic artifacts."""
    output_dir = tmp_path / "features"

    result = write_microstructure_features(repo_root=tmp_path, output_dir=output_dir)

    assert result["feature_build_status"] == "MICROSTRUCTURE_FEATURE_ROWS_BUILT_FROM_FIXTURES"
    assert result["feature_row_count"] == 2
    assert (output_dir / "microstructure_features.csv").exists()
    assert (output_dir / "feature_lineage.csv").exists()


def test_feature_builder_preserves_non_claims(tmp_path: Path) -> None:
    """Feature output should remain research-only and non-promotable."""
    output_dir = tmp_path / "features"

    write_microstructure_features(repo_root=tmp_path, output_dir=output_dir)

    report = json.loads((output_dir / "microstructure_features.json").read_text())
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    assert "NO_RUNTIME_EFFECT" in report["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in recommendation["honesty_flags"]
    assert recommendation["runtime_ready"] is False


def test_feature_builder_recommends_readiness_audit(tmp_path: Path) -> None:
    """After building features, readiness audit is the next step."""
    result = write_microstructure_features(repo_root=tmp_path, output_dir=tmp_path / "features")

    assert result["recommendation"] == "AUDIT_MICROSTRUCTURE_FEATURE_READINESS_FOR_ALPHA_RESEARCH"
    assert result["next_required_action"] == "RUN_MICROSTRUCTURE_RESEARCH_READINESS_AUDIT"
