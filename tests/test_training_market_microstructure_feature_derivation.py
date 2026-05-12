"""Tests for research-only microstructure feature derivation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.training.market_microstructure_book_normalizers import (
    normalize_kraken_book_payload_fixture,
)
from app.training.market_microstructure_feature_derivation import (
    derive_book_microstructure_features,
    write_microstructure_feature_derivation,
)


def test_book_feature_derivation_computes_spread_and_liquidity() -> None:
    """Feature derivation should compute deterministic book metrics."""
    event = normalize_kraken_book_payload_fixture(
        _snapshot_payload(),
        received_at=_received_at(),
    )

    row = derive_book_microstructure_features(event, depth_levels=2)

    assert row.best_bid == 0.5666
    assert row.best_ask == 0.5668
    assert row.mid_price == pytest.approx(0.5667)
    assert row.top_of_book_spread == pytest.approx(0.0002)
    assert row.relative_spread == pytest.approx(0.0002 / 0.5667)
    assert row.bid_depth_liquidity == pytest.approx(11489.98231095)
    assert row.ask_depth_liquidity == pytest.approx(9066.20182228)
    assert row.total_depth_liquidity == pytest.approx(20556.18413323)
    assert row.order_book_imbalance == pytest.approx(
        (11489.98231095 - 9066.20182228) / 20556.18413323
    )


def test_missing_book_side_leaves_spread_blank_but_keeps_depth() -> None:
    """One-sided updates should not invent spread or ask liquidity."""
    event = normalize_kraken_book_payload_fixture(
        _update_payload(),
        received_at=_received_at(),
    )

    row = derive_book_microstructure_features(event, depth_levels=2)

    assert row.best_bid == 0.5657
    assert row.best_ask is None
    assert row.top_of_book_spread is None
    assert row.relative_spread is None
    assert row.bid_depth_liquidity == pytest.approx(1098.3947558)
    assert row.ask_depth_liquidity == 0.0
    assert row.order_book_imbalance == 1.0


def test_depth_levels_must_be_positive() -> None:
    """Invalid derivation parameters should fail clearly."""
    event = normalize_kraken_book_payload_fixture(
        _snapshot_payload(),
        received_at=_received_at(),
    )

    with pytest.raises(ValueError, match="depth_levels must be positive"):
        derive_book_microstructure_features(event, depth_levels=0)


def test_feature_derivation_writer_persists_required_artifacts(tmp_path: Path) -> None:
    """The DU4 writer should persist deterministic feature artifacts."""
    output_dir = tmp_path / "feature_derivation"

    result = write_microstructure_feature_derivation(
        repo_root=tmp_path,
        output_dir=output_dir,
    )

    assert result["feature_derivation_status"] == (
        "RESEARCH_ONLY_FEATURE_DERIVATION_DEFINED"
    )
    for filename in (
        "manifest.json",
        "microstructure_feature_derivation.json",
        "microstructure_feature_derivation.md",
        "derived_feature_samples.csv",
        "feature_contract.csv",
        "derivation_rules.csv",
        "leakage_boundary_audit.csv",
        "blocked_decisions.csv",
        "next_actions.csv",
        "recommendation.json",
    ):
        assert (output_dir / filename).exists()


def test_feature_derivation_outputs_preserve_non_claims(tmp_path: Path) -> None:
    """Outputs should remain research-only and non-promotable."""
    output_dir = tmp_path / "feature_derivation"

    write_microstructure_feature_derivation(repo_root=tmp_path, output_dir=output_dir)

    report = json.loads((output_dir / "microstructure_feature_derivation.json").read_text())
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    for flag in (
        "RESEARCH_ONLY",
        "NO_RUNTIME_EFFECT",
        "NOT_BACKTEST",
        "NOT_RUNTIME_READY",
        "NOT_PROMOTABLE",
        "NO_PROFIT_CLAIM",
    ):
        assert flag in report["honesty_flags"]
        assert flag in recommendation["honesty_flags"]
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def test_feature_derivation_recommends_replay_audit_next(tmp_path: Path) -> None:
    """After DU4, the next safe batch should be coverage/replay audit."""
    output_dir = tmp_path / "feature_derivation"

    result = write_microstructure_feature_derivation(
        repo_root=tmp_path,
        output_dir=output_dir,
    )

    assert result["recommendation"] == (
        "ADD_COVERAGE_GAP_AND_REPLAY_DETERMINISM_REPORTS"
    )
    assert result["next_required_action"] == (
        "BUILD_MICROSTRUCTURE_COVERAGE_GAP_REPLAY_AUDIT"
    )
    blocked = _read_csv(output_dir / "blocked_decisions.csv")
    assert "stored_replay_coverage" in {row["decision"] for row in blocked}


def _snapshot_payload() -> dict[str, object]:
    return {
        "channel": "book",
        "type": "snapshot",
        "data": [
            {
                "symbol": "MATIC/USD",
                "bids": [
                    {"price": 0.5666, "qty": 4831.75496356},
                    {"price": 0.5665, "qty": 6658.22734739},
                ],
                "asks": [
                    {"price": 0.5668, "qty": 4410.79769741},
                    {"price": 0.5669, "qty": 4655.40412487},
                ],
                "checksum": 2439117997,
                "timestamp": "2023-10-06T17:35:55.440295Z",
            }
        ],
    }


def _update_payload() -> dict[str, object]:
    return {
        "channel": "book",
        "type": "update",
        "data": [
            {
                "symbol": "MATIC/USD",
                "bids": [{"price": 0.5657, "qty": 1098.3947558}],
                "asks": [],
                "checksum": 2114181697,
                "timestamp": "2023-10-06T17:35:55.440295Z",
            }
        ],
    }


def _received_at() -> datetime:
    return datetime(2023, 10, 6, 17, 35, 56, tzinfo=timezone.utc)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
