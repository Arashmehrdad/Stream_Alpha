"""Tests for research-only Kraken book payload fixture normalizers."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.training.market_microstructure_book_normalizers import (
    BookPayloadNormalizationError,
    normalize_kraken_book_payload_fixture,
    planned_kraken_book_subscription,
    write_market_microstructure_book_normalizer_contract,
)


def test_snapshot_fixture_normalizes_best_bid_and_ask() -> None:
    """A snapshot fixture should normalize levels and expose best prices."""
    event = normalize_kraken_book_payload_fixture(
        _snapshot_payload(),
        received_at=_received_at(),
    )

    assert event.source_exchange == "kraken"
    assert event.channel == "book"
    assert event.message_type == "snapshot"
    assert event.symbol == "MATIC/USD"
    assert event.checksum == 2439117997
    assert event.sequence_or_checksum == "2439117997"
    assert event.best_bid == 0.5666
    assert event.best_ask == 0.5668
    assert len(event.bids) == 2
    assert len(event.asks) == 2


def test_update_fixture_allows_empty_side() -> None:
    """A book update may contain an empty side and still normalize clearly."""
    event = normalize_kraken_book_payload_fixture(
        _update_payload(),
        received_at=_received_at(),
    )

    assert event.message_type == "update"
    assert len(event.bids) == 1
    assert len(event.asks) == 0
    assert event.best_ask is None


def test_invalid_payloads_fail_clearly() -> None:
    """Unsupported or incomplete fixtures should raise operator-readable errors."""
    with pytest.raises(BookPayloadNormalizationError, match="Unsupported channel"):
        normalize_kraken_book_payload_fixture(
            {"channel": "trade", "type": "snapshot", "data": []},
            received_at=_received_at(),
        )
    with pytest.raises(BookPayloadNormalizationError, match="Missing required field: data"):
        normalize_kraken_book_payload_fixture(
            {"channel": "book", "type": "snapshot"},
            received_at=_received_at(),
        )


def test_planned_subscription_contract_is_not_runtime_wired() -> None:
    """The planned subscription payload should be explicit and depth-validated."""
    payload = planned_kraken_book_subscription(("BTC/USD",), depth=10)

    assert payload["method"] == "subscribe"
    assert payload["params"]["channel"] == "book"
    assert payload["params"]["depth"] == 10
    with pytest.raises(ValueError, match="Kraken book depth"):
        planned_kraken_book_subscription(("BTC/USD",), depth=20)


def test_contract_writer_persists_required_artifacts(tmp_path: Path) -> None:
    """The fixture normalizer contract should write deterministic artifacts."""
    output_dir = tmp_path / "book_contract"

    result = write_market_microstructure_book_normalizer_contract(
        repo_root=tmp_path,
        output_dir=output_dir,
    )

    assert result["normalizer_status"] == "SAMPLE_BOOK_PAYLOAD_NORMALIZERS_DEFINED"
    for filename in (
        "manifest.json",
        "book_payload_normalizer_contract.json",
        "book_payload_normalizer_contract.md",
        "sample_payloads.json",
        "normalized_sample_events.csv",
        "parser_contract.csv",
        "validation_cases.csv",
        "leakage_boundary_audit.csv",
        "blocked_decisions.csv",
        "next_actions.csv",
        "recommendation.json",
    ):
        assert (output_dir / filename).exists()


def test_contract_outputs_preserve_research_only_non_claims(tmp_path: Path) -> None:
    """Outputs must not claim runtime, promotion, backtest, or profitability readiness."""
    output_dir = tmp_path / "book_contract"

    write_market_microstructure_book_normalizer_contract(
        repo_root=tmp_path,
        output_dir=output_dir,
    )

    report = json.loads((output_dir / "book_payload_normalizer_contract.json").read_text())
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


def test_contract_writer_recommends_feature_derivation_next(tmp_path: Path) -> None:
    """After fixture normalizers, the next safe batch is feature derivation."""
    output_dir = tmp_path / "book_contract"

    result = write_market_microstructure_book_normalizer_contract(
        repo_root=tmp_path,
        output_dir=output_dir,
    )

    assert result["recommendation"] == (
        "DERIVE_RESEARCH_ONLY_MICROSTRUCTURE_FEATURES_FROM_CONTRACTS"
    )
    assert result["next_required_action"] == (
        "BUILD_RESEARCH_ONLY_MICROSTRUCTURE_FEATURE_DERIVATION"
    )
    blocked = _read_csv(output_dir / "blocked_decisions.csv")
    assert {row["decision"] for row in blocked} == {
        "book_checksum_validation",
        "capture_service",
    }


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
