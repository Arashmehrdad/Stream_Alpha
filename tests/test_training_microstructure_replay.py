"""Tests for DU10 fixture microstructure replay."""

from __future__ import annotations

import json
from pathlib import Path

from app.common.time import parse_rfc3339
from app.training.market_microstructure_book_normalizers import (
    normalize_kraken_book_payload_fixture,
)
from app.training.microstructure_replay import replay_book_events, write_microstructure_replay


def test_replay_orders_and_applies_book_updates() -> None:
    """Replay should sort events and apply updates forward."""
    received_at = parse_rfc3339("2023-10-06T17:35:56.000000Z")
    events = [
        normalize_kraken_book_payload_fixture(_update_payload(), received_at=received_at),
        normalize_kraken_book_payload_fixture(_snapshot_payload(), received_at=received_at),
    ]

    rows = replay_book_events(events)

    assert [row.replay_sequence for row in rows] == [1, 2]
    assert rows[0].best_bid == 0.5666
    assert rows[1].best_bid == 0.5667
    assert rows[1].best_ask == 0.5668


def test_replay_is_deterministic() -> None:
    """Same input should produce identical replay rows."""
    received_at = parse_rfc3339("2023-10-06T17:35:56.000000Z")
    events = [
        normalize_kraken_book_payload_fixture(_snapshot_payload(), received_at=received_at),
        normalize_kraken_book_payload_fixture(_update_payload(), received_at=received_at),
    ]

    assert replay_book_events(events) == replay_book_events(events)


def test_replay_writer_preserves_non_claims(tmp_path: Path) -> None:
    """Replay artifacts should remain research-only."""
    output_dir = tmp_path / "replay"

    result = write_microstructure_replay(repo_root=tmp_path, output_dir=output_dir)

    assert result["replay_status"] == "FIXTURE_REPLAY_DETERMINISTIC"
    assert result["gap_count"] == 0
    report = json.loads((output_dir / "microstructure_replay.json").read_text())
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    assert "NO_RUNTIME_EFFECT" in report["honesty_flags"]
    assert recommendation["runtime_ready"] is False


def test_replay_writer_recommends_feature_builder(tmp_path: Path) -> None:
    """Replay output should route to feature builder next."""
    result = write_microstructure_replay(repo_root=tmp_path, output_dir=tmp_path / "replay")

    assert result["recommendation"] == "BUILD_MICROSTRUCTURE_FEATURE_ROWS_FROM_REPLAY"
    assert result["next_required_action"] == "IMPLEMENT_MICROSTRUCTURE_FEATURE_BUILDER"


def _snapshot_payload() -> dict[str, object]:
    return {
        "channel": "book",
        "type": "snapshot",
        "data": [
            {
                "symbol": "MATIC/USD",
                "bids": [{"price": 0.5666, "qty": 2.0}],
                "asks": [{"price": 0.5668, "qty": 3.0}],
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
                "bids": [{"price": 0.5667, "qty": 1.0}],
                "asks": [],
                "checksum": 2114181697,
                "timestamp": "2023-10-06T17:35:56.440295Z",
            }
        ],
    }
