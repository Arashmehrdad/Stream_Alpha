"""Focused tests for the local Kraken OHLCVT CSV importer."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from app.common.config import PostgresSettings
from app.common.models import OhlcEvent
from app.features.engine import MIN_FINALIZED_CANDLES
from app.features.state import FeatureStateManager
from app.ingestion import import_kraken_ohlcvt as import_module
from app.training.data_readiness import DataReadinessReport, SymbolReadinessSnapshot


def test_resolve_kraken_csv_files_selects_exact_5m_targets(tmp_path: Path) -> None:
    """Only the requested 5-minute BTC/ETH/SOL files should resolve."""
    for file_name in (
        "XBTUSD_5.csv",
        "ETHUSD_5.csv",
        "SOLUSD_5.csv",
        "ETHFIUSD_5.csv",
        "XBTUSD_15.csv",
        "._XBTUSD_5.csv",
    ):
        (tmp_path / file_name).write_text("1,1,1,1,1,1,1\n", encoding="utf-8")

    resolved = import_module.resolve_kraken_csv_files(
        tmp_path,
        symbols=("BTC/USD", "ETH/USD", "SOL/USD"),
        interval_minutes=5,
    )

    assert [file.symbol for file in resolved] == ["BTC/USD", "ETH/USD", "SOL/USD"]
    assert [file.path.name for file in resolved] == [
        "XBTUSD_5.csv",
        "ETHUSD_5.csv",
        "SOLUSD_5.csv",
    ]


def test_parse_kraken_csv_row_supports_downloadable_seven_column_schema(
    tmp_path: Path,
) -> None:
    """The downloadable CSV schema should parse without inventing extra columns."""
    parsed = import_module.parse_kraken_csv_row(
        ["1381095000", "122.0", "122.0", "122.0", "122.0", "0.1", "1"],
        interval_minutes=5,
        line_number=1,
        source_path=tmp_path / "XBTUSD_5.csv",
    )

    assert parsed.interval_begin == datetime(2013, 10, 6, 21, 30, tzinfo=timezone.utc)
    assert parsed.close_price == 122.0
    assert parsed.volume == 0.1
    assert parsed.trade_count == 1
    assert parsed.used_vwap_fallback is True
    assert parsed.vwap == parsed.close_price


class _FakeWriter:
    def __init__(self) -> None:
        self.written_events: list[OhlcEvent] = []

    async def connect(self) -> None:
        return

    async def close(self) -> None:
        return

    async def write_ohlc_batch(self, events) -> None:
        self.written_events.extend(events)


class _FakeStore:
    def __init__(
        self,
        *,
        raw_events: list[OhlcEvent],
        feature_rows,
    ) -> None:
        self.raw_events = list(raw_events)
        self.feature_rows = list(feature_rows)
        self.upserted_feature_rows = []

    async def connect(self) -> None:
        return

    async def close(self) -> None:
        return

    async def load_raw_candles(self, **kwargs):
        symbols = tuple(kwargs.get("symbols") or ())
        interval_minutes = kwargs.get("interval_minutes")
        start = kwargs.get("start")
        end = kwargs.get("end")
        rows = [
            event
            for event in self.raw_events
            if (not symbols or event.symbol in symbols)
            and event.interval_minutes == interval_minutes
            and (start is None or event.interval_begin >= start)
            and (end is None or event.interval_begin < end)
        ]
        return list(rows)

    async def load_feature_rows(self, **kwargs):
        symbols = tuple(kwargs.get("symbols") or ())
        interval_minutes = kwargs.get("interval_minutes")
        start = kwargs.get("start")
        end = kwargs.get("end")
        rows = [
            row
            for row in self.feature_rows
            if (not symbols or row.symbol in symbols)
            and row.interval_minutes == interval_minutes
            and (start is None or row.interval_begin >= start)
            and (end is None or row.interval_begin < end)
        ]
        return list(rows)

    async def upsert_feature_rows_batch(self, rows) -> None:
        self.upserted_feature_rows.extend(rows)


def _build_raw_events(symbol: str, *, count: int) -> list[OhlcEvent]:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    events: list[OhlcEvent] = []
    for index in range(count):
        interval_begin = start + timedelta(minutes=index * 5)
        close_price = 100.0 + index
        events.append(
            OhlcEvent(
                event_id=f"{symbol}-{index}",
                app_name="streamalpha",
                source_exchange="kraken",
                channel="ohlc",
                message_type="historical_csv_import",
                symbol=symbol,
                interval_minutes=5,
                interval_begin=interval_begin,
                interval_end=interval_begin + timedelta(minutes=5),
                open_price=close_price - 0.5,
                high_price=close_price + 0.5,
                low_price=close_price - 1.0,
                close_price=close_price,
                vwap=close_price,
                trade_count=10 + index,
                volume=1.0 + index,
                received_at=interval_begin + timedelta(minutes=5),
            )
        )
    return events


def _build_feature_rows(raw_events: list[OhlcEvent]):
    state = FeatureStateManager(grace_seconds=0, history_limit=MIN_FINALIZED_CANDLES + 16)
    rebuilt_rows = state.bootstrap(
        raw_events,
        now=raw_events[-1].interval_end + timedelta(minutes=5),
        computed_at=raw_events[-1].interval_end + timedelta(minutes=5),
    )
    return [import_module._normalize_feature_row(row) for row in rebuilt_rows]  # pylint: disable=protected-access


def test_sync_symbol_csv_file_skips_unchanged_reruns(tmp_path: Path) -> None:
    """Rerunning the same CSV import should not rewrite identical raw candles."""
    csv_path = tmp_path / "XBTUSD_5.csv"
    csv_path.write_text(
        "\n".join(
            (
                "1381095000,122.0,122.0,122.0,122.0,0.1,1",
                "1381095300,123.0,123.0,123.0,123.0,0.2,2",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    csv_file = import_module.KrakenCsvFile(
        symbol="BTC/USD",
        pair_code="XBTUSD",
        interval_minutes=5,
        path=csv_path,
    )
    writer = _FakeWriter()
    first_store = _FakeStore(raw_events=[], feature_rows=[])

    first_stats = asyncio.run(
        import_module._sync_symbol_csv_file(  # pylint: disable=protected-access
            writer=writer,
            store=first_store,
            csv_file=csv_file,
            app_name="streamalpha",
            import_window=import_module.ImportWindow(start=None, end=None),
            batch_size=100,
        )
    )
    second_store = _FakeStore(raw_events=writer.written_events, feature_rows=[])

    second_stats = asyncio.run(
        import_module._sync_symbol_csv_file(  # pylint: disable=protected-access
            writer=_FakeWriter(),
            store=second_store,
            csv_file=csv_file,
            app_name="streamalpha",
            import_window=import_module.ImportWindow(start=None, end=None),
            batch_size=100,
        )
    )

    assert first_stats.created_rows == 2
    assert first_stats.unchanged_rows == 0
    assert second_stats.created_rows == 0
    assert second_stats.updated_rows == 0
    assert second_stats.unchanged_rows == 2


def test_replay_symbol_feature_rows_is_idempotent() -> None:
    """Replaying the same imported raw history should not rewrite identical features."""
    raw_events = _build_raw_events("BTC/USD", count=MIN_FINALIZED_CANDLES + 6)
    rebuilt_rows = _build_feature_rows(raw_events)

    first_store = _FakeStore(raw_events=raw_events, feature_rows=[])
    first_stats = asyncio.run(
        import_module._replay_symbol_feature_rows(  # pylint: disable=protected-access
            store=first_store,
            symbol="BTC/USD",
            interval_minutes=5,
            history_limit=MIN_FINALIZED_CANDLES + 8,
            import_window=import_module.ImportWindow(start=None, end=None),
            batch_size=100,
        )
    )
    second_store = _FakeStore(raw_events=raw_events, feature_rows=first_store.upserted_feature_rows)
    second_stats = asyncio.run(
        import_module._replay_symbol_feature_rows(  # pylint: disable=protected-access
            store=second_store,
            symbol="BTC/USD",
            interval_minutes=5,
            history_limit=MIN_FINALIZED_CANDLES + 8,
            import_window=import_module.ImportWindow(start=None, end=None),
            batch_size=100,
        )
    )

    assert first_stats.created_rows == len(first_store.upserted_feature_rows)
    assert first_stats.updated_rows == 0
    assert second_stats.created_rows == 0
    assert second_stats.updated_rows == 0
    assert second_stats.unchanged_rows == first_stats.generated_rows


def test_run_import_replay_only_writes_import_operation_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Replay-only mode should still write the readiness bundle and import operation summary."""
    raw_events = _build_raw_events("BTC/USD", count=MIN_FINALIZED_CANDLES + 6)
    fake_store = _FakeStore(raw_events=raw_events, feature_rows=[])
    fake_writer = _FakeWriter()
    artifact_root = tmp_path / "artifacts"

    async def _fake_resolve_postgres_dsn(postgres) -> str:
        return "postgresql://test"

    monkeypatch.setattr(import_module, "_resolve_postgres_dsn", _fake_resolve_postgres_dsn)
    monkeypatch.setattr(import_module, "PostgresWriter", lambda dsn, tables: fake_writer)
    monkeypatch.setattr(import_module, "FeatureStore", lambda dsn, tables: fake_store)

    report = DataReadinessReport(
        config_path=str(tmp_path / "training.m7.json"),
        generated_at=datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc),
        artifact_root="artifacts/training/m7",
        source_table="feature_ohlc",
        raw_table="raw_ohlc",
        feature_table="feature_ohlc",
        source_exchange="kraken",
        interval_minutes=5,
        postgres_reachable=True,
        postgres_error=None,
        raw_table_exists=True,
        feature_table_exists=True,
        raw_rows_total=100,
        feature_rows_total=90,
        labeled_rows_total=80,
        unique_timestamps=40,
        required_unique_timestamps=9,
        ready_for_training=True,
        readiness_detail="feature_ohlc satisfies the configured walk-forward timestamp requirement.",
        earliest_usable_timestamp=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
        latest_usable_timestamp=datetime(2026, 4, 2, 0, 0, tzinfo=timezone.utc),
        overall_positive_label_rate=0.5,
        label_counts={"0": 40, "1": 40},
        regime_distribution_available=False,
        regime_distribution_detail="unavailable in test",
        regime_distribution={},
        opportunity_density={"overall": {}, "per_symbol": {}},
        warnings=(),
        symbol_summaries=(
            SymbolReadinessSnapshot(
                symbol="BTC/USD",
                raw_row_count=100,
                feature_row_count=90,
                labeled_row_count=80,
                raw_earliest_interval_begin=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
                raw_latest_interval_begin=datetime(2026, 4, 2, 0, 0, tzinfo=timezone.utc),
                feature_earliest_interval_begin=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
                feature_latest_interval_begin=datetime(2026, 4, 2, 0, 0, tzinfo=timezone.utc),
                labeled_earliest_as_of_time=datetime(2026, 4, 1, 0, 5, tzinfo=timezone.utc),
                labeled_latest_as_of_time=datetime(2026, 4, 2, 0, 5, tzinfo=timezone.utc),
                positive_label_rate=0.5,
                raw_missing_interval_count=0,
                feature_missing_interval_count=0,
                raw_gap_windows=(),
                feature_gap_windows=(),
                regime_distribution={},
            ),
        ),
    )
    monkeypatch.setattr(import_module, "build_data_readiness_report_from_path", lambda path: report)

    settings = SimpleNamespace(
        app_name="streamalpha",
        kraken=SimpleNamespace(symbols=("BTC/USD", "ETH/USD", "SOL/USD"), ohlc_interval_minutes=5),
        features=SimpleNamespace(bootstrap_candles=MIN_FINALIZED_CANDLES + 8),
        postgres=PostgresSettings(
            host="127.0.0.1",
            port=5432,
            database="streamalpha",
            user="streamalpha",
            password="change-me-local-only",
        ),
        tables=SimpleNamespace(raw_ohlc="raw_ohlc", feature_ohlc="feature_ohlc"),
    )
    arguments = argparse.Namespace(
        dataset_root=str(tmp_path / "master_q4"),
        symbols=("BTC/USD",),
        interval=5,
        start=None,
        end=None,
        batch_size=100,
        skip_raw_import=True,
        skip_feature_replay=False,
        report_only=False,
        training_config=str(tmp_path / "training.m7.json"),
        report_artifact_root=str(artifact_root),
        json=True,
    )

    summary = asyncio.run(import_module._run_import(arguments, settings))  # pylint: disable=protected-access
    report_dir = Path(summary["readiness_report_artifact_dir"])

    assert summary["skip_raw_import"] is True
    assert summary["feature_replay"]["generated_rows"] > 0
    assert (report_dir / "readiness_report.json").is_file()
    assert (report_dir / "summary.md").is_file()
    assert (report_dir / "import_operation.json").is_file()
    operation_payload = json.loads((report_dir / "import_operation.json").read_text(encoding="utf-8"))
    assert operation_payload["skip_raw_import"] is True
    assert operation_payload["feature_replay"]["generated_rows"] > 0
