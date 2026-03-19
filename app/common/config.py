"""Centralized environment-backed configuration for Stream Alpha."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple
from urllib.parse import quote_plus


def _get_required(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)).strip())


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)).strip())


def _split_csv(name: str, default: str) -> Tuple[str, ...]:
    raw_value = _get_required(name, default)
    values = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    if not values:
        raise ValueError(f"Environment variable {name} must contain at least one value")
    return values


@dataclass(frozen=True, slots=True)
class KrakenSettings:
    """Kraken public market-data connection settings."""

    ws_url: str
    symbols: Tuple[str, ...]
    ohlc_interval_minutes: int


@dataclass(frozen=True, slots=True)
class KafkaSettings:
    """Kafka-compatible broker connection settings."""

    bootstrap_servers: str
    client_id: str


@dataclass(frozen=True, slots=True)
class PostgresSettings:
    """PostgreSQL connection settings."""

    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def dsn(self) -> str:
        """Build a DSN string for async PostgreSQL clients."""
        user = quote_plus(self.user)
        password = quote_plus(self.password)
        return (
            f"postgresql://{user}:{password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True, slots=True)
class TopicSettings:
    """Topic names for normalized event publication."""

    raw_trades: str
    raw_ohlc: str
    raw_health: str


@dataclass(frozen=True, slots=True)
class TableSettings:
    """Database table names for essential persistence."""

    raw_trades: str
    raw_ohlc: str
    producer_heartbeat: str


@dataclass(frozen=True, slots=True)
class RetrySettings:
    """Reconnect and retry behavior for long-running services."""

    initial_delay_seconds: float
    max_delay_seconds: float
    multiplier: float
    jitter_seconds: float


@dataclass(frozen=True, slots=True)
class Settings:  # pylint: disable=too-many-instance-attributes
    """Application-wide settings assembled from environment variables."""

    app_name: str
    log_level: str
    service_name: str
    heartbeat_interval_seconds: int
    kraken: KrakenSettings
    kafka: KafkaSettings
    postgres: PostgresSettings
    topics: TopicSettings
    tables: TableSettings
    retry: RetrySettings

    @classmethod
    def from_env(cls) -> "Settings":
        """Load application settings from environment variables."""
        return cls(
            app_name=_get_required("APP_NAME", "streamalpha"),
            log_level=_get_required("LOG_LEVEL", "INFO").upper(),
            service_name=_get_required("PRODUCER_SERVICE_NAME", "producer"),
            heartbeat_interval_seconds=_get_int("PRODUCER_HEARTBEAT_INTERVAL_SECONDS", 15),
            kraken=KrakenSettings(
                ws_url=_get_required("KRAKEN_WS_URL", "wss://ws.kraken.com/v2"),
                symbols=_split_csv("KRAKEN_SYMBOLS", "BTC/USD,ETH/USD,SOL/USD"),
                ohlc_interval_minutes=_get_int("KRAKEN_OHLC_INTERVAL_MINUTES", 5),
            ),
            kafka=KafkaSettings(
                bootstrap_servers=_get_required("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092"),
                client_id=_get_required("KAFKA_CLIENT_ID", "streamalpha-producer"),
            ),
            postgres=PostgresSettings(
                host=_get_required("POSTGRES_HOST", "postgres"),
                port=_get_int("POSTGRES_PORT", 5432),
                database=_get_required("POSTGRES_DB", "streamalpha"),
                user=_get_required("POSTGRES_USER", "streamalpha"),
                password=_get_required("POSTGRES_PASSWORD", "change-me-local-only"),
            ),
            topics=TopicSettings(
                raw_trades=_get_required("TOPIC_RAW_TRADES", "raw.trades"),
                raw_ohlc=_get_required("TOPIC_RAW_OHLC", "raw.ohlc"),
                raw_health=_get_required("TOPIC_RAW_HEALTH", "raw.health"),
            ),
            tables=TableSettings(
                raw_trades=_get_required("TABLE_RAW_TRADES", "raw_trades"),
                raw_ohlc=_get_required("TABLE_RAW_OHLC", "raw_ohlc"),
                producer_heartbeat=_get_required("TABLE_PRODUCER_HEARTBEAT", "producer_heartbeat"),
            ),
            retry=RetrySettings(
                initial_delay_seconds=_get_float("RECONNECT_INITIAL_DELAY_SECONDS", 1.0),
                max_delay_seconds=_get_float("RECONNECT_MAX_DELAY_SECONDS", 30.0),
                multiplier=_get_float("RECONNECT_BACKOFF_MULTIPLIER", 2.0),
                jitter_seconds=_get_float("RECONNECT_JITTER_SECONDS", 0.5),
            ),
        )
