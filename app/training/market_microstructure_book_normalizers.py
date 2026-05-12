"""Research-only Kraken book payload fixture normalizers and contract writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.common.time import parse_rfc3339
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/book_payload_normalizer_contract"
M20_DECISION = "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "FIXTURE_ONLY",
    "NO_CAPTURE_SERVICE",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)


class BookPayloadNormalizationError(ValueError):
    """Raised when a sample Kraken book payload cannot be normalized."""


@dataclass(frozen=True, slots=True)
class ResearchBookLevel:
    """One normalized book level from a research fixture payload."""

    price: float
    quantity: float


@dataclass(frozen=True, slots=True)
class ResearchBookEvent:  # pylint: disable=too-many-instance-attributes
    """Normalized research-only Kraken book event."""

    source_exchange: str
    channel: str
    message_type: str
    symbol: str
    event_time: datetime
    received_at: datetime
    checksum: int
    sequence_or_checksum: str
    update_type: str
    bids: tuple[ResearchBookLevel, ...]
    asks: tuple[ResearchBookLevel, ...]
    payload: dict[str, Any]

    @property
    def best_bid(self) -> float | None:
        """Return the highest bid price if available."""
        if not self.bids:
            return None
        return max(level.price for level in self.bids)

    @property
    def best_ask(self) -> float | None:
        """Return the lowest ask price if available."""
        if not self.asks:
            return None
        return min(level.price for level in self.asks)

    def to_csv_row(self) -> dict[str, str]:
        """Return a deterministic compact row for artifact diagnostics."""
        return {
            "source_exchange": self.source_exchange,
            "channel": self.channel,
            "message_type": self.message_type,
            "symbol": self.symbol,
            "event_time": self.event_time.isoformat(),
            "received_at": self.received_at.isoformat(),
            "checksum": str(self.checksum),
            "sequence_or_checksum": self.sequence_or_checksum,
            "update_type": self.update_type,
            "bid_level_count": str(len(self.bids)),
            "ask_level_count": str(len(self.asks)),
            "best_bid": "" if self.best_bid is None else str(self.best_bid),
            "best_ask": "" if self.best_ask is None else str(self.best_ask),
        }


def normalize_kraken_book_payload_fixture(
    payload: Mapping[str, Any],
    *,
    received_at: datetime,
) -> ResearchBookEvent:
    """Normalize one Kraken WebSocket v2 book fixture payload."""
    channel = _require_string(payload, "channel")
    if channel != "book":
        raise BookPayloadNormalizationError(f"Unsupported channel: {channel}")
    message_type = _require_string(payload, "type")
    if message_type not in {"snapshot", "update"}:
        raise BookPayloadNormalizationError(f"Unsupported book message type: {message_type}")
    book = _single_book_payload(payload)
    checksum = int(_require(book, "checksum"))
    event_time = parse_rfc3339(_require_string(book, "timestamp"))
    return ResearchBookEvent(
        source_exchange="kraken",
        channel=channel,
        message_type=message_type,
        symbol=_require_string(book, "symbol"),
        event_time=event_time,
        received_at=received_at,
        checksum=checksum,
        sequence_or_checksum=str(checksum),
        update_type=message_type,
        bids=_normalize_levels(book, "bids"),
        asks=_normalize_levels(book, "asks"),
        payload=dict(payload),
    )


def planned_kraken_book_subscription(
    symbols: tuple[str, ...],
    *,
    depth: int = 10,
    snapshot: bool = True,
) -> dict[str, Any]:
    """Render the planned Kraken book subscription without wiring runtime capture."""
    if depth not in {10, 25, 100, 500, 1000}:
        raise ValueError("Kraken book depth must be one of 10, 25, 100, 500, or 1000")
    if not symbols:
        raise ValueError("At least one symbol is required")
    return {
        "method": "subscribe",
        "params": {
            "channel": "book",
            "symbol": list(symbols),
            "depth": depth,
            "snapshot": snapshot,
        },
    }


def write_market_microstructure_book_normalizer_contract(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write fixture-backed research-only book normalizer contract artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    samples = _sample_payloads()
    received_at = parse_rfc3339("2023-10-06T17:35:56.000000Z")
    normalized = [
        normalize_kraken_book_payload_fixture(sample, received_at=received_at)
        for sample in samples
    ]
    rows = _artifact_rows(normalized)
    recommendation = _recommendation()
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "book_payload_normalizer_contract_v1",
        "repo_root": str(root),
        "m20_research_decision": M20_DECISION,
        "normalizer_status": "SAMPLE_BOOK_PAYLOAD_NORMALIZERS_DEFINED",
        "sample_payload_count": len(samples),
        "normalized_event_count": len(normalized),
        "subscription_depths_supported": [10, 25, 100, 500, 1000],
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "schema_version": "book_payload_normalizer_contract_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(resolved_output_dir),
        "source_reference": "https://docs.kraken.com/api/docs/websocket-v2/book/",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    _write_outputs(
        output_files=output_files,
        manifest=manifest,
        report=report,
        recommendation=recommendation,
        rows_by_file={**rows, "next_actions_csv": recommendation["next_actions"]},
    )
    write_json_atomic(Path(output_files["sample_payloads_json"]), {"payloads": samples})
    Path(output_files["report_md"]).write_text(_markdown(report), encoding="utf-8")
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "sample_payloads": samples,
            "normalized_events": [asdict(event) for event in normalized],
            "parser_contract": rows["parser_contract_csv"],
            "recommendation_payload": recommendation,
        }
    )


def _single_book_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    data = _require(payload, "data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], Mapping):
        raise BookPayloadNormalizationError("Kraken book payload must contain one data object")
    return data[0]


def _normalize_levels(
    book: Mapping[str, Any],
    side: str,
) -> tuple[ResearchBookLevel, ...]:
    raw_levels = _require(book, side)
    if not isinstance(raw_levels, list):
        raise BookPayloadNormalizationError(f"{side} must be a list")
    levels: list[ResearchBookLevel] = []
    for raw_level in raw_levels:
        if not isinstance(raw_level, Mapping):
            raise BookPayloadNormalizationError(f"{side} level must be an object")
        levels.append(
            ResearchBookLevel(
                price=float(_require(raw_level, "price")),
                quantity=float(_require(raw_level, "qty")),
            )
        )
    return tuple(levels)


def _require(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise BookPayloadNormalizationError(f"Missing required field: {key}")
    return payload[key]


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = _require(payload, key)
    if not isinstance(value, str) or not value.strip():
        raise BookPayloadNormalizationError(f"Field {key} must be a non-empty string")
    return value.strip()


def _sample_payloads() -> list[dict[str, Any]]:
    return [
        {
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
        },
        {
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
        },
    ]


def _artifact_rows(
    normalized: list[ResearchBookEvent],
) -> dict[str, list[dict[str, str]]]:
    return {
        "normalized_sample_events_csv": [event.to_csv_row() for event in normalized],
        "parser_contract_csv": _parser_contract(),
        "validation_cases_csv": _validation_cases(),
        "leakage_boundary_audit_csv": _leakage_boundary_audit(),
        "blocked_decisions_csv": _blocked_decisions(),
    }


def _parser_contract() -> list[dict[str, str]]:
    return [
        _parser_row("channel", "book", "required", "payload root"),
        _parser_row("type", "snapshot|update", "required", "payload root"),
        _parser_row("data[0].symbol", "string", "required", "book object"),
        _parser_row("data[0].bids[].price", "float", "required", "bid level"),
        _parser_row("data[0].bids[].qty", "float", "required", "bid level"),
        _parser_row("data[0].asks[].price", "float", "required", "ask level"),
        _parser_row("data[0].asks[].qty", "float", "required", "ask level"),
        _parser_row("data[0].checksum", "integer", "required", "book object"),
        _parser_row("data[0].timestamp", "RFC3339", "required", "book object"),
    ]


def _parser_row(field: str, data_type: str, required: str, location: str) -> dict[str, str]:
    return {
        "field": field,
        "data_type": data_type,
        "required": required,
        "location": location,
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _validation_cases() -> list[dict[str, str]]:
    return [
        _case("snapshot_payload", "valid snapshot payload normalizes", "PASSING_FIXTURE"),
        _case("update_payload", "valid update payload normalizes", "PASSING_FIXTURE"),
        _case("missing_data", "missing data field fails clearly", "TESTED"),
        _case("unsupported_channel", "non-book channel fails clearly", "TESTED"),
        _case("invalid_depth", "unsupported subscription depth fails clearly", "TESTED"),
    ]


def _case(name: str, expectation: str, status: str) -> dict[str, str]:
    return {
        "case": name,
        "expectation": expectation,
        "status": status,
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _leakage_boundary_audit() -> list[dict[str, str]]:
    return [
        _boundary("payload_normalization", "uses only current book fixture payload fields"),
        _boundary("feature_derivation", "not implemented in this batch"),
        _boundary("capture_service", "not implemented in this batch"),
        _boundary("runtime_ingestion", "no service imports or subscriptions changed"),
    ]


def _boundary(name: str, rule: str) -> dict[str, str]:
    return {
        "boundary": name,
        "rule": rule,
        "uses_future_data": "False",
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _blocked_decisions() -> list[dict[str, str]]:
    return [
        {
            "decision": "book_checksum_validation",
            "blocker": "CHECKSUM_RECONSTRUCTION_NOT_IMPLEMENTED_IN_FIXTURE_BATCH",
            "required_action": "ADD_REPLAY_STATE_AND_CHECKSUM_VALIDATION_WITH_DU4_OR_DU5",
        },
        {
            "decision": "capture_service",
            "blocker": "NO_CAPTURE_SERVICE_APPROVAL",
            "required_action": "KEEP_CAPTURE_BLOCKED_UNTIL_DU6_APPROVAL",
        },
    ]


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": "DERIVE_RESEARCH_ONLY_MICROSTRUCTURE_FEATURES_FROM_CONTRACTS",
        "next_required_action": "BUILD_RESEARCH_ONLY_MICROSTRUCTURE_FEATURE_DERIVATION",
        "next_actions": [
            {
                "action": "BUILD_RESEARCH_ONLY_MICROSTRUCTURE_FEATURE_DERIVATION",
                "scope": "research_data_upgrade",
                "runtime_effect": "NO_RUNTIME_EFFECT",
                "m20_status": "M20_PAUSED",
            }
        ],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
    }


def _write_outputs(
    *,
    output_files: Mapping[str, str],
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    rows_by_file: Mapping[str, list[Mapping[str, Any]]],
) -> None:
    write_json_atomic(Path(output_files["manifest_json"]), dict(manifest))
    write_json_atomic(Path(output_files["report_json"]), dict(report))
    write_json_atomic(Path(output_files["recommendation_json"]), dict(recommendation))
    for key, rows in rows_by_file.items():
        write_csv(Path(output_files[key]), [dict(row) for row in rows])


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "book_payload_normalizer_contract.json"),
        "report_md": str(output_dir / "book_payload_normalizer_contract.md"),
        "sample_payloads_json": str(output_dir / "sample_payloads.json"),
        "normalized_sample_events_csv": str(output_dir / "normalized_sample_events.csv"),
        "parser_contract_csv": str(output_dir / "parser_contract.csv"),
        "validation_cases_csv": str(output_dir / "validation_cases.csv"),
        "leakage_boundary_audit_csv": str(output_dir / "leakage_boundary_audit.csv"),
        "blocked_decisions_csv": str(output_dir / "blocked_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Book Payload Normalizer Contract",
        "",
        f"- Normalizer status: `{report['normalizer_status']}`",
        f"- Sample payload count: `{report['sample_payload_count']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- M20 decision: `{report['m20_research_decision']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "This is fixture-only research normalization. It does not run capture,",
        "subscribe to Kraken book data, or change production ingestion.",
    ]
    return "\n".join(lines) + "\n"
