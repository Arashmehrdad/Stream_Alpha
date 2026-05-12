"""Research-only microstructure feature derivation from fixture contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.common.time import parse_rfc3339
from app.regime.artifacts import write_csv, write_json_atomic
from app.training.market_microstructure_book_normalizers import (
    ResearchBookEvent,
    normalize_kraken_book_payload_fixture,
)


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/microstructure_feature_derivation"
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


@dataclass(frozen=True, slots=True)
class MicrostructureFeatureRow:  # pylint: disable=too-many-instance-attributes
    """One research-only derived microstructure feature sample."""

    source_exchange: str
    symbol: str
    event_time: str
    received_at: str
    sequence_or_checksum: str
    update_type: str
    best_bid: float | None
    best_ask: float | None
    mid_price: float | None
    top_of_book_spread: float | None
    relative_spread: float | None
    bid_depth_liquidity: float
    ask_depth_liquidity: float
    total_depth_liquidity: float
    order_book_imbalance: float | None
    book_gap_flag: bool
    feature_version: str

    def to_csv_row(self) -> dict[str, str]:
        """Return a deterministic CSV row."""
        row = asdict(self)
        return {key: _csv_value(value) for key, value in row.items()}


def derive_book_microstructure_features(
    event: ResearchBookEvent,
    *,
    depth_levels: int = 2,
    feature_version: str = "microstructure_fixture_v1",
) -> MicrostructureFeatureRow:
    """Derive causal book features from one normalized research book event."""
    if depth_levels <= 0:
        raise ValueError("depth_levels must be positive")
    best_bid = event.best_bid
    best_ask = event.best_ask
    mid_price = None
    spread = None
    relative_spread = None
    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2.0
        relative_spread = spread / mid_price if mid_price else None
    bid_depth = sum(level.quantity for level in event.bids[:depth_levels])
    ask_depth = sum(level.quantity for level in event.asks[:depth_levels])
    total_depth = bid_depth + ask_depth
    imbalance = None
    if total_depth > 0:
        imbalance = (bid_depth - ask_depth) / total_depth
    return MicrostructureFeatureRow(
        source_exchange=event.source_exchange,
        symbol=event.symbol,
        event_time=event.event_time.isoformat(),
        received_at=event.received_at.isoformat(),
        sequence_or_checksum=event.sequence_or_checksum,
        update_type=event.update_type,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        top_of_book_spread=spread,
        relative_spread=relative_spread,
        bid_depth_liquidity=bid_depth,
        ask_depth_liquidity=ask_depth,
        total_depth_liquidity=total_depth,
        order_book_imbalance=imbalance,
        book_gap_flag=False,
        feature_version=feature_version,
    )


def write_microstructure_feature_derivation(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write research-only feature derivation artifacts from book fixtures."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    events = _fixture_events()
    feature_rows = [
        derive_book_microstructure_features(event, depth_levels=2) for event in events
    ]
    rows = _artifact_rows(feature_rows)
    recommendation = _recommendation()
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "microstructure_feature_derivation_v1",
        "repo_root": str(root),
        "m20_research_decision": M20_DECISION,
        "feature_derivation_status": "RESEARCH_ONLY_FEATURE_DERIVATION_DEFINED",
        "sample_event_count": len(events),
        "derived_feature_row_count": len(feature_rows),
        "feature_count": len(rows["feature_contract_csv"]),
        "blocked_decision_count": len(rows["blocked_decisions_csv"]),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "schema_version": "microstructure_feature_derivation_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(resolved_output_dir),
        "source_contract": str(
            root / "artifacts/research_data_upgrade/book_payload_normalizer_contract"
        ),
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
    Path(output_files["report_md"]).write_text(_markdown(report), encoding="utf-8")
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "derived_features": [asdict(row) for row in feature_rows],
            "feature_contract": rows["feature_contract_csv"],
            "recommendation_payload": recommendation,
        }
    )


def _fixture_events() -> list[ResearchBookEvent]:
    received_at = parse_rfc3339("2023-10-06T17:35:56.000000Z")
    return [
        normalize_kraken_book_payload_fixture(sample, received_at=received_at)
        for sample in _sample_payloads()
    ]


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
    feature_rows: list[MicrostructureFeatureRow],
) -> dict[str, list[dict[str, str]]]:
    return {
        "derived_feature_samples_csv": [row.to_csv_row() for row in feature_rows],
        "feature_contract_csv": _feature_contract(),
        "derivation_rules_csv": _derivation_rules(),
        "leakage_boundary_audit_csv": _leakage_boundary_audit(),
        "blocked_decisions_csv": _blocked_decisions(),
    }


def _feature_contract() -> list[dict[str, str]]:
    return [
        _feature("top_of_book_spread", "best_ask - best_bid"),
        _feature("relative_spread", "top_of_book_spread / mid_price"),
        _feature("bid_depth_liquidity", "sum bid quantity over configured depth"),
        _feature("ask_depth_liquidity", "sum ask quantity over configured depth"),
        _feature("total_depth_liquidity", "bid_depth_liquidity + ask_depth_liquidity"),
        _feature(
            "order_book_imbalance",
            "(bid_depth_liquidity - ask_depth_liquidity) / total_depth_liquidity",
        ),
    ]


def _feature(name: str, formula: str) -> dict[str, str]:
    return {
        "feature_name": name,
        "formula": formula,
        "input_scope": "current_or_past_book_replay_rows",
        "uses_future_data": "False",
        "uses_labels": "False",
        "uses_economic_outcomes": "False",
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _derivation_rules() -> list[dict[str, str]]:
    return [
        _rule("depth_levels", "default fixture depth is 2 levels"),
        _rule("missing_side", "spread is blank when bid or ask side is unavailable"),
        _rule("zero_depth", "imbalance is blank when total depth is zero"),
        _rule("gap_handling", "book_gap_flag remains false for static fixtures"),
    ]


def _rule(name: str, detail: str) -> dict[str, str]:
    return {"rule": name, "detail": detail, "runtime_effect": "NO_RUNTIME_EFFECT"}


def _leakage_boundary_audit() -> list[dict[str, str]]:
    return [
        _boundary("book_features", "uses normalized current fixture book levels only"),
        _boundary("trade_flow", "not derived until stored trade/book replay data exists"),
        _boundary("label_data", "labels and future returns are not inputs"),
        _boundary("runtime_ingestion", "no runtime imports or subscriptions changed"),
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
            "decision": "stored_replay_coverage",
            "blocker": "NO_STORED_MICROSTRUCTURE_REPLAY_ROWS_YET",
            "required_action": "ADD_COVERAGE_GAP_AND_REPLAY_DETERMINISM_REPORTS",
        },
        {
            "decision": "trade_flow_imbalance",
            "blocker": "REQUIRES_JOINED_RAW_TRADES_AND_BOOK_REPLAY_WINDOWS",
            "required_action": "DEFER_UNTIL_RESEARCH_REPLAY_ROWS_EXIST",
        },
        {
            "decision": "capture_service",
            "blocker": "NO_CAPTURE_SERVICE_APPROVAL",
            "required_action": "KEEP_CAPTURE_BLOCKED_UNTIL_DU6_APPROVAL",
        },
    ]


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": "ADD_COVERAGE_GAP_AND_REPLAY_DETERMINISM_REPORTS",
        "next_required_action": "BUILD_MICROSTRUCTURE_COVERAGE_GAP_REPLAY_AUDIT",
        "next_actions": [
            {
                "action": "BUILD_MICROSTRUCTURE_COVERAGE_GAP_REPLAY_AUDIT",
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
        "report_json": str(output_dir / "microstructure_feature_derivation.json"),
        "report_md": str(output_dir / "microstructure_feature_derivation.md"),
        "derived_feature_samples_csv": str(output_dir / "derived_feature_samples.csv"),
        "feature_contract_csv": str(output_dir / "feature_contract.csv"),
        "derivation_rules_csv": str(output_dir / "derivation_rules.csv"),
        "leakage_boundary_audit_csv": str(output_dir / "leakage_boundary_audit.csv"),
        "blocked_decisions_csv": str(output_dir / "blocked_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Microstructure Feature Derivation",
        "",
        f"- Feature derivation status: `{report['feature_derivation_status']}`",
        f"- Derived feature rows: `{report['derived_feature_row_count']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- M20 decision: `{report['m20_research_decision']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "This is fixture-only feature derivation. It does not read live streams,",
        "write tables, or alter production ingestion/runtime behavior.",
    ]
    return "\n".join(lines) + "\n"


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)
