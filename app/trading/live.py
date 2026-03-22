"""Explicit M12 guarded-live helpers for Stream Alpha."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339, utc_now
from app.trading.alpaca import AlpacaClientError, AlpacaTradingClient
from app.trading.config import PaperTradingConfig
from app.trading.schemas import (
    CanonicalSystemReliability,
    BrokerAccount,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    FeatureCandle,
    LiveSafetyState,
    OrderLifecycleEvent,
    OrderRequest,
    PaperPosition,
    LiveStartupCheck,
    LiveStartupChecklist,
    SignalDecision,
)


ALPACA_BROKER_NAME = "alpaca"
LIVE_CONFIRMATION_PHRASE = "I UNDERSTAND STREAM ALPHA LIVE TRADING IS ENABLED"
LIVE_DISABLED = "LIVE_DISABLED"
LIVE_ACCOUNT_ID_MISMATCH = "LIVE_ACCOUNT_ID_MISMATCH"
LIVE_ENVIRONMENT_MISMATCH = "LIVE_ENVIRONMENT_MISMATCH"
LIVE_NOT_ARMED = "LIVE_NOT_ARMED"
LIVE_CONFIRMATION_MISMATCH = "LIVE_CONFIRMATION_MISMATCH"
LIVE_STARTUP_CHECKS_NOT_PASSED = "LIVE_STARTUP_CHECKS_NOT_PASSED"
LIVE_MANUAL_DISABLE_ACTIVE = "LIVE_MANUAL_DISABLE_ACTIVE"
LIVE_FAILURE_HARD_STOP_ACTIVE = "LIVE_FAILURE_HARD_STOP_ACTIVE"
LIVE_SYMBOL_NOT_WHITELISTED = "LIVE_SYMBOL_NOT_WHITELISTED"
LIVE_MAX_ORDER_NOTIONAL_EXCEEDED = "LIVE_MAX_ORDER_NOTIONAL_EXCEEDED"
LIVE_BROKER_SUBMIT_FAILED = "LIVE_BROKER_SUBMIT_FAILED"
LIVE_ORDER_SUBMITTED = "LIVE_ORDER_SUBMITTED"
LIVE_ORDER_ACCEPTED = "LIVE_ORDER_ACCEPTED"
LIVE_ORDER_PARTIALLY_FILLED = "LIVE_ORDER_PARTIALLY_FILLED"
LIVE_ORDER_FILLED = "LIVE_ORDER_FILLED"
LIVE_ORDER_REJECTED = "LIVE_ORDER_REJECTED"
LIVE_ORDER_CANCELED = "LIVE_ORDER_CANCELED"
LIVE_ORDER_FAILED = "LIVE_ORDER_FAILED"
LIVE_RECONCILIATION_CLEAR = "LIVE_RECONCILIATION_CLEAR"
LIVE_RECONCILIATION_BROKER_UNAVAILABLE = "LIVE_RECONCILIATION_BROKER_UNAVAILABLE"
LIVE_RECONCILIATION_BLOCKED = "LIVE_RECONCILIATION_BLOCKED"
LIVE_RECONCILIATION_ORPHAN_ORDER = "LIVE_RECONCILIATION_ORPHAN_ORDER"
LIVE_RECONCILIATION_ORPHAN_POSITION = "LIVE_RECONCILIATION_ORPHAN_POSITION"
LIVE_RECONCILIATION_LOCAL_POSITION_MISSING_AT_BROKER = (
    "LIVE_RECONCILIATION_LOCAL_POSITION_MISSING_AT_BROKER"
)
LIVE_RECONCILIATION_ORDER_STATE_MISMATCH = "LIVE_RECONCILIATION_ORDER_STATE_MISMATCH"
LIVE_RECONCILIATION_POSITION_QTY_MISMATCH = (
    "LIVE_RECONCILIATION_POSITION_QTY_MISMATCH"
)
LIVE_HEALTH_GATE_CLEAR = "LIVE_HEALTH_GATE_CLEAR"
LIVE_SYSTEM_HEALTH_UNAVAILABLE = "LIVE_SYSTEM_HEALTH_UNAVAILABLE"
LIVE_SYSTEM_HEALTH_STALE = "LIVE_SYSTEM_HEALTH_STALE"
LIVE_SIGNAL_STALE = "LIVE_SIGNAL_STALE"
LIVE_PAPER_PROBE_SYMBOL_NOT_WHITELISTED = "LIVE_PAPER_PROBE_SYMBOL_NOT_WHITELISTED"
LIVE_PAPER_PROBE_INTEGER_QTY_REQUIRED = "LIVE_PAPER_PROBE_INTEGER_QTY_REQUIRED"
LIVE_PAPER_PROBE_MIN_ORDER_VALUE_REQUIRED = "LIVE_PAPER_PROBE_MIN_ORDER_VALUE_REQUIRED"
LIVE_PAPER_PROBE_MAX_ORDERS_PER_RUN_REACHED = "LIVE_PAPER_PROBE_MAX_ORDERS_PER_RUN_REACHED"

_LIVE_TERMINAL_ORDER_STATES = {"FILLED", "REJECTED", "CANCELED", "FAILED"}


@dataclass(frozen=True, slots=True)
class LiveReconciliationIncident:
    """One explicit live reconciliation mismatch requiring operator attention."""

    reason_code: str
    detail: str


class LiveStartupValidationError(RuntimeError):
    """Raised when M12 live startup validation fails."""


def resolve_live_submit_gate(  # pylint: disable=too-many-return-statements
    *,
    state: LiveSafetyState,
    expected_account_id: str | None = None,
    expected_environment: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve the operator-facing live submit gate from existing live truth."""
    if not state.live_enabled:
        return (
            LIVE_DISABLED,
            "Live execution is configured but live_enabled is false.",
        )
    if expected_environment is not None and state.environment_name is not None:
        if state.environment_name != expected_environment:
            return (
                LIVE_ENVIRONMENT_MISMATCH,
                "Validated broker environment does not match the configured live "
                f"environment: expected={expected_environment} "
                f"validated={state.environment_name}",
            )
    if expected_account_id is not None and state.account_id is not None:
        if state.account_id != expected_account_id:
            return (
                LIVE_ACCOUNT_ID_MISMATCH,
                "Validated broker account does not match the configured live "
                f"account: expected={expected_account_id} validated={state.account_id}",
            )
    if not state.startup_checks_passed:
        return (
            LIVE_STARTUP_CHECKS_NOT_PASSED,
            "Startup safety validation has not passed.",
        )
    if state.manual_disable_active:
        return (
            LIVE_MANUAL_DISABLE_ACTIVE,
            "Manual disable is active for guarded live trading.",
        )
    if state.failure_hard_stop_active:
        return (
            LIVE_FAILURE_HARD_STOP_ACTIVE,
            state.last_failure_reason or "Live failure hard-stop is active.",
        )
    if state.reconciliation_status != "CLEAR":
        return (
            state.reconciliation_reason_code or LIVE_RECONCILIATION_BLOCKED,
            "Broker reconciliation is not clear for live submit.",
        )
    if state.health_gate_status != "CLEAR":
        return (
            state.health_gate_reason_code
            or state.system_health_reason_code
            or LIVE_SYSTEM_HEALTH_UNAVAILABLE,
            state.health_gate_detail or "Canonical live health gating is not clear.",
        )
    return None, None


def derive_live_submit_state(
    *,
    state: LiveSafetyState,
    expected_account_id: str | None = None,
    expected_environment: str | None = None,
) -> LiveSafetyState:
    """Attach the canonical operator-facing live submit state to persisted safety state."""
    reason_code, block_detail = resolve_live_submit_gate(
        state=state,
        expected_account_id=expected_account_id,
        expected_environment=expected_environment,
    )
    return replace(
        state,
        can_submit_live_now=reason_code is None,
        primary_block_reason_code=reason_code,
        block_detail=block_detail,
    )


def is_runtime_live_enabled() -> bool:
    """Return whether the live runtime arming switch is explicitly enabled."""
    return os.getenv("STREAMALPHA_ENABLE_LIVE", "").strip().lower() == "true"


def runtime_confirmation_matches() -> bool:
    """Return whether the live confirmation phrase matches the required value."""
    return os.getenv("STREAMALPHA_LIVE_CONFIRM", "").strip() == LIVE_CONFIRMATION_PHRASE


def is_manual_disable_active(path: str) -> bool:
    """Return whether the local live manual-disable sentinel is active."""
    return Path(path).expanduser().exists()


async def validate_live_startup(  # pylint: disable=too-many-locals,too-many-statements
    *,
    config: PaperTradingConfig,
    broker_client: AlpacaTradingClient | None = None,
) -> tuple[LiveStartupChecklist, LiveSafetyState, AlpacaTradingClient | None]:
    """Validate the guarded live runtime before any broker submission is allowed."""
    checked_at = utc_now()
    live_config = config.execution.live
    checks: list[LiveStartupCheck] = []

    api_key_id = os.getenv("APCA_API_KEY_ID", "").strip()
    api_secret_key = os.getenv("APCA_API_SECRET_KEY", "").strip()
    alpaca_base_url = os.getenv("ALPACA_BASE_URL", "").strip()
    runtime_armed = is_runtime_live_enabled()
    confirmation_matches = runtime_confirmation_matches()
    manual_disable_active = is_manual_disable_active(live_config.manual_disable_path)

    checks.append(
        _startup_check(
            "execution_mode_live",
            config.execution.mode == "live",
            f"mode={config.execution.mode}",
        )
    )
    checks.append(
        _startup_check(
            "live_config_enabled",
            live_config.enabled,
            f"live.enabled={live_config.enabled}",
        )
    )
    checks.append(
        _startup_check(
            "runtime_live_armed",
            runtime_armed,
            "STREAMALPHA_ENABLE_LIVE must be true",
        )
    )
    checks.append(
        _startup_check(
            "runtime_confirmation",
            confirmation_matches,
            f"Expected exact phrase: {LIVE_CONFIRMATION_PHRASE}",
        )
    )
    checks.append(
        _startup_check(
            "alpaca_api_key_present",
            bool(api_key_id),
            "APCA_API_KEY_ID must be set",
        )
    )
    checks.append(
        _startup_check(
            "alpaca_api_secret_present",
            bool(api_secret_key),
            "APCA_API_SECRET_KEY must be set",
        )
    )
    checks.append(
        _startup_check(
            "alpaca_base_url_present",
            bool(alpaca_base_url),
            "ALPACA_BASE_URL must be set to the root domain only",
        )
    )

    resolved_client = broker_client
    validated_account_id: str | None = None
    validated_environment: str | None = None
    account_validated = False
    last_failure_reason: str | None = None
    broker_cash: float | None = None
    broker_equity: float | None = None

    if bool(api_key_id) and bool(api_secret_key) and bool(alpaca_base_url):
        try:
            if resolved_client is None:
                resolved_client = AlpacaTradingClient.from_env()
            account = await resolved_client.validate_account()
            validated_account_id = account.account_id
            validated_environment = account.environment_name
            account_validated = True
            broker_cash = account.cash
            broker_equity = account.equity
            checks.append(
                _startup_check(
                    "broker_authentication",
                    True,
                    "GET /v2/account succeeded",
                )
            )
            checks.append(
                _startup_check(
                    "account_environment_match",
                    account.environment_name == live_config.expected_environment,
                    (
                        f"expected={live_config.expected_environment} "
                        f"validated={account.environment_name}"
                    ),
                )
            )
            account_id_matches = (
                live_config.expected_account_id is None
                or account.account_id == live_config.expected_account_id
            )
            checks.append(
                _startup_check(
                    "account_id_match",
                    account_id_matches,
                    (
                        f"expected={live_config.expected_account_id or '<not set>'} "
                        f"validated={account.account_id}"
                    ),
                )
            )
        except (ValueError, AlpacaClientError) as error:
            last_failure_reason = str(error)
            checks.append(
                _startup_check(
                    "broker_authentication",
                    False,
                    str(error),
                )
            )
            checks.append(
                _startup_check(
                    "account_environment_match",
                    False,
                    "Account validation did not complete",
                )
            )
            checks.append(
                _startup_check(
                    "account_id_match",
                    False,
                    "Account validation did not complete",
                )
            )
    else:
        checks.append(
            _startup_check(
                "broker_authentication",
                False,
                "Broker validation skipped because required Alpaca env vars were missing",
            )
        )
        checks.append(
            _startup_check(
                "account_environment_match",
                False,
                "Account validation did not complete",
            )
        )
        checks.append(
            _startup_check(
                "account_id_match",
                False,
                "Account validation did not complete",
            )
        )

    checks.append(
        _startup_check(
            "symbol_whitelist_non_empty",
            bool(live_config.symbol_whitelist),
            f"whitelist={list(live_config.symbol_whitelist)}",
        )
    )
    checks.append(
        _startup_check(
            "max_order_notional_positive",
            live_config.max_order_notional > 0.0,
            f"max_order_notional={live_config.max_order_notional}",
        )
    )
    checks.append(
        _startup_check(
            "failure_hard_stop_threshold_positive",
            live_config.failure_hard_stop_threshold > 0,
            (
                "failure_hard_stop_threshold="
                f"{live_config.failure_hard_stop_threshold}"
            ),
        )
    )
    checks.append(
        _startup_check(
            "manual_disable_inactive",
            not manual_disable_active,
            f"manual_disable_path={live_config.manual_disable_path}",
        )
    )

    passed = all(check.passed for check in checks)
    state = LiveSafetyState(
        service_name=config.service_name,
        execution_mode=config.execution.mode,
        broker_name=ALPACA_BROKER_NAME,
        live_enabled=live_config.enabled,
        startup_checks_passed=passed,
        startup_checks_passed_at=checked_at if passed else None,
        account_validated=account_validated,
        account_id=validated_account_id,
        environment_name=validated_environment,
        manual_disable_active=manual_disable_active,
        consecutive_live_failures=0,
        failure_hard_stop_active=False,
        last_failure_reason=last_failure_reason,
        broker_cash=broker_cash,
        broker_equity=broker_equity,
        updated_at=checked_at,
    )
    state = derive_live_submit_state(
        state=state,
        expected_account_id=live_config.expected_account_id,
        expected_environment=live_config.expected_environment,
    )
    checklist = LiveStartupChecklist(
        service_name=config.service_name,
        execution_mode=config.execution.mode,
        broker_name=ALPACA_BROKER_NAME,
        checked_at=checked_at,
        passed=passed,
        expected_account_id=live_config.expected_account_id,
        validated_account_id=validated_account_id,
        expected_environment=live_config.expected_environment,
        validated_environment=validated_environment,
        live_enabled=live_config.enabled,
        runtime_armed=runtime_armed,
        runtime_confirmation_phrase=LIVE_CONFIRMATION_PHRASE,
        manual_disable_path=live_config.manual_disable_path,
        symbol_whitelist=live_config.symbol_whitelist,
        max_order_notional=live_config.max_order_notional,
        failure_hard_stop_threshold=live_config.failure_hard_stop_threshold,
        checks=tuple(checks),
    )
    return checklist, state, resolved_client


def assert_live_startup_passed(checklist: LiveStartupChecklist) -> None:
    """Raise a typed error when the M12 startup checklist did not pass."""
    if checklist.passed:
        return
    failed_checks = [check.name for check in checklist.checks if not check.passed]
    raise LiveStartupValidationError(
        "M12 live startup validation failed: "
        + ", ".join(failed_checks)
    )


def refresh_manual_disable_state(
    *,
    state: LiveSafetyState,
    manual_disable_path: str,
) -> LiveSafetyState:
    """Refresh the persisted live manual-disable flag from the local sentinel path."""
    return replace(
        state,
        manual_disable_active=is_manual_disable_active(manual_disable_path),
        updated_at=utc_now(),
    )


# pylint: disable=too-many-arguments
def reconcile_live_state(
    *,
    state: LiveSafetyState,
    account: BrokerAccount,
    broker_orders: list[BrokerOrderSnapshot],
    broker_positions: list[BrokerPositionSnapshot],
    local_order_events: list[OrderLifecycleEvent],
    local_open_positions: dict[str, PaperPosition],
) -> tuple[LiveSafetyState, tuple[LiveReconciliationIncident, ...]]:
    """Compare local tracked live state against broker truth and fail closed on mismatch."""
    incidents = _collect_live_reconciliation_incidents(
        broker_orders=broker_orders,
        broker_positions=broker_positions,
        local_order_events=local_order_events,
        local_open_positions=local_open_positions,
    )
    checked_at = utc_now()
    if incidents:
        return (
            replace(
                state,
                account_validated=True,
                account_id=account.account_id,
                environment_name=account.environment_name,
                broker_cash=account.cash,
                broker_equity=account.equity,
                reconciliation_status="BLOCKED",
                reconciliation_reason_code=incidents[0].reason_code,
                reconciliation_checked_at=checked_at,
                unresolved_incident_count=len(incidents),
                updated_at=checked_at,
            ),
            incidents,
        )
    return (
        replace(
            state,
            account_validated=True,
            account_id=account.account_id,
            environment_name=account.environment_name,
            broker_cash=account.cash,
            broker_equity=account.equity,
            reconciliation_status="CLEAR",
            reconciliation_reason_code=LIVE_RECONCILIATION_CLEAR,
            reconciliation_checked_at=checked_at,
            unresolved_incident_count=0,
            updated_at=checked_at,
        ),
        (),
    )


def mark_live_reconciliation_unavailable(
    *,
    state: LiveSafetyState,
    reason_code: str,
) -> LiveSafetyState:
    """Mark broker-truth reconciliation unavailable so live submit fails closed."""
    checked_at = utc_now()
    return replace(
        state,
        reconciliation_status="UNAVAILABLE",
        reconciliation_reason_code=reason_code,
        reconciliation_checked_at=checked_at,
        unresolved_incident_count=0,
        updated_at=checked_at,
    )


def assert_live_reconciliation_clear(state: LiveSafetyState) -> None:
    """Raise when live reconciliation is not clear enough for guarded startup."""
    if state.reconciliation_status == "CLEAR":
        return
    raise LiveStartupValidationError(
        "M12 live reconciliation failed closed: "
        f"{state.reconciliation_reason_code or LIVE_RECONCILIATION_BLOCKED}"
    )


def apply_live_health_gate(  # pylint: disable=too-many-arguments,too-many-return-statements
    *,
    state: LiveSafetyState,
    system_reliability: CanonicalSystemReliability | None,
    heartbeat_stale_after_seconds: int,
    signal: SignalDecision | None = None,
    candle: FeatureCandle | None = None,
    order_request: OrderRequest | None = None,
) -> LiveSafetyState:
    """Apply the canonical M13 live health gate and persist the latest decision."""
    checked_at = utc_now()
    if system_reliability is None:
        return replace(
            state,
            system_health_status="UNAVAILABLE",
            system_health_reason_code=LIVE_SYSTEM_HEALTH_UNAVAILABLE,
            system_health_checked_at=None,
            health_gate_status="UNAVAILABLE",
            health_gate_reason_code=LIVE_SYSTEM_HEALTH_UNAVAILABLE,
            health_gate_detail="Canonical system health could not be loaded",
            updated_at=checked_at,
        )

    reason_code = (
        system_reliability.reason_codes[0]
        if system_reliability.reason_codes
        else LIVE_HEALTH_GATE_CLEAR
    )
    snapshot_age_seconds = max(
        0.0,
        (checked_at - system_reliability.checked_at).total_seconds(),
    )
    if snapshot_age_seconds > heartbeat_stale_after_seconds:
        return replace(
            state,
            system_health_status=system_reliability.health_overall_status,
            system_health_reason_code=reason_code,
            system_health_checked_at=system_reliability.checked_at,
            health_gate_status="UNAVAILABLE",
            health_gate_reason_code=LIVE_SYSTEM_HEALTH_STALE,
            health_gate_detail=(
                "Canonical system health snapshot is stale: "
                f"age_seconds={snapshot_age_seconds:.3f} "
                f"threshold={heartbeat_stale_after_seconds}"
            ),
            updated_at=checked_at,
        )

    if system_reliability.health_overall_status != "HEALTHY":
        return replace(
            state,
            system_health_status=system_reliability.health_overall_status,
            system_health_reason_code=reason_code,
            system_health_checked_at=system_reliability.checked_at,
            health_gate_status=system_reliability.health_overall_status,
            health_gate_reason_code=reason_code,
            health_gate_detail=(
                "Canonical system health is not healthy: "
                f"{system_reliability.health_overall_status}"
            ),
            updated_at=checked_at,
        )

    if signal is not None and (
        signal.health_overall_status not in {None, "HEALTHY"}
        or signal.freshness_status not in {None, "FRESH"}
    ):
        return replace(
            state,
            system_health_status=system_reliability.health_overall_status,
            system_health_reason_code=reason_code,
            system_health_checked_at=system_reliability.checked_at,
            health_gate_status="BLOCKED",
            health_gate_reason_code=signal.reason_code or LIVE_SIGNAL_STALE,
            health_gate_detail=signal.reason,
            updated_at=checked_at,
        )

    if (
        order_request is not None
        and candle is not None
        and order_request.target_fill_interval_begin < candle.interval_begin
    ):
        return replace(
            state,
            system_health_status=system_reliability.health_overall_status,
            system_health_reason_code=reason_code,
            system_health_checked_at=system_reliability.checked_at,
            health_gate_status="BLOCKED",
            health_gate_reason_code=LIVE_SIGNAL_STALE,
            health_gate_detail=(
                "Order request target fill interval is stale: "
                f"target={to_rfc3339(order_request.target_fill_interval_begin)} "
                f"current={to_rfc3339(candle.interval_begin)}"
            ),
            updated_at=checked_at,
        )

    return replace(
        state,
        system_health_status=system_reliability.health_overall_status,
        system_health_reason_code=reason_code,
        system_health_checked_at=system_reliability.checked_at,
        health_gate_status="CLEAR",
        health_gate_reason_code=LIVE_HEALTH_GATE_CLEAR,
        health_gate_detail="Canonical live health gate is clear",
        updated_at=checked_at,
    )


def assert_live_health_gate_clear(state: LiveSafetyState) -> None:
    """Raise when the canonical M13 live gate is not clear enough for startup."""
    if state.health_gate_status == "CLEAR":
        return
    raise LiveStartupValidationError(
        "M13 live health gate failed closed: "
        f"{state.health_gate_reason_code or LIVE_SYSTEM_HEALTH_UNAVAILABLE}"
    )


def record_live_failure(
    *,
    state: LiveSafetyState,
    threshold: int,
    reason_code: str,
) -> LiveSafetyState:
    """Increment live failure state and activate the hard-stop when needed."""
    next_failures = state.consecutive_live_failures + 1
    return replace(
        state,
        consecutive_live_failures=next_failures,
        failure_hard_stop_active=next_failures >= threshold,
        last_failure_reason=reason_code,
        updated_at=utc_now(),
    )


def record_live_success(state: LiveSafetyState) -> LiveSafetyState:
    """Reset consecutive live failures after a successful broker submission."""
    return replace(
        state,
        consecutive_live_failures=0,
        last_failure_reason=None,
        updated_at=utc_now(),
    )


def write_startup_checklist_artifact(
    *,
    checklist: LiveStartupChecklist,
    artifact_path: str,
) -> None:
    """Write the redacted startup checklist artifact."""
    payload = {
        "service_name": checklist.service_name,
        "execution_mode": checklist.execution_mode,
        "broker_name": checklist.broker_name,
        "checked_at": to_rfc3339(checklist.checked_at),
        "passed": checklist.passed,
        "expected_account_id": checklist.expected_account_id,
        "validated_account_id": checklist.validated_account_id,
        "expected_environment": checklist.expected_environment,
        "validated_environment": checklist.validated_environment,
        "live_enabled": checklist.live_enabled,
        "runtime_armed": checklist.runtime_armed,
        "runtime_confirmation_phrase": checklist.runtime_confirmation_phrase,
        "manual_disable_path": checklist.manual_disable_path,
        "symbol_whitelist": list(checklist.symbol_whitelist),
        "max_order_notional": checklist.max_order_notional,
        "failure_hard_stop_threshold": checklist.failure_hard_stop_threshold,
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "detail": check.detail,
            }
            for check in checklist.checks
        ],
    }
    _write_json(Path(artifact_path), payload)


def write_live_status_artifact(
    *,
    state: LiveSafetyState,
    config: PaperTradingConfig,
    system_reliability: CanonicalSystemReliability | None = None,
) -> None:
    """Write the current guarded-live safety status artifact."""
    live_config = config.execution.live
    resolved_state = derive_live_submit_state(
        state=state,
        expected_account_id=live_config.expected_account_id,
        expected_environment=live_config.expected_environment,
    )
    payload = {
        "service_name": resolved_state.service_name,
        "execution_mode": resolved_state.execution_mode,
        "broker_name": resolved_state.broker_name,
        "live_enabled": resolved_state.live_enabled,
        "startup_checks_passed": resolved_state.startup_checks_passed,
        "startup_checks_passed_at": (
            None
            if resolved_state.startup_checks_passed_at is None
            else to_rfc3339(resolved_state.startup_checks_passed_at)
        ),
        "account_validated": resolved_state.account_validated,
        "account_id": resolved_state.account_id,
        "environment_name": resolved_state.environment_name,
        "broker_cash": resolved_state.broker_cash,
        "broker_equity": resolved_state.broker_equity,
        "expected_account_id": live_config.expected_account_id,
        "expected_environment": live_config.expected_environment,
        "research_source_exchange": config.source_exchange,
        "execution_broker_name": resolved_state.broker_name,
        "cross_venue_context": {
            "data_venue": config.source_exchange,
            "execution_venue": resolved_state.broker_name,
            "execution_environment": resolved_state.environment_name,
            "venue_mismatch": (
                config.source_exchange.lower() != resolved_state.broker_name.lower()
            ),
        },
        "order_submit_contract": (
            "Broker submit success records lifecycle events only; local portfolio "
            "and ledger mutations require broker-truth fill state or verified "
            "reconciliation."
        ),
        "portfolio_truth_source": "BROKER_RECONCILIATION_ONLY",
        "symbol_whitelist": list(live_config.symbol_whitelist),
        "max_order_notional": live_config.max_order_notional,
        "manual_disable_active": resolved_state.manual_disable_active,
        "manual_disable_path": live_config.manual_disable_path,
        "consecutive_live_failures": resolved_state.consecutive_live_failures,
        "failure_hard_stop_active": resolved_state.failure_hard_stop_active,
        "failure_hard_stop_threshold": live_config.failure_hard_stop_threshold,
        "last_failure_reason": resolved_state.last_failure_reason,
        "can_submit_live_now": resolved_state.can_submit_live_now,
        "primary_block_reason_code": resolved_state.primary_block_reason_code,
        "block_detail": resolved_state.block_detail,
        "system_health_status": resolved_state.system_health_status,
        "system_health_reason_code": resolved_state.system_health_reason_code,
        "system_health_checked_at": (
            None
            if resolved_state.system_health_checked_at is None
            else to_rfc3339(resolved_state.system_health_checked_at)
        ),
        "health_gate_status": resolved_state.health_gate_status,
        "health_gate_reason_code": resolved_state.health_gate_reason_code,
        "health_gate_detail": resolved_state.health_gate_detail,
        "reconciliation_status": resolved_state.reconciliation_status,
        "reconciliation_reason_code": resolved_state.reconciliation_reason_code,
        "reconciliation_checked_at": (
            None
            if resolved_state.reconciliation_checked_at is None
            else to_rfc3339(resolved_state.reconciliation_checked_at)
        ),
        "unresolved_incident_count": resolved_state.unresolved_incident_count,
        "canonical_system_health": (
            None
            if system_reliability is None
            else make_json_safe(asdict(system_reliability))
        ),
        "updated_at": (
            None if resolved_state.updated_at is None else to_rfc3339(resolved_state.updated_at)
        ),
    }
    _write_json(Path(live_config.live_status_path), payload)


def _startup_check(name: str, passed: bool, detail: str) -> LiveStartupCheck:
    return LiveStartupCheck(name=name, passed=passed, detail=detail)


def _collect_live_reconciliation_incidents(
    *,
    broker_orders: list[BrokerOrderSnapshot],
    broker_positions: list[BrokerPositionSnapshot],
    local_order_events: list[OrderLifecycleEvent],
    local_open_positions: dict[str, PaperPosition],
) -> tuple[LiveReconciliationIncident, ...]:
    latest_events_by_external_order = _latest_events_by_external_order(local_order_events)
    broker_positions_by_symbol = {
        position.symbol: position for position in broker_positions if position.quantity > 0.0
    }
    incidents: list[LiveReconciliationIncident] = []

    for broker_order in broker_orders:
        local_event = latest_events_by_external_order.get(broker_order.external_order_id)
        if local_event is None:
            incidents.append(
                LiveReconciliationIncident(
                    reason_code=LIVE_RECONCILIATION_ORPHAN_ORDER,
                    detail=(
                        "Broker open order is not linked to a local order event: "
                        f"{broker_order.external_order_id} {broker_order.symbol}"
                    ),
                )
            )
            continue
        if local_event.lifecycle_state in _LIVE_TERMINAL_ORDER_STATES:
            incidents.append(
                LiveReconciliationIncident(
                    reason_code=LIVE_RECONCILIATION_ORDER_STATE_MISMATCH,
                    detail=(
                        "Broker reports an open order while local state is terminal: "
                        f"{broker_order.external_order_id} local={local_event.lifecycle_state}"
                    ),
                )
            )

    for broker_position in broker_positions:
        local_position = local_open_positions.get(broker_position.symbol)
        if local_position is None:
            incidents.append(
                LiveReconciliationIncident(
                    reason_code=LIVE_RECONCILIATION_ORPHAN_POSITION,
                    detail=(
                        "Broker open position has no local live position: "
                        f"{broker_position.symbol} qty={broker_position.quantity}"
                    ),
                )
            )
            continue
        if abs(local_position.quantity - broker_position.quantity) > 1e-9:
            incidents.append(
                LiveReconciliationIncident(
                    reason_code=LIVE_RECONCILIATION_POSITION_QTY_MISMATCH,
                    detail=(
                        "Broker and local live position quantities differ: "
                        f"{broker_position.symbol} broker={broker_position.quantity} "
                        f"local={local_position.quantity}"
                    ),
                )
            )

    for symbol, local_position in local_open_positions.items():
        if symbol in broker_positions_by_symbol:
            continue
        incidents.append(
            LiveReconciliationIncident(
                reason_code=LIVE_RECONCILIATION_LOCAL_POSITION_MISSING_AT_BROKER,
                detail=(
                    "Local live position is missing at the broker: "
                    f"{symbol} qty={local_position.quantity}"
                ),
            )
        )

    return tuple(incidents)


def _latest_events_by_external_order(
    events: list[OrderLifecycleEvent],
) -> dict[str, OrderLifecycleEvent]:
    latest: dict[str, OrderLifecycleEvent] = {}
    for event in events:
        if event.external_order_id is None:
            continue
        existing = latest.get(event.external_order_id)
        if existing is None or (event.event_time, event.event_id or 0) > (
            existing.event_time,
            existing.event_id or 0,
        ):
            latest[event.external_order_id] = event
    return latest


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
