"""Explicit M12 guarded-live helpers for Stream Alpha."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339, utc_now
from app.trading.alpaca import AlpacaClientError, AlpacaTradingClient
from app.trading.config import PaperTradingConfig
from app.trading.schemas import (
    LiveSafetyState,
    LiveStartupCheck,
    LiveStartupChecklist,
)


ALPACA_BROKER_NAME = "alpaca"
LIVE_CONFIRMATION_PHRASE = "I UNDERSTAND STREAM ALPHA LIVE TRADING IS ENABLED"
LIVE_NOT_ARMED = "LIVE_NOT_ARMED"
LIVE_CONFIRMATION_MISMATCH = "LIVE_CONFIRMATION_MISMATCH"
LIVE_STARTUP_CHECKS_NOT_PASSED = "LIVE_STARTUP_CHECKS_NOT_PASSED"
LIVE_MANUAL_DISABLE_ACTIVE = "LIVE_MANUAL_DISABLE_ACTIVE"
LIVE_FAILURE_HARD_STOP_ACTIVE = "LIVE_FAILURE_HARD_STOP_ACTIVE"
LIVE_SYMBOL_NOT_WHITELISTED = "LIVE_SYMBOL_NOT_WHITELISTED"
LIVE_MAX_ORDER_NOTIONAL_EXCEEDED = "LIVE_MAX_ORDER_NOTIONAL_EXCEEDED"
LIVE_BROKER_SUBMIT_FAILED = "LIVE_BROKER_SUBMIT_FAILED"
LIVE_ORDER_ACCEPTED = "LIVE_ORDER_ACCEPTED"
LIVE_ORDER_FILLED = "LIVE_ORDER_FILLED"
LIVE_ORDER_REJECTED = "LIVE_ORDER_REJECTED"


class LiveStartupValidationError(RuntimeError):
    """Raised when M12 live startup validation fails."""


def is_runtime_live_enabled() -> bool:
    """Return whether the live runtime arming switch is explicitly enabled."""
    return os.getenv("STREAMALPHA_ENABLE_LIVE", "").strip().lower() == "true"


def runtime_confirmation_matches() -> bool:
    """Return whether the live confirmation phrase matches the required value."""
    return os.getenv("STREAMALPHA_LIVE_CONFIRM", "").strip() == LIVE_CONFIRMATION_PHRASE


def is_manual_disable_active(path: str) -> bool:
    """Return whether the local live manual-disable sentinel is active."""
    return Path(path).expanduser().exists()


async def validate_live_startup(  # pylint: disable=too-many-locals
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

    if bool(api_key_id) and bool(api_secret_key) and bool(alpaca_base_url):
        try:
            if resolved_client is None:
                resolved_client = AlpacaTradingClient.from_env()
            account = await resolved_client.validate_account()
            validated_account_id = account.account_id
            validated_environment = account.environment_name
            account_validated = True
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
        updated_at=checked_at,
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
) -> None:
    """Write the current guarded-live safety status artifact."""
    live_config = config.execution.live
    payload = {
        "service_name": state.service_name,
        "execution_mode": state.execution_mode,
        "broker_name": state.broker_name,
        "live_enabled": state.live_enabled,
        "startup_checks_passed": state.startup_checks_passed,
        "startup_checks_passed_at": (
            None
            if state.startup_checks_passed_at is None
            else to_rfc3339(state.startup_checks_passed_at)
        ),
        "account_validated": state.account_validated,
        "account_id": state.account_id,
        "environment_name": state.environment_name,
        "expected_account_id": live_config.expected_account_id,
        "expected_environment": live_config.expected_environment,
        "symbol_whitelist": list(live_config.symbol_whitelist),
        "max_order_notional": live_config.max_order_notional,
        "manual_disable_active": state.manual_disable_active,
        "manual_disable_path": live_config.manual_disable_path,
        "consecutive_live_failures": state.consecutive_live_failures,
        "failure_hard_stop_active": state.failure_hard_stop_active,
        "failure_hard_stop_threshold": live_config.failure_hard_stop_threshold,
        "last_failure_reason": state.last_failure_reason,
        "updated_at": None if state.updated_at is None else to_rfc3339(state.updated_at),
    }
    _write_json(Path(live_config.live_status_path), payload)


def _startup_check(name: str, passed: bool, detail: str) -> LiveStartupCheck:
    return LiveStartupCheck(name=name, passed=passed, detail=detail)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
