"""One-shot runtime validation for Stream Alpha M16 startup checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path

from app.common.config import Settings
from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339, utc_now
from app.regime.live import load_live_regime_runtime
from app.runtime.config import (
    resolve_runtime_profile,
    resolve_startup_report_path,
    resolve_trading_config_path,
)
from app.trading.config import PaperTradingConfig, load_paper_trading_config
from app.trading.live import LIVE_CONFIRMATION_PHRASE
from app.training.registry import resolve_inference_model_metadata


STARTUP_REPORT_SCHEMA_VERSION = "m16_startup_report_v1"
_DEPLOYED_PROFILES = {"paper", "shadow", "live"}


@dataclass(frozen=True, slots=True)
class StartupValidationCheck:
    """One explicit startup validation check result."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class StartupValidationReport:  # pylint: disable=too-many-instance-attributes
    """Structured startup validation report persisted for M16."""

    schema_version: str
    checked_at: str
    runtime_profile: str | None
    startup_validation_passed: bool
    startup_report_path: str
    checks: tuple[StartupValidationCheck, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    trading_config_path: str | None = None
    execution_mode: str | None = None
    model_artifact_path: str | None = None
    model_version: str | None = None
    model_version_source: str | None = None
    regime_artifact_path: str | None = None
    regime_run_id: str | None = None
    regime_signal_policy_path: str | None = None
    live_secret_presence: dict[str, bool] = field(default_factory=dict)
    runtime_flags: dict[str, bool] = field(default_factory=dict)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def build_startup_validation_report() -> StartupValidationReport:
    """Build the full M16 startup validation report without secrets."""
    checked_at = to_rfc3339(utc_now())
    startup_report_path = resolve_startup_report_path()
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[StartupValidationCheck] = []
    runtime_profile: str | None = None
    settings: Settings | None = None
    trading_config: PaperTradingConfig | None = None
    trading_config_path: Path | None = None
    model_metadata: dict[str, str] | None = None
    regime_runtime = None

    profile_env_value = os.getenv("STREAMALPHA_RUNTIME_PROFILE", "").strip()
    try:
        runtime_profile = resolve_runtime_profile(profile_env_value, default=None)
        checks.append(
            _check(
                "runtime_profile",
                True,
                f"profile={runtime_profile}",
            )
        )
    except ValueError as error:
        errors.append(str(error))
        checks.append(
            _check(
                "runtime_profile",
                False,
                str(error),
            )
        )

    try:
        settings = Settings.from_env()
        checks.append(_check("settings_env_parse", True, "Environment parsing succeeded"))
    except ValueError as error:
        errors.append(str(error))
        checks.append(
            _check(
                "settings_env_parse",
                False,
                str(error),
            )
        )

    if runtime_profile in _DEPLOYED_PROFILES:
        env_config_path = os.getenv("STREAMALPHA_TRADING_CONFIG_PATH", "").strip()
        try:
            trading_config_path = resolve_trading_config_path(
                env_config_path,
                default_profile=runtime_profile,
                use_profile_default=True,
            )
            config_exists = trading_config_path.is_file()
            checks.append(
                _check(
                    "trading_config_path",
                    config_exists,
                    f"path={trading_config_path}",
                )
            )
            if not config_exists:
                errors.append(
                    f"Trading config path does not exist: {trading_config_path}"
                )
            else:
                trading_config = load_paper_trading_config(trading_config_path)
                checks.append(
                    _check(
                        "trading_config_load",
                        True,
                        "Loaded trading config successfully",
                    )
                )
                mode_matches = trading_config.execution.mode == runtime_profile
                checks.append(
                    _check(
                        "profile_matches_execution_mode",
                        mode_matches,
                        (
                            f"profile={runtime_profile} "
                            f"execution.mode={trading_config.execution.mode}"
                        ),
                    )
                )
                if not mode_matches:
                    errors.append(
                        "Runtime profile does not match trading config execution.mode: "
                        f"profile={runtime_profile} "
                        f"execution.mode={trading_config.execution.mode}"
                    )
        except ValueError as error:
            errors.append(str(error))
            checks.append(_check("trading_config_load", False, str(error)))

        if settings is not None:
            try:
                model_metadata = resolve_inference_model_metadata(
                    settings.inference.model_path,
                )
                checks.append(
                    _check(
                        "model_artifact",
                        True,
                        "Resolved inference model artifact",
                    )
                )
            except ValueError as error:
                errors.append(str(error))
                checks.append(_check("model_artifact", False, str(error)))
            try:
                regime_runtime = load_live_regime_runtime(
                    thresholds_path=settings.inference.regime_thresholds_path,
                    signal_policy_path=settings.inference.regime_signal_policy_path,
                )
                checks.append(
                    _check(
                        "regime_artifact",
                        True,
                        "Resolved live regime runtime",
                    )
                )
            except ValueError as error:
                errors.append(str(error))
                checks.append(_check("regime_artifact", False, str(error)))

    if runtime_profile == "live":
        for secret_name in (
            "APCA_API_KEY_ID",
            "APCA_API_SECRET_KEY",
            "ALPACA_BASE_URL",
        ):
            is_present = bool(os.getenv(secret_name, "").strip())
            checks.append(
                _check(
                    f"{secret_name.lower()}_present",
                    is_present,
                    f"{secret_name} present={is_present}",
                )
            )
            if not is_present:
                errors.append(f"{secret_name} must be set for live profile startup")
        live_armed = os.getenv("STREAMALPHA_ENABLE_LIVE", "").strip().lower() == "true"
        confirmation_matches = (
            os.getenv("STREAMALPHA_LIVE_CONFIRM", "").strip()
            == LIVE_CONFIRMATION_PHRASE
        )
        checks.append(
            _check(
                "streamalpha_enable_live",
                live_armed,
                "STREAMALPHA_ENABLE_LIVE must be true for live startup",
            )
        )
        checks.append(
            _check(
                "streamalpha_live_confirm",
                confirmation_matches,
                "STREAMALPHA_LIVE_CONFIRM must match the required confirmation phrase",
            )
        )
        if not live_armed:
            errors.append("STREAMALPHA_ENABLE_LIVE must be true for live profile startup")
        if not confirmation_matches:
            errors.append(
                "STREAMALPHA_LIVE_CONFIRM must match the required live confirmation phrase"
            )

    if runtime_profile == "dev":
        config_path_env = os.getenv("STREAMALPHA_TRADING_CONFIG_PATH", "").strip()
        if config_path_env:
            warnings.append(
                "STREAMALPHA_TRADING_CONFIG_PATH is set but is not required for the dev profile"
            )

    startup_validation_passed = not errors
    return StartupValidationReport(
        schema_version=STARTUP_REPORT_SCHEMA_VERSION,
        checked_at=checked_at,
        runtime_profile=runtime_profile,
        startup_validation_passed=startup_validation_passed,
        startup_report_path=str(startup_report_path),
        checks=tuple(checks),
        errors=tuple(errors),
        warnings=tuple(warnings),
        trading_config_path=(
            None if trading_config_path is None else str(trading_config_path)
        ),
        execution_mode=(
            None if trading_config is None else trading_config.execution.mode
        ),
        model_artifact_path=(
            None if model_metadata is None else model_metadata["model_artifact_path"]
        ),
        model_version=None if model_metadata is None else model_metadata["model_version"],
        model_version_source=(
            None if model_metadata is None else model_metadata["model_version_source"]
        ),
        regime_artifact_path=(
            None if regime_runtime is None else regime_runtime.artifact_path
        ),
        regime_run_id=None if regime_runtime is None else regime_runtime.run_id,
        regime_signal_policy_path=(
            None
            if settings is None
            else settings.inference.regime_signal_policy_path
        ),
        live_secret_presence={
            "APCA_API_KEY_ID": bool(os.getenv("APCA_API_KEY_ID", "").strip()),
            "APCA_API_SECRET_KEY": bool(os.getenv("APCA_API_SECRET_KEY", "").strip()),
            "ALPACA_BASE_URL": bool(os.getenv("ALPACA_BASE_URL", "").strip()),
        },
        runtime_flags={
            "STREAMALPHA_ENABLE_LIVE": (
                os.getenv("STREAMALPHA_ENABLE_LIVE", "").strip().lower() == "true"
            ),
            "STREAMALPHA_LIVE_CONFIRM_MATCHED": (
                os.getenv("STREAMALPHA_LIVE_CONFIRM", "").strip()
                == LIVE_CONFIRMATION_PHRASE
            ),
        },
    )
# pylint: enable=too-many-locals,too-many-branches,too-many-statements


def write_startup_validation_report(report: StartupValidationReport) -> Path:
    """Persist the startup validation report to disk."""
    output_path = resolve_startup_report_path(report.startup_report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_json_safe(asdict(report)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    """Run startup validation once and exit non-zero on failure."""
    report = build_startup_validation_report()
    output_path = write_startup_validation_report(report)
    if report.startup_validation_passed:
        print(f"Startup validation passed: {output_path}")
        raise SystemExit(0)
    print(f"Startup validation failed: {output_path}")
    for error in report.errors:
        print(f"- {error}")
    raise SystemExit(1)


def _check(name: str, passed: bool, detail: str) -> StartupValidationCheck:
    return StartupValidationCheck(name=name, passed=passed, detail=detail)


if __name__ == "__main__":
    main()
