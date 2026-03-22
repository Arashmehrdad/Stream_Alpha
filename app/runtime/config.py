"""Shared runtime profile and startup-report helpers for Stream Alpha M16."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUNTIME_PROFILES = ("dev", "paper", "shadow", "live")
_PROFILE_CONFIG_FILES = {
    "paper": "paper_trading.paper.yaml",
    "shadow": "paper_trading.shadow.yaml",
    "live": "paper_trading.live.yaml",
}


def repo_root() -> Path:
    """Return the repository root from the runtime package."""
    return Path(__file__).resolve().parents[2]


def default_trading_config_path() -> Path:
    """Return the legacy checked-in trading config path."""
    return repo_root() / "configs" / "paper_trading.yaml"


def profile_trading_config_path(profile: str) -> Path:
    """Return the checked-in profile-specific trading config path."""
    normalized_profile = resolve_runtime_profile(profile, default=None)
    if normalized_profile == "dev":
        return default_trading_config_path()
    return repo_root() / "configs" / _PROFILE_CONFIG_FILES[normalized_profile]


def default_startup_report_path() -> Path:
    """Return the default M16 startup validation report path."""
    return repo_root() / "artifacts" / "runtime" / "startup_report.json"


def resolve_runtime_profile(
    value: str | None = None,
    *,
    default: str | None = "paper",
) -> str:
    """Resolve and validate the active runtime profile."""
    candidate = os.getenv("STREAMALPHA_RUNTIME_PROFILE", "") if value is None else value
    normalized = str(candidate).strip().lower()
    if not normalized:
        if default is None:
            raise ValueError(
                "STREAMALPHA_RUNTIME_PROFILE must be set to one of: "
                + ", ".join(RUNTIME_PROFILES)
            )
        normalized = default
    if normalized not in RUNTIME_PROFILES:
        raise ValueError(
            "STREAMALPHA_RUNTIME_PROFILE must be one of: "
            + ", ".join(RUNTIME_PROFILES)
        )
    return normalized


def resolve_trading_config_path(
    value: str | Path | None = None,
    *,
    default_profile: str = "paper",
    use_profile_default: bool = False,
) -> Path:
    """Resolve the active trading config path from env or a local default."""
    if value is None:
        raw_value = os.getenv("STREAMALPHA_TRADING_CONFIG_PATH", "").strip()
    else:
        raw_value = str(value).strip()
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    if use_profile_default:
        return profile_trading_config_path(default_profile).resolve()
    return default_trading_config_path().resolve()


def resolve_startup_report_path(value: str | Path | None = None) -> Path:
    """Resolve the active startup report path from env or the default."""
    if value is None:
        raw_value = os.getenv("STREAMALPHA_STARTUP_REPORT_PATH", "").strip()
    else:
        raw_value = str(value).strip()
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return default_startup_report_path().resolve()


def load_startup_report(path: str | Path | None = None) -> dict[str, Any] | None:
    """Load the current startup report when it exists."""
    report_path = resolve_startup_report_path(path)
    if not report_path.is_file():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Startup report must deserialize into a mapping: {report_path}"
        )
    return payload


@dataclass(frozen=True, slots=True)
class RuntimeMetadata:
    """Additive runtime metadata exposed to services and operator surfaces."""

    runtime_profile: str
    execution_mode: str | None
    startup_validation_passed: bool | None
    startup_report_path: str


def build_runtime_metadata(*, execution_mode: str | None) -> RuntimeMetadata:
    """Return normalized runtime metadata from env and the startup report."""
    runtime_profile = resolve_runtime_profile()
    startup_report_path = resolve_startup_report_path()
    startup_validation_passed: bool | None = None
    try:
        report = load_startup_report(startup_report_path)
    except (OSError, ValueError, json.JSONDecodeError):
        report = None
    if report is not None:
        startup_validation_passed = bool(
            report.get("startup_validation_passed")
        )
    return RuntimeMetadata(
        runtime_profile=runtime_profile,
        execution_mode=execution_mode,
        startup_validation_passed=startup_validation_passed,
        startup_report_path=str(startup_report_path),
    )
