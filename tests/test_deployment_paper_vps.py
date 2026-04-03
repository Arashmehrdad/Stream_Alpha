"""Focused tests for the paper-mode VPS deployment helper."""

from __future__ import annotations

import socket
from types import SimpleNamespace
from pathlib import Path

from app.deployment.paper_vps import (
    DEFAULT_PAPER_TRADING_CONFIG,
    DEFAULT_REMOTE_APP_DIR,
    DEFAULT_STARTUP_REPORT_PATH,
    _wrap_ssh_connect_error,
    build_deploy_plan,
    build_remote_env_text,
    load_env_lines,
    resolve_vps_connection,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_resolve_vps_connection_supports_existing_root_env_aliases(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            (
                "ipaddress=203.0.113.10",
                "username=streamalpha",
                "password=super-secret",
                "",
            )
        ),
        encoding="utf-8",
    )

    connection = resolve_vps_connection(load_env_lines(env_path))

    assert connection.host == "203.0.113.10"
    assert connection.user == "streamalpha"
    assert connection.password == "super-secret"
    assert connection.port == 22
    assert connection.remote_app_dir == DEFAULT_REMOTE_APP_DIR


def test_build_remote_env_text_strips_vps_keys_and_forces_paper_runtime(tmp_path: Path) -> None:
    env_path = _write_temp_env(
        tmp_path,
        [
            "APP_NAME=streamalpha",
            "POSTGRES_HOST=postgres",
            "STREAMALPHA_VPS_HOST=203.0.113.20",
            "STREAMALPHA_VPS_USER=streamalpha",
            "STREAMALPHA_VPS_PASSWORD=super-secret",
            "STREAMALPHA_RUNTIME_PROFILE=dev",
            "STREAMALPHA_TRADING_CONFIG_PATH=configs/paper_trading.shadow.yaml",
            "",
        ],
    )
    env_lines = load_env_lines(env_path)

    rendered = build_remote_env_text(env_lines)

    assert "STREAMALPHA_VPS_HOST=" not in rendered
    assert "STREAMALPHA_VPS_USER=" not in rendered
    assert "STREAMALPHA_VPS_PASSWORD=" not in rendered
    assert "APP_NAME=streamalpha" in rendered
    assert "POSTGRES_HOST=postgres" in rendered
    assert "STREAMALPHA_RUNTIME_PROFILE=paper" in rendered
    assert f"STREAMALPHA_TRADING_CONFIG_PATH={DEFAULT_PAPER_TRADING_CONFIG}" in rendered
    assert (
        f"STREAMALPHA_STARTUP_REPORT_PATH={DEFAULT_STARTUP_REPORT_PATH}" in rendered
    )


def test_build_deploy_plan_discovers_bounded_upload_set() -> None:
    env_path = _write_temp_env(
        _REPO_ROOT,
        [
            "STREAMALPHA_VPS_HOST=203.0.113.30",
            "STREAMALPHA_VPS_USER=streamalpha",
            "STREAMALPHA_VPS_PASSWORD=super-secret",
            "",
        ],
        name=".test-paper-vps.env",
    )
    try:
        plan = build_deploy_plan(
            repo_root=_REPO_ROOT,
            env_path=env_path,
            lookback_candles=128,
        )
    finally:
        env_path.unlink(missing_ok=True)

    assert plan["remote_host"] == "203.0.113.30"
    assert plan["remote_app_dir"] == DEFAULT_REMOTE_APP_DIR
    assert plan["paper_profile"] == "paper"
    assert plan["trading_config_path"] == DEFAULT_PAPER_TRADING_CONFIG
    assert "app" in plan["upload_entries"]
    assert "configs" in plan["upload_entries"]
    assert "docker-compose.yml" in plan["upload_entries"]
    assert "trader" in plan["started_services"]


def test_wrap_ssh_connect_error_explains_timeout_and_custom_port_path() -> None:
    fake_paramiko = SimpleNamespace(
        AuthenticationException=type("AuthenticationException", (Exception,), {}),
        SSHException=type("SSHException", (Exception,), {}),
        ssh_exception=SimpleNamespace(
            NoValidConnectionsError=type("NoValidConnectionsError", (Exception,), {})
        ),
    )

    error = _wrap_ssh_connect_error(
        host="203.0.113.50",
        port=22,
        error=socket.timeout("timed out"),
        paramiko_module=fake_paramiko,
    )

    message = str(error)
    assert "Timed out connecting to the VPS over SSH" in message
    assert "Host=203.0.113.50 Port=22" in message
    assert "STREAMALPHA_VPS_PORT" in message


def test_wrap_ssh_connect_error_explains_auth_failure() -> None:
    authentication_exception = type("AuthenticationException", (Exception,), {})
    fake_paramiko = SimpleNamespace(
        AuthenticationException=authentication_exception,
        SSHException=type("SSHException", (Exception,), {}),
        ssh_exception=SimpleNamespace(
            NoValidConnectionsError=type("NoValidConnectionsError", (Exception,), {})
        ),
    )

    error = _wrap_ssh_connect_error(
        host="203.0.113.51",
        port=2222,
        error=authentication_exception("bad credentials"),
        paramiko_module=fake_paramiko,
    )

    message = str(error)
    assert "SSH authentication failed" in message
    assert "username/password" in message


def _write_temp_env(base_dir: Path, lines: list[str], *, name: str = ".tmp-paper-vps.env") -> Path:
    env_path = base_dir / name
    env_path.write_text("\n".join(lines), encoding="utf-8")
    return env_path
