"""Repo-native Linux VPS deployment helpers for paper mode and challenger observation."""

from __future__ import annotations

import argparse
import io
import json
import os
import posixpath
import shlex
import socket
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ENV_PATH = Path(".env")
DEFAULT_REMOTE_APP_DIR = "~/stream_alpha_paper"
DEFAULT_PAPER_TRADING_CONFIG = "configs/paper_trading.paper.yaml"
DEFAULT_STARTUP_REPORT_PATH = "artifacts/runtime/startup_report.json"
DEFAULT_BACKFILL_LOOKBACK_CANDLES = 128
DEFAULT_RESEARCH_CHALLENGER_ARTIFACT_DIR = (
    "/workspace/artifacts/paper_trading/paper/research/policy_challengers"
)
DEFAULT_UPLOAD_ROOTS = (
    "app",
    "configs",
    "dashboards",
    "docker",
    "scripts",
    "artifacts/registry",
    "artifacts/regime",
    "docker-compose.yml",
    "requirements.txt",
    "README.md",
    ".env.example",
)
REMOTE_RUNTIME_ENV_OVERRIDES = {
    "STREAMALPHA_RUNTIME_PROFILE": "paper",
    "STREAMALPHA_TRADING_CONFIG_PATH": DEFAULT_PAPER_TRADING_CONFIG,
    "STREAMALPHA_STARTUP_REPORT_PATH": DEFAULT_STARTUP_REPORT_PATH,
}
VPS_HOST_ALIASES = (
    "STREAMALPHA_VPS_HOST",
    "VPS_HOST",
    "SSH_HOST",
    "ipaddress",
)
VPS_USER_ALIASES = (
    "STREAMALPHA_VPS_USER",
    "VPS_USER",
    "SSH_USER",
    "username",
)
VPS_PASSWORD_ALIASES = (
    "STREAMALPHA_VPS_PASSWORD",
    "VPS_PASSWORD",
    "SSH_PASSWORD",
    "password",
)
VPS_PORT_ALIASES = (
    "STREAMALPHA_VPS_PORT",
    "VPS_PORT",
    "SSH_PORT",
)
VPS_APP_DIR_ALIASES = (
    "STREAMALPHA_VPS_APP_DIR",
    "VPS_APP_DIR",
    "REMOTE_APP_DIR",
)
LOCAL_ONLY_ENV_ALIASES = frozenset(
    {
        *VPS_HOST_ALIASES,
        *VPS_USER_ALIASES,
        *VPS_PASSWORD_ALIASES,
        *VPS_PORT_ALIASES,
        *VPS_APP_DIR_ALIASES,
    }
)


@dataclass(frozen=True, slots=True)
class EnvLine:
    """One parsed line from a dotenv-style file."""

    raw: str
    key: str | None = None
    value: str | None = None


@dataclass(frozen=True, slots=True)
class VpsConnectionConfig:
    """Normalized VPS connection settings loaded from the local root .env."""

    host: str
    user: str
    password: str
    port: int
    remote_app_dir: str


@dataclass(frozen=True, slots=True)
class DockerAccessConfig:
    """How to run Docker commands on the remote host."""

    requires_sudo: bool


@dataclass(frozen=True, slots=True)
class RemoteCommandResult:
    """One remote shell command result."""

    command: str
    exit_status: int
    stdout: str
    stderr: str


def load_env_lines(env_path: Path) -> list[EnvLine]:
    """Load dotenv-style lines while preserving ordering and comments."""
    if not env_path.is_file():
        raise ValueError(f"Missing env file: {env_path}")
    lines: list[EnvLine] = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            lines.append(EnvLine(raw=raw_line))
            continue
        key, value = raw_line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            lines.append(EnvLine(raw=raw_line))
            continue
        lines.append(
            EnvLine(
                raw=raw_line,
                key=normalized_key,
                value=value,
            )
        )
    return lines


def resolve_vps_connection(env_lines: Iterable[EnvLine]) -> VpsConnectionConfig:
    """Resolve normalized VPS connection settings from supported aliases."""
    env_map = {
        line.key: line.value
        for line in env_lines
        if line.key is not None and line.value is not None
    }
    host = _resolve_first_alias(env_map, VPS_HOST_ALIASES)
    user = _resolve_first_alias(env_map, VPS_USER_ALIASES)
    password = _resolve_first_alias(env_map, VPS_PASSWORD_ALIASES)
    if host is None:
        raise ValueError("Missing VPS host in .env. Supported aliases include STREAMALPHA_VPS_HOST and ipaddress.")
    if user is None:
        raise ValueError("Missing VPS user in .env. Supported aliases include STREAMALPHA_VPS_USER and username.")
    if password is None:
        raise ValueError("Missing VPS password in .env. Supported aliases include STREAMALPHA_VPS_PASSWORD and password.")
    port_value = _resolve_first_alias(env_map, VPS_PORT_ALIASES)
    remote_app_dir = (
        _resolve_first_alias(env_map, VPS_APP_DIR_ALIASES) or DEFAULT_REMOTE_APP_DIR
    )
    return VpsConnectionConfig(
        host=host,
        user=user,
        password=password,
        port=int(port_value) if port_value is not None else 22,
        remote_app_dir=remote_app_dir,
    )


def build_remote_env_text(env_lines: Iterable[EnvLine]) -> str:
    """Build the sanitized remote .env for paper-mode VPS deployment."""
    filtered_lines: list[str] = []
    seen_keys: set[str] = set()
    for line in env_lines:
        if line.key is None:
            filtered_lines.append(line.raw)
            continue
        if line.key in LOCAL_ONLY_ENV_ALIASES:
            continue
        replacement_value = REMOTE_RUNTIME_ENV_OVERRIDES.get(line.key, line.value)
        seen_keys.add(line.key)
        filtered_lines.append(f"{line.key}={replacement_value}")
    for key, value in REMOTE_RUNTIME_ENV_OVERRIDES.items():
        if key not in seen_keys:
            filtered_lines.append(f"{key}={value}")
    return "\n".join(filtered_lines).rstrip() + "\n"


def discover_upload_paths(repo_root: Path) -> list[Path]:
    """Return the bounded deployment file/dir set required for the paper VPS."""
    resolved_paths: list[Path] = []
    missing_paths: list[str] = []
    for relative_path in DEFAULT_UPLOAD_ROOTS:
        candidate_path = repo_root / relative_path
        if candidate_path.exists():
            resolved_paths.append(candidate_path)
        else:
            missing_paths.append(relative_path)
    if missing_paths:
        raise ValueError(
            "Missing required deployment paths: " + ", ".join(sorted(missing_paths))
        )
    return resolved_paths


def create_upload_bundle(repo_root: Path, upload_paths: Iterable[Path]) -> Path:
    """Create a temporary tar.gz bundle containing the bounded deployment set."""
    temp_file = tempfile.NamedTemporaryFile(
        prefix="streamalpha-paper-vps-",
        suffix=".tar.gz",
        delete=False,
    )
    temp_file.close()
    bundle_path = Path(temp_file.name)
    with tarfile.open(bundle_path, "w:gz") as archive:
        for path in upload_paths:
            archive.add(
                str(path),
                arcname=str(path.resolve().relative_to(repo_root.resolve())),
            )
    return bundle_path


def build_deploy_plan(
    *,
    repo_root: Path,
    env_path: Path,
    lookback_candles: int,
) -> dict[str, Any]:
    """Build the local deployment plan without touching the remote host."""
    env_lines = load_env_lines(env_path)
    connection = resolve_vps_connection(env_lines)
    upload_paths = discover_upload_paths(repo_root)
    return {
        "action": "deploy",
        "remote_host": connection.host,
        "remote_app_dir": connection.remote_app_dir,
        "remote_port": connection.port,
        "paper_profile": "paper",
        "trading_config_path": DEFAULT_PAPER_TRADING_CONFIG,
        "startup_report_path": DEFAULT_STARTUP_REPORT_PATH,
        "lookback_candles": int(lookback_candles),
        "upload_entries": [
            str(path.resolve().relative_to(repo_root.resolve())) for path in upload_paths
        ],
        "started_services": [
            "redpanda",
            "redpanda-console",
            "postgres",
            "config-check",
            "producer",
            "features",
            "inference",
            "trader",
            "dashboard",
        ],
    }


class RemoteSession:
    """Small Paramiko-backed remote session wrapper."""

    def __init__(self, connection: VpsConnectionConfig) -> None:
        self.connection = connection
        self._client = None
        self._sftp = None

    def __enter__(self) -> "RemoteSession":
        try:
            import paramiko  # pylint: disable=import-outside-toplevel
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ValueError(
                "Paramiko is required for VPS deployment scripts. Install repo requirements first."
            ) from error
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=self.connection.host,
                port=self.connection.port,
                username=self.connection.user,
                password=self.connection.password,
                look_for_keys=False,
                allow_agent=False,
                timeout=30,
                auth_timeout=30,
                banner_timeout=30,
            )
        except Exception as error:  # pragma: no cover - network-dependent
            raise _wrap_ssh_connect_error(
                host=self.connection.host,
                port=self.connection.port,
                error=error,
                paramiko_module=paramiko,
            ) from error
        self._client = client
        self._sftp = client.open_sftp()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._sftp is not None:
            self._sftp.close()
        if self._client is not None:
            self._client.close()
        self._sftp = None
        self._client = None

    def run(
        self,
        command: str,
        *,
        sudo: bool = False,
        check: bool = True,
    ) -> RemoteCommandResult:
        """Run one remote bash command, optionally through sudo."""
        if self._client is None:
            raise RuntimeError("RemoteSession is not connected.")
        wrapped = f"bash -lc {shlex.quote(command)}"
        if sudo:
            wrapped = f"sudo -S -p '' {wrapped}"
        stdin, stdout, stderr = self._client.exec_command(wrapped, get_pty=sudo)
        if sudo:
            stdin.write(self.connection.password + "\n")
            stdin.flush()
        exit_status = stdout.channel.recv_exit_status()
        result = RemoteCommandResult(
            command=command,
            exit_status=exit_status,
            stdout=stdout.read().decode("utf-8", errors="replace"),
            stderr=stderr.read().decode("utf-8", errors="replace"),
        )
        if check and exit_status != 0:
            raise RuntimeError(
                f"Remote command failed with exit {exit_status}: {command}\n"
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        return result

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload one file via SFTP."""
        if self._sftp is None:
            raise RuntimeError("RemoteSession is not connected.")
        self._sftp.put(str(local_path), remote_path)

    def upload_text(self, text: str, remote_path: str) -> None:
        """Upload a UTF-8 text file via SFTP."""
        if self._sftp is None:
            raise RuntimeError("RemoteSession is not connected.")
        with self._sftp.file(remote_path, "wb") as remote_file:
            remote_file.write(text.encode("utf-8"))


def deploy_paper_vps(
    *,
    repo_root: Path,
    env_path: Path,
    lookback_candles: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Deploy the bounded paper-mode stack plus challenger observation to the VPS."""
    plan = build_deploy_plan(
        repo_root=repo_root,
        env_path=env_path,
        lookback_candles=lookback_candles,
    )
    if dry_run:
        return {
            **plan,
            "status": "dry_run",
            "tail_logs_hint": "ssh to the configured VPS and run: cd <remote_app_dir> && docker compose logs -f trader inference features producer",
            "inspect_challengers_hint": ".\\scripts\\show_live_policy_challengers_vps.ps1",
        }

    env_lines = load_env_lines(env_path)
    connection = resolve_vps_connection(env_lines)
    remote_env_text = build_remote_env_text(env_lines)
    upload_paths = discover_upload_paths(repo_root)
    bundle_path = create_upload_bundle(repo_root, upload_paths)
    try:
        with RemoteSession(connection) as session:
            remote_app_dir = _resolve_remote_app_dir(session, connection.remote_app_dir)
            docker_access = _ensure_remote_docker(session)
            remote_bundle_path = posixpath.join(remote_app_dir, ".streamalpha-paper.tar.gz")
            session.run(f"mkdir -p {shlex.quote(remote_app_dir)}")
            session.upload_file(bundle_path, remote_bundle_path)
            session.upload_text(remote_env_text, posixpath.join(remote_app_dir, ".env"))
            session.run(
                " && ".join(
                    [
                        f"cd {shlex.quote(remote_app_dir)}",
                        "rm -rf app configs dashboards docker scripts docker-compose.yml requirements.txt README.md .env.example artifacts/registry artifacts/regime",
                        "mkdir -p artifacts",
                        f"tar -xzf {shlex.quote(remote_bundle_path)} -C .",
                        f"rm -f {shlex.quote(remote_bundle_path)}",
                        "mkdir -p artifacts/paper_trading/paper/research/policy_challengers artifacts/runtime artifacts/operations artifacts/reliability artifacts/rationale artifacts/adaptation artifacts/continual_learning",
                    ]
                )
            )
            _run_remote_compose(
                session,
                docker_access=docker_access,
                remote_app_dir=remote_app_dir,
                compose_args=["--profile", "paper", "--env-file", ".env", "up", "-d", "--build"],
            )
            _run_remote_compose(
                session,
                docker_access=docker_access,
                remote_app_dir=remote_app_dir,
                compose_args=[
                    "--profile",
                    "paper",
                    "--env-file",
                    ".env",
                    "run",
                    "--rm",
                    "producer",
                    "python",
                    "-m",
                    "app.ingestion.backfill_ohlc",
                    "--lookback-candles",
                    str(int(lookback_candles)),
                ],
            )
            status = _collect_status(
                session,
                docker_access=docker_access,
                remote_app_dir=remote_app_dir,
                remote_host=connection.host,
            )
            return {
                **status,
                "action": "deploy",
                "tail_logs_hint": (
                    "ssh to the configured VPS and run: "
                    f"cd {remote_app_dir} && docker compose logs -f trader inference features producer"
                ),
                "inspect_challengers_hint": ".\\scripts\\show_live_policy_challengers_vps.ps1",
            }
    finally:
        bundle_path.unlink(missing_ok=True)


def status_paper_vps(
    *,
    env_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Return the current remote paper-stack status summary."""
    env_lines = load_env_lines(env_path)
    connection = resolve_vps_connection(env_lines)
    if dry_run:
        return {
            "action": "status",
            "status": "dry_run",
            "remote_host": connection.host,
            "remote_app_dir": connection.remote_app_dir,
            "command_hint": "docker compose ps plus challenger artifact existence checks",
        }
    with RemoteSession(connection) as session:
        remote_app_dir = _resolve_remote_app_dir(session, connection.remote_app_dir)
        docker_access = _ensure_remote_docker(session)
        return _collect_status(
            session,
            docker_access=docker_access,
            remote_app_dir=remote_app_dir,
            remote_host=connection.host,
        )


def stop_paper_vps(
    *,
    env_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Stop the deployed paper stack on the VPS without touching anything else."""
    env_lines = load_env_lines(env_path)
    connection = resolve_vps_connection(env_lines)
    if dry_run:
        return {
            "action": "stop",
            "status": "dry_run",
            "remote_host": connection.host,
            "remote_app_dir": connection.remote_app_dir,
            "command_hint": "docker compose down --remove-orphans",
        }
    with RemoteSession(connection) as session:
        remote_app_dir = _resolve_remote_app_dir(session, connection.remote_app_dir)
        docker_access = _ensure_remote_docker(session)
        _run_remote_compose(
            session,
            docker_access=docker_access,
            remote_app_dir=remote_app_dir,
            compose_args=["--profile", "paper", "--env-file", ".env", "down", "--remove-orphans"],
        )
        return {
            "action": "stop",
            "remote_host": connection.host,
            "remote_app_dir": remote_app_dir,
            "stopped": True,
        }


def show_live_policy_challengers_vps(
    *,
    env_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Rebuild and return the remote challenger scoreboard from the running paper stack."""
    env_lines = load_env_lines(env_path)
    connection = resolve_vps_connection(env_lines)
    if dry_run:
        return {
            "action": "show-challengers",
            "status": "dry_run",
            "remote_host": connection.host,
            "remote_app_dir": connection.remote_app_dir,
            "command_hint": "docker compose exec -T trader python -m app.training.live_policy_challenger --json",
        }
    with RemoteSession(connection) as session:
        remote_app_dir = _resolve_remote_app_dir(session, connection.remote_app_dir)
        docker_access = _ensure_remote_docker(session)
        scoreboard_result = _run_remote_compose(
            session,
            docker_access=docker_access,
            remote_app_dir=remote_app_dir,
            compose_args=[
                "--profile",
                "paper",
                "--env-file",
                ".env",
                "exec",
                "-T",
                "trader",
                "python",
                "-m",
                "app.training.live_policy_challenger",
                "--artifact-dir",
                DEFAULT_RESEARCH_CHALLENGER_ARTIFACT_DIR,
                "--json",
            ],
        )
        summary = json.loads(scoreboard_result.stdout)
        return {
            "action": "show-challengers",
            "remote_host": connection.host,
            "remote_app_dir": remote_app_dir,
            "scoreboard": summary,
        }


def _resolve_first_alias(
    env_map: dict[str, str | None],
    aliases: Iterable[str],
) -> str | None:
    for alias in aliases:
        candidate_value = env_map.get(alias)
        if candidate_value is None:
            continue
        if candidate_value.strip():
            return candidate_value.strip()
    return None


def _wrap_ssh_connect_error(
    *,
    host: str,
    port: int,
    error: Exception,
    paramiko_module: Any,
) -> ValueError:
    """Convert raw Paramiko/socket failures into actionable operator messages."""
    if isinstance(error, (TimeoutError, socket.timeout)):
        return ValueError(
            "Timed out connecting to the VPS over SSH. "
            f"Host={host} Port={port}. "
            "Check that the VPS is powered on, SSH is listening, and the firewall/security group allows the configured SSH port. "
            "If the VPS uses a custom SSH port, add STREAMALPHA_VPS_PORT=<port> to the root .env."
        )
    if isinstance(error, paramiko_module.AuthenticationException):
        return ValueError(
            "SSH authentication failed for the configured VPS user/password. "
            "Check the root .env VPS username/password values or switch the VPS to password SSH if the host only allows keys."
        )
    no_valid_connections = getattr(paramiko_module.ssh_exception, "NoValidConnectionsError", None)
    if no_valid_connections is not None and isinstance(error, no_valid_connections):
        return ValueError(
            "The VPS rejected the SSH TCP connection. "
            f"Host={host} Port={port}. "
            "Check that SSH is listening on the configured port and that the VPS firewall allows inbound SSH."
        )
    ssh_exception = getattr(paramiko_module, "SSHException", None)
    if ssh_exception is not None and isinstance(error, ssh_exception):
        return ValueError(
            "SSH transport setup failed while connecting to the VPS. "
            f"Host={host} Port={port}. "
            f"Details: {error}"
        )
    return ValueError(
        "Unable to connect to the VPS over SSH. "
        f"Host={host} Port={port}. "
        f"Details: {error}"
    )


def _resolve_remote_app_dir(session: RemoteSession, configured_path: str) -> str:
    home_directory = session.run("printf %s \"$HOME\"").stdout.strip()
    if configured_path.startswith("~/"):
        return posixpath.join(home_directory, configured_path[2:])
    if configured_path.startswith("/"):
        return configured_path
    return posixpath.join(home_directory, configured_path)


def _ensure_remote_docker(session: RemoteSession) -> DockerAccessConfig:
    docker_access = _detect_remote_docker_access(session)
    if docker_access is not None:
        return docker_access
    distro_id = session.run(". /etc/os-release && printf %s \"$ID\"").stdout.strip().lower()
    if distro_id not in {"ubuntu", "debian"}:
        raise ValueError(
            f"Unsupported VPS distro for the bounded deploy path: {distro_id or 'unknown'}"
        )
    curl_check = session.run("command -v curl >/dev/null 2>&1", check=False)
    if curl_check.exit_status != 0:
        session.run("apt-get update && apt-get install -y curl ca-certificates", sudo=True)
    session.run(
        "curl -fsSL https://get.docker.com -o /tmp/get-docker.sh && sh /tmp/get-docker.sh",
        sudo=True,
    )
    session.run(
        "systemctl enable --now docker && if ! docker compose version >/dev/null 2>&1; then apt-get update && apt-get install -y docker-compose-plugin; fi",
        sudo=True,
    )
    docker_access = _detect_remote_docker_access(session)
    if docker_access is None:
        raise RuntimeError("Docker installation completed but docker compose is still unavailable on the VPS.")
    return docker_access


def _detect_remote_docker_access(session: RemoteSession) -> DockerAccessConfig | None:
    for requires_sudo in (False, True):
        result = session.run("docker compose version", sudo=requires_sudo, check=False)
        if result.exit_status == 0:
            return DockerAccessConfig(requires_sudo=requires_sudo)
    return None


def _run_remote_compose(
    session: RemoteSession,
    *,
    docker_access: DockerAccessConfig,
    remote_app_dir: str,
    compose_args: list[str],
) -> RemoteCommandResult:
    compose_command = "docker compose " + " ".join(shlex.quote(argument) for argument in compose_args)
    return session.run(
        f"cd {shlex.quote(remote_app_dir)} && {compose_command}",
        sudo=docker_access.requires_sudo,
    )


def _collect_status(
    session: RemoteSession,
    *,
    docker_access: DockerAccessConfig,
    remote_app_dir: str,
    remote_host: str,
) -> dict[str, Any]:
    running_services_result = _run_remote_compose(
        session,
        docker_access=docker_access,
        remote_app_dir=remote_app_dir,
        compose_args=["--profile", "paper", "--env-file", ".env", "ps", "--services", "--filter", "status=running"],
    )
    running_services = [
        line.strip()
        for line in running_services_result.stdout.splitlines()
        if line.strip()
    ]
    challenger_check = session.run(
        "test -f "
        + shlex.quote(
            posixpath.join(
                remote_app_dir,
                "artifacts/paper_trading/paper/research/policy_challengers/latest_scoreboard.json",
            )
        ),
        sudo=False,
        check=False,
    )
    return {
        "action": "status",
        "remote_host": remote_host,
        "remote_app_dir": remote_app_dir,
        "running_services": running_services,
        "paper_runner_up": "trader" in running_services,
        "challenger_artifacts_exist": challenger_check.exit_status == 0,
        "challenger_scoreboard_remote_path": posixpath.join(
            remote_app_dir,
            "artifacts/paper_trading/paper/research/policy_challengers/latest_scoreboard.json",
        ),
    }


def main() -> None:
    """CLI entrypoint for the repo-native paper VPS helper."""
    parser = argparse.ArgumentParser(
        description="Deploy and inspect Stream Alpha paper mode on a Linux VPS",
    )
    parser.add_argument(
        "action",
        choices=("deploy", "status", "stop", "show-challengers"),
        help="Which VPS workflow action to run.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_PATH),
        help="Local root .env path containing the VPS connection aliases.",
    )
    parser.add_argument(
        "--lookback-candles",
        type=int,
        default=DEFAULT_BACKFILL_LOOKBACK_CANDLES,
        help="Bounded OHLC backfill used after the paper stack starts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the bounded plan without touching the remote VPS.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    arguments = parser.parse_args()

    repo_root = _repo_root()
    env_path = (repo_root / arguments.env_file).resolve()
    try:
        if arguments.action == "deploy":
            result = deploy_paper_vps(
                repo_root=repo_root,
                env_path=env_path,
                lookback_candles=arguments.lookback_candles,
                dry_run=arguments.dry_run,
            )
        elif arguments.action == "status":
            result = status_paper_vps(env_path=env_path, dry_run=arguments.dry_run)
        elif arguments.action == "stop":
            result = stop_paper_vps(env_path=env_path, dry_run=arguments.dry_run)
        else:
            result = show_live_policy_challengers_vps(
                env_path=env_path,
                dry_run=arguments.dry_run,
            )
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(result, sort_keys=True))
        return
    print(json.dumps(result, indent=2, sort_keys=True))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    main()
