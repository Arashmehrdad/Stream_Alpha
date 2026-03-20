"""Runtime loading and exact-row regime resolution for M9."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.regime.dataset import RegimeSourceRow
from app.regime.service import REGIME_LABELS, SymbolThresholds, classify_row


THRESHOLDS_SCHEMA_VERSION = "m8_thresholds_v1"
SIGNAL_POLICY_SCHEMA_VERSION = "m9_regime_signal_policy_v1"
REQUIRED_INPUTS = ("realized_vol_12", "momentum_3", "macd_line_12_26")


def repo_root() -> Path:
    """Return the repository root from the current module location."""
    return Path(__file__).resolve().parents[2]


def default_thresholds_root() -> Path:
    """Return the default M8 thresholds artifact root."""
    return repo_root() / "artifacts" / "regime" / "m8"


def default_signal_policy_path() -> Path:
    """Return the checked-in default M9 regime signal policy path."""
    return repo_root() / "configs" / "regime_signal_policy.json"


@dataclass(frozen=True, slots=True)
class RegimePolicy:
    """Per-regime M9 signal policy."""

    regime_label: str
    buy_prob_up: float
    sell_prob_up: float
    allow_new_long_entries: bool


@dataclass(frozen=True, slots=True)
class LoadedThresholdArtifact:
    """Validated M8 thresholds artifact ready for runtime use."""

    schema_version: str
    run_id: str
    source_table: str
    source_exchange: str
    interval_minutes: int
    required_inputs: tuple[str, ...]
    artifact_path: str
    thresholds_by_symbol: dict[str, SymbolThresholds]


@dataclass(frozen=True, slots=True)
class ResolvedRegime:
    """Exact-row regime resolution payload shared across M4 and M5."""

    symbol: str
    interval_begin: str
    as_of_time: str
    row_id: str
    regime_label: str
    regime_run_id: str
    regime_artifact_path: str
    high_vol_threshold: float
    trend_abs_threshold: float
    realized_vol_12: float
    momentum_3: float
    macd_line_12_26: float


@dataclass(frozen=True, slots=True)
class LiveRegimeRuntime:
    """Runtime wrapper around the saved M8 thresholds and M9 signal policy."""

    artifact: LoadedThresholdArtifact
    policies: dict[str, RegimePolicy]

    @property
    def run_id(self) -> str:
        """Return the loaded M8 run identifier."""
        return self.artifact.run_id

    @property
    def artifact_path(self) -> str:
        """Return the resolved thresholds artifact path."""
        return self.artifact.artifact_path

    def policy_for(self, regime_label: str) -> RegimePolicy:
        """Return the configured M9 signal policy for one regime label."""
        try:
            return self.policies[regime_label]
        except KeyError as error:
            raise ValueError(f"No M9 signal policy was configured for {regime_label}") from error

    def validate_runtime_compatibility(
        self,
        *,
        source_table: str,
        source_exchange: str,
        interval_minutes: int,
        symbols: tuple[str, ...],
    ) -> None:
        """Validate that the loaded thresholds artifact matches the live runtime."""
        if self.artifact.source_table != source_table:
            raise ValueError(
                "Regime thresholds artifact source_table does not match the live runtime. "
                f"Expected {source_table}, found {self.artifact.source_table}"
            )
        if self.artifact.source_exchange != source_exchange:
            raise ValueError(
                "Regime thresholds artifact source_exchange does not match the live runtime. "
                f"Expected {source_exchange}, found {self.artifact.source_exchange}"
            )
        if self.artifact.interval_minutes != interval_minutes:
            raise ValueError(
                "Regime thresholds artifact interval_minutes does not match the live runtime. "
                f"Expected {interval_minutes}, found {self.artifact.interval_minutes}"
            )
        missing_symbols = sorted(set(symbols) - set(self.artifact.thresholds_by_symbol))
        if missing_symbols:
            raise ValueError(
                "Regime thresholds artifact is missing configured symbols. "
                f"Missing: {missing_symbols}"
            )

    def resolve_feature_row_regime(self, row: dict[str, Any]) -> ResolvedRegime:
        """Resolve the regime for the exact canonical feature row used by M4."""
        missing_inputs = [
            column
            for column in ("symbol", "interval_begin", "as_of_time", *REQUIRED_INPUTS)
            if column not in row or row[column] is None
        ]
        if missing_inputs:
            raise ValueError(
                "Feature row is missing required regime inputs. "
                f"Missing columns: {missing_inputs}"
            )
        source_row = RegimeSourceRow(
            symbol=str(row["symbol"]),
            interval_begin=row["interval_begin"],
            as_of_time=row["as_of_time"],
            realized_vol_12=float(row["realized_vol_12"]),
            momentum_3=float(row["momentum_3"]),
            macd_line_12_26=float(row["macd_line_12_26"]),
        )
        regime_label = classify_row(source_row, self.artifact.thresholds_by_symbol)
        thresholds = self.artifact.thresholds_by_symbol[source_row.symbol]
        interval_begin = _to_rfc3339(source_row.interval_begin)
        as_of_time = _to_rfc3339(source_row.as_of_time)
        return ResolvedRegime(
            symbol=source_row.symbol,
            interval_begin=interval_begin,
            as_of_time=as_of_time,
            row_id=f"{source_row.symbol}|{interval_begin}",
            regime_label=regime_label,
            regime_run_id=self.artifact.run_id,
            regime_artifact_path=self.artifact.artifact_path,
            high_vol_threshold=thresholds.high_vol_threshold,
            trend_abs_threshold=thresholds.trend_abs_threshold,
            realized_vol_12=source_row.realized_vol_12,
            momentum_3=source_row.momentum_3,
            macd_line_12_26=source_row.macd_line_12_26,
        )


def load_live_regime_runtime(
    *,
    thresholds_path: str,
    signal_policy_path: str,
) -> LiveRegimeRuntime:
    """Load the runtime M8 thresholds and checked-in M9 signal policy."""
    artifact = load_thresholds_artifact(thresholds_path)
    policies = load_signal_policy(signal_policy_path)
    return LiveRegimeRuntime(artifact=artifact, policies=policies)


def resolve_thresholds_artifact_path(
    explicit_path: str,
    *,
    thresholds_root: Path | None = None,
) -> Path:
    """Resolve the explicit or latest thresholds artifact path."""
    if explicit_path.strip():
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Regime thresholds artifact does not exist: {path}")
        return path

    root = default_thresholds_root() if thresholds_root is None else Path(thresholds_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Regime thresholds root does not exist: {root}")

    candidates = sorted(
        (
            run_dir / "thresholds.json"
            for run_dir in root.iterdir()
            if run_dir.is_dir() and (run_dir / "thresholds.json").is_file()
        ),
        reverse=True,
    )
    if not candidates:
        raise ValueError(
            "No regime thresholds artifacts were found under "
            f"{root}. Run the M8 offline regime workflow first."
        )
    return candidates[0].resolve()


def load_thresholds_artifact(explicit_path: str) -> LoadedThresholdArtifact:
    """Load and validate one saved M8 thresholds artifact."""
    artifact_path = resolve_thresholds_artifact_path(explicit_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != THRESHOLDS_SCHEMA_VERSION:
        raise ValueError(
            "Regime thresholds artifact schema_version is not supported. "
            f"Expected {THRESHOLDS_SCHEMA_VERSION}, found {schema_version}"
        )

    required_inputs = tuple(str(value) for value in payload.get("required_inputs", []))
    missing_inputs = sorted(set(REQUIRED_INPUTS) - set(required_inputs))
    if missing_inputs:
        raise ValueError(
            "Regime thresholds artifact is missing required inputs. "
            f"Missing: {missing_inputs}"
        )

    payload_labels = tuple(str(value) for value in payload.get("regime_labels", []))
    if set(payload_labels) != set(REGIME_LABELS):
        raise ValueError(
            "Regime thresholds artifact regime_labels do not match the supported labels. "
            f"Found: {payload_labels}"
        )

    thresholds_payload = payload.get("thresholds_by_symbol")
    if not isinstance(thresholds_payload, dict) or not thresholds_payload:
        raise ValueError("Regime thresholds artifact thresholds_by_symbol must be populated")

    thresholds_by_symbol: dict[str, SymbolThresholds] = {}
    for symbol, threshold_payload in thresholds_payload.items():
        thresholds_by_symbol[str(symbol)] = SymbolThresholds(
            symbol=str(threshold_payload["symbol"]),
            fitted_row_count=int(threshold_payload["fitted_row_count"]),
            high_vol_threshold=float(threshold_payload["high_vol_threshold"]),
            trend_abs_threshold=float(threshold_payload["trend_abs_threshold"]),
        )

    return LoadedThresholdArtifact(
        schema_version=schema_version,
        run_id=str(payload["run_id"]),
        source_table=str(payload["source_table"]),
        source_exchange=str(payload["source_exchange"]),
        interval_minutes=int(payload["interval_minutes"]),
        required_inputs=required_inputs,
        artifact_path=str(artifact_path),
        thresholds_by_symbol=thresholds_by_symbol,
    )


def load_signal_policy(explicit_path: str) -> dict[str, RegimePolicy]:
    """Load and validate the checked-in M9 regime signal policy."""
    policy_path = (
        default_signal_policy_path()
        if not explicit_path.strip()
        else Path(explicit_path).expanduser().resolve()
    )
    if not policy_path.is_file():
        raise ValueError(f"Regime signal policy does not exist: {policy_path}")

    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != SIGNAL_POLICY_SCHEMA_VERSION:
        raise ValueError(
            "Regime signal policy schema_version is not supported. "
            f"Expected {SIGNAL_POLICY_SCHEMA_VERSION}, found {schema_version}"
        )

    raw_policies = payload.get("policies")
    if not isinstance(raw_policies, dict):
        raise ValueError("Regime signal policy must define a policies object")

    missing_policies = sorted(set(REGIME_LABELS) - set(raw_policies))
    if missing_policies:
        raise ValueError(
            "Regime signal policy is missing required regime labels. "
            f"Missing: {missing_policies}"
        )

    policies: dict[str, RegimePolicy] = {}
    for regime_label in REGIME_LABELS:
        policy_payload = raw_policies[regime_label]
        buy_prob_up = float(policy_payload["buy_prob_up"])
        sell_prob_up = float(policy_payload["sell_prob_up"])
        if sell_prob_up > buy_prob_up:
            raise ValueError(
                f"Regime signal policy sell threshold cannot exceed buy threshold for {regime_label}"
            )
        policies[regime_label] = RegimePolicy(
            regime_label=regime_label,
            buy_prob_up=buy_prob_up,
            sell_prob_up=sell_prob_up,
            allow_new_long_entries=bool(policy_payload["allow_new_long_entries"]),
        )
    return policies


def _to_rfc3339(value: Any) -> str:
    return value.isoformat().replace("+00:00", "Z")
