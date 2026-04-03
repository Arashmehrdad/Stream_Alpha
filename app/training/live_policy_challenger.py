"""Research-only live paper challenger scoring for top M7 policy candidates."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339
from app.reliability.artifacts import append_jsonl_artifact
from app.trading.config import PaperTradingConfig, load_paper_trading_config
from app.trading.schemas import FeatureCandle, RiskDecision, SignalDecision
from app.training.dataset import load_training_config
from app.training.policy_candidates import LongOnlyPolicyCandidate, find_policy_candidate
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_SHORTLIST_NAMES = (
    "m7_research_long_only_v1",
    "no_long_in_trend_down_high_vol_080",
    "range_or_trend_up_080",
    "per_regime_thresholds_v1",
    "range_only_080",
)
DEFAULT_M7_TRAINING_CONFIG_PATH = Path("configs") / "training.m7.json"
DEFAULT_TRADING_CONFIG_PATH = Path("configs") / "paper_trading.paper.yaml"
DEFAULT_ARTIFACT_SUBDIR = Path("research") / "policy_challengers"
LOW_TRADE_COUNT_WARNING_THRESHOLD = 20


@dataclass(frozen=True, slots=True)
class LivePolicyChallengerConfig:
    """Research-only challenger settings derived from the authoritative M7 config."""

    training_config_path: str
    evaluation_horizon_candles: int
    fee_rate: float
    candidates: tuple[LongOnlyPolicyCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe config payload for research artifacts."""
        return {
            "training_config_path": self.training_config_path,
            "evaluation_horizon_candles": int(self.evaluation_horizon_candles),
            "fee_rate": float(self.fee_rate),
            "candidate_names": [candidate.name for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class CandidateScoreDecision:
    """One research-only candidate decision on one observed live/paper signal row."""

    candidate_name: str
    decision: str
    would_trade: bool
    reason_code: str
    effective_prob_up_threshold: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe decision payload."""
        return {
            "candidate_name": self.candidate_name,
            "decision": self.decision,
            "would_trade": bool(self.would_trade),
            "reason_code": self.reason_code,
            "effective_prob_up_threshold": (
                None
                if self.effective_prob_up_threshold is None
                else float(self.effective_prob_up_threshold)
            ),
        }


@dataclass(frozen=True, slots=True)
class LivePolicyChallengerObservation:
    """One observed signal row scored by the research-only challenger shortlist."""

    observation_id: str
    service_name: str
    execution_mode: str
    symbol: str
    signal_row_id: str
    signal_interval_begin: str
    signal_as_of_time: str
    regime_label: str
    prob_up: float
    current_close_price: float
    target_eval_interval_begin: str
    active_production_decision: str
    active_production_trade_taken: bool
    active_production_reason_code: str
    candidate_decisions: tuple[CandidateScoreDecision, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe observation payload."""
        return {
            "observation_id": self.observation_id,
            "service_name": self.service_name,
            "execution_mode": self.execution_mode,
            "symbol": self.symbol,
            "signal_row_id": self.signal_row_id,
            "signal_interval_begin": self.signal_interval_begin,
            "signal_as_of_time": self.signal_as_of_time,
            "regime_label": self.regime_label,
            "prob_up": float(self.prob_up),
            "current_close_price": float(self.current_close_price),
            "target_eval_interval_begin": self.target_eval_interval_begin,
            "active_production_decision": self.active_production_decision,
            "active_production_trade_taken": bool(self.active_production_trade_taken),
            "active_production_reason_code": self.active_production_reason_code,
            "candidate_decisions": {
                decision.candidate_name: decision.to_dict()
                for decision in self.candidate_decisions
            },
        }


def build_live_policy_challenger_config(
    *,
    training_config_path: Path | None = None,
    shortlist_names: Iterable[str] | None = None,
) -> LivePolicyChallengerConfig:
    """Load the research-only M7 economics contract and challenger shortlist."""
    resolved_config_path = (
        Path(training_config_path).resolve()
        if training_config_path is not None
        else (_repo_root() / DEFAULT_M7_TRAINING_CONFIG_PATH).resolve()
    )
    training_config = load_training_config(resolved_config_path)
    candidate_names = tuple(shortlist_names or DEFAULT_SHORTLIST_NAMES)
    candidates = tuple(find_policy_candidate(candidate_name) for candidate_name in candidate_names)
    return LivePolicyChallengerConfig(
        training_config_path=str(resolved_config_path),
        evaluation_horizon_candles=int(training_config.label_horizon_candles),
        fee_rate=float(training_config.round_trip_fee_rate),
        candidates=candidates,
    )


def resolve_live_policy_challenger_artifact_dir(
    trading_artifact_dir: str | Path,
) -> Path:
    """Resolve the research-only policy challenger artifact directory."""
    return (Path(trading_artifact_dir) / DEFAULT_ARTIFACT_SUBDIR).resolve()


def resolve_live_policy_challenger_artifact_dir_from_trading_config(
    trading_config_path: Path | None = None,
) -> Path:
    """Resolve the research-only artifact directory from the checked-in paper config."""
    resolved_config_path = (
        Path(trading_config_path).resolve()
        if trading_config_path is not None
        else (_repo_root() / DEFAULT_TRADING_CONFIG_PATH).resolve()
    )
    trading_config = load_paper_trading_config(resolved_config_path)
    return resolve_live_policy_challenger_artifact_dir(trading_config.artifact_dir)


class LivePolicyChallengerTracker:
    """Append-only research scorer that never alters production execution behavior."""

    def __init__(
        self,
        *,
        trading_config: PaperTradingConfig,
        challenger_config: LivePolicyChallengerConfig,
        enabled: bool = True,
    ) -> None:
        self.trading_config = trading_config
        self.challenger_config = challenger_config
        self.enabled = enabled
        self.artifact_dir = resolve_live_policy_challenger_artifact_dir(
            trading_config.artifact_dir
        )
        self.observations_path = self.artifact_dir / "challenger_observations.jsonl"
        self.settlements_path = self.artifact_dir / "challenger_settlements.jsonl"
        self.latest_scoreboard_json_path = self.artifact_dir / "latest_scoreboard.json"
        self.latest_scoreboard_csv_path = self.artifact_dir / "latest_scoreboard.csv"
        self.summary_md_path = self.artifact_dir / "summary.md"
        self._pending_observations = self._load_pending_observations()

    def observe_signal(
        self,
        *,
        candle: FeatureCandle,
        signal: SignalDecision,
        risk_decision: RiskDecision,
        production_trade_taken: bool,
    ) -> None:
        """Score the challenger shortlist for one observed live/paper signal row."""
        if not self.enabled:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._settle_due_observations(candle)
        observation = build_live_policy_challenger_observation(
            trading_config=self.trading_config,
            challenger_config=self.challenger_config,
            candle=candle,
            signal=signal,
            risk_decision=risk_decision,
            production_trade_taken=production_trade_taken,
        )
        append_jsonl_artifact(self.observations_path, observation.to_dict())
        self._pending_observations[observation.observation_id] = observation.to_dict()

    def write_latest_scoreboard(self) -> dict[str, Any] | None:
        """Write the latest research-only challenger scoreboard from saved logs."""
        if not self.enabled:
            return None
        summary = build_live_policy_challenger_summary(
            artifact_dir=self.artifact_dir,
            challenger_config=self.challenger_config,
        )
        write_json_artifact(self.latest_scoreboard_json_path, summary)
        write_csv_artifact(
            self.latest_scoreboard_csv_path,
            [
                _flatten_candidate_summary(candidate_summary, summary["best_candidate"])
                for candidate_summary in summary["candidate_results"]
            ],
        )
        self.summary_md_path.write_text(
            _build_summary_markdown(summary),
            encoding="utf-8",
        )
        return summary

    def _load_pending_observations(self) -> dict[str, dict[str, Any]]:
        observations = {
            str(row["observation_id"]): row
            for row in _read_jsonl(self.observations_path)
        }
        settled_ids = {
            str(row["observation_id"])
            for row in _read_jsonl(self.settlements_path)
        }
        return {
            observation_id: row
            for observation_id, row in observations.items()
            if observation_id not in settled_ids
        }

    def _settle_due_observations(self, candle: FeatureCandle) -> None:
        due_observations = [
            row
            for row in self._pending_observations.values()
            if row["symbol"] == candle.symbol
            and row["target_eval_interval_begin"] == to_rfc3339(candle.interval_begin)
        ]
        for observation in sorted(
            due_observations,
            key=lambda row: (
                row["signal_as_of_time"],
                row["signal_interval_begin"],
                row["symbol"],
                row["observation_id"],
            ),
        ):
            settlement = _build_settlement_payload(
                observation=observation,
                candle=candle,
                fee_rate=self.challenger_config.fee_rate,
            )
            append_jsonl_artifact(self.settlements_path, settlement)
            self._pending_observations.pop(str(observation["observation_id"]), None)


def build_live_policy_challenger_observation(
    *,
    trading_config: PaperTradingConfig,
    challenger_config: LivePolicyChallengerConfig,
    candle: FeatureCandle,
    signal: SignalDecision,
    risk_decision: RiskDecision,
    production_trade_taken: bool,
) -> LivePolicyChallengerObservation:
    """Build one research-only challenger observation from the authoritative signal row."""
    regime_label = _normalized_regime_label(signal.regime_label)
    candidate_decisions = tuple(
        score_policy_candidate(
            candidate=candidate,
            prob_up=float(signal.prob_up),
            regime_label=regime_label,
        )
        for candidate in challenger_config.candidates
    )
    target_eval_interval_begin = candle.interval_begin + timedelta(
        minutes=trading_config.interval_minutes * challenger_config.evaluation_horizon_candles
    )
    return LivePolicyChallengerObservation(
        observation_id=f"{trading_config.service_name}|{trading_config.execution.mode}|{signal.row_id}",
        service_name=trading_config.service_name,
        execution_mode=trading_config.execution.mode,
        symbol=candle.symbol,
        signal_row_id=signal.row_id,
        signal_interval_begin=to_rfc3339(candle.interval_begin),
        signal_as_of_time=to_rfc3339(signal.as_of_time),
        regime_label=regime_label,
        prob_up=float(signal.prob_up),
        current_close_price=float(candle.close_price),
        target_eval_interval_begin=to_rfc3339(target_eval_interval_begin),
        active_production_decision="TAKE_LONG" if production_trade_taken else "FLAT",
        active_production_trade_taken=bool(production_trade_taken),
        active_production_reason_code=_production_reason_code(
            signal=signal,
            risk_decision=risk_decision,
            production_trade_taken=production_trade_taken,
        ),
        candidate_decisions=candidate_decisions,
    )


def score_policy_candidate(
    *,
    candidate: LongOnlyPolicyCandidate,
    prob_up: float,
    regime_label: str,
) -> CandidateScoreDecision:
    """Score one named long-only challenger candidate on one observed row."""
    if candidate.allowed_regimes is not None and regime_label not in candidate.allowed_regimes:
        return CandidateScoreDecision(
            candidate_name=candidate.name,
            decision="FLAT",
            would_trade=False,
            reason_code="REGIME_NOT_ALLOWED",
            effective_prob_up_threshold=None,
        )
    if regime_label in candidate.blocked_regimes:
        return CandidateScoreDecision(
            candidate_name=candidate.name,
            decision="FLAT",
            would_trade=False,
            reason_code="BLOCKED_REGIME",
            effective_prob_up_threshold=None,
        )
    effective_threshold = candidate.threshold_for_regime(regime_label)
    if effective_threshold is None:
        return CandidateScoreDecision(
            candidate_name=candidate.name,
            decision="FLAT",
            would_trade=False,
            reason_code="BLOCKED_REGIME",
            effective_prob_up_threshold=None,
        )
    if prob_up >= float(effective_threshold):
        return CandidateScoreDecision(
            candidate_name=candidate.name,
            decision="TAKE_LONG",
            would_trade=True,
            reason_code="PROB_UP_AT_OR_ABOVE_THRESHOLD",
            effective_prob_up_threshold=float(effective_threshold),
        )
    return CandidateScoreDecision(
        candidate_name=candidate.name,
        decision="FLAT",
        would_trade=False,
        reason_code="PROB_UP_BELOW_THRESHOLD",
        effective_prob_up_threshold=float(effective_threshold),
    )


def build_live_policy_challenger_summary(
    *,
    artifact_dir: Path,
    challenger_config: LivePolicyChallengerConfig | None = None,
) -> dict[str, Any]:
    """Build the latest challenger scoreboard from the append-only research logs."""
    resolved_artifact_dir = Path(artifact_dir).resolve()
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = challenger_config or build_live_policy_challenger_config()
    observations = _read_jsonl(resolved_artifact_dir / "challenger_observations.jsonl")
    settlements = _read_jsonl(resolved_artifact_dir / "challenger_settlements.jsonl")
    candidate_names = [candidate.name for candidate in resolved_config.candidates]
    production_baseline = _build_production_baseline(
        observations=observations,
        settlements=settlements,
    )
    candidate_results = [
        _build_candidate_summary(
            candidate_name=candidate_name,
            observations=observations,
            settlements=settlements,
            production_baseline=production_baseline,
        )
        for candidate_name in candidate_names
    ]
    best_candidate = _select_best_candidate(candidate_results)
    summary = {
        "artifact_dir": str(resolved_artifact_dir),
        "economics_contract": {
            **resolved_config.to_dict(),
            "name": "M7_LIVE_PAPER_RESEARCH_PROXY",
            "description": (
                "Research-only row-level long-entry proxy using the completed M7 label horizon "
                "and round-trip fee against observed paper/live signal rows."
            ),
        },
        "observed_window": _build_observed_window(observations, settlements),
        "candidate_results": candidate_results,
        "production_baseline": production_baseline,
        "best_candidate": best_candidate,
        "output_files": {
            "observations_jsonl": str((resolved_artifact_dir / "challenger_observations.jsonl").resolve()),
            "settlements_jsonl": str((resolved_artifact_dir / "challenger_settlements.jsonl").resolve()),
            "latest_scoreboard_json": str((resolved_artifact_dir / "latest_scoreboard.json").resolve()),
            "latest_scoreboard_csv": str((resolved_artifact_dir / "latest_scoreboard.csv").resolve()),
            "summary_md": str((resolved_artifact_dir / "summary.md").resolve()),
        },
    }
    return make_json_safe(summary)


def _build_production_baseline(
    *,
    observations: list[dict[str, Any]],
    settlements: list[dict[str, Any]],
) -> dict[str, Any]:
    trade_settlements = [
        settlement
        for settlement in _ordered_settlements(settlements)
        if bool(settlement["production_trade_taken"])
    ]
    cumulative_net_proxy = 0.0
    max_drawdown_proxy = 0.0
    running_peak = 0.0
    for settlement in trade_settlements:
        cumulative_net_proxy += float(settlement["production_net_proxy"])
        running_peak = max(running_peak, cumulative_net_proxy)
        max_drawdown_proxy = min(max_drawdown_proxy, cumulative_net_proxy - running_peak)
    settled_decision_count = len(settlements)
    observed_decision_count = len(observations)
    never_trades_trend_up = (
        any(observation["regime_label"] == "TREND_UP" for observation in observations)
        and not any(settlement["regime_label"] == "TREND_UP" for settlement in trade_settlements)
    )
    return {
        "policy_name": "active_production_policy",
        "observed_decision_count": observed_decision_count,
        "settled_decision_count": settled_decision_count,
        "unsettled_decision_count": max(observed_decision_count - settled_decision_count, 0),
        "hypothetical_trade_count": len(trade_settlements),
        "hypothetical_trade_rate": (
            len(trade_settlements) / settled_decision_count if settled_decision_count > 0 else 0.0
        ),
        "hypothetical_mean_net_proxy": (
            cumulative_net_proxy / settled_decision_count if settled_decision_count > 0 else 0.0
        ),
        "cumulative_net_proxy": cumulative_net_proxy,
        "max_drawdown_proxy": max_drawdown_proxy,
        "trades_in_trend_down": sum(
            int(settlement["regime_label"] == "TREND_DOWN") for settlement in trade_settlements
        ),
        "trades_in_trend_up": sum(
            int(settlement["regime_label"] == "TREND_UP") for settlement in trade_settlements
        ),
        "trades_in_range": sum(
            int(settlement["regime_label"] == "RANGE") for settlement in trade_settlements
        ),
        "trades_in_high_vol": sum(
            int(settlement["regime_label"] == "HIGH_VOL") for settlement in trade_settlements
        ),
        "never_trades_trend_up": never_trades_trend_up,
        "warnings": _build_candidate_warnings(
            trade_count=len(trade_settlements),
            cumulative_net_proxy=cumulative_net_proxy,
            max_drawdown_proxy=max_drawdown_proxy,
            never_trades_trend_up=never_trades_trend_up,
        ),
    }


def _build_candidate_summary(
    *,
    candidate_name: str,
    observations: list[dict[str, Any]],
    settlements: list[dict[str, Any]],
    production_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    trade_settlements = [
        settlement
        for settlement in _ordered_settlements(settlements)
        if bool(settlement["candidate_outcomes"][candidate_name]["would_trade"])
    ]
    cumulative_net_proxy = 0.0
    max_drawdown_proxy = 0.0
    running_peak = 0.0
    for settlement in trade_settlements:
        cumulative_net_proxy += float(settlement["candidate_outcomes"][candidate_name]["net_proxy"])
        running_peak = max(running_peak, cumulative_net_proxy)
        max_drawdown_proxy = min(max_drawdown_proxy, cumulative_net_proxy - running_peak)
    settled_decision_count = len(settlements)
    observed_decision_count = len(observations)
    trades_in_trend_up = sum(
        int(settlement["regime_label"] == "TREND_UP") for settlement in trade_settlements
    )
    never_trades_trend_up = (
        any(observation["regime_label"] == "TREND_UP" for observation in observations)
        and trades_in_trend_up == 0
    )
    warnings = _build_candidate_warnings(
        trade_count=len(trade_settlements),
        cumulative_net_proxy=cumulative_net_proxy,
        max_drawdown_proxy=max_drawdown_proxy,
        never_trades_trend_up=never_trades_trend_up,
    )
    return {
        "candidate_name": candidate_name,
        "observed_decision_count": observed_decision_count,
        "settled_decision_count": settled_decision_count,
        "unsettled_decision_count": max(observed_decision_count - settled_decision_count, 0),
        "hypothetical_trade_count": len(trade_settlements),
        "hypothetical_trade_rate": (
            len(trade_settlements) / settled_decision_count if settled_decision_count > 0 else 0.0
        ),
        "hypothetical_mean_net_proxy": (
            cumulative_net_proxy / settled_decision_count if settled_decision_count > 0 else 0.0
        ),
        "cumulative_net_proxy": cumulative_net_proxy,
        "max_drawdown_proxy": max_drawdown_proxy,
        "trades_in_trend_down": sum(
            int(settlement["regime_label"] == "TREND_DOWN") for settlement in trade_settlements
        ),
        "trades_in_trend_up": trades_in_trend_up,
        "trades_in_range": sum(
            int(settlement["regime_label"] == "RANGE") for settlement in trade_settlements
        ),
        "trades_in_high_vol": sum(
            int(settlement["regime_label"] == "HIGH_VOL") for settlement in trade_settlements
        ),
        "never_trades_trend_up": never_trades_trend_up,
        "positive_but_sparse": (
            cumulative_net_proxy > 0.0 and len(trade_settlements) < LOW_TRADE_COUNT_WARNING_THRESHOLD
        ),
        "delta_cumulative_net_vs_production": (
            cumulative_net_proxy - float(production_baseline["cumulative_net_proxy"])
        ),
        "delta_trade_count_vs_production": (
            len(trade_settlements) - int(production_baseline["hypothetical_trade_count"])
        ),
        "warnings": warnings,
    }


def _build_candidate_warnings(
    *,
    trade_count: int,
    cumulative_net_proxy: float,
    max_drawdown_proxy: float,
    never_trades_trend_up: bool,
) -> list[str]:
    warnings: list[str] = []
    if trade_count < LOW_TRADE_COUNT_WARNING_THRESHOLD:
        warnings.append("Trade count remains below 20.")
    if cumulative_net_proxy > 0.0 and abs(max_drawdown_proxy) >= cumulative_net_proxy:
        warnings.append("Positive cumulative net but drawdown is at least as large as total gain.")
    if never_trades_trend_up:
        warnings.append("Candidate never trades TREND_UP.")
    return warnings


def _flatten_candidate_summary(
    candidate_summary: Mapping[str, Any],
    best_candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "candidate_name": candidate_summary["candidate_name"],
        "observed_decision_count": int(candidate_summary["observed_decision_count"]),
        "settled_decision_count": int(candidate_summary["settled_decision_count"]),
        "unsettled_decision_count": int(candidate_summary["unsettled_decision_count"]),
        "hypothetical_trade_count": int(candidate_summary["hypothetical_trade_count"]),
        "hypothetical_trade_rate": float(candidate_summary["hypothetical_trade_rate"]),
        "hypothetical_mean_net_proxy": float(candidate_summary["hypothetical_mean_net_proxy"]),
        "cumulative_net_proxy": float(candidate_summary["cumulative_net_proxy"]),
        "max_drawdown_proxy": float(candidate_summary["max_drawdown_proxy"]),
        "trades_in_trend_down": int(candidate_summary["trades_in_trend_down"]),
        "trades_in_trend_up": int(candidate_summary["trades_in_trend_up"]),
        "trades_in_range": int(candidate_summary["trades_in_range"]),
        "trades_in_high_vol": int(candidate_summary["trades_in_high_vol"]),
        "never_trades_trend_up": bool(candidate_summary["never_trades_trend_up"]),
        "positive_but_sparse": bool(candidate_summary["positive_but_sparse"]),
        "delta_cumulative_net_vs_production": float(
            candidate_summary["delta_cumulative_net_vs_production"]
        ),
        "delta_trade_count_vs_production": int(
            candidate_summary["delta_trade_count_vs_production"]
        ),
        "warnings": " | ".join(candidate_summary["warnings"]),
        "is_best_candidate": (
            best_candidate is not None
            and candidate_summary["candidate_name"] == best_candidate["candidate_name"]
        ),
    }


def _select_best_candidate(
    candidate_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not candidate_results:
        return None
    trading_results = [
        candidate_result
        for candidate_result in candidate_results
        if int(candidate_result["hypothetical_trade_count"]) > 0
    ]
    ranked_candidates = trading_results or candidate_results
    return sorted(
        ranked_candidates,
        key=lambda candidate_result: (
            -float(candidate_result["cumulative_net_proxy"]),
            -int(candidate_result["hypothetical_trade_count"]),
            -float(candidate_result["hypothetical_mean_net_proxy"]),
            str(candidate_result["candidate_name"]),
        ),
    )[0]


def _build_observed_window(
    observations: list[dict[str, Any]],
    settlements: list[dict[str, Any]],
) -> dict[str, Any]:
    if not observations:
        return {
            "observed_decision_count": 0,
            "settled_decision_count": 0,
            "unsettled_decision_count": 0,
            "first_signal_as_of_time": None,
            "last_signal_as_of_time": None,
            "last_settled_as_of_time": None,
        }
    return {
        "observed_decision_count": len(observations),
        "settled_decision_count": len(settlements),
        "unsettled_decision_count": max(len(observations) - len(settlements), 0),
        "first_signal_as_of_time": min(row["signal_as_of_time"] for row in observations),
        "last_signal_as_of_time": max(row["signal_as_of_time"] for row in observations),
        "last_settled_as_of_time": (
            None
            if not settlements
            else max(row["evaluated_as_of_time"] for row in settlements)
        ),
    }


def _ordered_settlements(settlements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        settlements,
        key=lambda row: (
            row["evaluated_as_of_time"],
            row["signal_interval_begin"],
            row["symbol"],
            row["observation_id"],
        ),
    )


def _build_settlement_payload(
    *,
    observation: Mapping[str, Any],
    candle: FeatureCandle,
    fee_rate: float,
) -> dict[str, Any]:
    current_close_price = float(observation["current_close_price"])
    future_close_price = float(candle.close_price)
    future_return_3 = (
        0.0 if current_close_price == 0.0 else (future_close_price / current_close_price) - 1.0
    )
    production_trade_taken = bool(observation["active_production_trade_taken"])
    candidate_outcomes: dict[str, dict[str, Any]] = {}
    for candidate_name, decision in dict(observation["candidate_decisions"]).items():
        would_trade = bool(decision["would_trade"])
        gross_proxy = future_return_3 if would_trade else 0.0
        net_proxy = (future_return_3 - fee_rate) if would_trade else 0.0
        candidate_outcomes[str(candidate_name)] = {
            "decision": str(decision["decision"]),
            "would_trade": would_trade,
            "reason_code": str(decision["reason_code"]),
            "effective_prob_up_threshold": decision["effective_prob_up_threshold"],
            "gross_proxy": gross_proxy,
            "net_proxy": net_proxy,
        }
    production_gross_proxy = future_return_3 if production_trade_taken else 0.0
    production_net_proxy = (future_return_3 - fee_rate) if production_trade_taken else 0.0
    return {
        "observation_id": observation["observation_id"],
        "service_name": observation["service_name"],
        "execution_mode": observation["execution_mode"],
        "symbol": observation["symbol"],
        "signal_row_id": observation["signal_row_id"],
        "signal_interval_begin": observation["signal_interval_begin"],
        "signal_as_of_time": observation["signal_as_of_time"],
        "regime_label": observation["regime_label"],
        "prob_up": float(observation["prob_up"]),
        "evaluated_interval_begin": to_rfc3339(candle.interval_begin),
        "evaluated_as_of_time": to_rfc3339(candle.as_of_time),
        "future_close_price": future_close_price,
        "future_return_3": future_return_3,
        "production_trade_taken": production_trade_taken,
        "production_decision": observation["active_production_decision"],
        "production_reason_code": observation["active_production_reason_code"],
        "production_gross_proxy": production_gross_proxy,
        "production_net_proxy": production_net_proxy,
        "candidate_outcomes": candidate_outcomes,
    }


def _build_summary_markdown(summary: Mapping[str, Any]) -> str:
    best_candidate = summary["best_candidate"]
    production = summary["production_baseline"]
    lines = [
        "# Live Paper Policy Challengers",
        "",
        f"- Artifact directory: `{summary['artifact_dir']}`",
        f"- Observed decisions: `{int(summary['observed_window']['observed_decision_count'])}`",
        f"- Settled decisions: `{int(summary['observed_window']['settled_decision_count'])}`",
        f"- Unsettled decisions: `{int(summary['observed_window']['unsettled_decision_count'])}`",
        "",
        "## Production Baseline",
        "",
        (
            f"- Active production policy: trade_count={int(production['hypothetical_trade_count'])}, "
            f"cumulative_net={float(production['cumulative_net_proxy']):.6f}, "
            f"max_drawdown={float(production['max_drawdown_proxy']):.6f}"
        ),
        "",
        "## Best Research Candidate",
        "",
    ]
    if best_candidate is None:
        lines.append("- No challenger observations have been recorded yet.")
    else:
        lines.extend(
            [
                (
                    f"- Best candidate: `{best_candidate['candidate_name']}` "
                    f"(trade_count={int(best_candidate['hypothetical_trade_count'])}, "
                    f"cumulative_net={float(best_candidate['cumulative_net_proxy']):.6f}, "
                    f"max_drawdown={float(best_candidate['max_drawdown_proxy']):.6f})"
                ),
                (
                    f"- Routing: TREND_UP={int(best_candidate['trades_in_trend_up'])}, "
                    f"TREND_DOWN={int(best_candidate['trades_in_trend_down'])}, "
                    f"RANGE={int(best_candidate['trades_in_range'])}, "
                    f"HIGH_VOL={int(best_candidate['trades_in_high_vol'])}"
                ),
                (
                    f"- Too sparse: `{bool(best_candidate['positive_but_sparse'])}`; "
                    f"never trades TREND_UP: `{bool(best_candidate['never_trades_trend_up'])}`"
                ),
            ]
        )
        if best_candidate["warnings"]:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in best_candidate["warnings"])
    lines.extend(["", "## Output Files", ""])
    for label, path in summary["output_files"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "This challenger scoreboard is research-only. It never routes execution, places "
            "extra trades, or changes the active production policy.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            raw_line = line.strip()
            if not raw_line:
                continue
            rows.append(json.loads(raw_line))
    return rows


def _normalized_regime_label(regime_label: str | None) -> str:
    if regime_label is None or not regime_label.strip():
        return "UNKNOWN"
    return regime_label.strip()


def _production_reason_code(
    *,
    signal: SignalDecision,
    risk_decision: RiskDecision,
    production_trade_taken: bool,
) -> str:
    if production_trade_taken:
        return "PRODUCTION_BUY_APPROVED"
    if signal.signal != "BUY":
        return f"MODEL_{signal.signal}_NO_LONG"
    if risk_decision.primary_reason_code:
        return str(risk_decision.primary_reason_code)
    if risk_decision.reason_codes:
        return str(risk_decision.reason_codes[0])
    return f"RISK_{risk_decision.outcome}_NO_LONG"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    """Build and print the latest live paper challenger scoreboard."""
    parser = argparse.ArgumentParser(
        description="Build the Stream Alpha live paper challenger scoreboard",
    )
    parser.add_argument(
        "--artifact-dir",
        help=(
            "Optional research-only challenger artifact directory. Defaults to "
            "the paper trading artifact_dir plus research/policy_challengers."
        ),
    )
    parser.add_argument(
        "--trading-config",
        help="Optional paper trading YAML config path. Defaults to configs/paper_trading.paper.yaml.",
    )
    parser.add_argument(
        "--training-config",
        help="Optional M7 training config path. Defaults to configs/training.m7.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the scoreboard summary as JSON.",
    )
    arguments = parser.parse_args()
    try:
        challenger_config = build_live_policy_challenger_config(
            training_config_path=Path(arguments.training_config) if arguments.training_config else None
        )
        artifact_dir = (
            Path(arguments.artifact_dir).resolve()
            if arguments.artifact_dir
            else resolve_live_policy_challenger_artifact_dir_from_trading_config(
                Path(arguments.trading_config) if arguments.trading_config else None
            )
        )
        summary = build_live_policy_challenger_summary(
            artifact_dir=artifact_dir,
            challenger_config=challenger_config,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(summary), sort_keys=True))
        return

    best_candidate = summary["best_candidate"]
    print(f"artifact_dir={summary['artifact_dir']}")
    print(
        "production_baseline="
        f"trade_count={int(summary['production_baseline']['hypothetical_trade_count'])},"
        f" cumulative_net={float(summary['production_baseline']['cumulative_net_proxy']):.6f}"
    )
    if best_candidate is None:
        print("best_candidate=none")
        return
    print(
        "best_candidate="
        f"{best_candidate['candidate_name']}(cumulative_net={float(best_candidate['cumulative_net_proxy']):.6f},"
        f" trade_count={int(best_candidate['hypothetical_trade_count'])},"
        f" max_drawdown={float(best_candidate['max_drawdown_proxy']):.6f})"
    )
    print(f"positive_but_sparse={bool(best_candidate['positive_but_sparse'])}")
    print(f"never_trades_trend_up={bool(best_candidate['never_trades_trend_up'])}")
    if best_candidate["warnings"]:
        print(f"warnings={' | '.join(best_candidate['warnings'])}")


if __name__ == "__main__":
    main()
