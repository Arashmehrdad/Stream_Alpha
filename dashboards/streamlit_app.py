"""Read-only Stream Alpha M15 operator console."""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from app.common.config import Settings
from app.runtime.config import resolve_runtime_profile, resolve_trading_config_path
from app.trading.config import PaperTradingConfig, load_paper_trading_config
from dashboards.data_sources import DashboardDataSources, DecisionTraceSnapshot
from dashboards.view_models import (
    age_text,
    build_config_summary_rows,
    build_feature_lag_rows,
    build_feature_snapshot_rows,
    build_latest_blocked_trade_rows,
    build_latest_recovery_rows,
    build_latest_signal_rows,
    build_live_critical_state_strip,
    build_live_status_rows,
    build_model_reference_rows,
    build_open_position_rows,
    build_operator_banner,
    build_operator_incident_rows,
    build_overview_metrics,
    build_performance_by_regime_rows,
    build_recent_closed_trade_rows,
    build_recent_decision_trace_rows,
    build_recent_ledger_rows,
    build_recent_order_audit_rows,
    build_reliability_status_rows,
    build_service_health_rows,
    build_symbol_freshness_rows,
    build_trade_journal_rows,
    build_trader_freshness,
    build_venue_environment_rows,
    filter_trade_journal_traces,
    latest_feature_as_of,
)
from dashboards.widgets import (
    render_health_card,
    render_incidents_panel,
    render_live_critical_state_strip,
    render_operator_banner,
    render_summary_cards,
    render_table,
)

# pylint: disable=too-many-locals,too-many-arguments,too-many-statements,too-many-lines

_SNAPSHOT_SESSION_KEY = "streamalpha_dashboard_snapshot"


def main() -> None:
    """Run the read-only Stream Alpha M15 operator console."""
    settings = Settings.from_env()
    trading_config = load_paper_trading_config(resolve_dashboard_trading_config_path())

    st.set_page_config(
        page_title="Stream Alpha Operator Console",
        page_icon="SA",
        layout="wide",
    )

    st.title("Stream Alpha")
    st.caption("Operator console with accepted M15-M21 foundations")
    _render_refreshing_console(
        settings=settings,
        trading_config=trading_config,
    )


def _render_refreshing_console(
    *,
    settings: Settings,
    trading_config: PaperTradingConfig,
) -> None:
    snapshot = st.session_state.get(_SNAPSHOT_SESSION_KEY)
    if snapshot is None:
        snapshot = _refresh_dashboard_snapshot(
            settings=settings,
            trading_config=trading_config,
        )
    runtime_profile = resolve_display_runtime_profile(snapshot=snapshot)
    render_runtime_profile_badge(
        runtime_profile=runtime_profile,
        execution_mode=trading_config.execution.mode,
    )
    with st.sidebar:
        _render_sidebar(
            settings=settings,
            trading_config=trading_config,
            runtime_profile=runtime_profile,
        )

    refresh_seconds = max(int(settings.dashboard.refresh_seconds), 0)
    fragment_api = getattr(st, "fragment", None)

    if not callable(fragment_api):
        _render_dynamic_console(
            settings=settings,
            trading_config=trading_config,
        )
        return

    if refresh_seconds > 0:

        @fragment_api(run_every=refresh_seconds)
        def _refresh_fragment() -> None:
            _render_dynamic_console(
                settings=settings,
                trading_config=trading_config,
            )

    else:

        @fragment_api
        def _refresh_fragment() -> None:
            _render_dynamic_console(
                settings=settings,
                trading_config=trading_config,
            )

    _refresh_fragment()


def _load_dashboard_snapshot(
    *,
    settings: Settings,
    trading_config: PaperTradingConfig,
):
    return asyncio.run(
        DashboardDataSources(
            settings=settings,
            trading_config=trading_config,
        ).load_snapshot()
    )


def _refresh_dashboard_snapshot(
    *,
    settings: Settings,
    trading_config: PaperTradingConfig,
):
    snapshot = _load_dashboard_snapshot(
        settings=settings,
        trading_config=trading_config,
    )
    st.session_state[_SNAPSHOT_SESSION_KEY] = snapshot
    return snapshot


def _render_dynamic_console(
    *,
    settings: Settings,
    trading_config: PaperTradingConfig,
) -> None:
    snapshot = _refresh_dashboard_snapshot(
        settings=settings,
        trading_config=trading_config,
    )

    reference_time = max(snapshot.api_health.checked_at, snapshot.database.checked_at)
    incidents = build_operator_incident_rows(
        snapshot=snapshot,
        trading_config=trading_config,
        now=reference_time,
    )
    banner = build_operator_banner(
        snapshot=snapshot,
        trading_config=trading_config,
        incidents=incidents,
        now=reference_time,
    )
    venue_rows = build_venue_environment_rows(
        snapshot=snapshot,
        trading_config=trading_config,
    )
    config_summary_rows = build_config_summary_rows(
        settings=settings,
        trading_config=trading_config,
        snapshot=snapshot,
    )
    latest_signals = build_latest_signal_rows(
        symbols=trading_config.symbols,
        signals=snapshot.signals,
        now=reference_time,
    )
    latest_features = build_feature_snapshot_rows(
        symbols=trading_config.symbols,
        features=snapshot.database.latest_features,
        now=reference_time,
    )
    reliability_rows = build_reliability_status_rows(
        api_health=snapshot.api_health,
        system_reliability=snapshot.system_reliability,
        reliability_states=snapshot.database.reliability_states,
        latest_recovery_event=snapshot.database.latest_recovery_event,
        now=reference_time,
    )
    service_health_rows = build_service_health_rows(
        snapshot.system_reliability,
        now=reference_time,
    )
    lag_rows = build_feature_lag_rows(
        snapshot.system_reliability,
        now=reference_time,
    )
    freshness_rows = build_symbol_freshness_rows(
        snapshot.freshness,
        now=reference_time,
    )
    live_status_rows = build_live_status_rows(
        trading_config=trading_config,
        live_safety_state=snapshot.database.live_safety_state,
        now=reference_time,
    )
    live_critical_strip = build_live_critical_state_strip(
        snapshot=snapshot,
        trading_config=trading_config,
        now=reference_time,
    )
    recent_decision_trace_rows = build_recent_decision_trace_rows(
        snapshot.database.recent_decision_traces,
        now=reference_time,
    )
    latest_blocked_trade_rows = build_latest_blocked_trade_rows(
        snapshot.database.latest_blocked_trade,
        now=reference_time,
    )
    recent_order_audit_rows = build_recent_order_audit_rows(
        snapshot.database.recent_order_events,
        now=reference_time,
    )
    recent_ledger_rows = build_recent_ledger_rows(
        snapshot.database.recent_ledger_entries,
        now=reference_time,
    )
    recent_closed_rows = build_recent_closed_trade_rows(
        snapshot.database.recent_closed_positions,
        now=reference_time,
    )
    open_position_rows = build_open_position_rows(
        positions=snapshot.database.positions,
        latest_prices=snapshot.database.latest_prices,
        fee_bps=trading_config.risk.fee_bps,
    )
    performance_by_regime_rows = build_performance_by_regime_rows(
        snapshot=snapshot,
        trading_config=trading_config,
    )
    model_reference_rows = build_model_reference_rows(
        snapshot=snapshot,
        now=reference_time,
    )
    latest_recovery_rows = build_latest_recovery_rows(
        snapshot.database.latest_recovery_event,
        now=reference_time,
    )
    active_alert_rows = _build_active_alert_rows(
        snapshot=snapshot,
        now=reference_time,
    )
    incident_timeline_rows = _build_incident_timeline_rows(
        snapshot=snapshot,
        now=reference_time,
    )
    startup_safety_rows = _build_startup_safety_rows(
        snapshot=snapshot,
        now=reference_time,
    )
    daily_summary_rows = _build_daily_operations_summary_rows(snapshot=snapshot)
    overview_metrics = build_overview_metrics(
        snapshot=snapshot,
        trading_config=trading_config,
    )
    trader_freshness = build_trader_freshness(snapshot.database.engine_states)

    render_operator_banner(banner)
    render_live_critical_state_strip(live_critical_strip)
    render_summary_cards(
        title="Venue and Environment",
        items=_venue_summary_cards(venue_rows[0]),
    )

    tab_labels = [
        "Market",
        "Signals",
        "Trades",
        "Risk",
        "Health",
        "Models",
        "Incidents",
    ]
    (
        market_view,
        signals_view,
        trades_view,
        risk_view,
        health_view,
        models_view,
        incidents_view,
    ) = st.tabs(tab_labels)

    with market_view:
        _render_market_view(
            trading_config=trading_config,
            snapshot=snapshot,
            latest_features=latest_features,
            freshness_rows=freshness_rows,
            lag_rows=lag_rows,
            reference_time=reference_time,
        )

    with signals_view:
        _render_signals_view(
            latest_signals=latest_signals,
            recent_decision_trace_rows=recent_decision_trace_rows,
            snapshot=snapshot,
        )

    with trades_view:
        _render_trades_view(
            snapshot=snapshot,
            trading_config=trading_config,
            overview_metrics=overview_metrics,
            open_position_rows=open_position_rows,
            recent_closed_rows=recent_closed_rows,
            recent_ledger_rows=recent_ledger_rows,
            recent_order_audit_rows=recent_order_audit_rows,
            reference_time=reference_time,
        )

    with risk_view:
        _render_risk_view(
            snapshot=snapshot,
            performance_by_regime_rows=performance_by_regime_rows,
            latest_blocked_trade_rows=latest_blocked_trade_rows,
            reference_time=reference_time,
        )

    with health_view:
        _render_health_view(
            snapshot=snapshot,
            trading_config=trading_config,
            reliability_rows=reliability_rows,
            service_health_rows=service_health_rows,
            lag_rows=lag_rows,
            freshness_rows=freshness_rows,
            live_status_rows=live_status_rows,
            latest_recovery_rows=latest_recovery_rows,
            active_alert_rows=active_alert_rows,
            startup_safety_rows=startup_safety_rows,
            daily_summary_rows=daily_summary_rows,
            trader_freshness=trader_freshness,
        )

    with models_view:
        _render_models_view(
            snapshot=snapshot,
            model_reference_rows=model_reference_rows,
            config_summary_rows=config_summary_rows,
            recent_decision_trace_rows=recent_decision_trace_rows,
        )

    with incidents_view:
        _render_incidents_view(
            snapshot=snapshot,
            incidents=incidents,
            latest_recovery_rows=latest_recovery_rows,
            latest_blocked_trade_rows=latest_blocked_trade_rows,
            incident_timeline_rows=incident_timeline_rows,
        )


def resolve_dashboard_trading_config_path() -> Path:
    """Resolve the dashboard trading config path from runtime env."""
    return resolve_trading_config_path()


def resolve_display_runtime_profile(*, snapshot) -> str:
    """Resolve the runtime profile from API truth or env fallback."""
    snapshot_profile = getattr(snapshot.api_health, "runtime_profile", None)
    if snapshot_profile is not None and str(snapshot_profile).strip():
        try:
            return _normalize_runtime_profile(snapshot_profile)
        except ValueError:
            pass
    return _normalize_runtime_profile(resolve_runtime_profile())


def render_runtime_profile_badge(
    *,
    runtime_profile: str,
    execution_mode: str,
) -> None:
    """Render the compact runtime profile badge under the title."""
    profile = _normalize_runtime_profile(runtime_profile)
    st.markdown(
        (
            "<div style=\"display:flex; flex-wrap:wrap; gap:0.5rem; align-items:center; "
            "margin:0.1rem 0 0.9rem 0;\">"
            "<span style=\"font-size:0.95rem; font-weight:600;\">Runtime Profile:</span>"
            f"<span style=\"{_runtime_profile_badge_style(profile)}\">{profile}</span>"
            "<span style=\"font-size:0.95rem; font-weight:600; margin-left:0.75rem;\">"
            "Execution Mode:</span>"
            f"<span style=\"font-family:monospace; font-size:0.95rem;\">"
            f"{execution_mode}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_sidebar(
    *,
    settings: Settings,
    trading_config: PaperTradingConfig,
    runtime_profile: str,
) -> None:
    st.header("Runtime")
    st.write(f"Inference API: `{settings.dashboard.inference_api_base_url}`")
    st.write(f"Feature source table: `{settings.tables.feature_ohlc}`")
    st.write(f"Paper trader service: `{trading_config.service_name}`")
    st.write(f"Runtime profile: `{runtime_profile.lower()}`")
    st.write(f"Execution mode: `{trading_config.execution.mode}`")
    st.write(f"Refresh target: `{settings.dashboard.refresh_seconds}s`")
    if trading_config.execution.mode == "live":
        st.write("Broker: `alpaca`")
        st.write(
            "Expected environment: "
            f"`{trading_config.execution.live.expected_environment}`"
        )
    if st.button("Refresh now", width="stretch"):
        st.rerun()


def _normalize_runtime_profile(profile: str) -> str:
    """Normalize the operator-facing runtime profile label."""
    return resolve_runtime_profile(profile, default=None).upper()


def _runtime_profile_badge_style(profile: str) -> str:
    """Return the inline style for the runtime profile pill."""
    palette = {
        "DEV": ("#f3f4f6", "#374151", "#d1d5db"),
        "PAPER": ("#eff6ff", "#1d4ed8", "#bfdbfe"),
        "SHADOW": ("#fff7ed", "#c2410c", "#fed7aa"),
        "LIVE": ("#fef2f2", "#b91c1c", "#fecaca"),
    }
    background, foreground, border = palette[profile]
    return (
        "display:inline-block; padding:0.2rem 0.55rem; border-radius:999px; "
        f"background:{background}; color:{foreground}; border:1px solid {border}; "
        "font-size:0.82rem; font-weight:700; letter-spacing:0.02em;"
    )


def _render_market_view(
    *,
    trading_config: PaperTradingConfig,
    snapshot,
    latest_features,
    freshness_rows,
    lag_rows,
    reference_time: datetime,
) -> None:
    latest_feature_time = latest_feature_as_of(snapshot.database.latest_features)
    fresh_symbol_count = sum(
        int(row["freshness_status"] == "FRESH") for row in freshness_rows
    )
    lag_breach_count = sum(int(row["lag_breach"]) for row in lag_rows)
    render_summary_cards(
        title="Market Summary",
        items=[
            {"label": "Configured symbols", "value": str(len(trading_config.symbols))},
            {
                "label": "Latest feature time",
                "value": (
                    "-"
                    if latest_feature_time is None
                    else latest_feature_time.isoformat()
                ),
                "detail": age_text(latest_feature_time, reference_time) or "unavailable",
            },
            {
                "label": "Fresh symbols",
                "value": (
                    f"{fresh_symbol_count}/"
                    f"{len(freshness_rows) or len(trading_config.symbols)}"
                ),
                "detail": "Per-symbol exact-row freshness",
            },
            {
                "label": "Lag breaches",
                "value": str(lag_breach_count),
                "detail": "Feature consumer lag threshold breaches",
            },
        ],
    )
    render_table("Per-Symbol Freshness", freshness_rows)
    render_table("Latest Canonical Feature Snapshot", latest_features)
    render_table("Feature Consumer Lag", lag_rows)
    st.caption(
        "Market view is sourced from canonical feature rows, exact-row "
        "freshness, and M13 lag state."
    )


def _render_signals_view(
    *,
    latest_signals,
    recent_decision_trace_rows,
    snapshot,
) -> None:
    buy_count = sum(int(row["signal"] == "BUY") for row in latest_signals)
    hold_count = sum(int(row["signal"] == "HOLD") for row in latest_signals)
    unavailable_count = sum(int(row["signal"] == "UNAVAILABLE") for row in latest_signals)
    render_summary_cards(
        title="Signal Summary",
        items=[
            {"label": "BUY signals", "value": str(buy_count)},
            {"label": "HOLD signals", "value": str(hold_count)},
            {"label": "Unavailable", "value": str(unavailable_count)},
            {
                "label": "Regime runtime",
                "value": snapshot.api_health.regime_run_id or "-",
                "detail": snapshot.api_health.regime_artifact_path or "not loaded",
            },
        ],
    )
    _render_compact_bar_chart(
        title="Signal Distribution",
        rows=_build_signal_distribution_chart_rows(latest_signals),
        category_field="signal",
        value_field="count",
    )
    render_table("Latest Signals By Asset", latest_signals)
    render_table(
        "Continual Learning Signal Context",
        _build_continual_learning_signal_rows(snapshot=snapshot),
    )
    render_table("Recent Decision Traces", recent_decision_trace_rows)


def _render_trades_view(
    *,
    snapshot,
    trading_config: PaperTradingConfig,
    overview_metrics,
    open_position_rows,
    recent_closed_rows,
    recent_ledger_rows,
    recent_order_audit_rows,
    reference_time: datetime,
) -> None:
    render_summary_cards(
        title="Trade Summary",
        items=[
            {
                "label": "Open positions",
                "value": str(len(open_position_rows)),
            },
            {
                "label": "Closed trades loaded",
                "value": str(len(recent_closed_rows)),
            },
            {
                "label": "Recent order events",
                "value": str(len(recent_order_audit_rows)),
            },
            {
                "label": "Total PnL",
                "value": "-"
                if overview_metrics is None
                else f"{overview_metrics.total_pnl:,.2f}",
                "detail": "Unavailable if PostgreSQL trading state could not be read",
            },
        ],
    )
    _render_compact_bar_chart(
        title="Open Position Exposure",
        rows=_build_open_position_exposure_chart_rows(open_position_rows),
        category_field="symbol",
        value_field="entry_notional",
    )

    filtered_traces = _filter_trade_journal(
        traces=snapshot.database.recent_decision_traces,
        reference_time=reference_time,
    )
    trade_journal_rows = build_trade_journal_rows(
        filtered_traces,
        now=reference_time,
    )
    render_table("Trade Journal", trade_journal_rows)
    render_table("Open Positions", open_position_rows)
    render_table("Recent Closed Trades", recent_closed_rows)
    render_table("Recent Ledger Activity", recent_ledger_rows)
    render_table(
        "Recent Live Order Audit"
        if trading_config.execution.mode == "live"
        else "Recent Order Audit",
        recent_order_audit_rows,
    )
    _render_rationale_report_downloads(snapshot.database.recent_decision_traces)


def _render_risk_view(
    *,
    snapshot,
    performance_by_regime_rows,
    latest_blocked_trade_rows,
    reference_time: datetime,
) -> None:
    traces = snapshot.database.recent_decision_traces
    blocked_count = sum(int(trace.risk_outcome == "BLOCKED") for trace in traces)
    modified_count = sum(int(trace.risk_outcome == "MODIFIED") for trace in traces)
    latest_blocked_reason = (
        "-"
        if snapshot.database.latest_blocked_trade is None
        else snapshot.database.latest_blocked_trade.primary_reason_code or "-"
    )
    risk_rows = build_trade_journal_rows(
        traces,
        now=reference_time,
    )
    render_summary_cards(
        title="Risk Summary",
        items=[
            {"label": "Blocked decisions", "value": str(blocked_count)},
            {"label": "Modified decisions", "value": str(modified_count)},
            {"label": "Latest blocked reason", "value": latest_blocked_reason},
            {
                "label": "Latest risk trace",
                "value": "-"
                if not traces
                else traces[0].signal_as_of_time.isoformat(),
            },
        ],
    )
    _render_compact_bar_chart(
        title="Per-Regime Total PnL",
        rows=_build_regime_performance_chart_rows(performance_by_regime_rows),
        category_field="regime_label",
        value_field="total_pnl",
    )
    render_table("Latest Blocked Trade Rationale", latest_blocked_trade_rows)
    render_table("Recent Risk Decisions", risk_rows)
    render_table("Performance By Regime", performance_by_regime_rows)


def _render_health_view(
    *,
    snapshot,
    trading_config: PaperTradingConfig,
    reliability_rows,
    service_health_rows,
    lag_rows,
    freshness_rows,
    live_status_rows,
    latest_recovery_rows,
    active_alert_rows,
    startup_safety_rows,
    daily_summary_rows,
    trader_freshness,
) -> None:
    system_health = (
        "UNAVAILABLE"
        if snapshot.system_reliability is None
        else snapshot.system_reliability.health_overall_status or "UNKNOWN"
    )
    live_gate = live_status_rows[0].get("health_gate_status")
    render_summary_cards(
        title="Health Summary",
        items=[
            {"label": "System health", "value": system_health},
            {
                "label": "Lag breach active",
                "value": str(
                    False
                    if snapshot.system_reliability is None
                    else bool(snapshot.system_reliability.lag_breach_active)
                ),
            },
            {
                "label": "Trader freshness",
                "value": trader_freshness.state,
                "detail": trader_freshness.message,
            },
            {
                "label": "Live submit gate",
                "value": "-" if trading_config.execution.mode != "live" else str(live_gate),
            },
            {
                "label": "Active alerts",
                "value": str(
                    0 if snapshot.api_health.active_alert_count is None
                    else snapshot.api_health.active_alert_count
                ),
                "detail": snapshot.api_health.max_alert_severity or "none",
            },
        ],
    )

    health_columns = st.columns(4)
    with health_columns[0]:
        render_health_card(
            title="Inference API",
            state=snapshot.api_health.status,
            detail=(
                snapshot.api_health.error
                if snapshot.api_health.error is not None
                else f"Checked {age_text(snapshot.api_health.checked_at)} ago"
            ),
            healthy=snapshot.api_health.available,
        )
    with health_columns[1]:
        render_health_card(
            title="PostgreSQL",
            state="healthy" if snapshot.database.available else "unavailable",
            detail=(
                snapshot.database.error
                if snapshot.database.error is not None
                else f"Checked {age_text(snapshot.database.checked_at)} ago"
            ),
            healthy=snapshot.database.available,
        )
    with health_columns[2]:
        render_health_card(
            title="Model",
            state=snapshot.api_health.model_name or "not loaded",
            detail=snapshot.api_health.model_artifact_path or "Model artifact unavailable",
            healthy=snapshot.api_health.model_loaded,
        )
    with health_columns[3]:
        render_health_card(
            title="Trader",
            state=trader_freshness.state,
            detail=trader_freshness.message,
            healthy=trader_freshness.state == "healthy",
        )

    render_table("Reliability Summary", reliability_rows)
    _render_compact_bar_chart(
        title="Active Alert Severity",
        rows=_build_alert_severity_chart_rows(active_alert_rows),
        category_field="severity",
        value_field="count",
    )
    render_table("Active Alerts", active_alert_rows)
    render_table("Startup Safety", startup_safety_rows)
    render_table("Daily Operations Summary", daily_summary_rows)
    render_table("Live Safety State", live_status_rows)
    render_table("Per-Service Health", service_health_rows)
    render_table("Per-Symbol Freshness", freshness_rows)
    render_table("Feature Consumer Lag", lag_rows)
    render_table("Latest Recovery Event", latest_recovery_rows)


def _render_models_view(
    *,
    snapshot,
    model_reference_rows,
    config_summary_rows,
    recent_decision_trace_rows,
) -> None:
    latest_trace_count = len(snapshot.database.recent_decision_traces)
    latest_model_version = (
        "-"
        if not snapshot.database.recent_decision_traces
        else snapshot.database.recent_decision_traces[0].model_version
    )
    render_summary_cards(
        title="Model Summary",
        items=[
            {
                "label": "Model name",
                "value": snapshot.api_health.model_name or "-",
            },
            {
                "label": "Latest model version",
                "value": latest_model_version,
            },
            {
                "label": "Regime run",
                "value": snapshot.api_health.regime_run_id or "-",
            },
            {
                "label": "Decision traces loaded",
                "value": str(latest_trace_count),
            },
        ],
    )
    render_table("Model and Regime References", model_reference_rows)
    render_table("Operator Config Summary", config_summary_rows)
    render_table("Recent Decision Traces", recent_decision_trace_rows)
    _render_adaptation_section(snapshot=snapshot)
    _render_continual_learning_section(snapshot=snapshot)


def _render_adaptation_section(*, snapshot) -> None:
    adaptation = snapshot.adaptation
    summary = adaptation.summary
    render_summary_cards(
        title="Adaptation Summary",
        items=[
            {"label": "Execution mode", "value": snapshot.api_health.runtime_profile or "-"},
            {"label": "Active profile", "value": summary.active_profile_id or "-"},
            {"label": "Adaptation status", "value": summary.adaptation_status or "-"},
            {
                "label": "Freeze gate",
                "value": str(summary.frozen_by_health_gate),
                "detail": ", ".join(summary.reason_codes) if summary.reason_codes else None,
            },
        ],
    )
    if not summary.available:
        render_table(
            "Adaptation Status",
            [
                {
                    "status": "UNAVAILABLE",
                    "detail": summary.error or "adaptation endpoints unavailable",
                }
            ],
        )
        return
    render_table(
        "Adaptation Status",
        [
            {
                "current_execution_mode": snapshot.api_health.runtime_profile,
                "active_profile": summary.active_profile_id,
                "drift_status": summary.latest_drift_status,
                "latest_promotion": summary.latest_promotion_decision,
                "freeze_gate": summary.frozen_by_health_gate,
                "reason_codes": ", ".join(summary.reason_codes),
            }
        ],
    )
    render_table(
        "Adaptation Drift",
        [
            {
                "symbol": item.symbol,
                "regime_label": item.regime_label,
                "detector_name": item.detector_name,
                "window_id": item.window_id,
                "drift_score": item.drift_score,
                "status": item.status,
                "reason_code": item.reason_code,
            }
            for item in adaptation.drift
        ]
        or [{"status": "NONE"}],
    )
    render_table(
        "Adaptation Performance",
        [
            {
                "execution_mode": item.execution_mode,
                "symbol": item.symbol,
                "regime_label": item.regime_label,
                "window_id": item.window_id,
                "trade_count": item.trade_count,
                "net_pnl_after_costs": item.net_pnl_after_costs,
                "max_drawdown": item.max_drawdown,
                "profit_factor": item.profit_factor,
                "win_rate": item.win_rate,
                "blocked_trade_rate": item.blocked_trade_rate,
                "shadow_divergence_rate": item.shadow_divergence_rate,
            }
            for item in adaptation.performance
        ]
        or [{"status": "NONE"}],
    )
    render_table(
        "Adaptation Profiles",
        [
            {
                "profile_id": item.profile_id,
                "status": item.status,
                "execution_mode_scope": item.execution_mode_scope,
                "symbol_scope": item.symbol_scope,
                "regime_scope": item.regime_scope,
                "rollback_target_profile_id": item.rollback_target_profile_id,
                "activated_at": (
                    None if item.activated_at is None else item.activated_at.isoformat()
                ),
            }
            for item in adaptation.profiles
        ]
        or [{"status": "NONE"}],
    )
    render_table(
        "Adaptation Promotions",
        [
            {
                "decision_id": item.decision_id,
                "target_type": item.target_type,
                "target_id": item.target_id,
                "decision": item.decision,
                "summary_text": item.summary_text,
                "decided_at": item.decided_at.isoformat(),
                "reason_codes": ", ".join(item.reason_codes),
            }
            for item in adaptation.promotions
        ]
        or [{"status": "NONE"}],
    )


def _render_incidents_view(
    *,
    snapshot,
    incidents,
    latest_recovery_rows,
    latest_blocked_trade_rows,
    incident_timeline_rows,
) -> None:
    render_incidents_panel(incidents)
    _render_compact_bar_chart(
        title="Continual Learning Drift Cap Status",
        rows=_build_drift_cap_status_chart_rows(snapshot=snapshot),
        category_field="status",
        value_field="count",
    )
    render_table("Incident Timeline", incident_timeline_rows)
    render_table(
        "Continual Learning Events",
        _build_continual_learning_event_rows(snapshot=snapshot),
    )
    render_table(
        "Continual Learning Drift Alerts",
        _build_continual_learning_breached_drift_rows(snapshot=snapshot),
    )
    render_table(
        "Continual Learning Freeze Signals",
        _build_continual_learning_freeze_rows(snapshot=snapshot),
    )
    render_table("Latest Recovery Event", latest_recovery_rows)
    render_table("Latest Blocked Trade Rationale", latest_blocked_trade_rows)


def _render_continual_learning_section(*, snapshot) -> None:
    continual_learning = snapshot.continual_learning
    summary = continual_learning.summary
    active_profiles = [item for item in continual_learning.profiles if item.status == "ACTIVE"]
    rollback_target = None
    if len(active_profiles) == 1:
        rollback_target = active_profiles[0].rollback_target_profile_id
    render_summary_cards(
        title="Continual Learning Workflow",
        items=[
            {
                "label": "Status",
                "value": summary.continual_learning_status or "-",
                "detail": (
                    "Aggregated ALL-scope operator view"
                    if summary.aggregated_scope
                    else "Scope-specific operator view"
                ),
            },
            {"label": "Active profile", "value": summary.active_profile_id or "-"},
            {
                "label": "Latest promotion",
                "value": summary.latest_promotion_decision or "-",
            },
            {
                "label": "Latest event",
                "value": summary.latest_event_type or "-",
                "detail": "Manual and guarded workflow only",
            },
            {
                "label": "Drift-cap status",
                "value": summary.latest_drift_cap_status or "-",
            },
            {
                "label": "Rollback target",
                "value": rollback_target or "-",
                "detail": "Shown only when exactly one active profile is visible",
            },
        ],
    )
    render_table(
        "Continual Learning Workflow Status",
        _build_continual_learning_workflow_rows(snapshot=snapshot),
    )
    render_table(
        "Promotion Guardrail Status",
        _build_continual_learning_guardrail_rows(snapshot=snapshot),
    )
    render_table(
        "Continual Learning Profiles",
        [
            {
                "profile_id": item.profile_id,
                "status": item.status,
                "candidate_type": item.candidate_type,
                "execution_mode_scope": item.execution_mode_scope,
                "symbol_scope": item.symbol_scope,
                "regime_scope": item.regime_scope,
                "promotion_stage": item.promotion_stage,
                "baseline_target_id": item.baseline_target_id,
                "rollback_target_profile_id": item.rollback_target_profile_id,
            }
            for item in continual_learning.profiles
        ]
        or [{"status": "NONE"}],
    )
    render_table(
        "Continual Learning Promotions",
        [
            {
                "decision_id": item.decision_id,
                "target_id": item.target_id,
                "decision": item.decision,
                "summary_text": item.summary_text,
                "decided_at": item.decided_at.isoformat(),
                "reason_codes": ", ".join(item.reason_codes),
            }
            for item in continual_learning.promotions
        ]
        or [{"status": "NONE"}],
    )
    _render_continual_learning_operator_guidance(snapshot=snapshot)


def _build_continual_learning_workflow_rows(*, snapshot) -> list[dict[str, object]]:
    summary = snapshot.continual_learning.summary
    active_profiles = [
        item for item in snapshot.continual_learning.profiles if item.status == "ACTIVE"
    ]
    rollback_target = (
        None if len(active_profiles) != 1 else active_profiles[0].rollback_target_profile_id
    )
    if not summary.available:
        return [{"status": "UNAVAILABLE", "detail": summary.error or "unavailable"}]
    return [
        {
            "aggregated_status": summary.continual_learning_status,
            "active_profile_id": summary.active_profile_id,
            "rollback_target_profile_id": rollback_target,
            "latest_promotion_decision": summary.latest_promotion_decision,
            "latest_event_type": summary.latest_event_type,
            "drift_cap_status": summary.latest_drift_cap_status,
            "operator_note": "MANUAL_AND_GUARDED",
            "reason_codes": ", ".join(summary.reason_codes),
        }
    ]


def _build_continual_learning_guardrail_rows(*, snapshot) -> list[dict[str, object]]:
    drift_status_by_scope = {
        (
            item.execution_mode_scope,
            item.symbol_scope,
            item.regime_scope,
        ): item.status
        for item in snapshot.continual_learning.drift_caps
    }
    rows = []
    for item in snapshot.continual_learning.profiles:
        drift_status = drift_status_by_scope.get(
            (item.execution_mode_scope, item.symbol_scope, item.regime_scope)
        )
        operator_note = "LIVE_ELIGIBLE_ALLOWED"
        if drift_status == "BREACHED":
            operator_note = "BLOCKED_BY_BREACHED_DRIFT"
        elif item.candidate_type == "INCREMENTAL_SHADOW_CHALLENGER":
            operator_note = "SHADOW_ONLY_ONLY"
        rows.append(
            {
                "profile_id": item.profile_id,
                "candidate_type": item.candidate_type,
                "promotion_stage": item.promotion_stage,
                "live_eligible": item.live_eligible,
                "latest_drift_cap_status": drift_status,
                "rollback_target_profile_id": item.rollback_target_profile_id,
                "operator_note": operator_note,
            }
        )
    return rows or [{"status": "NONE"}]


def _render_continual_learning_operator_guidance(*, snapshot) -> None:
    active_profile_id = (
        snapshot.continual_learning.summary.active_profile_id or "cl-profile-id"
    )
    rollback_target = next(
        (
            item.rollback_target_profile_id
            for item in snapshot.continual_learning.profiles
            if item.status == "ACTIVE" and item.rollback_target_profile_id is not None
        ),
        None,
    )
    symbol = next(
        (
            signal.symbol
            for signal in snapshot.signals
            if signal.continual_learning_profile_id == active_profile_id
        ),
        "BTC/USD",
    )
    regime_label = next(
        (
            item.regime_scope
            for item in snapshot.continual_learning.profiles
            if item.profile_id == active_profile_id
        ),
        "ALL",
    )
    execution_mode = snapshot.api_health.runtime_profile or "paper"
    promote_payload = (
        "@'\n"
        "{\n"
        f"  \"decision_id\": \"promote:{active_profile_id}:20260322T120000Z\",\n"
        f"  \"profile_id\": \"{active_profile_id}\",\n"
        "  \"requested_promotion_stage\": \"PAPER_APPROVED\",\n"
        "  \"summary_text\": \"Manual guarded promotion after reviewing M21 evidence.\",\n"
        "  \"reason_codes\": [\n"
        "    \"OPERATOR_REVIEWED_EVIDENCE\",\n"
        "    \"MANUAL_M21_PROMOTION\"\n"
        "  ],\n"
        "  \"operator_confirmed\": true\n"
        "}\n"
        "'@ | Invoke-RestMethod -Method Post `\n"
        "  -Uri http://127.0.0.1:8000/continual-learning/promotions/promote-profile `\n"
        "  -ContentType 'application/json' `\n"
        "  -Body {$_}"
    )
    rollback_profile_id = rollback_target or "rollback-target-profile-id"
    rollback_payload = (
        "@'\n"
        "{\n"
        f"  \"decision_id\": \"rollback:{rollback_profile_id}:20260322T120500Z\",\n"
        f"  \"execution_mode\": \"{execution_mode}\",\n"
        f"  \"symbol\": \"{symbol}\",\n"
        f"  \"regime_label\": \"{regime_label}\",\n"
        "  \"summary_text\": \"Manual guarded rollback to the explicit prior M21 profile.\",\n"
        "  \"operator_confirmed\": true\n"
        "}\n"
        "'@ | Invoke-RestMethod -Method Post `\n"
        "  -Uri http://127.0.0.1:8000/continual-learning/promotions/rollback-active-profile `\n"
        "  -ContentType 'application/json' `\n"
        "  -Body {$_}"
    )
    with st.expander("Operator Guidance: Continual Learning Workflow"):
        st.caption(
            "Workflow remains local-first, manual, measurable, and reversible. "
            "The dashboard does not trigger writes directly in this packet."
        )
        st.markdown("Promotion payload")
        st.code(promote_payload, language="powershell")
        st.markdown("Rollback payload")
        st.code(rollback_payload, language="powershell")


def _build_continual_learning_signal_rows(*, snapshot) -> list[dict[str, object]]:
    rows = [
        {
            "symbol": signal.symbol,
            "continual_learning_status": signal.continual_learning_status,
            "continual_learning_profile_id": signal.continual_learning_profile_id,
            "continual_learning_frozen": signal.continual_learning_frozen,
            "candidate_type": signal.continual_learning_candidate_type,
            "promotion_stage": signal.continual_learning_promotion_stage,
            "baseline_target_type": signal.continual_learning_baseline_target_type,
            "baseline_target_id": signal.continual_learning_baseline_target_id,
            "drift_cap_status": signal.continual_learning_drift_cap_status,
            "reason_codes": ", ".join(signal.continual_learning_reason_codes),
        }
        for signal in snapshot.signals
    ]
    return rows or [{"status": "NONE"}]


def _build_continual_learning_event_rows(*, snapshot) -> list[dict[str, object]]:
    rows = [
        {
            "event_id": item.event_id,
            "event_type": item.event_type,
            "profile_id": item.profile_id,
            "experiment_id": item.experiment_id,
            "decision_id": item.decision_id,
            "reason_code": item.reason_code,
            "created_at": (
                None if item.created_at is None else item.created_at.isoformat()
            ),
            "highlight": item.event_type,
        }
        for item in snapshot.continual_learning.events
        if item.event_type in {"PROMOTION_APPLIED", "PROMOTION_BLOCKED", "ROLLBACK_APPLIED"}
    ]
    return rows or [{"status": "NONE"}]


def _build_continual_learning_breached_drift_rows(*, snapshot) -> list[dict[str, object]]:
    rows = [
        {
            "cap_id": item.cap_id,
            "execution_mode_scope": item.execution_mode_scope,
            "symbol_scope": item.symbol_scope,
            "regime_scope": item.regime_scope,
            "candidate_type": item.candidate_type,
            "status": item.status,
            "observed_drift_score": item.observed_drift_score,
            "reason_code": item.reason_code,
            "highlight": "DRIFT_BREACHED",
        }
        for item in snapshot.continual_learning.drift_caps
        if item.status == "BREACHED"
    ]
    return rows or [{"status": "NONE"}]


def _build_continual_learning_freeze_rows(*, snapshot) -> list[dict[str, object]]:
    rows = [
        {
            "symbol": signal.symbol,
            "continual_learning_profile_id": signal.continual_learning_profile_id,
            "continual_learning_status": signal.continual_learning_status,
            "continual_learning_frozen": signal.continual_learning_frozen,
            "reason_codes": ", ".join(signal.continual_learning_reason_codes),
            "highlight": "FROZEN_BY_HEALTH_GATE",
        }
        for signal in snapshot.signals
        if signal.continual_learning_frozen
        or "CONTINUAL_LEARNING_FROZEN_BY_HEALTH_GATE"
        in signal.continual_learning_reason_codes
    ]
    return rows or [{"status": "NONE"}]


def _build_active_alert_rows(*, snapshot, now: datetime) -> list[dict[str, object]]:
    if not snapshot.active_alerts.available:
        return [
            {
                "severity": "UNAVAILABLE",
                "category": "-",
                "execution_mode": "-",
                "symbol": "-",
                "reason_code": "-",
                "source_component": snapshot.active_alerts.error or "unavailable",
                "opened_at": None,
                "opened_age": None,
                "last_seen_at": None,
                "last_seen_age": None,
            }
        ]
    return [
        {
            "severity": row.severity,
            "category": row.category,
            "execution_mode": row.execution_mode,
            "symbol": row.symbol,
            "reason_code": row.reason_code,
            "source_component": row.source_component,
            "opened_at": row.opened_at.isoformat().replace("+00:00", "Z"),
            "opened_age": age_text(row.opened_at, now),
            "last_seen_at": row.last_seen_at.isoformat().replace("+00:00", "Z"),
            "last_seen_age": age_text(row.last_seen_at, now),
        }
        for row in snapshot.active_alerts.items
    ]


def _build_incident_timeline_rows(*, snapshot, now: datetime) -> list[dict[str, object]]:
    if not snapshot.alert_timeline.available:
        return [
            {
                "event_time": None,
                "event_age": None,
                "event_state": "UNAVAILABLE",
                "severity": "-",
                "category": "-",
                "symbol": "-",
                "reason_code": "-",
                "summary_text": snapshot.alert_timeline.error or "unavailable",
            }
        ]
    return [
        {
            "event_time": row.event_time.isoformat().replace("+00:00", "Z"),
            "event_age": age_text(row.event_time, now),
            "event_state": row.event_state,
            "severity": row.severity,
            "category": row.category,
            "symbol": row.symbol,
            "reason_code": row.reason_code,
            "summary_text": row.summary_text,
        }
        for row in snapshot.alert_timeline.items
    ]


def _build_startup_safety_rows(*, snapshot, now: datetime) -> list[dict[str, object]]:
    if not snapshot.startup_safety.available:
        return [
            {
                "startup_safety_passed": None,
                "primary_reason_code": None,
                "summary_text": snapshot.startup_safety.error or "unavailable",
                "startup_report_path": None,
                "generated_at": None,
                "generated_age": None,
            }
        ]
    row = snapshot.startup_safety
    return [
        {
            "startup_safety_passed": row.startup_safety_passed,
            "primary_reason_code": row.primary_reason_code,
            "summary_text": row.summary_text,
            "startup_report_path": row.startup_report_path,
            "generated_at": (
                None
                if row.generated_at is None
                else row.generated_at.isoformat().replace("+00:00", "Z")
            ),
            "generated_age": age_text(row.generated_at, now),
        }
    ]


def _build_daily_operations_summary_rows(*, snapshot) -> list[dict[str, object]]:
    if not snapshot.daily_operations_summary.available:
        return [
            {
                "summary_date": None,
                "unresolved_count": None,
                "highest_severity": None,
                "counts_by_category": snapshot.daily_operations_summary.error or "unavailable",
                "order_failure_counts": None,
                "drawdown_state": None,
                "actionable_signal_counts": None,
                "silence_flood_episodes": None,
                "live_mode_activation_count": None,
            }
        ]
    row = snapshot.daily_operations_summary
    return [
        {
            "summary_date": row.summary_date,
            "unresolved_count": row.unresolved_count,
            "highest_severity": row.highest_severity,
            "counts_by_category": row.counts_by_category,
            "order_failure_counts": row.order_failure_counts,
            "drawdown_state": row.drawdown_state,
            "actionable_signal_counts": row.actionable_signal_counts,
            "silence_flood_episodes": row.silence_flood_episodes,
            "live_mode_activation_count": row.live_mode_activation_count,
        }
    ]


def _venue_summary_cards(row: dict[str, object]) -> list[dict[str, str]]:
    mismatch_detail = "venues differ" if bool(row["venue_mismatch"]) else "venues aligned"
    return [
        {
            "label": "Market data venue",
            "value": str(row["market_data_venue"]),
        },
        {
            "label": "Execution venue",
            "value": str(row["execution_venue"]),
            "detail": mismatch_detail,
        },
        {
            "label": "Environment",
            "value": str(row["environment"]),
        },
        {
            "label": "Truth source",
            "value": str(row["portfolio_truth_source"]),
            "detail": (
                "account "
                + (
                    "-"
                    if row["validated_account_id"] in {None, ""}
                    else str(row["validated_account_id"])
                )
            ),
        },
    ]


def _render_compact_bar_chart(
    *,
    title: str,
    rows: list[dict[str, Any]],
    category_field: str,
    value_field: str,
) -> None:
    chart_rows = [
        row
        for row in rows
        if row.get(category_field) is not None and row.get(value_field) is not None
    ]
    if not chart_rows:
        return
    st.caption(title)
    st.vega_lite_chart(
        {
            "data": {"values": chart_rows},
            "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
            "encoding": {
                "x": {
                    "field": category_field,
                    "type": "nominal",
                    "sort": None,
                    "axis": {"labelAngle": 0, "title": None},
                },
                "y": {
                    "field": value_field,
                    "type": "quantitative",
                    "axis": {"title": None},
                },
                "tooltip": [
                    {"field": category_field, "type": "nominal"},
                    {"field": value_field, "type": "quantitative"},
                ],
            },
            "height": 220,
        },
        use_container_width=True,
    )


def _build_signal_distribution_chart_rows(
    latest_signals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0, "UNAVAILABLE": 0}
    for row in latest_signals:
        signal = str(row.get("signal") or "UNAVAILABLE")
        counts[signal] = counts.get(signal, 0) + 1
    return [
        {"signal": signal, "count": count}
        for signal, count in counts.items()
        if count > 0
    ]


def _build_regime_performance_chart_rows(
    performance_by_regime_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "regime_label": row["regime_label"],
            "total_pnl": float(row["total_pnl"]),
        }
        for row in performance_by_regime_rows
        if row.get("regime_label") is not None and row.get("total_pnl") is not None
    ]


def _build_alert_severity_chart_rows(
    active_alert_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in active_alert_rows:
        severity = str(row.get("severity") or "UNKNOWN")
        counts[severity] = counts.get(severity, 0) + 1
    return [
        {"severity": severity, "count": count}
        for severity, count in counts.items()
    ]


def _build_drift_cap_status_chart_rows(*, snapshot) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in snapshot.continual_learning.drift_caps:
        status = str(item.status or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    return [
        {"status": status, "count": count}
        for status, count in counts.items()
    ]


def _build_open_position_exposure_chart_rows(
    open_position_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    exposure_by_symbol: dict[str, float] = {}
    for row in open_position_rows:
        symbol = row.get("symbol")
        entry_notional = row.get("entry_notional")
        if symbol is None or entry_notional is None:
            continue
        exposure_by_symbol[str(symbol)] = exposure_by_symbol.get(str(symbol), 0.0) + float(
            entry_notional
        )
    return [
        {"symbol": symbol, "entry_notional": notional}
        for symbol, notional in exposure_by_symbol.items()
    ]


def _filter_trade_journal(
    *,
    traces: tuple[DecisionTraceSnapshot, ...],
    reference_time: datetime,
) -> tuple[DecisionTraceSnapshot, ...]:
    if not traces:
        return tuple()

    timestamps = sorted(trace.signal_as_of_time for trace in traces)
    min_timestamp = timestamps[0]
    max_timestamp = timestamps[-1]
    filter_columns = st.columns(4)
    selected_symbol = filter_columns[0].selectbox(
        "Symbol",
        options=["All"] + sorted({trace.symbol for trace in traces}),
        index=0,
    )
    selected_mode = filter_columns[1].selectbox(
        "Mode",
        options=["All"] + sorted({trace.execution_mode for trace in traces}),
        index=0,
    )
    selected_regime = filter_columns[2].selectbox(
        "Regime",
        options=["All"]
        + sorted({trace.regime_label or "UNKNOWN" for trace in traces}),
        index=0,
    )
    selected_reason_code = filter_columns[3].selectbox(
        "Reason code",
        options=["All"]
        + sorted(
            {
                trace.primary_reason_code or trace.signal_reason_code
                for trace in traces
                if (trace.primary_reason_code or trace.signal_reason_code) is not None
            }
        ),
        index=0,
    )

    range_columns = st.columns(4)
    selected_outcome = range_columns[0].selectbox(
        "Outcome category",
        options=["All"]
        + sorted({trace.risk_outcome or "SIGNAL_ONLY" for trace in traces}),
        index=0,
    )
    only_blocked = range_columns[1].checkbox("Only blocked", value=False)
    start_date = range_columns[2].date_input(
        "Start date",
        value=min_timestamp.date(),
    )
    start_clock = range_columns[3].time_input(
        "Start time (UTC)",
        value=time(min_timestamp.hour, min_timestamp.minute),
    )

    end_columns = st.columns(4)
    end_date = end_columns[0].date_input(
        "End date",
        value=max_timestamp.date(),
    )
    end_clock = end_columns[1].time_input(
        "End time (UTC)",
        value=time(max_timestamp.hour, max_timestamp.minute),
    )
    end_columns[2].metric("Newest trace age", age_text(max_timestamp, reference_time) or "-")
    end_columns[3].metric("Journal rows loaded", str(len(traces)))

    start_time = datetime.combine(start_date, start_clock, tzinfo=timezone.utc)
    end_time = datetime.combine(end_date, end_clock, tzinfo=timezone.utc)
    return filter_trade_journal_traces(
        traces,
        symbol=selected_symbol,
        mode=selected_mode,
        start_time=start_time,
        end_time=end_time,
        regime=selected_regime,
        reason_code=selected_reason_code,
        outcome_category=selected_outcome,
        only_blocked=only_blocked,
    )
def _render_rationale_report_downloads(recent_traces) -> None:
    if not recent_traces:
        return
    st.subheader("Rationale Reports")
    selected_label = st.selectbox(
        "Decision trace",
        options=[
            (
                f"{trace.decision_trace_id} | {trace.symbol} | "
                f"{trace.signal} | {trace.risk_outcome or 'PENDING'}"
            )
            for trace in recent_traces
        ],
        index=0,
    )
    selected_trace = next(
        trace
        for trace in recent_traces
        if selected_label.startswith(f"{trace.decision_trace_id} |")
    )
    st.write(
        "JSON report path: "
        f"`{selected_trace.json_report_path or 'unavailable'}`"
    )
    st.write(
        "Markdown report path: "
        f"`{selected_trace.markdown_report_path or 'unavailable'}`"
    )
    _render_download_button(
        label="Download JSON rationale report",
        report_path=selected_trace.json_report_path,
        mime_type="application/json",
    )
    _render_download_button(
        label="Download Markdown rationale report",
        report_path=selected_trace.markdown_report_path,
        mime_type="text/markdown",
    )


def _render_download_button(*, label: str, report_path: str | None, mime_type: str) -> None:
    if report_path is None:
        return
    path = Path(report_path)
    if not path.exists():
        st.caption(f"{label}: file not found at `{report_path}`")
        return
    st.download_button(
        label=label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime_type,
    )


if __name__ == "__main__":
    main()
