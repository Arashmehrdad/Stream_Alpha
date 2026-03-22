"""Read-only Stream Alpha M15 operator console."""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

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

# pylint: disable=too-many-locals,too-many-arguments,too-many-statements


def main() -> None:
    """Run the read-only Stream Alpha M15 operator console."""
    settings = Settings.from_env()
    trading_config = load_paper_trading_config(resolve_dashboard_trading_config_path())

    st.set_page_config(
        page_title="Stream Alpha Operator Console",
        page_icon="SA",
        layout="wide",
    )
    _maybe_enable_auto_refresh(settings.dashboard.refresh_seconds)

    st.title("Stream Alpha")
    st.caption("Milestone M15 operator console foundation")

    snapshot = asyncio.run(
        DashboardDataSources(
            settings=settings,
            trading_config=trading_config,
        ).load_snapshot()
    )
    runtime_profile = resolve_display_runtime_profile(snapshot=snapshot)
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
    overview_metrics = build_overview_metrics(
        snapshot=snapshot,
        trading_config=trading_config,
    )
    trader_freshness = build_trader_freshness(snapshot.database.engine_states)

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
            incidents=incidents,
            latest_recovery_rows=latest_recovery_rows,
            latest_blocked_trade_rows=latest_blocked_trade_rows,
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
    render_table("Latest Signals By Asset", latest_signals)
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


def _render_incidents_view(
    *,
    incidents,
    latest_recovery_rows,
    latest_blocked_trade_rows,
) -> None:
    render_incidents_panel(incidents)
    render_table("Latest Recovery Event", latest_recovery_rows)
    render_table("Latest Blocked Trade Rationale", latest_blocked_trade_rows)


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


def _maybe_enable_auto_refresh(refresh_seconds: int) -> None:
    if refresh_seconds <= 0:
        return
    components.html(
        f"""
        <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {refresh_seconds * 1000});
        </script>
        """,
        height=0,
        width=0,
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
