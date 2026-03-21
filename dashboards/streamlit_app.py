"""Read-only Stream Alpha M6 operator dashboard."""

from __future__ import annotations

import asyncio
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from app.common.config import Settings
from app.trading.config import load_paper_trading_config
from dashboards.data_sources import DashboardDataSources
from dashboards.view_models import (
    age_text,
    build_feature_lag_rows,
    build_reliability_status_rows,
    build_service_health_rows,
    build_latest_blocked_trade_rows,
    build_drawdown_curve_rows,
    build_equity_curve_rows,
    build_feature_snapshot_rows,
    build_latest_signal_rows,
    build_live_status_rows,
    build_open_position_rows,
    build_overview_metrics,
    build_performance_by_regime_rows,
    build_recent_decision_trace_rows,
    build_recent_closed_trade_rows,
    build_recent_ledger_rows,
    build_recent_order_audit_rows,
    build_symbol_freshness_rows,
    build_trader_freshness,
    latest_feature_as_of,
)
from dashboards.widgets import render_health_card, render_table


CONFIG_PATH = Path("configs/paper_trading.yaml")


def main() -> None:
    """Run the read-only Stream Alpha M6 dashboard."""
    settings = Settings.from_env()
    trading_config = load_paper_trading_config(CONFIG_PATH)

    st.set_page_config(
        page_title="Stream Alpha Dashboard",
        page_icon="SA",
        layout="wide",
    )
    _maybe_enable_auto_refresh(settings.dashboard.refresh_seconds)

    st.title("Stream Alpha")
    st.caption("Milestone M6 operator dashboard")

    with st.sidebar:
        st.header("Runtime")
        st.write(f"Inference API: `{settings.dashboard.inference_api_base_url}`")
        st.write(f"Feature source table: `{settings.tables.feature_ohlc}`")
        st.write(f"Paper trader service: `{trading_config.service_name}`")
        st.write(f"Execution mode: `{trading_config.execution.mode}`")
        if trading_config.execution.mode == "live":
            st.write("Broker: `alpaca`")
            st.write(
                "Expected environment: "
                f"`{trading_config.execution.live.expected_environment}`"
            )
        st.write(f"Refresh target: `{settings.dashboard.refresh_seconds}s`")
        if st.button("Refresh now", width="stretch"):
            st.rerun()

    snapshot = asyncio.run(
        DashboardDataSources(
            settings=settings,
            trading_config=trading_config,
        ).load_snapshot()
    )

    _render_mode_banner(trading_config=trading_config, snapshot=snapshot)

    overview_tab, signals_tab, trading_tab = st.tabs(
        ["Overview", "Signals and Features", "Trading"]
    )

    latest_signals = build_latest_signal_rows(
        symbols=trading_config.symbols,
        signals=snapshot.signals,
    )
    latest_features = build_feature_snapshot_rows(
        symbols=trading_config.symbols,
        features=snapshot.database.latest_features,
    )
    reliability_rows = build_reliability_status_rows(
        api_health=snapshot.api_health,
        system_reliability=snapshot.system_reliability,
        reliability_states=snapshot.database.reliability_states,
        latest_recovery_event=snapshot.database.latest_recovery_event,
    )
    service_health_rows = build_service_health_rows(snapshot.system_reliability)
    lag_rows = build_feature_lag_rows(snapshot.system_reliability)
    freshness_rows = build_symbol_freshness_rows(snapshot.freshness)

    with overview_tab:
        _render_overview(
            trading_config=trading_config,
            snapshot=snapshot,
            latest_signals=latest_signals,
            reliability_rows=reliability_rows,
            service_health_rows=service_health_rows,
            lag_rows=lag_rows,
            freshness_rows=freshness_rows,
        )

    with signals_tab:
        _render_signals_and_features(
            snapshot=snapshot,
            latest_signals=latest_signals,
            latest_features=latest_features,
        )

    with trading_tab:
        _render_trading(
            settings=settings,
            trading_config=trading_config,
            snapshot=snapshot,
        )

# pylint: disable=too-many-locals
def _render_overview(  # pylint: disable=too-many-arguments
    *,
    trading_config,
    snapshot,
    latest_signals,
    reliability_rows,
    service_health_rows,
    lag_rows,
    freshness_rows,
) -> None:
    health_columns = st.columns(4)
    api_health = snapshot.api_health
    db_health = snapshot.database
    trader_freshness = build_trader_freshness(snapshot.database.engine_states)
    overview_metrics = build_overview_metrics(
        snapshot=snapshot,
        trading_config=trading_config,
    )

    with health_columns[0]:
        render_health_card(
            title="Inference API",
            state=api_health.status,
            detail=(
                api_health.error
                if api_health.error is not None
                else f"Checked {age_text(api_health.checked_at)} ago"
            ),
            healthy=api_health.available,
        )
    with health_columns[1]:
        render_health_card(
            title="PostgreSQL",
            state="healthy" if db_health.available else "unavailable",
            detail=(
                db_health.error
                if db_health.error is not None
                else f"Checked {age_text(db_health.checked_at)} ago"
            ),
            healthy=db_health.available,
        )
    with health_columns[2]:
        model_loaded = api_health.model_loaded and api_health.model_name is not None
        render_health_card(
            title="Model",
            state=api_health.model_name or "not loaded",
            detail=api_health.model_artifact_path or "Model artifact unavailable",
            healthy=model_loaded,
        )
    with health_columns[3]:
        render_health_card(
            title="Paper Trader",
            state=trader_freshness.state,
            detail=trader_freshness.message,
            healthy=trader_freshness.state == "healthy",
        )

    if snapshot.system_reliability is None or not snapshot.system_reliability.available:
        st.warning(
            "Canonical reliability summary is unavailable. "
            "The dashboard is showing fallback degraded state where possible."
        )

    if overview_metrics is None:
        st.error("Trading KPIs are unavailable because PostgreSQL could not be read.")
    else:
        kpi_columns = st.columns(5)
        kpi_columns[0].metric("Realized PnL", f"{overview_metrics.realized_pnl:,.2f}")
        kpi_columns[1].metric("Unrealized PnL", f"{overview_metrics.unrealized_pnl:,.2f}")
        kpi_columns[2].metric("Total PnL", f"{overview_metrics.total_pnl:,.2f}")
        kpi_columns[3].metric("Max Drawdown", f"{overview_metrics.max_drawdown:.2%}")
        kpi_columns[4].metric("Open Positions", str(overview_metrics.open_position_count))

        secondary_columns = st.columns(4)
        secondary_columns[0].metric("Cash Balance", f"{overview_metrics.cash_balance:,.2f}")
        secondary_columns[1].metric("Win Rate", f"{overview_metrics.win_rate:.2%}")
        secondary_columns[2].metric("Turnover", f"{overview_metrics.turnover:.2f}")
        secondary_columns[3].metric("Sharpe-Like", f"{overview_metrics.sharpe_like:.2f}")

    render_table("Latest Signals", latest_signals)
    render_table("Reliability Status", reliability_rows)
    render_table("Per-Service Health", service_health_rows)
    render_table("Feature Consumer Lag", lag_rows)
    render_table("Per-Symbol Freshness", freshness_rows)
    if api_health.regime_loaded:
        st.caption(
            "Regime runtime: "
            f"`{api_health.regime_run_id}` from `{api_health.regime_artifact_path}`"
        )

    if snapshot.database.available:
        chart_rows = build_equity_curve_rows(
            positions=snapshot.database.positions,
            initial_cash=trading_config.risk.initial_cash,
            latest_prices=snapshot.database.latest_prices,
            fee_bps=trading_config.risk.fee_bps,
            as_of_time=latest_feature_as_of(snapshot.database.latest_features),
        )
        drawdown_rows = build_drawdown_curve_rows(chart_rows)
        chart_columns = st.columns(2)
        with chart_columns[0]:
            st.subheader("Equity and PnL")
            st.line_chart(
                chart_rows,
                x="timestamp",
                y=["equity", "cumulative_pnl"],
                width="stretch",
            )
        with chart_columns[1]:
            st.subheader("Drawdown")
            st.line_chart(
                drawdown_rows,
                x="timestamp",
                y="drawdown",
                width="stretch",
            )

        st.caption(
            "Latest canonical feature age: "
            f"{age_text(latest_feature_as_of(snapshot.database.latest_features))}"
        )

    st.caption(
        "Dashboard reads only from the accepted M4 inference API and M5 PostgreSQL tables."
    )
# pylint: enable=too-many-locals


def _render_signals_and_features(*, snapshot, latest_signals, latest_features) -> None:
    if not snapshot.api_health.available:
        st.warning(
            "Inference API is degraded. Signal rows below show unavailable states where applicable."
        )
    if not snapshot.database.available:
        st.error(
            "PostgreSQL is unavailable. "
            "Feature snapshots cannot be loaded until the database recovers."
        )

    render_table("Latest Signals By Asset", latest_signals)
    render_table("Latest Canonical Feature Snapshot", latest_features)

# pylint: disable=too-many-locals
def _render_trading(
    *,
    settings: Settings,
    trading_config,
    snapshot,
) -> None:
    if not snapshot.database.available:
        st.error("PostgreSQL is unavailable. Trading state cannot be rendered.")
        return

    freshness = build_trader_freshness(snapshot.database.engine_states)
    freshness_columns = st.columns(4)
    freshness_columns[0].metric("Tracked Symbols", str(freshness.symbols_tracked))
    freshness_columns[1].metric("Pending Signals", str(freshness.pending_signal_count))
    freshness_columns[2].metric(
        "Latest Processed",
        "-"
        if freshness.latest_processed_interval_begin is None
        else freshness.latest_processed_interval_begin.isoformat(),
    )
    freshness_columns[3].metric(
        "Slowest Processed",
        "-"
        if freshness.slowest_processed_interval_begin is None
        else freshness.slowest_processed_interval_begin.isoformat(),
    )

    open_position_rows = build_open_position_rows(
        positions=snapshot.database.positions,
        latest_prices=snapshot.database.latest_prices,
        fee_bps=trading_config.risk.fee_bps,
    )
    recent_closed_rows = build_recent_closed_trade_rows(
        snapshot.database.recent_closed_positions
    )
    recent_ledger_rows = build_recent_ledger_rows(snapshot.database.recent_ledger_entries)
    recent_order_audit_rows = build_recent_order_audit_rows(
        snapshot.database.recent_order_events
    )
    recent_decision_trace_rows = build_recent_decision_trace_rows(
        snapshot.database.recent_decision_traces
    )
    latest_blocked_trade_rows = build_latest_blocked_trade_rows(
        snapshot.database.latest_blocked_trade
    )
    live_status_rows = build_live_status_rows(
        trading_config=trading_config,
        live_safety_state=snapshot.database.live_safety_state,
    )
    by_regime_rows = build_performance_by_regime_rows(
        snapshot=snapshot,
        trading_config=trading_config,
    )

    st.caption(f"Execution mode: `{trading_config.execution.mode}`")
    if trading_config.execution.mode == "live":
        render_table("Live Status", live_status_rows)
    render_table("Open Positions", open_position_rows)
    render_table("Recent Closed Trades", recent_closed_rows)
    render_table("Recent Ledger Activity", recent_ledger_rows)
    render_table(
        "Recent Live Order Audit"
        if trading_config.execution.mode == "live"
        else "Recent Order Audit",
        recent_order_audit_rows,
    )
    render_table("Recent Decision Traces", recent_decision_trace_rows)
    render_table("Latest Blocked Trade Rationale", latest_blocked_trade_rows)
    _render_rationale_report_downloads(snapshot.database.recent_decision_traces)
    render_table("Performance By Regime", by_regime_rows)

    st.caption(
        "Recent closed trades limit: "
        f"{settings.dashboard.recent_trades_limit}; "
        f"recent ledger limit: {settings.dashboard.recent_ledger_limit}"
    )
# pylint: enable=too-many-locals


def _render_mode_banner(*, trading_config, snapshot) -> None:
    if trading_config.execution.mode == "live":
        live_state = snapshot.database.live_safety_state
        status_suffix = (
            "startup checks passed"
            if live_state is not None and live_state.startup_checks_passed
            else "startup checks not passed"
        )
        st.error(f"LIVE MODE ENABLED: {status_suffix}")
        return

    st.info(f"Execution mode: {trading_config.execution.mode}")


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
