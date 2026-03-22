# Stream Alpha

Stream Alpha is a local-first crypto market data pipeline built with Docker Compose, Redpanda Community Edition, PostgreSQL, and Python services.

Milestone `M1` ingests Kraken public WebSocket v2 market data for `BTC/USD`, `ETH/USD`, and `SOL/USD`, normalizes `trade` and `ohlc` events, publishes them into Redpanda, and writes raw rows into PostgreSQL.

Milestone `M2` adds an OHLC-only feature consumer. It reads `raw.ohlc`, finalizes closed candles safely, computes a minimal rolling feature set using only current and past finalized candles, and upserts typed rows into PostgreSQL table `feature_ohlc`.

Milestone `M3` adds an offline training pipeline. It builds a labeled dataset from `feature_ohlc`, evaluates naive and learned classifiers with purged expanding walk-forward validation, and saves model plus evaluation artifacts under `artifacts/training/m3/<run_id>/`.

Milestone `M4` adds a minimal FastAPI inference API. It loads the accepted saved M3 artifact, reads the latest canonical feature row from `feature_ohlc`, and serves health, latest-feature, prediction, signal, and JSON metrics endpoints.

Milestone `M5` adds a minimum correct paper-trading engine. It polls finalized canonical candles from `feature_ohlc`, asks the existing M4 `/signal` endpoint for the authoritative BUY/SELL/HOLD decision for each exact candle, simulates long-only spot fills, persists positions and trade ledger rows in PostgreSQL, and writes rolling paper-trading summaries under `artifacts/paper_trading/`.

Milestone `M6` adds a read-only Streamlit operator dashboard. It reads only from the accepted M4 inference API and PostgreSQL tables, then shows health, latest signals, canonical feature snapshots, open positions, recent trades, ledger activity, and paper-trading PnL/drawdown.

Milestone `M7` adds a local-first retraining and model-comparison workflow. It retrains challengers from canonical `feature_ohlc`, compares them to the current promoted champion with explicit file-based policy checks, promotes only passing models into a local registry, and supports rollback without retraining.

Milestone `M8` foundation adds an explicit offline regime workflow. It reads canonical `feature_ohlc`, fits deterministic per-symbol volatility and trend thresholds, classifies rows into `TREND_UP`, `TREND_DOWN`, `RANGE`, or `HIGH_VOL`, and writes explicit artifacts under `artifacts/regime/m8/<run_id>/`.

Milestone `M9` adds a minimum regime-aware live signal foundation. It loads the saved M8 `thresholds.json`, resolves regime labels for the exact canonical feature row used by M4, extends the accepted inference and paper-trading contracts additively, and surfaces by-regime performance in the existing dashboard and trading artifacts.

Milestone `M10` adds the authoritative pre-trade risk and sizing layer. It blocks unsafe BUY decisions explicitly, keeps SELL/HOLD behavior inspectable, records deterministic reason codes plus ordered sizing adjustments, and persists restart-safe risk state without moving signal authority away from M4.

Milestone `M11` adds a minimum execution abstraction after M10 risk approval. It keeps M4 authoritative for signals, keeps M10 authoritative for risk and sizing, records deterministic idempotent order requests plus lifecycle events, and supports two local-only execution modes: `paper` and `shadow`.

Milestone `M12` adds a guarded live trading foundation on top of the accepted execution layer. It adds a third `live` adapter mode for Alpaca Trading API account validation and order submission, requires explicit runtime arming plus local safety checks before any broker submission, keeps live orders tiny and whitelisted, and writes startup/live-status artifacts plus explicit live audit rows.

Milestone `M13` foundation adds explicit reliability and recovery state around the accepted runtime paths. Packet 1 adds checked-in reliability config, typed freshness and breaker primitives, and additive PostgreSQL reliability tables. Packet 2 wires heartbeats, exact-row freshness evaluation, a trader-side inference breaker, stale pending-signal recovery, reliability artifacts, and compact dashboard reliability views. Packet 3 finalizes M13 with explicit feature-consumer lag detection, canonical cross-service health aggregation, feed-freshness visibility, and one unified operator-facing reliability summary without changing M4, M10, M11, or M12 authority boundaries.

Milestone `M14` Packet 1 adds an M4-side explainability foundation only. It keeps M4 authoritative for prediction and signal generation, exposes `model_version` on decisions, adds deterministic top-feature contributions from one-at-a-time reference ablation against a persisted reference vector, and extends `/predict` plus `/signal` with additive explanation payloads without changing M10, M11, M12, or M13 behavior.

Milestone `M14` Packet 2 adds a canonical decision-trace and risk-rationale foundation around the accepted M4 -> M10 -> M11/M12 path. It persists one JSONB-backed `decision_traces` row per authoritative M4 signal, enriches that row with the M4 explanation payload plus explicit M10 risk rationale, and links `paper_risk_decisions`, `execution_order_requests`, and `paper_engine_state` to the same trace without changing any authority boundary or adding dashboard/report generation yet.

Milestone `M14` Packet 3 finishes the remaining explainability linkage without drifting into M15 console maturity. It carries the same canonical trace id through execution-facing rows, writes deterministic JSON plus Markdown rationale reports under `artifacts/rationale/`, persists those report paths back into `decision_traces`, and adds one minimal dashboard view for recent traces, blocked-trade rationale, and report downloads.

Milestone `M15` upgrades the Streamlit UI into an operator console foundation. It keeps all accepted M4, M10, M11, M12, M13, and M14 authority boundaries intact, replaces the older three-tab layout with explicit market, signals, trades, risk, health, models, and incidents views, adds a persistent mode and safety banner, surfaces venue and environment truth, adds a filterable trade journal from canonical decision traces, and makes unsafe state visibly obvious without adding alerting, broker import, or a new backend service.

Milestone `M16` adds the deployment and environment foundation. It keeps M4, M10, M11, M12, M13, M14, and M15 authority boundaries intact, adds one reusable app image, Compose profiles for `dev`, `paper`, `shadow`, and `live`, a one-shot startup validator, env-driven trading-config selection for the trader and dashboard, startup-report metadata on the existing inference APIs, PowerShell startup helpers, and explicit local retention rules without adding a new orchestration platform or alerting layer.

Milestone `M17` adds a local-first operational alerting foundation on top of the accepted M12 through M16 runtime truth. Packet 1 normalizes alert events and active state into PostgreSQL plus canonical startup-safety and daily-summary artifacts. Packet 2 adds read-only API surfaces and additive operator-console views for active alerts, incident timeline, startup safety, and daily operations summaries without changing trading, risk, execution, reliability, or explainability authority.

Milestone `M19` adds an adaptive intelligence foundation on top of the accepted M4, M10, M14, M15, M17, and M18 boundaries. It keeps M4 authoritative for prediction and signal generation, keeps M10 authoritative for final risk clamps and sizing safety, keeps all adaptation logic local-first and bounded, persists auditable adaptive state in PostgreSQL, and exposes read-only adaptation status through the existing inference API and operator dashboard.

## Deployment

M16 introduces one-command local startup helpers plus a deployment guide:

- [docs/deployment.md](docs/deployment.md) documents the exact `dev`, `paper`, `shadow`, and `live` startup flow.
- `.\scripts\start-stack.ps1 -Profile dev` starts the blank-clone-friendly development stack.
- `.\scripts\start-stack.ps1 -Profile paper` starts the full local stack only when model and regime artifacts already exist locally.
- `.\scripts\stop-stack.ps1` stops the Compose stack without deleting local state.
- `.\scripts\reset-state.ps1` removes Compose volumes plus runtime-only artifacts, but keeps long-lived training, regime, and registry evidence.
- `.\scripts\prune-runtime-artifacts.ps1 -RetentionDays 14` deletes old runtime artifacts without touching promoted models or regime artifacts.

Truthful M16 startup scope:

- Blank-clone `dev` works after copying `.env.example` to `.env`.
- Blank-clone `paper`, `shadow`, and `live` do not work honestly until local model and regime artifacts exist, or a local registry-backed champion exists under `artifacts/registry/`.
- `live` additionally requires `.env.secrets` plus explicit guarded-live arming env values, and startup validation fails closed when they are missing.

## M19 Scope

M19 adds:
- one bounded adaptation package under `app/adaptation/`
- one checked-in local config at `configs/adaptation.yaml`
- one local artifact root under `artifacts/adaptation/`
- five canonical PostgreSQL adaptive tables: `adaptive_drift_state`, `adaptive_performance_windows`, `adaptive_profiles`, `adaptive_challenger_runs`, and `adaptive_promotion_decisions`
- additive read-only inference endpoints: `GET /adaptation/summary`, `GET /adaptation/drift`, `GET /adaptation/performance`, `GET /adaptation/profiles`, and `GET /adaptation/promotions`
- additive `/health`, `/predict`, and `/signal` fields for adaptation status, calibrated confidence, effective thresholds, bounded adaptive sizing, and health-gate freeze visibility
- additive decision-trace context so adaptation rationale is auditable alongside M4 explanation and M10 risk rationale
- additive operator-console visibility in the existing Models view

M19 does not do:
- move signal authority away from M4
- move risk and final exposure authority away from M10
- create a new service, control plane, queue, or scheduler
- self-retrain or self-promote directly in `live`
- bypass reliability, freshness, startup-safety, or execution guardrails
- replace M7 promotion policy, M14 trace authority, M15 console structure, or M17 incident truth

M19 local-only artifact paths:
- `artifacts/adaptation/` for adaptation reports and evidence
- existing database tables for adaptive drift, performance, profiles, challenger runs, and promotion decisions
- existing decision-trace and dashboard paths for read-only adaptation visibility

M19 validation commands:

```powershell
pytest tests/test_adaptation_service.py tests/test_adaptation_repository.py tests/test_inference_api.py tests/test_dashboard_data_sources.py
pylint app/adaptation app/inference app/trading tests/test_adaptation_service.py tests/test_adaptation_repository.py tests/test_inference_api.py tests/test_dashboard_data_sources.py
```

M19 rollback commands:

```powershell
git diff -- app/adaptation configs/adaptation.yaml app/inference app/trading app/explainability dashboards tests README.md
git restore --source=HEAD --worktree --staged app/adaptation configs/adaptation.yaml app/inference app/trading app/explainability dashboards tests README.md
```

M19 rollback limitations:
- the `git restore` command above is only safe if you intend to discard the local M19 changes in those paths
- adaptive PostgreSQL rows and tables are not removed by git rollback and must be cleaned separately if you want a full datastore rollback
- `artifacts/adaptation/` is local evidence and can be deleted separately when a local-only cleanup is intended

## M20 Scope

Milestone `M20` is complete and accepted.

M20 adds the Dynamic Ensemble Foundation. Its purpose is to make prediction quality depend on market context instead of one static brain, while keeping the accepted local-first architecture inspectable and bounded.

M20 scope includes:
- multiple specialized candidate models instead of one always-on voting peer set
- regime-aware ensemble weighting so trend and range conditions can prefer different specialists
- explicit trend-versus-range comparison during offline ensemble research
- ensemble agreement as a first-class confidence input through agreement-adjusted effective confidence
- champion/challenger ensemble experiments with explicit evidence, activation, and rollback paths

### Canonical Runtime Design

The accepted M20 runtime path uses one canonical 3-role ensemble roster:
- `GENERALIST`
- `TREND_SPECIALIST`
- `RANGE_SPECIALIST`

The accepted runtime behavior is:
- scoped active ensemble profile loading from PostgreSQL
- registry-backed candidate scoring against the exact canonical M4 feature row
- regime-aware weight selection from the active ensemble profile
- agreement-adjusted effective confidence for downstream signal and adaptation use
- explicit fallback to the prior single-model path when the ensemble cannot run honestly
- truthful top-level active-ensemble identity:
  - `model_name = dynamic_ensemble`
  - `model_version = ensemble_profile:<profile_id>`

Nested ensemble payloads remain additive and explicit. They continue to show the actual participating candidate models, weights, and fallback context rather than hiding that detail behind the top-level ensemble identity.

### Model-Family Guidance

The accepted Packet 2 model-family direction in M20 is:
- AutoGluon for the `GENERALIST` role and for the champion/challenger baseline comparison path
- NeuralForecast specialist research pool for:
  - `NHITS`
  - `NBEATSx`
  - `TFT`
  - `PatchTST`
- River is not the primary production ensemble voter in M20

### Promotion And Rollback

The accepted M20 profile lifecycle is explicit and local-first:
- draft profile creation from one selected 3-role roster
- explicit promotion or activation evidence before a profile becomes active
- explicit rollback target behavior for returning to a known prior profile
- no automatic live mutation of the active runtime profile
- no uncontrolled runtime switching

### Evaluation And Evidence

Packet 2 ensemble research evaluates candidates honestly across these slices:
- `ALL`
- `TREND_COMBINED`
- `RANGE`
- `HIGH_VOL` as diagnostic-only context

The main evidence metrics are:
- net PnL after fees and slippage
- max drawdown
- Calmar ratio
- profit factor
- signal precision
- trade count
- blocked-trade rate

M20 keeps this evidence local-first and inspectable. Research outputs, draft profiles, promotion decisions, and rollback decisions are written as local artifacts and PostgreSQL records rather than hidden behind a remote control plane.

### Explainability Note

Active ensemble explainability remains in truthful placeholder mode in the accepted M20 runtime. Weighted aggregate ensemble explainability is still deferred, so the README does not claim that aggregated active-ensemble explainability is already implemented.

### Architecture Boundaries

M20 does not change the accepted authority boundaries:
- M4 remains authoritative for prediction and signal generation
- M10 remains authoritative for risk approval and sizing boundaries
- M11 remains authoritative for order lifecycle truth
- M12 remains authoritative for guarded live controls
- M13 remains authoritative for freshness and reliability HOLD behavior
- M14 remains authoritative for decision traces
- M18 remains authoritative for honest evaluation truth
- M19 remains authoritative for bounded adaptation

### Deferred Beyond M20

Intentionally deferred beyond the accepted M20 scope:
- weighted aggregate ensemble explainability
- M21 continual learning
- RL
- news and sentiment integration
- live self-retraining
- dashboard redesign beyond additive surfaces

### Validation And Quality

The accepted M20 implementation is documented as:
- validated with `pytest`
- held to a `pylint` target of `10.00/10` on touched files
- local-first and inspectable by design

## Repository Tree

```text
.
|-- .env.example
|-- .gitignore
|-- README.md
|-- app
|   |-- __init__.py
|   |-- common
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- logging.py
|   |   |-- models.py
|   |   |-- serialization.py
|   |   `-- time.py
|   |-- explainability
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- schemas.py
|   |   `-- service.py
|   |-- features
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- db.py
|   |   |-- engine.py
|   |   |-- main.py
|   |   |-- models.py
|   |   |-- service.py
|   |   `-- state.py
|   |-- inference
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- db.py
|   |   |-- main.py
|   |   |-- schemas.py
|   |   `-- service.py
|   |-- regime
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- artifacts.py
|   |   |-- config.py
|   |   |-- dataset.py
|   |   |-- live.py
|   |   `-- service.py
|   |-- reliability
|   |   |-- __init__.py
|   |   |-- artifacts.py
|   |   |-- config.py
|   |   |-- schemas.py
|   |   |-- service.py
|   |   `-- store.py
|   |-- trading
|   |   |-- __init__.py
|   |   |-- alpaca.py
|   |   |-- config.py
|   |   |-- engine.py
|   |   |-- execution.py
|   |   |-- live.py
|   |   |-- metrics.py
|   |   |-- repository.py
|   |   |-- risk.py
|   |   |-- runner.py
|   |   |-- schemas.py
|   |   `-- signal_client.py
|   |-- training
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- baselines.py
|   |   |-- compare.py
|   |   |-- dataset.py
|   |   |-- promote.py
|   |   |-- registry.py
|   |   |-- retrain.py
|   |   |-- rollback.py
|   |   |-- service.py
|   |   `-- splits.py
|   `-- ingestion
|       |-- __init__.py
|       |-- __main__.py
|       |-- backfill_ohlc.py
|       |-- db.py
|       |-- kraken.py
|       |-- main.py
|       |-- normalizers.py
|       |-- publisher.py
|       `-- service.py
|-- configs
|   |-- explainability.yaml
|   |-- paper_trading.yaml
|   |-- reliability.yaml
|   |-- regime.m8.json
|   |-- regime_signal_policy.json
|   |-- redpanda-console-config.yml
|   |-- training.m3.json
|   `-- training.m7.json
|-- dashboards
|   |-- __init__.py
|   |-- data_sources.py
|   |-- streamlit_app.py
|   |-- view_models.py
|   `-- widgets.py
|-- docker
|   `-- producer.Dockerfile
|-- docker-compose.yml
|-- docs
|   |-- m1-overview.md
|   `-- screenshots
|       `-- README.md
|-- requirements.txt
|-- scripts
|   |-- check-db.ps1
|   |-- check-topics.ps1
|   |-- m3-smoke-run.sh
|   |-- run_paper_trader.py
|   `-- tail-producer.ps1
`-- tests
    |-- __init__.py
    |-- test_dashboard_data_sources.py
    |-- test_dashboard_view_models.py
    |-- test_inference_api.py
    |-- test_inference_db.py
    |-- test_inference_model_loader.py
    |-- test_feature_engine.py
    |-- test_feature_state.py
    |-- test_normalizers.py
    |-- test_publish_smoke.py
    |-- test_training_compare.py
    |-- test_training_labels.py
    |-- test_training_registry.py
    |-- test_training_splits.py
    |-- training_workflow_helpers.py
    `-- trading
        |-- test_alpaca.py
        |-- test_execution.py
        |-- test_engine.py
        |-- test_live.py
        |-- test_metrics.py
        |-- test_risk.py
        `-- test_runner_idempotency.py
```

## Services

- `redpanda`: local Redpanda broker for topics and consumer groups
- `redpanda-console`: topic and cluster inspection UI
- `postgres`: local PostgreSQL for raw and feature tables
- `producer`: Kraken public WebSocket v2 ingestion service from M1
- `features`: OHLC-only feature consumer from M2
- `inference`: FastAPI prediction API from M4, run directly from the repo with the accepted M3 artifact
- `paper-trader`: long-only spot execution engine from M5/M9/M11/M12, run directly from the repo against canonical features plus authoritative M4 signals, M9 regime decisions, the M10 risk engine, and the configured paper, shadow, or guarded live execution mode
- `dashboard`: read-only Streamlit operator console from M6 through M15, run directly from the repo against the accepted API and PostgreSQL sources, including additive regime, reliability, live-safety, and decision-trace visibility

## M2 Scope

M2 does:
- consume only `raw.ohlc`
- finalize candles on next-interval arrival or grace-based sweep
- bootstrap rolling state from `raw_ohlc`
- compute typed feature rows for finalized candles only
- upsert rows into `feature_ohlc`

M2 does not do:
- consume `raw.trades`
- publish a feature topic
- create dashboards, APIs, training jobs, or inference services
- add trading, signals, or paper trading logic

## M3 Scope

M3 does:
- use `feature_ohlc` as the canonical offline training source
- construct next-3-candle binary direction labels
- evaluate `persistence_3` and `DummyClassifier(strategy="most_frequent")` baselines
- train `LogisticRegression` and `HistGradientBoostingClassifier`
- use a global purged expanding walk-forward split with 5 test folds
- save explicit run artifacts under `artifacts/training/m3/<run_id>/`

M3 does not do:
- serve inference online
- generate signals
- paper trade
- add FastAPI, dashboards, MLflow, RL, or sentiment/news modules

## M4 Scope

M4 does:
- load the accepted saved M3 model artifact directly from disk
- read the latest finalized canonical row from `feature_ohlc`
- serve `GET /health`, `GET /latest-features`, `GET /predict`, `GET /signal`, and `GET /metrics`
- return BUY, SELL, or HOLD using explicit probability thresholds
- maintain simple in-memory request and latency counters since startup

M4 does not do:
- retrain models
- accept custom feature payloads
- add batch endpoints, dashboards, paper trading, or signal-history storage
- introduce MLflow, Prometheus, Grafana, or deployment automation

## M5 Scope

M5 does:
- poll newly finalized canonical candles from `feature_ohlc`
- call the existing M4 `/signal` endpoint for each exact candle
- simulate long-only spot BUY/SELL/HOLD execution with next-open fills
- apply fixed-fraction sizing, cooldown, stop loss, take profit, max open positions, and max exposure per asset
- persist `paper_positions`, `paper_trade_ledger`, and `paper_engine_state`
- write rolling summary artifacts under `artifacts/paper_trading/`

M5 does not do:
- place live orders
- short, pyramid, or use leverage
- add broker integrations, dashboards, monitoring stacks, or retraining hooks

## M6 Scope

M6 does:
- run one read-only Streamlit dashboard locally from the repo
- call the accepted M4 `GET /health` and `GET /signal` endpoints
- read canonical `feature_ohlc` rows plus M5 trading tables from PostgreSQL
- show health, latest signals, latest canonical feature snapshots, open positions, recent trades, recent ledger activity, and paper-trading PnL/drawdown
- surface degraded states clearly when the inference API or PostgreSQL is unavailable

M6 does not do:
- retrain or reload models from any new workflow
- add a new backend service for the dashboard
- write dashboard state back into PostgreSQL
- add dashboards beyond Streamlit, live trading, RL, sentiment/news, MLflow, Grafana, Prometheus, or alerting

## M7 Scope

M7 does:
- retrain challengers from canonical `feature_ohlc` using the accepted M3 evaluation flow
- compare challengers to the current champion with explicit file-based policy checks
- use long-only after-cost `mean_long_only_net_value_proxy` as the primary promotion metric
- save regime-sliced long-only economics in `summary.json`
- require challengers to be positive after costs and beat both `persistence_3` and `dummy_most_frequent` before promotion
- store challenger artifacts under `artifacts/training/m7/<run_id>/`
- maintain a local immutable registry under `artifacts/registry/`
- support explicit rollback by `model_version`
- let M4 inference resolve the current champion from the registry when `INFERENCE_MODEL_PATH` is empty

M7 does not do:
- add MLflow, schedulers, or background retraining daemons
- add dashboard controls for retraining or promotion
- change M3 labels, models, splits, or artifact schema
- change M4 response contracts, add champion/challenger online serving, or retrain during rollback

## M8 Scope

M8 foundation does:
- read only from canonical `feature_ohlc`
- fit per-symbol `high_vol_threshold` from the 75th percentile of `realized_vol_12`
- fit per-symbol `trend_abs_threshold` from the 60th percentile of `abs(momentum_3)`
- classify rows with explicit deterministic rules into `TREND_UP`, `TREND_DOWN`, `RANGE`, or `HIGH_VOL`
- write explicit offline artifacts under `artifacts/regime/m8/<run_id>/`

M8 foundation does not do:
- change M4, M5, M6, or M7 behavior
- add a `/regime` endpoint, schedulers, background workers, or new PostgreSQL tables
- add clustering, RL, sentiment, anomaly detection, MLflow, or notebooks

## M9 Scope

M9 does:
- load the latest saved M8 `thresholds.json` by default, or an explicit threshold artifact path when configured
- resolve the regime for the exact canonical `feature_ohlc` row used by `GET /predict`, `GET /signal`, and `GET /regime`
- extend `GET /predict` with `regime_label` and `regime_run_id`
- extend `GET /signal` with `regime_label`, `regime_run_id`, and `trade_allowed`
- add a read-only `GET /regime` endpoint
- apply regime-dependent BUY/SELL thresholds and BUY no-trade rules in M4 only
- persist additive regime labels in `paper_engine_state`, `paper_positions`, and `paper_trade_ledger`
- extend paper-trading summaries with by-regime hit rate and PnL outputs
- surface regime fields and a compact by-regime performance table in the existing M6 dashboard

M9 does not do:
- add live trading, shorting, portfolio allocation, or position-sizing changes
- add a new PostgreSQL table, orchestration stack, scheduler, control plane, or background daemon
- duplicate regime logic in M5 or change M7 registry model resolution
- add RL, sentiment/news, anomaly detection, MLflow, or adaptive threshold tuning

## M10 Scope

M10 does:
- evaluate every authoritative M4 signal through an explicit pre-trade risk layer before any order request is created
- keep BUY approval bounded by cash, max open positions, per-asset exposure, total exposure, daily loss, drawdown, cooldown, loss-streak cooldown, minimum trade notional, and regime trade-allow rules
- keep SELL and HOLD behavior explicit and restart-safe
- persist additive `paper_risk_decisions` and `service_risk_state` rows with ordered adjustment steps, primary reason codes, and human-readable rationale text
- fail closed on blocked BUY decisions and keep rejection reasons inspectable

M10 does not do:
- replace M4 signal authority
- route execution directly to brokers
- add portfolio optimization, leverage, shorts, or adaptive risk tuning
- hide blocked or modified decisions behind implicit sizing behavior

## M11 Scope

M11 does:
- keep M4 authoritative for prediction and signal generation
- keep M10 authoritative for risk approval and sizing
- add deterministic idempotent order requests and lifecycle audit rows under `execution_order_requests` and `execution_order_events`
- support two local-only execution modes: `paper` and `shadow`
- keep the same risk-approved signal path in both modes
- record explicit `CREATED`, `SUBMITTED`, `ACCEPTED`, `PARTIALLY_FILLED`, `FILLED`, `REJECTED`, `CANCELED`, and `FAILED` lifecycle states where supported by the current adapter path
- keep paper mode functional while making every intended order traceable

M11 does not do:
- place live orders or add broker credentials
- add a live adapter, new backend service, scheduler, or control plane
- change M4 API contracts or move signal logic out of M4
- weaken accepted M1-M10 behavior to make execution abstraction easier

## M12 Scope

M12 does:
- add `live` as a third execution adapter mode inside the accepted M11 execution layer
- validate Alpaca account credentials with `GET /v2/account` using `APCA-API-KEY-ID` and `APCA-API-SECRET-KEY`
- require both checked-in config gating and runtime arming before live mode can start
- require `ALPACA_BASE_URL` to be the root domain only, then append paths like `/v2/account` and `/v2/orders` in code
- keep live order size tiny with `execution.live.max_order_notional`
- restrict live orders to the configured `execution.live.symbol_whitelist`
- check a local manual-disable sentinel before every live broker submission
- track consecutive live submit failures and activate a hard-stop when the configured threshold is reached
- keep live lifecycle audit rows explicit even when the broker path is non-terminal
- avoid mutating local live positions or ledger rows from submit success alone
- require broker-truth reconciliation before live state is considered clear enough to continue
- persist additive live safety state plus broker metadata on order lifecycle rows
- write a redacted startup checklist artifact and a current live status artifact
- surface live mode, live safety state, and recent live order audit rows in the existing dashboard

M12 does not do:
- auto-import broker fills into local portfolio state without explicit verified reconciliation
- add websocket streaming, advanced order types, or partial-fill management
- add a live trading control plane, scheduler, or background recovery daemon
- change M4 signal generation, M9 regime classification, or M10 risk sizing authority

## M13 Scope

M13 Packet 1 does:
- add checked-in `configs/reliability.yaml`
- add typed reliability config plus freshness, heartbeat, breaker, and recovery dataclasses
- add additive PostgreSQL tables `service_heartbeats`, `reliability_state`, and `reliability_events`
- add pure reliability helpers for freshness, breaker transitions, and pending-signal expiry

M13 Packet 2 does:
- write heartbeats from producer, features, inference, and the trading runner
- add exact-row `GET /freshness?symbol=...` from the M4 inference API
- degrade `GET /signal` to an explicit reliability `HOLD` when the exact canonical row is missing or stale
- keep regime freshness tied to exact-row compatibility and exact-row resolution, not artifact age alone
- add a trader-side inference breaker using the checked-in breaker primitives
- clear stale carried-over pending signals on runner startup and record deterministic recovery events
- write reliability artifacts under `artifacts/reliability/`
- surface compact reliability status and per-symbol freshness in the dashboard

M13 Packet 2 does not do:
- change M4 model authority, M10 risk authority, or M11/M12 execution routing
- add alert routing, deployment profiles, or a new orchestration stack
- add alert routing, recovery daemons, or explainability payload expansion

M13 Packet 3 does:
- track finalized feature time lag per symbol from `evaluated_at - latest_feature_as_of_time`
- track feature consumer processing lag per symbol from `latest_raw_event_received_at - latest_feature_as_of_time`
- persist lag snapshots under `reliability_lag_state` with explicit lag breach reason codes
- use producer heartbeat exchange-activity timestamps so feed staleness is visible in aggregate health
- add canonical cross-service aggregation across `producer`, `features`, `inference`, `trading_runner`, and `signal_client`
- persist canonical system snapshots under `reliability_system_state`
- add read-only `GET /reliability/system`
- write `artifacts/reliability/system_health.json` and `artifacts/reliability/lag_summary.json`
- gate guarded-live startup, runtime, and pre-submit paths on canonical system health and unresolved reconciliation
- surface one overall reliability status, per-service heartbeat health, lag breach state, and the latest recovery event in the existing dashboard

M13 Packet 3 does not do:
- change M4 prediction or signal generation authority
- change M10 risk approval or sizing authority
- change M11/M12 execution routing or live safety controls
- add alert routing, deployment profiles, explainability, RL, sentiment/news, or strategy rewrites

## M14 Scope

M14 Packet 1 does:
- keep M4 authoritative for prediction and signal generation
- add checked-in `configs/explainability.yaml`
- add shared explainability config, schemas, and service helpers under `app/explainability/`
- expose `model_version` and stable `model_version_source` on the loaded model metadata
- resolve a persisted reference vector from `artifacts/explainability/<model_version>/reference.json`
- build the reference vector deterministically from canonical `feature_ohlc` medians when the artifact is missing, then persist it
- compute top-feature contributions with one-at-a-time reference ablation against `prob_up`
- extend `GET /predict` with additive `model_version`, `top_features`, and `prediction_explanation`
- extend `GET /signal` with additive `model_version`, `top_features`, `prediction_explanation`, `threshold_snapshot`, `regime_reason`, and `signal_explanation`
- add explicit regime-reason codes `REGIME_HIGH_VOL`, `REGIME_TREND_UP`, `REGIME_TREND_DOWN`, and `REGIME_RANGE`

M14 Packet 1 does not do:
- add a new API service
- change M10 risk approval or sizing authority
- change M11/M12 execution routing, live safety, or audit behavior
- change M13 reliability HOLD behavior
- add SHAP, external explainability dependencies, threshold rewrites, or dashboard redesigns

M14 Packet 2 does:
- keep M4 authoritative by creating one canonical `decision_traces` row from the accepted `/signal` response
- persist the authoritative M4 prediction and signal explanation fields, `threshold_snapshot`, `regime_reason`, and `model_version` into the trace payload
- extend M10 outputs additively with `primary_reason_code`, `reason_texts`, ordered adjustment steps, and `blocked_stage="risk"` for blocked trades
- link `paper_risk_decisions.decision_trace_id`, `paper_risk_decisions.model_version`, `execution_order_requests.decision_trace_id`, `execution_order_requests.model_version`, and `paper_engine_state.pending_decision_trace_id`
- keep `decision_traces` as the canonical rationale holder and keep the existing tables on lightweight linkage fields only

M14 Packet 2 does not do:
- change M4 prediction or signal authority
- change M10 risk authority or sizing policy
- change M11/M12 execution routing behavior beyond carrying trace ids
- change M13 reliability behavior, dashboard layout, or report generation

M14 Packet 3 does:
- add `execution_order_events.decision_trace_id`, `paper_trade_ledger.decision_trace_id`, `paper_positions.entry_decision_trace_id`, and `paper_positions.exit_decision_trace_id`
- carry the same canonical trace id through created order requests, lifecycle events, entry fills, exit fills, ledger rows, and open/closed positions where an authoritative trace exists
- write one deterministic JSON report and one deterministic Markdown report per decision trace under `artifacts/rationale/`
- persist `json_report_path` and `markdown_report_path` back into `decision_traces`
- add a minimal dashboard table for recent decision traces, a latest blocked-trade rationale view, and local report download buttons without adding a new backend service

M14 Packet 3 does not do:
- change M4 signal authority, M10 risk authority, or M11/M12 execution routing authority
- add M15 operator-console redesign, alerting, deployment automation, or a new orchestration service
- duplicate the canonical rationale payload outside `decision_traces`

## M15 Scope

M15 does:
- keep the accepted M4, M10, M11, M12, M13, and M14 authority boundaries unchanged
- replace the old three-tab Streamlit layout with seven operator views: `Market`, `Signals`, `Trades`, `Risk`, `Health`, `Models`, and `Incidents`
- render one persistent top banner showing execution mode, safety posture, venue, environment, latest evaluation time, and the primary active blocking or degraded reason when applicable
- surface venue and environment truth near the banner, including validated account and environment fields when guarded-live state is available
- add freshness timestamps and compact age text across market, signals, risk, health, incidents, and recent trade or order activity tables
- add a filterable trade journal from canonical `decision_traces` and their linkage ids without inventing fill state
- aggregate current unsafe state from live safety, canonical reliability, freshness, breaker state, and latest blocked-trade truth into one incidents view
- add an operator-friendly config summary that shows active symbols, feature table, inference API URL, execution mode, venue and environment context, fee assumptions, major risk caps, live whitelist, and active model or regime references

M15 does not do:
- change strategy logic, M10 sizing policy, M11 execution routing, or M13 reliability semantics
- add alert routing, deployment automation, broker import, or a new orchestration layer
- pretend broker fills or reconciliation state exist when they do not
- add a new backend service or write dashboard state back into PostgreSQL

## M17 Scope

M17 does:
- keep M17 alerting local-first and inspectable by reading canonical PostgreSQL alert rows plus canonical JSON artifacts only
- expose read-only API endpoints: `GET /alerts/active`, `GET /alerts/timeline`, `GET /operations/daily-summary`, and `GET /operations/startup-safety`
- extend `GET /health` additively with `active_alert_count`, `max_alert_severity`, `startup_safety_status`, and `startup_safety_reason_code`
- keep active-alert and incident-timeline truth sourced from `operational_alert_state` and `operational_alert_events`
- keep startup-safety truth sourced from `artifacts/operations/startup_safety.json`
- keep daily-operations truth sourced from `artifacts/operations/daily/YYYY-MM-DD.json`
- surface active alerts, incident timeline, startup safety, and daily operations summaries in the existing operator console without adding write controls or a new backend service

M17 does not do:
- change M10 risk decisions, M11 execution routing, M12 guarded-live gating, M13 reliability logic, or M14 explainability payloads
- add ack, mute, resolve, or admin alert actions
- add alert routing, external paging, or a separate control plane

## M17 Operator Workflow

Canonical M17 artifact paths:

- `artifacts/operations/startup_safety.json`
- `artifacts/operations/daily/YYYY-MM-DD.json`

Read-only M17 API endpoints:

- `GET /alerts/active`
- `GET /alerts/timeline?limit=50&category=FEED_STALE&severity=WARNING&symbol=BTC/USD&active_only=true`
- `GET /operations/daily-summary?date=YYYY-MM-DD`
- `GET /operations/startup-safety`

Operator-console additions:

- `Health` view now includes `Active Alerts`, `Startup Safety`, and `Daily Operations Summary`
- `Incidents` view now includes `Incident Timeline`

Local validation commands:

```powershell
python -m pytest tests\test_inference_api.py -q
python -m pytest tests\test_dashboard_data_sources.py -q
python -m pytest -q
python -m pylint app\inference\main.py app\inference\schemas.py app\inference\service.py dashboards\data_sources.py dashboards\streamlit_app.py tests\test_inference_api.py tests\test_dashboard_data_sources.py
```

## Environment Variables

Copy `.env.example` to `.env` before running the stack.

| Variable | Purpose | Default |
| --- | --- | --- |
| `APP_NAME` | Logical app name embedded in events | `streamalpha` |
| `LOG_LEVEL` | Python log level | `INFO` |
| `KRAKEN_WS_URL` | Kraken public WebSocket v2 endpoint | `wss://ws.kraken.com/v2` |
| `KRAKEN_REST_OHLC_URL` | Kraken public REST OHLC endpoint used for explicit backfills | `https://api.kraken.com/0/public/OHLC` |
| `KRAKEN_SYMBOLS` | Comma-separated symbols | `BTC/USD,ETH/USD,SOL/USD` |
| `KRAKEN_OHLC_INTERVAL_MINUTES` | Candle interval in minutes | `5` |
| `KAFKA_BOOTSTRAP_SERVERS` | Redpanda bootstrap servers | `redpanda:9092` |
| `KAFKA_CLIENT_ID` | Kafka producer client id | `streamalpha-producer` |
| `POSTGRES_HOST` | PostgreSQL host | `postgres` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | Database name | `streamalpha` |
| `POSTGRES_USER` | Database user | `streamalpha` |
| `POSTGRES_PASSWORD` | Database password | `change-me-local-only` |
| `TOPIC_RAW_TRADES` | Trade topic name | `raw.trades` |
| `TOPIC_RAW_OHLC` | OHLC topic name | `raw.ohlc` |
| `TOPIC_RAW_HEALTH` | Health topic name | `raw.health` |
| `TABLE_RAW_TRADES` | Trade table name | `raw_trades` |
| `TABLE_RAW_OHLC` | Raw OHLC table name | `raw_ohlc` |
| `TABLE_FEATURE_OHLC` | Finalized OHLC feature table name | `feature_ohlc` |
| `TABLE_PRODUCER_HEARTBEAT` | Heartbeat table name | `producer_heartbeat` |
| `PRODUCER_SERVICE_NAME` | Producer service name | `producer` |
| `PRODUCER_HEARTBEAT_INTERVAL_SECONDS` | Producer heartbeat cadence | `15` |
| `FEATURE_CONSUMER_GROUP_ID` | Kafka consumer group id for M2 | `streamalpha-feature-consumer` |
| `FEATURE_SERVICE_NAME` | Feature service name used in logs and client id | `features` |
| `FEATURE_FINALIZATION_GRACE_SECONDS` | Grace period before stale open-candle finalization | `30` |
| `FEATURE_BOOTSTRAP_CANDLES` | Raw OHLC candles loaded per symbol on startup | `64` |
| `INFERENCE_MODEL_PATH` | Optional explicit `model.joblib` override; when empty M4 resolves the current promoted champion from the local registry | empty |
| `INFERENCE_SERVICE_NAME` | Inference API service name used in logs and responses | `inference` |
| `INFERENCE_SIGNAL_BUY_PROB_UP` | BUY threshold for `prob_up` | `0.55` |
| `INFERENCE_SIGNAL_SELL_PROB_UP` | SELL threshold for `prob_up` | `0.45` |
| `INFERENCE_REGIME_THRESHOLDS_PATH` | Optional explicit M8 `thresholds.json`; when empty M9 uses the latest run under `artifacts/regime/m8/` | empty |
| `INFERENCE_REGIME_SIGNAL_POLICY_PATH` | Optional explicit M9 regime signal policy path | `configs/regime_signal_policy.json` |
| `INFERENCE_API_BASE_URL` | Base URL for the accepted local M4 inference API | `http://127.0.0.1:8000` |
| `DASHBOARD_REFRESH_SECONDS` | Browser refresh cadence for the M6 Streamlit UI | `15` |
| `DASHBOARD_RECENT_TRADES_LIMIT` | Recent closed trades shown in the dashboard | `20` |
| `DASHBOARD_RECENT_LEDGER_LIMIT` | Recent ledger rows shown in the dashboard | `20` |
| `APCA_API_KEY_ID` | Alpaca Trading API key id used only by the M12 live adapter | empty |
| `APCA_API_SECRET_KEY` | Alpaca Trading API secret key used only by the M12 live adapter | empty |
| `ALPACA_BASE_URL` | Alpaca Trading API root URL only, for example `https://paper-api.alpaca.markets` | empty |
| `STREAMALPHA_ENABLE_LIVE` | Explicit runtime live arming switch; must be set to `true` for M12 live startup | empty |
| `STREAMALPHA_LIVE_CONFIRM` | Exact runtime confirmation phrase required for M12 live startup | empty |
| `RECONNECT_INITIAL_DELAY_SECONDS` | First reconnect delay | `1` |
| `RECONNECT_MAX_DELAY_SECONDS` | Reconnect delay cap | `30` |
| `RECONNECT_BACKOFF_MULTIPLIER` | Backoff multiplier | `2.0` |
| `RECONNECT_JITTER_SECONDS` | Random jitter added to backoff | `0.5` |

## Run Locally

### 1. Prepare the environment

```powershell
Copy-Item .env.example .env
```

### 2. Start the full local stack

```powershell
docker compose up --build -d
```

### 3. Follow the producer and feature consumer logs

```powershell
docker compose logs -f producer
docker compose logs -f features
```

### 4. Run only the feature consumer service after the shared stack is already up

```powershell
docker compose up --build -d redpanda postgres producer
docker compose up --build features
```

### 5. Run the feature consumer directly with Python

```powershell
python -m pip install -r requirements.txt
python -m app.features.main
```

### 6. Run the M3 offline training pipeline

```powershell
python -m pip install -r requirements.txt
python -m app.training --config configs/training.m3.json
```

### 7. Backfill recent Kraken REST OHLC history into the canonical raw and feature tables

```powershell
python -m app.ingestion.backfill_ohlc --symbols BTC/USD ETH/USD SOL/USD --interval 5
```

The backfill command upserts closed historical candles into `raw_ohlc` and then regenerates `feature_ohlc` directly from the existing raw/canonical flow without changing the M3 training source.

### 8. Run the WSL-based M3 smoke helper with bounded PostgreSQL readiness checks

```powershell
wsl -e bash /mnt/d/Github/Stream_Alpha/scripts/m3-smoke-run.sh
```

### 9. Run the M4 and M9 inference API from the repo

Set `INFERENCE_MODEL_PATH` to the accepted saved M3 artifact if you want a direct override. For example:

```powershell
$env:INFERENCE_MODEL_PATH = "D:\Github\Stream_Alpha\artifacts\training\m3\<run_id>\model.joblib"
$env:INFERENCE_REGIME_THRESHOLDS_PATH = "D:\Github\Stream_Alpha\artifacts\regime\m8\<run_id>\thresholds.json"
python -m app.inference
```

The API listens on `http://127.0.0.1:8000` by default.

When `INFERENCE_MODEL_PATH` is empty, M4 resolves the current champion from `artifacts/registry/current.json` instead.

When `INFERENCE_REGIME_THRESHOLDS_PATH` is empty, M9 resolves the latest saved `thresholds.json` under `artifacts/regime/m8/`.

### 10. Example inference requests

```powershell
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/latest-features?symbol=BTC/USD"
curl "http://127.0.0.1:8000/predict?symbol=BTC/USD"
curl "http://127.0.0.1:8000/regime?symbol=BTC/USD"
curl "http://127.0.0.1:8000/signal?symbol=BTC/USD"
curl "http://127.0.0.1:8000/freshness?symbol=BTC/USD"
curl "http://127.0.0.1:8000/reliability/system"
curl "http://127.0.0.1:8000/metrics"
```

Example `GET /predict` response:

```json
{
  "symbol": "BTC/USD",
  "model_name": "logistic_regression",
  "model_trained_at": "2026-03-19T22:30:02Z",
  "model_artifact_path": "D:/Github/Stream_Alpha/artifacts/training/m3/<run_id>/model.joblib",
  "row_id": "BTC/USD|2026-03-19T22:00:00Z",
  "interval_begin": "2026-03-19T22:00:00Z",
  "as_of_time": "2026-03-19T22:05:00Z",
  "prob_up": 0.61,
  "prob_down": 0.39,
  "predicted_class": "UP",
  "confidence": 0.61,
  "regime_label": "TREND_UP",
  "regime_run_id": "20260320T165813Z"
}
```

Example `GET /freshness` response:

```json
{
  "symbol": "BTC/USD",
  "row_id": "BTC/USD|2026-03-21T11:55:00Z",
  "interval_begin": "2026-03-21T11:55:00Z",
  "as_of_time": "2026-03-21T12:00:00Z",
  "health_overall_status": "HEALTHY",
  "freshness_status": "FRESH",
  "reason_code": "HEALTH_HEALTHY",
  "feature_freshness_status": "FRESH",
  "feature_reason_code": "FEATURE_FRESH",
  "feature_age_seconds": 0.0,
  "regime_freshness_status": "FRESH",
  "regime_reason_code": "REGIME_FRESH",
  "regime_age_seconds": 0.0,
  "detail": "Exact-row regime resolution succeeded"
}
```

### 11. Run the M5 paper trader once

Make sure PostgreSQL, canonical `feature_ohlc`, and the M4 inference API are already available first.

```powershell
python scripts\run_paper_trader.py --config configs\paper_trading.yaml --once
```

### 12. Run the M5 paper trader in polling mode

```powershell
python scripts\run_paper_trader.py --config configs\paper_trading.yaml
```

The paper trader is intentionally local-first and long-only. It consumes the existing M4 `/signal` endpoint as authoritative, including M9 regime-aware decisions, and never reimplements signal logic.

For M11, `configs/paper_trading.yaml` adds:

```yaml
execution:
  mode: paper
  idempotency_key_version: 1
```

Set `execution.mode: shadow` to run the same M4 and M10 path with isolated shadow rows and explicit order lifecycle audit events.

For M12 guarded live, extend the same `execution` block with:

```yaml
execution:
  mode: live
  idempotency_key_version: 1
  live:
    enabled: true
    expected_account_id: null
    expected_environment: paper
    symbol_whitelist:
      - BTC/USD
    max_order_notional: 25.0
    failure_hard_stop_threshold: 3
    manual_disable_path: artifacts/live/manual_disable.flag
    startup_checklist_path: artifacts/live/startup_checklist.json
    live_status_path: artifacts/live/live_status.json
```

M12 live startup still requires explicit runtime arming in addition to `mode: live`:

```powershell
$env:APCA_API_KEY_ID = "<alpaca key id>"
$env:APCA_API_SECRET_KEY = "<alpaca secret key>"
$env:ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
$env:STREAMALPHA_ENABLE_LIVE = "true"
$env:STREAMALPHA_LIVE_CONFIRM = "I UNDERSTAND STREAM ALPHA LIVE TRADING IS ENABLED"
python scripts\run_paper_trader.py --config configs\paper_trading.yaml --once
```

Keep `ALPACA_BASE_URL` at the root domain only. Do not include `/v2`; M12 appends `/v2/account` and `/v2/orders` internally.

To disable live broker submission immediately without editing config, create the configured sentinel file:

```powershell
New-Item -ItemType File artifacts\live\manual_disable.flag -Force
```

M12 writes:
- `artifacts/live/startup_checklist.json`
- `artifacts/live/live_status.json`

The live status artifact and `execution_live_safety_state` persist one authoritative operator-facing live submit gate from existing runtime truth:
- `can_submit_live_now`
- `primary_block_reason_code`
- `block_detail`

M13 writes:
- `artifacts/reliability/health_snapshot.json`
- `artifacts/reliability/freshness_summary.json`
- `artifacts/reliability/recovery_events.jsonl`
- `artifacts/reliability/system_health.json`
- `artifacts/reliability/lag_summary.json`

M14 writes:
- `artifacts/explainability/<model_version>/reference.json`

The operator console shows the configured execution mode, a persistent safety banner, a compact live critical-state strip, venue and environment truth, live safety state, canonical M13 reliability state, recent decision traces, rationale report download paths, a filterable trade journal, and a current incidents view. This remains a guarded live foundation; alert routing, broker import, and automatic recovery orchestration are still intentionally deferred.

### 13. Run the M6 Streamlit dashboard

Make sure PostgreSQL is reachable and the M4 inference API is running first.

```powershell
$env:POSTGRES_HOST = "127.0.0.1"
$env:INFERENCE_API_BASE_URL = "http://127.0.0.1:8000"
streamlit run dashboards/streamlit_app.py
```

The dashboard defaults to [http://localhost:8501](http://localhost:8501).

### 14. Example dashboard validation requests

```powershell
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/signal?symbol=BTC/USD"
curl "http://127.0.0.1:8000/reliability/system"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS feature_rows FROM feature_ohlc;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS open_positions FROM paper_positions WHERE status = 'OPEN';"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS ledger_rows FROM paper_trade_ledger;"
```

Example `GET /signal` response:

```json
{
  "symbol": "BTC/USD",
  "signal": "BUY",
  "reason": "prob_up 0.6100 >= buy threshold 0.54",
  "prob_up": 0.61,
  "prob_down": 0.39,
  "confidence": 0.61,
  "predicted_class": "UP",
  "thresholds": {
    "buy_prob_up": 0.54,
    "sell_prob_up": 0.44
  },
  "row_id": "BTC/USD|2026-03-19T22:00:00Z",
  "as_of_time": "2026-03-19T22:05:00Z",
  "model_name": "logistic_regression",
  "regime_label": "TREND_UP",
  "regime_run_id": "20260320T165813Z",
  "trade_allowed": true,
  "signal_status": "MODEL_SIGNAL",
  "decision_source": "model",
  "reason_code": "HEALTH_HEALTHY",
  "freshness_status": "FRESH",
  "health_overall_status": "HEALTHY"
}
```

Example `GET /regime` response:

```json
{
  "symbol": "BTC/USD",
  "row_id": "BTC/USD|2026-03-19T22:00:00Z",
  "interval_begin": "2026-03-19T22:00:00Z",
  "as_of_time": "2026-03-19T22:05:00Z",
  "regime_label": "TREND_UP",
  "regime_run_id": "20260320T165813Z",
  "regime_artifact_path": "D:/Github/Stream_Alpha/artifacts/regime/m8/20260320T165813Z/thresholds.json",
  "realized_vol_12": 0.03,
  "momentum_3": 0.03,
  "macd_line_12_26": 1.2,
  "high_vol_threshold": 0.05,
  "trend_abs_threshold": 0.02,
  "trade_allowed": true,
  "buy_prob_up": 0.54,
  "sell_prob_up": 0.44
}
```

### 15. Run the M7 retraining workflow

Run one challenger retraining job from the canonical source:

```powershell
python -m app.training.retrain --config configs/training.m7.json
```

This writes one timestamped challenger run under `artifacts/training/m7/<run_id>/`, including `comparison_vs_champion.json` and `run_manifest.json`.

Compare an existing challenger explicitly if needed:

```powershell
python -m app.training.compare --config configs/training.m7.json --run-dir artifacts/training/m7/<run_id>
```

Promote a passing challenger into the immutable local registry:

```powershell
python -m app.training.promote --run-dir artifacts/training/m7/<run_id> --model-version m7-<run_id>
```

Rollback to one previously promoted version without retraining:

```powershell
python -m app.training.rollback --model-version <model_version>
```

Bootstrap the registry from the current accepted artifact when you want registry-based inference before the first M7 challenger promotion:

```powershell
python -m app.training.promote --run-dir artifacts/training/m3/<run_id> --model-version m3-<run_id>
```

### 16. Run the M8 offline regime workflow

```powershell
python -m app.regime --config configs/regime.m8.json
```

Each run writes:

```text
artifacts/regime/m8/<run_id>/
|-- by_symbol_summary.csv
|-- overall_summary.json
|-- regime_predictions.csv
|-- run_config.json
|-- run_manifest.json
`-- thresholds.json
```

## Tests And Lint

```powershell
python -m pytest
python -m pylint app dashboards tests scripts\run_paper_trader.py
```

## Training Artifacts

Each M3 run writes a timestamped artifact directory:

```text
artifacts/training/m3/<run_id>/
|-- dataset_manifest.json
|-- feature_columns.json
|-- fold_metrics.csv
|-- model.joblib
|-- oof_predictions.csv
|-- run_config.json
`-- summary.json
```

Quick checks after a run:

```powershell
Get-ChildItem artifacts\training\m3
Get-Content artifacts\training\m3\<run_id>\summary.json
Get-Content artifacts\training\m3\<run_id>\dataset_manifest.json
```

## M7 Training Artifacts

Each M7 challenger run writes:

```text
artifacts/training/m7/<run_id>/
|-- comparison_vs_champion.json
|-- dataset_manifest.json
|-- feature_columns.json
|-- fold_metrics.csv
|-- model.joblib
|-- oof_predictions.csv
|-- run_config.json
|-- run_manifest.json
`-- summary.json
```

The immutable local registry lives under:

```text
artifacts/registry/
|-- current.json
|-- history.jsonl
`-- models
    `-- <model_version>/
        |-- comparison_vs_champion.json
        |-- dataset_manifest.json
        |-- feature_columns.json
        |-- fold_metrics.csv
        |-- model.joblib
        |-- oof_predictions.csv
        |-- registry_entry.json
        |-- run_config.json
        |-- run_manifest.json
        `-- summary.json
```

Quick checks after promotion:

```powershell
Get-Content artifacts\registry\current.json
Get-Content artifacts\registry\history.jsonl
Get-ChildItem artifacts\registry\models
```

## Paper-Trading Artifacts

Each M5 run updates a rolling artifact directory:

```text
artifacts/paper_trading/
|-- by_asset_summary.csv
|-- by_regime_summary.csv
|-- closed_positions.csv
|-- latest_summary.json
`-- open_positions.csv
```

Quick checks after a run:

```powershell
Get-Content artifacts\paper_trading\latest_summary.json
Get-Content artifacts\paper_trading\by_asset_summary.csv
Get-Content artifacts\paper_trading\by_regime_summary.csv
Get-Content artifacts\paper_trading\open_positions.csv
Get-Content artifacts\paper_trading\closed_positions.csv
```

For M11 execution-audit checks:

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT execution_mode, COUNT(*) AS order_requests FROM execution_order_requests GROUP BY execution_mode ORDER BY execution_mode;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT execution_mode, lifecycle_state, COUNT(*) AS event_rows FROM execution_order_events GROUP BY execution_mode, lifecycle_state ORDER BY execution_mode, lifecycle_state;"
```

For M12 guarded-live checks:

```powershell
Get-Content artifacts\live\startup_checklist.json
Get-Content artifacts\live\live_status.json
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT service_name, execution_mode, broker_name, startup_checks_passed, can_submit_live_now, primary_block_reason_code, manual_disable_active, failure_hard_stop_active, reconciliation_status, health_gate_status FROM execution_live_safety_state;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT execution_mode, lifecycle_state, broker_name, external_status, COUNT(*) AS event_rows FROM execution_order_events GROUP BY execution_mode, lifecycle_state, broker_name, external_status ORDER BY execution_mode, lifecycle_state, broker_name, external_status;"
```

## M13 Reliability Validation

1. Verify stale input downgrade to `HOLD`.
Use an exact `interval_begin` whose canonical row is older than `freshness.feature_max_age_seconds`, then confirm the inference API refuses to return a model-driven decision for that row.

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT symbol, interval_begin, as_of_time FROM feature_ohlc ORDER BY as_of_time ASC LIMIT 5;"
curl "http://127.0.0.1:8000/signal?symbol=BTC/USD&interval_begin=<older interval_begin>"
```

Expect `signal=HOLD`, `decision_source=reliability`, and `reason_code=RELIABILITY_HOLD_STALE_FEATURE_ROW` or `RELIABILITY_HOLD_MISSING_FEATURE_ROW`.

2. Verify stale pending-signal cleanup on restart.
Seed one stale `pending_signal_action` in `paper_engine_state`, restart the runner once, then confirm the carried signal was cleared and audited as a recovery action.

```powershell
python scripts\run_paper_trader.py --config configs\paper_trading.yaml --once
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT symbol, pending_signal_action, pending_signal_interval_begin FROM paper_engine_state ORDER BY symbol;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT component_name, event_type, reason_code, event_time FROM reliability_events WHERE reason_code = 'RECOVERY_STALE_PENDING_SIGNAL_CLEARED' ORDER BY event_time DESC LIMIT 10;"
```

3. Verify signal-client breaker transitions.
Temporarily make the inference API unreachable for the runner, run the trader until signal fetches fail, then restore the API and run again so the breaker can recover.

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT component_name, breaker_state, health_overall_status, reason_code, updated_at FROM reliability_state WHERE component_name = 'signal_client';"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT component_name, event_type, reason_code, event_time FROM reliability_events WHERE component_name = 'signal_client' ORDER BY event_time DESC LIMIT 10;"
```

Expect explicit `OPEN`, `HALF_OPEN`, and `CLOSED` transitions plus persisted transition events.

4. Verify heartbeat persistence.
All core services should continue writing additive heartbeats without changing their authority boundaries.

```powershell
Get-Content configs\reliability.yaml
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT service_name, component_name, heartbeat_at, health_overall_status, reason_code FROM service_heartbeats ORDER BY heartbeat_at DESC LIMIT 20;"
Get-Content artifacts\reliability\health_snapshot.json
Get-Content artifacts\reliability\freshness_summary.json
```

5. Verify lag breach visibility.
Keep producer ingesting while the feature consumer is stopped for longer than the configured lag thresholds, then inspect the persisted lag state, lag artifact, and transition events.

```powershell
Get-Content configs\reliability.yaml
Get-Content artifacts\reliability\lag_summary.json
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT symbol, health_overall_status, reason_code, time_lag_seconds, processing_lag_seconds, evaluated_at FROM reliability_lag_state ORDER BY symbol ASC;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT component_name, event_type, reason_code, event_time FROM reliability_events WHERE event_type = 'FEATURE_LAG_TRANSITION' ORDER BY event_time DESC LIMIT 10;"
```

Expect `lag_breach_active=true` and explicit lag reason codes such as `FEATURE_LAG_BREACH`, `FEATURE_TIME_LAG_BREACH`, and `FEATURE_PROCESSING_LAG_BREACH`.

6. Verify the canonical reliability summary.
The API, persisted snapshot, and dashboard should agree on one overall operator-visible view of producer, features, inference, runner, lag, feed freshness, and the latest recovery event.

```powershell
curl "http://127.0.0.1:8000/reliability/system"
Get-Content artifacts\reliability\system_health.json
Get-Content artifacts\reliability\recovery_events.jsonl
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT service_name, checked_at, health_overall_status, reason_codes, lag_breach_active FROM reliability_system_state ORDER BY checked_at DESC LIMIT 10;"
```

Expect one explicit overall status in `HEALTHY`, `DEGRADED`, or `UNAVAILABLE`, explicit aggregate reason codes, producer feed-freshness visibility, per-service heartbeat status, lag breach state, and the latest recovery event. The dashboard should show the same system-level summary in `Reliability Status`, `Per-Service Health`, and `Feature Consumer Lag`.

7. Verify live gating fails closed on degraded health or unresolved reconciliation.
Run guarded live with valid credentials, then either force `/reliability/system` away from `HEALTHY` or create a broker/local mismatch and confirm the runner suspends live submission until the gate clears.

```powershell
python scripts\run_paper_trader.py --config configs\paper_trading.yaml --once
Get-Content artifacts\live\live_status.json
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT service_name, reconciliation_status, reconciliation_reason_code, health_gate_status, health_gate_reason_code, unresolved_incident_count FROM execution_live_safety_state;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT component_name, event_type, reason_code, detail, event_time FROM reliability_events WHERE component_name IN ('live_reconciliation', 'live_health_gate') ORDER BY event_time DESC LIMIT 20;"
```

Expect `health_gate_status` and `reconciliation_status` to fail closed whenever canonical health is not `HEALTHY`, a lag breach is active, signal freshness is stale, or broker truth does not match local state. Expect `can_submit_live_now=false` plus one explicit `primary_block_reason_code` and `block_detail` whenever live submit is blocked.

## M14 Explainability Validation

1. Verify additive explainability on `/predict`.
Call `GET /predict` for a live symbol and confirm the response now includes `model_version`, `top_features`, and `prediction_explanation` without changing the existing prediction contract.

```powershell
Get-Content configs\explainability.yaml
curl "http://127.0.0.1:8000/predict?symbol=BTC/USD"
Get-ChildItem artifacts\explainability
Get-Content artifacts\explainability\<model_version>\reference.json
```

Expect `prediction_explanation.method=ONE_AT_A_TIME_REFERENCE_ABLATION`, a persisted `reference.json`, and `top_features` sorted by absolute `signed_contribution` descending.

2. Verify additive explainability on `/signal`.
Call `GET /signal` for a fresh symbol and confirm the response now includes `model_version`, `top_features`, `prediction_explanation`, `threshold_snapshot`, `regime_reason`, and `signal_explanation`.

```powershell
curl "http://127.0.0.1:8000/signal?symbol=BTC/USD"
curl "http://127.0.0.1:8000/regime?symbol=BTC/USD"
```

Expect `threshold_snapshot` to show the active regime thresholds, `regime_reason.reason_code` to be one of `REGIME_HIGH_VOL`, `REGIME_TREND_UP`, `REGIME_TREND_DOWN`, or `REGIME_RANGE`, and `signal_explanation.decision_source=model`.

3. Verify the M13 reliability HOLD path still wins when inputs are stale or missing.
Use an older exact candle or a missing exact candle and confirm the response is still an explicit reliability `HOLD` while the additive M14 fields remain present.

```powershell
curl "http://127.0.0.1:8000/signal?symbol=BTC/USD&interval_begin=<older interval_begin>"
```

Expect `signal=HOLD`, `decision_source=reliability`, the existing M13 `reason_code`, and additive `threshold_snapshot` plus `signal_explanation` fields without any change to the underlying HOLD behavior.

4. Verify canonical decision-trace persistence after one paper-trader pass.
Run the accepted paper trader once, then confirm the latest M4 signal wrote a `decision_traces` row and that the existing M10 and M11 tables now link back to it.

```powershell
python scripts\run_paper_trader.py --once
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT id, symbol, signal, model_version, risk_outcome, created_at FROM decision_traces ORDER BY id DESC LIMIT 5;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT id, decision_trace_id, model_version, outcome, reason_codes FROM paper_risk_decisions ORDER BY id DESC LIMIT 5;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT id, decision_trace_id, model_version, action, risk_outcome FROM execution_order_requests ORDER BY id DESC LIMIT 5;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT service_name, symbol, pending_decision_trace_id FROM paper_engine_state ORDER BY symbol;"
```

Expect one `decision_traces` row per processed signal, a populated `decision_trace_id` on the matching `paper_risk_decisions` row, the same `decision_trace_id` on any created `execution_order_requests` row, and `pending_decision_trace_id` to remain restart-safe while a signal is waiting to fill.

5. Verify blocked trades write an explicit risk rationale.
Use a setup that blocks a new BUY, then inspect the latest trace payload and confirm the canonical blocked-trade section is present with `blocked_stage="risk"`.

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT id, trace_payload->'risk' AS risk_section, trace_payload->'blocked_trade' AS blocked_trade FROM decision_traces ORDER BY id DESC LIMIT 3;"
```

Expect blocked traces to show `risk.primary_reason_code`, ordered `reason_codes`, explicit `reason_texts`, and `blocked_trade.blocked_stage = "risk"`.

6. Verify modified trades preserve ordered adjustment steps.
Use a setup that produces a `MODIFIED` BUY and confirm the trace shows the ordered adjustment path rather than only a flat list of reason codes.

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT id, trace_payload->'risk'->'ordered_adjustments' AS ordered_adjustments FROM decision_traces WHERE risk_outcome = 'MODIFIED' ORDER BY id DESC LIMIT 3;"
```

Expect `ordered_adjustments` to remain in the exact sequence applied by M10, with each step showing `reason_code`, `reason_text`, `before_notional`, and `after_notional`.

7. Verify execution-facing trace linkage and rationale report generation.
Run the trader through at least one created order request and one filled action, then confirm the same canonical trace id is visible on lifecycle rows, ledger rows, positions, and the persisted report paths.

```powershell
python scripts\run_paper_trader.py --once
Get-ChildItem artifacts\rationale -Recurse
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT id, signal_row_id, json_report_path, markdown_report_path FROM decision_traces ORDER BY id DESC LIMIT 10;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT order_request_id, lifecycle_state, decision_trace_id, reason_code FROM execution_order_events ORDER BY id DESC LIMIT 10;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT id, action, order_request_id, decision_trace_id, signal_row_id FROM paper_trade_ledger ORDER BY id DESC LIMIT 10;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT id, status, entry_decision_trace_id, exit_decision_trace_id, entry_signal_row_id, exit_signal_row_id FROM paper_positions ORDER BY id DESC LIMIT 10;"
```

Expect deterministic `artifacts/rationale/<service_name>/<execution_mode>/<decision_trace_id>.json` and `.md` files, persisted report paths on `decision_traces`, and matching `decision_trace_id` values across order events, ledger rows, and the related position entry or exit fields.

8. Verify minimal dashboard visibility for decision traces.
Open the existing dashboard and confirm the trading tab now shows recent decision traces, the latest blocked-trade rationale, and local report download controls without any broader console redesign.

```powershell
streamlit run dashboards\streamlit_app.py
```

Expect the persistent top banner, visible venue or environment summary, seven named views, a trade journal with filters, an incidents panel with explicit empty-safe or active-incident output, `Recent Decision Traces`, `Latest Blocked Trade Rationale`, and `Rationale Reports`. The selected report should show both stored paths and offer JSON plus Markdown downloads when the files exist.

## Inspect Topics

Open Redpanda Console at [http://localhost:8080](http://localhost:8080).

You can also inspect the raw topics directly:

```powershell
docker exec -it streamalpha-redpanda rpk topic list -X brokers=redpanda:9092
docker exec -it streamalpha-redpanda rpk topic consume raw.ohlc -n 5 --offset start -X brokers=redpanda:9092
docker exec -it streamalpha-redpanda rpk topic consume raw.health -n 5 --offset start -X brokers=redpanda:9092
```

## Validate Feature Rows In PostgreSQL

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS feature_rows FROM feature_ohlc;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT symbol, interval_begin, as_of_time, close_price, log_return_1, rsi_14, macd_line_12_26 FROM feature_ohlc ORDER BY interval_begin DESC LIMIT 10;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT symbol, COUNT(*) AS rows_per_symbol FROM feature_ohlc GROUP BY symbol ORDER BY symbol;"
```

You can still validate the raw OHLC source table if needed:

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS raw_ohlc_rows FROM raw_ohlc;"
```

## Validate Paper-Trading Rows In PostgreSQL

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS engine_state_rows FROM paper_engine_state;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS open_positions FROM paper_positions WHERE status = 'OPEN';"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS closed_positions FROM paper_positions WHERE status = 'CLOSED';"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c \"SELECT symbol, action, reason, fill_interval_begin, fill_price, quantity, fee FROM paper_trade_ledger ORDER BY fill_time DESC LIMIT 10;\"
```

## Warmup And Finalization Rules

- Feature rows are emitted only for finalized candles.
- A newer candle closes the previous candle but is not used in the previous candle's feature values.
- A stale current candle is finalized only when `now >= interval_end + FEATURE_FINALIZATION_GRACE_SECONDS`.
- The consumer bootstraps from the latest `FEATURE_BOOTSTRAP_CANDLES` raw OHLC rows per symbol before starting Kafka consumption.
- The effective bootstrap minimum is clamped to the feature-engine minimum warmup length so restart behavior remains safe.
- No row is emitted until all selected features are available. With the current feature set and standard MACD seeding, that means at least `26` finalized candles for a symbol stream.

## Known Limitations

- The feature consumer is OHLC-only in M2 and intentionally ignores `raw.trades`.
- Gaps are not forward-filled; features are computed only from the finalized candles that actually exist.
- Rolling standard deviation and z-score calculations use population standard deviation over the fixed 12-candle window.
- Bootstrap backfills the most recent feature rows idempotently through PostgreSQL upserts rather than relying on historical Kafka replay.
- M3 training drops the first 3 feature rows per symbol from the eligible labeled dataset so the official persistence baseline can be computed from the canonical source without pulling a second table.
- M3 training will stop with a clear error if `feature_ohlc` does not yet contain enough eligible labeled rows for the configured walk-forward split. With the current checked-in split config, that means at least `9` unique eligible timestamps.
- M5 fills signal-driven entries and exits at the next finalized candle open because only finalized canonical candles are processed.
- M5 uses the existing M4 inference service as authoritative and does not re-score features locally.
- M5 is long-only spot simulation with simple next-open fills plus barrier exits; it is intentionally not a live broker or advanced execution model.
- M7 promotion is fully file-based and explicit; there is no scheduler daemon or automatic retraining loop.
- M7 artifacts now treat long-only after-cost economics as the promotion contract; a model can win classification tie-breaks and still be explicitly non-acceptable if the after-cost proxy stays non-positive.
- M7 rollback only switches the promoted champion pointer and does not mutate old promoted snapshots.
- M9 resolves regimes from the latest saved M8 thresholds artifact by default; there is no online threshold fitting or adaptive retuning.
- M12 guarded live now fails closed on broker-truth mismatch and keeps local portfolio mutations out of the submit path. It does not yet auto-import broker positions or fills into local trading state; unresolved broker truth still blocks until an operator resolves it.
- M13 Packet 3 now gates live startup and live submit on canonical system health, lag, freshness, signal health, and reconciliation status. It still does not add alert routing or automatic recovery orchestration.
- This is still a single-broker local stack for development, not a highly available deployment.

## M12 Guarded Live Validation

Use Alpaca PAPER first.

1. Set the required env vars:

```powershell
$env:APCA_API_KEY_ID = "<alpaca paper key id>"
$env:APCA_API_SECRET_KEY = "<alpaca paper secret key>"
$env:ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
$env:STREAMALPHA_ENABLE_LIVE = "true"
$env:STREAMALPHA_LIVE_CONFIRM = "I UNDERSTAND STREAM ALPHA LIVE TRADING IS ENABLED"
```

2. Keep `execution.mode: live`, `execution.live.enabled: true`, and a tiny `max_order_notional` in `configs/paper_trading.yaml`.

3. Run the trader once:

```powershell
python scripts\run_paper_trader.py --config configs\paper_trading.yaml --once
```

4. Inspect the artifacts and audit rows:

```powershell
Get-Content artifacts\live\startup_checklist.json
Get-Content artifacts\live\live_status.json
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT * FROM execution_live_safety_state;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT execution_mode, lifecycle_state, reason_code, broker_name, external_status, account_id FROM execution_order_events ORDER BY id DESC LIMIT 10;"
```

Expect `artifacts\live\live_status.json` to show `research_source_exchange=kraken`, `execution_broker_name=alpaca`, explicit cross-venue context, explicit reconciliation and health-gate status, and the live submit contract explaining that local portfolio mutations require broker truth or verified reconciliation.

## References

- Kraken public WebSocket v2 trades: [docs.kraken.com/api/docs/websocket-v2/trade/](https://docs.kraken.com/api/docs/websocket-v2/trade/)
- Kraken public WebSocket v2 OHLC: [docs.kraken.com/api/docs/websocket-v2/ohlc/](https://docs.kraken.com/api/docs/websocket-v2/ohlc/)
- Kraken public WebSocket v2 status and heartbeat: [docs.kraken.com/api/docs/websocket-v2/status/](https://docs.kraken.com/api/docs/websocket-v2/status/) and [docs.kraken.com/api/docs/websocket-v2/heartbeat/](https://docs.kraken.com/api/docs/websocket-v2/heartbeat/)
- Redpanda single-broker Docker example: [docs.redpanda.com/current/get-started/quick-start/](https://docs.redpanda.com/current/get-started/quick-start/)
