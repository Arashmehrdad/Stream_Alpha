# AGENTS.md

## Project identity

Stream Alpha is a crypto trading research and operations platform.

The project is not trying to find one universal model. It is trying to discover conditional edge:

- which model works
- under which market context
- for which setup or strategy family
- with what evidence
- and when it should be ignored

A model can be globally weak but conditionally useful. Specialists should only be used where evidence says they work. HOLD elsewhere.

## Working mode

- Audit current repo state before edits:
  - git status
  - recent commits if context is unclear
  - relevant files and artifacts
- Use targeted checks, not broad rescans, unless the task requires a broad audit.
- Treat PLANS.md, latest commits, and existing artifacts as source of truth.
- Do not restart broad audits if PLANS.md already freezes findings.
- Prefer small, reversible batches.
- After each approved batch, update PLANS.md if the task changes research state, project direction, or validation rules.
- Report blockers separately from completed work.

## Core engineering rule

No one-off model work for collective problems.

If a task applies to many models, strategies, labels, policies, slices, or artifacts, build a reusable evaluator/tool once.

Wrong:

- Build a PatchTST-only evaluator.
- Build an NHITS-only evaluator.
- Build a MACD-only, RSI-only, breakout-only, or other one-family evaluator.
- Build separate one-off evaluators for range, volatility, volume, labels, slices, policies, or artifacts.

Only make model-specific code when repairing a truly model-specific export or format issue.

Correct:

- Build one specialist edge evaluator that accepts any prediction file.
- Build one strategy-family evaluator that accepts any strategy signal file.
- Build one policy evaluator that accepts any model/strategy candidate.

## M20 research rules

M20 is research-only unless explicitly promoted through evidence and safety gates.

Required research flags where applicable:

- RESEARCH_ONLY
- NO_RUNTIME_EFFECT
- NOT_BACKTEST
- NOT_RUNTIME_READY
- NOT_PROMOTABLE
- NO_PROFIT_CLAIM

Do not claim profitability from:

- labels
- top-k lift
- diagnostic precision
- proxy metrics
- slice evidence
- backtest-like research artifacts
- model confidence
- raw accuracy

If economics are unavailable, say so clearly:

- NET_PROXY_NOT_AVAILABLE
- ECONOMIC_POLICY_EVALUATION_REQUIRED

## Current M20 state

PatchTST confirmation export and analysis exist.

Current research interpretation:

- PatchTST: CONFIRMED_SELECTIVE_RANK_SLICE_RESEARCH_CANDIDATE
- NHITS: SECONDARY_WATCHLIST_OR_WEAKER_CANDIDATE
- Overall: RESEARCH_ONLY_NOT_PROMOTABLE
- Runtime: unchanged
- Promotion: unchanged
- Trading behavior: unchanged
- Profitability: no claim

The next architectural direction is generic evaluator tooling, not model-by-model experimentation.

## Forbidden changes unless explicitly requested

Do not change:

- runtime inference
- registry authority
- promotion logic
- paper/live execution
- trading behavior
- backtest behavior
- model training
- score-only export logic
- label generation
- validation workflow
- profit-claim behavior
- secrets or account configuration

Do not run:

- training
- score-only inference
- long jobs
- live/paper trading commands

unless Arash explicitly approves.

## Artifact discipline

Research tools should write deterministic outputs under:

<run-dir>/research_labels/vol_scaled/<tool_name>/

Prefer these artifacts:

- manifest.json
- report.json
- report.md
- candidate_decisions.csv
- model_metrics.csv
- topk_policy_metrics.csv
- threshold_policy_metrics.csv
- by_symbol.csv
- by_time.csv
- next_actions.csv
- recommendation.json

Do not commit generated research artifacts unless explicitly requested.

Do not commit caches, temporary files, notebook outputs, or accidental line-ending-only changes unless the task is specifically about them.

## Validation discipline

Run only validation relevant to touched files and touched paths.

Common validation:

- python -m pytest <targeted-tests> -q
- python -m py_compile <changed-python-files>
- python -m pylint --fail-under=10 <changed-python-files-or-test-slice>
- git diff --check

If the workflow requires pylint 10/10, keep the relevant slice at 10.00/10.

Do not weaken pylint config or workflow strictness unless explicitly requested.

## Commit discipline

Do not commit or push until Arash approves.

After approval:

- commit only approved files
- exclude generated artifacts unless explicitly requested
- report commit hash, pushed branch, final git status, and validation summary

## Truth rules

Code hooks, schemas, comments, research targets, or metadata do not count as trained model presence.

A model counts as present only if there is:

- real training or import support
- real artifact output
- registry/runtime discoverability, if runtime use is claimed
- actual runtime usability, if runtime use is claimed

Do not invent:

- profiles
- promotions
- experiments
- specialists
- model availability
- economic proof
- profitability
- runtime readiness

## Definition of done

A batch is done only when:

- code changes are written
- targeted tests/checks are run
- changed files are listed
- blockers are listed
- research status is honest
- PLANS.md is updated when project state changes
- Arash approves commit/push
