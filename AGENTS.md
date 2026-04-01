# AGENTS.md

## Working mode
- Use /plan before editing on complex tasks.
- Do not restart broad audits if PLANS.md already freezes findings.
- Prefer targeted checks over repo-wide rescans.
- Do not repeat DB count queries unless an edit directly affects them.
- Do not reread plan docs unless a specific discrepancy requires one fresh check.

## Frozen findings
- Treat PLANS.md "Frozen Findings" as established unless a specific code edit requires a targeted re-check.

## Patch discipline
- Work in small batches.
- After each batch, update PLANS.md.
- After each batch, run only targeted validation for touched files and touched runtime paths.
- Report blockers separately from completed work.

## Truth rules
- Code hooks, schemas, comments, research targets, or metadata do not count as trained model presence.
- A model counts as present only if there is real training or import support, real artifact output, registry/runtime discoverability, and actual runtime usability.
- Do not invent profiles, promotions, experiments, or specialists.

## Definition of done
A batch is done only when:
- code changes are written
- targeted tests/checks are run
- changed files are listed
- blockers are listed
- PLANS.md is updated