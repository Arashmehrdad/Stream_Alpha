# Release and Maintenance

## Update Docs After Code Changes

When runtime behavior changes, update the matching docs in the same batch:

- Service, profile, or Compose changes: `docs/docker.md`, `docs/services.md`, `docs/operations-runbook.md`.
- Environment variable changes: `docs/configuration.md`.
- API route or schema changes: `docs/api.md`.
- Model loading or registry changes: `docs/model-runtime.md`, `docs/training.md`.
- Data flow changes: `docs/data-pipeline.md`.

## Dependency Update Checklist

1. Decide whether the package is runtime or training only.
2. Update `requirements-runtime.txt` only for runtime service needs.
3. Update `requirements-training.txt` for training and research needs.
4. Keep `requirements.txt` as the local compatibility entrypoint unless the project changes that policy.
5. Run dry-run dependency checks.
6. Rebuild Docker app image if runtime dependencies changed.
7. Run inference model loader validation.

## Docker Image Size Check

```powershell
docker image ls -a `
  --format "table {{.Repository}}`t{{.Tag}}`t{{.ID}}`t{{.Size}}"

docker system df -v
```

Record meaningful size changes in `PLANS.md`.

## Runtime Validation Checklist

- `docker compose --profile paper --env-file .env config --services`
- `docker compose --profile paper --env-file .env build config-check inference`
- `docker compose --profile paper --env-file .env up -d`
- `docker compose --profile paper --env-file .env ps`
- `GET http://127.0.0.1:8000/health`
- Focused inference/runtime tests.

## What To Record In `PLANS.md`

For each batch, record:

- Scope.
- Changed files.
- Completed work.
- Validation commands and results.
- Blockers.
- Any truth caveats, especially around model promotion or runtime claims.

Do not record model-strength claims unless backed by runtime artifacts and validation.

