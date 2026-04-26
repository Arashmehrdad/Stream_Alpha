# Operations Runbook

## Start Stack

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  up -d --build
```

Or:

```powershell
.\scripts\start-stack.ps1 `
  -Profile paper
```

## Stop Stack Preserving Volumes

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  down --remove-orphans
```

Do not add `--volumes` unless intentionally deleting local data.

## Rebuild Services

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  build config-check inference dashboard trader producer features

docker compose `
  --profile paper `
  --env-file .env `
  up -d --force-recreate
```

## Check Health

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  ps

(Invoke-WebRequest `
  -UseBasicParsing `
  http://127.0.0.1:8000/health).Content
```

## Check Logs

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  logs -f config-check inference features producer trader dashboard
```

## Check Topics

```powershell
.\scripts\check-topics.ps1
```

## Check Database

```powershell
.\scripts\check-db.ps1
```

## Recover From Failed Build

1. Confirm Docker Desktop is running.
2. Rebuild the shared app image:

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  build config-check inference dashboard trader producer features
```

3. If dependency install fails, check `requirements-runtime.txt`.
4. If cache corruption is suspected, prune builder cache:

```powershell
docker builder prune -af
```

## Recover From Missing Dependency

1. Confirm whether the missing package is required at runtime or only for training.
2. Add runtime-only packages to `requirements-runtime.txt` only when a running service or current champion artifact needs them.
3. Add training-only packages to `requirements-training.txt`.
4. Rebuild app services and run the inference model loader test.

## Safe Docker Cleanup

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  down --remove-orphans

docker builder prune -af

docker image prune -af

docker compose `
  --profile paper `
  --env-file .env `
  up -d
```

Volumes are preserved by this sequence.

