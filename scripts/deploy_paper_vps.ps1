param(
    [string]$EnvFile = ".env",
    [ValidateRange(26, 512)]
    [int]$BackfillLookbackCandles = 128,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path $EnvFile)) {
    throw "Missing env file: $EnvFile"
}

$pythonArgs = @(
    "-m",
    "app.deployment.paper_vps",
    "deploy",
    "--env-file",
    $EnvFile,
    "--lookback-candles",
    $BackfillLookbackCandles,
    "--json"
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

$jsonOutput = python @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Paper VPS deploy command failed."
}

$summary = $jsonOutput | ConvertFrom-Json

if ($DryRun) {
    Write-Host "Paper VPS deploy dry run"
    Write-Host "env file: $EnvFile"
    Write-Host "remote host: $($summary.remote_host)"
    Write-Host "remote app dir: $($summary.remote_app_dir)"
    Write-Host "started services: $($summary.started_services -join ', ')"
    Write-Host "tail logs: $($summary.tail_logs_hint)"
    Write-Host "inspect challengers: $($summary.inspect_challengers_hint)"
    Write-Host "command: python $($pythonArgs -join ' ')"
    exit 0
}

Write-Host "Paper VPS deploy summary"
Write-Host "remote host: $($summary.remote_host)"
Write-Host "remote app dir: $($summary.remote_app_dir)"
Write-Host "started services: $($summary.running_services -join ', ')"
Write-Host "paper runner up: $($summary.paper_runner_up)"
Write-Host "challenger artifacts exist: $($summary.challenger_artifacts_exist)"
Write-Host "tail logs: $($summary.tail_logs_hint)"
Write-Host "inspect challengers: $($summary.inspect_challengers_hint)"
