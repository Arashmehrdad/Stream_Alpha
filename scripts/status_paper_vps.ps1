param(
    [string]$EnvFile = ".env",
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
    "status",
    "--env-file",
    $EnvFile,
    "--json"
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

$jsonOutput = python @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Paper VPS status command failed."
}

$summary = $jsonOutput | ConvertFrom-Json

if ($DryRun) {
    Write-Host "Paper VPS status dry run"
    Write-Host "env file: $EnvFile"
    Write-Host "remote host: $($summary.remote_host)"
    Write-Host "remote app dir: $($summary.remote_app_dir)"
    Write-Host "command: python $($pythonArgs -join ' ')"
    exit 0
}

Write-Host "Paper VPS status"
Write-Host "remote host: $($summary.remote_host)"
Write-Host "remote app dir: $($summary.remote_app_dir)"
Write-Host "running services: $($summary.running_services -join ', ')"
Write-Host "paper runner up: $($summary.paper_runner_up)"
Write-Host "challenger artifacts exist: $($summary.challenger_artifacts_exist)"
Write-Host "challenger scoreboard path: $($summary.challenger_scoreboard_remote_path)"
