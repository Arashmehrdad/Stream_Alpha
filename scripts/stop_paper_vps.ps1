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
    "stop",
    "--env-file",
    $EnvFile,
    "--json"
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

$jsonOutput = python @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Paper VPS stop command failed."
}

$summary = $jsonOutput | ConvertFrom-Json

if ($DryRun) {
    Write-Host "Paper VPS stop dry run"
    Write-Host "env file: $EnvFile"
    Write-Host "remote host: $($summary.remote_host)"
    Write-Host "remote app dir: $($summary.remote_app_dir)"
    Write-Host "command: python $($pythonArgs -join ' ')"
    exit 0
}

Write-Host "Paper VPS stop summary"
Write-Host "remote host: $($summary.remote_host)"
Write-Host "remote app dir: $($summary.remote_app_dir)"
Write-Host "stopped: $($summary.stopped)"
