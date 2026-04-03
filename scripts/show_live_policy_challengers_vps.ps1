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
    "show-challengers",
    "--env-file",
    $EnvFile,
    "--json"
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

$jsonOutput = python @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Paper VPS challenger scoreboard command failed."
}

$summary = $jsonOutput | ConvertFrom-Json

if ($DryRun) {
    Write-Host "Paper VPS challenger scoreboard dry run"
    Write-Host "env file: $EnvFile"
    Write-Host "remote host: $($summary.remote_host)"
    Write-Host "remote app dir: $($summary.remote_app_dir)"
    Write-Host "command: python $($pythonArgs -join ' ')"
    exit 0
}

$scoreboard = $summary.scoreboard
$bestCandidate = $scoreboard.best_candidate
$production = $scoreboard.production_baseline

Write-Host "Paper VPS challenger scoreboard"
Write-Host "remote host: $($summary.remote_host)"
Write-Host "remote app dir: $($summary.remote_app_dir)"
Write-Host "artifact dir: $($scoreboard.artifact_dir)"
Write-Host "production trade count: $($production.hypothetical_trade_count)"
Write-Host ("production cumulative net proxy: {0:N6}" -f [double]$production.cumulative_net_proxy)

if ($null -eq $bestCandidate) {
    Write-Host "best candidate: none yet"
    exit 0
}

Write-Host "best candidate: $($bestCandidate.candidate_name)"
Write-Host ("best candidate cumulative net proxy: {0:N6}" -f [double]$bestCandidate.cumulative_net_proxy)
Write-Host "best candidate trade count: $($bestCandidate.hypothetical_trade_count)"
Write-Host ("best candidate max drawdown proxy: {0:N6}" -f [double]$bestCandidate.max_drawdown_proxy)
Write-Host "positive but sparse: $($bestCandidate.positive_but_sparse)"
Write-Host "never trades TREND_UP: $($bestCandidate.never_trades_trend_up)"
if ($bestCandidate.warnings.Count -gt 0) {
    Write-Host "warnings: $($bestCandidate.warnings -join ' | ')"
}
