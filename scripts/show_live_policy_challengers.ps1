param(
    [string]$ConfigPath = ".\configs\paper_trading.paper.yaml",
    [string]$TrainingConfigPath = ".\configs\training.m7.json",
    [string]$ArtifactDir = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonArgs = @(
    "-m",
    "app.training.live_policy_challenger",
    "--trading-config",
    $ConfigPath,
    "--training-config",
    $TrainingConfigPath
)

if ($ArtifactDir -ne "") {
    $pythonArgs += @("--artifact-dir", $ArtifactDir)
}

if ($DryRun) {
    Write-Host "Live policy challenger observer dry run"
    Write-Host "trading config: $ConfigPath"
    Write-Host "training config: $TrainingConfigPath"
    if ($ArtifactDir -ne "") {
        Write-Host "challenger artifact dir override: $ArtifactDir"
    }
    Write-Host "command: python $($pythonArgs -join ' ')"
    exit 0
}

$jsonOutput = python @($pythonArgs + @("--json"))
if ($LASTEXITCODE -ne 0) {
    throw "Live policy challenger summary command failed."
}

$summary = $jsonOutput | ConvertFrom-Json -AsHashtable
$bestCandidate = $summary["best_candidate"]
$production = $summary["production_baseline"]

Write-Host "Live policy challenger scoreboard"
Write-Host "artifact dir: $($summary['artifact_dir'])"
Write-Host "production trade count: $($production['hypothetical_trade_count'])"
Write-Host ("production cumulative net proxy: {0:N6}" -f [double]$production["cumulative_net_proxy"])
if ($null -eq $bestCandidate) {
    Write-Host "best candidate: none yet"
    exit 0
}

Write-Host "best candidate: $($bestCandidate['candidate_name'])"
Write-Host ("best candidate cumulative net proxy: {0:N6}" -f [double]$bestCandidate["cumulative_net_proxy"])
Write-Host "best candidate trade count: $($bestCandidate['hypothetical_trade_count'])"
Write-Host ("best candidate max drawdown proxy: {0:N6}" -f [double]$bestCandidate["max_drawdown_proxy"])
Write-Host "positive but sparse: $($bestCandidate['positive_but_sparse'])"
Write-Host "never trades TREND_UP: $($bestCandidate['never_trades_trend_up'])"
if ($bestCandidate["warnings"].Count -gt 0) {
    Write-Host "warnings: $($bestCandidate['warnings'] -join ' | ')"
}
Write-Host "scoreboard json: $($summary['output_files']['latest_scoreboard_json'])"
Write-Host "scoreboard csv: $($summary['output_files']['latest_scoreboard_csv'])"
Write-Host "summary md: $($summary['output_files']['summary_md'])"
