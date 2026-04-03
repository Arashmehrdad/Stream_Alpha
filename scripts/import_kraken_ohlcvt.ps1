param(
    [string]$DatasetRoot,
    [string]$TrainingConfig = ".\configs\training.m7.json",
    [string[]]$Symbols = @("BTC/USD", "ETH/USD", "SOL/USD"),
    [string]$Start,
    [string]$End,
    [switch]$ReplayOnly,
    [switch]$SkipFeatureReplay,
    [switch]$ReportOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

function Import-StreamAlphaEnvFile {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }

    foreach ($line in Get-Content -Path $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line.TrimStart().StartsWith("#")) {
            continue
        }
        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) {
            continue
        }
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

function Resolve-DefaultKrakenDatasetRoot {
    $candidates = @(
        ".\Datasets\master_q4",
        ".\Datasets\Kraken_OHLCVT\master_q4"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return $null
}

Import-StreamAlphaEnvFile ".env"

$resolvedDatasetRoot = if ($DatasetRoot) {
    (Resolve-Path $DatasetRoot).Path
}
else {
    Resolve-DefaultKrakenDatasetRoot
}

if (-not $resolvedDatasetRoot -and -not $ReplayOnly -and -not $ReportOnly) {
    throw "Kraken dataset root was not found. Pass -DatasetRoot explicitly."
}

$command = @(
    "-m",
    "app.ingestion.import_kraken_ohlcvt",
    "--training-config",
    $TrainingConfig,
    "--json"
)
if ($resolvedDatasetRoot) {
    $command += @("--dataset-root", $resolvedDatasetRoot)
}
if ($Symbols.Count -gt 0) {
    $command += "--symbols"
    $command += $Symbols
}
if ($Start) {
    $command += @("--start", $Start)
}
if ($End) {
    $command += @("--end", $End)
}
if ($ReplayOnly) {
    $command += "--skip-raw-import"
}
if ($SkipFeatureReplay) {
    $command += "--skip-feature-replay"
}
if ($ReportOnly) {
    $command += "--report-only"
}

if ($DryRun) {
    Write-Host "Kraken OHLCVT import dry run"
    if ($resolvedDatasetRoot) {
        Write-Host "dataset root: $resolvedDatasetRoot"
    }
    Write-Host "training config: $TrainingConfig"
    Write-Host "symbols: $($Symbols -join ', ')"
    Write-Host "command: python $($command -join ' ')"
    exit 0
}

$json = & python @command
if ($LASTEXITCODE -ne 0) {
    throw "Kraken OHLCVT import command failed."
}
$summary = $json | ConvertFrom-Json
$rawStats = @($summary.raw_import)
$rawRowsTotal = if ($summary.readiness.raw_rows_total -ne $null) {
    [int]$summary.readiness.raw_rows_total
}
else {
    ($rawStats | Measure-Object -Property selected_rows -Sum).Sum
}
$importedSymbols = if ($summary.symbols) {
    @($summary.symbols) -join ", "
}
else {
    $Symbols -join ", "
}

Write-Host ""
Write-Host "Kraken OHLCVT import complete"
Write-Host "imported symbols: $importedSymbols"
if ($resolvedDatasetRoot) {
    Write-Host "dataset root: $resolvedDatasetRoot"
}
foreach ($stats in $rawStats) {
    Write-Host (
        "raw import {0}: parsed={1} selected={2} created={3} updated={4} unchanged={5}" -f
        $stats.symbol,
        $stats.parsed_rows,
        $stats.selected_rows,
        $stats.created_rows,
        $stats.updated_rows,
        $stats.unchanged_rows
    )
}
if ($summary.feature_replay -ne $null) {
    Write-Host (
        "feature replay: generated={0} created={1} updated={2} unchanged={3}" -f
        $summary.feature_replay.generated_rows,
        $summary.feature_replay.created_rows,
        $summary.feature_replay.updated_rows,
        $summary.feature_replay.unchanged_rows
    )
}
Write-Host "raw rows total: $rawRowsTotal"
Write-Host "feature rows total: $($summary.readiness.feature_rows_total)"
Write-Host "labeled rows total: $($summary.readiness.labeled_rows_total)"
Write-Host "readiness artifact path: $($summary.readiness_report_artifact_dir)"
Write-Host "ready for training: $($summary.readiness.ready_for_training)"
Write-Host "readiness detail: $($summary.readiness.readiness_detail)"
