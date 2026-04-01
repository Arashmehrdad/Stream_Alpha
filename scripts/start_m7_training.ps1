param(
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

function Get-M7TrainingReadiness {
    $json = & python -m app.training.readiness --config ".\configs\training.m7.json" --json
    if ($LASTEXITCODE -ne 0) {
        throw "Training readiness helper failed."
    }
    return ($json | ConvertFrom-Json)
}

function Require-TrainingReady {
    param([pscustomobject]$Status)

    if (-not $Status.config_ok) {
        throw "Training config is not loadable: $($Status.config_error)"
    }
    if (-not $Status.autogluon_installed) {
        throw "AutoGluon Tabular is not installed. Install dependencies before starting training."
    }
    if (-not $Status.postgres_reachable) {
        throw "PostgreSQL is not reachable for training: $($Status.postgres_error)"
    }
    if (-not $Status.feature_table_exists) {
        throw "Training source table $($Status.source_table) does not exist. Run .\scripts\prepare_m7_training.ps1 first."
    }
    if (-not $Status.ready_for_training) {
        throw "feature_ohlc is not ready for the configured M7 split. Run .\scripts\prepare_m7_training.ps1 first."
    }
}

Import-StreamAlphaEnvFile ".env"

$status = Get-M7TrainingReadiness
Require-TrainingReady -Status $status

$trainingCommand = @("-m", "app.training", "--config", ".\configs\training.m7.json")
if ($DryRun) {
    Write-Host "Dry run: would run python $($trainingCommand -join ' ')"
    Write-Host "Artifact root: $($status.artifact_root)"
    exit 0
}

$artifactRoot = Join-Path $repoRoot $status.artifact_root
if (-not (Test-Path $artifactRoot)) {
    New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
}
$existingArtifactDirs = @(
    Get-ChildItem -Path $artifactRoot -Directory -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty FullName
)

& python @trainingCommand
if ($LASTEXITCODE -ne 0) {
    throw "M7 training command failed."
}

$latestArtifactDir = @(
    Get-ChildItem -Path $artifactRoot -Directory |
    Where-Object { $_.FullName -notin $existingArtifactDirs } |
    Sort-Object LastWriteTimeUtc -Descending
) | Select-Object -First 1

if ($null -eq $latestArtifactDir) {
    $latestArtifactDir = Get-ChildItem -Path $artifactRoot -Directory |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
}

if ($null -eq $latestArtifactDir) {
    throw "Training completed but no artifact directory was found under $artifactRoot"
}

$summaryPath = Join-Path $latestArtifactDir.FullName "summary.json"
$winnerModelName = $null
$winnerAfterCostPositive = $null
$meetsAcceptanceTarget = $null

if (Test-Path $summaryPath) {
    $summary = Get-Content -Path $summaryPath -Raw | ConvertFrom-Json
    $winnerModelName = $summary.winner.model_name
    $winnerAfterCostPositive = $summary.acceptance.winner_after_cost_positive
    $meetsAcceptanceTarget = $summary.acceptance.meets_acceptance_target
}

Write-Host ""
Write-Host "M7 training completed"
Write-Host "newest artifact dir: $($latestArtifactDir.FullName)"
Write-Host "summary.json path: $summaryPath"
if ($winnerModelName) {
    Write-Host "winner model name: $winnerModelName"
}
if ($null -ne $winnerAfterCostPositive) {
    Write-Host "winner_after_cost_positive: $winnerAfterCostPositive"
}
if ($null -ne $meetsAcceptanceTarget) {
    Write-Host "meets_acceptance_target: $meetsAcceptanceTarget"
}
