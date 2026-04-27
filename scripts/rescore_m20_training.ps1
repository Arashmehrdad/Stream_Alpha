param(
    [string]$ConfigPath = ".\configs\training.m20.json",
    [string]$FittedModelsDir = ".\artifacts\training\m20\20260405T023104Z\fitted_models",
    [string]$ParquetDir = ".\exports\feature_ohlc_for_colab",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$resolvedConfigPath = (Resolve-Path $ConfigPath).Path
if (-not (Test-Path $FittedModelsDir)) {
    throw "Fitted models directory not found: $FittedModelsDir"
}
$resolvedFittedModelsDir = (Resolve-Path $FittedModelsDir).Path

$resolvedParquetDir = $null
if (-not [string]::IsNullOrWhiteSpace($ParquetDir) -and (Test-Path $ParquetDir)) {
    $resolvedParquetDir = (Resolve-Path $ParquetDir).Path
}

$config = Get-Content -Path $resolvedConfigPath -Raw | ConvertFrom-Json
$modelNames = @($config.models.PSObject.Properties.Name)
$modelLabels = $modelNames -join ", "
$trainingCommand = @(
    "python",
    "-m",
    "app.training",
    "--config",
    $resolvedConfigPath,
    "--score-only",
    $resolvedFittedModelsDir
)
if ($null -ne $resolvedParquetDir) {
    $trainingCommand += @("--parquet-dir", $resolvedParquetDir)
}
$trainingCommandText = $trainingCommand -join " "
$pythonArgs = $trainingCommand[1..($trainingCommand.Count - 1)]

if ($DryRun) {
    Write-Host "M20 specialist score-only dry run"
    Write-Host "config path: $resolvedConfigPath"
    Write-Host "fitted models dir: $resolvedFittedModelsDir"
    Write-Host "training source table: $($config.source_table)"
    Write-Host "symbols: $(@($config.symbols) -join ', ')"
    Write-Host "artifact root: $($config.artifact_root)"
    Write-Host "models: $modelLabels"
    if ($null -ne $resolvedParquetDir) {
        Write-Host "parquet dataset: $resolvedParquetDir"
    } else {
        Write-Host "parquet dataset: not found; score-only will use configured database source"
    }
    Write-Host "command: $trainingCommandText"
    exit 0
}

$artifactRoot = Join-Path $repoRoot $config.artifact_root
if (-not (Test-Path $artifactRoot)) {
    New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
}
$existingArtifactDirs = @(
    Get-ChildItem -Path $artifactRoot -Directory -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty FullName
)

Write-Host ""
Write-Host "Starting M20 specialist score-only rerun"
Write-Host "config path: $resolvedConfigPath"
Write-Host "fitted models dir: $resolvedFittedModelsDir"
if ($null -ne $resolvedParquetDir) {
    Write-Host "parquet dataset: $resolvedParquetDir"
}
Write-Host "models: $modelLabels"
Write-Host ""

$previousUseErrorActionPreference = $PSNativeCommandUseErrorActionPreference
$PSNativeCommandUseErrorActionPreference = $false
try {
    & python @pythonArgs
    $trainingExitCode = $LASTEXITCODE
} finally {
    $PSNativeCommandUseErrorActionPreference = $previousUseErrorActionPreference
}

$latestArtifactDir = Get-ChildItem -Path $artifactRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notin $existingArtifactDirs } |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
if ($null -eq $latestArtifactDir) {
    $latestArtifactDir = Get-ChildItem -Path $artifactRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
}

if ($trainingExitCode -ne 0) {
    if ($null -ne $latestArtifactDir) {
        Write-Host ""
        Write-Host "latest artifact dir: $($latestArtifactDir.FullName)"
    }
    throw "M20 specialist score-only command failed with exit code $trainingExitCode."
}

Write-Host ""
Write-Host "M20 specialist score-only completed"
if ($null -ne $latestArtifactDir) {
    Write-Host "latest artifact dir: $($latestArtifactDir.FullName)"
}

$summaryPath = $null
if ($null -ne $latestArtifactDir) {
    $summaryPath = Join-Path $latestArtifactDir.FullName "summary.json"
}
if ($summaryPath -and (Test-Path $summaryPath)) {
    $summary = Get-Content -Path $summaryPath -Raw | ConvertFrom-Json
    Write-Host "summary.json path: $summaryPath"
    Write-Host "winner model name: $($summary.winner.model_name)"
    Write-Host "acceptance scope: $($summary.acceptance.scope)"
    Write-Host "verdict basis: $($summary.acceptance.verdict_basis)"
    Write-Host "incumbent model version: $($summary.acceptance.incumbent_model_version)"
    Write-Host "meets_acceptance_target: $($summary.acceptance.meets_acceptance_target)"
    if ($null -ne $summary.specialist_verdicts) {
        foreach ($property in $summary.specialist_verdicts.PSObject.Properties) {
            $verdict = $property.Value
            Write-Host (
                "specialist verdict: {0} role={1} verdict={2} basis={3}" -f
                $property.Name,
                $verdict.candidate_role,
                $verdict.verdict,
                $verdict.verdict_basis
            )
        }
    }
}
