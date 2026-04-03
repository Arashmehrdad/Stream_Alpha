param(
    [string]$ConfigPath = ".\configs\training.m20.json",
    [switch]$RequireGpu,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot
$resolvedConfigPath = (Resolve-Path $ConfigPath).Path

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

Import-StreamAlphaEnvFile ".env"

$config = Get-Content -Path $resolvedConfigPath -Raw | ConvertFrom-Json
$trainingCommand = @("python", "-m", "app.training", "--config", $resolvedConfigPath)
$trainingCommandText = $trainingCommand -join " "
$modelNames = @($config.models.PSObject.Properties.Name)
$modelLabels = $modelNames -join ", "
$datasetModes = @(
    foreach ($modelProperty in $config.models.PSObject.Properties) {
        $mode = $modelProperty.Value.dataset_mode
        if ([string]::IsNullOrWhiteSpace($mode)) {
            $mode = "in_memory"
        }
        "$($modelProperty.Name)=$mode"
    }
)
$datasetModesLabel = $datasetModes -join ", "
$memoryProfiles = @(
    foreach ($modelProperty in $config.models.PSObject.Properties) {
        $modelConfig = $modelProperty.Value
        $modelKwargs = $modelConfig.model_kwargs
        $validBatchSize = if ($null -ne $modelKwargs.valid_batch_size) { $modelKwargs.valid_batch_size } else { "default" }
        $windowsBatchSize = if ($null -ne $modelKwargs.windows_batch_size) { $modelKwargs.windows_batch_size } else { "default" }
        $inferenceWindowsBatchSize = if ($null -ne $modelKwargs.inference_windows_batch_size) { $modelKwargs.inference_windows_batch_size } else { "default" }
        $stepSize = if ($null -ne $modelKwargs.step_size) { $modelKwargs.step_size } else { "default" }
        $precision = if ($null -ne $modelKwargs.precision) { $modelKwargs.precision } else { "default" }
        "$($modelProperty.Name)(batch=$($modelConfig.batch_size), valid=$validBatchSize, windows=$windowsBatchSize, infer_windows=$inferenceWindowsBatchSize, step=$stepSize, precision=$precision)"
    }
)
$memoryProfilesLabel = $memoryProfiles -join "; "

$allocatorHint = $env:PYTORCH_ALLOC_CONF
if ([string]::IsNullOrWhiteSpace($allocatorHint)) {
    $allocatorHint = $env:PYTORCH_CUDA_ALLOC_CONF
}
if ([string]::IsNullOrWhiteSpace($allocatorHint)) {
    $allocatorHint = "expandable_segments:True"
}
$env:PYTORCH_ALLOC_CONF = $allocatorHint
$env:PYTORCH_CUDA_ALLOC_CONF = $allocatorHint
if ([string]::IsNullOrWhiteSpace($env:NIXTLA_ID_AS_COL)) {
    $env:NIXTLA_ID_AS_COL = "1"
}

if ($DryRun) {
    Write-Host "M20 specialist training dry run"
    Write-Host "config path: $resolvedConfigPath"
    Write-Host "training source table: $($config.source_table)"
    Write-Host "symbols: $(@($config.symbols) -join ', ')"
    Write-Host "artifact root: $($config.artifact_root)"
    Write-Host "models: $modelLabels"
    Write-Host "specialist dataset mode: $datasetModesLabel"
    Write-Host "specialist memory profile: $memoryProfilesLabel"
    Write-Host "allocator hint: $allocatorHint"
    Write-Host "progress output: terminal bars disabled by config; see progress.log and progress_status.json inside the new artifact run directory"
    if ($RequireGpu) {
        Write-Host "gpu required: yes"
    } else {
        Write-Host "gpu required: no"
    }
    Write-Host "command: $trainingCommandText"
    exit 0
}

$preflightArgs = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $repoRoot "scripts\preflight_m20_training.ps1"),
    "-ConfigPath",
    $resolvedConfigPath
)
if ($RequireGpu) {
    $preflightArgs += "-RequireGpu"
}

& powershell @preflightArgs
if ($LASTEXITCODE -ne 0) {
    throw "M20 specialist preflight failed."
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
Write-Host "Starting M20 specialist training"
Write-Host "training source table: $($config.source_table)"
Write-Host "symbols: $(@($config.symbols) -join ', ')"
Write-Host "artifact root: $($config.artifact_root)"
Write-Host "models: $modelLabels"
Write-Host "specialist dataset mode: $datasetModesLabel"
Write-Host "specialist memory profile: $memoryProfilesLabel"
Write-Host "allocator hint: $allocatorHint"
Write-Host "progress output: terminal bars disabled by config; see progress.log and progress_status.json inside the new artifact run directory"
Write-Host ""

$previousUseErrorActionPreference = $PSNativeCommandUseErrorActionPreference
$PSNativeCommandUseErrorActionPreference = $false
try {
    & python -m app.training --config $resolvedConfigPath
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
    throw "M20 specialist training command failed with exit code $trainingExitCode."
}

$summaryPath = $null
if ($null -ne $latestArtifactDir) {
    $summaryPath = Join-Path $latestArtifactDir.FullName "summary.json"
}

Write-Host ""
Write-Host "M20 specialist training completed"
if ($null -ne $latestArtifactDir) {
    Write-Host "latest artifact dir: $($latestArtifactDir.FullName)"
}
if ($summaryPath -and (Test-Path $summaryPath)) {
    $summary = Get-Content -Path $summaryPath -Raw | ConvertFrom-Json
    Write-Host "summary.json path: $summaryPath"
    Write-Host "winner model name: $($summary.winner.model_name)"
    Write-Host "winner_after_cost_positive: $($summary.acceptance.winner_after_cost_positive)"
    Write-Host "meets_acceptance_target: $($summary.acceptance.meets_acceptance_target)"
}
