param(
    [string]$ArtifactRoot = ".\artifacts\training\m20",
    [string]$RunDir = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $null
    }
    return Get-Content -Path $Path -Raw | ConvertFrom-Json
}

function Resolve-M20RunDir {
    if (-not [string]::IsNullOrWhiteSpace($RunDir)) {
        if (-not (Test-Path $RunDir)) {
            throw "M20 run directory not found: $RunDir"
        }
        return (Resolve-Path $RunDir).Path
    }
    if (-not (Test-Path $ArtifactRoot)) {
        return $null
    }
    $latest = Get-ChildItem -Path $ArtifactRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        return $null
    }
    return $latest.FullName
}

function Resolve-ExecutionMode {
    param(
        [object]$RunConfig,
        [string]$ResolvedRunDir
    )
    if ($null -ne $RunConfig -and $null -ne $RunConfig.execution_mode) {
        return [string]$RunConfig.execution_mode
    }
    $fitManifestPath = Join-Path $ResolvedRunDir "fitted_models\manifest.json"
    $fitManifest = Read-JsonFile $fitManifestPath
    if ($null -ne $fitManifest -and $null -ne $fitManifest.mode) {
        return [string]$fitManifest.mode
    }
    return "unknown"
}

function Write-SpecialistVerdicts {
    param([object]$Summary)
    if ($null -eq $Summary -or $null -eq $Summary.specialist_verdicts) {
        return
    }
    foreach ($property in $Summary.specialist_verdicts.PSObject.Properties) {
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

Write-Host "M20 training status"
$resolvedRunDir = Resolve-M20RunDir
if ($null -eq $resolvedRunDir) {
    Write-Host "latest artifact dir: none"
    Write-Host "run status: no_artifacts"
    Write-Host "blocker: no M20 artifact directories found"
    exit 0
}

Write-Host "latest artifact dir: $resolvedRunDir"
$summaryPath = Join-Path $resolvedRunDir "summary.json"
$progressPath = Join-Path $resolvedRunDir "progress_status.json"
$checkpointPath = Join-Path $resolvedRunDir "checkpoint.json"
$runConfigPath = Join-Path $resolvedRunDir "run_config.json"

$summary = Read-JsonFile $summaryPath
$progress = Read-JsonFile $progressPath
$runConfig = Read-JsonFile $runConfigPath
$executionMode = Resolve-ExecutionMode -RunConfig $runConfig -ResolvedRunDir $resolvedRunDir
Write-Host "execution mode: $executionMode"

if ($null -ne $progress) {
    Write-Host "progress state: $($progress.state)"
    Write-Host "progress stage: $($progress.stage)"
    if ($null -ne $progress.event) {
        Write-Host "progress event: $($progress.event)"
    }
    if ($null -ne $progress.model_name) {
        Write-Host "progress model: $($progress.model_name)"
    }
}

if ($null -ne $summary) {
    Write-Host "run status: completed"
    Write-Host "winner model name: $($summary.winner.model_name)"
    Write-Host "acceptance scope: $($summary.acceptance.scope)"
    Write-Host "verdict basis: $($summary.acceptance.verdict_basis)"
    Write-Host "incumbent model version: $($summary.acceptance.incumbent_model_version)"
    Write-Host "meets_acceptance_target: $($summary.acceptance.meets_acceptance_target)"
    Write-SpecialistVerdicts -Summary $summary
} else {
    Write-Host "run status: incomplete"
    Write-Host "blocker: summary.json missing"
}

if (Test-Path $checkpointPath) {
    $checkpoint = Get-Item $checkpointPath
    Write-Host "checkpoint: present ($($checkpoint.Length) bytes)"
}
