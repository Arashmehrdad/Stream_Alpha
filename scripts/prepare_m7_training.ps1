param(
    [switch]$DryRun,
    [switch]$SkipServiceStart,
    [ValidateRange(15, 600)]
    [int]$WaitSeconds = 90,
    [ValidateRange(2, 30)]
    [int]$PollSeconds = 5
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

function Get-RecommendedNextCommand {
    param(
        [pscustomobject]$Status,
        [bool]$CanStartServices,
        [bool]$ServiceStartWasSkipped
    )

    if (-not $Status.config_ok) {
        return "Fix .\configs\training.m7.json and rerun .\scripts\prepare_m7_training.ps1"
    }
    if (-not $Status.autogluon_installed) {
        return "python -m pip install -r .\requirements.txt"
    }
    if ($Status.ready_for_training) {
        return ".\scripts\start_m7_training.ps1"
    }
    if (-not $CanStartServices) {
        return "Copy .env.example to .env, then rerun .\scripts\prepare_m7_training.ps1"
    }
    if ($ServiceStartWasSkipped) {
        return ".\scripts\prepare_m7_training.ps1"
    }
    return ".\scripts\prepare_m7_training.ps1"
}

function Write-M7TrainingSummary {
    param(
        [pscustomobject]$Status,
        [string]$RecommendedNextCommand
    )

    $autogluonVersion = if ($Status.autogluon_installed) {
        $Status.autogluon_version
    }
    else {
        "missing"
    }
    $featureExists = if ($null -eq $Status.feature_table_exists) {
        "unknown"
    }
    elseif ($Status.feature_table_exists) {
        "yes"
    }
    else {
        "no"
    }
    $rowCount = if ($null -eq $Status.row_count) {
        "unknown"
    }
    else {
        [string]$Status.row_count
    }
    $fastaiStatus = if (-not $Status.fastai_installed) {
        "missing (optional breadth only, not a blocker)"
    }
    else {
        "$($Status.fastai_version) ($($Status.fastai_detail))"
    }

    Write-Host ""
    Write-Host "M7 training prep status"
    Write-Host "config ok: $($Status.config_ok)"
    if (-not $Status.config_ok -and $Status.config_error) {
        Write-Host "config detail: $($Status.config_error)"
    }
    Write-Host "autogluon version: $autogluonVersion"
    Write-Host "fastai optional breadth: $fastaiStatus"
    Write-Host "postgres reachable: $($Status.postgres_reachable)"
    if (-not $Status.postgres_reachable -and $Status.postgres_error) {
        Write-Host "postgres detail: $($Status.postgres_error)"
    }
    Write-Host "feature_ohlc exists: $featureExists"
    Write-Host "feature_ohlc row count: $rowCount"
    if ($null -ne $Status.unique_timestamps -and $null -ne $Status.required_unique_timestamps) {
        Write-Host "eligible unique timestamps: $($Status.unique_timestamps) / $($Status.required_unique_timestamps)"
    }
    if ($Status.readiness_detail) {
        Write-Host "readiness detail: $($Status.readiness_detail)"
    }
    Write-Host "recommended next command: $RecommendedNextCommand"
}

Import-StreamAlphaEnvFile ".env"

$status = Get-M7TrainingReadiness
$canStartServices = Test-Path ".env"
$needsServiceStart = (
    $Status.config_ok -and
    (
        (-not $Status.postgres_reachable) -or
        (-not $Status.feature_table_exists) -or
        (-not $Status.ready_for_training)
    )
)
$serviceStartWasSkipped = $false

if ($needsServiceStart) {
    if (-not $canStartServices) {
        Write-Host "Dev services were not started because .env is missing."
    }
    elseif ($SkipServiceStart -or $DryRun) {
        $serviceStartWasSkipped = $true
        Write-Host "Dry run / skip requested; dev services were not started."
        Write-Host "Would run: .\scripts\start-stack.ps1 -Profile dev"
    }
    else {
        & ".\scripts\start-stack.ps1" -Profile dev
        $deadline = (Get-Date).AddSeconds($WaitSeconds)
        do {
            Start-Sleep -Seconds $PollSeconds
            $status = Get-M7TrainingReadiness
        } while ((Get-Date) -lt $deadline -and -not $status.ready_for_training)
    }
}

$recommendedNextCommand = Get-RecommendedNextCommand `
    -Status $status `
    -CanStartServices $canStartServices `
    -ServiceStartWasSkipped $serviceStartWasSkipped

Write-M7TrainingSummary -Status $status -RecommendedNextCommand $recommendedNextCommand
