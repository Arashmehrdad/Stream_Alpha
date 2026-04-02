param(
    [switch]$DryRun,
    [ValidateRange(5, 120)]
    [int]$StatusSeconds = 15
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot
$localTrainingTempRoot = Join-Path $repoRoot "artifacts\tmp\autogluon"
$trainingConfigPath = ".\configs\training.m7.json"

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
    $json = & python -m app.training.readiness --config $trainingConfigPath --json
    if ($LASTEXITCODE -ne 0) {
        throw "Training readiness helper failed."
    }
    return ($json | ConvertFrom-Json)
}

function Get-TrainingBudgetSeconds {
    $config = Get-Content -Path $trainingConfigPath -Raw | ConvertFrom-Json
    if (
        $null -eq $config.models -or
        $null -eq $config.models.autogluon_tabular -or
        $null -eq $config.models.autogluon_tabular.time_limit
    ) {
        return $null
    }
    return [int]$config.models.autogluon_tabular.time_limit
}

function Get-LogicalCpuCount {
    try {
        $computerSystem = Get-CimInstance Win32_ComputerSystem
        if ($null -ne $computerSystem.NumberOfLogicalProcessors) {
            return [int]$computerSystem.NumberOfLogicalProcessors
        }
    }
    catch {
    }
    return [Environment]::ProcessorCount
}

function Resolve-CurrentFitDirectory {
    param(
        [datetime]$StartedAt
    )

    $fitDirs = @(
        Get-ChildItem -Path $localTrainingTempRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "streamalpha-autogluon-fit-*" } |
        Where-Object { $_.LastWriteTime -ge $StartedAt.AddMinutes(-2) } |
        Sort-Object LastWriteTime -Descending
    )
    if ($fitDirs.Count -eq 0) {
        return $null
    }
    return $fitDirs[0].FullName
}

function Get-AutoGluonFitProgress {
    param(
        [string]$FitDirectory
    )

    if ([string]::IsNullOrWhiteSpace($FitDirectory) -or -not (Test-Path $FitDirectory)) {
        return $null
    }
    $json = & python -m app.training.progress --fit-dir $FitDirectory
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($json)) {
        return $null
    }
    return ($json | ConvertFrom-Json)
}

function Format-Duration {
    param([timespan]$Duration)

    if ($Duration.TotalSeconds -lt 0) {
        $Duration = [timespan]::Zero
    }
    return "{0:00}:{1:00}:{2:00}" -f $Duration.Hours, $Duration.Minutes, $Duration.Seconds
}

function Write-TrainingHeartbeat {
    param(
        [datetime]$StartedAt,
        [int]$BudgetSeconds,
        [pscustomobject]$ProgressSnapshot
    )

    $elapsed = (Get-Date) - $StartedAt
    $elapsedText = Format-Duration -Duration $elapsed
    $statusParts = @("elapsed $elapsedText")
    $percentComplete = 0

    if ($BudgetSeconds -gt 0) {
        $budget = [timespan]::FromSeconds($BudgetSeconds)
        $budgetText = Format-Duration -Duration $budget
        $statusParts += "budget $budgetText"
        if ($elapsed.TotalSeconds -lt $BudgetSeconds) {
            $remaining = [timespan]::FromSeconds($BudgetSeconds - [int]$elapsed.TotalSeconds)
            $statusParts += "eta ~$(Format-Duration -Duration $remaining)"
            $percentComplete = [math]::Min(
                99,
                [int][math]::Floor(($elapsed.TotalSeconds / $BudgetSeconds) * 100.0)
            )
        }
        else {
            $overrun = [timespan]::FromSeconds([int]$elapsed.TotalSeconds - $BudgetSeconds)
            $statusParts += "over budget by $(Format-Duration -Duration $overrun)"
            $percentComplete = 99
        }
    }

    if ($null -ne $ProgressSnapshot) {
        if ($ProgressSnapshot.current_best_model) {
            $statusParts += "best $($ProgressSnapshot.current_best_model)"
        }
        if ($ProgressSnapshot.latest_model_name) {
            $statusParts += "latest $($ProgressSnapshot.latest_model_name)"
        }
        if ($ProgressSnapshot.total_model_count) {
            $statusParts += (
                "discovered $($ProgressSnapshot.discovered_model_count)/$($ProgressSnapshot.total_model_count) model dirs"
            )
        }
        elseif ($ProgressSnapshot.discovered_model_count -gt 0) {
            $statusParts += "discovered $($ProgressSnapshot.discovered_model_count) model dirs"
        }
    }

    Write-Progress `
        -Activity "Training M7 AutoGluon" `
        -Status ($statusParts -join " | ") `
        -PercentComplete $percentComplete
}

function Get-NewestArtifactDirectory {
    param(
        [string]$ArtifactRoot,
        [string[]]$ExistingArtifactDirectories
    )

    $newArtifactDirectory = @(
        Get-ChildItem -Path $ArtifactRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notin $ExistingArtifactDirectories } |
        Sort-Object LastWriteTimeUtc -Descending
    ) | Select-Object -First 1
    if ($null -ne $newArtifactDirectory) {
        return $newArtifactDirectory
    }
    return Get-ChildItem -Path $ArtifactRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
}

function Test-CompletedTrainingArtifact {
    param([System.IO.DirectoryInfo]$ArtifactDirectory)

    if ($null -eq $ArtifactDirectory) {
        return $false
    }
    $requiredFiles = @(
        "summary.json",
        "model.joblib",
        "fold_metrics.csv",
        "oof_predictions.csv",
        "feature_columns.json"
    )
    foreach ($fileName in $requiredFiles) {
        if (-not (Test-Path (Join-Path $ArtifactDirectory.FullName $fileName))) {
            return $false
        }
    }
    return $true
}

function Get-LogTail {
    param(
        [string]$Path,
        [int]$LineCount = 40
    )

    if (-not (Test-Path $Path)) {
        return @()
    }
    return @(Get-Content -Path $Path -Tail $LineCount)
}

function Move-TrainingLogsIntoArtifact {
    param(
        [string]$StdoutPath,
        [string]$StderrPath,
        [System.IO.DirectoryInfo]$ArtifactDirectory
    )

    if ($null -eq $ArtifactDirectory) {
        return @{
            stdout = $StdoutPath
            stderr = $StderrPath
        }
    }

    $resolvedStdoutPath = $StdoutPath
    $resolvedStderrPath = $StderrPath
    $artifactStdoutPath = Join-Path $ArtifactDirectory.FullName "training.stdout.log"
    $artifactStderrPath = Join-Path $ArtifactDirectory.FullName "training.stderr.log"

    if (Test-Path $StdoutPath) {
        Move-Item -Force -Path $StdoutPath -Destination $artifactStdoutPath
        $resolvedStdoutPath = $artifactStdoutPath
    }
    if (Test-Path $StderrPath) {
        Move-Item -Force -Path $StderrPath -Destination $artifactStderrPath
        $resolvedStderrPath = $artifactStderrPath
    }

    return @{
        stdout = $resolvedStdoutPath
        stderr = $resolvedStderrPath
    }
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

function Set-LocalTrainingTempEnvironment {
    param([string]$TempRoot)

    New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
    [Environment]::SetEnvironmentVariable("STREAMALPHA_LOCAL_TRAINING_TEMP_ROOT", $TempRoot, "Process")
    [Environment]::SetEnvironmentVariable("TMP", $TempRoot, "Process")
    [Environment]::SetEnvironmentVariable("TEMP", $TempRoot, "Process")
    [Environment]::SetEnvironmentVariable("TMPDIR", $TempRoot, "Process")
    [Environment]::SetEnvironmentVariable("RAY_TMPDIR", $TempRoot, "Process")
}

Import-StreamAlphaEnvFile ".env"
Set-LocalTrainingTempEnvironment -TempRoot $localTrainingTempRoot

$status = Get-M7TrainingReadiness
Require-TrainingReady -Status $status
$trainingBudgetSeconds = Get-TrainingBudgetSeconds
$logicalCpuCount = Get-LogicalCpuCount

$trainingCommand = @("-m", "app.training", "--config", $trainingConfigPath)
if ($DryRun) {
    Write-Host "Local training temp root: $localTrainingTempRoot"
    Write-Host "CPU mode: sequential_local bagging plus AutoGluon model-level multithreading on up to $logicalCpuCount logical CPUs"
    if ($null -ne $trainingBudgetSeconds) {
        Write-Host "AutoGluon time budget: $trainingBudgetSeconds seconds"
    }
    Write-Host "Dry run: would run python $($trainingCommand -join ' ')"
    Write-Host "Artifact root: $($status.artifact_root)"
    exit 0
}

$startedAt = Get-Date
Write-Host "Local training temp root: $localTrainingTempRoot"
Write-Host "CPU mode: sequential_local bagging plus AutoGluon model-level multithreading on up to $logicalCpuCount logical CPUs"
if ($null -ne $trainingBudgetSeconds) {
    Write-Host "AutoGluon time budget: $trainingBudgetSeconds seconds"
}

$artifactRoot = Join-Path $repoRoot $status.artifact_root
if (-not (Test-Path $artifactRoot)) {
    New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
}
$existingArtifactDirs = @(
    Get-ChildItem -Path $artifactRoot -Directory -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty FullName
)
$runLabel = Get-Date -Format "yyyyMMddTHHmmssfff"
$stdoutLogPath = Join-Path $localTrainingTempRoot "m7-training-$runLabel.stdout.log"
$stderrLogPath = Join-Path $localTrainingTempRoot "m7-training-$runLabel.stderr.log"

$process = Start-Process `
    -FilePath "python" `
    -ArgumentList $trainingCommand `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLogPath `
    -RedirectStandardError $stderrLogPath `
    -PassThru
$fitDirectory = $null

while (-not $process.HasExited) {
    if ($null -eq $fitDirectory) {
        $fitDirectory = Resolve-CurrentFitDirectory -StartedAt $startedAt
    }
    $progressSnapshot = Get-AutoGluonFitProgress -FitDirectory $fitDirectory
    if ($null -ne $progressSnapshot) {
        $fitDirectory = $progressSnapshot.fit_dir
    }
    Write-TrainingHeartbeat `
        -StartedAt $startedAt `
        -BudgetSeconds $trainingBudgetSeconds `
        -ProgressSnapshot $progressSnapshot
    Start-Sleep -Seconds $StatusSeconds
    $process.Refresh()
}
Write-Progress -Activity "Training M7 AutoGluon" -Completed

$latestArtifactDir = Get-NewestArtifactDirectory `
    -ArtifactRoot $artifactRoot `
    -ExistingArtifactDirectories $existingArtifactDirs
$logPaths = Move-TrainingLogsIntoArtifact `
    -StdoutPath $stdoutLogPath `
    -StderrPath $stderrLogPath `
    -ArtifactDirectory $latestArtifactDir

if ($null -eq $latestArtifactDir) {
    throw "Training completed but no artifact directory was found under $artifactRoot"
}

if ($process.ExitCode -ne 0) {
    if (-not (Test-CompletedTrainingArtifact -ArtifactDirectory $latestArtifactDir)) {
        Write-Host "training stdout log: $($logPaths.stdout)"
        Write-Host "training stderr log: $($logPaths.stderr)"
        $stderrTail = Get-LogTail -Path $logPaths.stderr
        if ($stderrTail.Count -gt 0) {
            Write-Host ""
            Write-Host "stderr tail:"
            $stderrTail | ForEach-Object { Write-Host $_ }
        }
        $stdoutTail = Get-LogTail -Path $logPaths.stdout
        if ($stdoutTail.Count -gt 0) {
            Write-Host ""
            Write-Host "stdout tail:"
            $stdoutTail | ForEach-Object { Write-Host $_ }
        }
        throw "M7 training command failed with exit code $($process.ExitCode)."
    }
    Write-Warning "python -m app.training exited with code $($process.ExitCode) after a complete artifact was written; treating the run as completed and surfacing the log paths."
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
Write-Host "local training temp root: $localTrainingTempRoot"
Write-Host "newest artifact dir: $($latestArtifactDir.FullName)"
Write-Host "summary.json path: $summaryPath"
Write-Host "training stdout log: $($logPaths.stdout)"
Write-Host "training stderr log: $($logPaths.stderr)"
if ($winnerModelName) {
    Write-Host "winner model name: $winnerModelName"
}
if ($null -ne $winnerAfterCostPositive) {
    Write-Host "winner_after_cost_positive: $winnerAfterCostPositive"
}
if ($null -ne $meetsAcceptanceTarget) {
    Write-Host "meets_acceptance_target: $meetsAcceptanceTarget"
}
