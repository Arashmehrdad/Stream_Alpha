param(
    [string]$RunDir,
    [double[]]$Thresholds,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot
$artifactRoot = Join-Path $repoRoot "artifacts\training\m7"

function Resolve-M7RunDirectory {
    param([string]$RequestedRunDir)

    if (-not [string]::IsNullOrWhiteSpace($RequestedRunDir)) {
        if (-not (Test-Path $RequestedRunDir)) {
            throw "Requested M7 run directory does not exist: $RequestedRunDir"
        }
        $resolvedPath = (Resolve-Path $RequestedRunDir).Path
        if (-not (Test-Path $resolvedPath -PathType Container)) {
            throw "Requested M7 run path is not a directory: $resolvedPath"
        }
        return $resolvedPath
    }

    if (-not (Test-Path $artifactRoot -PathType Container)) {
        throw "No M7 artifact root exists yet: $artifactRoot"
    }

    $newestRun = Get-ChildItem -Path $artifactRoot -Directory -ErrorAction Stop |
        Where-Object { -not $_.Name.StartsWith("_") } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $newestRun) {
        throw "No M7 run directories were found under $artifactRoot"
    }
    return $newestRun.FullName
}

function Require-RunArtifactFiles {
    param([string]$ResolvedRunDir)

    $requiredPaths = @(
        (Join-Path $ResolvedRunDir "summary.json"),
        (Join-Path $ResolvedRunDir "oof_predictions.csv")
    )
    foreach ($requiredPath in $requiredPaths) {
        if (-not (Test-Path $requiredPath -PathType Leaf)) {
            throw "Required completed-run file is missing: $requiredPath"
        }
    }
}

$resolvedRunDir = Resolve-M7RunDirectory -RequestedRunDir $RunDir
Require-RunArtifactFiles -ResolvedRunDir $resolvedRunDir

$analysisCommand = @(
    "-m",
    "app.training.threshold_analysis",
    "--run-dir",
    $resolvedRunDir,
    "--json"
)
if ($Thresholds.Count -gt 0) {
    $analysisCommand += "--thresholds"
    $analysisCommand += @($Thresholds | ForEach-Object { "{0:0.00}" -f $_ })
}

if ($DryRun) {
    Write-Host "Resolved M7 run dir: $resolvedRunDir"
    Write-Host "Dry run: would run python $($analysisCommand -join ' ')"
    Write-Host "Expected analysis dir: $(Join-Path $resolvedRunDir 'threshold_analysis')"
    exit 0
}

$json = & python @analysisCommand
if ($LASTEXITCODE -ne 0) {
    throw "M7 threshold analysis command failed."
}
$analysis = $json | ConvertFrom-Json

function Format-PolicyLine {
    param([object]$Policy)

    if ($null -eq $Policy) {
        return "none"
    }
    return (
        "$($Policy.policy_name) @ threshold $([double]$Policy.threshold)" +
        " (trade_count=$([int]$Policy.trade_count), " +
        "trade_rate=$('{0:N4}' -f [double]$Policy.trade_rate), " +
        "mean_net=$('{0:N6}' -f [double]$Policy.mean_long_only_net_value_proxy), " +
        "after_cost_positive=$([bool]$Policy.after_cost_positive))"
    )
}

Write-Host ""
Write-Host "M7 threshold analysis completed"
Write-Host "run dir: $($analysis.run_dir)"
Write-Host "analysis dir: $($analysis.analysis_dir)"
Write-Host "model analyzed: $($analysis.model_name)"
Write-Host "best global threshold by mean_long_only_net_value_proxy: $(Format-PolicyLine -Policy $analysis.best_global_threshold_policy)"
Write-Host "best threshold with TREND_DOWN blocked: $(Format-PolicyLine -Policy $analysis.best_no_long_in_trend_down_policy)"
Write-Host "any tested policy after-cost positive: $($analysis.any_after_cost_positive)"
if ($analysis.worst_fold_for_best_overall -ne $null) {
    Write-Host (
        "worst fold under best overall policy: fold $($analysis.worst_fold_for_best_overall.fold_index) " +
        "(trade_count=$($analysis.worst_fold_for_best_overall.trade_count), " +
        "trade_rate=$('{0:N4}' -f [double]$analysis.worst_fold_for_best_overall.trade_rate), " +
        "mean_net=$('{0:N6}' -f [double]$analysis.worst_fold_for_best_overall.mean_long_only_net_value_proxy))"
    )
}
Write-Host "saved summary: $($analysis.output_files.summary_md)"
Write-Host "saved threshold_sweep.json: $($analysis.output_files.threshold_sweep_json)"
Write-Host "saved threshold_sweep.csv: $($analysis.output_files.threshold_sweep_csv)"
Write-Host "saved regime_policy_comparison.json: $($analysis.output_files.regime_policy_comparison_json)"
Write-Host "saved regime_policy_comparison.csv: $($analysis.output_files.regime_policy_comparison_csv)"
Write-Host "saved fold_policy_breakdown.csv: $($analysis.output_files.fold_policy_breakdown_csv)"
