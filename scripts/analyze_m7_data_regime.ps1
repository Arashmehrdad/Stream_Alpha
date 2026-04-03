param(
    [string]$RunDir,
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
        throw "No completed M7 run directories were found under $artifactRoot"
    }
    return $newestRun.FullName
}

function Require-RunArtifactFiles {
    param([string]$ResolvedRunDir)

    $requiredPaths = @(
        (Join-Path $ResolvedRunDir "summary.json"),
        (Join-Path $ResolvedRunDir "oof_predictions.csv"),
        (Join-Path $ResolvedRunDir "fold_metrics.csv"),
        (Join-Path $ResolvedRunDir "dataset_manifest.json")
    )
    foreach ($requiredPath in $requiredPaths) {
        if (-not (Test-Path $requiredPath -PathType Leaf)) {
            throw "Required completed-run diagnostics file is missing: $requiredPath"
        }
    }
}

$resolvedRunDir = Resolve-M7RunDirectory -RequestedRunDir $RunDir
Require-RunArtifactFiles -ResolvedRunDir $resolvedRunDir

$analysisCommand = @(
    "-m",
    "app.training.data_regime_diagnostics",
    "--run-dir",
    $resolvedRunDir,
    "--json"
)

if ($DryRun) {
    Write-Host "Resolved M7 run dir: $resolvedRunDir"
    Write-Host "Dry run: would run python $($analysisCommand -join ' ')"
    Write-Host "Expected analysis dir: $(Join-Path $resolvedRunDir 'data_regime_diagnostics')"
    exit 0
}

$json = & python @analysisCommand
if ($LASTEXITCODE -ne 0) {
    throw "M7 data/regime diagnostics command failed."
}
$analysis = $json | ConvertFrom-Json
$overallLabelRate = [double]$analysis.label_diagnostics.overall.positive_label_rate
$twentyBps = $analysis.opportunity_density.twenty_bps_summary
$weakestFold = $analysis.fold_diagnostics.weakest_fold
$suspiciousFindings = @($analysis.regime_routing.suspicious_findings)
$bestCandidate = $analysis.best_named_candidate

Write-Host ""
Write-Host "M7 data/regime diagnostics completed"
Write-Host "run dir: $($analysis.run_dir)"
Write-Host "analysis dir: $($analysis.analysis_dir)"
Write-Host "overall positive label rate: $('{0:N4}' -f $overallLabelRate)"
Write-Host (
    ">=20 bps opportunities: $([int]$twentyBps.opportunity_count) / " +
    "$([int]$twentyBps.prediction_count) " +
    "(rate=$('{0:N4}' -f [double]$twentyBps.opportunity_rate), " +
    "sparse=$([bool]$analysis.opportunity_density.twenty_bps_sparse))"
)
Write-Host (
    "best named candidate: $($bestCandidate.policy_name) " +
    "(trade_count=$([int]$bestCandidate.trade_count), " +
    "mean_net=$('{0:N6}' -f [double]$bestCandidate.mean_long_only_net_value_proxy), " +
    "after_cost_positive=$([bool]$bestCandidate.after_cost_positive))"
)
Write-Host (
    "weakest fold: fold $([int]$weakestFold.fold_index) " +
    "(best_named_mean_net=$('{0:N6}' -f [double]$weakestFold.best_named_mean_long_only_net_value_proxy), " +
    "default_mean_net=$('{0:N6}' -f [double]$weakestFold.default_mean_long_only_net_value_proxy))"
)
Write-Host (
    "suspicious regime routing: " +
    "$(if ($suspiciousFindings.Count -gt 0) { $suspiciousFindings[0] } else { 'none' })"
)
if ($analysis.warnings.Count -gt 0) {
    Write-Host "warnings: $($analysis.warnings -join ' | ')"
}
Write-Host "saved summary: $($analysis.output_files.summary_md)"
Write-Host "saved diagnostics.json: $($analysis.output_files.diagnostics_json)"
Write-Host "saved fold_diagnostics.csv: $($analysis.output_files.fold_diagnostics_csv)"
