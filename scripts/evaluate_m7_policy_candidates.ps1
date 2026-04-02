param(
    [string]$RunDir,
    [string[]]$Candidate,
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
    "app.training.policy_candidate_analysis",
    "--run-dir",
    $resolvedRunDir,
    "--json"
)
if ($Candidate.Count -gt 0) {
    foreach ($candidateName in $Candidate) {
        $analysisCommand += "--candidate"
        $analysisCommand += $candidateName
    }
}

if ($DryRun) {
    Write-Host "Resolved M7 run dir: $resolvedRunDir"
    Write-Host "Dry run: would run python $($analysisCommand -join ' ')"
    Write-Host "Expected analysis dir: $(Join-Path $resolvedRunDir 'policy_candidate_analysis')"
    exit 0
}

$json = & python @analysisCommand
if ($LASTEXITCODE -ne 0) {
    throw "M7 policy-candidate analysis command failed."
}
$analysis = $json | ConvertFrom-Json

function Format-CandidateLine {
    param([object]$Policy)

    if ($null -eq $Policy) {
        return "none"
    }
    return (
        "$($Policy.policy_name)" +
        " (trade_count=$([int]$Policy.trade_count), " +
        "trade_rate=$('{0:N4}' -f [double]$Policy.trade_rate), " +
        "mean_net=$('{0:N6}' -f [double]$Policy.mean_long_only_net_value_proxy), " +
        "after_cost_positive=$([bool]$Policy.after_cost_positive))"
    )
}

Write-Host ""
Write-Host "M7 policy-candidate evaluation completed"
Write-Host "run dir: $($analysis.run_dir)"
Write-Host "analysis dir: $($analysis.analysis_dir)"
Write-Host "model analyzed: $($analysis.model_name)"
Write-Host "best candidate by mean_long_only_net_value_proxy: $(Format-CandidateLine -Policy $analysis.best_candidate)"
Write-Host "any named candidate after-cost positive: $($analysis.any_after_cost_positive)"
Write-Host "best candidate trade count: $($analysis.best_candidate.trade_count)"
if ($analysis.worst_fold_for_best_candidate -ne $null) {
    Write-Host (
        "weakest fold for best candidate: fold $($analysis.worst_fold_for_best_candidate.fold_index) " +
        "(trade_count=$($analysis.worst_fold_for_best_candidate.trade_count), " +
        "trade_rate=$('{0:N4}' -f [double]$analysis.worst_fold_for_best_candidate.trade_rate), " +
        "mean_net=$('{0:N6}' -f [double]$analysis.worst_fold_for_best_candidate.mean_long_only_net_value_proxy))"
    )
}
if (-not [string]::IsNullOrWhiteSpace($analysis.best_candidate.caution_text)) {
    Write-Host "caution: $($analysis.best_candidate.caution_text)"
}
Write-Host "saved summary: $($analysis.output_files.summary_md)"
Write-Host "saved policy_candidate_summary.json: $($analysis.output_files.policy_candidate_summary_json)"
Write-Host "saved policy_candidate_summary.csv: $($analysis.output_files.policy_candidate_summary_csv)"
Write-Host "saved policy_candidate_fold_breakdown.csv: $($analysis.output_files.policy_candidate_fold_breakdown_csv)"
