param(
    [string]$ArtifactRoot = ".\artifacts\training\m7",
    [string[]]$Candidate,
    [ValidateRange(1, 1000)]
    [int]$MinRunCount = 1,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

function Resolve-M7ArtifactRoot {
    param([string]$RequestedArtifactRoot)

    if (-not (Test-Path $RequestedArtifactRoot)) {
        throw "Requested M7 artifact root does not exist: $RequestedArtifactRoot"
    }
    $resolvedPath = (Resolve-Path $RequestedArtifactRoot).Path
    if (-not (Test-Path $resolvedPath -PathType Container)) {
        throw "Requested M7 artifact root is not a directory: $resolvedPath"
    }
    return $resolvedPath
}

$resolvedArtifactRoot = Resolve-M7ArtifactRoot -RequestedArtifactRoot $ArtifactRoot
$analysisCommand = @(
    "-m",
    "app.training.multi_run_policy_analysis",
    "--artifact-root",
    $resolvedArtifactRoot,
    "--min-run-count",
    "$MinRunCount",
    "--json"
)
if ($Candidate.Count -gt 0) {
    foreach ($candidateName in $Candidate) {
        $analysisCommand += "--candidate"
        $analysisCommand += $candidateName
    }
}

if ($DryRun) {
    Write-Host "Resolved M7 artifact root: $resolvedArtifactRoot"
    Write-Host "Dry run: would run python $($analysisCommand -join ' ')"
    Write-Host "Expected analysis dir: $(Join-Path $resolvedArtifactRoot '_analysis\policy_candidates')"
    exit 0
}

$json = & python @analysisCommand
if ($LASTEXITCODE -ne 0) {
    throw "Multi-run M7 policy-candidate analysis command failed."
}
$analysis = $json | ConvertFrom-Json
$bestCandidate = $analysis.best_candidate

Write-Host ""
Write-Host "M7 multi-run policy-candidate evaluation completed"
Write-Host "artifact root: $($analysis.artifact_root)"
Write-Host "analysis dir: $($analysis.analysis_dir)"
Write-Host "scanned run directories: $($analysis.scanned_run_count)"
Write-Host "complete runs with required files: $($analysis.complete_run_count)"
Write-Host "analyzable runs: $($analysis.analyzable_run_count)"
Write-Host "skipped runs: $($analysis.skipped_runs.Count)"
Write-Host (
    "best candidate by median net value proxy: $($bestCandidate.policy_name) " +
    "(median_net=$('{0:N6}' -f [double]$bestCandidate.median_net_value_proxy_across_runs), " +
    "positive_run_rate=$('{0:N2}' -f [double]$bestCandidate.positive_run_rate), " +
    "total_trade_count=$([int]$bestCandidate.total_trade_count))"
)
Write-Host "positive_run_rate: $('{0:N2}' -f [double]$bestCandidate.positive_run_rate)"
Write-Host "total_trade_count: $($bestCandidate.total_trade_count)"
Write-Host "evidence still too sparse: $([bool]$bestCandidate.evidence_too_sparse)"
if ($bestCandidate.warnings.Count -gt 0) {
    Write-Host "warnings: $($bestCandidate.warnings -join ' | ')"
}
Write-Host "saved summary: $($analysis.output_files.summary_md)"
Write-Host "saved multi_run_policy_summary.json: $($analysis.output_files.multi_run_policy_summary_json)"
Write-Host "saved multi_run_policy_summary.csv: $($analysis.output_files.multi_run_policy_summary_csv)"
Write-Host "saved multi_run_policy_run_breakdown.csv: $($analysis.output_files.multi_run_policy_run_breakdown_csv)"
