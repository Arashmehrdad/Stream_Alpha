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
    "app.training.policy_replay_analysis",
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
    Write-Host "Expected analysis dir: $(Join-Path $resolvedRunDir 'policy_replay_analysis')"
    exit 0
}

$json = & python @analysisCommand
if ($LASTEXITCODE -ne 0) {
    throw "M7 policy replay analysis command failed."
}
$analysis = $json | ConvertFrom-Json
$bestCandidate = $analysis.best_candidate

Write-Host ""
Write-Host "M7 policy replay analysis completed"
Write-Host "run dir: $($analysis.run_dir)"
Write-Host "analysis dir: $($analysis.analysis_dir)"
Write-Host "model analyzed: $($analysis.model_name)"
Write-Host (
    "best candidate by cumulative net proxy: $($bestCandidate.policy_name) " +
    "(cumulative_net=$('{0:N6}' -f [double]$bestCandidate.cumulative_net_proxy), " +
    "trade_count=$([int]$bestCandidate.trade_count), " +
    "max_drawdown=$('{0:N6}' -f [double]$bestCandidate.max_drawdown_proxy))"
)
Write-Host "evidence still thin: $([bool]$bestCandidate.evidence_still_thin)"
if ($bestCandidate.warnings.Count -gt 0) {
    Write-Host "warnings: $($bestCandidate.warnings -join ' | ')"
}
Write-Host "saved summary: $($analysis.output_files.summary_md)"
Write-Host "saved replay_summary.json: $($analysis.output_files.replay_summary_json)"
Write-Host "saved replay_summary.csv: $($analysis.output_files.replay_summary_csv)"
Write-Host "saved replay_trade_ledger.csv: $($analysis.output_files.replay_trade_ledger_csv)"
