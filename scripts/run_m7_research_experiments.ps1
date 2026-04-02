param(
    [string]$ConfigDirectory = ".\configs",
    [string]$ArtifactRoot = ".\artifacts\training\m7",
    [ValidateRange(5, 120)]
    [int]$StatusSeconds = 15,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

function Resolve-ExistingRunDirectories {
    param([string]$ResolvedArtifactRoot)

    if (-not (Test-Path $ResolvedArtifactRoot -PathType Container)) {
        return @()
    }
    return @(
        Get-ChildItem -Path $ResolvedArtifactRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { -not $_.Name.StartsWith("_") } |
        Select-Object -ExpandProperty FullName
    )
}

function Resolve-NewRunDirectory {
    param(
        [string]$ResolvedArtifactRoot,
        [string[]]$ExistingRunDirectories
    )

    $candidate = @(
        Get-ChildItem -Path $ResolvedArtifactRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { -not $_.Name.StartsWith("_") } |
        Where-Object { $_.FullName -notin $ExistingRunDirectories } |
        Sort-Object LastWriteTimeUtc -Descending
    ) | Select-Object -First 1
    if ($null -ne $candidate) {
        return $candidate.FullName
    }
    throw "Could not identify a new completed M7 run directory under $ResolvedArtifactRoot"
}

function Get-ResearchConfigurations {
    param([string]$ResolvedConfigDirectory)

    $json = & python -m app.training.research_experiments --config-dir $ResolvedConfigDirectory --list-configs --json
    if ($LASTEXITCODE -ne 0) {
        throw "Research config discovery failed."
    }
    $payload = $json | ConvertFrom-Json
    if ($null -eq $payload.configurations -or $payload.configurations.Count -eq 0) {
        throw "No bounded M7 research configs were discovered."
    }
    return @($payload.configurations)
}

$resolvedConfigDirectory = (Resolve-Path $ConfigDirectory).Path
$resolvedArtifactRoot = if (Test-Path $ArtifactRoot) {
    (Resolve-Path $ArtifactRoot).Path
}
else {
    $createdArtifactRoot = Join-Path $repoRoot $ArtifactRoot
    New-Item -ItemType Directory -Force -Path $createdArtifactRoot | Out-Null
    (Resolve-Path $createdArtifactRoot).Path
}
$researchConfigs = Get-ResearchConfigurations -ResolvedConfigDirectory $resolvedConfigDirectory

if ($DryRun) {
    Write-Host "Resolved research config directory: $resolvedConfigDirectory"
    Write-Host "Resolved M7 artifact root: $resolvedArtifactRoot"
    Write-Host "Discovered bounded research configs:"
    foreach ($config in $researchConfigs) {
        Write-Host "  - $($config.config_name): $($config.config_path)"
    }
    foreach ($config in $researchConfigs) {
        Write-Host ""
        Write-Host "Dry run: would run .\scripts\start_m7_training.ps1 -ConfigPath $($config.config_path) -StatusSeconds $StatusSeconds"
        Write-Host "Dry run: would then run .\scripts\evaluate_m7_policy_candidates.ps1 -RunDir <new_run_for_$($config.config_name)>"
    }
    Write-Host ""
    Write-Host "Dry run: would then write summary under $(Join-Path $resolvedArtifactRoot '_analysis\research_experiments')"
    exit 0
}

$experimentArguments = @()
$startScriptPath = Join-Path $repoRoot "scripts\start_m7_training.ps1"
$candidateEvalScriptPath = Join-Path $repoRoot "scripts\evaluate_m7_policy_candidates.ps1"

foreach ($config in $researchConfigs) {
    Write-Host ""
    Write-Host "Running bounded M7 research config: $($config.config_name)"
    Write-Host "config path: $($config.config_path)"

    $existingRunDirectories = Resolve-ExistingRunDirectories -ResolvedArtifactRoot $resolvedArtifactRoot
    & $startScriptPath -ConfigPath $config.config_path -StatusSeconds $StatusSeconds
    $newRunDirectory = Resolve-NewRunDirectory `
        -ResolvedArtifactRoot $resolvedArtifactRoot `
        -ExistingRunDirectories $existingRunDirectories

    & $candidateEvalScriptPath -RunDir $newRunDirectory
    $experimentArguments += "$($config.config_name)::$($config.config_path)::$newRunDirectory"
}

$summaryCommand = @(
    "-m",
    "app.training.research_experiments",
    "--artifact-root",
    $resolvedArtifactRoot,
    "--json"
)
foreach ($experimentArgument in $experimentArguments) {
    $summaryCommand += "--experiment"
    $summaryCommand += $experimentArgument
}

$summaryJson = & python @summaryCommand
if ($LASTEXITCODE -ne 0) {
    throw "M7 research experiment summary command failed."
}
$summary = $summaryJson | ConvertFrom-Json
$bestExperiment = $summary.best_experiment

$tableRows = @(
    foreach ($experiment in $summary.experiments) {
        [pscustomobject]@{
            config_name = $experiment.config_name
            run_id = $experiment.run_id
            winner_model = $experiment.winner_model_name
            winner_after_cost_positive = $experiment.winner_after_cost_positive
            meets_acceptance_target = $experiment.meets_acceptance_target
            best_named_policy = $experiment.best_policy_name
            best_policy_net = ('{0:N6}' -f [double]$experiment.best_policy_mean_long_only_net_value_proxy)
            best_policy_trade_count = [int]$experiment.best_policy_trade_count
            best_policy_positive = $experiment.best_policy_after_cost_positive
        }
    }
)

Write-Host ""
Write-Host "M7 research experiments completed"
Write-Host "best experiment: $($bestExperiment.config_name) -> $($bestExperiment.run_id)"
Write-Host "best policy: $($bestExperiment.best_policy_name)"
Write-Host "best policy mean net proxy: $('{0:N6}' -f [double]$bestExperiment.best_policy_mean_long_only_net_value_proxy)"
Write-Host "best policy trade_count: $($bestExperiment.best_policy_trade_count)"
Write-Host ""
Write-Host ($tableRows | Format-Table -AutoSize | Out-String).TrimEnd()
Write-Host ""
Write-Host "saved summary: $($summary.output_files.summary_md)"
Write-Host "saved experiment_summary.json: $($summary.output_files.experiment_summary_json)"
Write-Host "saved experiment_summary.csv: $($summary.output_files.experiment_summary_csv)"
