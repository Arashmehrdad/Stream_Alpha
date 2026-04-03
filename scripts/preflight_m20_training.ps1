param(
    [string]$ConfigPath = ".\configs\training.m20.json",
    [switch]$RequireGpu,
    [switch]$DryRun
)

$resolvedConfigPath = (Resolve-Path $ConfigPath).Path
$commandParts = @(
    "python",
    "-m",
    "app.training.preflight_m20",
    "--config",
    $resolvedConfigPath
)
if ($RequireGpu) {
    $commandParts += "--require-gpu"
}
$commandString = ($commandParts | ForEach-Object {
    if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
}) -join ' '

if ($DryRun) {
    Write-Host "M20 specialist preflight dry run"
    Write-Host "config path: $resolvedConfigPath"
    if ($RequireGpu) {
        Write-Host "gpu required: yes"
    } else {
        Write-Host "gpu required: no"
    }
    Write-Host "command: $commandString"
    exit 0
}

$pythonArgs = @(
    "-m",
    "app.training.preflight_m20",
    "--config",
    $resolvedConfigPath
)
if ($RequireGpu) {
    $pythonArgs += "--require-gpu"
}

$jsonOutput = & python @pythonArgs 2>&1
$exitCode = $LASTEXITCODE
$jsonText = ($jsonOutput | Out-String).Trim()
$jsonLines = @($jsonText -split "`r?`n" | Where-Object { $_.Trim() -ne "" })
$jsonCandidate = if ($jsonLines.Count -gt 0) {
    $jsonLines[-1]
} else {
    $jsonText
}

try {
    $payload = $jsonText | ConvertFrom-Json
    if ($payload -is [string]) {
        $payload = $payload | ConvertFrom-Json
    }
} catch {
    try {
        $payload = $jsonCandidate | ConvertFrom-Json
        if ($payload -is [string]) {
            $payload = $payload | ConvertFrom-Json
        }
    } catch {
        $jsonMatch = [regex]::Match($jsonText, '\{[\s\S]*\}')
        if ($jsonMatch.Success) {
            try {
                $payload = $jsonMatch.Value | ConvertFrom-Json
                if ($payload -is [string]) {
                    $payload = $payload | ConvertFrom-Json
                }
            } catch {
                Write-Host $jsonText
                throw "M20 specialist preflight returned unreadable output."
            }
        } else {
            Write-Host $jsonText
            throw "M20 specialist preflight returned unreadable output."
        }
    }
}

$configuredModels = @($payload.configured_models) -join ", "
$registrySpecialists = [int]$payload.registry.specialist_entry_count
$cudaAvailable = [bool]$payload.runtime.torch.cuda_available
$deviceLabel = if ($cudaAvailable) { "GPU" } else { "CPU" }

Write-Host "M20 specialist preflight"
Write-Host "config path: $($payload.config_path)"
Write-Host "training source table: $($payload.source_table)"
Write-Host "symbols: $(@($payload.symbols) -join ', ')"
Write-Host "configured models: $configuredModels"
Write-Host "lightning installed: $($payload.runtime.lightning.installed)"
Write-Host "neuralforecast installed: $($payload.runtime.neuralforecast.installed)"
Write-Host "torch version: $($payload.runtime.torch.version)"
Write-Host "preferred execution device: $($payload.preferred_execution_device.ToUpperInvariant())"
Write-Host "cuda available: $cudaAvailable"
if ($cudaAvailable -and @($payload.runtime.torch.device_names).Count -gt 0) {
    Write-Host "visible GPU devices: $(@($payload.runtime.torch.device_names) -join ', ')"
}
Write-Host "current registry specialist candidates: $registrySpecialists"
Write-Host "ready for manual training: $($payload.ready_for_manual_training)"

if (@($payload.warnings).Count -gt 0) {
    Write-Host "warnings:"
    foreach ($warning in @($payload.warnings)) {
        Write-Host " - $warning"
    }
}

if (@($payload.blockers).Count -gt 0) {
    Write-Host "blockers:"
    foreach ($blocker in @($payload.blockers)) {
        Write-Host " - $blocker"
    }
    throw "M20 specialist preflight failed."
}

Write-Host "manual run mode: $deviceLabel"
