param(
    [ValidateSet("dev", "paper", "shadow", "live")]
    [string]$Profile = "dev"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not (Test-Path ".env")) {
    throw "Missing .env. Copy .env.example to .env before starting the stack."
}

$startupReportPath = "artifacts/runtime/startup_report.json"
$tradingConfigPath = switch ($Profile) {
    "paper" { "configs/paper_trading.paper.yaml" }
    "shadow" { "configs/paper_trading.shadow.yaml" }
    "live" { "configs/paper_trading.live.yaml" }
    default { "configs/paper_trading.yaml" }
}

New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "artifacts/runtime") | Out-Null

$previousRuntimeProfile = $env:STREAMALPHA_RUNTIME_PROFILE
$previousTradingConfigPath = $env:STREAMALPHA_TRADING_CONFIG_PATH
$previousStartupReportPath = $env:STREAMALPHA_STARTUP_REPORT_PATH

$env:STREAMALPHA_RUNTIME_PROFILE = $Profile
$env:STREAMALPHA_TRADING_CONFIG_PATH = $tradingConfigPath
$env:STREAMALPHA_STARTUP_REPORT_PATH = $startupReportPath

$composeArgs = @("compose", "--profile", $Profile, "--env-file", ".env")
if (Test-Path ".env.secrets") {
    $composeArgs += @("--env-file", ".env.secrets")
}
$composeArgs += @("up", "-d", "--build")

try {
    & docker @composeArgs
}
finally {
    if ($null -eq $previousRuntimeProfile) {
        Remove-Item Env:STREAMALPHA_RUNTIME_PROFILE -ErrorAction SilentlyContinue
    }
    else {
        $env:STREAMALPHA_RUNTIME_PROFILE = $previousRuntimeProfile
    }
    if ($null -eq $previousTradingConfigPath) {
        Remove-Item Env:STREAMALPHA_TRADING_CONFIG_PATH -ErrorAction SilentlyContinue
    }
    else {
        $env:STREAMALPHA_TRADING_CONFIG_PATH = $previousTradingConfigPath
    }
    if ($null -eq $previousStartupReportPath) {
        Remove-Item Env:STREAMALPHA_STARTUP_REPORT_PATH -ErrorAction SilentlyContinue
    }
    else {
        $env:STREAMALPHA_STARTUP_REPORT_PATH = $previousStartupReportPath
    }
}

