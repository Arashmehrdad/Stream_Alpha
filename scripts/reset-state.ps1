param()

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

& docker compose down --volumes --remove-orphans

$runtimeArtifactDirs = @(
    "artifacts/live",
    "artifacts/paper_trading",
    "artifacts/rationale",
    "artifacts/reliability",
    "artifacts/runtime"
)

foreach ($relativePath in $runtimeArtifactDirs) {
    $targetPath = Join-Path $repoRoot $relativePath
    if (Test-Path $targetPath) {
        Remove-Item -Recurse -Force $targetPath
    }
}

