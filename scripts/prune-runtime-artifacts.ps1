param(
    [int]$RetentionDays = 14
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cutoff = (Get-Date).AddDays(-1 * $RetentionDays)

$runtimeArtifactDirs = @(
    "artifacts/live",
    "artifacts/paper_trading",
    "artifacts/rationale",
    "artifacts/reliability",
    "artifacts/runtime"
)

foreach ($relativePath in $runtimeArtifactDirs) {
    $targetPath = Join-Path $repoRoot $relativePath
    if (-not (Test-Path $targetPath)) {
        continue
    }
    Get-ChildItem -Path $targetPath -Recurse -File |
        Where-Object { $_.LastWriteTimeUtc -lt $cutoff.ToUniversalTime() } |
        Remove-Item -Force
}
