param(
    [switch]$RemoveOrphans = $true
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$composeArgs = @("compose", "down")
if ($RemoveOrphans) {
    $composeArgs += "--remove-orphans"
}

& docker @composeArgs

