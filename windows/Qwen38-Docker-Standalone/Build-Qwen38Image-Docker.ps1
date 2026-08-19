[CmdletBinding()]
param(
    [string]$Image = "qwen38-b70-docker:2026.08.18",
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
if (-not (Test-Path -LiteralPath $docker)) { $docker = (Get-Command docker.exe -ErrorAction Stop).Source }
. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-QwenDocker -DockerPath $docker

$expected = @{
    "patch_mtp_nightly.py"  = "4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14"
    "patch_mtp_boundary.py" = "41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50"
}
Write-Host "[Docker build] Verifying pinned compatibility patches..."
foreach ($name in $expected.Keys) {
    $path = Join-Path $PSScriptRoot "patches\$name"
    if (-not (Test-Path -LiteralPath $path)) { throw "Required pinned patch is missing: $path" }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    if ($actual -ne $expected[$name]) { throw "Hash mismatch for $name. Expected $($expected[$name]), received $actual" }
    Write-Host "[Docker build] Verified $name" -ForegroundColor Green
}

Write-Host "[Docker build] Building $Image"
Write-Host "[Docker build] The first build downloads several large layers; later builds reuse Docker's cache."
$args = @("build", "--tag", $Image, "--file", (Join-Path $PSScriptRoot "Dockerfile"))
if ($NoCache) { $args += "--no-cache" }
$args += $PSScriptRoot
& $docker @args
if ($LASTEXITCODE -ne 0) { throw "Docker image build failed with exit code $LASTEXITCODE." }
Write-Host "[Docker build] Image ready: $Image" -ForegroundColor Green
