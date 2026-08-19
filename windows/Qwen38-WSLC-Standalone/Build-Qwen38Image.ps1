[CmdletBinding()]
param(
    [string]$Image = "qwen38-b70-friendly:2026.08.18",
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
. (Join-Path $Root "Common.ps1")
$PatchDir = Join-Path $Root "patches"
New-Item -ItemType Directory -Force -Path $PatchDir | Out-Null

Write-Host "[Build] Preparing two pinned compatibility patches..."

$patches = @(
    @{
        Name = "patch_mtp_nightly.py"
        Uri = "https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/5c6b6b1/patches/patch_mtp_nightly.py"
        Sha256 = "4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14"
    },
    @{
        Name = "patch_mtp_boundary.py"
        Uri = "https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/5c6b6b1/patches/patch_mtp_boundary.py"
        Sha256 = "41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50"
    }
)

foreach ($item in $patches) {
    Write-Host "[Build] Downloading $($item.Name)..."
    $destination = Join-Path $PatchDir $item.Name
    $download = "$destination.download"
    Invoke-WebRequest -UseBasicParsing -Uri $item.Uri -OutFile $download

    # Windows PowerShell may convert downloaded text to CRLF. The cookbook
    # publishes hashes for the original LF bytes, and the files are copied into
    # a Linux image, so normalize deterministically before checking the digest.
    $content = [System.IO.File]::ReadAllText($download)
    $content = $content.Replace("`r`n", "`n").Replace("`r", "`n")
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    if (Test-Path -LiteralPath $destination) {
        Remove-Item -Force -LiteralPath $destination
    }
    [System.IO.File]::WriteAllText($destination, $content, $utf8NoBom)
    Remove-Item -Force -LiteralPath $download

    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
    if ($actual -ne $item.Sha256) {
        throw "Hash mismatch for $($item.Name). Expected $($item.Sha256), received $actual"
    }
    Write-Host "[Build] Verified $($item.Name)" -ForegroundColor Green
}

Write-Host "[Build] Asking WSLC to build image $Image"
Write-Host "[Build] The first build may download a large base image. Cached layers will be reused."
$args = @("build", "--pull", "--tag", $Image, "--file", (Join-Path $Root "Dockerfile"))
if ($NoCache) { $args += "--no-cache" }
$args += $Root
& wslc.exe @args
if ($LASTEXITCODE -ne 0) { throw "WSLC image build failed with exit code $LASTEXITCODE" }

Write-Host "[Build] Image ready: $Image" -ForegroundColor Green
