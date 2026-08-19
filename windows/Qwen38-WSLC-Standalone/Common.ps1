function Write-QwenBanner {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 68) -ForegroundColor Cyan
    Write-Host ("  " + $Title) -ForegroundColor Cyan
    Write-Host ("=" * 68) -ForegroundColor Cyan
    Write-Host ""
}

function Write-QwenStep {
    param([string]$Title, [string]$Detail)
    Write-Host ""
    Write-Host ("-" * 68) -ForegroundColor DarkCyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("-" * 68) -ForegroundColor DarkCyan
    if ($Detail) { Write-Host $Detail }
    Write-Host ""
}

function Write-HuggingFaceTokenStatus {
    param([string]$Token = $env:HF_TOKEN, [switch]$RequireToken)
    if ($Token) {
        Write-Host "[Hugging Face] HF_TOKEN detected." -ForegroundColor Green
        Write-Host "Authenticated downloads and higher rate limits are available."
        return
    }

    Write-Host "[Hugging Face] No HF_TOKEN was detected." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The model is public, so setup can continue, but Hugging Face may"
    Write-Host "apply lower rate limits and the download could take longer."
    Write-Host ""
    Write-Host 'To use a read-only token in this PowerShell session:'
    Write-Host '  $env:HF_TOKEN = "hf_your_token_here"' -ForegroundColor White
    Write-Host ""
    if ($RequireToken) {
        throw "HF_TOKEN is required because -RequireHuggingFaceToken was specified."
    }
    Write-Host "Press Ctrl+C now if you would prefer to set a token first."
    Write-Host "Continuing without authentication in 10 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}
