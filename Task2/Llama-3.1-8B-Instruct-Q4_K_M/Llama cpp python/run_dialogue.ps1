param(
    [string]$CondaEnv = "5112Project",
    [int]$Gpu = 0,
    [switch]$SkipConda
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Working directory: $scriptDir" -ForegroundColor Gray

if (-not $SkipConda) {
    Write-Host "Activating conda environment: $CondaEnv" -ForegroundColor Green
    try {
        & conda activate $CondaEnv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to activate conda environment"
        }
    } catch {
        Write-Host "Error: failed to activate conda environment '$CondaEnv'" -ForegroundColor Red
        Write-Host "Use -SkipConda to bypass activation if you already activated manually." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "Skipping conda activation (per user request)." -ForegroundColor Yellow
}

if (-not $env:LLAMA_CUBLAS) {
    $env:LLAMA_CUBLAS = "1"
    Write-Host "LLAMA_CUBLAS set to 1 for GPU acceleration." -ForegroundColor Cyan
}

if (-not [string]::IsNullOrWhiteSpace($Gpu) -and -not $env:CUDA_VISIBLE_DEVICES) {
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"
    Write-Host "CUDA_VISIBLE_DEVICES set to $Gpu" -ForegroundColor Cyan
}

Write-Host "Running GPU dialogue system..." -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Gray

try {
    & python dialogue_system.py
} catch {
    Write-Host "`nScript execution failed: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
