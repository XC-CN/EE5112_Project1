# PowerShell script to activate conda environment and run dialogue system
# Usage: .\run_dialogue.ps1

param(
    [string]$CondaEnv = "5112Project"
)

Write-Host "Activating conda environment: $CondaEnv..." -ForegroundColor Green

# Try to activate conda environment
try {
    & conda activate $CondaEnv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate conda environment"
    }
} catch {
    Write-Host "Error: Failed to activate conda environment '$CondaEnv'" -ForegroundColor Red
    Write-Host "Please make sure conda is installed and the environment exists" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Environment activated successfully!" -ForegroundColor Green
Write-Host "Running dialogue system..." -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Gray

# Run the Python script
try {
    & python dialogue_system.py
} catch {
    Write-Host "`nScript execution failed: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}