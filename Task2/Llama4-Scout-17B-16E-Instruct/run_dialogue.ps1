#!/usr/bin/env pwsh
# Llama-4-Scout Dialogue System - PowerShell Script
# Requires 5112Project conda environment with llama-cpp-python[cuda]

Write-Host "Starting Llama-4-Scout Dialogue System..." -ForegroundColor Green
Write-Host "GPU: RTX 5080 16GB Optimized" -ForegroundColor Cyan
Write-Host ""

try {
    # Activate conda environment
    conda activate 5112Project
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate 5112Project conda environment"
    }
    
    # Run the dialogue system
    python dialogue_system.py $args
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "" 
        Write-Host "Dialogue system exited with error code $LASTEXITCODE" -ForegroundColor Red
    }
}
catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please ensure the 5112Project conda environment exists and try again" -ForegroundColor Yellow
}
finally {
    Write-Host "Llama-4-Scout session ended." -ForegroundColor Green
    Read-Host "Press Enter to continue..."
}