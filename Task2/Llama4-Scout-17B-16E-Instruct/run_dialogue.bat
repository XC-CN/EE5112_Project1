@echo off
REM Llama-4-Scout Dialogue System - Windows Batch Script
REM Requires 5112Project conda environment with llama-cpp-python[cuda]

echo Starting Llama-4-Scout Dialogue System...
echo GPU: RTX 5080 16GB Optimized
echo.

REM Activate conda environment and run
conda activate 5112Project
if %errorlevel% neq 0 (
    echo Error: Failed to activate 5112Project conda environment
    echo Please ensure the environment exists and try again
    pause
    exit /b 1
)

python dialogue_system.py %*

REM Keep window open on error
if %errorlevel% neq 0 (
    echo.
    echo Dialogue system exited with error code %errorlevel%
    pause
)

echo Llama-4-Scout session ended.
pause