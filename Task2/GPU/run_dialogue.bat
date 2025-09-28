@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"

echo Working directory: %SCRIPT_DIR%

if "%SKIP_CONDA%"=="1" (
    echo Skipping conda activation (SKIP_CONDA=1).
) else (
    set "CONDA_ENV=%CONDA_ENV%"
    if "%CONDA_ENV%"=="" set "CONDA_ENV=5112Project"
    echo Activating conda environment: %CONDA_ENV%
    call conda activate %CONDA_ENV%
    if errorlevel 1 (
        echo Failed to activate conda environment "%CONDA_ENV%".
        echo Set SKIP_CONDA=1 if the environment is already active.
        pause
        exit /b 1
    )
)

if "%LLAMA_CUBLAS%"=="" (
    set LLAMA_CUBLAS=1
    echo LLAMA_CUBLAS set to 1 (enable GPU backend).
)

if "%CUDA_VISIBLE_DEVICES%"=="" (
    set "CUDA_VISIBLE_DEVICES=%GPU_ID%"
    if "%CUDA_VISIBLE_DEVICES%"=="" set CUDA_VISIBLE_DEVICES=0
    echo CUDA_VISIBLE_DEVICES set to %CUDA_VISIBLE_DEVICES%.
)

echo ============================================
echo Launching GPU dialogue system...
echo ============================================

python dialogue_system.py
if errorlevel 1 (
    echo Python script exited with error %errorlevel%.
    pause
)

popd
endlocal
