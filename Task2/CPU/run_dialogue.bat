@echo off
REM Batch script to activate conda environment and run dialogue system
REM Usage: run_dialogue.bat

echo Activating conda environment: 5112Project...
call conda activate 5112Project

if %ERRORLEVEL% neq 0 (
    echo Error: Failed to activate conda environment '5112Project'
    echo Please make sure conda is installed and the environment exists
    pause
    exit /b 1
)

echo Environment activated successfully!
echo Running dialogue system...

REM Run the Python script
python dialogue_system.py

REM Keep window open if there's an error
if %ERRORLEVEL% neq 0 (
    echo.
    echo Script execution failed. Press any key to exit...
    pause
)