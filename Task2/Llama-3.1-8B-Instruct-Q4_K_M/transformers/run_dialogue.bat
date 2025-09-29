@echo off
setlocal

REM Navigate to directory containing this script
echo Changing directory to script location...
pushd "%~dp0"
if %ERRORLEVEL% neq 0 (
    echo Error: failed to locate script directory.
    pause
    exit /b 1
)

echo Activating conda environment: 5112Project
call conda activate 5112Project
if %ERRORLEVEL% neq 0 (
    echo Error: failed to activate conda environment '5112Project'.
    echo Please make sure conda is installed and the environment exists.
    pause
    popd
    exit /b 1
)

echo Environment activated successfully.
set SCRIPT=dialogue_system.py

REM Forward any additional parameters to the Python script
echo Running %SCRIPT% %*
python "%SCRIPT%" %*
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% neq 0 (
    echo Script execution failed with exit code %EXIT_CODE%.
    echo Press any key to close...
    pause >nul
)

REM Restore original directory
popd
endlocal
exit /b %EXIT_CODE%
