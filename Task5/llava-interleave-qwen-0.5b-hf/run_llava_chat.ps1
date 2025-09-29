param(
    [string]$CondaEnv = "5112Project",
    [string]$Image,
    [switch]$Save,
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
    }
    catch {
        Write-Host "[ERROR] Unable to activate environment '$CondaEnv'" -ForegroundColor Red
        Write-Host "If already activated manually, add -SkipConda parameter" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "Skipping conda activation (handled manually by user)" -ForegroundColor Yellow
}

$pythonExe = 'python'
$chatScript = Join-Path $scriptDir 'scripts\interactive_llava_chat.py'
if (-not (Test-Path $chatScript)) {
    Write-Host "[ERROR] Script not found: $chatScript" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$arguments = @($chatScript)

if ($Image) {
    $resolvedImage = Resolve-Path $Image -ErrorAction SilentlyContinue
    if (-not $resolvedImage) {
        Write-Host "[ERROR] Specified image does not exist: $Image" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    $arguments += @('--image', $resolvedImage.Path)
} else {
    Write-Host "No image specified, will list Task5/data directory contents at runtime" -ForegroundColor Cyan
}

if ($Save) {
    $arguments += '--save'
}

Write-Host "=============================================="
Write-Host "Starting LLaVA multi-turn conversation" -ForegroundColor Cyan
Write-Host "Python  : $pythonExe"
Write-Host "Script  : $chatScript"
if ($Image) {
    Write-Host "Image   : $((Resolve-Path $Image).Path)"
}
Write-Host "=============================================="

try {
    & $pythonExe @arguments
    $exit = $LASTEXITCODE
}
catch {
    Write-Host "[ERROR] Exception occurred during execution: $($_.Exception.Message)" -ForegroundColor Red
    $exit = 1
}

if ($exit -ne 0) {
    Write-Host "[ERROR] Session execution failed (exit code $exit)." -ForegroundColor Red
} else {
    Write-Host "[INFO] Conversation ended, welcome back next time." -ForegroundColor Green
}

Read-Host "Press Enter to close window"
