param(
    [string]$CondaEnv = "5112Project",
    [string]$Image,
    [switch]$Save,
    [switch]$SkipConda
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "工作路径: $scriptDir" -ForegroundColor Gray

if (-not $SkipConda) {
    Write-Host "激活 conda 环境: $CondaEnv" -ForegroundColor Green
    try {
        & conda activate $CondaEnv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to activate conda environment"
        }
    }
    catch {
        Write-Host "[ERROR] 无法激活环境 '$CondaEnv'" -ForegroundColor Red
        Write-Host "若已手动激活可添加 -SkipConda 参数" -ForegroundColor Yellow
        Read-Host "按回车退出"
        exit 1
    }
} else {
    Write-Host "跳过 conda 激活 (已由用户手动处理)" -ForegroundColor Yellow
}

$pythonExe = 'python'
$chatScript = Join-Path $scriptDir 'scripts\interactive_llava_chat.py'
if (-not (Test-Path $chatScript)) {
    Write-Host "[ERROR] 未找到脚本: $chatScript" -ForegroundColor Red
    Read-Host "按回车退出"
    exit 1
}

$arguments = @($chatScript)

if ($Image) {
    $resolvedImage = Resolve-Path $Image -ErrorAction SilentlyContinue
    if (-not $resolvedImage) {
        Write-Host "[ERROR] 指定的图片不存在: $Image" -ForegroundColor Red
        Read-Host "按回车退出"
        exit 1
    }
    $arguments += @('--image', $resolvedImage.Path)
} else {
    Write-Host "未指定图片，将在运行时列出 Task5/data 目录内容" -ForegroundColor Cyan
}

if ($Save) {
    $arguments += '--save'
}

Write-Host "=============================================="
Write-Host "启动 LLaVA 多轮对话" -ForegroundColor Cyan
Write-Host "Python  : $pythonExe"
Write-Host "脚本    : $chatScript"
if ($Image) {
    Write-Host "图片    : $((Resolve-Path $Image).Path)"
}
Write-Host "=============================================="

try {
    & $pythonExe @arguments
    $exit = $LASTEXITCODE
}
catch {
    Write-Host "[ERROR] 执行过程中发生异常: $($_.Exception.Message)" -ForegroundColor Red
    $exit = 1
}

if ($exit -ne 0) {
    Write-Host "[ERROR] 会话执行失败 (exit code $exit)。" -ForegroundColor Red
} else {
    Write-Host "[INFO] 对话结束，欢迎下次使用。" -ForegroundColor Green
}

Read-Host "按回车关闭窗口"
