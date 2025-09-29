param(
    [string],
    [switch]
)

 = Split-Path -Parent .MyCommand.Path
Set-Location 

 = 'D:\Work\Miniconda\envs\5112Project\python.exe'
 = Join-Path  'scripts\interactive_llava_chat.py'

if (-not (Test-Path )) {
    Write-Host "[ERROR] 找不到 Python 可执行文件：" -ForegroundColor Red
    exit 1
}

 = @()

if () {
    if (-not (Test-Path )) {
        Write-Host "[ERROR] 指定的图片不存在：" -ForegroundColor Red
        exit 1
    }
     += @('--image', (Resolve-Path ))
}

if () {
     += '--save'
}

Write-Host "=============================================="
Write-Host "LLaVA 多轮对话启动中……" -ForegroundColor Cyan
Write-Host "Python  : "
Write-Host "脚本    : "
if () {
    Write-Host "图片    : 0"
} else {
    Write-Host "图片    : 运行时从 data 目录选择"
}
Write-Host "=============================================="

&  @arguments

if ( -ne 0) {
    Write-Host "[ERROR] 会话执行失败 (exit code )。" -ForegroundColor Red
    exit 
}

Write-Host "[INFO] 对话结束，欢迎下次使用。" -ForegroundColor Green
