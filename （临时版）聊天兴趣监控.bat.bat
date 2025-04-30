@echo off
CHCP 65001 > nul
setlocal enabledelayedexpansion

REM 查找venv虚拟环境
set "venv_path=%~dp0venv\Scripts\activate.bat"
if not exist "%venv_path%" (
    echo 错误: 未找到虚拟环境，请确保venv目录存在
    pause
    exit /b 1
)

REM 激活虚拟环境
call "%venv_path%"
if %ERRORLEVEL% neq 0 (
    echo 错误: 虚拟环境激活失败
    pause
    exit /b 1
)

echo 虚拟环境已激活，正在启动 GUI...

REM 运行 Python 脚本
python scripts/interest_monitor_gui.py

pause 