@echo off
chcp 65001 >nul
echo ========================================
echo TrajRL Web Demo 启动脚本
echo ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.x
    pause
    exit /b 1
)

:: 检查依赖
echo [1/3] 检查依赖...
python -c "import numpy, sklearn" >nul 2>&1
if errorlevel 1 (
    echo [2/3] 安装依赖: numpy, scikit-learn...
    pip install numpy scikit-learn
) else (
    echo [2/3] 依赖已安装 ✓
)

:: 启动服务
echo [3/3] 启动服务于 http://localhost:8601 ...
echo 按 Ctrl+C 停止服务
echo ========================================
python web_server.py --port 8601

pause
