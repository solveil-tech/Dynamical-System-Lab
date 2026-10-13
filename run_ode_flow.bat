@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"

"%PYTHON%" -c "import numpy, PySide6, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Required Python packages are missing.
    echo Install them with:
    echo   "%PYTHON%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%PYTHON%" "%~dp0ode_flow.py"
if errorlevel 1 pause
