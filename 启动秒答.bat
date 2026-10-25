@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Vocab Solver

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python38\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python38\python.exe"
if not defined PY (
  echo [X] Python not found. Install Python 3.8+ and tick "Add Python to PATH".
  pause
  exit /b 1
)

"%PY%" -c "import rapidocr_onnxruntime, keyboard, PIL, numpy" >nul 2>nul
if errorlevel 1 (
  echo [!] Missing dependencies, installing now. This happens once and may take a few minutes...
  "%PY%" -m pip install rapidocr-onnxruntime keyboard pillow numpy
  if errorlevel 1 (
    echo [X] Install failed. Check your network and try again.
    pause
    exit /b 1
  )
)

echo.
echo   ============================================
echo     Vocab Solver
echo     F7 = select region (do this first)
echo     F9 = solve
echo     F8 = quit
echo   ============================================
echo.
echo   If F9 does nothing, close this window and run
echo   this file as administrator.
echo.

"%PY%" -u "%~dp0vocab_solver.py"

echo.
echo Exited.
pause
