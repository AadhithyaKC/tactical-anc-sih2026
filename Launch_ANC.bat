@echo off
title Tactical AI-ANC Monitor (SIH26052)
cd /d "E:\SIH_ANC"
call venv\Scripts\activate.bat
python stream_windows.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Script crashed with error code %errorlevel%.
    pause
)
