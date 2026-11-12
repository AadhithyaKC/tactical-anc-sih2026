@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python stream_windows.py
pause
