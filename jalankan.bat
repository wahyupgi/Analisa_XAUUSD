@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    echo Menjalankan Analisis XAUUSD menggunakan Virtual Environment (.venv)...
    ".venv\Scripts\python.exe" xauusd1.py
) else (
    echo Menjalankan Analisis XAUUSD menggunakan Python sistem...
    python xauusd1.py
)
pause


