@echo off
setlocal

cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set "PYTHON_BIN=venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

if not defined HOST set "HOST=0.0.0.0"
if not defined PORT set "PORT=5000"

echo Hosting Stock Price Forecasting System on %HOST%:%PORT%
echo Open this on your laptop: http://localhost:%PORT%
"%PYTHON_BIN%" serve_waitress.py
