@echo off
echo 🦅 Starting Vihangam - Disaster Management Drone System
echo =================================================

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo ❌ Virtual environment not found. Please run setup first.
    pause
    exit /b 1
)

echo ✅ Virtual environment found

REM Set environment variables
set "PATH=%CD%\venv\Scripts;%PATH%"
set "VIRTUAL_ENV=%CD%\venv"

echo 🔧 Environment activated

REM Navigate to Django project directory
cd disaster_dashboard

echo 🚀 Starting Django development server...
echo 📱 Access the application at: http://127.0.0.1:8000
echo ⏹️  Press Ctrl+C to stop the server
echo.

REM Start Django development server
"%CD%\..\venv\Scripts\python.exe" manage.py runserver