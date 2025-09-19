@echo off
echo Starting Vihangam Disaster Management Development Server...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Navigate to Django project directory
cd disaster_dashboard

REM Check for any issues
echo Checking Django configuration...
python manage.py check

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Django check failed. Please fix the issues above.
    pause
    exit /b 1
)

echo.
echo Django check passed! Starting development server...
echo.
echo Access your application at: http://127.0.0.1:8000/
echo Press Ctrl+C to stop the server
echo.

REM Start Django development server
python manage.py runserver 0.0.0.0:8000