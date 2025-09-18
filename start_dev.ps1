# Vihangam Development Environment Setup Script
# This script sets up the Python environment and starts the Django development server

Write-Host "🦅 Starting Vihangam - Disaster Management Drone System" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# Check if virtual environment exists
if (!(Test-Path "venv\Scripts\python.exe")) {
    Write-Host "❌ Virtual environment not found. Please run setup first." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Virtual environment found" -ForegroundColor Green

# Set environment variables to use the virtual environment
$env:PATH = "$PWD\venv\Scripts;$env:PATH"
$env:VIRTUAL_ENV = "$PWD\venv"

Write-Host "🔧 Environment activated" -ForegroundColor Yellow

# Navigate to Django project directory
Set-Location disaster_dashboard

# Start Django development server
Write-Host "🚀 Starting Django development server..." -ForegroundColor Green
Write-Host "📱 Access the application at: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "⏹️  Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "" 

& "$PWD\..\venv\Scripts\python.exe" manage.py runserver