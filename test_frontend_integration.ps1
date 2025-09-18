# 🚁 Vihangam YOLO Frontend Integration Test Script

Write-Host "🚁 Vihangam YOLO Frontend Integration Test" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green

$pythonExe = "venv\Scripts\python.exe"

# 1. Check if Django dashboard exists
Write-Host "`n1️⃣ DJANGO PROJECT CHECK" -ForegroundColor Cyan
Write-Host "-" * 30

if (Test-Path "disaster_dashboard") {
    Write-Host "✅ Django project found: disaster_dashboard" -ForegroundColor Green
    
    # Check key frontend files
    $frontendFile = "disaster_dashboard\templates\detection\index.html"
    if (Test-Path $frontendFile) {
        $fileSize = [math]::Round((Get-Item $frontendFile).Length / 1KB, 1)
        Write-Host "✅ Frontend template found: $frontendFile ($fileSize KB)" -ForegroundColor Green
    } else {
        Write-Host "❌ Frontend template missing: $frontendFile" -ForegroundColor Red
    }
    
    # Check consumer file
    $consumerFile = "disaster_dashboard\apps\detection\consumers.py"
    if (Test-Path $consumerFile) {
        Write-Host "✅ WebSocket consumer found: $consumerFile" -ForegroundColor Green
    } else {
        Write-Host "❌ WebSocket consumer missing: $consumerFile" -ForegroundColor Red
    }
    
} else {
    Write-Host "❌ Django project not found: disaster_dashboard" -ForegroundColor Red
    Write-Host "⚠️  Make sure you're in the correct directory" -ForegroundColor Yellow
    exit 1
}

# 2. Check Python dependencies for Django
Write-Host "`n2️⃣ DJANGO DEPENDENCIES CHECK" -ForegroundColor Cyan
Write-Host "-" * 30

$djangoDeps = @("django", "channels", "ultralytics", "cv2")
$allDepsOk = $true

foreach ($dep in $djangoDeps) {
    Write-Host "Checking $dep... " -NoNewline
    try {
        $result = & $pythonExe -c "import $dep; print('OK')" 2>$null
        if ($result -eq "OK") {
            Write-Host "✅" -ForegroundColor Green
        } else {
            Write-Host "❌" -ForegroundColor Red
            $allDepsOk = $false
        }
    } catch {
        Write-Host "❌" -ForegroundColor Red
        $allDepsOk = $false
    }
}

if (-not $allDepsOk) {
    Write-Host "⚠️  Some dependencies missing. Install with:" -ForegroundColor Yellow
    Write-Host "pip install django channels django-channels" -ForegroundColor White
}

# 3. Check YOLO model availability
Write-Host "`n3️⃣ YOLO MODEL CHECK" -ForegroundColor Cyan
Write-Host "-" * 30

$modelPath = "runs/detect/disaster_demo_20250918_201832/weights/best.pt"
if (Test-Path $modelPath) {
    $modelSize = [math]::Round((Get-Item $modelPath).Length / 1MB, 2)
    Write-Host "✅ Vihangam model found: $modelSize MB" -ForegroundColor Green
} else {
    Write-Host "❌ Vihangam model missing: $modelPath" -ForegroundColor Red
    Write-Host "⚠️  Frontend will show error without model" -ForegroundColor Yellow
}

# 4. Check Django project structure
Write-Host "`n4️⃣ DJANGO PROJECT STRUCTURE" -ForegroundColor Cyan
Write-Host "-" * 30

$djangoFiles = @(
    "disaster_dashboard\manage.py",
    "disaster_dashboard\disaster_dashboard\settings.py",
    "disaster_dashboard\apps\detection\urls.py",
    "disaster_dashboard\apps\detection\views.py"
)

foreach ($file in $djangoFiles) {
    if (Test-Path $file) {
        Write-Host "✅ $file" -ForegroundColor Green
    } else {
        Write-Host "❌ $file" -ForegroundColor Red
    }
}

# 5. Test Django server startup (dry run)
Write-Host "`n5️⃣ DJANGO SERVER TEST" -ForegroundColor Cyan
Write-Host "-" * 30

try {
    Write-Host "Testing Django configuration..." -ForegroundColor Yellow
    Push-Location "disaster_dashboard"
    
    $djangoCheck = & ..\$pythonExe manage.py check --verbosity 0 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Django configuration is valid" -ForegroundColor Green
    } else {
        Write-Host "❌ Django configuration has issues:" -ForegroundColor Red
        Write-Host $djangoCheck -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ Failed to test Django configuration" -ForegroundColor Red
} finally {
    Pop-Location
}

# 6. Frontend Integration Summary
Write-Host "`n6️⃣ INTEGRATION SUMMARY" -ForegroundColor Cyan
Write-Host "-" * 30

Write-Host "🎯 Frontend Features Updated:" -ForegroundColor White
Write-Host "  • WebSocket connection to /ws/detection/" -ForegroundColor Green
Write-Host "  • Real-time detection display with bounding boxes" -ForegroundColor Green  
Write-Host "  • Vihangam class colors (Red: Human, Orange: Debris)" -ForegroundColor Green
Write-Host "  • Toast notifications for status updates" -ForegroundColor Green
Write-Host "  • Interactive detection list and statistics" -ForegroundColor Green
Write-Host "  • Auto-reconnection on connection loss" -ForegroundColor Green

Write-Host "`n📡 WebSocket Commands Supported:" -ForegroundColor White
Write-Host "  • start_detection (with confidence threshold)" -ForegroundColor Green
Write-Host "  • stop_detection" -ForegroundColor Green
Write-Host "  • get_model_info" -ForegroundColor Green
Write-Host "  • update_settings" -ForegroundColor Green

# 7. Next Steps
Write-Host "`n7️⃣ NEXT STEPS TO TEST" -ForegroundColor Cyan
Write-Host "-" * 30

Write-Host "To test the frontend integration:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Navigate to Django directory:" -ForegroundColor White
Write-Host "   cd disaster_dashboard" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Activate virtual environment:" -ForegroundColor White
Write-Host "   ..\\venv\\Scripts\\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Start Django development server:" -ForegroundColor White
Write-Host "   python manage.py runserver" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Open browser and go to:" -ForegroundColor White
Write-Host "   http://localhost:8000/detection/" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Open browser DevTools (F12) and check Console for:" -ForegroundColor White
Write-Host "   🚁 Connecting to Vihangam Detection WebSocket" -ForegroundColor Gray
Write-Host "   ✅ WebSocket connected to Vihangam Detection System" -ForegroundColor Gray
Write-Host ""
Write-Host "6. Click 'Start Detection' and watch for live updates!" -ForegroundColor White

Write-Host "`n🎉 Frontend Integration Test Complete!" -ForegroundColor Green

# 8. Quick Launch Option
Write-Host "`n8️⃣ QUICK LAUNCH" -ForegroundColor Cyan
Write-Host "-" * 30

$launch = Read-Host "Would you like to start the Django server now? (y/n)"
if ($launch -eq 'y' -or $launch -eq 'yes') {
    Write-Host "🚀 Starting Django development server..." -ForegroundColor Green
    Push-Location "disaster_dashboard"
    try {
        Write-Host "📱 Server will start at: http://localhost:8000/detection/" -ForegroundColor Yellow
        Write-Host "💡 Press Ctrl+C to stop the server" -ForegroundColor Yellow
        Write-Host ""
        & ..\$pythonExe manage.py runserver
    } finally {
        Pop-Location
    }
} else {
    Write-Host "✅ Test complete! Use the steps above to manually start the server." -ForegroundColor Green
}

Write-Host "`n🚁 Vihangam YOLO Frontend Test Complete!" -ForegroundColor Green