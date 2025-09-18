# 🚁 Vihangam YOLO Simple Test Script
Write-Host "🚁 Vihangam YOLO Object Detection System - Simple Test" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green

$pythonExe = "venv\Scripts\python.exe"

# 1. Check Python
Write-Host "`n1️⃣ Python Check:" -ForegroundColor Cyan
if (Test-Path $pythonExe) {
    $version = & $pythonExe --version 2>&1
    Write-Host "✅ $version" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found" -ForegroundColor Red
    exit 1
}

# 2. Check Dependencies
Write-Host "`n2️⃣ Dependencies Check:" -ForegroundColor Cyan
$deps = @("ultralytics", "torch", "cv2", "PIL", "numpy")
foreach ($dep in $deps) {
    $check = & $pythonExe -c "import $dep; print('OK')" 2>$null
    if ($check -eq "OK") {
        Write-Host "✅ $dep" -ForegroundColor Green
    } else {
        Write-Host "❌ $dep missing" -ForegroundColor Red
    }
}

# 3. Check Model
Write-Host "`n3️⃣ Model Check:" -ForegroundColor Cyan
$modelPath = "runs/detect/disaster_demo_20250918_201832/weights/best.pt"
if (Test-Path $modelPath) {
    $size = [math]::Round((Get-Item $modelPath).Length / 1MB, 2)
    Write-Host "✅ Model found ($size MB)" -ForegroundColor Green
} else {
    Write-Host "❌ Model not found" -ForegroundColor Red
}

# 4. Check Test Images
Write-Host "`n4️⃣ Test Images Check:" -ForegroundColor Cyan
$testImg = "disaster_test_20250918_205115.jpg"
if (Test-Path $testImg) {
    Write-Host "✅ Test image found: $testImg" -ForegroundColor Green
} else {
    Write-Host "❌ Test image not found: $testImg" -ForegroundColor Red
    exit 1
}

# 5. Run Detection Test
Write-Host "`n5️⃣ Detection Test:" -ForegroundColor Cyan
Write-Host "Running detection..." -ForegroundColor Yellow
try {
    & $pythonExe detect_objects.py --image $testImg --confidence 0.25
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Detection successful!" -ForegroundColor Green
    } else {
        Write-Host "❌ Detection failed!" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Detection error!" -ForegroundColor Red
}

# 6. Check Results
Write-Host "`n6️⃣ Results Check:" -ForegroundColor Cyan
if (Test-Path "detection_results") {
    $count = (Get-ChildItem "detection_results/*.jpg" -ErrorAction SilentlyContinue).Count
    Write-Host "✅ $count result images found" -ForegroundColor Green
} else {
    Write-Host "⚠️  No results folder" -ForegroundColor Yellow
}

Write-Host "`n🎉 Test Complete!" -ForegroundColor Green
Write-Host "`n💡 To run detection manually:" -ForegroundColor Cyan
Write-Host "venv\Scripts\python.exe detect_objects.py --image $testImg --confidence 0.25" -ForegroundColor White