# 🚁 Vihangam YOLO Quick Test Script
# Automated testing and verification script

Write-Host "🚁 Vihangam YOLO Object Detection System - Quick Test" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green

# Function to check command success
function Test-Command {
    param($CommandResult, $TestName)
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $TestName - PASSED" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ $TestName - FAILED" -ForegroundColor Red
        return $false
    }
}

$testResults = @{}

# 1. Environment Verification
Write-Host "`n1️⃣ ENVIRONMENT CHECK" -ForegroundColor Cyan
Write-Host "-" * 30

Write-Host "📍 Current Directory: " -NoNewline
$currentDir = (Get-Location).Path
Write-Host $currentDir -ForegroundColor Yellow

Write-Host "🐍 Python Version: " -NoNewline
$pythonVersion = venv\Scripts\python.exe --version 2>&1
Write-Host $pythonVersion -ForegroundColor Yellow

Write-Host "🔧 Virtual Environment: " -NoNewline
if ($env:VIRTUAL_ENV) {
    Write-Host "ACTIVE ✅" -ForegroundColor Green
    $testResults["VirtualEnv"] = $true
} else {
    Write-Host "NOT ACTIVE ⚠️" -ForegroundColor Yellow
    $testResults["VirtualEnv"] = $false
}

# 2. Dependencies Check
Write-Host "`n2️⃣ DEPENDENCIES CHECK" -ForegroundColor Cyan
Write-Host "-" * 30

$dependencies = @("ultralytics", "torch", "cv2", "PIL", "numpy", "yaml")
$allDepsOk = $true

foreach ($dep in $dependencies) {
    Write-Host "Checking $dep... " -NoNewline
    $result = venv\Scripts\python.exe -c "try: import $dep; print('OK')" 2>$null
    if ($result -eq "OK") {
        Write-Host "✅" -ForegroundColor Green
    } else {
        Write-Host "❌" -ForegroundColor Red
        $allDepsOk = $false
    }
}

$testResults["Dependencies"] = $allDepsOk

# 3. Model Check
Write-Host "`n3️⃣ MODEL VERIFICATION" -ForegroundColor Cyan
Write-Host "-" * 30

$modelPath = "runs/detect/disaster_demo_20250918_201832/weights/best.pt"
Write-Host "Model Path: " -NoNewline
Write-Host $modelPath -ForegroundColor Yellow

if (Test-Path $modelPath) {
    Write-Host "✅ Model file exists" -ForegroundColor Green
    $modelSize = [math]::Round((Get-Item $modelPath).Length / 1MB, 2)
    Write-Host "📊 Model size: $modelSize MB" -ForegroundColor Yellow
    
    # Test model loading
    Write-Host "Testing model loading... " -NoNewline
    $modelTest = venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('$modelPath'); print('SUCCESS')" 2>$null
    if ($modelTest -eq "SUCCESS") {
        Write-Host "✅" -ForegroundColor Green
        $testResults["ModelLoading"] = $true
    } else {
        Write-Host "❌" -ForegroundColor Red
        $testResults["ModelLoading"] = $false
    }
} else {
    Write-Host "❌ Model file not found" -ForegroundColor Red
    $testResults["ModelLoading"] = $false
}

# 4. Test Images Check
Write-Host "`n4️⃣ TEST IMAGES CHECK" -ForegroundColor Cyan
Write-Host "-" * 30

$testImages = @("disaster_test_20250918_205115.jpg", "real_disaster_image.jpg")
$availableImages = @()

foreach ($img in $testImages) {
    Write-Host "Checking $img... " -NoNewline
    if (Test-Path $img) {
        Write-Host "✅" -ForegroundColor Green
        $availableImages += $img
    } else {
        Write-Host "❌" -ForegroundColor Red
    }
}

if ($availableImages.Count -gt 0) {
    Write-Host "📸 Found $($availableImages.Count) test images" -ForegroundColor Green
    $testResults["TestImages"] = $true
    $primaryTestImage = $availableImages[0]
} else {
    Write-Host "⚠️ No test images found" -ForegroundColor Yellow
    $testResults["TestImages"] = $false
}

# 5. Detection Test
Write-Host "`n5️⃣ DETECTION TEST" -ForegroundColor Cyan
Write-Host "-" * 30

if ($testResults["ModelLoading"] -and $testResults["TestImages"]) {
    Write-Host "Running detection on: $primaryTestImage"
    Write-Host "Please wait..." -ForegroundColor Yellow
    
    $detectionOutput = venv\Scripts\python.exe detect_objects.py --image $primaryTestImage --confidence 0.25 2>&1
    $detectionSuccess = Test-Command -CommandResult $LASTEXITCODE -TestName "Object Detection"
    $testResults["Detection"] = $detectionSuccess
    
    if ($detectionSuccess) {
        Write-Host "🎯 Detection completed successfully!" -ForegroundColor Green
        
        # Check if results were saved
        if (Test-Path "detection_results") {
            $resultCount = (Get-ChildItem "detection_results/*.jpg").Count
            Write-Host "📊 Detection results: $resultCount images saved" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "⚠️ Skipping detection test (prerequisites not met)" -ForegroundColor Yellow
    $testResults["Detection"] = $false
}

# 6. Performance Benchmark (Quick)
Write-Host "`n6️⃣ PERFORMANCE BENCHMARK" -ForegroundColor Cyan
Write-Host "-" * 30

if ($testResults["Detection"]) {
    Write-Host "Running performance test..." -ForegroundColor Yellow
    
    $pythonCode = @"
import time
from ultralytics import YOLO
model = YOLO('$modelPath')
start = time.time()
results = model('$primaryTestImage', verbose=False)
inference_time = time.time() - start
print(f'Inference time: {inference_time:.3f}s')
print(f'FPS: {1/inference_time:.1f}')
"@
    
    $perfTest = venv\Scripts\python.exe -c $pythonCode 2>$null

    if ($perfTest) {
        Write-Host $perfTest -ForegroundColor Yellow
        $testResults["Performance"] = $true
    } else {
        Write-Host "❌ Performance test failed" -ForegroundColor Red
        $testResults["Performance"] = $false
    }
} else {
    Write-Host "⚠️ Skipping performance test" -ForegroundColor Yellow
    $testResults["Performance"] = $false
}

# 7. Results Summary
Write-Host "`n📊 TEST RESULTS SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 40 -ForegroundColor Cyan

$passedTests = 0
$totalTests = $testResults.Count

foreach ($test in $testResults.GetEnumerator()) {
    $status = if ($test.Value) { "✅ PASS" } else { "❌ FAIL" }
    $color = if ($test.Value) { "Green" } else { "Red" }
    Write-Host "$($test.Key): " -NoNewline
    Write-Host $status -ForegroundColor $color
    
    if ($test.Value) { $passedTests++ }
}

Write-Host "`nOVERALL SCORE: $passedTests/$totalTests tests passed" -ForegroundColor Cyan

# 8. Recommendations
Write-Host "`n💡 RECOMMENDATIONS" -ForegroundColor Cyan
Write-Host "-" * 30

if (-not $testResults["VirtualEnv"]) {
    Write-Host "⚠️  Activate virtual environment: venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}

if (-not $testResults["Dependencies"]) {
    Write-Host "⚠️  Install missing dependencies: pip install ultralytics torch opencv-python pillow numpy pyyaml" -ForegroundColor Yellow
}

if (-not $testResults["ModelLoading"]) {
    Write-Host "⚠️  Check model file or retrain: python train_yolo_model.py" -ForegroundColor Yellow
}

if (-not $testResults["TestImages"]) {
    Write-Host "⚠️  Add test images to current directory" -ForegroundColor Yellow
}

if ($passedTests -eq $totalTests) {
    Write-Host "`n🎉 ALL TESTS PASSED! System ready for production!" -ForegroundColor Green
    Write-Host "📁 Check detection_results/ folder for outputs" -ForegroundColor Yellow
    Write-Host "📖 Read COMPLETE_SETUP_GUIDE.md for advanced usage" -ForegroundColor Yellow
} elseif ($passedTests -ge 3) {
    Write-Host "`n✅ Core functionality working! Address warnings for optimal performance" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  System needs attention. Check failed tests above" -ForegroundColor Yellow
}

# 9. Quick Commands Reference
Write-Host "`n🔧 QUICK COMMANDS REFERENCE" -ForegroundColor Cyan
Write-Host "-" * 30
Write-Host "• Single detection:    python detect_objects.py --image image.jpg" -ForegroundColor White
Write-Host "• Batch detection:     python detect_objects.py --directory images/" -ForegroundColor White
Write-Host "• Low confidence:      python detect_objects.py --image image.jpg --confidence 0.1" -ForegroundColor White
Write-Host "• No save results:     python detect_objects.py --image image.jpg --no-save" -ForegroundColor White
Write-Host "• Validate model:      python validate_yolo_model.py" -ForegroundColor White

Write-Host "`n🚁 Vihangam YOLO Quick Test Complete!" -ForegroundColor Green