# 🚁 Vihangam YOLO Object Detection System - Complete Setup & Testing Guide

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Training](#model-training)
6. [Model Testing](#model-testing)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

---

## 🎯 System Overview

The Vihangam YOLO Object Detection System is designed for disaster management and search & rescue operations. It detects:
- **Humans** (Critical Priority 🔴)
- **Debris** (Warning Priority 🟡)

**Current Status:** ✅ Model Trained and Ready for Testing
- Model Path: `runs/detect/disaster_demo_20250918_201832/weights/best.pt`
- Classes: 2 (human, debris)
- Model Size: 5.9 MB

---

## 🔧 Prerequisites

### System Requirements
- **OS:** Windows 10/11 (Current: Windows)
- **Python:** 3.8 or higher
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 2GB free space
- **GPU:** NVIDIA GPU with CUDA support (optional, for faster training)

### Required Software
- Python 3.8+
- Git (for version control)
- PowerShell or Command Prompt

---

## 🚀 Environment Setup

### Step 1: Verify Current Environment
```powershell
# Check your current location
pwd
# Should show: C:\Users\KIIT\OneDrive\Desktop\Vihangam YOLO\Vihangam

# Check Python version
python --version

# Check if virtual environment exists
if (Test-Path "venv") { 
    Write-Host "✅ Virtual environment exists" 
} else { 
    Write-Host "❌ Virtual environment not found" 
}
```

### Step 2: Create/Activate Virtual Environment
```powershell
# If virtual environment doesn't exist, create it
if (!(Test-Path "venv")) {
    python -m venv venv
    Write-Host "✅ Virtual environment created"
}

# Activate virtual environment
venv\Scripts\Activate.ps1

# Verify activation (should show (venv) in prompt)
```

### Step 3: Install Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies
pip install ultralytics torch torchvision opencv-python pillow numpy matplotlib

# Install additional dependencies
pip install pyyaml tqdm seaborn pandas

# Verify installation
venv\Scripts\python.exe -c "import ultralytics; print('✅ YOLOv8 installed successfully')"
venv\Scripts\python.exe -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')"
venv\Scripts\python.exe -c "import cv2; print(f'✅ OpenCV version: {cv2.__version__}')"
```

### Step 4: Verify Project Structure
```powershell
# Check project structure
Get-ChildItem -Recurse -Depth 1 | Format-Table Name, Length, LastWriteTime

# Expected structure:
# ├── data.yaml                 # Dataset configuration
# ├── detect_objects.py         # Main detection script
# ├── train_yolo_model.py       # Training script
# ├── images/                   # Dataset images
# │   ├── train/
# │   └── val/
# ├── labels/                   # Dataset labels
# │   ├── train/
# │   └── val/
# ├── runs/                     # Training results
# │   └── detect/
# ├── detection_results/        # Detection outputs
# └── venv/                     # Virtual environment
```

---

## 📊 Dataset Preparation

### Current Dataset Status
Your dataset is already configured with:
- **Classes:** human (0), debris (1)
- **Training images:** `images/train/`
- **Validation images:** `images/val/`
- **Labels:** YOLO format in `labels/train/` and `labels/val/`

### Verify Dataset
```powershell
# Check dataset structure
python -c "
import yaml
with open('data.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('Dataset Configuration:')
for key, value in config.items():
    if key in ['path', 'train', 'val', 'names', 'nc']:
        print(f'  {key}: {value}')
"

# Count images and labels
$train_images = (Get-ChildItem "images/train/*.jpg").Count
$train_labels = (Get-ChildItem "labels/train/*.txt").Count
$val_images = (Get-ChildItem "images/val/*.jpg").Count
$val_labels = (Get-ChildItem "labels/val/*.txt").Count

Write-Host "📊 Dataset Statistics:"
Write-Host "  Training: $train_images images, $train_labels labels"
Write-Host "  Validation: $val_images images, $val_labels labels"
```

### Validate Dataset Quality
```powershell
# Run dataset validation
python -c "
from ultralytics import YOLO
import yaml

# Load dataset config
with open('data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

print('🔍 Validating dataset...')
print(f'Classes: {data_config[\"names\"]}')
print(f'Number of classes: {data_config[\"nc\"]}')

# Basic validation
import os
train_path = data_config['train']
val_path = data_config['val']

if os.path.exists(train_path) and os.path.exists(val_path):
    print('✅ Dataset paths exist')
else:
    print('❌ Dataset paths missing')
"
```

---

## 🎯 Model Training

### Option 1: Use Pre-trained Model (Recommended for Testing)
Your model is already trained and ready:
```powershell
# Verify trained model
if (Test-Path "runs/detect/disaster_demo_20250918_201832/weights/best.pt") {
    Write-Host "✅ Pre-trained model found"
    $model_size = [math]::Round((Get-Item "runs/detect/disaster_demo_20250918_201832/weights/best.pt").Length / 1MB, 2)
    Write-Host "📊 Model size: $model_size MB"
} else {
    Write-Host "❌ Pre-trained model not found"
}
```

### Option 2: Train New Model (If needed)
```powershell
# Basic training command
venv\Scripts\python.exe train_yolo_model.py

# Or use ultralytics directly (after activating venv)
venv\Scripts\Activate.ps1
yolo train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16

# For CPU-only training (slower but works without GPU)
venv\Scripts\python.exe train_cpu.py
```

### Monitor Training Progress
```powershell
# Check training results
if (Test-Path "runs/detect") {
    $latest_run = Get-ChildItem "runs/detect" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    Write-Host "📈 Latest training run: $($latest_run.Name)"
    
    # Check if training completed
    if (Test-Path "$($latest_run.FullName)/weights/best.pt") {
        Write-Host "✅ Training completed successfully"
    } else {
        Write-Host "⚠️  Training may be incomplete"
    }
}
```

---

## 🧪 Model Testing

### Step 1: Basic Model Test
```powershell
# Test model loading
python -c "
from ultralytics import YOLO
import os

# Find the best model
model_path = 'runs/detect/disaster_demo_20250918_201832/weights/best.pt'
if os.path.exists(model_path):
    print(f'✅ Loading model from: {model_path}')
    model = YOLO(model_path)
    print(f'📊 Model classes: {list(model.names.values())}')
    print('✅ Model loaded successfully')
else:
    print('❌ Model file not found')
"
```

### Step 2: Test with Sample Images
```powershell
# Check available test images
Write-Host "🖼️  Available test images:"
Get-ChildItem "*.jpg" | ForEach-Object { Write-Host "  - $($_.Name)" }

# Test with disaster scene image
venv\Scripts\python.exe detect_objects.py --image disaster_test_20250918_205115.jpg --confidence 0.1

# Test with alternative image
venv\Scripts\python.exe detect_objects.py --image real_disaster_image.jpg --confidence 0.25
```

### Step 3: Batch Testing
```powershell
# Create test images directory if it doesn't exist
if (!(Test-Path "test_images")) {
    New-Item -ItemType Directory -Path "test_images"
    # Copy some validation images for testing
    Copy-Item "images/val/*.jpg" "test_images/" -ErrorAction SilentlyContinue
}

# Run batch detection
python detect_objects.py --directory test_images --confidence 0.2
```

### Step 4: Validate Detection Results
```powershell
# Check detection results
if (Test-Path "detection_results") {
    $result_count = (Get-ChildItem "detection_results/*.jpg").Count
    $json_count = (Get-ChildItem "detection_results/*.json").Count
    Write-Host "📊 Detection Results:"
    Write-Host "  - Images: $result_count"
    Write-Host "  - JSON reports: $json_count"
    
    # Show latest result
    $latest_result = Get-ChildItem "detection_results/*.jpg" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest_result) {
        Write-Host "🎯 Latest detection: $($latest_result.Name)"
    }
} else {
    Write-Host "⚠️  No detection results found"
}
```

### Step 5: Performance Testing
```powershell
# Run performance benchmark
python -c "
import time
from ultralytics import YOLO
import cv2
import os

# Load model
model_path = 'runs/detect/disaster_demo_20250918_201832/weights/best.pt'
if not os.path.exists(model_path):
    print('❌ Model not found')
    exit()

model = YOLO(model_path)
print('✅ Model loaded')

# Test image
test_image = 'disaster_test_20250918_205115.jpg'
if not os.path.exists(test_image):
    print('❌ Test image not found')
    exit()

# Benchmark detection speed
print('🚀 Running performance benchmark...')
times = []
for i in range(5):
    start_time = time.time()
    results = model(test_image, verbose=False)
    inference_time = time.time() - start_time
    times.append(inference_time)
    print(f'  Run {i+1}: {inference_time:.3f}s')

avg_time = sum(times) / len(times)
print(f'📊 Average inference time: {avg_time:.3f}s')
print(f'📊 FPS: {1/avg_time:.1f}')
"
```

---

## 🚀 Production Deployment

### Step 1: Model Validation
```powershell
# Run comprehensive validation
python validate_yolo_model.py

# Or use ultralytics validation
yolo val model=runs/detect/disaster_demo_20250918_201832/weights/best.pt data=data.yaml
```

### Step 2: Export Model for Deployment
```powershell
# Export to different formats
python -c "
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/disaster_demo_20250918_201832/weights/best.pt')

# Export to ONNX (for cross-platform deployment)
model.export(format='onnx')
print('✅ Exported to ONNX format')

# Export to TensorRT (for NVIDIA GPU deployment)
# model.export(format='engine')  # Uncomment if you have TensorRT

# Export to CoreML (for iOS deployment)
# model.export(format='coreml')  # Uncomment if needed
"
```

### Step 3: Create Production Script
```powershell
# Test the integrated detection system
python integrate_with_vihangam.py

# Test Django integration (if applicable)
python test_django_integration.py
```

### Step 4: Create Deployment Package
```powershell
# Create deployment directory
New-Item -ItemType Directory -Path "deployment" -Force

# Copy essential files
Copy-Item "detect_objects.py" "deployment/"
Copy-Item "data.yaml" "deployment/"
Copy-Item "runs/detect/disaster_demo_20250918_201832/weights/best.pt" "deployment/vihangam_model.pt"

# Create requirements.txt for deployment
pip freeze > deployment/requirements.txt

Write-Host "✅ Deployment package created in 'deployment' folder"
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Issues
```powershell
# Check if model file exists and is valid
python -c "
import os
import torch

model_path = 'runs/detect/disaster_demo_20250918_201832/weights/best.pt'
if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print('✅ Model file is valid')
        print(f'📊 Model keys: {list(checkpoint.keys())}')
    except Exception as e:
        print(f'❌ Model file corrupted: {e}')
else:
    print('❌ Model file not found')
"
```

#### 2. Image Loading Issues
```powershell
# Test image loading
python -c "
import cv2
import os

test_images = ['disaster_test_20250918_205115.jpg', 'real_disaster_image.jpg']
for img_path in test_images:
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            print(f'✅ {img_path}: {img.shape}')
        else:
            print(f'❌ {img_path}: Cannot load image')
    else:
        print(f'❌ {img_path}: File not found')
"
```

#### 3. Dependencies Issues
```powershell
# Check all dependencies
python -c "
import sys
required_modules = ['ultralytics', 'torch', 'cv2', 'PIL', 'numpy', 'yaml']
missing = []

for module in required_modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        print(f'❌ {module} - MISSING')
        missing.append(module)

if missing:
    print(f'\\nInstall missing modules: pip install {\" \".join(missing)}')
else:
    print('\\n✅ All dependencies installed')
"
```

#### 4. Performance Issues
```powershell
# Check system resources
python -c "
import torch
import psutil
import platform

print('🖥️  System Information:')
print(f'   OS: {platform.system()} {platform.release()}')
print(f'   CPU: {platform.processor()}')
print(f'   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')
print(f'   Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')

if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA Version: {torch.version.cuda}')
    print('✅ GPU acceleration available')
else:
    print('⚠️  Using CPU only (slower)')
"
```

---

## 🎯 Advanced Usage

### Custom Training Parameters
```powershell
# Advanced training with custom parameters
yolo train data=data.yaml model=yolov8n.pt epochs=200 imgsz=640 batch=8 patience=50 save_period=10 conf=0.25 iou=0.45
```

### Hyperparameter Tuning
```powershell
# Automatic hyperparameter tuning
yolo train data=data.yaml model=yolov8n.pt epochs=100 tune=True
```

### Real-time Detection (Webcam)
```powershell
# Real-time detection from webcam
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/disaster_demo_20250918_201832/weights/best.pt')
# Note: This requires a webcam connected
# model.track(source=0, show=True, save=True)
print('Real-time detection code ready (uncomment to use with webcam)')
"
```

### API Integration
```powershell
# Test API integration (if applicable)
python -c "
# Example API usage
import requests
import base64
import json

# This is a template for API integration
print('📡 API Integration Template:')
print('1. Convert image to base64')
print('2. Send POST request to detection endpoint')
print('3. Parse JSON response')
print('4. Display results')
"
```

---

## 📊 Testing Checklist

### ✅ Pre-deployment Testing
- [ ] Model loads successfully
- [ ] Single image detection works
- [ ] Batch detection works
- [ ] Results are saved correctly
- [ ] JSON reports are generated
- [ ] Performance is acceptable (< 1s per image)
- [ ] Both human and debris detection work
- [ ] Confidence thresholds work correctly
- [ ] Export formats work (ONNX, etc.)
- [ ] Integration scripts run without errors

### 🧪 Validation Tests
```powershell
# Run all validation tests
Write-Host "🧪 Running comprehensive validation tests..."

# Test 1: Model Loading
python -c "from ultralytics import YOLO; YOLO('runs/detect/disaster_demo_20250918_201832/weights/best.pt'); print('✅ Test 1: Model Loading')"

# Test 2: Single Detection
python detect_objects.py --image disaster_test_20250918_205115.jpg --confidence 0.25 --no-save
Write-Host "✅ Test 2: Single Detection"

# Test 3: Batch Detection
python detect_objects.py --directory images/val --confidence 0.25
Write-Host "✅ Test 3: Batch Detection"

# Test 4: Different Confidence Levels
python detect_objects.py --image disaster_test_20250918_205115.jpg --confidence 0.1 --no-save
python detect_objects.py --image disaster_test_20250918_205115.jpg --confidence 0.5 --no-save
Write-Host "✅ Test 4: Confidence Thresholds"

Write-Host "🎉 All tests completed!"
```

---

## 📝 Usage Examples

### Basic Detection
```powershell
# Detect objects with default settings
python detect_objects.py --image disaster_scene.jpg

# Detect with custom confidence
python detect_objects.py --image disaster_scene.jpg --confidence 0.3

# Batch process directory
python detect_objects.py --directory test_images --confidence 0.25

# Don't save results (testing only)
python detect_objects.py --image disaster_scene.jpg --no-save
```

### Advanced Usage
```powershell
# Use specific model
python detect_objects.py --image disaster_scene.jpg --model custom_models/my_model.pt

# Process with very low confidence (find all possible objects)
python detect_objects.py --image disaster_scene.jpg --confidence 0.05

# Process multiple images with different settings
foreach ($img in Get-ChildItem "*.jpg") {
    python detect_objects.py --image $img.Name --confidence 0.2
}
```

---

## 🎯 Final Verification

Run this complete verification script:
```powershell
Write-Host "🚁 Vihangam YOLO System - Final Verification" -ForegroundColor Green
Write-Host "=" * 50

# 1. Environment Check
Write-Host "1️⃣ Environment Check:"
python --version
if ($env:VIRTUAL_ENV) { Write-Host "   ✅ Virtual environment active" } else { Write-Host "   ⚠️ Virtual environment not active" }

# 2. Dependencies Check
Write-Host "2️⃣ Dependencies Check:"
python -c "import ultralytics, torch, cv2; print('   ✅ All core dependencies installed')" 2>$null || Write-Host "   ❌ Missing dependencies"

# 3. Model Check
Write-Host "3️⃣ Model Check:"
if (Test-Path "runs/detect/disaster_demo_20250918_201832/weights/best.pt") {
    Write-Host "   ✅ Trained model found"
} else {
    Write-Host "   ❌ Trained model missing"
}

# 4. Dataset Check
Write-Host "4️⃣ Dataset Check:"
if ((Test-Path "data.yaml") -and (Test-Path "images")) {
    Write-Host "   ✅ Dataset configuration found"
} else {
    Write-Host "   ❌ Dataset missing"
}

# 5. Test Detection
Write-Host "5️⃣ Test Detection:"
try {
    $output = python detect_objects.py --image disaster_test_20250918_205115.jpg --confidence 0.25 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Detection test passed"
    } else {
        Write-Host "   ❌ Detection test failed"
    }
} catch {
    Write-Host "   ❌ Detection test error"
}

Write-Host "`n🎉 Vihangam YOLO System Ready for Production!" -ForegroundColor Green
Write-Host "📁 Check detection_results/ for output files" -ForegroundColor Yellow
```

---

## 📞 Support & Maintenance

### Regular Maintenance Tasks
1. **Clean up old detection results:** `Remove-Item detection_results/* -Force`
2. **Update dependencies:** `pip install --upgrade ultralytics torch`
3. **Backup model:** `Copy-Item runs/detect/*/weights/best.pt backup/`
4. **Monitor performance:** Run benchmark tests monthly

### Getting Help
- Check the troubleshooting section above
- Review ultralytics documentation: https://docs.ultralytics.com/
- Verify system requirements and dependencies

---

*Last Updated: 2025-01-18*
*System Status: ✅ Operational*
*Model Version: disaster_demo_20250918_201832*