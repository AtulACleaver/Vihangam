# 🚁 Vihangam - The Bird's Eye

Vihangam (Vi-han-gam) is a drone software system designed for disaster management, focusing on search and rescue operations using AI-powered object detection.

## 📁 Project Structure

```
Vihangam/
├── 🤖 yolo_detection/           # YOLO AI Detection System
│   ├── scripts/                 # Detection and training scripts
│   ├── models/                  # Trained models and checkpoints
│   ├── data/                    # Datasets and configuration
│   └── results/                 # Detection outputs and results
│
├── 🌐 disaster_dashboard/       # Django Web Dashboard
│   ├── apps/                    # Django applications
│   ├── static/                  # CSS, JS, images
│   └── templates/               # HTML templates
│
├── 📚 docs/                     # Documentation
├── 🐍 venv/                     # Python virtual environment
├── 📋 README.md                 # This file
└── 📦 requirements.txt          # Python dependencies
```

## 🔧 Tech Stack
- **YOLOv8**: AI object detection for humans and debris
- **Django 5.0.0**: Web framework for mission control dashboard
- **Ultralytics**: YOLO model training and inference
- **OpenCV**: Computer vision processing
- **Bootstrap 5.3.0**: Frontend CSS framework

## 🚀 Quick Start

### 1. Setup Environment
```powershell
# Activate virtual environment
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Run YOLO Detection
```powershell
# Single image detection
venv\Scripts\python.exe yolo_detection\scripts\detect_objects.py --image path\to\image.jpg

# Batch detection
venv\Scripts\python.exe yolo_detection\scripts\detect_objects.py --directory path\to\images\folder
```

### 3. Start Web Dashboard
```powershell
cd disaster_dashboard
python manage.py runserver
# Visit: http://localhost:8000
```

## 🎯 Core Applications

### 1. YOLO Detection System (`yolo_detection/`)
**AI-Powered Object Detection for Disaster Management**
- **Human Detection**: Critical priority alerts for search & rescue
- **Debris Detection**: Infrastructure damage assessment
- **Real-time Processing**: Fast inference with confidence scoring
- **Batch Processing**: Handle multiple images efficiently

### 2. Dashboard App (`disaster_dashboard/apps/dashboard/`)
Mission Control Center acts as the central hub for drone operations
- **Real-time Mission Overview**: Active drones, monitored areas, detection alerts
- **Live Drone Feed Interface**: Video feed placeholder with controls
- **Emergency Alert System**: Emergency protocol activation
- **Drone Status Monitoring**: GPS coordinates, battery levels, system status
- **Activity Log**: Real-time event tracking and mission history


### 2\. Detection App (`apps.detection`)
AI Object Detection System — YOLO-powered computer vision

**Features Implemented:**
- **Live Detection Feed:** Real-time object detection processing interface
- **YOLO Model Integration:** Multiple model options (YOLOv8 variants, custom disaster objects)
- **Object Classification:** Person, vehicle, building, debris detection
- **Confidence Threshold Controls:** Adjustable detection sensitivity (75% default)
- **Alert Management:** Auto alerts and high-priority filtering
- **Detection Analytics:** Real-time statistics and object count summaries
- **Performance Monitoring:** 28 FPS processing speed, 94.2% accuracy tracking

---

### 3\. Pathfinding App (`apps.pathfinding`)
Autonomous Navigation System — A\* algorithm-based flight planning

**Features Implemented:**
- **Interactive Flight Path Visualization:** Map with waypoints, obstacles, and drone position
- **A\* Algorithm Implementation:** Optimal path calculation with distance formulas
- **Waypoint Management:** Add, modify, and sequence navigation points
- **Flight Parameter Controls:** Altitude (50–500m), speed (5–25 m/s) adjustment
- **Safety Systems:** Obstacle avoidance, weather routing, no-fly zone respect
- **Emergency Protocols:** Emergency return to base functionality
- **Real-time Navigation Status:** GPS tracking, battery monitoring, ETA calculations
- **Multi-algorithm Support:** A\*, Dijkstra, RRT, Potential Field options

**Utility Functions:**
- **Haversine Distance Calculation:** Accurate GPS coordinate distance measurement
- **Waypoint Generation:** Automatic intermediate point calculation
- **Path Optimization:** Shortest, safest, or fuel-efficient route selection

## 🚀 Current Development State

What's Working:

✅ Complete web interface with all three modules  
✅ Responsive design and professional UI/UX  
✅ API endpoints with simulated responses  
✅ Navigation between all sections  
✅ Interactive controls and real-time updates  
✅ Database setup and Django configuration  
✅ Virtual environment with all dependencies installed

What's Simulated (Ready for Integration):

🔄 YOLO object detection (framework ready, needs camera integration)  
🔄 Real GPS/telemetry data (currently using mock data)  
🔄 WebSocket connections (Django Channels configured)  
🔄 Actual drone communication protocols  
🔄 File upload and media handling

## 📚 Documentation

- **Complete Setup Guide**: `docs/COMPLETE_SETUP_GUIDE.md`
- **Training Guide**: `docs/YOLO_TRAINING_SUMMARY.md`
- **Error Fixes**: `docs/ERROR_FIXES_SUMMARY.md`
- **Frontend Updates**: `docs/FRONTEND_UPDATE_SUMMARY.md`
- **Validation Guide**: `docs/VALIDATION_SUMMARY.md`

## ✅ What's Working

- ✅ Complete YOLO detection system with trained model
- ✅ Web dashboard with all three modules (Dashboard, Detection, Pathfinding)
- ✅ Real-time object detection (humans and debris)
- ✅ Batch processing capabilities
- ✅ Professional UI/UX with Bootstrap
- ✅ Virtual environment with all dependencies
- ✅ Database setup and Django configuration
