## Vihangam - the bird's eye 🦅
Vihangam (Vi-han-gam) is a drone software system designed for disaster management, focusing on search and rescue operations.

## Project Architecture

### Tech Stack
- Django 5.0.0: Web framework for the dashboard interface 
- Django Channels: WebSocket support for real-time communication (configured but not yet implemented)
- Ultralytics: YOLO-based AI object detection (installed in venv)
- OpenCV: Computer vision processing (installed in venv)
- SQLite: Database for development
- Bootstrap 5.3.0: Frontend CSS framework
- Font Awesome 6.0: Icons and UI elements

## Core Applications
### 1\. Dashboard App (`apps.dashboard`)  
   Mission Control Center acts as the central hub for drone operations

   **Features Implemented:**
   - **Real-time Mission Overview:** Active drones (3), monitored areas (12), detection alerts (7)
   - **Live Drone Feed Interface:** Video feed placeholder with controls
   - **Emergency Alert System:** Emergency protocol activation
   - **Drone Status Monitoring:** GPS coordinates, battery levels, system status
   - **Activity Log:** Real-time event tracking and mission history
   - **Quick Action Controls:** Start detection, auto pathfinding, return to base, emergency land


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

Development Environment Ready:

- Virtual environment with Django 5.0.0, Ultralytics, OpenCV
- Database migrations ready
- Static files configuration complete
- Template system fully functional