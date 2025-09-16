# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Vihangam is drone software for disaster management focused on achieving better speed, accuracy, and precision. This is an early-stage project that will likely involve real-time data processing, flight control systems, image/video analysis, and emergency response coordination.

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install Django==5.0.0 channels ultralytics opencv-python
```

### Django Development
```bash
# Activate virtual environment (always do this first)
source venv/bin/activate

# Run development server
cd disaster_dashboard
python manage.py runserver

# Create new Django app
python manage.py startapp <app_name>

# Database migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic
```

### Testing
```bash
# Run Django tests
cd disaster_dashboard
python manage.py test

# Run specific test
python manage.py test <app_name>.tests.<TestClass>.<test_method>
```

### Development Patterns
Key development areas for this drone disaster management system:
- Building real-time WebSocket connections (using Django Channels)
- Processing drone sensor data with OpenCV
- AI-powered object detection using Ultralytics YOLO
- Real-time disaster response coordination
- Flight control system integration

## Architecture Considerations

Given the nature of drone disaster management software, the architecture will likely involve:

### Core Components (Expected)
- **Flight Control Module**: Real-time drone control and navigation
- **Sensor Processing**: Image, video, and environmental data analysis
- **Communication System**: Emergency response coordination and data transmission
- **Data Pipeline**: Real-time processing and storage of disaster monitoring data
- **Safety Systems**: Failsafes and emergency protocols
- **User Interface**: Ground control station for operators

### Technical Requirements (Anticipated)
- Low-latency processing for real-time control
- Robust error handling for critical safety systems
- Scalable data processing for multiple drone coordination
- Secure communication protocols for emergency scenarios
- Cross-platform compatibility for various deployment environments

## Development Guidelines

### Safety-Critical Code
- All flight control and safety systems require extensive testing
- Implement redundant failsafe mechanisms
- Validate all sensor inputs and communication channels
- Document all emergency procedures and system limitations

### Real-Time Requirements
- Profile performance-critical code paths
- Consider memory allocation patterns in real-time loops
- Test under various load and network conditions
- Implement proper timeouts and resource management

### Disaster Management Context
- Design for unreliable network conditions
- Plan for offline operation capabilities
- Ensure data integrity during emergency scenarios
- Consider multi-drone coordination protocols

## Project Structure

### Current Structure
```
Vihangam/
├── venv/                    # Python virtual environment
├── disaster_dashboard/      # Django project root
│   ├── apps/               # Django applications
│   │   ├── detection/      # YOLO integration for object detection
│   │   ├── pathfinding/    # A* algorithm for drone navigation
│   │   └── dashboard/      # Main web interface
│   ├── disaster_dashboard/ # Django settings and configuration
│   ├── static/             # CSS, JS, images
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   ├── templates/          # HTML templates
│   │   ├── dashboard/
│   │   ├── detection/
│   │   └── pathfinding/
│   ├── media/              # Uploaded files (images, videos)
│   └── manage.py           # Django management script
├── README.md
└── WARP.md                 # This file
```

### Technology Stack
- **Django 5.0.0**: Web framework for the dashboard interface
- **Django Channels**: WebSocket support for real-time communication
- **Ultralytics**: YOLO-based AI object detection for drone imagery
- **OpenCV**: Computer vision processing for sensor data
- **Python**: Primary development language

### Django Apps
Current Django apps in the system:
- `apps.detection/` - YOLO-based object detection for drone imagery analysis
- `apps.pathfinding/` - A* algorithm implementation for optimal drone navigation
- `apps.dashboard/` - Main web interface for disaster management control

### Future Apps (Expected)
As the project develops, additional apps may include:
- `apps.communication/` - Emergency response coordination
- `apps.sensors/` - Real-time sensor data processing
- `apps.api/` - REST API for drone-to-server communication
