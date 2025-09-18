# Vihangam Development Environment Setup

This guide will help you set up the development environment for the Vihangam disaster management drone system.

## 🦅 About Vihangam

Vihangam is a Django-based web application for drone-powered disaster management and search & rescue operations. The system includes:

- **Mission Control Dashboard**: Central hub for drone operations
- **AI Object Detection**: YOLO-powered computer vision for person/object detection
- **Autonomous Pathfinding**: A* algorithm-based flight planning
- **Real-time Communication**: WebSocket support via Django Channels

## 📋 Prerequisites

- Python 3.11+ (Currently using Python 3.13.3)
- Git
- Windows PowerShell or Command Prompt

## 🚀 Quick Start

### 1. Navigate to Project Directory
```powershell
cd C:\Users\KIIT\Documents\GitHub\Vihangam
```

### 2. Start the Development Server

**Option A: Using Batch Script (Recommended)**
```cmd
start_dev.bat
```

**Option B: Using PowerShell Script**
```powershell
.\start_dev.ps1
```

**Option C: Manual Start**
```powershell
venv\Scripts\python.exe disaster_dashboard\manage.py runserver
```

### 3. Access the Application
- Open your web browser
- Navigate to: http://127.0.0.1:8000

## 📁 Project Structure

```
Vihangam/
├── disaster_dashboard/          # Django project root
│   ├── apps/                   # Django applications
│   │   ├── dashboard/          # Mission control interface
│   │   ├── detection/          # AI object detection
│   │   └── pathfinding/        # Autonomous navigation
│   ├── disaster_dashboard/     # Project settings
│   ├── static/                 # CSS, JS, images
│   ├── templates/              # HTML templates
│   └── manage.py               # Django management script
├── venv/                       # Python virtual environment
├── requirements.txt            # Python dependencies
├── start_dev.bat               # Development server script (Windows)
├── start_dev.ps1               # Development server script (PowerShell)
├── .env.example                # Environment variables template
└── SETUP.md                    # This file
```

## 🔧 Environment Details

### Python Virtual Environment
- Location: `./venv/`
- Python Version: 3.13.3
- Activated automatically by startup scripts

### Installed Dependencies
- **Django 5.0.0**: Web framework
- **Django Channels 4.3.1**: WebSocket support
- **Django Extensions 4.1**: Development utilities
- **Daphne 4.2.1**: ASGI server
- **Redis 6.4.0**: Caching and message broker
- **Requests 2.32.5**: HTTP library

### Database
- **Type**: SQLite3
- **Location**: `disaster_dashboard/db.sqlite3`
- **Status**: Migrations applied and ready to use

## 🎯 Available Applications

1. **Dashboard** (`/`) - Mission control center
2. **Detection** (`/detection/`) - AI object detection interface
3. **Pathfinding** (`/pathfinding/`) - Autonomous flight planning

## 🔮 Future AI/ML Setup

The system is prepared for AI/ML integration. When ready, install these packages:

```powershell
venv\Scripts\python.exe -m pip install ultralytics opencv-python torch numpy Pillow
```

## 🛠️ Development Commands

### Run Development Server
```powershell
venv\Scripts\python.exe disaster_dashboard\manage.py runserver
```

### Run System Check
```powershell
venv\Scripts\python.exe disaster_dashboard\manage.py check
```

### Create Migrations
```powershell
venv\Scripts\python.exe disaster_dashboard\manage.py makemigrations
```

### Apply Migrations
```powershell
venv\Scripts\python.exe disaster_dashboard\manage.py migrate
```

### Create Superuser
```powershell
venv\Scripts\python.exe disaster_dashboard\manage.py createsuperuser
```

## 🌐 Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key variables:
- `DEBUG`: Development mode (True/False)
- `SECRET_KEY`: Django secret key
- `ALLOWED_HOSTS`: Comma-separated host list

## 🔧 Troubleshooting

### PowerShell Execution Policy Error
If you get an execution policy error, use the batch file instead:
```cmd
start_dev.bat
```

### Virtual Environment Issues
If the virtual environment is corrupted, recreate it:
```powershell
rmdir /s venv
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Port Already in Use
If port 8000 is busy, specify a different port:
```powershell
venv\Scripts\python.exe disaster_dashboard\manage.py runserver 8080
```

## 📞 Support

- Check the main `README.md` for project documentation
- Review `WARP.md` for additional technical details
- Ensure all dependencies in `requirements.txt` are installed

---

🦅 **Happy Coding with Vihangam!**