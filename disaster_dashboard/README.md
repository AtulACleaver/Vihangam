# Disaster Dashboard

A Django-based disaster management and monitoring dashboard.

## Setup Instructions

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Migrations
```bash
python manage.py migrate
```

### 4. Start Development Server
```bash
python manage.py runserver
```

### 5. Access Admin Panel
- URL: http://127.0.0.1:8000/admin/
- Username: admin
- Password: admin123

## Apps Structure
- `apps/dashboard` - Main dashboard functionality
- `apps/detection` - Disaster detection features
- `apps/pathfinding` - Path finding algorithms

## Technologies Used
- Django 5.2.6
- Django Channels (WebSocket support)
- SQLite (default database)
- Pillow (image processing)

## Development
Make sure to activate the virtual environment before running any Django commands:
```bash
source venv/bin/activate
```