# 🚁 Vihangam - Drone Disaster Management System

<div align="center">
  <img src="static/images/logo.png" alt="Vihangam Logo" width="200" height="200">
  <h3>Advanced AI-Powered Disaster Response Platform</h3>
  <p><em>Low-altitude navigation, real-time detection, and intelligent pathfinding for emergency response operations</em></p>
</div>

---

## 🌟 Features

- **🎯 Real-time AI Object Detection** - YOLO-based disaster detection with confidence scoring
- **🧠 A* Pathfinding Algorithm** - Low-altitude navigation with dynamic obstacle avoidance
- **📊 Interactive Dashboard** - Comprehensive mission control and analytics
- **🗺️ Terrain-Aware Navigation** - Elevation mapping and clearance monitoring
- **⚡ Real-time Updates** - WebSocket-powered live data streaming
- **📱 Responsive Design** - Dark theme with glassmorphism UI

---

## 🛠️ Tech Stack

### **Backend Framework**
<div align="center">
  <img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=green" alt="Django">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</div>

- **Django 5.2.6** - High-level Python web framework
- **Django Channels** - WebSocket support for real-time communication
- **Python 3.x** - Core programming language

### **Frontend Technologies**
<div align="center">
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5">
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3">
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript">
  <img src="https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white" alt="Bootstrap">
</div>

- **HTML5** - Semantic markup and structure
- **CSS3** - Advanced styling with glassmorphism effects
- **JavaScript ES6+** - Interactive functionality and A* algorithm implementation
- **Bootstrap 5.3** - Responsive grid system and components

### **UI/UX Libraries**
<div align="center">
  <img src="https://img.shields.io/badge/Font_Awesome-339AF0?style=for-the-badge&logo=fontawesome&logoColor=white" alt="Font Awesome">
  <img src="https://img.shields.io/badge/Leaflet-199900?style=for-the-badge&logo=leaflet&logoColor=white" alt="Leaflet">
  <img src="https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chart.js&logoColor=white" alt="Chart.js">
</div>

- **Font Awesome 6.0** - Comprehensive icon library
- **Leaflet 1.9.4** - Interactive mapping and geolocation
- **Chart.js 4.4** - Data visualization and analytics charts

### **AI & Computer Vision**
<div align="center">
  <img src="https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLO">
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Pillow-306998?style=for-the-badge&logo=python&logoColor=white" alt="Pillow">
</div>

- **YOLOv8** - Real-time object detection for disaster identification
- **Pillow (PIL)** - Image processing and manipulation
- **Custom A* Implementation** - Pathfinding algorithm for low-altitude navigation

### **Database & Storage**
<div align="center">
  <img src="https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
</div>

- **SQLite** - Lightweight database for development
- **Django ORM** - Object-relational mapping

### **Development Tools**
<div align="center">
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git">
  <img src="https://img.shields.io/badge/VS_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white" alt="VS Code">
  <img src="https://img.shields.io/badge/pip-3776AB?style=for-the-badge&logo=pypi&logoColor=white" alt="pip">
</div>

- **Git** - Version control system
- **pip** - Package management
- **Virtual Environment** - Isolated Python environment

---

## 📁 Project Architecture

```
vihangam_disaster_dashboard/
├── 🎮 apps/
│   ├── 📊 dashboard/          # Mission control and analytics
│   ├── 🎯 detection/          # AI object detection system
│   └── 🧠 pathfinding/        # A* navigation algorithms
├── 🎨 static/
│   ├── css/                   # Custom stylesheets
│   ├── js/                    # JavaScript modules
│   └── images/                # Assets and logos
├── 📄 templates/              # HTML templates
├── 🌐 disaster_dashboard/     # Django project settings
└── 🔧 manage.py              # Django management script
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment support

### 1. **Environment Setup**
```bash
# Clone the repository
git clone <repository-url>
cd disaster_dashboard

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Database Migration**
```bash
python manage.py migrate
```

### 4. **Collect Static Files**
```bash
python manage.py collectstatic --noinput
```

### 5. **Run Development Server**
```bash
python manage.py runserver
```

### 6. **Access the Application**
- **Main Dashboard**: http://127.0.0.1:8000/
- **AI Detection**: http://127.0.0.1:8000/detection/
- **A* Pathfinding**: http://127.0.0.1:8000/pathfinding/
- **Admin Panel**: http://127.0.0.1:8000/admin/
  - Username: `admin`
  - Password: `admin123`

---

## 🎮 Application Modules

### 📊 **Dashboard Module**
- **Real-time mission monitoring**
- **Drone status tracking**
- **Weather and environmental data**
- **Mission statistics and analytics**

### 🎯 **Detection Module**
- **YOLO-based object detection**
- **Confidence threshold adjustment**
- **Real-time bounding box visualization**
- **Detection history and analytics**
- **Export functionality**

### 🧠 **Pathfinding Module**
- **A* algorithm implementation**
- **Low-altitude navigation (10-100m)**
- **Dynamic obstacle avoidance**
- **Terrain awareness system**
- **Real-time path recalculation**

---

## 🔧 Development

### **Adding New Features**
1. Create new Django app: `python manage.py startapp feature_name`
2. Add to `INSTALLED_APPS` in `settings.py`
3. Create templates in `templates/feature_name/`
4. Add CSS in `static/css/feature_name.css`
5. Run migrations: `python manage.py makemigrations && python manage.py migrate`

### **CSS Architecture**
- `base.css` - Global styles and dark theme
- `dashboard.css` - Dashboard-specific styling
- `detection.css` - Detection module styling
- `pathfinding.css` - A* pathfinding styling

### **JavaScript Modules**
- A* pathfinding algorithm implementation
- Real-time data updates
- Interactive UI components
- Chart.js integrations

---

## 📊 Performance Features

- **⚡ Real-time Updates**: WebSocket-powered live data
- **🎨 Glassmorphism UI**: Modern dark theme with blur effects
- **📱 Responsive Design**: Mobile-first approach
- **🚀 Optimized Assets**: Minified CSS/JS and image optimization
- **🧠 Efficient Algorithms**: Optimized A* pathfinding implementation
