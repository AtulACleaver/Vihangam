Vihangam 🦅 - Disaster Management Drone System

Vihangam (Vi-han-gam) is a drone software system designed for disaster management, focusing on search and rescue operations.

✨ Features

🧑‍🤝‍🧑 Object Detection: Humans and debris detection using custom YOLO models

🖼️ Web Interface: Upload and analyze images directly

⚡ Real-time Processing: Confidence scoring with immediate results

🛰️ Pathfinding Algorithms: Smart rescue coordination and navigation

🖥️ Mission Dashboard: Django-based control panel for planning and monitoring

🏗️ Architecture

Vihangam System

├── YOLO Detection Engine
├── Django Web Dashboard
├── Pathfinding Module
├── Analytics
└── Alert System

📂 Project Structure
Vihangam/
├── disaster_dashboard/         # Django web app
│   ├── apps/
│   ├── templates/
│   ├── static/
│   └── manage.py
│
├── yolo_detection/             # YOLO training & detection
│   ├── data/                   # Training datasets
│   ├── models/                 # Model configs
│   └── results/                # Detection outputs
│
├── scripts/                    # Utilities
│   ├── training/               # Model training
│   ├── testing/                # Testing tools
│   └── integration/            # System integration
│
├── docs/
├── requirements.txt
└── README.md

⚙️ Setup
# Clone and setup
git clone <repository-url>
cd Vihangam
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup database
cd disaster_dashboard
python manage.py migrate
python manage.py createsuperuser

# Start server
python manage.py runserver


🔗 Access the dashboard at: http://localhost:8000

🧠 Model Training

Place training images in:

yolo_detection/data/images/train/
yolo_detection/data/labels/train/


Train model:

cd scripts/training
python train_yolo_model.py


Test model:

cd scripts/testing
python test_yolo_model.py image.jpg --model model.pt

🧪 Testing
# Test single image
python scripts/testing/test_yolo_model.py image.jpg --confidence 0.25

# Test directory
python scripts/testing/test_yolo_model.py images/ --save

# Validate model
python scripts/testing/validate_yolo_model.py --model model.pt

🚀 Current Development State
✅ What’s Working

Complete Django web interface with mission dashboard

Upload and analyze images with detection results

Responsive UI/UX with Bootstrap and Font Awesome

Database setup and migrations configured

Model training and testing scripts ready

Command-line utilities functional

🔄 What’s Simulated (Ready for Integration)

Real-time YOLO detection from drone camera feed (currently image upload only)

Live GPS/telemetry data (mock data in use)

WebSocket real-time updates (framework in place)

Drone communication protocols (to be implemented)

Advanced analytics and alert workflows

🛠️ Development Environment Ready

Virtual environment with Django, Ultralytics, OpenCV

Requirements managed in requirements.txt

Static files and templates fully configured

Documentation folder initialized

🚀 Usage
Web Interface

Upload images through the dashboard

View YOLO detection results

Plan rescue operations based on findings

Command Line
python scripts/testing/test_yolo_model.py disaster_scene.jpg --save
python scripts/training/train_yolo_model.py --epochs 100

📊 Performance

⚡ Detection speed: 8–20 FPS

👤 Human detection accuracy: 95%+

⏱️ Response time: < 2 seconds

👩‍💻 Development

Create feature branches for new modules

Add tests in scripts/testing/ for validation

Follow modular design for scalability

📜 License

Developed for disaster management and emergency response 🌍

Would you like me to also design a visual flow diagram (architecture + workflow) for this, like the tree structure but more diagrammatic (boxes + arrows) so it’s presentation-ready?

You said:
im not able to add the bullet list symbol, please give it to me in a format where i can copy that too
ChatGPT said:

Got it 👍 I’ll give you the full README again but this time with proper Markdown bullet symbols (-) so you can directly copy-paste without losing the formatting.

Vihangam 🦅 - Disaster Management Drone System

Vihangam (Vi-han-gam) is a drone software system designed for disaster management, focusing on search and rescue operations.

✨ Features

🧑‍🤝‍🧑 Object Detection: Humans and debris detection using custom YOLO models

🖼️ Web Interface: Upload and analyze images directly

⚡ Real-time Processing: Confidence scoring with immediate results

🛰️ Pathfinding Algorithms: Smart rescue coordination and navigation

🖥️ Mission Dashboard: Django-based control panel for planning and monitoring

🏗️ Architecture

Vihangam System

├── YOLO Detection Engine
├── Django Web Dashboard
├── Pathfinding Module
├── Analytics
└── Alert System

📂 Project Structure
Vihangam/
├── disaster_dashboard/         # Django web app
│   ├── apps/
│   ├── templates/
│   ├── static/
│   └── manage.py
│
├── yolo_detection/             # YOLO training & detection
│   ├── data/                   # Training datasets
│   ├── models/                 # Model configs
│   └── results/                # Detection outputs
│
├── scripts/                    # Utilities
│   ├── training/               # Model training
│   ├── testing/                # Testing tools
│   └── integration/            # System integration
│
├── docs/
├── requirements.txt
└── README.md

⚙️ Setup
# Clone and setup
git clone <repository-url>
cd Vihangam
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup database
cd disaster_dashboard
python manage.py migrate
python manage.py createsuperuser

# Start server
python manage.py runserver


🔗 Access the dashboard at: http://localhost:8000

🧠 Model Training

Place training images in:

yolo_detection/data/images/train/
yolo_detection/data/labels/train/


Train model:

cd scripts/training
python train_yolo_model.py


Test model:

cd scripts/testing
python test_yolo_model.py image.jpg --model model.pt

🧪 Testing
# Test single image
python scripts/testing/test_yolo_model.py image.jpg --confidence 0.25

# Test directory
python scripts/testing/test_yolo_model.py images/ --save

# Validate model
python scripts/testing/validate_yolo_model.py --model model.pt

🚀 Current Development State
✅ What’s Working

Complete Django web interface with mission dashboard

Upload and analyze images with detection results

Responsive UI/UX with Bootstrap and Font Awesome

Database setup and migrations configured

Model training and testing scripts ready

Command-line utilities functional

🔄 What’s Simulated (Ready for Integration)

Real-time YOLO detection from drone camera feed (currently image upload only)

Live GPS/telemetry data (mock data in use)

WebSocket real-time updates (framework in place)

Drone communication protocols (to be implemented)

Advanced analytics and alert workflows

🛠️ Development Environment Ready

Virtual environment with Django, Ultralytics, OpenCV

Requirements managed in requirements.txt

Static files and templates fully configured

Documentation folder initialized

🚀 Usage
Web Interface

Upload images through the dashboard

View YOLO detection results

Plan rescue operations based on findings

Command Line
python scripts/testing/test_yolo_model.py disaster_scene.jpg --save
python scripts/training/train_yolo_model.py --epochs 100

📊 Performance

⚡ Detection speed: 8–20 FPS

👤 Human detection accuracy: 95%+

⏱️ Response time: < 2 seconds

👩‍💻 Development

Create feature branches for new modules

Add tests in scripts/testing/ for validation

Follow modular design for scalability

📜 License

Developed for disaster management and emergency response 🌍
