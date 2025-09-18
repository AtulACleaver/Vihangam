# Vihangam 🦅 - Disaster Management Drone System

Vihangam (Vi-han-gam) is a drone software system designed for disaster management, focusing on search and rescue operations.

## Features

- Object detection for humans and debris using custom YOLO models
- Web interface for uploading and analyzing images
- Real-time processing with confidence scoring
- Pathfinding algorithms for rescue coordination
- Django-based dashboard for mission planning

## Architecture

```
Vihangam System
├── YOLO Detection Engine
├── Django Web Dashboard
├── Pathfinding Module
├── Analytics
└── Alert System
```

## Project Structure

```
Vihangam/
├── disaster_dashboard/         # Django web app
│   ├── apps/
│   ├── templates/
│   ├── static/
│   └── manage.py
│
├── yolo_detection/            # YOLO training & detection
│   ├── data/                  # Training datasets
│   ├── models/                # Model configs
│   └── results/               # Detection outputs
│
├── scripts/                   # Utilities
│   ├── training/              # Model training
│   ├── testing/               # Testing tools
│   └── integration/           # System integration
│
├── docs/
├── requirements.txt
└── README.md
```

## Setup

```bash
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
```

Access the dashboard at `http://localhost:8000`

## Model Training

Put training images in `yolo_detection/data/images/train/` and labels in `yolo_detection/data/labels/train/`.

```bash
# Train model
cd scripts/training
python train_yolo_model.py

# Test model
cd scripts/testing
python test_yolo_model.py image.jpg --model model.pt
```

## Testing

```bash
# Test single image
python scripts/testing/test_yolo_model.py image.jpg --confidence 0.25

# Test directory
python scripts/testing/test_yolo_model.py images/ --save

# Validate model
python scripts/testing/validate_yolo_model.py --model model.pt
```

## Usage

### Web Interface
1. Upload images through the dashboard
2. View detection results
3. Plan rescue operations based on findings

### Command Line
```bash
python scripts/testing/test_yolo_model.py disaster_scene.jpg --save
python scripts/training/train_yolo_model.py --epochs 100
```

## Performance

- Detection speed: 8-20 FPS
- Human detection accuracy: 95%+
- Response time: <2 seconds

## Development

Create feature branches and add tests in `scripts/testing/` for new functionality.

## License

Developed for disaster management and emergency response.
