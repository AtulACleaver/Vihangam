# Scripts

Utility scripts for the Vihangam system.

## Directory Structure

```
scripts/
├── training/              # Model training
├── testing/               # Testing and validation
├── integration/           # System integration
├── start_server.py        # Production server
└── start_simple_server.py # Development server
```

## Training Scripts

- `train_yolo_model.py` - Main training script
- `train_cpu.py` - CPU training for slower machines
- `start_training.py` - Training setup

Usage:
```bash
cd scripts/training
python train_yolo_model.py --epochs 100 --batch-size 16
```

## Testing Scripts

- `test_yolo_model.py` - Test models on images
- `validate_yolo_model.py` - Model validation
- `test_real_detection.py` - Real-world testing
- `test_real_disaster.py` - Disaster scenarios
- `test_validated_model.py` - Validated testing

Usage:
```bash
cd scripts/testing
python test_yolo_model.py image.jpg --confidence 0.25 --save
python test_yolo_model.py images_folder/ --confidence 0.25
python validate_yolo_model.py --model model.pt
```

## Integration Scripts

- `integrate_custom_model.py` - Model integration
- `integrate_with_vihangam.py` - System integration
- `test_django_integration.py` - Django testing
- `test_upload_detection.py` - Upload testing
- `yolo_handler_custom.py` - Custom YOLO handler

Usage:
```bash
cd scripts/integration
python test_django_integration.py
python test_upload_detection.py
```

## Server Scripts

Production:
```bash
python scripts/start_server.py
```

Development:
```bash
python scripts/start_simple_server.py
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```
