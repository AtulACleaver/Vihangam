# 🤖 YOLO Detection System

AI-powered object detection for disaster management focusing on human and debris detection.

## 📁 Folder Structure

```
yolo_detection/
├── scripts/             # Main detection and training scripts
│   ├── detect_objects.py    # Run object detection
│   └── demo_training.py     # Training demo script
├── models/              # Trained models and training outputs
│   ├── runs/               # Training run results
│   └── custom_models/      # Custom model configurations
├── data/                # Dataset and configuration
│   ├── data.yaml          # Dataset configuration
│   ├── images/            # Training and validation images
│   └── labels/            # YOLO format labels
└── results/             # Detection outputs
    └── detection_results/  # Annotated images and JSON reports
```

## 🚀 Quick Usage

### Run Detection on Single Image
```powershell
venv\Scripts\python.exe yolo_detection\scripts\detect_objects.py --image path\to\image.jpg --confidence 0.25
```

### Batch Process Multiple Images
```powershell
venv\Scripts\python.exe yolo_detection\scripts\detect_objects.py --directory path\to\images\folder --confidence 0.25
```

### Available Options
- `--image`: Single image file
- `--directory`: Folder with multiple images  
- `--confidence`: Detection confidence threshold (0.0-1.0, default: 0.25)
- `--model`: Custom model path (optional)
- `--no-save`: Don't save results (for testing only)

## 🎯 Detection Classes

- **Human** 🔴 **CRITICAL PRIORITY**: Search and rescue operations
- **Debris** 🟡 **WARNING PRIORITY**: Infrastructure damage assessment

## 📊 Outputs

Detection results are saved to `results/detection_results/`:
- **Images**: Annotated images with bounding boxes
- **JSON**: Detailed detection reports with coordinates and confidence scores
- **Batch Results**: Summary reports for multiple images

## 🔧 Current Model

- **Model**: `models/runs/detect/disaster_demo_20250918_201832/weights/best.pt`
- **Size**: ~6MB
- **Classes**: 2 (human, debris)
- **Performance**: Fast inference suitable for real-time processing

## 📚 Documentation

For detailed setup and training information, see `../docs/COMPLETE_SETUP_GUIDE.md`