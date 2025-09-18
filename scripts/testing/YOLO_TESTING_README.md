# 🚁 Vihangam YOLO Testing

Standalone testing environment for the Vihangam disaster detection YOLO model.

## Usage

### Test a single image:
```bash
python test_yolo_model.py path/to/image.jpg
```

### Test with custom model:
```bash
python test_yolo_model.py image.jpg --model path/to/custom/model.pt
```

### Test with lower confidence:
```bash
python test_yolo_model.py image.jpg --confidence 0.1
```

### Test and save annotated images:
```bash
python test_yolo_model.py image.jpg --save
```

### Test all images in a directory:
```bash
python test_yolo_model.py /path/to/images/ --save
```

## Requirements

- ultralytics
- opencv-python
- pillow
- numpy

## Model Classes

- **human**: Critical priority (red bounding boxes)
- **debris**: Warning priority (orange bounding boxes)

## Examples

```bash
# Test with YOLOv8n (will download automatically)
python test_yolo_model.py disaster.jpg --model yolov8n.pt

# Test custom model with low confidence
python test_yolo_model.py disaster.jpg --model best.pt --confidence 0.1 --save
```