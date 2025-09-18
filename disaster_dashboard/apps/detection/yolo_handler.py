#!/usr/bin/env python3
"""
Enhanced YOLO Handler for Vihangam Disaster Detection
Integrates custom-trained YOLOv8 model for human and debris detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import logging
from django.conf import settings
import os
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Disaster-specific class mapping for custom model
DISASTER_CLASSES = {
    0: 'human',
    1: 'debris'
}

PRIORITY_MAPPING = {
    'human': 'CRITICAL',
    'debris': 'WARNING'  
}

ALERT_COLORS = {
    'human': '#FF0000',      # Red for humans
    'debris': '#FFA500'      # Orange for debris
}

class VihangamYOLOHandler:
    def __init__(self, model_path=None, confidence_threshold=0.25):
        """
        Initialize Vihangam YOLO Handler with custom disaster detection model
        
        Args:
            model_path: Path to custom trained model
            confidence_threshold: Minimum confidence for detections
        """
        # Use custom model path or find it automatically
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = self.find_custom_model()
            
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_loaded = False
        
        logger.info(f"Initializing Vihangam YOLO Handler with model: {self.model_path}")
        
    def find_custom_model(self):
        """Find the custom disaster detection model"""
        # Check for custom model in Django app
        django_model_paths = [
            "apps/detection/models/vihangam_disaster_model_20250918.pt",
            "disaster_dashboard/apps/detection/models/vihangam_disaster_model_20250918.pt",
            os.path.join(os.path.dirname(__file__), "models", "vihangam_disaster_model_20250918.pt")
        ]
        
        for model_path in django_model_paths:
            if os.path.exists(model_path):
                return model_path
        
        # Check in reorganized yolo_detection directory
        yolo_detection_paths = [
            "../../../yolo_detection/models/runs/detect/disaster_demo_20250918_201832/weights/best.pt",
            "../../yolo_detection/models/runs/detect/disaster_demo_20250918_201832/weights/best.pt",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                        "yolo_detection", "models", "runs", "detect", 
                        "disaster_demo_20250918_201832", "weights", "best.pt")
        ]
        
        for model_path in yolo_detection_paths:
            if os.path.exists(model_path):
                return model_path
        
        # Fallback to training directory (legacy)
        training_paths = [
            "runs/detect/disaster_demo_20250918_201832/weights/best.pt",
            "../runs/detect/disaster_demo_20250918_201832/weights/best.pt",
            "../../runs/detect/disaster_demo_20250918_201832/weights/best.pt"
        ]
        
        for model_path in training_paths:
            if os.path.exists(model_path):
                return model_path
                
        # Final fallback to standard YOLOv8
        logger.warning("Custom model not found, using YOLOv8n")
        return 'yolov8n.pt'
        
    def load_model(self):
        """Load the custom YOLO model"""
        try:
            logger.info("Loading custom disaster detection model...")
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            logger.info("✅ Custom model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def detect_objects(self, image_source, confidence=None):
        """
        Detect objects in image using custom model
        
        Args:
            image_source: Image path, numpy array, or PIL image
            confidence: Confidence threshold (optional)
            
        Returns:
            dict: Detection results with disaster-specific information
        """
        if not self.is_loaded:
            if not self.load_model():
                return {"error": "Model not loaded"}
        
        conf_threshold = confidence if confidence is not None else self.confidence_threshold
        
        try:
            start_time = time.time()
            
            # Handle different input types
            if isinstance(image_source, str):
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"Image file not found: {image_source}")
                input_image = image_source
            elif isinstance(image_source, Image.Image):
                input_image = image_source
            elif isinstance(image_source, np.ndarray):
                input_image = image_source
            else:
                raise ValueError("Unsupported image type")
            
            # Run inference
            results = self.model(input_image, conf=conf_threshold, verbose=False)
            
            # Process results
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                
                if boxes is not None:
                    for box in boxes:
                        # Extract detection data
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = DISASTER_CLASSES.get(class_id, 'unknown')
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': round(conf, 3),
                            'class_id': class_id,
                            'class_name': class_name,
                            'priority': PRIORITY_MAPPING.get(class_name, 'INFO'),
                            'alert_color': ALERT_COLORS.get(class_name, '#808080'),
                            'area': int((x2 - x1) * (y2 - y1)),
                            'is_disaster_related': True,  # All our classes are disaster-related
                            'is_high_priority': class_name == 'human'
                        }
                        detections.append(detection)
            
            processing_time = time.time() - start_time
            
            # Create summary
            human_count = sum(1 for d in detections if d['class_name'] == 'human')
            debris_count = sum(1 for d in detections if d['class_name'] == 'debris')
            
            return {
                'detections': detections,
                'summary': {
                    'total_objects': len(detections),
                    'human_count': human_count,
                    'debris_count': debris_count,
                    'critical_alerts': human_count,
                    'warning_alerts': debris_count,
                    'average_confidence': round(np.mean([d['confidence'] for d in detections]), 3) if detections else 0
                },
                'count': len(detections),  # Legacy compatibility
                'disaster_related_count': len(detections),  # All detections are disaster-related
                'high_priority_count': human_count,
                'processing_time': round(processing_time, 3),
                'inference_time_seconds': round(processing_time, 3),
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'model_path': self.model_path,
                    'confidence_threshold': conf_threshold,
                    'classes': list(DISASTER_CLASSES.values())
                }
            }
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {"error": str(e)}
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
            
        try:
            model_size = os.path.getsize(self.model_path) / (1024*1024)  # MB
            
            return {
                'model_path': self.model_path,
                'model_size_mb': round(model_size, 1),
                'classes': list(DISASTER_CLASSES.values()),
                'class_count': len(DISASTER_CLASSES),
                'confidence_threshold': self.confidence_threshold,
                'is_loaded': self.is_loaded
            }
        except Exception as e:
            return {"error": str(e)}
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            raise ValueError("Could not load image")
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            class_name = detection['class_name']
            confidence = detection['confidence']
            priority = detection['priority']
            
            # Color based on priority
            if priority == 'CRITICAL':
                color = (0, 0, 255)  # Red in BGR
            elif priority == 'WARNING':
                color = (0, 165, 255)  # Orange in BGR
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Label background
            cv2.rectangle(image, (x1, y1 - label_height - 4), (x1 + label_width, y1), color, -1)
            
            # Label text
            cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image

# Legacy YOLOHandler class for backward compatibility
class YOLOHandler(VihangamYOLOHandler):
    def __init__(self, model_path=None):
        """Legacy constructor for backward compatibility"""
        super().__init__(model_path)
        
        # Legacy attributes for compatibility
        self.disaster_classes = DISASTER_CLASSES
        self.high_priority_classes = [0]  # human class ID
    
    def is_disaster_class(self, class_id):
        """Check if detected object is disaster-related (legacy compatibility)"""
        return class_id in DISASTER_CLASSES

# Global handler instance
yolo_handler = VihangamYOLOHandler()

# Compatibility functions for existing Django views
def get_yolo_handler():
    """Get the global YOLO handler instance"""
    return yolo_handler

def get_yolo_detector(model_path=None):
    """Get the global YOLO detector instance (legacy compatibility)"""
    if model_path:
        # Create new handler with specific model
        return VihangamYOLOHandler(model_path)
    return yolo_handler

def reset_yolo_detector():
    """Reset the global detector instance"""
    global yolo_handler
    yolo_handler = VihangamYOLOHandler()

def detect_objects_in_frame(frame):
    """Detect objects in a video frame"""
    return yolo_handler.detect_objects(frame)

def load_yolo_model():
    """Load the YOLO model"""
    return yolo_handler.load_model()

def get_model_information():
    """Get model information"""
    return yolo_handler.get_model_info()
