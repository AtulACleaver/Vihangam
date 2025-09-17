import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import logging
from django.conf import settings
import os
import time

logger = logging.getLogger(__name__)

class YOLOHandler:
    def __init__(self, model_path=None):
        """
        Initialize YOLO model
        Args:
            model_path: Path to custom model, defaults to YOLOv8n
        """
        self.model_path = model_path or 'yolov8n.pt'
        self.model = None
        self.load_model()
        
        # Disaster-specific classes mapping
        self.disaster_classes = {
            0: 'person',
            2: 'car',
            3: 'motorbike',
            5: 'bus',
            7: 'truck',
            8: 'boat',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow'
        }
        
        # Priority classes for disaster response
        self.high_priority_classes = [0, 2, 3, 5, 7]  # person, car, motorbike, bus, truck
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_objects(self, image, confidence_threshold=0.5):
        """
        Perform object detection on image
        Args:
            image: PIL Image or numpy array or file path
            confidence_threshold: Minimum confidence for detections
        Returns:
            dict: Detection results with bounding boxes, classes, confidences
        """
        if self.model is None:
            raise ValueError("YOLO model not loaded")
        
        try:
            start_time = time.time()
            
            # Handle different input types
            if isinstance(image, str):
                # File path
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                input_image = image
            elif isinstance(image, Image.Image):
                # PIL Image
                input_image = image
            elif isinstance(image, np.ndarray):
                # Numpy array
                input_image = image
            else:
                raise ValueError("Unsupported image type")
            
            # Run inference
            results = self.model(input_image, conf=confidence_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names.get(class_id, 'unknown')
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': round(confidence, 3),
                            'class_id': class_id,
                            'class_name': class_name,
                            'is_disaster_related': self.is_disaster_class(class_id),
                            'is_high_priority': class_id in self.high_priority_classes,
                            'area': int((x2 - x1) * (y2 - y1))
                        }
                        detections.append(detection)
            
            processing_time = time.time() - start_time
            
            # Sort detections by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'detections': detections,
                'count': len(detections),
                'disaster_related_count': sum(1 for det in detections if det['is_disaster_related']),
                'high_priority_count': sum(1 for det in detections if det['is_high_priority']),
                'processing_time': round(processing_time, 3),
                'average_confidence': round(np.mean([d['confidence'] for d in detections]), 3) if detections else 0,
                'image_shape': getattr(input_image, 'shape', None)
            }
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                'detections': [], 
                'count': 0, 
                'disaster_related_count': 0,
                'high_priority_count': 0,
                'processing_time': 0,
                'average_confidence': 0,
                'error': str(e)
            }
    
    def is_disaster_class(self, class_id):
        """Check if detected object is disaster-related"""
        # Customize this based on your specific disaster detection needs
        disaster_relevant_classes = [0, 2, 3, 5, 7, 8, 15, 16, 17, 18, 19, 20]
        return class_id in disaster_relevant_classes
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            raise ValueError("Could not load image")
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            is_high_priority = detection.get('is_high_priority', False)
            
            # Choose color based on priority
            if is_high_priority:
                color = (0, 0, 255)  # Red for high priority
            elif detection['is_disaster_related']:
                color = (0, 165, 255)  # Orange for disaster-related
            else:
                color = (0, 255, 0)  # Green for normal objects
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return None
        
        return {
            'model_path': self.model_path,
            'model_type': type(self.model).__name__,
            'class_names': list(self.model.names.values()),
            'num_classes': len(self.model.names),
            'disaster_classes': list(self.disaster_classes.values()),
            'high_priority_classes': [self.model.names.get(i, f'class_{i}') for i in self.high_priority_classes]
        }
    
    def process_video_frame(self, frame, confidence_threshold=0.5):
        """Process a single video frame for real-time detection"""
        return self.detect_objects(frame, confidence_threshold)


# Global instance to avoid reloading model
_yolo_detector = None

def get_yolo_detector(model_path=None):
    """Get or create YOLO detector instance"""
    global _yolo_detector
    if _yolo_detector is None or (model_path and _yolo_detector.model_path != model_path):
        _yolo_detector = YOLOHandler(model_path)
    return _yolo_detector

def reset_yolo_detector():
    """Reset the global YOLO detector (useful for model switching)"""
    global _yolo_detector
    _yolo_detector = None