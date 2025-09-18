import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import logging
from django.conf import settings
import os
import time

logger = logging.getLogger(__name__)

class VihangamCustomYOLOHandler:
    def __init__(self, model_path=None):
        """
        Initialize custom YOLO model for Vihangam disaster detection
        
        Args:
            model_path: Path to custom model, defaults to trained disaster model
        """
        self.model_path = model_path or 'custom_models\vihangam_disaster_detection.pt'
        self.model = None
        self.load_model()
        
        # Custom disaster-specific classes
        self.disaster_classes = {
            0: 'human',
            1: 'debris'
        }
        
        # Priority classes for disaster response (human is critical)
        self.high_priority_classes = [0]  # human
    
    def load_model(self):
        """Load the custom YOLO model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Custom model not found: {self.model_path}")
                # Fallback to default YOLOv8n
                self.model_path = 'yolov8n.pt'
                logger.info("Falling back to YOLOv8n pretrained model")
            
            self.model = YOLO(self.model_path)
            logger.info(f"Custom Vihangam model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            raise
    
    def detect_objects(self, image, confidence_threshold=0.25):
        """
        Perform disaster object detection on image
        Optimized for human and debris detection
        
        Args:
            image: PIL Image or numpy array or file path
            confidence_threshold: Minimum confidence for detections (lowered for custom model)
        Returns:
            dict: Detection results with disaster-specific metadata
        """
        if self.model is None:
            raise ValueError("Custom YOLO model not loaded")
        
        try:
            start_time = time.time()
            
            # Handle different input types
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                input_image = image
            elif isinstance(image, Image.Image):
                input_image = image
            elif isinstance(image, np.ndarray):
                input_image = image
            else:
                raise ValueError("Unsupported image type")
            
            # Run inference with custom model
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
                        
                        # Use our custom class names
                        class_name = self.disaster_classes.get(class_id, f'class_{class_id}')
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': round(confidence, 3),
                            'class_id': class_id,
                            'class_name': class_name,
                            'is_disaster_related': True,  # All our classes are disaster-related
                            'is_high_priority': class_id in self.high_priority_classes,
                            'area': int((x2 - x1) * (y2 - y1)),
                            'model_type': 'custom_vihangam'
                        }
                        detections.append(detection)
            
            processing_time = time.time() - start_time
            
            # Sort detections by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate disaster-specific metrics
            human_count = sum(1 for det in detections if det['class_name'] == 'human')
            debris_count = sum(1 for det in detections if det['class_name'] == 'debris')
            
            return {
                'detections': detections,
                'count': len(detections),
                'human_count': human_count,
                'debris_count': debris_count,
                'disaster_related_count': len(detections),  # All are disaster-related
                'high_priority_count': human_count,  # Humans are high priority
                'processing_time': round(processing_time, 3),
                'average_confidence': round(np.mean([d['confidence'] for d in detections]), 3) if detections else 0,
                'model_info': {
                    'type': 'custom_vihangam_v1',
                    'classes': ['human', 'debris'],
                    'optimized_for': 'disaster_response'
                }
            }
            
        except Exception as e:
            logger.error(f"Custom detection failed: {e}")
            return {
                'detections': [], 
                'count': 0, 
                'human_count': 0,
                'debris_count': 0,
                'disaster_related_count': 0,
                'high_priority_count': 0,
                'processing_time': 0,
                'average_confidence': 0,
                'error': str(e)
            }
    
    def is_disaster_class(self, class_id):
        """All our custom classes are disaster-related"""
        return class_id in self.disaster_classes
    
    def get_model_info(self):
        """Get information about the custom model"""
        if self.model is None:
            return None
        
        return {
            'model_path': self.model_path,
            'model_type': 'Custom Vihangam Disaster Detection v1.0',
            'class_names': list(self.disaster_classes.values()),
            'num_classes': len(self.disaster_classes),
            'disaster_classes': list(self.disaster_classes.values()),
            'high_priority_classes': [self.disaster_classes[i] for i in self.high_priority_classes],
            'optimized_for': 'disaster_management',
            'training_data': 'human_and_debris_detection'
        }

# Global instance for the custom model
_vihangam_detector = None

def get_vihangam_detector(model_path=None):
    """Get or create custom Vihangam detector instance"""
    global _vihangam_detector
    if _vihangam_detector is None:
        _vihangam_detector = VihangamCustomYOLOHandler(model_path)
    return _vihangam_detector

def reset_vihangam_detector():
    """Reset the global detector"""
    global _vihangam_detector
    _vihangam_detector = None
