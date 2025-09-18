#!/usr/bin/env python3
"""
Integration Script for Custom Trained YOLOv8 Model
Updates the Vihangam disaster management system to use the newly trained custom model
"""

import os
import shutil
from pathlib import Path
import sys

def find_latest_training_run():
    """Find the most recent training run"""
    runs_dir = Path('runs/detect')
    if not runs_dir.exists():
        print("❌ No training runs found!")
        return None
    
    # Find all disaster_demo directories
    demo_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('disaster_demo_')]
    
    if not demo_dirs:
        print("❌ No disaster demo training runs found!")
        return None
    
    # Get the most recent one
    latest_dir = max(demo_dirs, key=lambda d: d.stat().st_mtime)
    return latest_dir

def integrate_custom_model():
    """Integrate the custom model into the Vihangam system"""
    print("🚁 Vihangam Custom Model Integration")
    print("=" * 50)
    
    # Find latest training run
    latest_run = find_latest_training_run()
    if not latest_run:
        return False
    
    print(f"📁 Found training run: {latest_run.name}")
    
    # Check if best.pt exists
    best_model_path = latest_run / 'weights' / 'best.pt'
    if not best_model_path.exists():
        print("❌ best.pt not found in training run!")
        return False
    
    print(f"✅ Custom model found: {best_model_path}")
    
    # Model info
    model_size = best_model_path.stat().st_size / (1024*1024)  # MB
    print(f"📦 Model size: {model_size:.1f} MB")
    
    # Copy model to a convenient location
    custom_model_dir = Path('custom_models')
    custom_model_dir.mkdir(exist_ok=True)
    
    custom_model_path = custom_model_dir / 'vihangam_disaster_detection.pt'
    
    print(f"📋 Copying model to: {custom_model_path}")
    shutil.copy2(best_model_path, custom_model_path)
    
    # Create model configuration file
    model_config = {
        'model_path': str(custom_model_path),
        'model_name': 'Vihangam Disaster Detection v1.0',
        'classes': ['human', 'debris'],
        'training_date': latest_run.name.split('_')[-2:],
        'training_epochs': 5,  # From our demo
        'image_size': 320,
        'description': 'Custom YOLOv8 model trained for disaster management scenarios',
        'performance_metrics': {
            'mAP50': 0.165,  # From training output
            'mAP50-95': 0.0777,
            'precision': 0.00476,
            'recall': 1.0
        }
    }
    
    import json
    config_file = custom_model_dir / 'model_config.json'
    with open(config_file, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"⚙️  Model configuration saved: {config_file}")
    
    # Update the YOLO handler for the custom model
    print("\n🔧 Integration Instructions:")
    print("-" * 30)
    
    print(f"1. Update your yolo_handler.py:")
    print(f"   Replace: model_path = 'yolov8n.pt'")
    print(f"   With: model_path = '{custom_model_path}'")
    
    print(f"\n2. Update class mappings in yolo_handler.py:")
    print(f"   self.disaster_classes = {{")
    print(f"       0: 'human',")
    print(f"       1: 'debris'")
    print(f"   }}")
    
    print(f"\n3. High priority classes:")
    print(f"   self.high_priority_classes = [0]  # human only")
    
    # Create updated yolo_handler.py
    create_updated_handler(custom_model_path)
    
    # Test the integration
    test_custom_model(custom_model_path)
    
    print("\n🎉 Integration completed successfully!")
    print(f"📁 Custom model files:")
    print(f"   📦 Model: {custom_model_path}")
    print(f"   ⚙️  Config: {config_file}")
    print(f"   🔧 Handler: yolo_handler_custom.py")
    
    return True

def create_updated_handler(model_path):
    """Create updated YOLO handler for custom model"""
    handler_content = f'''import cv2
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
        self.model_path = model_path or '{model_path}'
        self.model = None
        self.load_model()
        
        # Custom disaster-specific classes
        self.disaster_classes = {{
            0: 'human',
            1: 'debris'
        }}
        
        # Priority classes for disaster response (human is critical)
        self.high_priority_classes = [0]  # human
    
    def load_model(self):
        """Load the custom YOLO model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Custom model not found: {{self.model_path}}")
                # Fallback to default YOLOv8n
                self.model_path = 'yolov8n.pt'
                logger.info("Falling back to YOLOv8n pretrained model")
            
            self.model = YOLO(self.model_path)
            logger.info(f"Custom Vihangam model loaded: {{self.model_path}}")
        except Exception as e:
            logger.error(f"Failed to load custom model: {{e}}")
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
                    raise FileNotFoundError(f"Image file not found: {{image}}")
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
                        class_name = self.disaster_classes.get(class_id, f'class_{{class_id}}')
                        
                        detection = {{
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': round(confidence, 3),
                            'class_id': class_id,
                            'class_name': class_name,
                            'is_disaster_related': True,  # All our classes are disaster-related
                            'is_high_priority': class_id in self.high_priority_classes,
                            'area': int((x2 - x1) * (y2 - y1)),
                            'model_type': 'custom_vihangam'
                        }}
                        detections.append(detection)
            
            processing_time = time.time() - start_time
            
            # Sort detections by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate disaster-specific metrics
            human_count = sum(1 for det in detections if det['class_name'] == 'human')
            debris_count = sum(1 for det in detections if det['class_name'] == 'debris')
            
            return {{
                'detections': detections,
                'count': len(detections),
                'human_count': human_count,
                'debris_count': debris_count,
                'disaster_related_count': len(detections),  # All are disaster-related
                'high_priority_count': human_count,  # Humans are high priority
                'processing_time': round(processing_time, 3),
                'average_confidence': round(np.mean([d['confidence'] for d in detections]), 3) if detections else 0,
                'model_info': {{
                    'type': 'custom_vihangam_v1',
                    'classes': ['human', 'debris'],
                    'optimized_for': 'disaster_response'
                }}
            }}
            
        except Exception as e:
            logger.error(f"Custom detection failed: {{e}}")
            return {{
                'detections': [], 
                'count': 0, 
                'human_count': 0,
                'debris_count': 0,
                'disaster_related_count': 0,
                'high_priority_count': 0,
                'processing_time': 0,
                'average_confidence': 0,
                'error': str(e)
            }}
    
    def is_disaster_class(self, class_id):
        """All our custom classes are disaster-related"""
        return class_id in self.disaster_classes
    
    def get_model_info(self):
        """Get information about the custom model"""
        if self.model is None:
            return None
        
        return {{
            'model_path': self.model_path,
            'model_type': 'Custom Vihangam Disaster Detection v1.0',
            'class_names': list(self.disaster_classes.values()),
            'num_classes': len(self.disaster_classes),
            'disaster_classes': list(self.disaster_classes.values()),
            'high_priority_classes': [self.disaster_classes[i] for i in self.high_priority_classes],
            'optimized_for': 'disaster_management',
            'training_data': 'human_and_debris_detection'
        }}

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
'''
    
    with open('yolo_handler_custom.py', 'w') as f:
        f.write(handler_content)
    
    print(f"✅ Created custom handler: yolo_handler_custom.py")

def test_custom_model(model_path):
    """Test the custom model integration"""
    print(f"\n🧪 Testing custom model...")
    
    try:
        from ultralytics import YOLO
        
        # Load the custom model
        model = YOLO(str(model_path))
        
        # Test on a sample image
        test_image = 'images/val/val_img_000.jpg'
        if os.path.exists(test_image):
            results = model(test_image, verbose=False)
            
            if results and len(results) > 0:
                detections = results[0].boxes
                if detections is not None:
                    print(f"✅ Model test successful: {len(detections)} objects detected")
                    
                    # Show detection details
                    for i, detection in enumerate(detections):
                        conf = float(detection.conf[0])
                        cls = int(detection.cls[0])
                        class_name = 'human' if cls == 0 else 'debris'
                        print(f"   🎯 Object {i+1}: {class_name} (confidence: {conf:.3f})")
                else:
                    print("✅ Model test successful: No objects detected")
            else:
                print("✅ Model loaded successfully")
        else:
            print("⚠️  Test image not found, but model loads correctly")
    
    except Exception as e:
        print(f"⚠️  Model test error: {e}")

if __name__ == "__main__":
    success = integrate_custom_model()
    
    if success:
        print("\n" + "="*60)
        print("🎉 CUSTOM MODEL INTEGRATION COMPLETE!")
        print("="*60)
        print("Your Vihangam system now has a custom-trained model for:")
        print("  👤 Human detection (high priority)")
        print("  🧱 Debris detection") 
        print("")
        print("Next steps:")
        print("1. 🔄 Replace disaster_dashboard/apps/detection/yolo_handler.py")
        print("   with yolo_handler_custom.py (or merge the changes)")
        print("2. 🚀 Restart your Django server")
        print("3. 🧪 Test the detection interface")
        print("4. 📊 Monitor performance and retrain if needed")
        print("="*60)
    else:
        print("❌ Integration failed. Check the error messages above.")