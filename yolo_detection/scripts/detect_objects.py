#!/usr/bin/env python3
"""
Object Detection Script for Custom-Trained YOLOv8 Model
Performs detection on new images and saves results with bounding boxes
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import json

from ultralytics import YOLO
import torch

class VihangamObjectDetector:
    def __init__(self, model_path=None):
        """
        Initialize object detector with custom model
        
        Args:
            model_path: Path to custom trained model
        """
        self.model_path = self.find_model_path(model_path)
        self.model = None
        self.results_dir = Path("detection_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Disaster-specific class colors and priorities
        self.class_colors = {
            'human': (0, 0, 255),      # Red - High priority
            'debris': (255, 165, 0),   # Orange - Medium priority
        }
        
        self.class_priorities = {
            'human': '🔴 CRITICAL',
            'debris': '🟡 WARNING'
        }
        
        print("🚁 Vihangam Object Detection System")
        print("=" * 50)
        print(f"📦 Model: {self.model_path}")
        
    def find_model_path(self, provided_path):
        """Find the trained model file"""
        # Check provided path
        if provided_path and os.path.exists(provided_path):
            return provided_path
            
        # Check common locations
        possible_paths = [
            "runs/detect/train/weights/best.pt",
            "runs/detect/disaster_demo_20250918_201832/weights/best.pt",
            "custom_models/vihangam_disaster_detection.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found model at: {path}")
                return path
                
        # Search for any best.pt in runs directory
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            best_models = list(runs_dir.rglob("best.pt"))
            if best_models:
                latest_model = max(best_models, key=lambda p: p.stat().st_mtime)
                print(f"✅ Using latest model: {latest_model}")
                return str(latest_model)
        
        raise FileNotFoundError("❌ No trained model found!")
    
    def load_model(self):
        """Load the custom trained model"""
        try:
            print(f"🤖 Loading custom model...")
            self.model = YOLO(self.model_path)
            
            # Model info
            model_size = os.path.getsize(self.model_path) / (1024*1024)
            print(f"📊 Model size: {model_size:.1f} MB")
            print(f"🎯 Classes: {list(self.model.names.values())}")
            
            if hasattr(self.model.model, 'parameters'):
                total_params = sum(p.numel() for p in self.model.model.parameters())
                print(f"📈 Parameters: {total_params:,}")
            
            print("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def detect_objects(self, image_path, confidence=0.25, save_results=True):
        """
        Perform object detection on an image
        
        Args:
            image_path: Path to input image
            confidence: Confidence threshold for detections
            save_results: Whether to save detection results
            
        Returns:
            dict: Detection results with metadata
        """
        if not self.model:
            raise ValueError("Model not loaded!")
            
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        print(f"\n🔍 Processing: {Path(image_path).name}")
        print(f"🎚️  Confidence threshold: {confidence}")
        
        try:
            # Load image for processing
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            original_height, original_width = image.shape[:2]
            print(f"📐 Image size: {original_width}x{original_height}")
            
            # Run inference
            start_time = datetime.now()
            results = self.model(image_path, conf=confidence, verbose=False)
            inference_time = (datetime.now() - start_time).total_seconds()
            
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
                        class_name = self.model.names[class_id]
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': round(conf, 3),
                            'class_id': class_id,
                            'class_name': class_name,
                            'priority': self.class_priorities.get(class_name, '⚪ INFO'),
                            'area': int((x2 - x1) * (y2 - y1))
                        }
                        detections.append(detection)
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Create results summary
            human_count = sum(1 for d in detections if d['class_name'] == 'human')
            debris_count = sum(1 for d in detections if d['class_name'] == 'debris')
            
            results_summary = {
                'image_path': image_path,
                'image_size': (original_width, original_height),
                'timestamp': datetime.now().isoformat(),
                'inference_time_seconds': round(inference_time, 3),
                'confidence_threshold': confidence,
                'detections': detections,
                'summary': {
                    'total_objects': len(detections),
                    'human_count': human_count,
                    'debris_count': debris_count,
                    'high_priority_alerts': human_count,
                    'average_confidence': round(np.mean([d['confidence'] for d in detections]), 3) if detections else 0
                }
            }
            
            # Display results
            self.display_results(results_summary)
            
            # Save results if requested
            if save_results:
                output_path = self.save_detection_results(image_path, image, results_summary)
                results_summary['output_path'] = output_path
            
            return results_summary
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            return None
    
    def display_results(self, results):
        """Display detection results in console"""
        print("\n📊 DETECTION RESULTS")
        print("=" * 30)
        
        summary = results['summary']
        print(f"⚡ Processing time: {results['inference_time_seconds']}s")
        print(f"🎯 Objects detected: {summary['total_objects']}")
        
        if summary['human_count'] > 0:
            print(f"🔴 CRITICAL: {summary['human_count']} human(s) detected!")
            
        if summary['debris_count'] > 0:
            print(f"🟡 WARNING: {summary['debris_count']} debris detected")
            
        if summary['total_objects'] == 0:
            print("✅ No objects detected")
            return
        
        print(f"📈 Average confidence: {summary['average_confidence']:.3f}")
        
        print("\n🔍 Detailed detections:")
        for i, detection in enumerate(results['detections'][:5]):  # Show top 5
            bbox = detection['bbox']
            print(f"  {i+1}. {detection['priority']} {detection['class_name']}")
            print(f"     Confidence: {detection['confidence']:.3f}")
            print(f"     Bounding box: ({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})")
            print(f"     Area: {detection['area']} pixels")
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get color for class
            color = self.class_colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            if font:
                bbox_font = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_font[2] - bbox_font[0]
                text_height = bbox_font[3] - bbox_font[1]
            else:
                text_width, text_height = len(label) * 10, 20
            
            # Label background
            draw.rectangle([x1, y1-text_height-4, x1+text_width+8, y1], fill=color)
            
            # Label text
            text_color = (255, 255, 255)  # White text
            if font:
                draw.text((x1+4, y1-text_height-2), label, fill=text_color, font=font)
            else:
                draw.text((x1+4, y1-text_height-2), label, fill=text_color)
        
        # Convert back to BGR for OpenCV
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result_image
    
    def save_detection_results(self, input_path, original_image, results):
        """Save detection results including annotated image and JSON"""
        input_name = Path(input_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output filename
        output_name = f"{input_name}_detected_{timestamp}"
        
        # Draw detections on image
        annotated_image = self.draw_detections(original_image, results['detections'])
        
        # Save annotated image
        image_output_path = self.results_dir / f"{output_name}.jpg"
        cv2.imwrite(str(image_output_path), annotated_image)
        
        # Save results JSON
        json_output_path = self.results_dir / f"{output_name}.json"
        with open(json_output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   📸 Annotated image: {image_output_path}")
        print(f"   📄 Detection data: {json_output_path}")
        
        return str(image_output_path)
    
    def batch_detect(self, image_directory, confidence=0.25):
        """Perform detection on multiple images in a directory"""
        image_dir = Path(image_directory)
        if not image_dir.exists():
            print(f"❌ Directory not found: {image_directory}")
            return
            
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"⚠️  No image files found in {image_directory}")
            return
            
        print(f"🔍 Found {len(image_files)} images for batch processing")
        
        batch_results = []
        for i, image_path in enumerate(image_files):
            print(f"\n📸 Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            try:
                result = self.detect_objects(str(image_path), confidence)
                if result:
                    batch_results.append(result)
            except Exception as e:
                print(f"❌ Failed to process {image_path.name}: {e}")
        
        # Save batch summary
        batch_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images_processed': len(batch_results),
            'total_objects_detected': sum(r['summary']['total_objects'] for r in batch_results),
            'total_humans_detected': sum(r['summary']['human_count'] for r in batch_results),
            'total_debris_detected': sum(r['summary']['debris_count'] for r in batch_results),
            'results': batch_results
        }
        
        batch_file = self.results_dir / f"batch_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
            
        print(f"\n📊 BATCH SUMMARY")
        print("=" * 20)
        print(f"✅ Processed: {len(batch_results)} images")
        print(f"🎯 Total objects: {batch_summary['total_objects_detected']}")
        print(f"🔴 Humans: {batch_summary['total_humans_detected']}")
        print(f"🟡 Debris: {batch_summary['total_debris_detected']}")
        print(f"💾 Batch report: {batch_file}")

def main():
    """Main detection function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Object Detection with Custom YOLOv8")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--directory", type=str, help="Directory containing images for batch processing")
    parser.add_argument("--model", type=str, help="Path to custom model (optional)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = VihangamObjectDetector(args.model)
    except FileNotFoundError as e:
        print(e)
        return
    
    # Load model
    if not detector.load_model():
        return
    
    # Process single image
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
            
        print(f"\n🚀 Starting detection on: {args.image}")
        result = detector.detect_objects(
            args.image, 
            confidence=args.confidence, 
            save_results=not args.no_save
        )
        
        if result:
            print("\n🎉 Detection completed successfully!")
            if result.get('output_path'):
                print(f"📁 Check results in: {detector.results_dir}")
        else:
            print("❌ Detection failed!")
            
    # Process directory
    elif args.directory:
        print(f"\n🚀 Starting batch detection on: {args.directory}")
        detector.batch_detect(args.directory, confidence=args.confidence)
        
    # Interactive mode
    else:
        print("\n🖼️  Interactive Detection Mode")
        print("=" * 40)
        
        # Check for sample images
        sample_dirs = ["images/val", "images/train", "."]
        sample_image = None
        
        for directory in sample_dirs:
            if os.path.exists(directory):
                for ext in ['.jpg', '.jpeg', '.png']:
                    images = list(Path(directory).glob(f"*{ext}"))
                    if images:
                        sample_image = str(images[0])
                        break
                if sample_image:
                    break
        
        if sample_image:
            print(f"📸 Found sample image: {sample_image}")
            choice = input("Process this image? (y/n): ").lower().strip()
            
            if choice == 'y' or choice == 'yes':
                result = detector.detect_objects(sample_image, confidence=args.confidence)
                if result:
                    print("\n🎉 Detection completed!")
            else:
                print("Please provide --image or --directory parameter")
        else:
            print("No sample images found. Please provide --image or --directory parameter")
            print("\nUsage examples:")
            print("  python detect_objects.py --image path/to/image.jpg")
            print("  python detect_objects.py --directory path/to/images/")
            print("  python detect_objects.py --image image.jpg --confidence 0.5")

if __name__ == "__main__":
    main()