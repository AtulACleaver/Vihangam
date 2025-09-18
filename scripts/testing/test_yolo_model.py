#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import time

class YOLOTester:
    def __init__(self):
        self.model = None
        self.model_path = None
        
        self.disaster_classes = {
            0: 'human',
            1: 'debris'
        }
        
        self.colors = {
            'human': (0, 0, 255),
            'debris': (0, 165, 255)
        }
    
    def find_custom_model(self):
        possible_paths = [
            "runs/detect/disaster_demo_20250918_201832/weights/best.pt",
            "../runs/detect/disaster_demo_20250918_201832/weights/best.pt",
            "models/best.pt",
            "weights/best.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def load_model(self, model_path=None):
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = self.find_custom_model()
        
        if not self.model_path:
            print("Model not found. Please specify path with --model")
            print("Available options:")
            print("- yolov8n.pt (will be downloaded)")
            print("- path/to/your/custom/model.pt")
            return False
        
        try:
            print(f"Loading model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def test_image(self, image_path, confidence=0.25, save_result=False):
        if not self.model:
            print("Model not loaded")
            return False
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return False
        
        print(f"\nProcessing: {image_path}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            results = self.model(image_path, conf=confidence, verbose=False)
            processing_time = time.time() - start_time
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.disaster_classes.get(class_id, f'class_{class_id}')
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'class_id': class_id,
                            'class_name': class_name,
                            'area': int((x2 - x1) * (y2 - y1))
                        }
                        detections.append(detection)
            
            print(f"Processing time: {processing_time:.3f}s")
            print(f"Detections found: {len(detections)}")
            
            if detections:
                print("\nDetection Details:")
                for i, det in enumerate(detections, 1):
                    priority = "CRITICAL" if det['class_name'] == 'human' else "WARNING"
                    print(f"  {i}. {det['class_name'].upper()} - {det['confidence']:.3f} ({det['confidence']*100:.1f}%)")
                    print(f"     {priority}")
                    print(f"     Area: {det['area']} pixels")
                    print(f"     Box: [{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]")
                    print()
                
                if save_result:
                    output_path = self.save_annotated_image(image_path, detections)
                    if output_path:
                        print(f"Annotated image saved: {output_path}")
            else:
                print("No objects detected (try lowering confidence threshold)")
            
            human_count = sum(1 for d in detections if d['class_name'] == 'human')
            debris_count = sum(1 for d in detections if d['class_name'] == 'debris')
            
            print(f"\nSummary:")
            print(f"   Humans: {human_count} {'(CRITICAL)' if human_count > 0 else ''}")
            print(f"   Debris: {debris_count}")
            print(f"   Speed: {1/processing_time:.1f} FPS")
            
            return True
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return False
    
    def save_annotated_image(self, image_path, detections):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                color = self.colors.get(class_name, (128, 128, 128))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {confidence:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                cv2.rectangle(image, (x1, y1 - label_height - 8), (x1 + label_width, y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_detected.jpg"
            cv2.imwrite(output_path, image)
            
            return output_path
            
        except Exception as e:
            print(f"Error saving annotated image: {e}")
            return None
    
    def test_directory(self, directory_path, confidence=0.25, save_results=False):
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file in os.listdir(directory_path):
            if os.path.splitext(file.lower())[1] in image_extensions:
                image_files.append(os.path.join(directory_path, file))
        
        if not image_files:
            print(f"No image files found in: {directory_path}")
            return
        
        print(f"Testing {len(image_files)} images in: {directory_path}")
        print("=" * 60)
        
        successful = 0
        
        for image_file in image_files:
            success = self.test_image(image_file, confidence, save_results)
            if success:
                successful += 1
        
        print(f"\nProcessed {successful}/{len(image_files)} images successfully")

def main():
    parser = argparse.ArgumentParser(description='YOLO Model Tester')
    parser.add_argument('input', help='Image file or directory to test')
    parser.add_argument('-m', '--model', help='Path to YOLO model (default: auto-detect custom model)')
    parser.add_argument('-c', '--confidence', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('-s', '--save', action='store_true', help='Save annotated images')
    
    args = parser.parse_args()
    
    print("YOLO Model Tester")
    print("=" * 50)
    
    tester = YOLOTester()
    
    if not tester.load_model(args.model):
        return 1
    
    if os.path.isfile(args.input):
        tester.test_image(args.input, args.confidence, args.save)
    elif os.path.isdir(args.input):
        tester.test_directory(args.input, args.confidence, args.save)
    else:
        print(f"Input not found: {args.input}")
        return 1
    
    print("\nTesting complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())