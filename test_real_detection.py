#!/usr/bin/env python3
"""
Real Detection Test - Create and test with a more recognizable image
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
from datetime import datetime

def create_test_image_with_objects():
    """Create a test image with recognizable human-like and debris-like objects"""
    
    # Create a disaster scene background
    img = np.ones((480, 640, 3), dtype=np.uint8) * 50  # Dark background
    
    # Add some texture to simulate disaster scene
    for i in range(100):
        x, y = np.random.randint(0, 640), np.random.randint(0, 480)
        cv2.circle(img, (x, y), np.random.randint(2, 8), (80, 70, 60), -1)
    
    # Create human-like figure (stick figure style)
    # Head
    cv2.circle(img, (200, 100), 25, (220, 180, 120), -1)  # Skin tone
    # Body 
    cv2.rectangle(img, (185, 125), (215, 220), (100, 50, 200), -1)  # Body
    # Arms
    cv2.rectangle(img, (160, 140), (185, 155), (220, 180, 120), -1)  # Left arm
    cv2.rectangle(img, (215, 140), (240, 155), (220, 180, 120), -1)  # Right arm
    # Legs
    cv2.rectangle(img, (185, 220), (200, 280), (50, 50, 150), -1)  # Left leg
    cv2.rectangle(img, (200, 220), (215, 280), (50, 50, 150), -1)  # Right leg
    
    # Create another human figure
    cv2.circle(img, (450, 120), 20, (200, 160, 100), -1)  # Head
    cv2.rectangle(img, (430, 140), (470, 230), (120, 80, 60), -1)  # Body
    cv2.rectangle(img, (470, 150), (490, 165), (200, 160, 100), -1)  # Arm
    cv2.rectangle(img, (410, 150), (430, 165), (200, 160, 100), -1)  # Arm
    cv2.rectangle(img, (430, 230), (445, 290), (60, 60, 120), -1)  # Leg
    cv2.rectangle(img, (455, 230), (470, 290), (60, 60, 120), -1)  # Leg
    
    # Add debris objects (rectangular and irregular shapes)
    # Large debris piece
    debris_points = np.array([[100, 350], [180, 320], [200, 380], [120, 400]], np.int32)
    cv2.fillPoly(img, [debris_points], (60, 60, 60))  # Gray debris
    
    # Smaller debris
    cv2.rectangle(img, (350, 300), (420, 340), (80, 60, 40), -1)  # Brown debris
    cv2.circle(img, (500, 350), 30, (70, 70, 70), -1)  # Round debris
    
    # Add some scattered small debris
    for i in range(15):
        x = np.random.randint(50, 590)
        y = np.random.randint(300, 450)
        size = np.random.randint(5, 15)
        cv2.circle(img, (x, y), size, (np.random.randint(40, 100), 
                                      np.random.randint(40, 80), 
                                      np.random.randint(30, 70)), -1)
    
    # Add some emergency/disaster context
    # Smoke/dust effect
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (640, 150), (120, 120, 120), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    
    # Add text indicating this is a disaster scene
    cv2.putText(img, "DISASTER SIMULATION", (200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def test_detection_with_custom_image():
    """Create and test detection on a custom disaster simulation image"""
    
    print("🏗️  Creating disaster simulation test image...")
    
    # Create test image
    test_img = create_test_image_with_objects()
    
    # Save test image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_image_path = f"disaster_test_{timestamp}.jpg"
    cv2.imwrite(test_image_path, test_img)
    
    print(f"✅ Test image created: {test_image_path}")
    print(f"📐 Image contains:")
    print(f"   👤 2 human figures")
    print(f"   🗂️  Multiple debris objects")
    print(f"   🌫️  Disaster scene simulation")
    
    # Test with the custom YOLOv8 model
    print(f"\n🔍 Running detection on test image...")
    
    # Import detection system
    from detect_objects import VihangamObjectDetector
    
    # Initialize detector
    detector = VihangamObjectDetector()
    
    # Load model
    if not detector.load_model():
        print("❌ Failed to load model")
        return
    
    # Run detection with multiple confidence levels
    confidence_levels = [0.5, 0.3, 0.1, 0.05]
    
    for conf in confidence_levels:
        print(f"\n🎚️  Testing with confidence threshold: {conf}")
        
        try:
            results = detector.detect_objects(test_image_path, confidence=conf)
            
            if results and 'error' not in results:
                summary = results['summary']
                if summary['total_objects'] > 0:
                    print(f"🎯 SUCCESS! Objects detected:")
                    print(f"   👤 Humans: {summary['human_count']}")
                    print(f"   🗂️  Debris: {summary['debris_count']}")
                    print(f"   📊 Total: {summary['total_objects']}")
                    print(f"   ⚡ Processing time: {results['inference_time_seconds']}s")
                    print(f"   📈 Avg confidence: {summary['average_confidence']:.3f}")
                    
                    # Show detailed detections
                    for i, detection in enumerate(results['detections'][:3]):
                        print(f"     {i+1}. {detection['class_name']}: {detection['confidence']:.3f}")
                    
                    break
                else:
                    print(f"   ⚪ No objects detected at this confidence level")
            else:
                print(f"   ❌ Detection error: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Detection failed: {e}")
    
    print(f"\n💾 Test image saved as: {test_image_path}")
    print(f"🔍 You can also manually run: python detect_objects.py --image {test_image_path} --confidence 0.1")

if __name__ == "__main__":
    test_detection_with_custom_image()