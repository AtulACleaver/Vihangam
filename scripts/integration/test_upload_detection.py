#!/usr/bin/env python3
"""
Test script to verify image upload and detection functionality
"""

import os
import sys
import django
from pathlib import Path

# Add the Django project to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'disaster_dashboard'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'disaster_dashboard.settings')

# Setup Django
django.setup()

from apps.detection.yolo_handler import get_yolo_detector
from PIL import Image
import requests
import io

def test_yolo_handler():
    """Test the YOLO handler directly"""
    print("🧪 Testing YOLO Handler")
    print("-" * 30)
    
    try:
        # Initialize detector
        detector = get_yolo_detector()
        print(f"✅ Detector initialized: {type(detector).__name__}")
        
        # Get model info
        model_info = detector.get_model_info()
        print(f"📊 Model Info: {model_info}")
        
        # Test with a sample image (create a simple test image)
        # Create a simple test image if none exists
        test_image_path = "test_image.jpg"
        if not os.path.exists(test_image_path):
            # Create a simple colored rectangle as test image
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (640, 480), color='blue')
            draw = ImageDraw.Draw(img)
            draw.rectangle([100, 100, 300, 300], fill='red')
            draw.rectangle([400, 200, 500, 350], fill='green')
            img.save(test_image_path)
            print(f"📷 Created test image: {test_image_path}")
        
        # Test detection
        print("🔍 Running detection on test image...")
        results = detector.detect_objects(test_image_path, confidence=0.1)
        
        print(f"🎯 Detection Results:")
        print(f"   Total detections: {results.get('count', 0)}")
        print(f"   Processing time: {results.get('processing_time', 'N/A')}s")
        print(f"   Model path: {results.get('model_info', {}).get('model_path', 'N/A')}")
        
        if results.get('detections'):
            print("📋 Detections:")
            for i, det in enumerate(results['detections'][:5]):  # Show first 5
                print(f"   {i+1}. {det.get('class_name', 'unknown')}: {det.get('confidence', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing YOLO handler: {e}")
        return False

def test_django_views():
    """Test the Django detection views"""
    print("\n🌐 Testing Django Views")
    print("-" * 30)
    
    try:
        from apps.detection.views import model_info
        from django.test import RequestFactory
        
        factory = RequestFactory()
        request = factory.get('/detection/api/model-info/')
        
        response = model_info(request)
        print(f"✅ Model info endpoint: Status {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Django views: {e}")
        return False

def main():
    print("🚁 Vihangam Detection System Test")
    print("=" * 40)
    
    # Test YOLO handler
    yolo_success = test_yolo_handler()
    
    # Test Django views 
    django_success = test_django_views()
    
    print(f"\n📊 Test Results:")
    print(f"   YOLO Handler: {'✅ PASS' if yolo_success else '❌ FAIL'}")
    print(f"   Django Views: {'✅ PASS' if django_success else '❌ FAIL'}")
    
    if yolo_success and django_success:
        print("\n🎉 All tests passed! The system should work correctly.")
        print("\nTo test the full system:")
        print("1. Run: python ../start_server.py")
        print("2. Open: http://localhost:8000/detection/")
        print("3. Switch to 'Upload' mode")
        print("4. Upload an image and click 'Process'")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return 0 if (yolo_success and django_success) else 1

if __name__ == "__main__":
    sys.exit(main())