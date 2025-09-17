from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import json
import random
import base64
import io
from PIL import Image
import cv2
import numpy as np
from .yolo_handler import get_yolo_detector
import logging
import os
from django.conf import settings

logger = logging.getLogger(__name__)


def index(request):
    """
    Object detection interface view for Vihangam AI vision system.
    Displays live detection feed, configuration controls, and analytics.
    """
    context = {
        'page_title': 'AI Object Detection System',
        'objects_detected': 247,
        'high_priority_alerts': 12,
        'detection_accuracy': 94.2,
        'processing_speed': 28,
        'active_model': 'YOLOv8 - General Objects',
        'confidence_threshold': 75
    }
    return render(request, 'detection/index.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def start_detection(request):
    """
    Start the object detection process.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        model_type = data.get('model', 'yolov8_general')
        confidence_threshold = data.get('confidence_threshold', 0.75)
        classes = data.get('classes', ['person', 'vehicle', 'debris'])
        
        # Simulate detection start
        response_data = {
            'status': 'success',
            'message': 'Object detection started successfully',
            'session_id': f'detection_{random.randint(1000, 9999)}',
            'model': model_type,
            'confidence_threshold': confidence_threshold,
            'active_classes': classes,
            'processing_fps': random.randint(25, 30),
            'timestamp': '2024-01-16T18:04:00Z'
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to start detection: {str(e)}'
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def stop_detection(request):
    """
    Stop the object detection process.
    """
    try:
        response_data = {
            'status': 'success',
            'message': 'Object detection stopped',
            'total_objects_detected': random.randint(200, 300),
            'session_duration': '00:15:32',
            'timestamp': '2024-01-16T18:04:30Z'
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to stop detection: {str(e)}'
        }, status=400)


@require_http_methods(["GET"])
def get_detection_results(request):
    """
    Get latest detection results and statistics.
    """
    # Simulate real-time detection results
    results = {
        'current_detections': [
            {
                'class': 'person',
                'confidence': round(random.uniform(0.75, 0.95), 2),
                'bbox': [random.randint(100, 400), random.randint(50, 300), 
                        random.randint(50, 100), random.randint(80, 120)],
                'timestamp': '2024-01-16T18:04:45Z'
            },
            {
                'class': 'vehicle',
                'confidence': round(random.uniform(0.80, 0.98), 2),
                'bbox': [random.randint(200, 500), random.randint(100, 350), 
                        random.randint(80, 150), random.randint(40, 80)],
                'timestamp': '2024-01-16T18:04:44Z'
            }
        ],
        'statistics': {
            'total_detections': random.randint(245, 250),
            'person_count': random.randint(85, 90),
            'vehicle_count': random.randint(40, 45),
            'debris_count': random.randint(20, 25),
            'building_count': random.randint(10, 15),
            'average_confidence': round(random.uniform(0.90, 0.96), 2),
            'processing_fps': random.randint(26, 30)
        },
        'alerts': [
            {
                'type': 'high_priority',
                'message': 'Person detected in restricted zone',
                'confidence': 0.94,
                'timestamp': '2024-01-16T18:04:12Z'
            }
        ],
        'system_status': 'active',
        'last_update': '2024-01-16T18:04:45Z'
    }
    
    return JsonResponse({
        'status': 'success',
        'data': results
    })


@csrf_exempt
@require_http_methods(["POST"])
def update_detection_settings(request):
    """
    Update detection configuration settings.
    """
    try:
        data = json.loads(request.body)
        
        settings = {
            'model': data.get('model', 'yolov8_general'),
            'confidence_threshold': data.get('confidence_threshold', 0.75),
            'classes': data.get('classes', []),
            'auto_alerts': data.get('auto_alerts', True),
            'priority_only': data.get('priority_only', False)
        }
        
        response_data = {
            'status': 'success',
            'message': 'Detection settings updated successfully',
            'settings': settings,
            'timestamp': '2024-01-16T18:04:00Z'
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to update settings: {str(e)}'
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def process_image(request):
    """
    Process uploaded image with YOLO detection
    """
    try:
        # Get image data from request
        if 'image' in request.FILES:
            # Handle file upload
            image_file = request.FILES['image']
            image = Image.open(image_file)
        elif request.body:
            # Handle base64 encoded image
            data = json.loads(request.body)
            image_data = data.get('image', '')
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        # Get detection parameters
        confidence_threshold = float(request.POST.get('confidence', 0.5))
        if request.body:
            data = json.loads(request.body)
            confidence_threshold = data.get('confidence', confidence_threshold)
        
        # Perform detection
        detector = get_yolo_detector()
        results = detector.detect_objects(image, confidence_threshold)
        
        # Process results for response
        response_data = {
            'status': 'success',
            'total_detections': results['count'],
            'disaster_objects': results['disaster_related_count'],
            'high_priority_objects': results['high_priority_count'],
            'processing_time': results['processing_time'],
            'average_confidence': results['average_confidence'],
            'detections': results['detections'][:10],  # Limit to top 10
            'timestamp': '2024-01-16T18:04:45Z'
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return JsonResponse({
            'status': 'error', 
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def live_detection(request):
    """
    Handle live video stream detection (simulated for now)
    """
    try:
        data = json.loads(request.body) if request.body else {}
        confidence_threshold = data.get('confidence', 0.5)
        
        # In a real implementation, this would connect to your drone's video stream
        # For now, we'll use the existing YOLO detector with simulation
        detector = get_yolo_detector()
        
        # Simulated detection results with real YOLO model info
        model_info = detector.get_model_info()
        
        mock_results = {
            'status': 'success',
            'session_id': f'live_{random.randint(1000, 9999)}',
            'timestamp': '2024-01-16T18:04:45Z',
            'frame_id': random.randint(10000, 99999),
            'detections': [
                {
                    'bbox': [random.randint(50, 200), random.randint(50, 200), 
                            random.randint(100, 150), random.randint(100, 150)],
                    'confidence': round(random.uniform(0.7, 0.95), 3),
                    'class_name': random.choice(['person', 'car', 'truck', 'motorbike']),
                    'is_disaster_related': True,
                    'is_high_priority': random.choice([True, False])
                }
            ],
            'total_count': random.randint(1, 5),
            'disaster_count': random.randint(0, 3),
            'high_priority_count': random.randint(0, 2),
            'fps': random.randint(25, 30),
            'accuracy': round(random.uniform(90, 96), 1),
            'model_info': model_info
        }
        
        return JsonResponse(mock_results)
        
    except Exception as e:
        logger.error(f"Live detection error: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["GET"])
def model_info(request):
    """
    Get information about the current YOLO model
    """
    try:
        detector = get_yolo_detector()
        info = detector.get_model_info()
        
        return JsonResponse({
            'status': 'success',
            'model_info': info,
            'available_models': [
                'yolov8n.pt',
                'yolov8s.pt',
                'yolov8m.pt',
                'yolov8l.pt',
                'yolov8x.pt'
            ],
            'supported_formats': ['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            'max_image_size': '4096x4096'
        })
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def switch_model(request):
    """
    Switch to a different YOLO model
    """
    try:
        data = json.loads(request.body)
        model_path = data.get('model_path', 'yolov8n.pt')
        
        # Reset detector to load new model
        from .yolo_handler import reset_yolo_detector
        reset_yolo_detector()
        
        # Get new detector with specified model
        detector = get_yolo_detector(model_path)
        model_info = detector.get_model_info()
        
        return JsonResponse({
            'status': 'success',
            'message': f'Switched to model: {model_path}',
            'model_info': model_info
        })
        
    except Exception as e:
        logger.error(f"Model switch error: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def detect_from_url(request):
    """
    Detect objects from an image URL
    """
    try:
        data = json.loads(request.body)
        image_url = data.get('url')
        confidence_threshold = data.get('confidence', 0.5)
        
        if not image_url:
            return JsonResponse({'error': 'No image URL provided'}, status=400)
        
        # Download and process image
        import requests
        response = requests.get(image_url, timeout=10)
        image = Image.open(io.BytesIO(response.content))
        
        # Perform detection
        detector = get_yolo_detector()
        results = detector.detect_objects(image, confidence_threshold)
        
        response_data = {
            'status': 'success',
            'source': 'url',
            'url': image_url,
            'total_detections': results['count'],
            'disaster_objects': results['disaster_related_count'],
            'high_priority_objects': results['high_priority_count'],
            'processing_time': results['processing_time'],
            'average_confidence': results['average_confidence'],
            'detections': results['detections'],
            'timestamp': '2024-01-16T18:04:45Z'
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"URL detection error: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
