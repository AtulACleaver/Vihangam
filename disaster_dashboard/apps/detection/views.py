from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import random
import logging

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
    Simulate image processing for demonstration
    """
    try:
        # Simulate processing
        response_data = {
            'status': 'success',
            'message': 'Image processing simulated',
            'total_detections': random.randint(0, 5),
            'processing_time': round(random.uniform(0.1, 0.3), 3),
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
    Simulate live detection for demonstration
    """
    try:
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
            'accuracy': round(random.uniform(90, 96), 1)
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
    Get basic model information
    """
    try:
        return JsonResponse({
            'status': 'success',
            'model_info': {
                'name': 'Detection System',
                'version': '1.0',
                'status': 'Simulation Mode'
            },
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
    Simulate model switching
    """
    try:
        data = json.loads(request.body)
        model_path = data.get('model_path', 'default')
        
        return JsonResponse({
            'status': 'success',
            'message': f'Switched to model: {model_path} (simulated)',
            'model_info': {
                'name': f'Model {model_path}',
                'status': 'Loaded (simulation)'
            }
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
    Simulate detection from image URL
    """
    try:
        data = json.loads(request.body)
        image_url = data.get('url')
        
        if not image_url:
            return JsonResponse({'error': 'No image URL provided'}, status=400)
        
        response_data = {
            'status': 'success',
            'source': 'url',
            'url': image_url,
            'total_detections': random.randint(0, 3),
            'processing_time': round(random.uniform(0.1, 0.5), 3),
            'message': 'URL detection simulated',
            'timestamp': '2024-01-16T18:04:45Z'
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"URL detection error: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
