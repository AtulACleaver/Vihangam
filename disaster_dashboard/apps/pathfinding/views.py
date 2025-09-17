from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import random
import math


def index(request):
    """
    Pathfinding navigation interface view for Vihangam autonomous flight system.
    Displays flight path visualization, navigation controls, and route analytics.
    """
    context = {
        'page_title': 'Autonomous Navigation System',
        'current_distance': 2.4,  # km
        'estimated_time': 12,     # minutes
        'obstacles_detected': 3,
        'path_efficiency': 92,    # percentage
        'algorithm': 'A* (A-Star)',
        'cruise_altitude': 150,   # meters
        'max_speed': 12          # m/s
    }
    return render(request, 'pathfinding/index.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def calculate_path(request):
    """
    Calculate optimal flight path using selected algorithm.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        
        # Extract parameters
        start_lat = data.get('start_lat', 28.6139)
        start_lng = data.get('start_lng', 77.2090)
        dest_lat = data.get('dest_lat', 28.6143)
        dest_lng = data.get('dest_lng', 77.2095)
        algorithm = data.get('algorithm', 'astar')
        priority = data.get('priority', 'shortest')
        altitude = data.get('altitude', 150)
        max_speed = data.get('max_speed', 12)
        
        # Simulate path calculation
        distance = calculate_distance(start_lat, start_lng, dest_lat, dest_lng)
        flight_time = distance / (max_speed / 1000 * 60)  # minutes
        battery_usage = distance * 0.1  # percentage per km
        
        # Generate waypoints
        waypoints = generate_waypoints(start_lat, start_lng, dest_lat, dest_lng)
        
        response_data = {
            'status': 'success',
            'message': 'Flight path calculated successfully',
            'path_id': f'path_{random.randint(1000, 9999)}',
            'algorithm': algorithm,
            'distance': round(distance, 2),
            'flight_time': round(flight_time, 1),
            'battery_usage': round(battery_usage, 1),
            'altitude': altitude,
            'waypoints': waypoints,
            'obstacles': random.randint(2, 5),
            'efficiency': round(random.uniform(88, 96), 1),
            'timestamp': '2024-01-16T18:04:00Z'
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to calculate path: {str(e)}'
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def start_navigation(request):
    """
    Start autonomous navigation along calculated path.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        path_id = data.get('path_id', 'path_default')
        
        response_data = {
            'status': 'success',
            'message': 'Navigation started successfully',
            'navigation_id': f'nav_{random.randint(1000, 9999)}',
            'path_id': path_id,
            'start_time': '2024-01-16T18:04:00Z',
            'current_waypoint': 1,
            'total_waypoints': 3,
            'estimated_arrival': '2024-01-16T18:16:00Z',
            'status': 'navigating'
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to start navigation: {str(e)}'
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def emergency_return(request):
    """
    Initiate emergency return to base protocol.
    """
    try:
        response_data = {
            'status': 'success',
            'message': 'Emergency return protocol activated',
            'return_id': f'emrg_return_{random.randint(1000, 9999)}',
            'return_path_calculated': True,
            'estimated_return_time': '8 minutes',
            'priority': 'high',
            'all_systems_override': True,
            'timestamp': '2024-01-16T18:04:00Z'
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to initiate emergency return: {str(e)}'
        }, status=400)


@require_http_methods(["GET"])
def get_navigation_status(request):
    """
    Get current navigation status and telemetry.
    """
    # Simulate real-time navigation data
    status = {
        'current_position': {
            'lat': 28.6139 + random.uniform(-0.001, 0.001),
            'lng': 77.2090 + random.uniform(-0.001, 0.001),
            'alt': 150 + random.uniform(-5, 5)
        },
        'navigation_status': 'navigating',
        'current_waypoint': 2,
        'total_waypoints': 3,
        'distance_remaining': round(random.uniform(1.5, 2.0), 2),
        'time_remaining': round(random.uniform(8, 12), 1),
        'current_speed': round(random.uniform(10, 14), 1),
        'heading': random.randint(0, 359),
        'battery_remaining': round(random.uniform(75, 85), 1),
        'obstacles_ahead': random.randint(0, 3),
        'weather_conditions': 'favorable',
        'gps_signal_strength': 'strong',
        'communication_status': 'excellent',
        'last_update': '2024-01-16T18:04:45Z'
    }
    
    return JsonResponse({
        'status': 'success',
        'data': status
    })


@csrf_exempt
@require_http_methods(["POST"])
def add_waypoint(request):
    """
    Add a new waypoint to the current flight path.
    """
    try:
        data = json.loads(request.body)
        
        lat = data.get('lat')
        lng = data.get('lng')
        alt = data.get('alt', 150)
        
        if not lat or not lng:
            return JsonResponse({
                'status': 'error',
                'message': 'Latitude and longitude are required'
            }, status=400)
        
        waypoint = {
            'id': f'wp_{random.randint(100, 999)}',
            'lat': float(lat),
            'lng': float(lng),
            'alt': float(alt),
            'order': random.randint(1, 10),
            'added_at': '2024-01-16T18:04:00Z'
        }
        
        response_data = {
            'status': 'success',
            'message': 'Waypoint added successfully',
            'waypoint': waypoint,
            'path_recalculation_required': True
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to add waypoint: {str(e)}'
        }, status=400)


# Utility functions
def calculate_distance(lat1, lng1, lat2, lng2):
    """
    Calculate distance between two GPS coordinates using Haversine formula.
    Returns distance in kilometers.
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def generate_waypoints(start_lat, start_lng, dest_lat, dest_lng):
    """
    Generate intermediate waypoints between start and destination.
    """
    waypoints = [
        {'lat': start_lat, 'lng': start_lng, 'type': 'start'},
        {'lat': start_lat + (dest_lat - start_lat) * 0.33, 
         'lng': start_lng + (dest_lng - start_lng) * 0.33, 'type': 'intermediate'},
        {'lat': start_lat + (dest_lat - start_lat) * 0.67, 
         'lng': start_lng + (dest_lng - start_lng) * 0.67, 'type': 'intermediate'},
        {'lat': dest_lat, 'lng': dest_lng, 'type': 'destination'}
    ]
    
    return waypoints
