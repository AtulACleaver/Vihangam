from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json


def index(request):
    """
    Main dashboard view for Vihangam drone disaster management system.
    Displays real-time drone status, mission control interface, and system overview.
    """
    context = {
        'page_title': 'Mission Control Dashboard',
        'active_drones': 3,
        'areas_monitored': 12,
        'detection_alerts': 7,
        'mission_time': '2h 45m',
        'system_status': 'operational'
    }
    return render(request, 'dashboard/index.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def emergency_alert(request):
    """
    Handle emergency alert activation from the dashboard.
    """
    try:
        data = json.loads(request.body)
        alert_type = data.get('type', 'general')
        message = data.get('message', 'Emergency protocol activated')
        
        # In a real implementation, this would trigger actual emergency protocols
        # For now, we'll just return a success response
        
        response_data = {
            'status': 'success',
            'message': f'Emergency alert activated: {message}',
            'alert_id': 'EMRG-001',
            'timestamp': '2024-01-16T18:04:00Z'
        }
        
        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to activate emergency alert: {str(e)}'
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def refresh_data(request):
    """
    Refresh dashboard data - simulates fetching latest drone telemetry.
    """
    # Simulate updated drone data
    updated_data = {
        'active_drones': 3,
        'drone_positions': [
            {'id': 'alpha', 'lat': 28.6139, 'lng': 77.2090, 'alt': 150, 'battery': 87},
            {'id': 'beta', 'lat': 28.6141, 'lng': 77.2092, 'alt': 145, 'battery': 92},
            {'id': 'gamma', 'lat': 28.6138, 'lng': 77.2088, 'alt': 155, 'battery': 78}
        ],
        'recent_activities': [
            {'time': '18:04:45', 'drone': 'Alpha', 'event': 'Position Update', 'status': 'info'},
            {'time': '18:04:30', 'drone': 'Beta', 'event': 'Battery Check', 'status': 'success'},
            {'time': '18:04:15', 'drone': 'Gamma', 'event': 'Route Adjustment', 'status': 'warning'}
        ],
        'system_health': {
            'overall': 'good',
            'communication': 'excellent',
            'gps_signal': 'strong',
            'weather_conditions': 'favorable'
        },
        'timestamp': '2024-01-16T18:04:45Z'
    }
    
    return JsonResponse({
        'status': 'success',
        'data': updated_data,
        'last_update': updated_data['timestamp']
    })
