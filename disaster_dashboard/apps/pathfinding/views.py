from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import random
import math
from .astar import (
    AStarPathfinder, Coordinate, Priority, Obstacle, 
    TerrainType, haversine_distance, PathResult, astar,
    astar_grid, astar_grid_8dir, grid_to_coordinates, create_test_grid
)
from .pathfinding_utils import (
    PathSmoother, PathValidator, DynamicReplanner, PathOptimizer,
    FlightConstraints, PathCache, convert_path_to_mavlink,
    estimate_battery_consumption, create_kml_export
)


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


# Initialize pathfinding components (module-level for caching)
pathfinder = AStarPathfinder()
path_cache = PathCache(max_cache_size=50)
flight_constraints = FlightConstraints(
    max_altitude=500.0,
    min_altitude=50.0,
    max_speed=25.0,
    min_speed=5.0,
    safety_margin=100.0
)
path_validator = PathValidator(flight_constraints)
path_smoother = PathSmoother(flight_constraints)

@csrf_exempt
@require_http_methods(["POST"])
def calculate_path_simple(request):
    """
    Simple path calculation using the astar function.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        
        # Extract basic parameters
        start_lat = float(data.get('start_lat', 28.6139))
        start_lng = float(data.get('start_lng', 77.2090))
        goal_lat = float(data.get('dest_lat', 28.6143))
        goal_lng = float(data.get('dest_lng', 77.2095))
        priority = data.get('priority', 'shortest')
        altitude = float(data.get('altitude', 150.0))
        max_speed = float(data.get('max_speed', 12.0))
        avoid_obstacles = data.get('avoid_obstacles', True)
        
        # Use the simple astar function
        result = astar(
            start_lat=start_lat,
            start_lng=start_lng,
            goal_lat=goal_lat,
            goal_lng=goal_lng,
            priority=priority,
            altitude=altitude,
            max_speed=max_speed,
            avoid_obstacles=avoid_obstacles
        )
        
        # Return the result
        return JsonResponse({
            'status': 'success' if result['success'] else 'error',
            'message': 'Path calculated using simple A* function',
            'path_id': f'simple_path_{random.randint(1000, 9999)}',
            **result
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Simple path calculation failed: {str(e)}',
            'error_type': 'simple_pathfinding_error'
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def calculate_path(request):
    """
    Calculate optimal flight path using grid-based A* algorithm.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        
        # Extract parameters
        start_lat = data.get('start_lat', 28.6139)
        start_lng = data.get('start_lng', 77.2090)
        dest_lat = data.get('dest_lat', 28.6143)
        dest_lng = data.get('dest_lng', 77.2095)
        algorithm = data.get('algorithm', 'astar_grid')
        use_8_dir = data.get('use_8_dir', True)  # Use 8-directional or 4-directional movement
        grid_size = data.get('grid_size', 20)   # Grid dimensions
        obstacle_density = data.get('obstacle_density', 0.2)  # Obstacle density
        custom_grid = data.get('custom_grid', None)  # Custom grid if provided
        altitude = data.get('altitude', 150)
        max_speed = data.get('max_speed', 12)
        
        # Define a grid (0 = free space, 1 = obstacle)
        if custom_grid:
            # Use provided custom grid
            grid = custom_grid
        else:
            # Define a static grid with some obstacles for demonstration
            if grid_size <= 10:
                # Small grid with predefined obstacles
                grid = [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                    [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]
            else:
                # Generate random grid for larger sizes
                grid = create_test_grid(
                    width=grid_size,
                    height=grid_size,
                    obstacle_density=obstacle_density
                )
        
        # Define start and end points in grid coordinates
        grid_height = len(grid)
        grid_width = len(grid[0]) if grid_height > 0 else 0
        
        # Convert GPS coordinates to grid coordinates (simplified mapping)
        # In a real application, you'd use proper coordinate transformation
        start_x = max(0, min(grid_height - 1, int((start_lat - 28.6) * 1000) % grid_height))
        start_y = max(0, min(grid_width - 1, int((start_lng - 77.2) * 1000) % grid_width))
        end_x = max(0, min(grid_height - 1, int((dest_lat - 28.6) * 1000) % grid_height))
        end_y = max(0, min(grid_width - 1, int((dest_lng - 77.2) * 1000) % grid_width))
        
        start = (start_x, start_y)
        end = (end_x, end_y)
        
        # Ensure start and end points are not obstacles
        if grid[start[0]][start[1]] == 1:
            grid[start[0]][start[1]] = 0
        if grid[end[0]][end[1]] == 1:
            grid[end[0]][end[1]] = 0
        
        # Use appropriate A* algorithm
        if use_8_dir:
            path = astar_grid_8dir(grid, start, end)
        else:
            path = astar_grid(grid, start, end)
        
        if path:
            # Convert grid path back to GPS coordinates
            origin_lat = start_lat
            origin_lng = start_lng
            grid_resolution = 0.001  # ~100m per grid cell
            
            # Calculate coordinates for each grid point
            gps_coordinates = []
            for x, y in path:
                lat = origin_lat + (x - start[0]) * grid_resolution
                lng = origin_lng + (y - start[1]) * grid_resolution
                gps_coordinates.append({
                    'lat': round(lat, 6),
                    'lng': round(lng, 6),
                    'alt': altitude,
                    'grid_x': x,
                    'grid_y': y
                })
            
            # Calculate path metrics
            total_distance = calculate_grid_path_distance(path, grid_resolution)
            flight_time = (total_distance * 1000) / max_speed / 60  # minutes
            battery_usage = min(100.0, total_distance * 2.5)  # ~2.5% per km
            
            response_data = {
                'status': 'success',
                'message': f'Path found using {algorithm} algorithm',
                'path_id': f'grid_path_{random.randint(1000, 9999)}',
                'algorithm': f'{algorithm}_{"8dir" if use_8_dir else "4dir"}',
                'grid_path': path,
                'gps_path': gps_coordinates,
                'path_coordinates': gps_coordinates,  # For compatibility
                'waypoints': [
                    {
                        'lat': coord['lat'],
                        'lng': coord['lng'],
                        'alt': coord['alt'],
                        'type': 'start' if i == 0 else 'destination' if i == len(gps_coordinates)-1 else 'waypoint'
                    }
                    for i, coord in enumerate(gps_coordinates)
                ],
                'grid_info': {
                    'grid_size': f'{grid_height}x{grid_width}',
                    'start_grid': start,
                    'end_grid': end,
                    'obstacles_in_grid': sum(sum(row) for row in grid),
                    'grid_resolution': grid_resolution
                },
                'distance': round(total_distance, 2),
                'flight_time': round(flight_time, 1),
                'battery_usage': round(battery_usage, 1),
                'path_length': len(path),
                'efficiency': round((total_distance / haversine_distance(
                    Coordinate(start_lat, start_lng, altitude),
                    Coordinate(dest_lat, dest_lng, altitude)
                )) * 100, 1) if total_distance > 0 else 100,
                'timestamp': '2024-01-16T18:04:00Z'
            }
        else:
            response_data = {
                'status': 'error',
                'message': 'No path found',
                'error_details': {
                    'grid_size': f'{grid_height}x{grid_width}',
                    'start_grid': start,
                    'end_grid': end,
                    'algorithm': algorithm,
                    'reason': 'Path blocked by obstacles or invalid start/end points'
                }
            }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to calculate path: {str(e)}',
            'error_type': 'grid_pathfinding_error'
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def calculate_path_grid(request):
    """
    Calculate path using a specific static grid as requested.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        
        # Extract basic parameters
        use_8_dir = data.get('use_8_dir', True)
        custom_start = data.get('start', None)
        custom_end = data.get('end', None)
        
        # Define a grid (0 = free space, 1 = obstacle) - exactly as requested
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        
        # Define start and end points - exactly as requested
        start = custom_start if custom_start else (0, 0)
        end = custom_end if custom_end else (4, 4)
        
        # Use appropriate A* algorithm
        if use_8_dir:
            path = astar_grid_8dir(grid, start, end)
        else:
            path = astar_grid(grid, start, end)
        
        if path:
            # Calculate distance
            grid_resolution = 0.001  # ~100m per grid cell
            total_distance = calculate_grid_path_distance(path, grid_resolution)
            
            response_data = {
                'status': 'success',
                'message': f'Path found using grid-based A* algorithm',
                'path_id': f'static_grid_path_{random.randint(1000, 9999)}',
                'algorithm': f'astar_grid_{"8dir" if use_8_dir else "4dir"}',
                'path': path,
                'grid': grid,
                'start': start,
                'end': end,
                'distance_km': round(total_distance, 3),
                'path_length': len(path),
                'grid_size': '5x5',
                'obstacles': [(1,1), (1,3), (3,1), (3,2)],  # Obstacle positions
                'path_found': True,
                'timestamp': '2024-01-16T18:04:00Z'
            }
        else:
            response_data = {
                'status': 'error',
                'message': 'No path found',
                'path': None,
                'grid': grid,
                'start': start,
                'end': end,
                'path_found': False,
                'reason': 'Path blocked by obstacles'
            }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Grid pathfinding failed: {str(e)}',
            'error_type': 'static_grid_error'
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
def calculate_grid_path_distance(path, grid_resolution):
    """
    Calculate the total distance of a grid-based path.
    
    Args:
        path: List of (x, y) grid coordinates
        grid_resolution: Distance per grid cell in degrees
    
    Returns:
        Total distance in kilometers
    """
    if len(path) < 2:
        return 0.0
    
    total_distance = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        
        # Calculate Euclidean distance in grid units
        grid_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Convert to kilometers (approximate)
        # 1 degree ≈ 111 km at equator
        km_distance = grid_distance * grid_resolution * 111
        total_distance += km_distance
    
    return total_distance


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


def _format_pathfinding_response(result: PathResult, calculation_method: str = 'calculated', 
                                  validation_result=None) -> dict:
    """
    Format PathResult into JSON response format.
    """
    waypoints = [
        {
            'lat': coord.lat,
            'lng': coord.lng,
            'alt': coord.altitude,
            'type': 'start' if i == 0 else 'destination' if i == len(result.path)-1 else 'waypoint',
            'terrain_type': coord.terrain_type.name.lower() if coord.terrain_type else 'flat'
        }
        for i, coord in enumerate(result.path)
    ]
    
    response_data = {
        'status': 'success',
        'message': f'Flight path {calculation_method} successfully',
        'path_id': f'path_{random.randint(1000, 9999)}',
        'algorithm': result.algorithm_used,
        'priority': result.priority_mode.value,
        'distance': round(result.total_distance, 2),
        'flight_time': round(result.total_time, 1),
        'battery_usage': round(result.battery_usage, 1),
        'safety_score': round(result.safety_score, 1),
        'obstacles_avoided': result.obstacles_avoided,
        'waypoints': waypoints,
        'path_coordinates': [{'lat': c.lat, 'lng': c.lng, 'alt': c.altitude} for c in result.path],
        'efficiency': round((result.total_distance / haversine_distance(result.path[0], result.path[-1])) * 100, 1) if len(result.path) >= 2 else 100,
        'calculation_method': calculation_method,
        'timestamp': '2024-01-16T18:04:00Z'
    }
    
    # Add validation information if available
    if validation_result:
        response_data.update({
            'validation': {
                'is_valid': validation_result.is_valid,
                'violations': validation_result.violations,
                'safety_score': validation_result.safety_score,
                'fuel_efficiency': validation_result.fuel_efficiency_score,
                'recommended_fixes': validation_result.recommended_fixes
            }
        })
    
    return response_data


@csrf_exempt
@require_http_methods(["POST"])
def validate_path(request):
    """
    Validate an existing flight path for safety and efficiency.
    """
    try:
        data = json.loads(request.body)
        path_coordinates = data.get('path', [])
        
        if not path_coordinates:
            return JsonResponse({
                'status': 'error',
                'message': 'No path coordinates provided'
            }, status=400)
        
        # Convert to Coordinate objects
        path = [
            Coordinate(
                lat=coord['lat'],
                lng=coord['lng'],
                altitude=coord.get('alt', 150.0)
            )
            for coord in path_coordinates
        ]
        
        # Validate the path
        validation_result = path_validator.validate_path(path, pathfinder.obstacles)
        
        return JsonResponse({
            'status': 'success',
            'validation': {
                'is_valid': validation_result.is_valid,
                'violations': validation_result.violations,
                'safety_score': validation_result.safety_score,
                'estimated_completion_time': validation_result.estimated_completion_time,
                'fuel_efficiency_score': validation_result.fuel_efficiency_score,
                'recommended_fixes': validation_result.recommended_fixes
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Path validation failed: {str(e)}'
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def replan_path(request):
    """
    Perform dynamic replanning from current drone position.
    """
    try:
        data = json.loads(request.body)
        
        current_lat = data.get('current_lat')
        current_lng = data.get('current_lng')
        current_alt = data.get('current_alt', 150.0)
        dest_lat = data.get('dest_lat')
        dest_lng = data.get('dest_lng')
        dest_alt = data.get('dest_alt', 150.0)
        priority_str = data.get('priority', 'fastest')
        new_obstacles = data.get('obstacles', [])
        
        if not all([current_lat, current_lng, dest_lat, dest_lng]):
            return JsonResponse({
                'status': 'error',
                'message': 'Missing required coordinates'
            }, status=400)
        
        # Create coordinates
        current_pos = Coordinate(lat=current_lat, lng=current_lng, altitude=current_alt)
        destination = Coordinate(lat=dest_lat, lng=dest_lng, altitude=dest_alt)
        
        # Convert priority
        priority_map = {
            'shortest': Priority.SHORTEST,
            'fastest': Priority.FASTEST,
            'safest': Priority.SAFEST,
            'battery_optimal': Priority.BATTERY_OPTIMAL
        }
        priority = priority_map.get(priority_str, Priority.FASTEST)
        
        # Add new obstacles if provided
        obstacles = []
        for obs_data in new_obstacles:
            obstacle = Obstacle(
                center=Coordinate(
                    lat=obs_data['lat'],
                    lng=obs_data['lng'],
                    altitude=obs_data.get('alt', 0)
                ),
                radius=obs_data.get('radius', 100),
                height=obs_data.get('height', 300),
                severity=obs_data.get('severity', 2)
            )
            obstacles.append(obstacle)
        
        # Initialize replanner
        replanner = DynamicReplanner(pathfinder, flight_constraints)
        
        # Perform replanning
        result = replanner.replan_from_current_position(
            current_pos=current_pos,
            original_destination=destination,
            new_obstacles=obstacles,
            priority=priority
        )
        
        response = _format_pathfinding_response(result, 'replanned')
        response['replan_reason'] = 'Emergency replanning requested'
        
        return JsonResponse(response)
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Replanning failed: {str(e)}'
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def optimize_path(request):
    """
    Optimize existing path for wind conditions and other factors.
    """
    try:
        data = json.loads(request.body)
        path_coordinates = data.get('path', [])
        wind_direction = data.get('wind_direction', 0)  # degrees
        wind_speed = data.get('wind_speed', 0)  # m/s
        inspection_points = data.get('inspection_points', [])
        
        if not path_coordinates:
            return JsonResponse({
                'status': 'error',
                'message': 'No path coordinates provided'
            }, status=400)
        
        # Convert to Coordinate objects
        path = [
            Coordinate(
                lat=coord['lat'],
                lng=coord['lng'],
                altitude=coord.get('alt', 150.0)
            )
            for coord in path_coordinates
        ]
        
        # Apply wind optimization
        if wind_speed > 0:
            path = PathOptimizer.optimize_for_wind(path, wind_direction, wind_speed)
        
        # Add inspection waypoints if provided
        if inspection_points:
            inspection_coords = [
                Coordinate(
                    lat=point['lat'],
                    lng=point['lng'],
                    altitude=point.get('alt', 150.0)
                )
                for point in inspection_points
            ]
            path = PathOptimizer.add_inspection_waypoints(path, inspection_coords)
        
        # Apply smoothing
        path = path_smoother.smooth_path(path, smoothing_factor=0.6)
        
        # Calculate metrics for optimized path
        total_distance = sum(haversine_distance(path[i], path[i+1]) for i in range(len(path)-1))
        battery_usage = estimate_battery_consumption(path)
        
        optimized_waypoints = [
            {
                'lat': coord.lat,
                'lng': coord.lng,
                'alt': coord.altitude,
                'type': 'start' if i == 0 else 'destination' if i == len(path)-1 else 'waypoint'
            }
            for i, coord in enumerate(path)
        ]
        
        return JsonResponse({
            'status': 'success',
            'message': 'Path optimized successfully',
            'optimized_path': optimized_waypoints,
            'optimization_applied': {
                'wind_compensation': wind_speed > 0,
                'inspection_points_added': len(inspection_points) > 0,
                'path_smoothing': True
            },
            'metrics': {
                'total_distance': round(total_distance, 2),
                'estimated_battery_usage': round(battery_usage, 1),
                'waypoint_count': len(path)
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Path optimization failed: {str(e)}'
        }, status=400)


@require_http_methods(["GET"])
def export_path_kml(request):
    """
    Export the last calculated path to KML format.
    """
    try:
        # For demo, create a sample path (in production, this would come from session/database)
        sample_path = [
            Coordinate(lat=28.6139, lng=77.2090, altitude=150.0),
            Coordinate(lat=28.6200, lng=77.2150, altitude=160.0),
            Coordinate(lat=28.6300, lng=77.2200, altitude=155.0),
            Coordinate(lat=28.6400, lng=77.2250, altitude=150.0)
        ]
        
        filename = create_kml_export(sample_path, "vihangam_flight_path.kml")
        
        return JsonResponse({
            'status': 'success',
            'message': 'KML export created successfully',
            'filename': filename,
            'download_url': f'/media/exports/{filename}'  # Adjust path as needed
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'KML export failed: {str(e)}'
        }, status=500)


def generate_waypoints(start_lat, start_lng, dest_lat, dest_lng):
    """
    Generate intermediate waypoints between start and destination (legacy function).
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
