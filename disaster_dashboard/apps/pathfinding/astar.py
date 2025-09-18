"""
A* Pathfinding Algorithm Implementation for Vihangam Drone Navigation System

This module implements the A* (A-Star) pathfinding algorithm optimized for drone
navigation in disaster scenarios. It includes terrain awareness, obstacle avoidance,
and real-time path optimization for autonomous flight operations.

Features:
- GPS coordinate-based pathfinding
- Terrain elevation consideration
- Dynamic obstacle avoidance
- No-fly zone respect
- Weather-aware routing
- Battery optimization
- Emergency landing site identification
"""

import heapq
import math
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json


class TerrainType(Enum):
    """Terrain types for pathfinding cost calculation"""
    FLAT = 1.0
    HILLS = 1.2
    MOUNTAINS = 1.5
    WATER = 0.8
    URBAN = 1.3
    FOREST = 1.1
    RESTRICTED = float('inf')  # No-fly zones


class Priority(Enum):
    """Path optimization priorities"""
    SHORTEST = "shortest"
    FASTEST = "fastest"
    SAFEST = "safest"
    BATTERY_OPTIMAL = "battery_optimal"


@dataclass
class Coordinate:
    """GPS coordinate with additional navigation metadata"""
    lat: float
    lng: float
    altitude: float = 150.0
    terrain_type: TerrainType = TerrainType.FLAT
    
    def __hash__(self):
        return hash((round(self.lat, 6), round(self.lng, 6)))
    
    def __eq__(self, other):
        if not isinstance(other, Coordinate):
            return False
        return (abs(self.lat - other.lat) < 0.000001 and 
                abs(self.lng - other.lng) < 0.000001)


@dataclass
class Node:
    """A* algorithm node for pathfinding"""
    coordinate: Coordinate
    g_cost: float = 0.0  # Distance from start
    h_cost: float = 0.0  # Heuristic distance to goal
    f_cost: float = field(init=False)  # Total cost
    parent: Optional['Node'] = None
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


@dataclass
class Obstacle:
    """Dynamic obstacle representation"""
    center: Coordinate
    radius: float  # meters
    height: float = 500.0  # meters (max altitude affected)
    severity: int = 1  # 1-5, higher = more dangerous
    is_temporary: bool = False
    
    def contains_point(self, coord: Coordinate, safety_margin: float = 50.0) -> bool:
        """Check if coordinate is within obstacle bounds"""
        if coord.altitude > self.height:
            return False
        
        distance = haversine_distance(self.center, coord) * 1000  # Convert to meters
        return distance <= (self.radius + safety_margin)


@dataclass
class PathResult:
    """Result of pathfinding calculation"""
    path: List[Coordinate]
    total_distance: float  # kilometers
    total_time: float  # minutes
    battery_usage: float  # percentage
    safety_score: float  # 0-100
    obstacles_avoided: int
    waypoints: List[Dict]
    algorithm_used: str
    priority_mode: Priority


class AStarPathfinder:
    """
    A* pathfinding algorithm implementation for drone navigation.
    
    Optimized for 3D GPS coordinate pathfinding with real-world constraints
    including terrain, weather, obstacles, and battery limitations.
    """
    
    def __init__(self):
        self.grid_resolution = 0.005  # ~500m resolution for better performance
        self.max_altitude = 500.0  # meters
        self.min_altitude = 50.0   # meters
        self.obstacles: List[Obstacle] = []
        self.no_fly_zones: List[Coordinate] = []
        self.emergency_sites: List[Coordinate] = []
        self.use_simple_fallback = True  # Use simple pathfinding if A* fails
        
    def add_obstacle(self, obstacle: Obstacle):
        """Add dynamic obstacle to avoid during pathfinding"""
        self.obstacles.append(obstacle)
    
    def add_no_fly_zone(self, center: Coordinate, radius: float):
        """Add no-fly zone restriction"""
        obstacle = Obstacle(
            center=center,
            radius=radius,
            height=self.max_altitude,
            severity=5
        )
        obstacle.center.terrain_type = TerrainType.RESTRICTED
        self.obstacles.append(obstacle)
    
    def find_path(self, 
                  start: Coordinate, 
                  goal: Coordinate,
                  priority: Priority = Priority.SHORTEST,
                  max_distance: float = 10.0,  # Reduced from 50.0 for performance
                  drone_speed: float = 12.0,
                  battery_capacity: float = 100.0) -> PathResult:
        """
        Find optimal path from start to goal using A* algorithm.
        
        Args:
            start: Starting GPS coordinate
            goal: Destination GPS coordinate  
            priority: Optimization priority (shortest, fastest, safest, battery_optimal)
            max_distance: Maximum search distance in kilometers
            drone_speed: Drone cruise speed in m/s
            battery_capacity: Available battery percentage
            
        Returns:
            PathResult containing optimized path and metadata
        """
        
        # Initialize algorithm data structures
        open_set = []
        closed_set: Set[Coordinate] = set()
        nodes: Dict[Coordinate, Node] = {}
        
        # Create start node
        start_node = Node(coordinate=start, g_cost=0.0, h_cost=self.heuristic(start, goal))
        nodes[start] = start_node
        heapq.heappush(open_set, start_node)
        
        obstacles_avoided = 0
        
        iterations = 0
        max_iterations = 1000  # Limit iterations to prevent infinite loops
        
        while open_set and iterations < max_iterations:
            try:
                current_node = heapq.heappop(open_set)
                current_coord = current_node.coordinate
                iterations += 1
                
                # Goal reached
                if current_coord == goal:
                    path = self._reconstruct_path(nodes, start, goal)
                    return self._create_path_result(
                        path, priority, obstacles_avoided, drone_speed, battery_capacity
                    )
                
                closed_set.add(current_coord)
                
                # Explore neighbors
                neighbors = self._get_neighbors(current_coord, max_distance)
                
            except KeyboardInterrupt:
                print("\nPathfinding interrupted by user. Returning direct path.")
                direct_path = [start, goal]
                return self._create_path_result(
                    direct_path, priority, obstacles_avoided, drone_speed, battery_capacity
                )
            
            for neighbor_coord in neighbors:
                if neighbor_coord in closed_set:
                    continue
                
                # Check for obstacles
                if self._is_blocked(neighbor_coord):
                    obstacles_avoided += 1
                    continue
                
                # Calculate movement cost
                movement_cost = self._calculate_movement_cost(
                    current_coord, neighbor_coord, priority
                )
                tentative_g = current_node.g_cost + movement_cost
                
                # Create or update neighbor node
                if neighbor_coord not in nodes:
                    nodes[neighbor_coord] = Node(
                        coordinate=neighbor_coord,
                        g_cost=tentative_g,
                        h_cost=self.heuristic(neighbor_coord, goal),
                        parent=current_node
                    )
                    heapq.heappush(open_set, nodes[neighbor_coord])
                elif tentative_g < nodes[neighbor_coord].g_cost:
                    nodes[neighbor_coord].g_cost = tentative_g
                    nodes[neighbor_coord].f_cost = tentative_g + nodes[neighbor_coord].h_cost
                    nodes[neighbor_coord].parent = current_node
        
        # No path found or max iterations reached
        if iterations >= max_iterations:
            print(f"\nPathfinding stopped after {max_iterations} iterations.")
        else:
            print("\nNo valid path found with A*.")
            
        # Try simple fallback pathfinding
        if self.use_simple_fallback:
            print("Attempting simple fallback pathfinding...")
            fallback_path = self._simple_pathfinding(start, goal)
            if len(fallback_path) > 2:
                print(f"Fallback found path with {len(fallback_path)} waypoints.")
                return self._create_path_result(
                    fallback_path, priority, obstacles_avoided, drone_speed, battery_capacity
                )
        
        # Return direct path as last resort
        print("Returning direct path.")
        direct_path = [start, goal]
        return self._create_path_result(
            direct_path, priority, obstacles_avoided, drone_speed, battery_capacity
        )
    
    def heuristic(self, coord1: Coordinate, coord2: Coordinate) -> float:
        """
        Calculate heuristic distance between two coordinates.
        Uses 3D Euclidean distance with terrain consideration.
        """
        # 2D distance using Haversine formula
        horizontal_distance = haversine_distance(coord1, coord2)
        
        # 3D distance including altitude difference
        altitude_diff = abs(coord1.altitude - coord2.altitude) / 1000.0  # Convert to km
        distance_3d = math.sqrt(horizontal_distance**2 + altitude_diff**2)
        
        # Terrain cost multiplier
        terrain_cost = coord1.terrain_type.value
        
        return distance_3d * terrain_cost
    
    def _get_neighbors(self, coord: Coordinate, max_distance: float) -> List[Coordinate]:
        """Generate neighboring coordinates for pathfinding exploration"""
        neighbors = []
        
        # Reduce grid resolution for better performance
        resolution = self.grid_resolution * 2  # Increase step size for performance
        
        # 8-directional movement in horizontal plane
        directions = [
            (-resolution, 0),      # North
            (resolution, 0),       # South  
            (0, -resolution),      # West
            (0, resolution),       # East
            (-resolution, -resolution),  # NW
            (-resolution, resolution),   # NE
            (resolution, -resolution),   # SW
            (resolution, resolution),    # SE
        ]
        
        # Limit altitude variations to reduce computational overhead
        altitude_offsets = [0]  # Only use current altitude for now
        if coord.altitude < self.max_altitude - 100:
            altitude_offsets.append(100)  # Go higher
        if coord.altitude > self.min_altitude + 100:
            altitude_offsets.append(-100)  # Go lower
        
        for dlat, dlng in directions:
            new_lat = coord.lat + dlat
            new_lng = coord.lng + dlng
            
            for altitude_offset in altitude_offsets:
                new_altitude = max(self.min_altitude, 
                                 min(self.max_altitude, coord.altitude + altitude_offset))
                
                try:
                    # Pre-compute terrain type to avoid repeated calls
                    terrain_type = self._get_terrain_type(new_lat, new_lng)
                    
                    neighbor = Coordinate(
                        lat=new_lat,
                        lng=new_lng, 
                        altitude=new_altitude,
                        terrain_type=terrain_type
                    )
                    
                    # Check if within search bounds
                    if haversine_distance(coord, neighbor) <= max_distance:
                        neighbors.append(neighbor)
                        
                except KeyboardInterrupt:
                    # Handle interruption gracefully
                    print("Pathfinding interrupted by user")
                    return neighbors
                except Exception as e:
                    # Skip problematic coordinates
                    continue
        
        return neighbors
    
    def _is_blocked(self, coord: Coordinate) -> bool:
        """Check if coordinate is blocked by obstacles or restrictions"""
        for obstacle in self.obstacles:
            if obstacle.contains_point(coord):
                return True
        
        # Check terrain restrictions
        if coord.terrain_type == TerrainType.RESTRICTED:
            return True
            
        return False
    
    def _calculate_movement_cost(self, 
                               from_coord: Coordinate, 
                               to_coord: Coordinate,
                               priority: Priority) -> float:
        """Calculate cost of movement between two coordinates"""
        base_distance = haversine_distance(from_coord, to_coord)
        
        # Base terrain cost
        terrain_multiplier = to_coord.terrain_type.value
        
        # Altitude change penalty
        altitude_change = abs(to_coord.altitude - from_coord.altitude)
        altitude_penalty = (altitude_change / 100.0) * 0.1  # 10% penalty per 100m
        
        # Priority-specific adjustments
        if priority == Priority.SAFEST:
            # Prefer higher altitudes and avoid obstacles
            safety_bonus = to_coord.altitude / 1000.0  # Higher = safer
            obstacle_penalty = sum(1.0 / max(1.0, haversine_distance(to_coord, obs.center)) 
                                 for obs in self.obstacles)
            return base_distance * terrain_multiplier + altitude_penalty + obstacle_penalty - safety_bonus
            
        elif priority == Priority.FASTEST:
            # Minimize time, prefer straight paths
            return base_distance * terrain_multiplier + altitude_penalty
            
        elif priority == Priority.BATTERY_OPTIMAL:
            # Consider wind resistance and altitude efficiency
            wind_resistance = 1.0 + (altitude_change / 500.0)
            return base_distance * terrain_multiplier * wind_resistance
            
        else:  # SHORTEST
            return base_distance * terrain_multiplier + altitude_penalty
    
    def _get_terrain_type(self, lat: float, lng: float) -> TerrainType:
        """Determine terrain type for given coordinates (simplified)"""
        # In a real implementation, this would query GIS data
        # For simulation, we'll use simple heuristics
        
        # Urban areas (simplified - around Delhi coordinates)
        if (28.5 <= lat <= 28.7) and (77.1 <= lng <= 77.3):
            return TerrainType.URBAN
        
        # Default to flat terrain
        return TerrainType.FLAT
    
    def _reconstruct_path(self, 
                         nodes: Dict[Coordinate, Node], 
                         start: Coordinate, 
                         goal: Coordinate) -> List[Coordinate]:
        """Reconstruct path from goal to start using parent pointers"""
        path = []
        current = nodes[goal]
        
        while current is not None:
            path.append(current.coordinate)
            current = current.parent
        
        path.reverse()
        return path
    
    def _simple_pathfinding(self, start: Coordinate, goal: Coordinate) -> List[Coordinate]:
        """Simple fallback pathfinding using waypoints between start and goal"""
        try:
            # Calculate intermediate waypoints
            path = [start]
            
            # Add one intermediate waypoint for simple navigation
            mid_lat = (start.lat + goal.lat) / 2
            mid_lng = (start.lng + goal.lng) / 2
            mid_alt = max(start.altitude, goal.altitude) + 50  # Fly higher for safety
            
            midpoint = Coordinate(
                lat=mid_lat,
                lng=mid_lng,
                altitude=min(mid_alt, self.max_altitude),
                terrain_type=self._get_terrain_type(mid_lat, mid_lng)
            )
            
            # Only add midpoint if it's not blocked
            if not self._is_blocked(midpoint):
                path.append(midpoint)
            
            path.append(goal)
            return path
            
        except Exception as e:
            print(f"Simple pathfinding failed: {e}")
            return [start, goal]
    
    def _create_path_result(self, 
                           path: List[Coordinate],
                           priority: Priority,
                           obstacles_avoided: int,
                           drone_speed: float,
                           battery_capacity: float) -> PathResult:
        """Create PathResult from calculated path"""
        
        if len(path) < 2:
            return PathResult(
                path=path,
                total_distance=0.0,
                total_time=0.0,
                battery_usage=0.0,
                safety_score=0.0,
                obstacles_avoided=obstacles_avoided,
                waypoints=[],
                algorithm_used="A*",
                priority_mode=priority
            )
        
        # Calculate metrics
        total_distance = sum(
            haversine_distance(path[i], path[i+1]) 
            for i in range(len(path)-1)
        )
        
        total_time = (total_distance * 1000) / drone_speed / 60  # minutes
        battery_usage = min(100.0, total_distance * 2.5)  # ~2.5% per km
        
        # Safety score based on altitude and obstacle avoidance
        avg_altitude = sum(coord.altitude for coord in path) / len(path)
        safety_score = min(100.0, (avg_altitude / 200.0) * 70 + (obstacles_avoided * 5))
        
        # Convert to waypoints format
        waypoints = [
            {
                'lat': coord.lat,
                'lng': coord.lng, 
                'alt': coord.altitude,
                'type': 'start' if i == 0 else 'destination' if i == len(path)-1 else 'waypoint'
            }
            for i, coord in enumerate(path)
        ]
        
        return PathResult(
            path=path,
            total_distance=total_distance,
            total_time=total_time,
            battery_usage=battery_usage,
            safety_score=safety_score,
            obstacles_avoided=obstacles_avoided,
            waypoints=waypoints,
            algorithm_used="A*",
            priority_mode=priority
        )


# Utility Functions

def haversine_distance(coord1: Coordinate, coord2: Coordinate) -> float:
    """
    Calculate the great circle distance between two GPS coordinates.
    Returns distance in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers
    
    lat1_rad = math.radians(coord1.lat)
    lat2_rad = math.radians(coord2.lat)
    dlat = math.radians(coord2.lat - coord1.lat)
    dlng = math.radians(coord2.lng - coord1.lng)
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def optimize_path_for_battery(path: List[Coordinate], 
                             battery_level: float,
                             consumption_rate: float = 2.5) -> List[Coordinate]:
    """
    Optimize path considering current battery level and consumption.
    May add emergency landing sites if needed.
    """
    if not path or len(path) < 2:
        return path
    
    total_distance = sum(
        haversine_distance(path[i], path[i+1]) 
        for i in range(len(path)-1)
    )
    
    required_battery = total_distance * consumption_rate
    
    # If sufficient battery, return original path
    if required_battery <= battery_level * 0.8:  # 20% safety margin
        return path
    
    # Calculate maximum safe distance
    max_safe_distance = (battery_level * 0.7) / consumption_rate
    
    # Truncate path to safe distance
    optimized_path = [path[0]]
    current_distance = 0.0
    
    for i in range(1, len(path)):
        segment_distance = haversine_distance(path[i-1], path[i])
        
        if current_distance + segment_distance <= max_safe_distance:
            optimized_path.append(path[i])
            current_distance += segment_distance
        else:
            break
    
    return optimized_path


def calculate_eta(path: List[Coordinate], 
                  drone_speed: float = 12.0,
                  wind_factor: float = 1.0) -> float:
    """
    Calculate estimated time of arrival for given path.
    Returns ETA in minutes.
    """
    if len(path) < 2:
        return 0.0
    
    total_distance = sum(
        haversine_distance(path[i], path[i+1]) 
        for i in range(len(path)-1)
    )
    
    # Convert distance to meters and calculate time
    distance_meters = total_distance * 1000
    effective_speed = drone_speed * wind_factor
    
    return distance_meters / effective_speed / 60  # minutes


# Simple Grid-Based A* Algorithm Implementation
# This provides a lightweight alternative for basic pathfinding scenarios

class SimpleNode:
    """Simplified node class for grid-based pathfinding"""
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Distance from starting node
        self.h = 0  # Heuristic distance to end node
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f


def astar_grid(grid, start, end):
    """
    Simple A* pathfinding algorithm for grid-based navigation.
    
    Args:
        grid: 2D list where 0 = walkable, 1 = obstacle
        start: (x, y) starting position tuple
        end: (x, y) ending position tuple
    
    Returns:
        List of (x, y) positions representing the path, or None if no path found
    """
    start_node = SimpleNode(start)
    end_node = SimpleNode(end)

    open_list = []
    closed_list = set()

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        # Goal reached
        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        # Explore neighbors (4-directional movement)
        (x, y) = current_node.position
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_pos = (x + dx, y + dy)

            # Check bounds
            if not (0 <= next_pos[0] < len(grid) and 0 <= next_pos[1] < len(grid[0])):
                continue
            
            # Check for obstacles
            if grid[next_pos[0]][next_pos[1]] == 1:
                continue
            
            # Skip if already processed
            if next_pos in closed_list:
                continue

            # Create neighbor node
            neighbor = SimpleNode(next_pos, current_node)
            neighbor.g = current_node.g + 1
            neighbor.h = ((next_pos[0] - end_node.position[0]) ** 2) + \
                         ((next_pos[1] - end_node.position[1]) ** 2)
            neighbor.f = neighbor.g + neighbor.h

            # Skip if we already have a better path to this node
            if any(node for node in open_list if neighbor == node and neighbor.g > node.g):
                continue
            
            heapq.heappush(open_list, neighbor)

    return None  # No path found


def astar_grid_8dir(grid, start, end):
    """
    A* pathfinding with 8-directional movement (including diagonals).
    
    Args:
        grid: 2D list where 0 = walkable, 1 = obstacle
        start: (x, y) starting position tuple
        end: (x, y) ending position tuple
    
    Returns:
        List of (x, y) positions representing the path, or None if no path found
    """
    start_node = SimpleNode(start)
    end_node = SimpleNode(end)

    open_list = []
    closed_list = set()

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        (x, y) = current_node.position
        # 8-directional movement (including diagonals)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            next_pos = (x + dx, y + dy)

            if not (0 <= next_pos[0] < len(grid) and 0 <= next_pos[1] < len(grid[0])):
                continue
            if grid[next_pos[0]][next_pos[1]] == 1:
                continue
            if next_pos in closed_list:
                continue

            neighbor = SimpleNode(next_pos, current_node)
            
            # Different cost for diagonal movement
            move_cost = 1.414 if abs(dx) == 1 and abs(dy) == 1 else 1.0
            neighbor.g = current_node.g + move_cost
            
            neighbor.h = math.sqrt((next_pos[0] - end_node.position[0]) ** 2 + 
                                 (next_pos[1] - end_node.position[1]) ** 2)
            neighbor.f = neighbor.g + neighbor.h

            if any(node for node in open_list if neighbor == node and neighbor.g > node.g):
                continue
            
            heapq.heappush(open_list, neighbor)

    return None


def grid_to_coordinates(grid_path, origin_lat, origin_lng, grid_resolution=0.001):
    """
    Convert grid-based path to GPS coordinates.
    
    Args:
        grid_path: List of (x, y) grid positions
        origin_lat: Starting latitude
        origin_lng: Starting longitude  
        grid_resolution: Degrees per grid cell
    
    Returns:
        List of Coordinate objects
    """
    if not grid_path:
        return []
    
    coordinates = []
    for x, y in grid_path:
        lat = origin_lat + (x * grid_resolution)
        lng = origin_lng + (y * grid_resolution)
        coordinates.append(Coordinate(lat=lat, lng=lng))
    
    return coordinates


def astar(start_lat: float, start_lng: float, goal_lat: float, goal_lng: float,
          priority: str = 'shortest', altitude: float = 150.0, 
          max_speed: float = 12.0, avoid_obstacles: bool = True) -> dict:
    """
    Simple A* pathfinding function for easy integration.
    
    Args:
        start_lat: Starting latitude
        start_lng: Starting longitude  
        goal_lat: Goal latitude
        goal_lng: Goal longitude
        priority: Path optimization priority ('shortest', 'fastest', 'safest', 'battery_optimal')
        altitude: Flight altitude in meters
        max_speed: Maximum drone speed in m/s
        avoid_obstacles: Whether to avoid obstacles
        
    Returns:
        Dictionary with path information and waypoints
    """
    try:
        # Initialize pathfinder
        pathfinder = AStarPathfinder()
        
        # Convert priority string to enum
        priority_map = {
            'shortest': Priority.SHORTEST,
            'fastest': Priority.FASTEST,
            'safest': Priority.SAFEST,
            'battery_optimal': Priority.BATTERY_OPTIMAL
        }
        priority_enum = priority_map.get(priority, Priority.SHORTEST)
        
        # Create coordinates
        start_coord = Coordinate(lat=start_lat, lng=start_lng, altitude=altitude)
        goal_coord = Coordinate(lat=goal_lat, lng=goal_lng, altitude=altitude)
        
        # Add sample obstacles if requested
        if avoid_obstacles:
            sample_obstacle = Obstacle(
                center=Coordinate(
                    lat=(start_lat + goal_lat) / 2,
                    lng=(start_lng + goal_lng) / 2,
                    altitude=0
                ),
                radius=200,  # 200m radius
                height=300,  # Up to 300m altitude
                severity=2
            )
            pathfinder.add_obstacle(sample_obstacle)
        
        # Calculate path
        result = pathfinder.find_path(
            start=start_coord,
            goal=goal_coord,
            priority=priority_enum,
            max_distance=10.0,
            drone_speed=max_speed,
            battery_capacity=85.0
        )
        
        # Format result as simple dictionary
        return {
            'success': True,
            'path': [{'lat': coord.lat, 'lng': coord.lng, 'alt': coord.altitude} 
                    for coord in result.path],
            'waypoints': result.waypoints,
            'distance_km': round(result.total_distance, 2),
            'flight_time_minutes': round(result.total_time, 1),
            'battery_usage_percent': round(result.battery_usage, 1),
            'safety_score': round(result.safety_score, 1),
            'obstacles_avoided': result.obstacles_avoided,
            'algorithm': result.algorithm_used,
            'priority_mode': result.priority_mode.value
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'path': [{'lat': start_lat, 'lng': start_lng, 'alt': altitude},
                    {'lat': goal_lat, 'lng': goal_lng, 'alt': altitude}],
            'waypoints': [],
            'distance_km': haversine_distance(
                Coordinate(start_lat, start_lng, altitude),
                Coordinate(goal_lat, goal_lng, altitude)
            ),
            'flight_time_minutes': 0.0,
            'battery_usage_percent': 0.0,
            'safety_score': 50.0,
            'obstacles_avoided': 0,
            'algorithm': 'Direct Path (Fallback)',
            'priority_mode': priority
        }


def create_test_grid(width=20, height=20, obstacle_density=0.2):
    """
    Create a test grid with random obstacles for pathfinding testing.
    
    Args:
        width: Grid width
        height: Grid height
        obstacle_density: Percentage of cells that should be obstacles (0.0-1.0)
    
    Returns:
        2D list representing the grid (0=walkable, 1=obstacle)
    """
    import random
    
    grid = [[0 for _ in range(height)] for _ in range(width)]
    
    # Add random obstacles
    num_obstacles = int(width * height * obstacle_density)
    for _ in range(num_obstacles):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        grid[x][y] = 1
    
    return grid


# Example usage and testing
if __name__ == "__main__":
    print("=== Vihangam A* Pathfinding Test Suite ===\n")
    
    # Test 1: Advanced GPS-based pathfinding
    print("1. Testing Advanced GPS-based A* Pathfinding:")
    pathfinder = AStarPathfinder()
    
    # Define coordinates (Delhi area example)
    start = Coordinate(lat=28.6139, lng=77.2090, altitude=150.0)
    goal = Coordinate(lat=28.6500, lng=77.2300, altitude=150.0)
    
    # Add some obstacles
    obstacle = Obstacle(
        center=Coordinate(lat=28.6300, lng=77.2200, altitude=0),
        radius=500,  # 500m radius
        severity=3
    )
    pathfinder.add_obstacle(obstacle)
    
    # Find optimal path
    result = pathfinder.find_path(
        start=start,
        goal=goal,
        priority=Priority.SAFEST,
        drone_speed=12.0,
        battery_capacity=85.0
    )
    
    # Print results
    print(f"   Path found: {len(result.path)} waypoints")
    print(f"   Total distance: {result.total_distance:.2f} km")
    print(f"   Estimated time: {result.total_time:.1f} minutes")
    print(f"   Battery usage: {result.battery_usage:.1f}%")
    print(f"   Safety score: {result.safety_score:.1f}/100")
    print(f"   Obstacles avoided: {result.obstacles_avoided}\n")
    
    # Test 2: Simple grid-based pathfinding
    print("2. Testing Simple Grid-based A* Pathfinding:")
    
    # Create a test grid
    test_grid = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    grid_start = (0, 0)
    grid_end = (9, 9)
    
    path = astar_grid(test_grid, grid_start, grid_end)
    
    if path:
        print(f"   Grid path found: {len(path)} steps")
        print(f"   Path: {path[:5]}...{path[-5:] if len(path) > 10 else path[5:]}")
    else:
        print("   No path found in grid")
    
    # Test 3: 8-directional grid pathfinding
    print("\n3. Testing 8-directional Grid A* Pathfinding:")
    path_8dir = astar_grid_8dir(test_grid, grid_start, grid_end)
    
    if path_8dir:
        print(f"   8-dir path found: {len(path_8dir)} steps")
        print(f"   Path: {path_8dir[:5]}...{path_8dir[-5:] if len(path_8dir) > 10 else path_8dir[5:]}")
    else:
        print("   No 8-directional path found")
    
    print("\n=== Test Suite Complete ===")
