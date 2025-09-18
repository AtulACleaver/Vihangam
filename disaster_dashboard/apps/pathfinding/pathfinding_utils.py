"""
Enhanced Pathfinding Utilities for Vihangam Drone Navigation System

This module provides advanced utilities for drone pathfinding including:
- Path smoothing and optimization
- Real-time path validation 
- Dynamic replanning capabilities
- Emergency response handling
- Performance optimization
"""

import math
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from .astar import (
    Coordinate, AStarPathfinder, PathResult, Priority, 
    Obstacle, haversine_distance, TerrainType
)


@dataclass
class FlightConstraints:
    """Flight constraints for path validation"""
    max_speed: float = 25.0  # m/s
    min_speed: float = 5.0   # m/s
    max_altitude: float = 500.0  # meters
    min_altitude: float = 50.0   # meters
    max_climb_rate: float = 5.0  # m/s
    max_descent_rate: float = 3.0  # m/s
    max_bank_angle: float = 45.0  # degrees
    min_turning_radius: float = 50.0  # meters
    safety_margin: float = 100.0  # meters from obstacles


@dataclass 
class PathValidationResult:
    """Result of path validation checks"""
    is_valid: bool
    violations: List[str]
    safety_score: float
    estimated_completion_time: float
    fuel_efficiency_score: float
    recommended_fixes: List[str]


class PathSmoother:
    """Smooth and optimize calculated paths for realistic drone flight"""
    
    def __init__(self, constraints: FlightConstraints):
        self.constraints = constraints
    
    def smooth_path(self, path: List[Coordinate], smoothing_factor: float = 0.5) -> List[Coordinate]:
        """
        Apply path smoothing to reduce sharp turns and create realistic flight paths.
        
        Args:
            path: Original path coordinates
            smoothing_factor: Smoothing intensity (0.0 = no smoothing, 1.0 = maximum)
            
        Returns:
            Smoothed path coordinates
        """
        if len(path) < 3:
            return path
            
        smoothed = [path[0]]  # Keep start point
        
        for i in range(1, len(path) - 1):
            prev_point = path[i-1]
            current_point = path[i]
            next_point = path[i+1]
            
            # Calculate smoothed position using weighted average
            smooth_lat = (
                prev_point.lat * (1 - smoothing_factor) / 2 +
                current_point.lat * smoothing_factor +
                next_point.lat * (1 - smoothing_factor) / 2
            )
            
            smooth_lng = (
                prev_point.lng * (1 - smoothing_factor) / 2 +
                current_point.lng * smoothing_factor +
                next_point.lng * (1 - smoothing_factor) / 2
            )
            
            smooth_alt = (
                prev_point.altitude * (1 - smoothing_factor) / 2 +
                current_point.altitude * smoothing_factor +
                next_point.altitude * (1 - smoothing_factor) / 2
            )
            
            smoothed_coord = Coordinate(
                lat=smooth_lat,
                lng=smooth_lng,
                altitude=smooth_alt,
                terrain_type=current_point.terrain_type
            )
            
            smoothed.append(smoothed_coord)
        
        smoothed.append(path[-1])  # Keep end point
        return smoothed
    
    def add_banking_waypoints(self, path: List[Coordinate], bank_factor: float = 0.7) -> List[Coordinate]:
        """Add intermediate waypoints for realistic banking in turns"""
        if len(path) < 3:
            return path
            
        enhanced_path = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev_point = path[i-1]
            current_point = path[i]
            next_point = path[i+1]
            
            # Calculate turn angle
            bearing1 = self._calculate_bearing(prev_point, current_point)
            bearing2 = self._calculate_bearing(current_point, next_point)
            turn_angle = abs(bearing2 - bearing1)
            
            # Add banking waypoints for sharp turns
            if turn_angle > 30:  # degrees
                bank_waypoint = self._calculate_bank_waypoint(
                    prev_point, current_point, next_point, bank_factor
                )
                enhanced_path.append(bank_waypoint)
            
            enhanced_path.append(current_point)
        
        enhanced_path.append(path[-1])
        return enhanced_path
    
    def _calculate_bearing(self, coord1: Coordinate, coord2: Coordinate) -> float:
        """Calculate bearing between two coordinates in degrees"""
        lat1 = math.radians(coord1.lat)
        lat2 = math.radians(coord2.lat)
        diff_lng = math.radians(coord2.lng - coord1.lng)
        
        x = math.sin(diff_lng) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(diff_lng))
        
        bearing = math.atan2(x, y)
        return math.degrees(bearing) % 360
    
    def _calculate_bank_waypoint(self, prev: Coordinate, current: Coordinate, 
                                next: Coordinate, bank_factor: float) -> Coordinate:
        """Calculate banking waypoint for smooth turns"""
        # Simple banking calculation - offset perpendicular to flight path
        mid_lat = (prev.lat + next.lat) / 2
        mid_lng = (prev.lng + next.lng) / 2
        
        # Offset slightly for banking effect
        bank_offset = 0.0001 * bank_factor
        
        return Coordinate(
            lat=current.lat + (mid_lat - current.lat) * 0.1,
            lng=current.lng + (mid_lng - current.lng) * 0.1,
            altitude=current.altitude + 10,  # Slight altitude gain for banking
            terrain_type=current.terrain_type
        )


class PathValidator:
    """Comprehensive path validation for flight safety"""
    
    def __init__(self, constraints: FlightConstraints):
        self.constraints = constraints
    
    def validate_path(self, path: List[Coordinate], obstacles: List[Obstacle] = None) -> PathValidationResult:
        """
        Comprehensive path validation against flight constraints and safety requirements.
        
        Args:
            path: Path to validate
            obstacles: List of known obstacles
            
        Returns:
            Detailed validation result
        """
        if not path or len(path) < 2:
            return PathValidationResult(
                is_valid=False,
                violations=["Path too short or empty"],
                safety_score=0.0,
                estimated_completion_time=0.0,
                fuel_efficiency_score=0.0,
                recommended_fixes=["Provide valid start and end coordinates"]
            )
        
        violations = []
        safety_score = 100.0
        recommended_fixes = []
        
        # Check altitude constraints
        altitude_violations = self._check_altitude_constraints(path)
        violations.extend(altitude_violations)
        if altitude_violations:
            safety_score -= len(altitude_violations) * 10
            recommended_fixes.append("Adjust altitude to stay within limits")
        
        # Check climb/descent rates
        climb_violations = self._check_climb_rates(path)
        violations.extend(climb_violations)
        if climb_violations:
            safety_score -= len(climb_violations) * 15
            recommended_fixes.append("Add intermediate waypoints for gradual climbs/descents")
        
        # Check turning constraints
        turn_violations = self._check_turning_constraints(path)
        violations.extend(turn_violations)
        if turn_violations:
            safety_score -= len(turn_violations) * 5
            recommended_fixes.append("Apply path smoothing to reduce sharp turns")
        
        # Check obstacle clearance
        if obstacles:
            obstacle_violations = self._check_obstacle_clearance(path, obstacles)
            violations.extend(obstacle_violations)
            if obstacle_violations:
                safety_score -= len(obstacle_violations) * 20
                recommended_fixes.append("Recalculate path to avoid obstacles")
        
        # Calculate performance metrics
        total_distance = sum(haversine_distance(path[i], path[i+1]) for i in range(len(path)-1))
        estimated_time = total_distance * 1000 / 12.0 / 60  # Assume 12 m/s cruise speed
        fuel_efficiency = self._calculate_fuel_efficiency(path)
        
        return PathValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            safety_score=max(0.0, safety_score),
            estimated_completion_time=estimated_time,
            fuel_efficiency_score=fuel_efficiency,
            recommended_fixes=recommended_fixes
        )
    
    def _check_altitude_constraints(self, path: List[Coordinate]) -> List[str]:
        """Check if path respects altitude constraints"""
        violations = []
        for i, coord in enumerate(path):
            if coord.altitude > self.constraints.max_altitude:
                violations.append(f"Waypoint {i}: Altitude {coord.altitude}m exceeds maximum {self.constraints.max_altitude}m")
            elif coord.altitude < self.constraints.min_altitude:
                violations.append(f"Waypoint {i}: Altitude {coord.altitude}m below minimum {self.constraints.min_altitude}m")
        return violations
    
    def _check_climb_rates(self, path: List[Coordinate]) -> List[str]:
        """Check climb and descent rates between waypoints"""
        violations = []
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            
            altitude_change = next_point.altitude - current.altitude
            horizontal_distance = haversine_distance(current, next_point) * 1000  # meters
            
            if horizontal_distance > 0:
                climb_angle = math.degrees(math.atan(altitude_change / horizontal_distance))
                
                if altitude_change > 0:  # Climbing
                    max_climb_angle = math.degrees(math.atan(self.constraints.max_climb_rate / 10))  # Assume 10 m/s forward speed
                    if climb_angle > max_climb_angle:
                        violations.append(f"Segment {i}-{i+1}: Climb rate too steep ({climb_angle:.1f}° > {max_climb_angle:.1f}°)")
                else:  # Descending
                    max_descent_angle = math.degrees(math.atan(self.constraints.max_descent_rate / 10))
                    if abs(climb_angle) > max_descent_angle:
                        violations.append(f"Segment {i}-{i+1}: Descent rate too steep ({abs(climb_angle):.1f}° > {max_descent_angle:.1f}°)")
        
        return violations
    
    def _check_turning_constraints(self, path: List[Coordinate]) -> List[str]:
        """Check turning radius constraints"""
        violations = []
        if len(path) < 3:
            return violations
            
        for i in range(1, len(path) - 1):
            prev_point = path[i-1]
            current_point = path[i]
            next_point = path[i+1]
            
            # Calculate turn angle
            bearing1 = math.atan2(current_point.lng - prev_point.lng, current_point.lat - prev_point.lat)
            bearing2 = math.atan2(next_point.lng - current_point.lng, next_point.lat - current_point.lat)
            turn_angle = abs(bearing2 - bearing1)
            
            # Check if turn is too sharp
            if turn_angle > math.radians(self.constraints.max_bank_angle):
                violations.append(f"Waypoint {i}: Turn too sharp ({math.degrees(turn_angle):.1f}° > {self.constraints.max_bank_angle}°)")
        
        return violations
    
    def _check_obstacle_clearance(self, path: List[Coordinate], obstacles: List[Obstacle]) -> List[str]:
        """Check minimum clearance from obstacles"""
        violations = []
        
        for i, coord in enumerate(path):
            for j, obstacle in enumerate(obstacles):
                distance = haversine_distance(coord, obstacle.center) * 1000  # Convert to meters
                
                if distance < (obstacle.radius + self.constraints.safety_margin):
                    violations.append(
                        f"Waypoint {i}: Too close to obstacle {j} "
                        f"({distance:.1f}m < {obstacle.radius + self.constraints.safety_margin:.1f}m required)"
                    )
        
        return violations
    
    def _calculate_fuel_efficiency(self, path: List[Coordinate]) -> float:
        """Calculate fuel efficiency score (0-100)"""
        if len(path) < 2:
            return 0.0
        
        total_distance = sum(haversine_distance(path[i], path[i+1]) for i in range(len(path)-1))
        direct_distance = haversine_distance(path[0], path[-1])
        
        if direct_distance == 0:
            return 100.0
        
        efficiency_ratio = direct_distance / total_distance
        return min(100.0, efficiency_ratio * 100)


class DynamicReplanner:
    """Handle real-time path replanning for dynamic obstacles and conditions"""
    
    def __init__(self, pathfinder: AStarPathfinder, constraints: FlightConstraints):
        self.pathfinder = pathfinder
        self.constraints = constraints
        self.last_replan_time = 0
        self.replan_interval = 10.0  # seconds
        
    def should_replan(self, current_pos: Coordinate, current_path: List[Coordinate], 
                     new_obstacles: List[Obstacle] = None) -> bool:
        """Determine if path replanning is necessary"""
        current_time = time.time()
        
        # Time-based replanning
        if current_time - self.last_replan_time < self.replan_interval:
            return False
        
        # Check for new obstacles in current path
        if new_obstacles:
            for obstacle in new_obstacles:
                for waypoint in current_path:
                    if obstacle.contains_point(waypoint):
                        return True
        
        # Check if significantly off course
        if len(current_path) > 1:
            next_waypoint = current_path[1]
            distance_to_next = haversine_distance(current_pos, next_waypoint) * 1000
            
            # If more than 200m off course, replan
            if distance_to_next > 200:
                return True
        
        return False
    
    def replan_from_current_position(self, current_pos: Coordinate, 
                                   original_destination: Coordinate,
                                   new_obstacles: List[Obstacle] = None,
                                   priority: Priority = Priority.FASTEST) -> PathResult:
        """Perform emergency replanning from current position"""
        self.last_replan_time = time.time()
        
        # Add new obstacles to pathfinder
        if new_obstacles:
            for obstacle in new_obstacles:
                self.pathfinder.add_obstacle(obstacle)
        
        # Calculate new path from current position
        return self.pathfinder.find_path(
            start=current_pos,
            goal=original_destination,
            priority=priority,
            max_distance=50.0,
            drone_speed=12.0,
            battery_capacity=70.0  # Conservative estimate for replanning
        )
    
    def get_emergency_landing_sites(self, current_pos: Coordinate, 
                                  max_distance: float = 5.0) -> List[Coordinate]:
        """Find suitable emergency landing sites near current position"""
        landing_sites = []
        
        # Generate potential landing sites in a grid pattern
        search_resolution = 0.01  # ~1km resolution
        search_range = int(max_distance / 111.0 / search_resolution)  # Convert km to degrees
        
        for i in range(-search_range, search_range + 1):
            for j in range(-search_range, search_range + 1):
                candidate = Coordinate(
                    lat=current_pos.lat + i * search_resolution,
                    lng=current_pos.lng + j * search_resolution,
                    altitude=50.0,  # Low altitude for landing
                    terrain_type=TerrainType.FLAT
                )
                
                # Check if site is clear of obstacles
                is_clear = True
                for obstacle in self.pathfinder.obstacles:
                    if obstacle.contains_point(candidate, safety_margin=200.0):
                        is_clear = False
                        break
                
                if is_clear:
                    distance = haversine_distance(current_pos, candidate)
                    if distance <= max_distance:
                        landing_sites.append(candidate)
        
        # Sort by distance from current position
        landing_sites.sort(key=lambda site: haversine_distance(current_pos, site))
        
        return landing_sites[:5]  # Return top 5 closest sites


class PathOptimizer:
    """Advanced path optimization techniques"""
    
    @staticmethod
    def optimize_for_wind(path: List[Coordinate], wind_direction: float, 
                         wind_speed: float) -> List[Coordinate]:
        """Optimize path considering wind conditions"""
        if len(path) < 2:
            return path
            
        optimized = [path[0]]
        
        for i in range(1, len(path)):
            current = path[i-1]
            next_point = path[i]
            
            # Calculate flight bearing
            bearing = math.atan2(next_point.lng - current.lng, next_point.lat - current.lat)
            bearing_deg = math.degrees(bearing) % 360
            
            # Calculate wind effect
            wind_angle_diff = abs(bearing_deg - wind_direction)
            if wind_angle_diff > 180:
                wind_angle_diff = 360 - wind_angle_diff
            
            # Adjust altitude based on wind
            if wind_angle_diff < 45:  # Headwind
                altitude_adjustment = 20  # Climb for efficiency
            elif wind_angle_diff > 135:  # Tailwind
                altitude_adjustment = -10  # Descend slightly
            else:  # Crosswind
                altitude_adjustment = 0
            
            optimized_coord = Coordinate(
                lat=next_point.lat,
                lng=next_point.lng,
                altitude=max(50, min(500, next_point.altitude + altitude_adjustment)),
                terrain_type=next_point.terrain_type
            )
            
            optimized.append(optimized_coord)
        
        return optimized
    
    @staticmethod
    def add_inspection_waypoints(path: List[Coordinate], 
                               inspection_points: List[Coordinate],
                               max_detour: float = 2.0) -> List[Coordinate]:
        """Add inspection waypoints to path if they don't add significant distance"""
        if not inspection_points:
            return path
            
        enhanced_path = [path[0]]
        
        for i in range(1, len(path)):
            current = path[i-1]
            next_point = path[i]
            
            # Check if any inspection points are near this segment
            for inspection_point in inspection_points:
                # Calculate if inspection point is worth a detour
                original_distance = haversine_distance(current, next_point)
                detour_distance = (haversine_distance(current, inspection_point) + 
                                 haversine_distance(inspection_point, next_point))
                
                if detour_distance - original_distance <= max_detour:
                    enhanced_path.append(inspection_point)
            
            enhanced_path.append(next_point)
        
        return enhanced_path


class PathCache:
    """Caching system for frequently used paths"""
    
    def __init__(self, max_cache_size: int = 100):
        self.cache: Dict[str, PathResult] = {}
        self.cache_times: Dict[str, float] = {}
        self.max_cache_size = max_cache_size
        self.cache_ttl = 300  # 5 minutes
    
    def _generate_cache_key(self, start: Coordinate, goal: Coordinate, 
                          priority: Priority) -> str:
        """Generate unique cache key for path request"""
        return f"{start.lat:.6f},{start.lng:.6f},{start.altitude:.1f}->{goal.lat:.6f},{goal.lng:.6f},{goal.altitude:.1f}:{priority.value}"
    
    def get_cached_path(self, start: Coordinate, goal: Coordinate, 
                       priority: Priority) -> Optional[PathResult]:
        """Retrieve cached path if available and valid"""
        cache_key = self._generate_cache_key(start, goal, priority)
        
        if cache_key in self.cache:
            cache_time = self.cache_times[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return self.cache[cache_key]
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
                del self.cache_times[cache_key]
        
        return None
    
    def cache_path(self, start: Coordinate, goal: Coordinate, 
                  priority: Priority, result: PathResult):
        """Cache a calculated path result"""
        cache_key = self._generate_cache_key(start, goal, priority)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_cache_size:
            oldest_key = min(self.cache_times.keys(), key=lambda k: self.cache_times[k])
            del self.cache[oldest_key]
            del self.cache_times[oldest_key]
        
        self.cache[cache_key] = result
        self.cache_times[cache_key] = time.time()
    
    def clear_cache(self):
        """Clear all cached paths"""
        self.cache.clear()
        self.cache_times.clear()


# Utility functions for integration

def convert_path_to_mavlink(path: List[Coordinate]) -> List[Dict]:
    """Convert path to MAVLink waypoint format for drone communication"""
    waypoints = []
    
    for i, coord in enumerate(path):
        waypoint = {
            'seq': i,
            'frame': 3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
            'command': 16,  # MAV_CMD_NAV_WAYPOINT
            'current': 1 if i == 0 else 0,
            'autocontinue': 1,
            'param1': 0,  # Hold time
            'param2': 0,  # Accept radius
            'param3': 0,  # Pass radius
            'param4': 0,  # Yaw
            'x': coord.lat,
            'y': coord.lng,
            'z': coord.altitude
        }
        waypoints.append(waypoint)
    
    return waypoints


def estimate_battery_consumption(path: List[Coordinate], 
                               drone_weight: float = 2.5,  # kg
                               base_consumption: float = 200) -> float:  # Wh/hour
    """Estimate battery consumption for given path"""
    if len(path) < 2:
        return 0.0
    
    total_distance = sum(haversine_distance(path[i], path[i+1]) for i in range(len(path)-1))
    total_altitude_change = sum(abs(path[i+1].altitude - path[i].altitude) for i in range(len(path)-1))
    
    # Base consumption based on distance (assuming 12 m/s cruise speed)
    flight_time_hours = (total_distance * 1000 / 12) / 3600
    base_power = base_consumption * flight_time_hours
    
    # Additional power for altitude changes (climbing costs more energy)
    altitude_power = (total_altitude_change / 100) * 50  # 50Wh per 100m climb
    
    # Weight factor
    weight_factor = 1 + (drone_weight - 2.0) * 0.1  # 10% more power per kg above 2kg
    
    total_consumption = (base_power + altitude_power) * weight_factor
    return total_consumption


def create_kml_export(path: List[Coordinate], filename: str = "flight_path.kml"):
    """Export path to KML format for Google Earth visualization"""
    kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Vihangam Flight Path</name>
    <description>Generated drone navigation path</description>
    <Style id="yellowLineGreenPoly">
      <LineStyle>
        <color>7f00ffff</color>
        <width>4</width>
      </LineStyle>
      <PolyStyle>
        <color>7f00ff00</color>
      </PolyStyle>
    </Style>
    <Placemark>
      <name>Flight Path</name>
      <description>Autonomous drone navigation route</description>
      <styleUrl>#yellowLineGreenPoly</styleUrl>
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>
'''
    
    for coord in path:
        kml_content += f'          {coord.lng},{coord.lat},{coord.altitude}\n'
    
    kml_content += '''        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>'''
    
    with open(filename, 'w') as f:
        f.write(kml_content)
    
    return filename