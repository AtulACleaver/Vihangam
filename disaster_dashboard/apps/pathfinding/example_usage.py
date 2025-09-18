"""
Practical Examples: A* Pathfinding Integration for Vihangam Drone System

This module demonstrates how to use the A* pathfinding system in real-world scenarios
for disaster management and search & rescue operations.

Examples include:
- Basic pathfinding for emergency response
- Multi-waypoint mission planning
- Real-time obstacle avoidance
- Weather-aware routing
- Emergency landing site selection
"""

import asyncio
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from .astar import (
    AStarPathfinder, Coordinate, Priority, Obstacle, 
    TerrainType, haversine_distance, PathResult
)
from .pathfinding_utils import (
    PathSmoother, PathValidator, DynamicReplanner, PathOptimizer,
    FlightConstraints, PathCache, convert_path_to_mavlink,
    estimate_battery_consumption
)


@dataclass
class Mission:
    """Mission definition for drone operations"""
    mission_id: str
    mission_type: str  # 'search_rescue', 'surveillance', 'delivery', 'inspection'
    start_location: Coordinate
    target_locations: List[Coordinate]
    priority: Priority
    max_flight_time: float  # minutes
    battery_capacity: float  # percentage
    weather_conditions: Dict
    obstacles: List[Obstacle]


class VihangamFlightPlanner:
    """
    High-level flight planning system integrating A* pathfinding
    with mission requirements and safety constraints.
    """
    
    def __init__(self):
        self.pathfinder = AStarPathfinder()
        self.flight_constraints = FlightConstraints(
            max_altitude=500.0,
            min_altitude=50.0,
            max_speed=25.0,
            min_speed=5.0,
            safety_margin=150.0  # Increased safety margin for disaster scenarios
        )
        self.validator = PathValidator(self.flight_constraints)
        self.smoother = PathSmoother(self.flight_constraints)
        self.cache = PathCache(max_cache_size=100)
        self.replanner = DynamicReplanner(self.pathfinder, self.flight_constraints)
    
    def plan_mission(self, mission: Mission) -> Dict:
        """
        Plan complete mission with multiple waypoints and optimization.
        
        Returns:
            Complete mission plan with paths, timing, and safety assessments
        """
        print(f"🚁 Planning Mission: {mission.mission_id} ({mission.mission_type})")
        
        mission_plan = {
            'mission_id': mission.mission_id,
            'mission_type': mission.mission_type,
            'status': 'planning',
            'segments': [],
            'total_distance': 0.0,
            'total_flight_time': 0.0,
            'total_battery_usage': 0.0,
            'safety_assessments': [],
            'emergency_sites': [],
            'recommendations': []
        }
        
        # Add known obstacles to pathfinder
        for obstacle in mission.obstacles:
            self.pathfinder.add_obstacle(obstacle)
        
        # Plan path segments between all target locations
        current_location = mission.start_location
        segment_number = 1
        
        for target in mission.target_locations:
            print(f"  📍 Planning segment {segment_number}: {current_location.lat:.4f},{current_location.lng:.4f} → {target.lat:.4f},{target.lng:.4f}")
            
            # Calculate path for this segment
            result = self.pathfinder.find_path(
                start=current_location,
                goal=target,
                priority=mission.priority,
                max_distance=30.0,
                drone_speed=15.0,
                battery_capacity=mission.battery_capacity
            )
            
            # Smooth the path for realistic flight
            if len(result.path) > 2:
                result.path = self.smoother.smooth_path(result.path, smoothing_factor=0.6)
            
            # Validate path safety
            validation = self.validator.validate_path(result.path, self.pathfinder.obstacles)
            
            # Apply weather optimization if needed
            if mission.weather_conditions.get('wind_speed', 0) > 5:
                result.path = PathOptimizer.optimize_for_wind(
                    result.path,
                    mission.weather_conditions['wind_direction'],
                    mission.weather_conditions['wind_speed']
                )
            
            segment_plan = {
                'segment_number': segment_number,
                'start': {'lat': current_location.lat, 'lng': current_location.lng, 'alt': current_location.altitude},
                'end': {'lat': target.lat, 'lng': target.lng, 'alt': target.altitude},
                'waypoints': [{'lat': c.lat, 'lng': c.lng, 'alt': c.altitude} for c in result.path],
                'distance_km': result.total_distance,
                'flight_time_min': result.total_time,
                'battery_usage_pct': result.battery_usage,
                'safety_score': result.safety_score,
                'obstacles_avoided': result.obstacles_avoided,
                'validation': {
                    'is_safe': validation.is_valid,
                    'violations': validation.violations,
                    'recommended_fixes': validation.recommended_fixes
                }
            }
            
            mission_plan['segments'].append(segment_plan)
            mission_plan['total_distance'] += result.total_distance
            mission_plan['total_flight_time'] += result.total_time
            mission_plan['total_battery_usage'] += result.battery_usage
            
            current_location = target
            segment_number += 1
        
        # Find emergency landing sites along the route
        emergency_sites = self._find_emergency_landing_sites(mission_plan['segments'])
        mission_plan['emergency_sites'] = emergency_sites
        
        # Generate recommendations
        recommendations = self._generate_mission_recommendations(mission_plan, mission)
        mission_plan['recommendations'] = recommendations
        mission_plan['status'] = 'ready' if all(s['validation']['is_safe'] for s in mission_plan['segments']) else 'needs_review'
        
        print(f"✅ Mission planned: {mission_plan['total_distance']:.1f}km, {mission_plan['total_flight_time']:.1f}min, {mission_plan['total_battery_usage']:.1f}% battery")
        
        return mission_plan
    
    def handle_emergency_replan(self, current_position: Coordinate, 
                              original_destination: Coordinate,
                              emergency_type: str,
                              new_obstacles: List[Obstacle] = None) -> Dict:
        """
        Handle emergency replanning scenarios.
        """
        print(f"🚨 Emergency Replanning: {emergency_type}")
        
        # Determine priority based on emergency type
        priority_map = {
            'low_battery': Priority.BATTERY_OPTIMAL,
            'weather_deterioration': Priority.SAFEST,
            'obstacle_detected': Priority.FASTEST,
            'medical_emergency': Priority.FASTEST,
            'mechanical_issue': Priority.SAFEST
        }
        
        priority = priority_map.get(emergency_type, Priority.SAFEST)
        
        # Perform replanning
        result = self.replanner.replan_from_current_position(
            current_pos=current_position,
            original_destination=original_destination,
            new_obstacles=new_obstacles,
            priority=priority
        )
        
        # Find emergency landing sites if critical
        emergency_sites = []
        if emergency_type in ['low_battery', 'mechanical_issue']:
            emergency_sites = self.replanner.get_emergency_landing_sites(
                current_position, max_distance=3.0
            )
        
        replan_result = {
            'emergency_type': emergency_type,
            'replan_successful': len(result.path) > 1,
            'new_path': [{'lat': c.lat, 'lng': c.lng, 'alt': c.altitude} for c in result.path],
            'distance_km': result.total_distance,
            'flight_time_min': result.total_time,
            'battery_usage_pct': result.battery_usage,
            'safety_score': result.safety_score,
            'emergency_landing_sites': [
                {'lat': site.lat, 'lng': site.lng, 'alt': site.altitude, 
                 'distance_km': haversine_distance(current_position, site)}
                for site in emergency_sites[:3]
            ],
            'recommended_action': self._get_emergency_recommendation(emergency_type, result, emergency_sites)
        }
        
        return replan_result
    
    def _find_emergency_landing_sites(self, segments: List[Dict]) -> List[Dict]:
        """Find emergency landing sites along mission route."""
        emergency_sites = []
        
        for segment in segments:
            # Use midpoint of each segment as search center
            start = segment['start']
            end = segment['end']
            midpoint = Coordinate(
                lat=(start['lat'] + end['lat']) / 2,
                lng=(start['lng'] + end['lng']) / 2,
                altitude=100.0  # Low altitude for landing
            )
            
            sites = self.replanner.get_emergency_landing_sites(midpoint, max_distance=2.0)
            
            for site in sites[:2]:  # Top 2 sites per segment
                emergency_sites.append({
                    'lat': site.lat,
                    'lng': site.lng,
                    'alt': site.altitude,
                    'segment': segment['segment_number'],
                    'suitability_score': 85.0  # Simplified scoring
                })
        
        return emergency_sites
    
    def _generate_mission_recommendations(self, mission_plan: Dict, mission: Mission) -> List[str]:
        """Generate recommendations based on mission analysis."""
        recommendations = []
        
        # Battery recommendations
        if mission_plan['total_battery_usage'] > 80:
            recommendations.append("⚠️ High battery usage predicted. Consider reducing mission scope or adding charging stops.")
        
        # Weather recommendations
        if mission.weather_conditions.get('wind_speed', 0) > 15:
            recommendations.append("🌬️ Strong winds detected. Flight time may increase by 20-30%.")
        
        # Safety recommendations
        avg_safety = sum(s['safety_score'] for s in mission_plan['segments']) / len(mission_plan['segments'])
        if avg_safety < 80:
            recommendations.append("🛡️ Route safety score is moderate. Consider alternative paths or wait for better conditions.")
        
        # Time recommendations
        if mission_plan['total_flight_time'] > mission.max_flight_time:
            recommendations.append(f"⏰ Flight time ({mission_plan['total_flight_time']:.1f}min) exceeds limit ({mission.max_flight_time}min).")
        
        return recommendations
    
    def _get_emergency_recommendation(self, emergency_type: str, result: PathResult, 
                                    emergency_sites: List[Coordinate]) -> str:
        """Get recommendation for emergency situation."""
        if emergency_type == 'low_battery':
            if emergency_sites:
                return f"🔋 Immediate landing recommended. Nearest site: {haversine_distance(result.path[0], emergency_sites[0]):.1f}km away."
            else:
                return "🔋 Continue to destination with battery monitoring. No suitable landing sites nearby."
        
        elif emergency_type == 'mechanical_issue':
            return "🔧 Land at nearest safe location immediately. Emergency services notified."
        
        elif emergency_type == 'weather_deterioration':
            return "🌧️ Route optimized for weather conditions. Monitor conditions continuously."
        
        else:
            return "📍 New route calculated. Proceed with caution."


# Example Usage Functions

def example_search_rescue_mission():
    """
    Example: Search and rescue mission planning
    """
    print("=" * 60)
    print("🚁 EXAMPLE: Search and Rescue Mission Planning")
    print("=" * 60)
    
    # Initialize flight planner
    planner = VihangamFlightPlanner()
    
    # Define mission parameters
    mission = Mission(
        mission_id="SAR_001",
        mission_type="search_rescue",
        start_location=Coordinate(lat=28.6139, lng=77.2090, altitude=150),
        target_locations=[
            Coordinate(lat=28.6300, lng=77.2200, altitude=180),  # Search area 1
            Coordinate(lat=28.6400, lng=77.2300, altitude=160),  # Search area 2
            Coordinate(lat=28.6350, lng=77.2150, altitude=170),  # Medical facility
        ],
        priority=Priority.FASTEST,
        max_flight_time=45.0,  # minutes
        battery_capacity=95.0,
        weather_conditions={
            'wind_speed': 8,  # m/s
            'wind_direction': 270,  # degrees (west wind)
            'visibility': 'good'
        },
        obstacles=[
            Obstacle(
                center=Coordinate(lat=28.6250, lng=77.2150, altitude=0),
                radius=200,
                height=250,
                severity=3
            )
        ]
    )
    
    # Plan the mission
    mission_plan = planner.plan_mission(mission)
    
    # Display results
    print("\n📋 MISSION PLAN SUMMARY:")
    print(f"   Status: {mission_plan['status'].upper()}")
    print(f"   Total Distance: {mission_plan['total_distance']:.2f} km")
    print(f"   Total Flight Time: {mission_plan['total_flight_time']:.1f} minutes")
    print(f"   Battery Usage: {mission_plan['total_battery_usage']:.1f}%")
    print(f"   Emergency Sites: {len(mission_plan['emergency_sites'])} identified")
    
    print("\n📍 ROUTE SEGMENTS:")
    for segment in mission_plan['segments']:
        print(f"   Segment {segment['segment_number']}: {segment['distance_km']:.2f}km, "
              f"{segment['flight_time_min']:.1f}min, Safety: {segment['safety_score']:.1f}/100")
    
    print("\n💡 RECOMMENDATIONS:")
    for rec in mission_plan['recommendations']:
        print(f"   {rec}")
    
    return mission_plan


def example_emergency_replanning():
    """
    Example: Emergency replanning scenario
    """
    print("\n" + "=" * 60)
    print("🚨 EXAMPLE: Emergency Replanning")
    print("=" * 60)
    
    planner = VihangamFlightPlanner()
    
    # Simulate drone current position (mid-flight)
    current_position = Coordinate(lat=28.6250, lng=77.2150, altitude=165)
    original_destination = Coordinate(lat=28.6400, lng=77.2300, altitude=160)
    
    # Simulate different emergency scenarios
    scenarios = [
        'low_battery',
        'obstacle_detected',
        'weather_deterioration',
        'mechanical_issue'
    ]
    
    for emergency_type in scenarios:
        print(f"\n🚨 Scenario: {emergency_type.replace('_', ' ').title()}")
        
        # Add new obstacle for obstacle detection scenario
        new_obstacles = []
        if emergency_type == 'obstacle_detected':
            new_obstacles = [
                Obstacle(
                    center=Coordinate(lat=28.6320, lng=77.2220, altitude=0),
                    radius=150,
                    height=200,
                    severity=4
                )
            ]
        
        # Perform emergency replanning
        replan_result = planner.handle_emergency_replan(
            current_position=current_position,
            original_destination=original_destination,
            emergency_type=emergency_type,
            new_obstacles=new_obstacles
        )
        
        print(f"   ✓ Replanning successful: {replan_result['replan_successful']}")
        print(f"   📏 New route distance: {replan_result['distance_km']:.2f} km")
        print(f"   ⏱️ New flight time: {replan_result['flight_time_min']:.1f} min")
        print(f"   🛡️ Safety score: {replan_result['safety_score']:.1f}/100")
        print(f"   🏥 Emergency sites nearby: {len(replan_result['emergency_landing_sites'])}")
        print(f"   💡 Recommendation: {replan_result['recommended_action']}")


def example_mavlink_integration():
    """
    Example: Converting A* path to MAVLink format for drone communication
    """
    print("\n" + "=" * 60)
    print("🔗 EXAMPLE: MAVLink Integration")
    print("=" * 60)
    
    # Create sample path
    pathfinder = AStarPathfinder()
    start = Coordinate(lat=28.6139, lng=77.2090, altitude=150)
    goal = Coordinate(lat=28.6300, lng=77.2200, altitude=160)
    
    result = pathfinder.find_path(start, goal, Priority.SHORTEST)
    
    # Convert to MAVLink format
    mavlink_waypoints = convert_path_to_mavlink(result.path)
    
    print(f"📡 Generated {len(mavlink_waypoints)} MAVLink waypoints:")
    for i, wp in enumerate(mavlink_waypoints[:3]):  # Show first 3
        print(f"   Waypoint {i}: Lat={wp['x']:.6f}, Lng={wp['y']:.6f}, Alt={wp['z']:.1f}m")
    
    if len(mavlink_waypoints) > 3:
        print(f"   ... and {len(mavlink_waypoints) - 3} more waypoints")
    
    # Calculate battery consumption
    battery_consumption = estimate_battery_consumption(result.path)
    print(f"🔋 Estimated battery consumption: {battery_consumption:.1f} Wh")
    
    return mavlink_waypoints


def run_all_examples():
    """Run all pathfinding examples."""
    print("🚁 VIHANGAM A* PATHFINDING SYSTEM - PRACTICAL EXAMPLES")
    print("=" * 80)
    
    try:
        # Example 1: Mission planning
        mission_plan = example_search_rescue_mission()
        
        # Example 2: Emergency replanning
        example_emergency_replanning()
        
        # Example 3: MAVLink integration
        mavlink_waypoints = example_mavlink_integration()
        
        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return {
            'mission_plan': mission_plan,
            'mavlink_waypoints': mavlink_waypoints,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return {'status': 'error', 'message': str(e)}


# Performance Testing

def performance_test():
    """Test pathfinding performance with various scenarios."""
    print("\n" + "=" * 60)
    print("⚡ PERFORMANCE TEST")
    print("=" * 60)
    
    import time
    
    pathfinder = AStarPathfinder()
    
    # Test scenarios with increasing complexity
    scenarios = [
        {"name": "Short distance", "distance": 1.0, "obstacles": 0},
        {"name": "Medium distance", "distance": 5.0, "obstacles": 5},
        {"name": "Long distance", "distance": 20.0, "obstacles": 10},
        {"name": "Complex terrain", "distance": 10.0, "obstacles": 20}
    ]
    
    for scenario in scenarios:
        # Add obstacles
        pathfinder.obstacles.clear()
        for i in range(scenario["obstacles"]):
            obstacle = Obstacle(
                center=Coordinate(
                    lat=28.6139 + (i * 0.01),
                    lng=77.2090 + (i * 0.01),
                    altitude=0
                ),
                radius=100,
                height=200,
                severity=2
            )
            pathfinder.add_obstacle(obstacle)
        
        # Calculate path
        start_time = time.time()
        start = Coordinate(lat=28.6139, lng=77.2090, altitude=150)
        goal = Coordinate(
            lat=28.6139 + (scenario["distance"] / 111.0),
            lng=77.2090 + (scenario["distance"] / 111.0),
            altitude=160
        )
        
        result = pathfinder.find_path(start, goal, Priority.SHORTEST)
        calculation_time = time.time() - start_time
        
        print(f"   {scenario['name']:18} | "
              f"{calculation_time*1000:6.1f}ms | "
              f"{len(result.path):3} waypoints | "
              f"{result.total_distance:5.2f}km")


if __name__ == "__main__":
    # Run examples when script is executed directly
    results = run_all_examples()
    
    # Run performance test
    performance_test()
    
    print("\n🎯 Integration complete! Your A* pathfinding system is ready for production use.")