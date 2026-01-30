"""
Tracking analysis functions for ball trajectory analysis

Functions for analyzing trajectory, detecting release/impact points, and generating reports.

Version: 2.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: January 30, 2026
"""

import numpy as np
import json
from typing import List, Dict, Optional, Tuple


def smooth_trajectory(trajectory: List[Dict], window_size: int = 5) -> List[Dict]:
    """
    Apply moving average smoothing to trajectory.
    
    Args:
        trajectory: List of trajectory points
        window_size: Window size for moving average
        
    Returns:
        Smoothed trajectory
    """
    if len(trajectory) < window_size:
        return trajectory
    
    smoothed = []
    half_window = window_size // 2
    
    for i, point in enumerate(trajectory):
        if i < half_window or i >= len(trajectory) - half_window:
            # Keep edge points as-is
            smoothed.append(point)
        else:
            # Apply moving average
            start = i - half_window
            end = i + half_window + 1
            window_points = trajectory[start:end]
            
            avg_x = np.mean([p['center'][0] for p in window_points])
            avg_y = np.mean([p['center'][1] for p in window_points])
            
            smoothed_point = point.copy()
            smoothed_point['center'] = (int(avg_x), int(avg_y))
            smoothed.append(smoothed_point)
    
    return smoothed


def detect_release_point(trajectory: List[Dict],
                          velocity_threshold: float = 3.0,
                          zone_y: float = 0.70,
                          boundary_data: Optional[Dict] = None) -> Optional[Dict]:
    """
    Detect ball release point from trajectory.
    
    Release point is first point where velocity exceeds threshold
    and ball is in release zone (bottom portion of lane).
    
    Args:
        trajectory: List of trajectory points
        velocity_threshold: Minimum speed to detect release (px/frame)
        zone_y: Release zone as fraction from top (0.7 = bottom 30% of lane)
        boundary_data: Boundary data to determine zone
        
    Returns:
        Release point data or None
    """
    if len(trajectory) < 5:
        return None
    
    # Determine release zone Y coordinate
    if boundary_data:
        top_y = int(boundary_data['top_boundary']['y_position'])
        bottom_y = boundary_data['median_foul_params']['center_y']
        zone_threshold = top_y + (bottom_y - top_y) * zone_y
    else:
        zone_threshold = None
    
    # Find first point exceeding velocity threshold in release zone
    for i, point in enumerate(trajectory):
        # Check if in release zone
        if zone_threshold and point['center'][1] < zone_threshold:
            continue
        
        vel = point.get('velocity', (0, 0))
        speed = np.sqrt(vel[0]**2 + vel[1]**2)
        
        if speed > velocity_threshold:
            return {
                'frame': point['frame'],
                'position': point['center'],
                'velocity': vel,
                'speed': float(speed),
                'index': i
            }
    
    return None


def detect_impact_point(trajectory: List[Dict],
                         velocity_threshold: float = 2.0,
                         zone_y: float = 0.20,
                         boundary_data: Optional[Dict] = None) -> Optional[Dict]:
    """
    Detect ball impact point at pins from trajectory.
    
    Impact point is in top portion of lane where ball is still moving.
    
    Args:
        trajectory: List of trajectory points
        velocity_threshold: Minimum speed at impact (px/frame)
        zone_y: Impact zone as fraction from top (0.2 = top 20% of lane)
        boundary_data: Boundary data to determine zone
        
    Returns:
        Impact point data or None
    """
    if len(trajectory) < 5:
        return None
    
    # Determine impact zone Y coordinate
    if boundary_data:
        top_y = int(boundary_data['top_boundary']['y_position'])
        bottom_y = boundary_data['median_foul_params']['center_y']
        zone_threshold = top_y + (bottom_y - top_y) * zone_y
    else:
        zone_threshold = None
    
    # Find first point in impact zone
    for i, point in enumerate(trajectory):
        # Check if in impact zone
        if zone_threshold and point['center'][1] > zone_threshold:
            continue
        
        vel = point.get('velocity', (0, 0))
        speed = np.sqrt(vel[0]**2 + vel[1]**2)
        
        if speed > velocity_threshold:
            return {
                'frame': point['frame'],
                'position': point['center'],
                'velocity': vel,
                'speed': float(speed),
                'index': i
            }
    
    return None


def save_trajectory_json(trajectory: List[Dict],
                          release_point: Optional[Dict],
                          impact_point: Optional[Dict],
                          statistics: Dict,
                          output_path: str):
    """
    Save trajectory data to JSON file.
    
    Args:
        trajectory: List of trajectory points
        release_point: Release point data
        impact_point: Impact point data
        statistics: Tracking statistics
        output_path: Path to save JSON file
    """
    data = {
        'version': '2.0.0',
        'trajectory': [
            {
                'frame': p['frame'],
                'center': list(p['center']),
                'radius': p.get('radius', 0),
                'velocity': list(p.get('velocity', (0, 0))),
                'confidence': p.get('confidence', 1.0),
                'detected': p.get('detected', True)
            }
            for p in trajectory
        ],
        'release_point': release_point if release_point else None,
        'impact_point': impact_point if impact_point else None,
        'statistics': statistics
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_analysis_report(trajectory: List[Dict],
                          release_point: Optional[Dict],
                          impact_point: Optional[Dict],
                          statistics: Dict,
                          output_path: str):
    """
    Save text analysis report.
    
    Args:
        trajectory: List of trajectory points
        release_point: Release point data
        impact_point: Impact point data
        statistics: Tracking statistics
        output_path: Path to save report file
    """
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BALL TRACKING ANALYSIS REPORT\n")
        f.write("Phase 2: Ball Detection and Tracking\n")
        f.write("="*70 + "\n\n")
        
        # Tracking statistics
        f.write("TRACKING STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total trajectory points: {statistics.get('total_points', 0)}\n")
        f.write(f"Detected points: {statistics.get('detected_points', 0)}\n")
        f.write(f"Predicted points: {statistics.get('predicted_points', 0)}\n")
        f.write(f"Detection rate: {100*statistics.get('detection_rate', 0):.1f}%\n")
        f.write(f"Average confidence: {statistics.get('avg_confidence', 0):.3f}\n\n")
        
        # Velocity statistics
        f.write("VELOCITY STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Maximum speed: {statistics.get('max_speed', 0):.2f} px/frame\n")
        f.write(f"Average speed: {statistics.get('avg_speed', 0):.2f} px/frame\n\n")
        
        # Release point
        if release_point:
            f.write("RELEASE POINT\n")
            f.write("-" * 70 + "\n")
            f.write(f"Frame number: {release_point['frame']}\n")
            f.write(f"Position (x, y): {release_point['position']}\n")
            f.write(f"Velocity (vx, vy): ({release_point['velocity'][0]:.2f}, "
                   f"{release_point['velocity'][1]:.2f}) px/frame\n")
            f.write(f"Speed: {release_point['speed']:.2f} px/frame\n\n")
        else:
            f.write("RELEASE POINT\n")
            f.write("-" * 70 + "\n")
            f.write("Not detected\n\n")
        
        # Impact point
        if impact_point:
            f.write("IMPACT POINT\n")
            f.write("-" * 70 + "\n")
            f.write(f"Frame number: {impact_point['frame']}\n")
            f.write(f"Position (x, y): {impact_point['position']}\n")
            f.write(f"Velocity (vx, vy): ({impact_point['velocity'][0]:.2f}, "
                   f"{impact_point['velocity'][1]:.2f}) px/frame\n")
            f.write(f"Speed: {impact_point['speed']:.2f} px/frame\n\n")
        else:
            f.write("IMPACT POINT\n")
            f.write("-" * 70 + "\n")
            f.write("Not detected\n\n")
        
        # Trajectory summary
        f.write("TRAJECTORY SUMMARY\n")
        f.write("-" * 70 + "\n")
        if trajectory:
            start = trajectory[0]
            end = trajectory[-1]
            f.write(f"Start frame: {start['frame']}\n")
            f.write(f"Start position: {start['center']}\n")
            f.write(f"End frame: {end['frame']}\n")
            f.write(f"End position: {end['center']}\n")
            
            dx = end['center'][0] - start['center'][0]
            dy = end['center'][1] - start['center'][1]
            distance = np.sqrt(dx**2 + dy**2)
            f.write(f"Total distance traveled: {distance:.2f} pixels\n")
            f.write(f"Duration: {end['frame'] - start['frame']} frames\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("End of Report\n")
        f.write("="*70 + "\n")
