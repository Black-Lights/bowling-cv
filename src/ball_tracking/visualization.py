"""
Visualization functions for ball tracking

Functions for drawing tracking overlays and creating plots.

Version: 2.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: January 30, 2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple


def draw_tracking(frame: np.ndarray,
                  track_data: Optional[Dict],
                  trajectory: List[Dict],
                  config,
                  release_point: Optional[Dict] = None,
                  impact_point: Optional[Dict] = None) -> np.ndarray:
    """
    Draw tracking visualization on frame.
    
    Args:
        frame: Input frame (BGR)
        track_data: Current tracking data
        trajectory: Full trajectory history
        config: Configuration module
        release_point: Release point data
        impact_point: Impact point data
        
    Returns:
        Frame with tracking overlay
    """
    vis_frame = frame.copy()
    
    if track_data is None:
        return vis_frame
    
    center = track_data['center']
    radius = track_data.get('radius', 0)
    confidence = track_data.get('confidence', 1.0)
    detected = track_data.get('detected', True)
    
    # Choose color based on detection status
    if detected:
        color = config.COLOR_DETECTED  # Green for detected
    else:
        color = config.COLOR_PREDICTED  # Yellow for predicted
    
    # Draw ball circle
    if config.DRAW_BALL_CIRCLE and radius > 0:
        cv2.circle(vis_frame, center, radius, color, 2)
    
    # Draw ball center
    if config.DRAW_BALL_CENTER:
        cv2.circle(vis_frame, center, 5, color, -1)
    
    # Draw trajectory trail
    if config.DRAW_TRAJECTORY and len(trajectory) > 1:
        trail_length = min(config.TRAJECTORY_LENGTH, len(trajectory))
        trail = trajectory[-trail_length:]
        
        for i in range(1, len(trail)):
            pt1 = trail[i-1]['center']
            pt2 = trail[i]['center']
            
            # Fade color based on age
            alpha = i / len(trail)
            trail_color = tuple(int(c * alpha) for c in config.COLOR_TRAJECTORY)
            
            cv2.line(vis_frame, pt1, pt2, trail_color, 2)
    
    # Draw velocity vector
    if config.DRAW_VELOCITY_VECTOR:
        velocity = track_data.get('velocity', (0, 0))
        if velocity != (0, 0):
            vel_scale = config.VELOCITY_VECTOR_SCALE
            end_point = (
                int(center[0] + velocity[0] * vel_scale),
                int(center[1] + velocity[1] * vel_scale)
            )
            cv2.arrowedLine(vis_frame, center, end_point, (255, 0, 255), 2, tipLength=0.3)
    
    # Draw status text
    status_text = "DETECTED" if detected else "PREDICTED"
    conf_text = f"Conf: {confidence:.2f}"
    
    cv2.putText(vis_frame, status_text, (center[0] + 15, center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(vis_frame, conf_text, (center[0] + 15, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw release point if available
    if release_point and release_point['frame'] <= track_data['frame']:
        release_pos = release_point['position']
        cv2.circle(vis_frame, release_pos, 8, config.COLOR_RELEASE, 3)
        cv2.putText(vis_frame, "RELEASE", (release_pos[0] - 30, release_pos[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_RELEASE, 2)
    
    # Draw impact point if available
    if impact_point and impact_point['frame'] <= track_data['frame']:
        impact_pos = impact_point['position']
        cv2.circle(vis_frame, impact_pos, 8, config.COLOR_IMPACT, 3)
        cv2.putText(vis_frame, "IMPACT", (impact_pos[0] - 30, impact_pos[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_IMPACT, 2)
    
    return vis_frame


def plot_trajectory(trajectory: List[Dict],
                    boundary_data: Dict,
                    output_path: str,
                    release_point: Optional[Dict] = None,
                    impact_point: Optional[Dict] = None):
    """
    Create trajectory plot with lane boundaries.
    
    Args:
        trajectory: List of trajectory points
        boundary_data: Lane boundary data from Phase 1
        output_path: Path to save plot
        release_point: Release point data
        impact_point: Impact point data
    """
    if not trajectory:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Extract data
    frames = [p['frame'] for p in trajectory]
    x_coords = [p['center'][0] for p in trajectory]
    y_coords = [p['center'][1] for p in trajectory]
    detected = [p.get('detected', True) for p in trajectory]
    
    # Plot 1: 2D trajectory view
    ax1.set_title('Ball Trajectory (Top-Down View)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.invert_yaxis()  # Invert Y to match image coordinates
    ax1.grid(True, alpha=0.3)
    
    # Draw lane boundaries
    left = boundary_data['master_left']
    right = boundary_data['master_right']
    top_y = int(boundary_data['top_boundary']['y_position'])
    bottom_y = boundary_data['median_foul_params']['center_y']
    
    ax1.plot([left['x_top'], left['x_bottom']], [top_y, bottom_y],
             'b--', linewidth=2, label='Left Boundary', alpha=0.7)
    ax1.plot([right['x_top'], right['x_bottom']], [top_y, bottom_y],
             'b--', linewidth=2, label='Right Boundary', alpha=0.7)
    ax1.axhline(y=top_y, color='g', linestyle='--', linewidth=2, label='Top Boundary', alpha=0.7)
    ax1.axhline(y=bottom_y, color='r', linestyle='--', linewidth=2, label='Foul Line', alpha=0.7)
    
    # Plot trajectory
    detected_x = [x for x, d in zip(x_coords, detected) if d]
    detected_y = [y for y, d in zip(y_coords, detected) if d]
    predicted_x = [x for x, d in zip(x_coords, detected) if not d]
    predicted_y = [y for y, d in zip(y_coords, detected) if not d]
    
    if detected_x:
        ax1.scatter(detected_x, detected_y, c='green', s=20, alpha=0.6, label='Detected', zorder=3)
    if predicted_x:
        ax1.scatter(predicted_x, predicted_y, c='yellow', s=20, alpha=0.6, label='Predicted', zorder=3)
    
    # Draw trajectory line
    ax1.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.3, zorder=2)
    
    # Mark start and end
    if trajectory:
        ax1.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start', zorder=4)
        ax1.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End', zorder=4)
    
    # Mark release point
    if release_point:
        rx, ry = release_point['position']
        ax1.plot(rx, ry, 'r^', markersize=12, label='Release', zorder=5)
    
    # Mark impact point
    if impact_point:
        ix, iy = impact_point['position']
        ax1.plot(ix, iy, 'bs', markersize=12, label='Impact', zorder=5)
    
    ax1.legend(loc='best', fontsize=9)
    
    # Plot 2: X, Y coordinates over time
    ax2.set_title('Ball Position Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Position (pixels)')
    ax2.grid(True, alpha=0.3)
    
    ax2.plot(frames, x_coords, 'b-', linewidth=2, label='X Position', alpha=0.7)
    ax2.plot(frames, y_coords, 'r-', linewidth=2, label='Y Position', alpha=0.7)
    
    # Mark release and impact on time plot
    if release_point:
        ax2.axvline(x=release_point['frame'], color='red', linestyle='--', alpha=0.5, label='Release')
    if impact_point:
        ax2.axvline(x=impact_point['frame'], color='blue', linestyle='--', alpha=0.5, label='Impact')
    
    ax2.legend(loc='best')
    
    # Add statistics box
    stats_text = f"Total Points: {len(trajectory)}\n"
    stats_text += f"Detected: {sum(detected)}\n"
    stats_text += f"Predicted: {len(detected) - sum(detected)}\n"
    
    if trajectory:
        velocities = [p.get('velocity', (0, 0)) for p in trajectory]
        speeds = [np.sqrt(v[0]**2 + v[1]**2) for v in velocities]
        max_speed = max(speeds) if speeds else 0
        avg_speed = np.mean(speeds) if speeds else 0
        stats_text += f"Max Speed: {max_speed:.2f} px/frame\n"
        stats_text += f"Avg Speed: {avg_speed:.2f} px/frame"
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_velocity(trajectory: List[Dict], output_path: str):
    """
    Create velocity over time plot.
    
    Args:
        trajectory: List of trajectory points
        output_path: Path to save plot
    """
    if not trajectory:
        return
    
    # Extract velocity data
    frames = [p['frame'] for p in trajectory]
    velocities = [p.get('velocity', (0, 0)) for p in trajectory]
    vx = [v[0] for v in velocities]
    vy = [v[1] for v in velocities]
    speeds = [np.sqrt(v[0]**2 + v[1]**2) for v in velocities]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Velocity components
    ax1.set_title('Velocity Components Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Velocity (pixels/frame)')
    ax1.grid(True, alpha=0.3)
    
    ax1.plot(frames, vx, 'b-', linewidth=1.5, label='Vx (horizontal)', alpha=0.7)
    ax1.plot(frames, vy, 'r-', linewidth=1.5, label='Vy (vertical)', alpha=0.7)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.legend(loc='best')
    
    # Plot 2: Speed (magnitude)
    ax2.set_title('Ball Speed Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Speed (pixels/frame)')
    ax2.grid(True, alpha=0.3)
    
    ax2.plot(frames, speeds, 'g-', linewidth=2, label='Speed', alpha=0.7)
    ax2.fill_between(frames, speeds, alpha=0.3, color='green')
    
    # Add statistics
    max_speed = max(speeds) if speeds else 0
    avg_speed = np.mean(speeds) if speeds else 0
    ax2.axhline(y=avg_speed, color='orange', linestyle='--', linewidth=1.5, 
                label=f'Avg: {avg_speed:.2f}', alpha=0.7)
    ax2.axhline(y=max_speed, color='red', linestyle='--', linewidth=1.5,
                label=f'Max: {max_speed:.2f}', alpha=0.7)
    
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
