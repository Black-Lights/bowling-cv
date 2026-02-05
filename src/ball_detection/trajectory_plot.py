"""
Trajectory Plotting Module
Plots ball trajectory on both original and transformed (overhead) views
Saves trajectory points for post-processing
"""

import cv2
import numpy as np
import json
from pathlib import Path


def save_trajectory_points(tracker, output_dir, video_name, config):
    """
    Save trajectory points (original and transformed) to JSON for post-processing
    
    Args:
        tracker: BallTracker instance with trajectory data
        output_dir: Output directory path
        video_name: Base name of the video
        config: Configuration object
    """
    output_dir = Path(output_dir)
    
    # Prepare original trajectory points
    original_points = []
    for i, (cx, cy, frame_idx, radius) in enumerate(tracker.trajectory):
        is_interpolated = False
        if tracker.interpolated_points:
            # Check if this point is interpolated
            for interp_point in tracker.interpolated_points:
                if abs(interp_point[0] - cx) < 0.1 and abs(interp_point[1] - cy) < 0.1:
                    is_interpolated = True
                    break
        
        original_points.append({
            'index': i,
            'frame_number': int(frame_idx),
            'x': float(cx),
            'y': float(cy),
            'radius': float(radius),
            'interpolated': is_interpolated
        })
    
    # Prepare transformed trajectory points (apply homography)
    transformed_points = []
    
    # Load homography data
    homography_path = output_dir / 'homography_data.json'
    if not homography_path.exists():
        homography_path = output_dir.parent / 'homography_data.json'
    
    if homography_path.exists():
        with open(homography_path, 'r') as f:
            homography_data = json.load(f)
        
        H = np.array(homography_data['homography_matrix'], dtype=np.float32)
        
        # Transform all trajectory points
        trajectory_points = np.array([[cx, cy] for cx, cy, _, _ in tracker.trajectory], dtype=np.float32)
        if len(trajectory_points) > 0:
            transformed = cv2.perspectiveTransform(trajectory_points.reshape(-1, 1, 2), H)
            transformed = transformed.reshape(-1, 2)
            
            for i, (cx, cy, frame_idx, radius) in enumerate(tracker.trajectory):
                is_interpolated = False
                
                if tracker.interpolated_points:
                    for interp_point in tracker.interpolated_points:
                        if abs(interp_point[0] - cx) < 0.1 and abs(interp_point[1] - cy) < 0.1:
                            is_interpolated = True
                            break
                
                transformed_points.append({
                    'index': i,
                    'frame_number': int(frame_idx),
                    'x': float(transformed[i][0]),
                    'y': float(transformed[i][1]),
                    'radius': float(radius),
                    'interpolated': is_interpolated
                })
    
    # Prepare complete data structure
    trajectory_data = {
        'video_name': video_name,
        'total_frames_processed': tracker.stop_frame if tracker.trajectory_complete else None,
        'trajectory_complete': tracker.trajectory_complete,
        'stop_info': {
            'stopped_at_frame': tracker.stop_frame if tracker.trajectory_complete else None,
            'stop_threshold_y': float(tracker.stop_threshold_y) if tracker.stop_threshold_y is not None else None,
            'top_boundary_y': float(tracker.top_boundary_y) if tracker.top_boundary_y is not None else None
        },
        'trajectory_points': {
            'original': original_points,
            'transformed': transformed_points if transformed_points else None
        },
        'interpolated_endpoints': {
            'original': [{'x': float(x), 'y': float(y), 'radius': float(r)} for x, y, r in tracker.interpolated_points] if tracker.interpolated_points else [],
            'transformed': []
        },
        'statistics': {
            'total_points': len(tracker.trajectory),
            'real_points': len([p for p in original_points if not p['interpolated']]),
            'interpolated_points': len([p for p in original_points if p['interpolated']]),
            'extrapolated_endpoints': len(tracker.interpolated_points) if tracker.interpolated_points else 0
        }
    }
    
    # Transform interpolated points if homography is available
    if homography_path.exists() and tracker.interpolated_points:
        with open(homography_path, 'r') as f:
            homography_data = json.load(f)
        H = np.array(homography_data['homography_matrix'], dtype=np.float32)
        
        # Extract x, y from (x, y, radius) tuples
        interp_points = np.array([[x, y] for x, y, _ in tracker.interpolated_points], dtype=np.float32)
        transformed_interp = cv2.perspectiveTransform(interp_points.reshape(-1, 1, 2), H)
        transformed_interp = transformed_interp.reshape(-1, 2)
        
        trajectory_data['interpolated_endpoints']['transformed'] = [
            {'x': float(x), 'y': float(y), 'radius': float(tracker.interpolated_points[i][2])} 
            for i, (x, y) in enumerate(transformed_interp)
        ]
    
    # Save to JSON
    output_path = output_dir / f'{video_name}_trajectory_data.json'
    with open(output_path, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    print(f">>> Saved trajectory data: {output_path.name}")
    print(f"    Total points: {trajectory_data['statistics']['total_points']}")
    print(f"    Extrapolated endpoints: {trajectory_data['statistics']['extrapolated_endpoints']}")
    if transformed_points:
        print(f"    Includes transformed (overhead) coordinates")
    
    return output_path


def plot_trajectory_on_overhead(tracker, output_dir, video_name, config):
    """
    Plot ball trajectory on the transformed overhead view
    Applies homography transformation to trajectory points
    
    Args:
        tracker: BallTracker instance with trajectory data
        output_dir: Output directory path
        video_name: Base name of the video
        config: Configuration object
    """
    output_dir = Path(output_dir)
    
    # Load homography data (try both possible locations)
    homography_path = output_dir / 'homography_data.json'
    if not homography_path.exists():
        homography_path = output_dir.parent / 'homography_data.json'
    
    if not homography_path.exists():
        print(f"  Warning: Homography data not found")
        print(f"  Skipping overhead trajectory plot")
        return
    
    with open(homography_path, 'r') as f:
        homography_data = json.load(f)
    
    H = np.array(homography_data['homography_matrix'], dtype=np.float32)
    
    # Get dimensions (different structure for homography vs transform data)
    if 'transformed_width' in homography_data:
        transformed_width = homography_data['transformed_width']
        transformed_height = homography_data['transformed_height']
    else:
        # Calculate from lane dimensions and pixels_per_inch
        # Default: 20 pixels per inch
        pixels_per_inch = 20
        lane_dims = homography_data.get('lane_dimensions', {})
        transformed_width = int(lane_dims.get('length_inches', 720) * pixels_per_inch)
        transformed_height = int(lane_dims.get('width_inches', 41.5) * pixels_per_inch)
    
    # Load a sample transformed frame (background)
    transformed_video_path = output_dir.parent / f'cropped_{video_name}_transformed.mp4'
    if not transformed_video_path.exists():
        print(f"  Warning: Transformed video not found at {transformed_video_path}")
        print(f"  Creating blank background")
        background = np.zeros((transformed_height, transformed_width, 3), dtype=np.uint8)
        background.fill(40)  # Dark gray
    else:
        cap = cv2.VideoCapture(str(transformed_video_path))
        ret, background = cap.read()
        cap.release()
        if not ret:
            background = np.zeros((transformed_height, transformed_width, 3), dtype=np.uint8)
            background.fill(40)
    
    # Create visualization canvas
    vis = background.copy()
    
    # Transform trajectory points
    if len(tracker.trajectory) == 0:
        print(f"  Warning: No trajectory points to plot")
        return
    
    # Convert trajectory to numpy array (extract x, y only from 4-tuple)
    trajectory_array = np.array([[x, y] for x, y, _, _ in tracker.trajectory], dtype=np.float32)
    
    # Apply homography transformation to all points
    # cv2.perspectiveTransform expects shape (N, 1, 2)
    trajectory_reshaped = trajectory_array.reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(trajectory_reshaped, H)
    transformed_trajectory = transformed_points.reshape(-1, 2)
    
    # Draw transformed trajectory (solid blue line)
    for i in range(1, len(transformed_trajectory)):
        pt1 = tuple(transformed_trajectory[i-1].astype(int))
        pt2 = tuple(transformed_trajectory[i].astype(int))
        cv2.line(vis, pt1, pt2, (255, 165, 0), 3)  # Orange line
    
    # Mark start point (green)
    start_pt = tuple(transformed_trajectory[0].astype(int))
    cv2.circle(vis, start_pt, 10, (0, 255, 0), -1)
    cv2.putText(vis, "START", (start_pt[0] + 15, start_pt[1]),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Mark end point (yellow)
    end_pt = tuple(transformed_trajectory[-1].astype(int))
    cv2.circle(vis, end_pt, 10, (0, 255, 255), -1)
    cv2.putText(vis, "END (Real)", (end_pt[0] + 15, end_pt[1]),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Transform and draw interpolated points (if any)
    if tracker.interpolated_points:
        # Extract only x,y from 3-tuples (x, y, frame_idx)
        interp_xy = [[pt[0], pt[1]] for pt in tracker.interpolated_points]
        interp_array = np.array(interp_xy, dtype=np.float32).reshape(-1, 1, 2)
        transformed_interp = cv2.perspectiveTransform(interp_array, H).reshape(-1, 2)
        
        # Dashed line from last real to interpolated
        last_real = transformed_trajectory[-1]
        for interp_pt in transformed_interp:
            # Draw dashes
            num_dashes = 15
            for j in range(num_dashes):
                t1 = j / num_dashes
                t2 = (j + 0.5) / num_dashes
                x1 = int(last_real[0] + t1 * (interp_pt[0] - last_real[0]))
                y1 = int(last_real[1] + t1 * (interp_pt[1] - last_real[1]))
                x2 = int(last_real[0] + t2 * (interp_pt[0] - last_real[0]))
                y2 = int(last_real[1] + t2 * (interp_pt[1] - last_real[1]))
                cv2.line(vis, (x1, y1), (x2, y2), config.INTERPOLATION_COLOR, 3)
            
            # Mark interpolated endpoint (orange)
            interp_pt_int = tuple(interp_pt.astype(int))
            cv2.circle(vis, interp_pt_int, 8, config.INTERPOLATION_COLOR, -1)
            cv2.putText(vis, "EXTRAPOLATED", (interp_pt_int[0] + 15, interp_pt_int[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.INTERPOLATION_COLOR, 2)
    
    # Add info panel
    info_y = 30
    cv2.putText(vis, "OVERHEAD VIEW (Transformed)", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    info_y += 40
    cv2.putText(vis, f"Trajectory Points: {len(tracker.trajectory)}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    info_y += 30
    if tracker.interpolated_points:
        cv2.putText(vis, f"Interpolated Points: {len(tracker.interpolated_points)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.INTERPOLATION_COLOR, 2)
    
    # Add scale reference (if available)
    if 'pixels_per_inch' in homography_data:
        ppi = homography_data['pixels_per_inch']
        info_y += 30
        cv2.putText(vis, f"Scale: {ppi} pixels/inch", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Save plot
    output_path = output_dir / f'{video_name}_overhead_trajectory.png'
    cv2.imwrite(str(output_path), vis)
    print(f">>> Saved overhead trajectory plot: {output_path.name}")
    
    return str(output_path)


def plot_trajectory_on_original(tracker, frames, tracking_results, output_dir, video_name, config):
    """
    Plot ball trajectory on original perspective view
    
    Args:
        tracker: BallTracker instance with trajectory data
        frames: List of original frames
        tracking_results: List of tracking result dicts
        output_dir: Output directory path
        video_name: Base name of the video
        config: Configuration object
    """
    if len(frames) == 0:
        return
    
    output_dir = Path(output_dir)
    
    # Use middle frame as background
    mid_frame = frames[len(frames) // 2].copy()
    vis = mid_frame.copy()
    
    # Draw trajectory (solid orange line)
    if len(tracker.trajectory) > 1:
        for i in range(1, len(tracker.trajectory)):
            pt1 = (int(tracker.trajectory[i-1][0]), int(tracker.trajectory[i-1][1]))
            pt2 = (int(tracker.trajectory[i][0]), int(tracker.trajectory[i][1]))
            cv2.line(vis, pt1, pt2, (0, 165, 255), 3)  # Orange
    
    # Mark start point (green)
    if len(tracker.trajectory) > 0:
        start_pt = (int(tracker.trajectory[0][0]), int(tracker.trajectory[0][1]))
        cv2.circle(vis, start_pt, 10, (0, 255, 0), -1)
        cv2.putText(vis, "START", (start_pt[0] + 15, start_pt[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Mark end point (yellow)
        end_pt = (int(tracker.trajectory[-1][0]), int(tracker.trajectory[-1][1]))
        cv2.circle(vis, end_pt, 10, (0, 255, 255), -1)
        cv2.putText(vis, "END", (end_pt[0] + 15, end_pt[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Draw interpolated points (dashed line)
    if tracker.interpolated_points:
        last_real = (tracker.trajectory[-1][0], tracker.trajectory[-1][1])
        for interp_pt in tracker.interpolated_points:
            # Draw dashes
            num_dashes = 15
            for j in range(num_dashes):
                t1 = j / num_dashes
                t2 = (j + 0.5) / num_dashes
                x1 = int(last_real[0] + t1 * (interp_pt[0] - last_real[0]))
                y1 = int(last_real[1] + t1 * (interp_pt[1] - last_real[1]))
                x2 = int(last_real[0] + t2 * (interp_pt[0] - last_real[0]))
                y2 = int(last_real[1] + t2 * (interp_pt[1] - last_real[1]))
                cv2.line(vis, (x1, y1), (x2, y2), config.INTERPOLATION_COLOR, 3)
            
            # Mark interpolated endpoint
            interp_pt_int = (int(interp_pt[0]), int(interp_pt[1]))
            cv2.circle(vis, interp_pt_int, 8, config.INTERPOLATION_COLOR, -1)
            cv2.putText(vis, "EXTRAPOLATED", (interp_pt_int[0] + 15, interp_pt_int[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.INTERPOLATION_COLOR, 2)
    
    # Draw stop threshold line (if available)
    if tracker.stop_threshold_y is not None and config.SHOW_STOP_THRESHOLD_LINE:
        height = vis.shape[0]
        width = vis.shape[1]
        stop_y = int(tracker.stop_threshold_y)  # Convert to int for OpenCV
        cv2.line(vis, (0, stop_y), (width, stop_y),
                config.STOP_THRESHOLD_COLOR, 2)
        cv2.putText(vis, f"Stop Threshold Y={stop_y}", (width - 300, stop_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.STOP_THRESHOLD_COLOR, 2)
    
    # Add info panel
    info_y = 30
    cv2.putText(vis, "ORIGINAL VIEW (Perspective)", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    info_y += 40
    cv2.putText(vis, f"Trajectory Points: {len(tracker.trajectory)}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    info_y += 30
    if tracker.interpolated_points:
        cv2.putText(vis, f"Interpolated Points: {len(tracker.interpolated_points)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.INTERPOLATION_COLOR, 2)
    
    # Save plot
    output_path = output_dir / f'{video_name}_original_trajectory.png'
    cv2.imwrite(str(output_path), vis)
    print(f">>> Saved original trajectory plot: {output_path.name}")
    
    return str(output_path)
