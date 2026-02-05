"""
Integrated Tracking Visualization - Stage C+D+E Combined
Generates diagnostic videos showing the complete tracking pipeline

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Last Updated: February 2, 2026
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def generate_integrated_tracking_videos(video_name, frames, masks, tracking_results, config, output_dir):
    """
    Generate visualization videos for integrated tracking pipeline
    
    Args:
        video_name: Name of the video being processed
        frames: List of BGR frames
        masks: List of binary masks
        tracking_results: List of tracking result dicts from BallTracker.process_frame()
        config: Configuration module
        output_dir: Output directory for videos
    
    Returns:
        dict: Mapping of video names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(frames) == 0:
        return {}
    
    height, width = frames[0].shape[:2]
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    result_paths = {}
    
    print("\nGenerating integrated tracking visualization videos...")
    
    # Video 1: All Validated Candidates + Selected
    if config.SAVE_BLOB_FINAL_FILTERED_VIDEO:
        output_path = output_dir / f"{video_name}_integrated_candidates.mp4"
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame, result in tqdm(zip(frames, tracking_results), 
                                  total=len(frames), 
                                  desc="Candidates View"):
            vis = frame.copy()
            
            # Filter candidates based on mode
            if result['mode'] == 'local' and result['roi_box']:
                # In local mode, only show candidates within ROI
                x1, y1, x2, y2 = result['roi_box']
                candidates_to_show = [
                    c for c in result['all_candidates']
                    if (x1 <= c['center'][0] <= x2 and y1 <= c['center'][1] <= y2)
                ]
                
                # Draw ROI box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate and display ROI size info
                roi_size = result['roi_size']
                if result['detection']:
                    y_ball = result['detection']['center'][1]
                    dynamic_buffer = int(config.K_SCALE * y_ball)
                    cv2.putText(vis, f"ROI: {roi_size}px (B_min={config.B_MIN}, k*y={dynamic_buffer})", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # In global mode, show all candidates
                candidates_to_show = result['all_candidates']
            
            # Draw validated candidates (cyan)
            for candidate in candidates_to_show:
                cx, cy = candidate['center']
                radius = candidate['radius']
                cv2.circle(vis, (cx, cy), radius, (255, 255, 0), 2)  # Cyan
                cv2.circle(vis, (cx, cy), 2, (255, 255, 0), -1)
            
            # Draw selected candidate (yellow, larger)
            if result['detection']:
                cx, cy = result['detection']['center']
                radius = result['detection']['radius']
                cv2.circle(vis, (cx, cy), radius + 5, (0, 255, 255), 3)  # Yellow
                cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)
            
            # Add info overlay
            if result['mode'] == 'local' and result['roi_box']:
                info_text = [
                    f"Mode: {result['mode'].upper()}",
                    f"Total Candidates: {result['candidates_count']}",
                    f"In ROI: {len(candidates_to_show)}",
                    f"Detected: {'YES' if result['detection'] else 'NO'}"
                ]
            else:
                info_text = [
                    f"Mode: {result['mode'].upper()}",
                    f"Candidates: {result['candidates_count']}",
                    f"Detected: {'YES' if result['detection'] else 'NO'}"
                ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(vis, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            writer.write(vis)
        
        writer.release()
        result_paths['candidates'] = str(output_path)
        print(f">>> Saved: {output_path.name}")
    
    # Video 2: Selection Strategy (Global vs Local)
    output_path = output_dir / f"{video_name}_integrated_selection.mp4"
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame, result in tqdm(zip(frames, tracking_results), 
                              total=len(frames), 
                              desc="Selection Strategy"):
        vis = frame.copy()
        
        # Draw all validated candidates (faded cyan)
        for candidate in result['all_candidates']:
            cx, cy = candidate['center']
            radius = candidate['radius']
            cv2.circle(vis, (cx, cy), radius, (200, 200, 100), 1)
        
        if result['mode'] == 'global':
            # Global search visualization
            search_type = result.get('search_type', 'initial')
            
            if search_type == 'initial':
                # Draw exclusion zone (upper 30%)
                exclusion_y = int(height * config.FOUL_LINE_EXCLUSION_FACTOR)
                overlay = vis.copy()
                cv2.rectangle(overlay, (0, 0), (width, exclusion_y), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
                cv2.putText(vis, "EXCLUSION ZONE", (width//2 - 100, exclusion_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw search zone
                cv2.line(vis, (0, exclusion_y), (width, exclusion_y), (0, 255, 0), 2)
                cv2.putText(vis, "SEARCH ZONE", (10, exclusion_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            else:  # reactivation
                # Draw search boundary using actual last_known_y from tracker
                if result.get('last_known_y') is not None:
                    max_y_search = result['last_known_y'] + config.REACTIVATION_SEARCH_MARGIN
                else:
                    # Fallback if last_known_y not available
                    max_y_search = height // 2
                
                # Clamp to frame bounds
                max_y_search = max(0, min(height, int(max_y_search)))
                
                cv2.line(vis, (0, max_y_search), (width, max_y_search), (0, 255, 0), 3)
                cv2.putText(vis, f"Search Above Y={max_y_search} (toward pins)", (10, max(max_y_search - 30, 25)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if result.get('last_known_y') is not None:
                    # Show last known position
                    last_y = int(result['last_known_y'])
                    cv2.line(vis, (0, last_y), (width, last_y), (255, 100, 0), 2)
                    cv2.putText(vis, f"Last Known Y={last_y}", (10, last_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                
                # Shade restricted zone (BELOW search line = toward foul line, larger Y values)
                overlay = vis.copy()
                cv2.rectangle(overlay, (0, max_y_search), (width, height), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
                cv2.putText(vis, "NO SEARCH (below last position)", (10, min(max_y_search + 30, height - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        else:  # local tracking
            # Draw ROI box
            if result['roi_box']:
                x1, y1, x2, y2 = result['roi_box']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"ROI: {result['roi_size']}px", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw Kalman prediction
            if result['prediction']:
                px, py = int(result['prediction']['x']), int(result['prediction']['y'])
                cv2.drawMarker(vis, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(vis, "Prediction", (px + 10, py - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw selected candidate (yellow)
        if result['detection']:
            cx, cy = result['detection']['center']
            radius = result['detection']['radius']
            cv2.circle(vis, (cx, cy), radius + 5, (0, 255, 255), 3)
        
        # STAGE F: Draw stop threshold line (if enabled)
        if config.SHOW_STOP_THRESHOLD_LINE and result.get('stop_threshold_y') is not None:
            stop_y = int(result['stop_threshold_y'])  # Convert to int for OpenCV
            color = config.STOP_THRESHOLD_COLOR  # Magenta
            cv2.line(vis, (0, stop_y), (width, stop_y), color, 2)
            cv2.putText(vis, f"Stop Threshold Y={stop_y}", (width - 250, stop_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show trajectory status if complete
            if result.get('trajectory_complete', False):
                cv2.putText(vis, "TRAJECTORY COMPLETE", (width//2 - 150, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)
        
        # Info overlay
        mode_text = f"{result['mode'].upper()}"
        if result['mode'] == 'global':
            mode_text += f" ({result.get('search_type', 'initial').upper()})"
        
        cv2.putText(vis, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"Candidates: {result['candidates_count']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        writer.write(vis)
    
    writer.release()
    result_paths['selection'] = str(output_path)
    print(f">>> Saved: {output_path.name}")
    
    # Video 3: Trajectory View
    output_path = output_dir / f"{video_name}_integrated_trajectory.mp4"
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    trajectory_points = []
    interpolated_added = False
    
    for frame_idx, (frame, result) in enumerate(tqdm(zip(frames, tracking_results), 
                                                      total=len(frames), 
                                                      desc="Trajectory View")):
        vis = frame.copy()
        
        # Collect trajectory
        if result['detection']:
            cx, cy = result['detection']['center']
            trajectory_points.append((cx, cy))
        
        # Add interpolated points (only once when trajectory complete)
        if (not interpolated_added and 
            result.get('trajectory_complete', False) and 
            result.get('interpolated_points')):
            interpolated_added = True
        
        # Draw REAL trajectory trail (solid line)
        if len(trajectory_points) > 1:
            trail = trajectory_points[-50:]
            for i in range(1, len(trail)):
                # Fade older points
                alpha = i / len(trail)
                color = (int(255 * alpha), int(165 * alpha), 0)  # Orange gradient
                cv2.line(vis, trail[i-1], trail[i], color, 2)
        
        # Draw INTERPOLATED trajectory (dashed line)
        if interpolated_added and result.get('interpolated_points'):
            last_real = trajectory_points[-1]
            for interp_point in result['interpolated_points']:
                # Extract x, y from 3-tuple (x, y, frame_idx)
                interp_x, interp_y = int(interp_point[0]), int(interp_point[1])
                
                # Dashed line from last real to interpolated
                num_dashes = 10
                for j in range(num_dashes):
                    t1 = j / num_dashes
                    t2 = (j + 0.5) / num_dashes
                    x1 = int(last_real[0] + t1 * (interp_x - last_real[0]))
                    y1 = int(last_real[1] + t1 * (interp_y - last_real[1]))
                    x2 = int(last_real[0] + t2 * (interp_x - last_real[0]))
                    y2 = int(last_real[1] + t2 * (interp_y - last_real[1]))
                    cv2.line(vis, (x1, y1), (x2, y2), config.INTERPOLATION_COLOR, 2)
                
                # Mark interpolated endpoint
                cv2.circle(vis, (interp_x, interp_y), 6, config.INTERPOLATION_COLOR, -1)
                cv2.putText(vis, "Extrapolated", (interp_x + 10, interp_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.INTERPOLATION_COLOR, 2)
        
        # Draw current position
        if result['detection']:
            cx, cy = result['detection']['center']
            cv2.circle(vis, (cx, cy), 8, (0, 255, 255), -1)
        
        # Show trajectory count
        cv2.putText(vis, f"Trajectory Points: {len(trajectory_points)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Info
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Trajectory Points: {len(trajectory_points)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        writer.write(vis)
    
    writer.release()
    result_paths['trajectory'] = str(output_path)
    print(f">>> Saved: {output_path.name}")
    
    # Video 4: Debug Overlay (Complete)
    output_path = output_dir / f"{video_name}_integrated_debug.mp4"
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    trajectory_points = []
    for frame_idx, (frame, result) in enumerate(tqdm(zip(frames, tracking_results), 
                                                      total=len(frames), 
                                                      desc="Debug Overlay")):
        vis = frame.copy()
        
        # Collect trajectory
        if result['detection']:
            trajectory_points.append(result['detection']['center'])
        
        # Draw trajectory
        if len(trajectory_points) > 1:
            for i in range(1, len(trajectory_points)):
                cv2.line(vis, trajectory_points[i-1], trajectory_points[i], (0, 165, 255), 2)
        
        # Draw all candidates (small cyan)
        for candidate in result['all_candidates']:
            cx, cy = candidate['center']
            cv2.circle(vis, (cx, cy), 3, (255, 200, 100), 1)
        
        # Draw ROI if tracking
        if result['roi_box']:
            x1, y1, x2, y2 = result['roi_box']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Draw selected (yellow)
        if result['detection']:
            cx, cy = result['detection']['center']
            radius = result['detection']['radius']
            cv2.circle(vis, (cx, cy), radius, (0, 255, 255), 2)
        
        # Comprehensive info overlay
        info_bg = np.zeros((120, 300, 3), dtype=np.uint8)
        info_bg[:] = (0, 0, 0)
        
        mode_color = (0, 255, 0) if result['mode'] == 'local' else (255, 100, 0)
        cv2.putText(info_bg, f"Mode: {result['mode'].upper()}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        if result['mode'] == 'global':
            cv2.putText(info_bg, f"Type: {result.get('search_type', 'N/A').upper()}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(info_bg, f"Frame: {frame_idx}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_bg, f"Candidates: {result['candidates_count']}", (10, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        detection_color = (0, 255, 0) if result['detection'] else (0, 0, 255)
        cv2.putText(info_bg, f"Detected: {'YES' if result['detection'] else 'NO'}", (150, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection_color, 1)
        
        # Blend info panel
        vis[10:130, 10:310] = cv2.addWeighted(vis[10:130, 10:310], 0.3, info_bg, 0.7, 0)
        
        writer.write(vis)
    
    writer.release()
    result_paths['debug'] = str(output_path)
    print(f">>> Saved: {output_path.name}")
    
    print(f"\n>>> Generated {len(result_paths)} integrated tracking videos")
    return result_paths

