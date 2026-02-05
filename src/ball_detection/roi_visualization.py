"""
Stage C: ROI Logic Visualization

Generates intermediate debug videos showing:
1. Global search mode
2. Local tracking with ROI boxes
3. Kalman predictions
4. Mode comparison
5. Perspective scaling
6. Full pipeline overview

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Date: February 1, 2026
"""

import cv2
import numpy as np
import os
from pathlib import Path
import subprocess
import shutil
from tqdm import tqdm
from .roi_logic import BallTracker
from .motion_detection import apply_background_subtraction
from .mask_video import create_masked_lane_video


def draw_velocity_vector(frame, start_pos, velocity, color=(0, 255, 0), scale=5):
    """Draw velocity vector arrow"""
    x, y = start_pos
    vx, vy = velocity
    
    end_x = int(x + vx * scale)
    end_y = int(y + vy * scale)
    
    cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y), color, 2, tipLength=0.3)


def draw_roi_box(frame, roi_box, color=(0, 255, 255), thickness=2):
    """Draw ROI bounding box"""
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_detection(frame, center, radius, color=(0, 0, 255), label=None):
    """Draw detected ball"""
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    
    cv2.circle(frame, (cx, cy), r, color, 2)
    cv2.circle(frame, (cx, cy), 3, color, -1)  # Center dot
    
    if label:
        cv2.putText(frame, label, (cx + r + 5, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_prediction(frame, prediction, color=(255, 0, 0)):
    """Draw Kalman prediction"""
    if prediction:
        px = int(prediction['x'])
        py = int(prediction['y'])
        
        # Draw crosshair
        cv2.drawMarker(frame, (px, py), color, cv2.MARKER_CROSS, 15, 2)
        
        # Draw velocity vector
        if 'vx' in prediction and 'vy' in prediction:
            draw_velocity_vector(frame, (px, py), (prediction['vx'], prediction['vy']), color)


def draw_trajectory(frame, trajectory, color=(0, 255, 0), max_points=30):
    """Draw trajectory trail"""
    if len(trajectory) < 2:
        return
    
    points = trajectory[-max_points:]
    for i in range(1, len(points)):
        # Extract x, y from (x, y, frame_idx, radius) tuple
        pt1 = (int(points[i-1][0]), int(points[i-1][1]))
        pt2 = (int(points[i][0]), int(points[i][1]))
        
        # Fade effect
        alpha = i / len(points)
        thickness = max(1, int(alpha * 3))
        
        cv2.line(frame, pt1, pt2, color, thickness)


def add_text_overlay(frame, text_lines, position=(10, 30), bg_color=(0, 0, 0), alpha=0.7):
    """Add text overlay with semi-transparent background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 25
    
    # Calculate background size
    max_width = 0
    for text in text_lines:
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        max_width = max(max_width, w)
    
    bg_height = len(text_lines) * line_height + 10
    bg_width = max_width + 20
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                 (position[0] - 5, position[1] - 20),
                 (position[0] + bg_width, position[1] + bg_height - 20),
                 bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw text
    y = position[1]
    for text in text_lines:
        cv2.putText(frame, text, (position[0], y), font, font_scale, (255, 255, 255), thickness)
        y += line_height


def save_video_ffmpeg(frames, output_path, fps, desc="Saving video"):
    """Save frames as video using ffmpeg"""
    if not frames:
        return False
    
    temp_dir = os.path.join(os.path.dirname(output_path), 'temp_frames_roi')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Ensure dimensions are even (required for yuv420p encoding)
        first_frame = frames[0]
        h, w = first_frame.shape[:2]
        
        # Pad to even dimensions if necessary
        if h % 2 != 0 or w % 2 != 0:
            new_h = h if h % 2 == 0 else h + 1
            new_w = w if w % 2 == 0 else w + 1
            padded_frames = []
            for frame in frames:
                # Create padded frame (add black pixels at bottom/right)
                padded = np.zeros((new_h, new_w, 3), dtype=frame.dtype)
                padded[:h, :w] = frame
                padded_frames.append(padded)
            frames = padded_frames
        
        # Save frames as PNG
        for i, frame in enumerate(tqdm(frames, desc=desc, disable=False)):
            frame_path = os.path.join(temp_dir, f'frame_{i:06d}.png')
            cv2.imwrite(frame_path, frame)
        
        # Combine with ffmpeg
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        # FFmpeg may return warnings but still create valid output
        if result.returncode != 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"⚠ FFmpeg warning (video created successfully): {desc}")
                # Optionally show last part of stderr for debugging
                # if result.stderr:
                #     print(f"  {result.stderr.strip()[-200:]}")
            else:
                print(f"✗ FFmpeg failed for {desc}")
                if result.stderr:
                    print(f"  Error: {result.stderr.strip()}")
                return False
        
        return True
        
    finally:
        # Clean up temp frames
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def generate_roi_videos(video_path, config, output_base_dir):
    """
    Generate all Stage C intermediate visualization videos
    
    Args:
        video_path: Path to input video
        config: Configuration module
        output_base_dir: Base output directory
        
    Returns:
        dict: Paths to generated videos
    """
    video_name = Path(video_path).stem
    intermediate_dir = os.path.join(output_base_dir, 'ball_detection', 'intermediate')
    os.makedirs(intermediate_dir, exist_ok=True)
    
    if config.VERBOSE:
        print(f"\n{'='*80}")
        print(f"STAGE C: ROI LOGIC VISUALIZATION")
        print(f"Video: {video_name}")
        print(f"Output: {intermediate_dir}")
        print(f"{'='*80}\n")
    
    # Get boundary data for foul line position
    boundary_file = os.path.join(output_base_dir, 'boundary_data.json')
    if not os.path.exists(boundary_file):
        raise FileNotFoundError(f"Boundary data not found: {boundary_file}")
    
    import json
    with open(boundary_file, 'r') as f:
        boundary_data = json.load(f)
    
    foul_line_y = boundary_data['median_foul_params']['center_y']
    
    # Get masked frames and motion detection
    masked_frames_gen = create_masked_lane_video(video_path, config, save_video=False)
    motion_gen = apply_background_subtraction(masked_frames_gen, config, save_videos=False)
    
    # Get video properties from original video file
    full_video_path = video_path
    if not os.path.isabs(video_path):
        full_video_path = os.path.join(config.ASSETS_DIR, video_path)
    
    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {full_video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if width == 0 or height == 0:
        raise ValueError(f"Invalid video dimensions: {width}x{height}")
    
    # Initialize tracker
    tracker = BallTracker(config, width, height, foul_line_y)
    
    # Storage for different visualizations
    global_search_frames = []
    local_tracking_frames = []
    kalman_frames = []
    comparison_frames = []
    scaling_frames = []
    pipeline_frames = []
    
    # Process all frames
    if config.VERBOSE:
        print("Processing frames with ROI logic...")
    
    for frame_idx, denoised_mask, metadata, intermediate_masks in tqdm(
        motion_gen,
        desc="Stage C: ROI Logic",
        disable=not config.VERBOSE
    ):
        original_masked = intermediate_masks['original_masked']
        
        # Process frame with tracker
        result = tracker.process_frame(denoised_mask, frame_idx)
        
        # Video 1: Global Search Visualization
        if config.SAVE_ROI_GLOBAL_SEARCH_VIDEO:
            frame1 = original_masked.copy()
            
            if result['mode'] == 'global':
                # Draw all contours
                contours, _ = cv2.findContours(denoised_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame1, contours, -1, (100, 100, 100), 1)
                
                # Highlight foul line priority zone
                priority_start = int(foul_line_y - config.FOUL_LINE_PRIORITY_ZONE)
                cv2.rectangle(frame1, (0, priority_start), (width, int(foul_line_y)), 
                            (50, 50, 0), 2)
                cv2.putText(frame1, "Priority Zone", (10, priority_start + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                # Draw detection if found
                if result['detection']:
                    draw_detection(frame1, result['detection']['center'], 
                                 result['detection']['radius'], (0, 255, 0), "DETECTED!")
                    
                    if 'velocity' in result['detection']:
                        vx, vy = result['detection']['velocity']
                        color = (0, 255, 0) if vy < 0 else (0, 0, 255)
                        draw_velocity_vector(frame1, result['detection']['center'], 
                                           (vx, vy), color)
            
            text = [
                f"Mode: GLOBAL SEARCH",
                f"Frame: {frame_idx}",
                f"Status: {'TRACKING!' if result['mode'] == 'local' else 'SEARCHING...'}",
            ]
            add_text_overlay(frame1, text)
            global_search_frames.append(frame1)
        
        # Video 2: Local Tracking Visualization
        if config.SAVE_ROI_LOCAL_TRACKING_VIDEO:
            frame2 = original_masked.copy()
            
            if result['mode'] == 'local':
                # Draw ROI box
                if result['roi_box']:
                    # Color based on size (green=large, blue=small)
                    roi_size = result['roi_size']
                    ratio = (roi_size - config.B_MIN) / (config.MAX_BALL_RADIUS - config.B_MIN)
                    color = (int(255 * (1-ratio)), int(255 * ratio), 0)
                    draw_roi_box(frame2, result['roi_box'], color, 3)
                    
                    # Draw ROI size text
                    x1, y1, x2, y2 = result['roi_box']
                    roi_text = f"ROI: {x2-x1}x{y2-y1}px"
                    cv2.putText(frame2, roi_text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw prediction
                if result['prediction']:
                    draw_prediction(frame2, result['prediction'], (255, 0, 0))
                
                # Draw detection
                if result['detection']:
                    draw_detection(frame2, result['detection']['center'],
                                 result['detection']['radius'], (0, 255, 0))
                
                # Draw trajectory
                draw_trajectory(frame2, tracker.trajectory, (0, 255, 255))
            
            text = [
                f"Mode: {'LOCAL TRACKING' if result['mode'] == 'local' else 'GLOBAL SEARCH'}",
                f"Frame: {frame_idx}",
                f"ROI Size: {result['roi_size']}px" if result['roi_size'] else "ROI: N/A",
                f"Trajectory: {len(tracker.trajectory)} points"
            ]
            add_text_overlay(frame2, text)
            local_tracking_frames.append(frame2)
        
        # Video 3: Kalman Prediction Visualization
        if config.SAVE_KALMAN_PREDICTION_VIDEO:
            frame3 = cv2.cvtColor(denoised_mask, cv2.COLOR_GRAY2BGR)
            
            # Draw trajectory
            draw_trajectory(frame3, tracker.trajectory, (0, 255, 0), max_points=50)
            
            # Draw prediction (blue)
            if result['prediction']:
                draw_prediction(frame3, result['prediction'], (255, 0, 0))
                pred_text = f"Pred: ({result['prediction']['x']:.1f}, {result['prediction']['y']:.1f})"
                cv2.putText(frame3, pred_text, (10, height - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw detection (red)
            if result['detection']:
                draw_detection(frame3, result['detection']['center'],
                             result['detection']['radius'], (0, 0, 255))
                det_text = f"Meas: ({result['detection']['center'][0]:.1f}, {result['detection']['center'][1]:.1f})"
                cv2.putText(frame3, det_text, (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            text = [
                f"Kalman Filter Tracking",
                f"Blue = Prediction | Red = Measurement",
                f"Green = Trajectory Trail"
            ]
            add_text_overlay(frame3, text)
            kalman_frames.append(frame3)
        
        # Video 4: Mode Comparison (side-by-side)
        if config.SAVE_ROI_MODE_COMPARISON_VIDEO:
            # Left: Full frame with mode indicator
            left_frame = original_masked.copy()
            
            if result['mode'] == 'global':
                cv2.putText(left_frame, "GLOBAL SEARCH", (width//2 - 100, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(left_frame, "LOCAL TRACKING", (width//2 - 100, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if result['detection']:
                draw_detection(left_frame, result['detection']['center'],
                             result['detection']['radius'], (0, 255, 0))
            
            # Right: ROI zoomed view (if in local mode)
            if result['mode'] == 'local' and result['roi_box']:
                x1, y1, x2, y2 = result['roi_box']
                roi_crop = original_masked[y1:y2, x1:x2].copy()
                
                # Resize to match left frame height
                scale = left_frame.shape[0] / roi_crop.shape[0] if roi_crop.shape[0] > 0 else 1
                right_frame = cv2.resize(roi_crop, None, fx=scale, fy=scale)
                
                # Pad to same height
                if right_frame.shape[0] < left_frame.shape[0]:
                    pad_height = left_frame.shape[0] - right_frame.shape[0]
                    right_frame = cv2.copyMakeBorder(right_frame, 0, pad_height, 0, 0,
                                                    cv2.BORDER_CONSTANT, value=0)
            else:
                right_frame = np.zeros_like(left_frame)
                cv2.putText(right_frame, "Waiting for ball...", (50, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            
            # Ensure same height
            min_height = min(left_frame.shape[0], right_frame.shape[0])
            left_frame = left_frame[:min_height]
            right_frame = right_frame[:min_height]
            
            comparison = np.hstack([left_frame, right_frame])
            comparison_frames.append(comparison)
        
        # Video 5: Scaling Demonstration
        if config.SAVE_ROI_SCALING_DEMO_VIDEO:
            frame5 = original_masked.copy()
            
            # Draw multiple ROI examples at different Y positions
            y_positions = [height - 100, height // 2, 200, 100]
            
            for y_pos in y_positions:
                from .roi_logic import calculate_roi_size, create_roi_box
                roi_size = calculate_roi_size(y_pos, config)
                # Avoid division by zero
                ratio = y_pos / height if height > 0 else 0
                color = (0, int(255 * ratio), int(255 * (1 - ratio)))
                
                # Draw example ROI
                x_center = width // 2
                roi = create_roi_box(x_center, y_pos, roi_size, width, height)
                draw_roi_box(frame5, roi, color, 2)
                
                # Add label
                label = f"y={y_pos}: ROI={roi_size}px"
                cv2.putText(frame5, label, (roi[0] + 5, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw actual ball if tracking
            if result['detection']:
                draw_detection(frame5, result['detection']['center'],
                             result['detection']['radius'], (0, 255, 0))
                
                if result['roi_box']:
                    draw_roi_box(frame5, result['roi_box'], (255, 255, 0), 3)
            
            text = [
                f"Perspective Scaling: B_t = max({config.B_MIN}, {config.K_SCALE} * y)",
                f"Near foul line (large y): Large ROI",
                f"Near pins (small y): Small ROI (min={config.B_MIN}px)"
            ]
            add_text_overlay(frame5, text, position=(10, height - 80))
            scaling_frames.append(frame5)
        
        # Video 6: Full Pipeline (2x3 grid)
        if config.SAVE_FULL_ROI_PIPELINE_VIDEO:
            # Create 2x3 grid
            def add_label_to_frame(frame, label):
                labeled = frame.copy()
                cv2.putText(labeled, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                return labeled
            
            # Resize all to same dimensions
            target_h = height // 2
            target_w = width // 2
            
            # Row 1
            frame_orig = cv2.resize(original_masked, (target_w, target_h))
            frame_orig = add_label_to_frame(frame_orig, "1. Original Masked")
            
            frame_denoised = cv2.resize(cv2.cvtColor(denoised_mask, cv2.COLOR_GRAY2BGR), 
                                       (target_w, target_h))
            frame_denoised = add_label_to_frame(frame_denoised, "2. Denoised Mask")
            
            # Row 2
            frame_roi = original_masked.copy()
            if result['roi_box']:
                draw_roi_box(frame_roi, result['roi_box'], (0, 255, 255), 2)
            if result['detection']:
                draw_detection(frame_roi, result['detection']['center'],
                             result['detection']['radius'], (0, 255, 0))
            frame_roi = cv2.resize(frame_roi, (target_w, target_h))
            frame_roi = add_label_to_frame(frame_roi, f"3. ROI ({result['mode'].upper()})")
            
            frame_kalman = cv2.cvtColor(denoised_mask, cv2.COLOR_GRAY2BGR)
            if result['prediction']:
                draw_prediction(frame_kalman, result['prediction'], (255, 0, 0))
            if result['detection']:
                draw_detection(frame_kalman, result['detection']['center'],
                             result['detection']['radius'], (0, 0, 255))
            frame_kalman = cv2.resize(frame_kalman, (target_w, target_h))
            frame_kalman = add_label_to_frame(frame_kalman, "4. Kalman Filter")
            
            # Row 3
            frame_detect = original_masked.copy()
            if result['detection']:
                draw_detection(frame_detect, result['detection']['center'],
                             result['detection']['radius'], (0, 255, 0))
            frame_detect = cv2.resize(frame_detect, (target_w, target_h))
            frame_detect = add_label_to_frame(frame_detect, "5. Detection")
            
            frame_traj = original_masked.copy()
            draw_trajectory(frame_traj, tracker.trajectory, (0, 255, 255), max_points=100)
            if result['detection']:
                draw_detection(frame_traj, result['detection']['center'],
                             result['detection']['radius'], (0, 255, 0))
            frame_traj = cv2.resize(frame_traj, (target_w, target_h))
            frame_traj = add_label_to_frame(frame_traj, "6. Full Trajectory")
            
            # Stack into grid
            row1 = np.hstack([frame_orig, frame_denoised])
            row2 = np.hstack([frame_roi, frame_kalman])
            row3 = np.hstack([frame_detect, frame_traj])
            grid = np.vstack([row1, row2, row3])
            
            pipeline_frames.append(grid)
    
    # Save all videos
    saved_videos = {}
    
    if config.VERBOSE:
        print(f"\nGenerating ROI visualization videos...")
    
    if config.SAVE_ROI_GLOBAL_SEARCH_VIDEO and global_search_frames:
        path = os.path.join(intermediate_dir, f'{video_name}_roi_global_search.mp4')
        if save_video_ffmpeg(global_search_frames, path, fps, "ROI Global Search"):
            saved_videos['global_search'] = path
            if config.VERBOSE:
                print(f"✓ Saved: {path}")
    
    if config.SAVE_ROI_LOCAL_TRACKING_VIDEO and local_tracking_frames:
        path = os.path.join(intermediate_dir, f'{video_name}_roi_local_tracking.mp4')
        if save_video_ffmpeg(local_tracking_frames, path, fps, "ROI Local Tracking"):
            saved_videos['local_tracking'] = path
            if config.VERBOSE:
                print(f"✓ Saved: {path}")
    
    if config.SAVE_KALMAN_PREDICTION_VIDEO and kalman_frames:
        path = os.path.join(intermediate_dir, f'{video_name}_kalman_prediction.mp4')
        if save_video_ffmpeg(kalman_frames, path, fps, "Kalman Prediction"):
            saved_videos['kalman'] = path
            if config.VERBOSE:
                print(f"✓ Saved: {path}")
    
    if config.SAVE_ROI_MODE_COMPARISON_VIDEO and comparison_frames:
        path = os.path.join(intermediate_dir, f'{video_name}_roi_mode_comparison.mp4')
        if save_video_ffmpeg(comparison_frames, path, fps, "Mode Comparison"):
            saved_videos['comparison'] = path
            if config.VERBOSE:
                print(f"✓ Saved: {path}")
    
    if config.SAVE_ROI_SCALING_DEMO_VIDEO and scaling_frames:
        path = os.path.join(intermediate_dir, f'{video_name}_roi_scaling_demo.mp4')
        if save_video_ffmpeg(scaling_frames, path, fps, "Scaling Demo"):
            saved_videos['scaling'] = path
            if config.VERBOSE:
                print(f"✓ Saved: {path}")
    
    if config.SAVE_FULL_ROI_PIPELINE_VIDEO and pipeline_frames:
        path = os.path.join(intermediate_dir, f'{video_name}_full_roi_pipeline.mp4')
        if save_video_ffmpeg(pipeline_frames, path, fps, "Full Pipeline"):
            saved_videos['pipeline'] = path
            if config.VERBOSE:
                print(f"✓ Saved: {path}")
    
    if config.VERBOSE:
        print(f"\nStage C Complete! Generated {len(saved_videos)} videos.")
    
    return saved_videos


# Import at end to avoid circular imports
from .roi_logic import calculate_roi_size, create_roi_box
