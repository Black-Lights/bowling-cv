"""
Main script for bowling lane detection with master lines
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import json

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(__file__))

# Import configuration
from config import *

# Import modules
from detection_functions import detect_horizontal_line, detect_vertical_boundaries_approach1
from master_line_computation import compute_master_line_from_collection, visualize_bin_analysis
from intermediate_visualization import create_intermediate_video
from tracking_analysis import analyze_master_line_tracking, plot_master_line_tracking, create_summary_plot


def save_boundary_data(output_dir, master_left, master_right, median_foul_params, video_name):
    """
    Save detected boundary data to JSON file for reuse in top boundary detection.
    """
    boundary_data = {
        'video_name': video_name,
        'master_left': {
            'x_top': int(master_left['x_top']),
            'y_top': int(master_left['y_top']),
            'x_bottom': int(master_left['x_bottom']),
            'y_bottom': int(master_left['y_bottom']),
            'x_intersect': int(master_left['x_intersect']),
            'slope': float(master_left['slope']),
            'median_angle': float(master_left.get('median_angle', 0))
        },
        'master_right': {
            'x_top': int(master_right['x_top']),
            'y_top': int(master_right['y_top']),
            'x_bottom': int(master_right['x_bottom']),
            'y_bottom': int(master_right['y_bottom']),
            'x_intersect': int(master_right['x_intersect']),
            'slope': float(master_right['slope']),
            'median_angle': float(master_right.get('median_angle', 0))
        },
        'median_foul_params': {
            'center_y': int(median_foul_params['center_y']),
            'slope': float(median_foul_params['slope']),
            'y_left': int(median_foul_params.get('y_left', 0)),
            'y_right': int(median_foul_params.get('y_right', 0))
        }
    }
    
    boundary_file = os.path.join(output_dir, 'boundary_data.json')
    with open(boundary_file, 'w') as f:
        json.dump(boundary_data, f, indent=2)
    
    print(f"  Saved boundary data: boundary_data.json")
    return boundary_file


def load_boundary_data(output_dir):
    """
    Load previously saved boundary data from JSON file.
    Returns None if file doesn't exist.
    """
    boundary_file = os.path.join(output_dir, 'boundary_data.json')
    
    if not os.path.exists(boundary_file):
        return None
    
    with open(boundary_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded boundary data from: {boundary_file}")
    return data


def collect_lines_from_frames(video_path, num_frames=100, angle_mode='from_horizontal'):
    """
    Phase 1: Collect lines from first num_frames using approach1.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1: Collecting lines from first {num_frames} frames")
    print(f"Video: {video_path}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(num_frames, total_frames)
    
    left_lines_collected = []
    right_lines_collected = []
    foul_params_list = []
    
    for frame_idx in tqdm(range(frames_to_process), desc="Collecting lines"):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        _, _, foul_params = detect_horizontal_line(frame)
        
        if foul_params:
            foul_params_list.append(foul_params)
            
            left_bound, right_bound, _, left_lines, right_lines = detect_vertical_boundaries_approach1(
                frame, foul_params, angle_mode
            )
            
            if left_bound:
                # Calculate angle from the line coordinates
                x1, y1, x2, y2 = left_bound[0]
                from detection_utils import calculate_line_angle, normalize_line_direction
                x1_norm, y1_norm, x2_norm, y2_norm = normalize_line_direction(x1, y1, x2, y2)
                line_angle = calculate_line_angle(x1_norm, y1_norm, x2_norm, y2_norm, angle_mode)
                
                left_lines_collected.append({
                    'line': left_bound[0],
                    'x_intersect': left_bound[1],
                    'slope': left_bound[2],
                    'angle': line_angle,
                    'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                })
            
            if right_bound:
                # Calculate angle from the line coordinates
                x1, y1, x2, y2 = right_bound[0]
                from detection_utils import calculate_line_angle, normalize_line_direction
                x1_norm, y1_norm, x2_norm, y2_norm = normalize_line_direction(x1, y1, x2, y2)
                line_angle = calculate_line_angle(x1_norm, y1_norm, x2_norm, y2_norm, angle_mode)
                
                right_lines_collected.append({
                    'line': right_bound[0],
                    'x_intersect': right_bound[1],
                    'slope': right_bound[2],
                    'angle': line_angle,
                    'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                })
    
    cap.release()
    
    if foul_params_list:
        median_center_y = int(np.median([fp['center_y'] for fp in foul_params_list]))
        median_foul_params = foul_params_list[0].copy()
        median_foul_params['center_y'] = median_center_y
        
        print(f"\nCollected {len(left_lines_collected)} left lines")
        print(f"Collected {len(right_lines_collected)} right lines")
        print(f"Median foul line Y: {median_center_y}")
    else:
        median_foul_params = None
    
    return {
        'left_lines': left_lines_collected,
        'right_lines': right_lines_collected,
        'foul_params_list': foul_params_list,
        'median_foul_params': median_foul_params
    }


def process_video_with_master_lines(video_path, output_path, master_left, master_right,
                                    median_foul_params, visualization_mode='final'):
    """
    Phase 3: Apply master lines to full video.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 3: Applying master lines to full video")
    print(f"Visualization: {visualization_mode}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    codecs_to_try = [('avc1', '.mp4'), ('XVID', '.avi'), ('MJPG', '.avi')]
    
    out = None
    final_output_path = output_path
    
    for codec, extension in codecs_to_try:
        try:
            if not output_path.endswith(extension):
                final_output_path = os.path.splitext(output_path)[0] + extension
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
            
            if out.isOpened():
                print(f"Using codec: {codec}")
                break
            else:
                out.release()
                out = None
        except:
            continue
    
    if out is None or not out.isOpened():
        print("Error: Could not initialize video writer!")
        cap.release()
        return
    
    frame_count = 0
    
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        result_frame = frame.copy()
        
        # Draw foul line FIRST (so it appears under other lines)
        if median_foul_params and visualization_mode in ['final', 'with_stats']:
            cv2.line(result_frame,
                    (0, median_foul_params['y_left']),
                    (width, median_foul_params['y_right']),
                    (255, 0, 255), 3)
            cv2.circle(result_frame,
                      (median_foul_params['center_x'], median_foul_params['center_y']),
                      10, (0, 255, 255), -1)
        
        # Draw based on visualization mode
        if visualization_mode in ['final', 'master_lines_only', 'with_stats']:
            # Draw master left line
            if master_left:
                cv2.line(result_frame,
                        (master_left['x_top'], master_left['y_top']),
                        (master_left['x_bottom'], master_left['y_bottom']),
                        (0, 255, 0), 3)
                
                if median_foul_params:
                    cv2.circle(result_frame,
                             (master_left['x_intersect'], median_foul_params['center_y']),
                             8, (0, 255, 0), -1)
            
            # Draw master right line
            if master_right:
                cv2.line(result_frame,
                        (master_right['x_top'], master_right['y_top']),
                        (master_right['x_bottom'], master_right['y_bottom']),
                        (0, 0, 255), 3)
                
                if median_foul_params:
                    cv2.circle(result_frame,
                             (master_right['x_intersect'], median_foul_params['center_y']),
                             8, (0, 0, 255), -1)
            
            # Add statistics overlay
            if visualization_mode == 'with_stats':
                info_y = 30
                cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                info_y += 30
                
                if master_left:
                    cv2.putText(result_frame, 
                               f"Left: X={master_left['x_intersect']}, Angle={master_left['median_angle']:.1f}deg",
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    info_y += 25
                
                if master_right:
                    cv2.putText(result_frame,
                               f"Right: X={master_right['x_intersect']}, Angle={master_right['median_angle']:.1f}deg",
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    info_y += 25
                
                if master_left and master_right:
                    lane_width = master_right['x_intersect'] - master_left['x_intersect']
                    cv2.putText(result_frame, f"Lane Width: {lane_width} pixels",
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        out.write(result_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"Processed {frame_count} frames. Saved to: {final_output_path}")


def process_single_video(video_path, output_dir):
    """Process a single video file."""
    # Handle both absolute paths and filenames
    if not os.path.isabs(video_path):
        # If just filename, look in ASSETS_DIR
        full_video_path = os.path.join(ASSETS_DIR, video_path)
    else:
        full_video_path = video_path
    
    if not os.path.exists(full_video_path):
        print(f"Error: Video not found: {full_video_path}")
        return None
    
    video_name = os.path.splitext(os.path.basename(full_video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"# Processing: {video_name}")
    print(f"{'#'*70}")
    
    # Determine angle mode
    angle_mode = 'from_vertical' if USE_ABSOLUTE_ANGLES else 'from_horizontal'
    
    # PHASE 1: Collect lines
    collection_data = collect_lines_from_frames(full_video_path, NUM_COLLECTION_FRAMES, angle_mode)
    
    if not collection_data or not collection_data['median_foul_params']:
        print(f"Error: Could not collect lines from {video_name}")
        return
    
    median_foul_params = collection_data['median_foul_params']
    
    # PHASE 2: Compute master lines
    print(f"\n{'='*60}")
    print(f"PHASE 2: Computing master lines")
    print(f"{'='*60}")
    
    # Debug: Show some sample line coordinates
    if DEBUG_MODE and collection_data['left_lines']:
        print(f"\nDEBUG - Sample left line:")
        sample = collection_data['left_lines'][0]
        x1, y1, x2, y2 = sample['line']
        print(f"  Coords: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  dx={x2-x1}, dy={y2-y1}")
        print(f"  Intersect X: {sample['x_intersect']}")
        print(f"  Foul center X: {median_foul_params['center_x']}")
    
    if DEBUG_MODE and collection_data['right_lines']:
        print(f"\nDEBUG - Sample right line:")
        sample = collection_data['right_lines'][0]
        x1, y1, x2, y2 = sample['line']
        print(f"  Coords: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  dx={x2-x1}, dy={y2-y1}")
        print(f"  Intersect X: {sample['x_intersect']}")
        print(f"  Foul center X: {median_foul_params['center_x']}")
    
    master_left, debug_left = compute_master_line_from_collection(
        collection_data['left_lines'],
        median_foul_params,
        bin_width=BIN_WIDTH,
        vote_threshold=VOTE_THRESHOLD,
        angle_tolerance=ANGLE_TOLERANCE,
        side='left',
        angle_mode=angle_mode
    )
    
    master_right, debug_right = compute_master_line_from_collection(
        collection_data['right_lines'],
        median_foul_params,
        bin_width=BIN_WIDTH,
        vote_threshold=VOTE_THRESHOLD,
        angle_tolerance=ANGLE_TOLERANCE,
        side='right',
        angle_mode=angle_mode
    )
    
    if master_left:
        print(f"\nLeft master line:")
        print(f"  X-intersect: {master_left['x_intersect']}")
        if angle_mode == 'from_vertical':
            print(f"  Angle: {master_left['median_angle']:.2f}° (from vertical, 0°=vertical)")
            print(f"         → Positive = leans RIGHT (toward center due to perspective)")
        else:
            print(f"  Angle: {master_left['median_angle']:.2f}° (from horizontal)")
    else:
        print("\nWarning: Could not compute left master line")
    
    if master_right:
        print(f"\nRight master line:")
        print(f"  X-intersect: {master_right['x_intersect']}")
        if angle_mode == 'from_vertical':
            print(f"  Angle: {master_right['median_angle']:.2f}° (from vertical, 0°=vertical)")
            print(f"         → Negative = leans LEFT (toward center due to perspective)")
        else:
            print(f"  Angle: {master_right['median_angle']:.2f}° (from horizontal)")
    else:
        print("\nWarning: Could not compute right master line")
    
    # Save bin analysis
    if SAVE_BIN_ANALYSIS:
        if debug_left:
            visualize_bin_analysis(debug_left, 'left',
                                  os.path.join(video_output_dir, 'bin_analysis_left.png'),
                                  angle_mode)
        if debug_right:
            visualize_bin_analysis(debug_right, 'right',
                                  os.path.join(video_output_dir, 'bin_analysis_right.png'),
                                  angle_mode)
    # Save boundary data for reuse
    save_boundary_data(video_output_dir, master_left, master_right, median_foul_params, video_name)
    
    
    # PHASE 3: Process full video
    output_video = os.path.join(video_output_dir, 
                               f'master_{VISUALIZATION_MODE}_{os.path.basename(full_video_path)}')
    
    process_video_with_master_lines(
        full_video_path,
        output_video,
        master_left,
        master_right,
        median_foul_params,
        VISUALIZATION_MODE
    )
    
    # PHASE 4: Create intermediate visualization videos
    if SAVE_INTERMEDIATE_VIDEOS:
        print(f"\n{'='*60}")
        print(f"PHASE 4: Creating intermediate visualization videos")
        print(f"{'='*60}")
        
        intermediate_dir = os.path.join(video_output_dir, 'intermediate')
        os.makedirs(intermediate_dir, exist_ok=True)
        
        for mode in INTERMEDIATE_MODES:
            output_path = os.path.join(intermediate_dir, f'{mode}_{os.path.basename(full_video_path)}')
            create_intermediate_video(full_video_path, output_path, mode, detect_horizontal_line)
    
    # PHASE 5: Tracking analysis
    if GENERATE_TRACKING_PLOTS and master_left and master_right:
        print(f"\n{'='*60}")
        print(f"PHASE 5: Tracking analysis")
        print(f"{'='*60}")
        
        tracking_data = analyze_master_line_tracking(
            full_video_path,
            master_left,
            master_right,
            median_foul_params
        )
        
        if tracking_data:
            plot_master_line_tracking(
                tracking_data,
                video_name,
                video_output_dir,
                master_left,
                master_right
            )
    
    print(f"\n✓ Completed: {video_name}")
    print(f"  Output: {video_output_dir}")
    
    return tracking_data if (GENERATE_TRACKING_PLOTS and master_left and master_right) else None


def main():
    """Main entry point."""
    print("="*70)
    print(" BOWLING LANE BOUNDARY DETECTION - MASTER LINE APPROACH")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Videos to process: {len(VIDEO_FILES)}")
    print(f"  Collection frames: {NUM_COLLECTION_FRAMES}")
    print(f"  Bin width: {BIN_WIDTH} pixels")
    print(f"  Vote threshold: {VOTE_THRESHOLD*100}%")
    print(f"  Angle tolerance: ±{ANGLE_TOLERANCE}°")
    print(f"  Visualization: {VISUALIZATION_MODE}")
    print(f"  Angle mode: {'from_vertical' if USE_ABSOLUTE_ANGLES else 'from_horizontal'}")
    print(f"  Intermediate videos: {SAVE_INTERMEDIATE_VIDEOS}")
    print(f"  Tracking analysis: {GENERATE_TRACKING_PLOTS}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Storage for tracking data from all videos
    all_tracking_data = {}
    
    # Process each video
    for video_file in VIDEO_FILES:
        try:
            tracking_data = process_single_video(video_file, OUTPUT_DIR)
            if tracking_data:
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                all_tracking_data[video_name] = tracking_data
        except Exception as e:
            print(f"\n✗ Error processing {video_file}: {str(e)}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
    
    # Create summary plot comparing all videos
    if CREATE_SUMMARY_PLOT and all_tracking_data:
        print(f"\n{'='*70}")
        print(" CREATING SUMMARY PLOT")
        print("="*70)
        create_summary_plot(all_tracking_data, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print(" ALL VIDEOS PROCESSED")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  - Master line videos")
    print(f"  - Bin analysis plots")
    if SAVE_INTERMEDIATE_VIDEOS:
        print(f"  - Intermediate visualization videos")
    if GENERATE_TRACKING_PLOTS:
        print(f"  - Tracking analysis plots")
    if CREATE_SUMMARY_PLOT and all_tracking_data:
        print(f"  - Summary comparison plot")


if __name__ == "__main__":
    main()
