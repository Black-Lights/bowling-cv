"""
Mask video frames based on lane boundaries from Phase 1

This creates a pre-processed video where everything outside the lane is blacked out,
which helps Phase 2 ball detection focus only on the lane area.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: January 30, 2026
"""

import cv2
import numpy as np
import json
import os
import sys
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

def create_proper_lane_mask(frame_shape, boundaries):
    """
    Create lane mask between top boundary, foul line, and lateral boundaries.
    
    Parameters:
    -----------
    frame_shape : tuple
        (height, width) of frame
    boundaries : dict
        Boundary data from Phase 1
        
    Returns:
    --------
    mask : np.ndarray
        Binary mask (255 = lane area, 0 = black out)
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Get boundary parameters
    left = boundaries['master_left']
    right = boundaries['master_right']
    top_y = int(boundaries['top_boundary']['y_position'])
    foul_y = int(boundaries['median_foul_params']['center_y'])
    
    # Get slopes and x-intercepts
    left_x_intersect = left['x_intersect']
    right_x_intersect = right['x_intersect']
    left_slope = left['slope']
    right_slope = right['slope']
    
    # Calculate x positions at top boundary and foul line
    # Line equation: x = x_intersect + (y - foul_y) / slope
    
    # Left boundary
    left_x_top = int(left_x_intersect + (top_y - foul_y) / left_slope) if left_slope != 0 else left_x_intersect
    left_x_foul = left_x_intersect
    
    # Right boundary
    right_x_top = int(right_x_intersect + (top_y - foul_y) / right_slope) if right_slope != 0 else right_x_intersect
    right_x_foul = right_x_intersect
    
    # Create polygon: top-left, top-right, foul-right, foul-left
    polygon = np.array([
        [left_x_top, top_y],      # Top-left corner
        [right_x_top, top_y],     # Top-right corner
        [right_x_foul, foul_y],   # Foul line right
        [left_x_foul, foul_y]     # Foul line left
    ], dtype=np.int32)
    
    # Fill polygon
    cv2.fillPoly(mask, [polygon], 255)
    
    return mask


def mask_video_with_boundaries(input_video_path, boundary_data_path, output_video_path):
    """
    Mask video frames to show only the lane area based on detected boundaries.
    
    Parameters:
    -----------
    input_video_path : str
        Path to input video (Phase 1 output with boundary visualization)
    boundary_data_path : str
        Path to boundary_data.json from Phase 1
    output_video_path : str
        Path to save masked output video
        
    Returns:
    --------
    bool : Success status
    """
    print("\n" + "="*60)
    print("VIDEO MASKING - LANE AREA ISOLATION")
    print("="*60)
    
    # Load boundary data
    print(f"\nLoading boundary data from: {boundary_data_path}")
    with open(boundary_data_path, 'r') as f:
        boundaries = json.load(f)
    
    print(f"Boundaries loaded:")
    print(f"  Left: X={boundaries['master_left']['x_intersect']}, angle={boundaries['master_left']['median_angle']:.1f}°")
    print(f"  Right: X={boundaries['master_right']['x_intersect']}, angle={boundaries['master_right']['median_angle']:.1f}°")
    print(f"  Top: Y={boundaries['top_boundary']['y_position']:.0f}")
    print(f"  Foul: Y={boundaries['median_foul_params']['center_y']}")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {input_video_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nInput video: {input_video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    
    # Create lane mask (between top boundary, foul line, and lateral boundaries)
    print(f"\nCreating lane mask...")
    print(f"  Masking area: Y={boundaries['top_boundary']['y_position']:.0f} (top) to Y={boundaries['median_foul_params']['center_y']} (foul)")
    lane_mask = create_proper_lane_mask((height, width), boundaries)
    
    # Count lane area
    lane_area = np.count_nonzero(lane_mask)
    total_area = height * width
    lane_percentage = (lane_area / total_area) * 100
    print(f"Lane area: {lane_area:,} pixels ({lane_percentage:.1f}% of frame)")
    
    # Create output video writer
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"ERROR: Cannot create output video {output_video_path}")
        cap.release()
        return False
    
    print(f"\nProcessing frames...")
    print(f"Output: {output_video_path}")
    
    # Process each frame
    frame_count = 0
    with tqdm(total=total_frames, desc="  Masking frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply lane mask (black out everything outside lane)
            masked_frame = cv2.bitwise_and(frame, frame, mask=lane_mask)
            
            # Write masked frame
            out.write(masked_frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n✅ Masked video saved: {output_video_path}")
    print(f"✅ Processed {frame_count} frames")
    
    return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mask video with lane boundaries')
    parser.add_argument('--video', required=True, help='Video name (e.g., cropped_test3)')
    parser.add_argument('--input-dir', default='output', help='Input directory (default: output)')
    parser.add_argument('--output-name', default='masked_video.mp4', help='Output filename')
    
    args = parser.parse_args()
    
    # Construct paths
    video_name = args.video.replace('.mp4', '')
    video_dir = os.path.join(args.input_dir, video_name)
    
    input_video = os.path.join(video_dir, f'final_all_boundaries_{video_name}.mp4')
    boundary_data = os.path.join(video_dir, 'boundary_data.json')
    output_video = os.path.join(video_dir, args.output_name)
    
    # Check if files exist
    if not os.path.exists(input_video):
        print(f"ERROR: Input video not found: {input_video}")
        return 1
    
    if not os.path.exists(boundary_data):
        print(f"ERROR: Boundary data not found: {boundary_data}")
        return 1
    
    # Mask video
    success = mask_video_with_boundaries(input_video, boundary_data, output_video)
    
    if success:
        print("\n" + "="*60)
        print("MASKING COMPLETE")
        print("="*60)
        print(f"\nNext step: Run Phase 2 on masked video:")
        print(f"  python -m src.ball_tracking.main --video {output_video}")
        return 0
    else:
        print("\n" + "="*60)
        print("MASKING FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
