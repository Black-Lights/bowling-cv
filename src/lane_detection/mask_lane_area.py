"""
Mask video to show only the bowling lane area (between boundaries, above foul line)
Black out everything else to focus top boundary detection
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import subprocess
import shutil


def create_lane_mask(frame_shape, master_left, master_right, median_foul_params):
    """
    Create a binary mask for the bowling lane area.
    
    Parameters:
    -----------
    frame_shape : tuple
        (height, width) of the frame
    master_left : dict
        Left boundary parameters with x_intersect and slope
    master_right : dict
        Right boundary parameters with x_intersect and slope
    median_foul_params : dict
        Foul line parameters with center_y and slope
        
    Returns:
    --------
    mask : numpy.ndarray
        Binary mask (255 = keep, 0 = black out)
    """
    height, width = frame_shape[:2]
    
    # Create blank mask (all black)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Get boundary positions
    left_x = master_left['x_intersect']
    right_x = master_right['x_intersect']
    foul_y = median_foul_params['center_y']
    
    # For sloped boundaries, we need to account for the angle
    left_slope = master_left.get('slope', 0)
    right_slope = master_right.get('slope', 0)
    
    # Create polygon points for the lane area
    # We'll draw a polygon from top-left to top-right to bottom-right to bottom-left
    
    # Calculate x positions at top (y=0) and foul line (y=foul_y)
    # Using line equation: x = x_intersect + (y - y_intersect) / slope
    # Assuming x_intersect is at foul line
    
    # Left boundary at top (y=0)
    left_x_top = int(left_x + (0 - foul_y) / left_slope) if left_slope != 0 else left_x
    # Left boundary at foul line
    left_x_foul = left_x
    
    # Right boundary at top (y=0)  
    right_x_top = int(right_x + (0 - foul_y) / right_slope) if right_slope != 0 else right_x
    # Right boundary at foul line
    right_x_foul = right_x
    
    # Create polygon points (clockwise from top-left)
    polygon_points = np.array([
        [left_x_top, 0],           # Top-left
        [right_x_top, 0],          # Top-right
        [right_x_foul, foul_y],    # Bottom-right (at foul line)
        [left_x_foul, foul_y]      # Bottom-left (at foul line)
    ], dtype=np.int32)
    
    # Fill the polygon with white (255 = keep this area)
    cv2.fillPoly(mask, [polygon_points], 255)
    
    return mask


def apply_mask_to_video(video_path, output_path, master_left, master_right, 
                       median_foul_params, mask_color=(0, 0, 0)):
    """
    Apply lane mask to video, blacking out areas outside the bowling lane.
    
    Parameters:
    -----------
    video_path : str
        Path to input video
    output_path : str
        Path to save masked video
    master_left : dict
        Left boundary parameters
    master_right : dict
        Right boundary parameters
    median_foul_params : dict
        Foul line parameters
    mask_color : tuple
        Color for masked areas (default: black)
        
    Returns:
    --------
    dict : Information about the masked video
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Read first frame to create mask
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return None
    
    # Create the mask
    print("\nCreating lane mask...")
    mask = create_lane_mask(first_frame.shape, master_left, master_right, median_foul_params)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Create temp directory for frames
    temp_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"\nApplying mask to frames...")
    frame_count = 0
    
    for i in tqdm(range(total_frames), desc="Masking frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create masked frame
        # Where mask is 0 (black), use mask_color
        # Where mask is 255 (white), keep original frame
        masked_frame = frame.copy()
        masked_frame[mask == 0] = mask_color
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{i:05d}.jpg")
        cv2.imwrite(frame_path, masked_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_count += 1
    
    cap.release()
    print(f"Masked {frame_count} frames")
    
    # Use ffmpeg to combine frames into video
    print(f"\nCombining frames with ffmpeg...")
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%05d.jpg'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',  # High quality
        output_path
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        print(f"Video created: {output_path}")
        success = True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr}")
        success = False
    except FileNotFoundError:
        print("ffmpeg not found. Installing with OpenCV VideoWriter fallback...")
        success = False
        
        # Fallback to OpenCV if ffmpeg not available
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Error: Could not initialize VideoWriter!")
            return None
        
        # Re-read and write frames
        for i in range(frame_count):
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.jpg")
            frame = cv2.imread(frame_path)
            out.write(frame)
        
        out.release()
        success = True
        print(f"Video created with OpenCV: {output_path}")
    
    # Clean up temp frames
    if success:
        print("Cleaning up temporary frames...")
        shutil.rmtree(temp_dir)
    
    if success:
        return {
            'output_path': output_path,
            'dimensions': {'width': width, 'height': height},
            'frames_processed': frame_count
        }
    else:
        print(f"\nFrames saved to: {temp_dir}")
        return None


if __name__ == "__main__":
    # Test - Load boundary data and apply mask
    from main import load_boundary_data
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from config import VIDEO_FILES, ASSETS_DIR, OUTPUT_DIR
    
    video_name = VIDEO_FILES[0]
    video_path = os.path.join(ASSETS_DIR, video_name)
    
    print(f"Video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found!")
        exit(1)
    
    # Load boundary data from previous detection
    video_base_name = video_name.replace('.mp4', '')
    output_dir = os.path.join(OUTPUT_DIR, video_base_name)
    
    boundary_data = load_boundary_data(output_dir)
    
    if not boundary_data:
        print(f"Error: No boundary_data.json found in {output_dir}")
        print("Please run main.py first to generate boundary data.")
        exit(1)
    
    # Extract boundary parameters
    master_left = boundary_data['master_left']
    master_right = boundary_data['master_right']
    median_foul_params = boundary_data['median_foul_params']
    
    print(f"\nUsing detected boundaries:")
    print(f"  Left boundary X: {master_left['x_intersect']}")
    print(f"  Right boundary X: {master_right['x_intersect']}")
    print(f"  Foul line Y: {median_foul_params['center_y']}")
    
    # Create output path
    output_path = os.path.join(output_dir, f'masked_{video_name}')
    
    # Apply mask
    print(f"\n{'='*70}")
    print("MASKING VIDEO TO LANE AREA ONLY")
    print(f"{'='*70}\n")
    
    result = apply_mask_to_video(
        video_path,
        output_path,
        master_left,
        master_right,
        median_foul_params
    )
    
    if result:
        print(f"\n{'='*70}")
        print(f"SUCCESS! Masked video saved to:")
        print(f"  {result['output_path']}")
        print(f"{'='*70}")
