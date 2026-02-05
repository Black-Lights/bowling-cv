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


def create_lane_mask(frame_shape, master_left, master_right, median_foul_params, top_boundary=None):
    """
    Create a binary mask for the bowling lane area.
    
    Parameters:
    -----------
    frame_shape : tuple
        (height, width) of the frame
    master_left : dict
        Left boundary parameters with x_intersect, x_top, x_bottom, and slope
    master_right : dict
        Right boundary parameters with x_intersect, x_top, x_bottom, and slope
    median_foul_params : dict
        Foul line parameters with center_y and slope
    top_boundary : dict, optional
        Top boundary parameters with y_position. If None, masks from frame top (y=0)
        
    Returns:
    --------
    mask : numpy.ndarray
        Binary mask (255 = keep, 0 = black out)
    """
    height, width = frame_shape[:2]
    
    # Create blank mask (all black)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Get bottom boundary position (keep at actual foul line for polygon shape)
    foul_y = int(median_foul_params['center_y'])
    
    # Determine top boundary position
    if top_boundary is not None:
        # Use detected top boundary (4-side masking for Phase 2)
        top_y = int(top_boundary['y_position'])
        # Calculate x positions where master lines intersect the top boundary
        # Using line equation: x = x_intersect + (y - y_intersect) / slope
        left_x_intersect = master_left['x_intersect']
        right_x_intersect = master_right['x_intersect']
        left_slope = master_left.get('slope', 0)
        right_slope = master_right.get('slope', 0)
        
        # Calculate x at top_y using the line equation from foul line
        left_x_top = int(left_x_intersect + (top_y - foul_y) / left_slope) if left_slope != 0 else left_x_intersect
        right_x_top = int(right_x_intersect + (top_y - foul_y) / right_slope) if right_slope != 0 else right_x_intersect
    else:
        # Use frame top (3-side masking for Phase 1)
        top_y = 0
        # Calculate x positions at frame top using slope
        left_x_intersect = master_left['x_intersect']
        right_x_intersect = master_right['x_intersect']
        left_slope = master_left.get('slope', 0)
        right_slope = master_right.get('slope', 0)
        
        left_x_top = int(left_x_intersect + (0 - foul_y) / left_slope) if left_slope != 0 else left_x_intersect
        right_x_top = int(right_x_intersect + (0 - foul_y) / right_slope) if right_slope != 0 else right_x_intersect
    
    # Get bottom boundary positions at foul line
    # Use x_intersect since that's the x position at the foul line (y=center_y)
    left_x_foul = master_left['x_intersect']
    right_x_foul = master_right['x_intersect']
    
    # Create polygon points (clockwise from top-left)
    polygon_points = np.array([
        [left_x_top, top_y],           # Top-left
        [right_x_top, top_y],          # Top-right
        [right_x_foul, foul_y],        # Bottom-right (at foul line)
        [left_x_foul, foul_y]          # Bottom-left (at foul line)
    ], dtype=np.int32)
    
    # Fill the polygon with white (255 = keep this area)
    cv2.fillPoly(mask, [polygon_points], 255)
    
    # Black out everything BELOW the foul line (including foul line markers)
    # This ensures foul line dots are completely masked out
    mask_cutoff = int(foul_y - 30)  # Start masking 30 pixels above foul line
    mask[mask_cutoff:, :] = 0  # Set all rows from cutoff to bottom as black
    
    return mask


def get_masked_frames_generator(video_path, master_left, master_right, 
                               median_foul_params, top_boundary=None, mask_color=(0, 0, 0)):
    """
    Generator that yields masked frames without creating a video file.
    Memory-efficient for processing frames directly without saving.
    
    Parameters:
    -----------
    video_path : str
        Path to input video
    master_left : dict
        Left boundary parameters
    master_right : dict
        Right boundary parameters
    median_foul_params : dict
        Foul line parameters
    top_boundary : dict, optional
        Top boundary parameters. If None, masks from frame top (3-side masking)
    mask_color : tuple
        Color for masked areas (default: black)
        
    Yields:
    -------
    tuple : (frame_index, masked_frame, metadata)
        frame_index: int - Frame number (0-indexed)
        masked_frame: numpy.ndarray - Masked frame
        metadata: dict - Video info (fps, width, height, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    metadata = {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames
    }
    
    # Read first frame to create mask
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame")
    
    # Create the mask once
    mask = create_lane_mask(first_frame.shape, master_left, master_right, median_foul_params, top_boundary)
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Yield masked frames
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply mask
        masked_frame = frame.copy()
        masked_frame[mask == 0] = mask_color
        
        yield (frame_index, masked_frame, metadata)
        frame_index += 1
    
    cap.release()


def apply_mask_to_video(video_path, output_path, master_left, master_right, 
                       median_foul_params, top_boundary=None, mask_color=(0, 0, 0),
                       save_video=True):
    """
    Apply lane mask to video, optionally saving to file or just returning frames.
    
    Parameters:
    -----------
    video_path : str
        Path to input video
    output_path : str
        Path to save masked video (used only if save_video=True)
    master_left : dict
        Left boundary parameters
    master_right : dict
        Right boundary parameters
    median_foul_params : dict
        Foul line parameters
    top_boundary : dict, optional
        Top boundary parameters. If None, masks from frame top (3-side masking)
    mask_color : tuple
        Color for masked areas (default: black)
    save_video : bool
        If True, saves video file. If False, returns generator for frames.
        
    Returns:
    --------
    dict or generator:
        If save_video=True: dict with video information
        If save_video=False: generator yielding (frame_index, masked_frame, metadata)
    """
    # If not saving video, return generator
    if not save_video:
        return get_masked_frames_generator(video_path, master_left, master_right, 
                                          median_foul_params, top_boundary, mask_color)
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
    mask = create_lane_mask(first_frame.shape, master_left, master_right, median_foul_params, top_boundary)
    
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
