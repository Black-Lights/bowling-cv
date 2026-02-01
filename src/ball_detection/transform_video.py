"""
Perspective Transformation Utilities

Creates perspective-corrected video of the bowling lane using homography matrix.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 1, 2026
"""

import os
import cv2
import json
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm

# Import homography utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ball_detection.homography import (
    calculate_homography, 
    apply_perspective_transform,
    save_homography_data,
    LANE_WIDTH_INCHES,
    LANE_LENGTH_INCHES
)


def create_transformed_video(video_path: str, config):
    """
    Create perspective-corrected video of the bowling lane.
    
    Applies homography transformation to convert tilted lane view to
    overhead rectangular view matching real-world dimensions.
    
    Args:
        video_path (str): Path to input video
        config: Configuration module with settings
        
    Returns:
        dict: Results with output_path and metadata
        
    Raises:
        FileNotFoundError: If video or boundary data not found
    """
    # Validate video path
    if not os.path.isabs(video_path):
        assets_dir = getattr(config, 'ASSETS_DIR', os.getcwd())
        video_path = os.path.join(assets_dir, video_path)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Setup output directory
    video_name = Path(video_path).stem
    output_dir = os.path.join(config.OUTPUT_DIR, video_name)
    ball_detection_dir = os.path.join(output_dir, 'ball_detection')
    intermediate_dir = os.path.join(ball_detection_dir, 'intermediate')
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Load boundary data from Phase 1
    boundary_file = os.path.join(output_dir, 'boundary_data.json')
    if not os.path.exists(boundary_file):
        raise FileNotFoundError(
            f"Boundary data not found: {boundary_file}\n"
            f"Please run Phase 1 (lane detection) first!"
        )
    
    with open(boundary_file, 'r') as f:
        boundary_data = json.load(f)
    
    if config.VERBOSE:
        print(f"\n{'='*80}")
        print(f"Creating Perspective-Corrected Video")
        print(f"Video: {video_name}")
        print(f"{'='*80}\n")
    
    # Calculate homography matrix
    if config.VERBOSE:
        print("Calculating homography matrix...")
    
    H, corners_image, corners_real = calculate_homography(boundary_data)
    
    # Save homography data
    save_homography_data(ball_detection_dir, H, corners_image, corners_real)
    
    if config.VERBOSE:
        print(f"✓ Homography matrix calculated")
        print(f"\nImage corners (pixels):")
        for i, corner in enumerate(corners_image):
            labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
            print(f"  {labels[i]:12}: ({corner[0]:6.1f}, {corner[1]:6.1f})")
        
        print(f"\nReal-world corners (inches):")
        for i, corner in enumerate(corners_real):
            labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
            print(f"  {labels[i]:12}: ({corner[0]:6.1f}, {corner[1]:6.1f})")
        
        print(f"\nOutput dimensions:")
        scale = config.TRANSFORM_SCALE
        output_width = int(LANE_WIDTH_INCHES * scale)
        output_height = int(LANE_LENGTH_INCHES * scale)
        print(f"  Scale: {scale} pixels/inch")
        print(f"  Width: {output_width} pixels ({LANE_WIDTH_INCHES} inches)")
        print(f"  Height: {output_height} pixels ({LANE_LENGTH_INCHES} inches)")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate output dimensions
    scale = config.TRANSFORM_SCALE
    output_width = int(LANE_WIDTH_INCHES * scale)
    output_height = int(LANE_LENGTH_INCHES * scale)
    
    # Ensure dimensions are divisible by 2 for video encoding
    if output_width % 2 != 0:
        output_width += 1
    if output_height % 2 != 0:
        output_height += 1
    
    if config.VERBOSE:
        print(f"\nTransforming {total_frames} frames...")
    
    # Create temp directory for frames
    output_path = os.path.join(intermediate_dir, f'{video_name}_transformed.mp4')
    temp_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Transform frames
    frame_count = 0
    for i in tqdm(range(total_frames), desc="Transforming frames", disable=not config.VERBOSE):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply perspective transformation
        transformed = apply_perspective_transform(frame, H, output_width, output_height)
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{i:05d}.jpg")
        cv2.imwrite(frame_path, transformed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_count += 1
    
    cap.release()
    
    if config.VERBOSE:
        print(f"Transformed {frame_count} frames")
        print(f"\nCombining frames with ffmpeg...")
    
    # Use ffmpeg to combine frames into video
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
        if config.VERBOSE:
            print(f"Video created: {output_path}")
        success = True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr}")
        success = False
    except FileNotFoundError:
        print("ffmpeg not found. Using OpenCV VideoWriter fallback...")
        
        # Fallback to OpenCV if ffmpeg not available
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        if not out.isOpened():
            print("Error: Could not initialize VideoWriter!")
            cap.release()
            return None
        
        # Re-read and write frames
        for i in range(frame_count):
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.jpg")
            frame = cv2.imread(frame_path)
            out.write(frame)
        
        out.release()
        success = True
        if config.VERBOSE:
            print(f"Video created with OpenCV: {output_path}")
    
    # Clean up temp frames
    if success:
        if config.VERBOSE:
            print("Cleaning up temporary frames...")
        shutil.rmtree(temp_dir)
    
    if success and config.VERBOSE:
        print(f"\n{'='*80}")
        print(f"✓ Success!")
        print(f"{'='*80}")
        print(f"Transformed video saved to:")
        print(f"  {output_path}")
        print(f"\nHomography data saved to:")
        print(f"  {os.path.join(ball_detection_dir, 'homography_data.json')}")
        print(f"\nThe video shows overhead view of the lane:")
        print(f"  - Corrected perspective (no tilt)")
        print(f"  - Real-world proportions (60ft x 41.5in)")
        print(f"  - Ready for accurate ball tracking")
        print(f"{'='*80}\n")
    
    if success:
        return {
            'output_path': output_path,
            'video_name': video_name,
            'homography_matrix': H.tolist(),
            'dimensions': {
                'width': output_width,
                'height': output_height,
                'width_inches': LANE_WIDTH_INCHES,
                'height_inches': LANE_LENGTH_INCHES
            },
            'fps': fps,
            'frame_count': frame_count
        }
    else:
        return None
