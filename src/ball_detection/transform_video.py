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
    
    Applies homography transformation to MASKED frames to convert tilted 
    lane view to overhead rectangular view matching real-world dimensions.
    
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
        scale_width = getattr(config, 'TRANSFORM_SCALE_WIDTH', config.TRANSFORM_SCALE if hasattr(config, 'TRANSFORM_SCALE') else 10)
        scale_height = getattr(config, 'TRANSFORM_SCALE_HEIGHT', config.TRANSFORM_SCALE if hasattr(config, 'TRANSFORM_SCALE') else 10)
        output_width = int(LANE_WIDTH_INCHES * scale_width)
        output_height = int(LANE_LENGTH_INCHES * scale_height)
        # Ensure divisible by 2
        if output_width % 2 != 0:
            output_width += 1
        if output_height % 2 != 0:
            output_height += 1
        print(f"  Scale: {scale_width} px/in (width), {scale_height} px/in (height)")
        print(f"  Width: {output_width} pixels ({LANE_WIDTH_INCHES} inches)")
        print(f"  Height: {output_height} pixels ({LANE_LENGTH_INCHES} inches)")
        print(f"  Aspect ratio: {output_width}:{output_height} (~1:{output_height/output_width:.1f})")
    
    # Get masked frames generator (don't save video, just get frames)
    if config.VERBOSE:
        print(f"\nGetting masked frames for transformation...")
    
    # Import mask_video here to avoid circular import
    from ball_detection.mask_video import create_masked_lane_video
    
    # Get frame generator (save_video=False for memory efficiency)
    masked_frames_gen = create_masked_lane_video(video_path, config, save_video=False)
    
    # Get scale factor from config (same for both dimensions to preserve aspect ratio)
    scale = getattr(config, 'TRANSFORM_SCALE', 10)
    auto_crop = getattr(config, 'AUTO_CROP_TRANSFORMED', True)
    
    output_width = int(LANE_WIDTH_INCHES * scale)
    output_height = int(LANE_LENGTH_INCHES * scale)
    
    # Ensure dimensions are divisible by 2 for video encoding
    if output_width % 2 != 0:
        output_width += 1
    if output_height % 2 != 0:
        output_height += 1
    
    if config.VERBOSE:
        print(f"\nOutput dimensions:")
        print(f"  Scale: {scale} px/in (uniform - preserves shapes)")
        print(f"  Width: {output_width} pixels ({LANE_WIDTH_INCHES} inches)")
        print(f"  Height: {output_height} pixels ({LANE_LENGTH_INCHES} inches)")
        print(f"  Aspect ratio: {output_width}:{output_height} (~{output_width/output_height:.1f}:1)")
        print(f"  Auto-crop: {'Enabled' if auto_crop else 'Disabled'}")
        print()
        print(f"Transforming masked frames...")
    
    # Create temp directory for frames
    output_path = os.path.join(intermediate_dir, f'{video_name}_transformed.mp4')
    temp_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get metadata from first frame
    first_frame_data = next(masked_frames_gen)
    frame_idx, first_masked_frame, metadata = first_frame_data
    fps = metadata['fps']
    total_frames = metadata['total_frames']
    
    # Transform first frame (auto_crop already defined above)
    transformed = apply_perspective_transform(first_masked_frame, H, output_width, output_height, auto_crop=auto_crop)
    
    # Get actual dimensions
    actual_height, actual_width = transformed.shape[:2]
    
    if config.VERBOSE:
        if auto_crop:
            print(f"  Cropped dimensions: {actual_width}x{actual_height} (removed black borders)")
        else:
            print(f"  Output dimensions: {actual_width}x{actual_height}")
    
    frame_path = os.path.join(temp_dir, f"frame_{frame_idx:05d}.png")
    cv2.imwrite(frame_path, transformed)  # PNG is lossless, no quality parameter needed
    frame_count = 1
    
    # Transform remaining frames with progress bar
    for frame_idx, masked_frame, _ in tqdm(masked_frames_gen, 
                                           desc="Transforming frames", 
                                           total=total_frames-1,
                                           disable=not config.VERBOSE):
        # Apply perspective transformation
        transformed = apply_perspective_transform(masked_frame, H, output_width, output_height, auto_crop=auto_crop)
        
        # Save frame as PNG (lossless)
        frame_path = os.path.join(temp_dir, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(frame_path, transformed)
        frame_count += 1
    
    if config.VERBOSE:
        print(f"Transformed {frame_count} frames")
        print(f"\nCombining frames with ffmpeg...")
    
    # Use ffmpeg to combine frames into video with high quality encoding
    # yuv444p prevents chroma subsampling artifacts (purple patches) on narrow videos
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%05d.png'),  # PNG input (lossless)
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv444p',  # No chroma subsampling (prevents purple artifacts)
        '-preset', 'veryslow',  # Best compression quality
        '-crf', '15',  # Near-lossless quality
        '-profile:v', 'high444',  # Required for yuv444p
        '-tune', 'film',  # Optimize for high quality video content
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
        out = cv2.VideoWriter(output_path, fourcc, fps, (actual_width, actual_height))
        
        if not out.isOpened():
            print("Error: Could not initialize VideoWriter!")
            return None
        
        # Re-read and write frames
        for i in range(frame_count):
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
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
                'width': actual_width,
                'height': actual_height,
                'width_inches': LANE_WIDTH_INCHES,
                'height_inches': LANE_LENGTH_INCHES
            },
            'fps': fps,
            'frame_count': frame_count
        }
    else:
        return None
