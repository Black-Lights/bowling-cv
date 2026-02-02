"""
Video Masking Utilities for Ball Detection

Creates masked video focusing only on the bowling lane area
using boundary data from Phase 1.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 1, 2026
"""

import os
import json
from pathlib import Path

# Reuse Phase 1 masking function (now supports 4-side masking)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from lane_detection.mask_lane_area import apply_mask_to_video


def create_masked_lane_video(video_path: str, config, save_video=None):
    """
    Create a masked video or return masked frames generator.
    
    Uses boundary data from Phase 1 to mask out everything outside
    the lane boundaries (left, right, top, bottom).
    
    Args:
        video_path (str): Path to input video
        config: Configuration module with settings
        save_video (bool, optional): If True, saves video. If False, returns frame generator.
                                     If None, uses config.SAVE_MASKED_VIDEO
        
    Returns:
        dict or generator:
            If save_video=True: dict with output_path and metadata
            If save_video=False: generator yielding (frame_index, masked_frame, metadata)
        
    Raises:
        FileNotFoundError: If video or boundary data not found
    """
    # Determine whether to save video
    if save_video is None:
        save_video = config.SAVE_MASKED_VIDEO
    
    # Validate video path
    if not os.path.isabs(video_path):
        assets_dir = getattr(config, 'ASSETS_DIR', os.getcwd())
        video_path = os.path.join(assets_dir, video_path)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Setup output directory
    video_name = Path(video_path).stem
    output_dir = os.path.join(config.OUTPUT_DIR, video_name)
    
    # Create ball_detection subdirectory
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
        if save_video:
            print(f"Creating Masked Lane Video")
        else:
            print(f"Preparing Masked Frame Generator")
        print(f"Video: {video_name}")
        print(f"{'='*80}\n")
        print(f"Loaded boundary data from Phase 1")
        print(f"  - Left boundary: x={boundary_data['master_left']['x_intersect']}")
        print(f"  - Right boundary: x={boundary_data['master_right']['x_intersect']}")
        print(f"  - Top boundary: y={boundary_data['top_boundary']['y_position']:.1f}")
        print(f"  - Bottom boundary (foul): y={boundary_data['median_foul_params']['center_y']}")
    
    # Determine output path
    output_path = os.path.join(intermediate_dir, f'{video_name}_lane_masked.mp4') if save_video else None
    
    if config.VERBOSE:
        if save_video:
            print(f"\nMasking video to lane area only (all 4 boundaries)...")
        else:
            print(f"\nCreating frame generator (no video file will be saved)...")
    
    # Apply mask using Phase 1 function
    result = apply_mask_to_video(
        video_path,
        output_path,
        boundary_data['master_left'],
        boundary_data['master_right'],
        boundary_data['median_foul_params'],
        top_boundary=boundary_data['top_boundary'],  # Include top boundary for 4-side masking
        save_video=save_video
    )
    
    # If not saving video, result is a generator - wrap it with metadata
    if not save_video:
        if config.VERBOSE:
            print(f">>> Frame generator ready")
            print(f"  Use: for idx, frame, meta in generator:")
            print(f"       # Process frame directly without video I/O")
        
        # Return the generator with boundary data accessible via closure
        return result
    
    # If saving video, result is a dict
    if result:
        if config.VERBOSE:
            print(f"\n{'='*80}")
            print(f">>> Success!")
            print(f"{'='*80}")
            print(f"Masked video saved to:")
            print(f"  {output_path}")
            print(f"\nNext steps:")
            print(f"  - Review the masked video")
            print(f"  - Proceed with ball detection implementation")
            print(f"{'='*80}\n")
        
        return {
            'output_path': output_path,
            'video_name': video_name,
            'boundary_data': boundary_data,
            **result
        }
    else:
        if config.VERBOSE:
            print("ERROR: Could not create masked video")
        return None


if __name__ == "__main__":
    # Test masking
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import config
    
    print(f"\n{'#'*80}")
    print(f"# LANE MASKING TEST - Phase 2")
    print(f"{'#'*80}\n")
    
    # Process first video
    video_file = config.VIDEO_FILES[0]
    
    try:
        result = create_masked_lane_video(video_file, config)
        
        if result:
            print("\n>>> Masking test successful!")
        else:
            print("\nERROR: Masking test failed!")
            
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
