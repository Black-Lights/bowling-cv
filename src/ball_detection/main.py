"""
Main entry point for bowling ball detection - Phase 2.

Currently implements:
- Step 1: Video masking to focus on lane area
- Step 2: Perspective transformation to overhead view

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Last Updated: February 1, 2026

Usage:
    python -m src.ball_detection.main
    
    Or process a single video:
    python -m src.ball_detection.main --video cropped_test3.mp4
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ball_detection import config
from ball_detection.mask_video import create_masked_lane_video
from ball_detection.transform_video import create_transformed_video


def main():
    """
    Main entry point for ball detection pipeline.
    
    Currently implements:
    - Step 1: Create masked video (lane area only)
    - Step 2: Create perspective-corrected video (overhead view)
    
    Future steps:
    - Step 3: Ball detection (color/motion/hybrid)
    - Step 4: Ball tracking
    - Step 5: Trajectory analysis
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bowling Ball Detection Pipeline - Phase 2')
    parser.add_argument('--video', type=str, help='Process single video file')
    args = parser.parse_args()
    
    # Determine which videos to process
    if args.video:
        videos = [args.video]
    else:
        videos = config.VIDEO_FILES
    
    print(f"\n{'#'*80}")
    print(f"# BOWLING BALL DETECTION - PHASE 2")
    print(f"# Step 1: Lane Masking")
    print(f"# Step 2: Perspective Transformation")
    print(f"# Processing {len(videos)} video(s)")
    print(f"{'#'*80}\n")
    
    # Process each video
    for video_file in videos:
        try:
            # Step 1: Create masked video (if enabled)
            if config.SAVE_MASKED_VIDEO:
                result = create_masked_lane_video(video_file, config, save_video=True)
                if not result:
                    print(f"✗ Failed to create masked video: {video_file}\n")
            
            # Step 2: Create perspective-corrected video (if enabled)
            if config.SAVE_TRANSFORMED_VIDEO:
                result = create_transformed_video(video_file, config)
                if not result:
                    print(f"✗ Failed to create transformed video: {video_file}\n")
                
        except FileNotFoundError as e:
            print(f"\n✗ Error processing {video_file}:")
            print(f"  {e}\n")
            continue
        except Exception as e:
            print(f"\n✗ Error processing {video_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'#'*80}")
    print(f"# BALL DETECTION PIPELINE COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == '__main__':
    main()
