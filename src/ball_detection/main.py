"""
Main entry point for bowling ball detection - Phase 2.

Currently implements Step 1: Video masking to focus on lane area.

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


def main():
    """
    Main entry point for ball detection pipeline.
    
    Currently implements:
    - Step 1: Create masked video (lane area only)
    
    Future steps:
    - Step 2: Ball detection (color/motion/hybrid)
    - Step 3: Ball tracking
    - Step 4: Trajectory analysis
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
    print(f"# Processing {len(videos)} video(s)")
    print(f"{'#'*80}\n")
    
    # Process each video
    for video_file in videos:
        try:
            result = create_masked_lane_video(video_file, config)
            
            if not result:
                print(f"✗ Failed to process: {video_file}\n")
                continue
                
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
    print(f"# LANE MASKING COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == '__main__':
    main()
