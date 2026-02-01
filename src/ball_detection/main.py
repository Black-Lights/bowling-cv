"""
Main entry point for bowling ball detection - Phase 2.

Currently implements:
- Step 1: Video masking to focus on lane area
- Step 2: Perspective transformation to overhead view
- Step 3: Motion detection (background subtraction)

Version: 1.1.0
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
from ball_detection.motion_detection import save_motion_detection_videos


def main():
    """
    Main entry point for ball detection pipeline.
    
    Currently implements:
    - Step 1: Create masked video (lane area only)
    - Step 2: Create perspective-corrected video (overhead view)
    - Step 3: Motion detection with background subtraction (Stage B)
    
    Future steps:
    - Step 4: Ball detection (color/motion/hybrid)
    - Step 5: Ball tracking
    - Step 6: Trajectory analysis
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bowling Ball Detection Pipeline - Phase 2')
    parser.add_argument('--video', type=str, help='Process single video file')
    parser.add_argument('--skip-masking', action='store_true', help='Skip Step 1 (masked video)')
    parser.add_argument('--skip-transform', action='store_true', help='Skip Step 2 (transformed video)')
    parser.add_argument('--skip-motion', action='store_true', help='Skip Step 3 (motion detection)')
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
    print(f"# Step 3: Motion Detection (Background Subtraction)")
    print(f"# Processing {len(videos)} video(s)")
    print(f"{'#'*80}\n")
    
    # Process each video
    for video_file in videos:
        try:
            # Get output directory for this video
            video_name = Path(video_file).stem
            output_base_dir = config.OUTPUT_DIR / Path(video_name)
            
            # Step 1: Create masked video (if enabled and not skipped)
            if not args.skip_masking and config.SAVE_MASKED_VIDEO:
                result = create_masked_lane_video(video_file, config, save_video=True)
                if not result:
                    print(f"✗ Failed to create masked video: {video_file}\n")
            
            # Step 2: Create perspective-corrected video (if enabled and not skipped)
            if not args.skip_transform and config.SAVE_TRANSFORMED_VIDEO:
                result = create_transformed_video(video_file, config)
                if not result:
                    print(f"✗ Failed to create transformed video: {video_file}\n")
            
            # Step 3: Motion detection with background subtraction (Stage B)
            if not args.skip_motion:
                print(f"\n{'='*60}")
                print(f"Step 3: Motion Detection - {video_file}")
                print(f"{'='*60}")
                
                result = save_motion_detection_videos(video_file, config, str(output_base_dir))
                
                if result:
                    print(f"\n✓ Motion detection complete for {video_file}")
                    print(f"  Generated {len(result)} videos:")
                    for name, path in result.items():
                        print(f"    - {name}: {Path(path).name}")
                else:
                    print(f"✗ Failed motion detection: {video_file}\n")
                
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
