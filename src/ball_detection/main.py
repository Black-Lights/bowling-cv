"""
Main entry point for bowling ball detection - Phase 2.

Currently implements:
- Step 1: Video masking to focus on lane area
- Step 2: Perspective transformation to overhead view
- Step 3: Motion detection (background subtraction)
- Step 4: ROI logic (dual-mode search strategy with Kalman filter)

Version: 1.2.0
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
from ball_detection.roi_visualization import generate_roi_videos
from ball_detection.blob_analysis import BlobAnalyzer
from ball_detection.blob_visualization import generate_blob_videos


def main():
    """
    Main entry point for ball detection pipeline.
    
    Currently implements:
    - Step 1: Create masked video (lane area only)
    - Step 2: Create perspective-corrected video (overhead view)
    - Step 3: Motion detection with background subtraction (Stage B)
    - Step 4: ROI logic with Kalman filter tracking (Stage C)
    - Step 5: Blob analysis and filtering (Stage D)
    
    Future steps:
    - Step 6: Trajectory analysis and export
    - Step 7: Spin/rotation analysis
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bowling Ball Detection Pipeline - Phase 2')
    parser.add_argument('--video', type=str, help='Process single video file')
    parser.add_argument('--skip-masking', action='store_true', help='Skip Step 1 (masked video)')
    parser.add_argument('--skip-transform', action='store_true', help='Skip Step 2 (transformed video)')
    parser.add_argument('--skip-motion', action='store_true', help='Skip Step 3 (motion detection)')
    parser.add_argument('--skip-roi', action='store_true', help='Skip Step 4 (ROI logic)')
    parser.add_argument('--skip-blob', action='store_true', help='Skip Step 5 (blob analysis)')
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
    print(f"# Step 4: ROI Logic (Kalman Filter Tracking)")
    print(f"# Step 5: Blob Analysis & Filtering")
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
            
            # Step 4: ROI Logic with Kalman filter tracking (Stage C)
            if not args.skip_roi:
                print(f"\n{'='*60}")
                print(f"Step 4: ROI Logic & Tracking - {video_file}")
                print(f"{'='*60}")
                
                result = generate_roi_videos(video_file, config, str(output_base_dir))
                
                if result:
                    print(f"\n✓ ROI tracking complete for {video_file}")
                    print(f"  Generated {len(result)} videos:")
                    for name, path in result.items():
                        print(f"    - {name}: {Path(path).name}")
                else:
                    print(f"✗ Failed ROI tracking: {video_file}\n")
            
            # Step 5: Integrated ROI Tracking + Blob Analysis (Stage C+D)
            if not args.skip_roi and not args.skip_blob:
                print(f"\n{'='*60}")
                print(f"Step 5: ROI Tracking + Blob Analysis - {video_file}")
                print(f"{'='*60}")
                
                # Import dependencies
                import cv2
                import numpy as np
                import json
                from ball_detection.mask_video import create_masked_lane_video
                from ball_detection.motion_detection import apply_background_subtraction
                from ball_detection.roi_logic import BallTracker
                
                # Get video path
                video_path = Path(config.ASSETS_DIR) / video_file
                if not video_path.exists():
                    print(f"✗ Video not found: {video_path}")
                    continue
                    
                # Load boundary data
                boundary_path = output_base_dir / 'boundary_data.json'
                if not boundary_path.exists():
                    print(f"✗ Boundary data not found: {boundary_path}")
                    print("  Run Phase 1 first to generate boundary data")
                    continue
                    
                with open(boundary_path, 'r') as f:
                    boundary_data = json.load(f)
                
                foul_line_y = boundary_data['median_foul_params']['center_y']
                
                # Get video dimensions
                cap = cv2.VideoCapture(str(video_path))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # Initialize tracker and blob analyzer
                tracker = BallTracker(config, width, height, foul_line_y)
                analyzer = BlobAnalyzer(config)
                
                print("Processing frames with ROI tracking + blob filtering...")
                
                # Get masked frames generator
                masked_frames_gen = create_masked_lane_video(video_file, config, save_video=False)
                
                # Apply motion detection
                motion_gen = apply_background_subtraction(masked_frames_gen, config, save_videos=False)
                
                frames = []
                masks = []
                roi_results = []
                blob_metrics_list = []
                
                for frame_idx, denoised_mask, metadata, intermediate in motion_gen:
                    frame = intermediate['original_masked']
                    
                    # Stage C: ROI tracking
                    roi_result = tracker.process_frame(denoised_mask, frame_idx)
                    
                    # Stage D: Blob filtering within ROI
                    if roi_result['roi_box'] is not None:
                        # Extract ROI region
                        x1, y1, x2, y2 = roi_result['roi_box']
                        roi_mask = np.zeros_like(denoised_mask)
                        roi_mask[y1:y2, x1:x2] = denoised_mask[y1:y2, x1:x2]
                        
                        # Filter blobs only in ROI
                        blob_metrics = analyzer.filter_blobs(frame, roi_mask, frame_idx)
                    else:
                        # No ROI - analyze full frame (global search)
                        blob_metrics = analyzer.filter_blobs(frame, denoised_mask, frame_idx)
                    
                    frames.append(frame)
                    masks.append(denoised_mask)
                    roi_results.append(roi_result)
                    blob_metrics_list.append(blob_metrics)
                
                # Generate visualization videos
                blob_output_dir = str(output_base_dir / 'ball_detection' / 'intermediate')
                generate_blob_videos(video_name, frames, masks, blob_metrics_list, 
                                   analyzer, blob_output_dir, config)
                
                print(f"\n✓ ROI tracking + Blob analysis complete for {video_file}")
                
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
