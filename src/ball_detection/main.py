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
            # Respects config.MASK_TOP_BOUNDARY setting (3 or 4 boundary masking)
            if not args.skip_masking and config.SAVE_MASKED_VIDEO:
                result = create_masked_lane_video(video_file, config, save_video=True)
                if not result:
                    print(f"ERROR: Failed to create masked video: {video_file}\n")
            
            # Step 2: Create perspective-corrected video (if enabled and not skipped)
            if not args.skip_transform and config.SAVE_TRANSFORMED_VIDEO:
                result = create_transformed_video(video_file, config)
                if not result:
                    print(f"ERROR: Failed to create transformed video: {video_file}\n")
            
            # Step 3: Motion detection with background subtraction (Stage B)
            if not args.skip_motion:
                print(f"\n{'='*60}")
                print(f"Step 3: Motion Detection - {video_file}")
                print(f"{'='*60}")
                
                result = save_motion_detection_videos(video_file, config, str(output_base_dir))
                
                if result:
                    print(f"\n>>> Motion detection complete for {video_file}")
                    print(f"  Generated {len(result)} videos:")
                    for name, path in result.items():
                        print(f"    - {name}: {Path(path).name}")
                else:
                    print(f"ERROR: Failed motion detection: {video_file}\n")
            
            # Step 4: ROI Logic with Kalman filter tracking (Stage C) - LEGACY VISUALIZATION
            # This step generates visualization videos using the OLD separate Stage C approach
            # For actual ball detection, use Step 5 (integrated C+D+E)
            if not args.skip_roi and args.skip_blob:
                print(f"\n{'='*60}")
                print(f"Step 4: ROI Logic & Tracking (Legacy) - {video_file}")
                print(f"{'='*60}")
                print("Note: This is legacy ROI-only visualization.")
                print("For full ball detection, don't skip blob analysis (Step 5).")
                
                result = generate_roi_videos(video_file, config, str(output_base_dir))
                
                if result:
                    print(f"\n>>> ROI tracking complete for {video_file}")
                    print(f"  Generated {len(result)} videos:")
                    for name, path in result.items():
                        print(f"    - {name}: {Path(path).name}")
                else:
                    print(f"ERROR: Failed ROI tracking: {video_file}\n")
            
            # Step 5: Integrated Ball Detection (Stage C+D+E) - NEW ARCHITECTURE
            # This is the main ball detection step using Tracking-by-Detection
            # Filter ALL → Select based on state → Update Kalman
            if not args.skip_blob:
                print(f"\n{'='*60}")
                print(f"Step 5: Integrated Ball Detection (Stages C+D+E)")
                print(f"{'='*60}")
                
                # Import dependencies
                import cv2
                import numpy as np
                import json
                from ball_detection.motion_detection import apply_background_subtraction
                from ball_detection.roi_logic import BallTracker
                # Note: create_masked_lane_video already imported at top of file
                
                # Get video path
                video_path = Path(config.ASSETS_DIR) / video_file
                if not video_path.exists():
                    print(f"ERROR: Video not found: {video_path}")
                    continue
                    
                # Load boundary data
                boundary_path = output_base_dir / 'boundary_data.json'
                if not boundary_path.exists():
                    print(f"ERROR: Boundary data not found: {boundary_path}")
                    print("  Run Phase 1 first to generate boundary data")
                    continue
                    
                with open(boundary_path, 'r') as f:
                    boundary_data = json.load(f)
                
                foul_line_y = boundary_data['median_foul_params']['center_y']
                top_boundary_y = boundary_data.get('top_boundary', {}).get('y_position', None)
                
                # Get video dimensions
                cap = cv2.VideoCapture(str(video_path))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # Initialize blob analyzer and tracker (integrated)
                analyzer = BlobAnalyzer(config)
                tracker = BallTracker(config, width, height, foul_line_y, blob_analyzer=analyzer)
                
                # Set top boundary for stop condition (Stage F)
                if top_boundary_y is not None:
                    tracker.set_boundaries(top_boundary_y)
                else:
                    print("  Warning: Top boundary not found in boundary data. Stop condition disabled.")
                
                print("Architecture: Filter ALL (Stage D) -> Select (Stage C) -> Track (Stage E)")
                print("Processing frames with integrated tracking...")
                
                # Get masked frames generator
                # Respects config.MASK_TOP_BOUNDARY (False = 3 boundaries, True = 4 boundaries)
                # Stage A (homography) always uses 4 boundaries regardless
                masked_frames_gen = create_masked_lane_video(video_file, config, save_video=False)
                
                # Apply motion detection
                motion_gen = apply_background_subtraction(masked_frames_gen, config, save_videos=False)
                
                frames = []
                masks = []
                tracking_results = []
                
                for frame_idx, denoised_mask, metadata, intermediate in motion_gen:
                    frame = intermediate['original_masked']
                    
                    # INTEGRATED: Tracker handles everything
                    # 1. Filters ALL candidates (Stage D full-frame)
                    # 2. Selects based on tracking state (Stage C)
                    # 3. Updates Kalman filter (Stage E)
                    result = tracker.process_frame(denoised_mask, frame_idx, frame)
                    
                    # Auto-calibrate blob analyzer in first frames
                    if not analyzer.is_calibrated:
                        # Get contours for calibration
                        contours, _ = cv2.findContours(denoised_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        analyzer.auto_calibrate(frame, contours, height, foul_line_y)
                    
                    frames.append(frame)
                    masks.append(denoised_mask)
                    tracking_results.append(result)
                    
                    # Logging progress
                    if frame_idx % 30 == 0:
                        print(f"  Frame {frame_idx}: Mode={result['mode']}, "
                              f"Candidates={result['candidates_count']}, "
                              f"Detected={'Yes' if result['detection'] else 'No'}")
                    
                    # STAGE F: Break loop if trajectory complete (ball reached pins)
                    if result.get('trajectory_complete', False):
                        print(f"\n  -> Trajectory complete at frame {frame_idx}. Stopping tracking.")
                        break
                
                print(f"\n>>> Integrated tracking complete for {video_file}")
                print(f"  Total frames processed: {len(frames)}")
                print(f"  Architecture: Tracking-by-Detection (Filter -> Select -> Track)")
                
                if tracking_results and tracking_results[-1].get('trajectory_complete', False):
                    print(f"  Trajectory: Complete (stopped at pin area)")
                    print(f"  Valid trajectory points: {len(tracker.trajectory)}")
                    if tracker.interpolated_points:
                        print(f"  Interpolated points: {len(tracker.interpolated_points)}")
                
                # Generate visualization videos
                from ball_detection.integrated_visualization import generate_integrated_tracking_videos
                
                integrated_output_dir = str(output_base_dir / 'ball_detection' / 'intermediate')
                result = generate_integrated_tracking_videos(
                    video_name, frames, masks, tracking_results, config, integrated_output_dir
                )
                
                if result:
                    print(f"\n>>> Visualization videos generated:")
                    for name, path in result.items():
                        print(f"    - {name}: {Path(path).name}")
                
                # Save trajectory points for post-processing
                print(f"\nSaving trajectory data...")
                from ball_detection.trajectory_plot import save_trajectory_points
                
                save_trajectory_points(
                    tracker, output_base_dir / 'ball_detection',
                    video_name, config
                )
                
                # Generate trajectory plots (original and overhead views)
                print(f"\nGenerating trajectory plots...")
                from ball_detection.trajectory_plot import plot_trajectory_on_original, plot_trajectory_on_overhead
                
                # Plot on original perspective view
                plot_trajectory_on_original(
                    tracker, frames, tracking_results,
                    output_base_dir / 'ball_detection', video_name, config
                )
                
                # Plot on transformed overhead view (with homography)
                plot_trajectory_on_overhead(
                    tracker, output_base_dir / 'ball_detection',
                    video_name, config
                )
                
        except FileNotFoundError as e:
            print(f"\nERROR: Error processing {video_file}:")
            print(f"  {e}\n")
            continue
        except Exception as e:
            print(f"\nERROR: Error processing {video_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'#'*80}")
    print(f"# BALL DETECTION PIPELINE COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == '__main__':
    main()
