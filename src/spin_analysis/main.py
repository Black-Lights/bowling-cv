"""
Main entry point for spin analysis - Phase 3

Currently implements:
- Stage A: Data Preparation & Trajectory Loading
- Stage B: Optical Flow Detection (with test visualizations)
- Stage C: 3D Projection & Feature Filtering

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026

Usage:
    python -m src.spin_analysis.main --video cropped_test3.mp4
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spin_analysis import config
from spin_analysis import utils
from spin_analysis import optical_flow
from spin_analysis import projection_3d


def run_stage_a(video_name: str, config) -> tuple:
    """
    Execute Stage A: Data Preparation & Trajectory Loading.
    
    Args:
        video_name: Name of video without extension
        config: Configuration module
    
    Returns:
        tuple: (trajectory_df, output_dir, video_path)
    """
    print(f"\n{'='*70}")
    print(f"STAGE A: DATA PREPARATION & TRAJECTORY LOADING")
    print(f"{'='*70}")
    
    # Setup output directory
    output_dir = utils.setup_output_directory(video_name, config)
    debug_dir = output_dir / config.DEBUG_SUBDIR
    
    # Load trajectory from ball detection output
    trajectory_df, trajectory_path = utils.load_trajectory_from_ball_detection(video_name, config)
    
    # Validate trajectory data
    stats = utils.validate_trajectory_data(trajectory_df, config)
    
    # Save prepared trajectory
    output_csv = debug_dir / config.STAGE_A_TRAJECTORY_CSV
    trajectory_df.to_csv(output_csv, index=False)
    if config.VERBOSE:
        print(f"[OK] Prepared trajectory saved: {output_csv.name}")
    
    # Create validation visualization (optional)
    if config.SAVE_DEBUG_IMAGES and config.GENERATE_STAGE_A_PLOT:
        utils.visualize_stage_a(trajectory_df, stats, debug_dir, config)
    
    # Get video path
    video_path = Path(config.ASSETS_DIR) / f"{video_name}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Get video info
    video_info = utils.get_video_info(str(video_path))
    if config.VERBOSE:
        print(f"\nVideo Information:")
        print(f"  Resolution: {video_info['width']}x{video_info['height']}")
        print(f"  FPS: {video_info['fps']:.2f}")
        print(f"  Total frames: {video_info['total_frames']}")
    
    print(f"\n{'='*70}")
    print(f"[COMPLETE] STAGE A")
    print(f"{'='*70}")
    print(f"  • Trajectory loaded: {len(trajectory_df)} frames")
    if config.GENERATE_STAGE_A_PLOT:
        print(f"  • Validation plot: {config.STAGE_A_VALIDATION_PLOT}")
    print(f"  • Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    return trajectory_df, output_dir, str(video_path)


def run_stage_b(video_path: str, trajectory_df, output_dir: Path, config):
    """
    Execute Stage B: Optical Flow Detection.
    
    Processes optical flow on frames and generates visualizations.
    - Full mode: Process all consecutive frame pairs
    - Test mode: Process only configured test frames
    
    Args:
        video_path: Path to video file
        trajectory_df: Trajectory DataFrame from Stage A
        output_dir: Output directory
        config: Configuration module
    """
    print(f"\n{'='*70}")
    if config.FULL_VIDEO_MODE:
        print(f"STAGE B: OPTICAL FLOW DETECTION (FULL VIDEO)")
    else:
        print(f"STAGE B: OPTICAL FLOW DETECTION (TEST PHASE)")
    print(f"{'='*70}")
    
    debug_dir = output_dir / config.DEBUG_SUBDIR
    
    # Determine which frames to process
    if config.FULL_VIDEO_MODE:
        # Process all consecutive frame pairs
        all_frames = sorted(trajectory_df['frame'].unique())
        frames_to_process = [f for f in all_frames if f + 1 in all_frames]
        if config.VERBOSE:
            print(f"\nProcessing optical flow on {len(frames_to_process)} consecutive frame pairs...")
            print(f"Frame range: {frames_to_process[0]} to {frames_to_process[-1]}")
    else:
        # Test mode: only process configured test frames
        frames_to_process = config.TEST_FRAMES
        if config.VERBOSE:
            print(f"\nTesting optical flow on {len(frames_to_process)} frame pairs...")
    
    # Process optical flow on selected frames
    test_results = []
    successful_frames = 0
    failed_frames = 0
    
    for test_frame in frames_to_process:
        if config.VERBOSE:
            print(f"Testing frame {test_frame}...")
        
        result = optical_flow.test_optical_flow_on_frame(
            video_path, trajectory_df, test_frame, config
        )
        
        if result is not None:
            successful_frames += 1
            
            # Print detailed statistics (only in test mode or verbose)
            if config.VERBOSE and not config.FULL_VIDEO_MODE:
                radius = result['ball_radius']
                
                # Calculate threshold being used
                if config.USE_EXPONENTIAL_THRESHOLD:
                    max_radius = 60.0
                    min_radius = 10.0
                    radius_normalized = (radius - min_radius) / (max_radius - min_radius)
                    radius_normalized = max(0.0, min(1.0, radius_normalized))
                    threshold = config.BASE_MOVEMENT_THRESHOLD * (radius_normalized ** config.MOVEMENT_THRESHOLD_DECAY)
                    threshold = max(0.5, threshold)
                    threshold_type = "exponential"
                else:
                    threshold = radius / config.LOW_THRESHOLD_FACTOR
                    threshold_type = "linear"
                
                # Calculate FB error stats
                fb_errors = result['fb_errors']
                fb_median = float(np.median(fb_errors))
                fb_mean = float(np.mean(fb_errors))
                
                print(f"    Ball: radius={radius:.1f}px")
                print(f"    Movement threshold: {threshold:.2f}px ({threshold_type})")
                print(f"    Features: {result['num_good']}/{result['total_features']} passed")
                print(f"    FB error: median={fb_median:.2f}px, mean={fb_mean:.2f}px")
            elif config.FULL_VIDEO_MODE and successful_frames % 10 == 0:
                # Progress update every 10 frames in full mode
                print(f"  Processed {successful_frames}/{len(frames_to_process)} frames...")
            
            test_results.append(result)
        else:
            failed_frames += 1
    
    # Generate visualization (only in test mode or for sample frames in full mode)
    if test_results and config.SAVE_DEBUG_IMAGES and config.GENERATE_STAGE_B_PLOT:
        if config.FULL_VIDEO_MODE:
            # In full mode, visualize only sample frames for debugging
            sample_indices = [0, len(test_results)//4, len(test_results)//2, 
                            3*len(test_results)//4, len(test_results)-1]
            sample_results = [test_results[i] for i in sample_indices if i < len(test_results)]
            optical_flow.visualize_optical_flow_test(sample_results, debug_dir, config)
        else:
            # Test mode: visualize all test frames
            optical_flow.visualize_optical_flow_test(test_results, debug_dir, config)
    
    # Create feature count summary plot
    if test_results and config.GENERATE_STAGE_B_SUMMARY:
        import matplotlib.pyplot as plt
        
        frames = [r['frame'] for r in test_results]
        good_counts = [r['num_good'] for r in test_results]
        total_counts = [r['total_features'] for r in test_results]
        
        plt.figure(figsize=(14, 6))
        
        # Plot 1: Tracked features count (all detected features)
        plt.subplot(1, 2, 1)
        plt.plot(frames, total_counts, 'o-', color='blue', linewidth=2, markersize=4, label='Tracked Features')
        plt.fill_between(frames, total_counts, alpha=0.3, color='blue')
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Number of Features', fontsize=12)
        plt.title('Tracked Features per Frame\n(All Detected by Optical Flow)', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Good features count (passed all filters)
        plt.subplot(1, 2, 2)
        plt.plot(frames, good_counts, 'o-', color='green', linewidth=2, markersize=4, label='Good Features')
        plt.fill_between(frames, good_counts, alpha=0.3, color='green')
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Number of Features', fontsize=12)
        plt.title('Good Features per Frame\n(Passed All Filters)', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.suptitle('Stage B: Optical Flow Feature Statistics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_plot_path = debug_dir / config.STAGE_B_FEATURE_SUMMARY
        plt.savefig(summary_plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        if config.VERBOSE:
            print(f"[OK] Stage B feature summary plot saved: {summary_plot_path.name}")
    
    # Summary statistics
    if test_results:
        total_features = sum(r['total_features'] for r in test_results)
        total_good = sum(r['num_good'] for r in test_results)
        avg_pass_rate = (total_good / total_features * 100) if total_features > 0 else 0
        
        print(f"\n{'='*70}")
        if config.FULL_VIDEO_MODE:
            print(f"[COMPLETE] STAGE B (FULL VIDEO)")
        else:
            print(f"[COMPLETE] STAGE B (TEST PHASE)")
        print(f"{'='*70}")
        print(f"  • Frames processed: {successful_frames}/{len(frames_to_process)}")
        if failed_frames > 0:
            print(f"  • Frames skipped: {failed_frames}")
        print(f"  • Total features detected: {total_features}")
        print(f"  • Total good features: {total_good}")
        print(f"  • Average pass rate: {avg_pass_rate:.1f}%")
        if config.FULL_VIDEO_MODE:
            print(f"  • Sample visualization: {config.STAGE_B_OPTICAL_FLOW_TEST}")
            print(f"  • Feature summary plot: {config.STAGE_B_FEATURE_SUMMARY}")
        else:
            print(f"  • Test visualization: {config.STAGE_B_OPTICAL_FLOW_TEST}")
            print(f"  • Feature summary plot: {config.STAGE_B_FEATURE_SUMMARY}")
        print(f"{'='*70}\n")
        
        # Quality check
        if avg_pass_rate < 20:
            print(f"⚠ WARNING: Low feature pass rate ({avg_pass_rate:.1f}%)")
            print(f"  Consider adjusting:")
            print(f"  - FB_ERROR_THRESHOLD (current: {config.FB_ERROR_THRESHOLD})")
            if config.USE_EXPONENTIAL_THRESHOLD:
                print(f"  - BASE_MOVEMENT_THRESHOLD (current: {config.BASE_MOVEMENT_THRESHOLD})")
                print(f"  - MOVEMENT_THRESHOLD_DECAY (current: {config.MOVEMENT_THRESHOLD_DECAY})")
            else:
                print(f"  - LOW_THRESHOLD_FACTOR (current: {config.LOW_THRESHOLD_FACTOR})")
            print(f"  - BALL_MASK_RADIUS_FACTOR (current: {config.BALL_MASK_RADIUS_FACTOR})")
        elif avg_pass_rate > 50:
            print(f"[OK] Good feature pass rate ({avg_pass_rate:.1f}%)")
            print(f"  Optical flow parameters are working well!")
    else:
        print(f"\n⚠ WARNING: No successful optical flow tests")
        print(f"  Check if test frames exist in trajectory")
    
    return test_results


def run_stage_c(test_results: list, trajectory_df, output_dir: Path, config):
    """
    Execute Stage C: 3D Projection & Feature Filtering.
    
    Projects 2D optical flow features onto 3D sphere surface.
    
    Args:
        test_results: Results from Stage B (optical flow)
        trajectory_df: Trajectory DataFrame
        output_dir: Output directory
        config: Configuration module
        
    Returns:
        list: 3D projection results
    """
    print(f"\n{'='*70}")
    print(f"STAGE C: 3D PROJECTION & FEATURE FILTERING")
    print(f"{'='*70}")
    
    debug_dir = output_dir / config.DEBUG_SUBDIR
    projection_results = []
    total_2d = 0
    total_3d = 0
    
    if config.VERBOSE:
        print(f"\nProjecting {len(test_results)} frame pairs to 3D...")
    
    for result in test_results:
        if result is None or result['num_good'] == 0:
            continue
        
        # Get ball centers (IMAGE coordinates from CSV) and radius
        # Also get ROI offsets to convert features from ROI coords to IMAGE coords
        ball_center_1 = result['ball_center1']  # CSV coordinates (consistent reference!)
        ball_center_2 = result['ball_center2']  # CSV coordinates
        ball_radius = result['ball_radius']
        roi_offset_1 = result['roi_offset1']    # To convert ROI → Image
        roi_offset_2 = result['roi_offset2']    # To convert ROI → Image
        
        # Process 3D projection
        # Features are in ROI coords, will be converted to image coords inside
        projection_result = projection_3d.process_frame_pair_3d(
            result['good_features'],
            ball_center_1,
            ball_center_2,
            ball_radius,
            roi_offset_1,
            roi_offset_2,
            config
        )
        
        # Add frame number
        projection_result['frame'] = result['frame']
        projection_results.append(projection_result)
        
        total_2d += projection_result['num_input']
        total_3d += projection_result['num_output']
        
        if config.VERBOSE and len(projection_results) % 10 == 0:
            print(f"  Processed {len(projection_results)} frame pairs...")
    
    # Generate visualization
    if projection_results and config.SAVE_DEBUG_IMAGES and config.GENERATE_STAGE_C_PLOT:
        projection_3d.visualize_3d_projection(projection_results, debug_dir, config)
    
    # Calculate statistics
    avg_projection_rate = (total_3d / total_2d * 100) if total_2d > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"[COMPLETE] STAGE C")
    print(f"{'='*70}")
    print(f"  • Frame pairs processed: {len(projection_results)}")
    print(f"  • Total 2D features: {total_2d}")
    print(f"  • Total 3D points: {total_3d}")
    print(f"  • Projection success rate: {avg_projection_rate:.1f}%")
    if config.GENERATE_STAGE_C_PLOT:
        print(f"  • 3D projection plot: {config.STAGE_C_3D_PROJECTION_PLOT}")
    print(f"{'='*70}\n")
    
    return projection_results


def main():
    """Main entry point for spin analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Spin Analysis - Phase 3 (Stages A, B & C)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--video',
        type=str,
        required=False,
        help='Video file to process (e.g., cropped_test3.mp4)'
    )
    
    args = parser.parse_args()
    
    # Determine which videos to process
    if args.video:
        video_name = Path(args.video).stem
        videos = [video_name]
    else:
        videos = [Path(v).stem for v in config.VIDEO_FILES]
    
    print(f"\n{'#'*80}")
    print(f"# SPIN ANALYSIS - PHASE 3")
    print(f"# Stages: A (Data Prep) + B (Optical Flow) + C (3D Projection)")
    print(f"# Processing {len(videos)} video(s)")
    print(f"{'#'*80}\n")
    
    # Process each video
    for video_name in videos:
        try:
            print(f"\n{'='*70}")
            print(f"Processing: {video_name}")
            print(f"{'='*70}")
            
            # Stage A: Data Preparation
            trajectory_df, output_dir, video_path = run_stage_a(video_name, config)
            
            # Stage B: Optical Flow Detection
            test_results = run_stage_b(video_path, trajectory_df, output_dir, config)
            
            # Stage C: 3D Projection
            projection_results = run_stage_c(test_results, trajectory_df, output_dir, config)
            
            print(f"\n{'='*70}")
            print(f"[COMPLETE] {video_name}")
            print(f"{'='*70}")
            print(f"Output: {output_dir}")
            print(f"{'='*70}\n")
            
        except FileNotFoundError as e:
            print(f"\n[ERROR] {video_name}")
            print(f"  {e}\n")
            continue
        except Exception as e:
            print(f"\n[ERROR] {video_name}")
            print(f"  {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    print(f"\n{'#'*80}")
    print(f"# SPIN ANALYSIS COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
