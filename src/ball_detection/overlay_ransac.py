"""
RANSAC Overlay Video Generation

Generate ball tracking overlay video using RANSAC fitted radius.
Shows cleaned ball circle (using RANSAC exponential model) and trajectory 
overlaid on original video frames.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026

Usage:
    python -m src.ball_detection.overlay_ransac <video_file>

Example:
    python -m src.ball_detection.overlay_ransac cropped_test6.mp4
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ball_detection import config
from ball_detection.post_processing import create_ball_overlay_video


def generate_ransac_overlay(video_path: str, config) -> dict:
    """
    Generate overlay videos based on config flags for different radius sources.
    
    Creates overlay video(s) with:
    - Yellow circles showing ball radius (source depends on flags)
    - Magenta trajectory path showing ball movement
    - Frame information with radius values
    
    Generates up to 3 videos based on config flags:
    - RANSAC fitted radius (exponential decay model)
    - Cleaned measured radius (after MAD outlier removal)
    - Raw measured radius (original detections)
    
    Args:
        video_path (str): Path to input video file
        config: Configuration module with settings
    
    Returns:
        dict: Results with output_paths and metadata
        
    Raises:
        FileNotFoundError: If video or trajectory CSV not found
    """
    # Validate video path
    if not Path(video_path).is_absolute():
        video_path = Path(config.ASSETS_DIR) / video_path
    
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Setup paths
    video_name = Path(video_path).stem
    output_dir = Path(config.OUTPUT_DIR) / video_name / "ball_detection"
    trajectory_csv_path = output_dir / "trajectory_processed_original.csv"
    
    # Check if trajectory CSV exists
    if not trajectory_csv_path.exists():
        raise FileNotFoundError(
            f"Processed trajectory CSV not found: {trajectory_csv_path}\n"
            f"Please run the ball detection pipeline first:\n"
            f"  python -m src.ball_detection.main --video {video_name}.mp4"
        )
    
    if config.VERBOSE:
        print(f"\n{'='*80}")
        print(f"Generating Overlay Videos - Stage H")
        print(f"Video: {video_name}")
        print(f"{'='*80}\n")
    
    results = {
        'video_name': video_name,
        'output_paths': []
    }
    
    # Generate RANSAC fitted radius overlay
    if config.SAVE_OVERLAY_RANSAC_FITTED:
        if config.VERBOSE:
            print(f"Generating RANSAC fitted radius overlay...")
        
        create_ball_overlay_video(
            video_path=str(video_path),
            trajectory_csv_path=str(trajectory_csv_path),
            output_path=output_dir,
            fps=config.OVERLAY_FPS,
            circle_color=config.OVERLAY_CIRCLE_COLOR,
            trajectory_color=config.OVERLAY_TRAJECTORY_COLOR,
            line_width=config.OVERLAY_LINE_WIDTH,
            radius_source="fitted",
            verbose=config.VERBOSE
        )
        
        output_file = output_dir / "ball_tracking_overlay_ransac.mp4"
        results['output_paths'].append(str(output_file))
        
        if config.VERBOSE:
            print(f"  ✓ RANSAC fitted: {output_file.name}")
    
    # Generate cleaned measured radius overlay
    if config.SAVE_OVERLAY_MEASURED_CLEANED:
        if config.VERBOSE:
            print(f"Generating cleaned measured radius overlay...")
        
        create_ball_overlay_video(
            video_path=str(video_path),
            trajectory_csv_path=str(trajectory_csv_path),
            output_path=output_dir,
            fps=config.OVERLAY_FPS,
            circle_color=config.OVERLAY_CIRCLE_COLOR,
            trajectory_color=config.OVERLAY_TRAJECTORY_COLOR,
            line_width=config.OVERLAY_LINE_WIDTH,
            radius_source="measured",
            verbose=config.VERBOSE
        )
        
        output_file = output_dir / "ball_tracking_overlay_measured_cleaned.mp4"
        results['output_paths'].append(str(output_file))
        
        if config.VERBOSE:
            print(f"  ✓ Cleaned measured: {output_file.name}")
    
    # Generate raw measured radius overlay
    if config.SAVE_OVERLAY_MEASURED_RAW:
        if config.VERBOSE:
            print(f"Generating raw measured radius overlay...")
        
        create_ball_overlay_video(
            video_path=str(video_path),
            trajectory_csv_path=str(trajectory_csv_path),
            output_path=output_dir,
            fps=config.OVERLAY_FPS,
            circle_color=config.OVERLAY_CIRCLE_COLOR,
            trajectory_color=config.OVERLAY_TRAJECTORY_COLOR,
            line_width=config.OVERLAY_LINE_WIDTH,
            radius_source="raw",
            verbose=config.VERBOSE
        )
        
        output_file = output_dir / "ball_tracking_overlay_measured_raw.mp4"
        results['output_paths'].append(str(output_file))
        
        if config.VERBOSE:
            print(f"  ✓ Raw measured: {output_file.name}")
    
    if config.VERBOSE and results['output_paths']:
        print(f"\n✓ Generated {len(results['output_paths'])} overlay video(s)")
    
    return results


def main():
    """
    Main entry point for command-line usage.
    
    Generates overlay video(s) for specified input video based on config flags.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.ball_detection.overlay_ransac <video_file>")
        print("\nExample:")
        print("  python -m src.ball_detection.overlay_ransac cropped_test6.mp4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    
    # Build list of enabled overlay types
    overlay_types = []
    if config.SAVE_OVERLAY_RANSAC_FITTED:
        overlay_types.append('RANSAC Fitted')
    if config.SAVE_OVERLAY_MEASURED_CLEANED:
        overlay_types.append('Cleaned Measured')
    if config.SAVE_OVERLAY_MEASURED_RAW:
        overlay_types.append('Raw Measured')
    
    print(f"\n{'#'*80}")
    print(f"# STAGE H: OVERLAY VIDEO GENERATION")
    print(f"# Radius Sources: {', '.join(overlay_types) if overlay_types else 'None'}")
    print(f"{'#'*80}\n")
    
    try:
        result = generate_ransac_overlay(video_file, config)
        
        print(f"\n{'='*80}")
        print(f"✓ STAGE H: OVERLAY VIDEO GENERATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Generated {len(result['output_paths'])} overlay video(s):")
        for path in result['output_paths']:
            print(f"  - {Path(path).name}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
