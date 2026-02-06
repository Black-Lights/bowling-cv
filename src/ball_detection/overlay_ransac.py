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
    Generate RANSAC fitted radius overlay video for the specified video.
    
    Creates overlay video with:
    - Yellow circles showing RANSAC fitted radius (exponential decay model)
    - Magenta trajectory path showing ball movement
    - Frame information with radius values
    
    Args:
        video_path (str): Path to input video file
        config: Configuration module with settings
    
    Returns:
        dict: Results with output_path and metadata
        
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
        print(f"Generating RANSAC Overlay Video")
        print(f"Video: {video_name}")
        print(f"{'='*80}\n")
    
    # Generate overlay video using RANSAC fitted radius
    create_ball_overlay_video(
        video_path=str(video_path),
        trajectory_csv_path=str(trajectory_csv_path),
        output_path=output_dir,
        fps=config.OVERLAY_FPS,
        circle_color=config.OVERLAY_CIRCLE_COLOR,
        trajectory_color=config.OVERLAY_TRAJECTORY_COLOR,
        line_width=config.OVERLAY_LINE_WIDTH,
        radius_source=config.OVERLAY_RADIUS_SOURCE,
        verbose=config.VERBOSE
    )
    
    output_file = output_dir / "ball_tracking_overlay_ransac.mp4"
    
    if config.VERBOSE:
        print(f"✓ RANSAC overlay video generated")
        print(f"  Output: {output_file}")
    
    return {
        'output_path': str(output_file),
        'video_name': video_name,
        'radius_source': 'RANSAC fitted'
    }


def main():
    """
    Main entry point for command-line usage.
    
    Generates RANSAC overlay video for specified input video.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.ball_detection.overlay_ransac <video_file>")
        print("\nExample:")
        print("  python -m src.ball_detection.overlay_ransac cropped_test6.mp4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    
    print(f"\n{'#'*80}")
    print(f"# RANSAC OVERLAY VIDEO GENERATION")
    print(f"# Radius Source: RANSAC Fitted Exponential Model")
    print(f"{'#'*80}\n")
    
    try:
        result = generate_ransac_overlay(video_file, config)
        
        print(f"\n{'='*80}")
        print(f"✓ RANSAC OVERLAY VIDEO GENERATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Output: {result['output_path']}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
