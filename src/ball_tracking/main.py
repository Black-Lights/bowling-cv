"""
Ball Tracking - Phase 2
Main entry point for ball tracking pipeline

Usage:
    python -m src.ball_tracking.main --video path/to/video.mp4
    
    Or with custom boundary data:
    python -m src.ball_tracking.main --video video.mp4 --boundary boundary_data.json

Version: 2.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: January 30, 2026
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ball_tracking.ball_tracker import BallTracker
from src.ball_tracking import config


def load_boundary_data(boundary_path: str) -> dict:
    """Load boundary data from JSON file."""
    try:
        with open(boundary_path, 'r') as f:
            boundary_data = json.load(f)
        print(f"✓ Loaded boundary data from: {boundary_path}")
        return boundary_data
    except FileNotFoundError:
        print(f"✗ Error: Boundary file not found: {boundary_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in boundary file: {e}")
        sys.exit(1)


def find_boundary_data_for_video(video_path: str) -> str:
    """
    Find boundary data file for given video.
    Assumes Phase 1 output structure: output/<video_name>/boundary_data.json
    """
    video_name = Path(video_path).stem
    boundary_path = f"output/{video_name}/boundary_data.json"
    
    if os.path.exists(boundary_path):
        return boundary_path
    else:
        return None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Ball Tracking - Phase 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use auto-detected boundary data from Phase 1
  python -m src.ball_tracking.main --video assets/input/cropped_test8.mp4
  
  # Use custom boundary data
  python -m src.ball_tracking.main --video video.mp4 --boundary custom_boundary.json
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--boundary',
        type=str,
        default=None,
        help='Path to boundary data JSON file (optional, will auto-detect if not provided)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (optional, default: output/<video_name>/tracking)'
    )
    
    args = parser.parse_args()
    
    # Validate video file
    if not os.path.exists(args.video):
        print(f"✗ Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Find or validate boundary data
    boundary_path = args.boundary
    if boundary_path is None:
        print("Searching for boundary data from Phase 1...")
        boundary_path = find_boundary_data_for_video(args.video)
        
        if boundary_path is None:
            print(f"✗ Error: No boundary data found for video: {args.video}")
            print("Please run Phase 1 (Lane Detection) first, or provide --boundary path")
            sys.exit(1)
    
    # Load boundary data
    boundary_data = load_boundary_data(boundary_path)
    
    # Print configuration
    print("\n" + "="*60)
    print("BALL TRACKING - PHASE 2")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Boundary Data: {boundary_path}")
    print(f"Output: {args.output or 'auto-generated'}")
    print("="*60 + "\n")
    
    # Initialize tracker
    tracker = BallTracker(
        video_path=args.video,
        boundary_data=boundary_data,
        config=config,
        output_dir=args.output
    )
    
    # Run tracking
    results = tracker.track_all()
    
    if results:
        print("\n" + "="*60)
        print("TRACKING COMPLETE")
        print("="*60)
        print(f"✓ Tracked frames: {len(results['trajectory'])}")
        if results.get('release_point'):
            print(f"✓ Release point: Frame {results['release_point']['frame']}")
        if results.get('impact_point'):
            print(f"✓ Impact point: Frame {results['impact_point']['frame']}")
        print(f"✓ Results saved to: {tracker.output_dir}")
        print("="*60)
    else:
        print("\n✗ Tracking failed - no results generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
