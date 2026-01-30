"""
Main entry point for bowling lane detection pipeline.

This script runs the complete Phase 1 lane detection using the LaneDetector class.
Processes all configured videos and outputs all 4 boundaries + intersection points.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Last Updated: January 30, 2026

Usage:
    python main.py
    
    Or process a single video:
    python main.py --video cropped_test3.mp4
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lane_detection import LaneDetector
from lane_detection import config


def main():
    """
    Main entry point for complete lane detection pipeline.
    
    Processes configured videos using LaneDetector class to detect
    all 4 boundaries and calculate all intersection points.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bowling Lane Detection Pipeline')
    parser.add_argument('--video', type=str, help='Process single video file')
    args = parser.parse_args()
    
    # Determine which videos to process
    if args.video:
        videos = [args.video]
    else:
        videos = config.VIDEO_FILES
    
    print(f"\n{'#'*80}")
    print(f"# BOWLING LANE DETECTION PIPELINE v{LaneDetector.VERSION}")
    print(f"# Processing {len(videos)} video(s)")
    print(f"{'#'*80}\n")
    
    # Process each video
    successful = 0
    failed = 0
    
    for video_file in videos:
        try:
            # Initialize detector
            detector = LaneDetector(video_file, config)
            
            # Run complete pipeline
            boundaries, intersections = detector.detect_all()
            
            # Save all results
            detector.save()
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"📊 SUMMARY: {detector.video_name}")
            print(f"{'='*70}")
            print(f"  Boundaries Detected:")
            print(f"    ✅ Bottom (Foul):  Y={boundaries['bottom']['center_y']}")
            print(f"    ✅ Left Master:    X={boundaries['left']['x_intersect']}, θ={boundaries['left']['median_angle']:.1f}°")
            print(f"    ✅ Right Master:   X={boundaries['right']['x_intersect']}, θ={boundaries['right']['median_angle']:.1f}°")
            print(f"    ✅ Top (Pin Area): Y={boundaries['top']['y_position']:.1f}")
            print(f"\n  Intersection Points:")
            print(f"    • Top-Left:     ({intersections['top_left']['x']}, {intersections['top_left']['y']})")
            print(f"    • Top-Right:    ({intersections['top_right']['x']}, {intersections['top_right']['y']})")
            print(f"    • Bottom-Left:  ({intersections['bottom_left']['x']}, {intersections['bottom_left']['y']})")
            print(f"    • Bottom-Right: ({intersections['bottom_right']['x']}, {intersections['bottom_right']['y']})")
            print(f"\n  Output: {detector.output_dir}")
            print(f"{'='*70}\n")
            
            successful += 1
            
        except Exception as e:
            print(f"\n❌ ERROR processing {video_file}: {e}")
            failed += 1
            
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
    
    # Final summary
    print(f"\n{'#'*80}")
    print(f"# PIPELINE COMPLETE")
    print(f"# Successful: {successful}/{len(videos)}")
    if failed > 0:
        print(f"# Failed: {failed}/{len(videos)}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
