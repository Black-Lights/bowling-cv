"""
Main Pipeline for Lane Detection

Orchestrates the complete Phase 1 lane detection pipeline using LaneDetector class.
Detects all 4 boundaries and calculates intersection points.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lane_detection import LaneDetector, config


def detect_lane_boundaries(video_file, config_module):
    """
    Run complete lane detection pipeline for a single video.
    
    Parameters:
    -----------
    video_file : str
        Name of the video file (with or without extension)
    config_module : module
        Configuration module containing settings
        
    Returns:
    --------
    dict : Detection results with boundaries and intersections
    """
    # Initialize detector
    detector = LaneDetector(video_file, config_module)
    
    # Run complete pipeline
    boundaries, intersections = detector.detect_all()
    
    # Save all results
    detector.save()
    
    # Return results
    return {
        'video_name': detector.video_name,
        'boundaries': boundaries,
        'intersections': intersections,
        'output_dir': str(detector.output_dir)
    }


def main():
    """
    Main entry point for lane detection pipeline.
    
    Processes configured videos using LaneDetector class to detect
    all 4 boundaries and calculate all intersection points.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bowling Lane Detection Pipeline - Phase 1')
    parser.add_argument('--video', type=str, help='Process single video file')
    args = parser.parse_args()
    
    # Determine which videos to process
    if args.video:
        videos = [args.video]
    else:
        videos = config.VIDEO_FILES
    
    print(f"\n{'#'*80}")
    print(f"# BOWLING LANE DETECTION - PHASE 1")
    print(f"# LaneDetector v{LaneDetector.VERSION}")
    print(f"# Processing {len(videos)} video(s)")
    print(f"{'#'*80}\n")
    
    # Process each video
    successful = 0
    failed = 0
    
    for video_file in videos:
        try:
            result = detect_lane_boundaries(video_file, config)
            
            boundaries = result['boundaries']
            intersections = result['intersections']
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"SUMMARY: {result['video_name']}")
            print(f"{'='*80}")
            print(f"  Boundaries Detected:")
            print(f"    Bottom (Foul):  Y={boundaries['bottom']['center_y']}")
            print(f"    Left Master:    X={boundaries['left']['x_intersect']}, angle={boundaries['left']['median_angle']:.1f}deg")
            print(f"    Right Master:   X={boundaries['right']['x_intersect']}, angle={boundaries['right']['median_angle']:.1f}deg")
            print(f"    Top (Pin Area): Y={boundaries['top']['y_position']:.1f}")
            print(f"\n  Intersection Points:")
            print(f"    Top-Left:     ({intersections['top_left']['x']}, {intersections['top_left']['y']})")
            print(f"    Top-Right:    ({intersections['top_right']['x']}, {intersections['top_right']['y']})")
            print(f"    Bottom-Left:  ({intersections['bottom_left']['x']}, {intersections['bottom_left']['y']})")
            print(f"    Bottom-Right: ({intersections['bottom_right']['x']}, {intersections['bottom_right']['y']})")
            print(f"\n  Output: {result['output_dir']}")
            print(f"{'='*80}\n")
            
            successful += 1
            
        except Exception as e:
            print(f"\nERROR: Error processing {video_file}:")
            print(f"  {e}\n")
            failed += 1
            
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
    
    # Final summary
    print(f"\n{'#'*80}")
    print(f"# PHASE 1: LANE DETECTION COMPLETE")
    print(f"# Successful: {successful}/{len(videos)}")
    if failed > 0:
        print(f"# Failed: {failed}/{len(videos)}")
    print(f"{'#'*80}\n")


if __name__ == '__main__':
    main()
