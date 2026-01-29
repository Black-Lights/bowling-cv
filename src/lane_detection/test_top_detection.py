"""
Test script for top boundary detection using Sobel edge detection
Generates 3 output videos for each input:
1. Sobel filter visualization (red heatmap)
2. Preprocessed video with detected top line
3. Final video with all 4 boundaries
"""

import os
import sys
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

import config
from main import load_boundary_data
from mask_lane_area import apply_mask_to_video
from preprocess_frames import create_preprocessed_video
from top_boundary_detection import (
    detect_top_boundary_all_frames,
    create_visualization_videos,
    plot_intersection_y_coordinates,
    fit_msac_line
)


def test_top_detection_all_videos():
    """
    Run top boundary detection on all test videos.
    Creates 3 output videos for each input video.
    """
    print("="*70)
    print(" TOP BOUNDARY DETECTION - SOBEL EDGE DETECTION")
    print("="*70)
    
    for video_file in config.VIDEO_FILES:
        video_name = video_file.replace('.mp4', '')
        print(f"\n{'#'*70}")
        print(f"# Processing: {video_name}")
        print(f"{'#'*70}")
        
        # Paths
        video_path = os.path.join(config.ASSETS_DIR, video_file)
        output_dir = os.path.join(config.OUTPUT_DIR, video_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"  ⚠ Video not found: {video_path}")
            continue
        
        # Load boundary data
        boundary_data = load_boundary_data(output_dir)
        
        if not boundary_data:
            print(f"  ⚠ No boundary_data.json found. Running main detection first...")
            from main import process_single_video
            process_single_video(video_file, config.OUTPUT_DIR)
            boundary_data = load_boundary_data(output_dir)
            
            if not boundary_data:
                print(f"  ✗ Could not generate boundary data. Skipping {video_name}")
                continue
        
        print(f"\n  Loaded boundaries:")
        print(f"    Left: X={boundary_data['master_left']['x_intersect']}")
        print(f"    Right: X={boundary_data['master_right']['x_intersect']}")
        print(f"    Foul: Y={boundary_data['median_foul_params']['center_y']}")
        
        # Create masked video if it doesn't exist
        masked_video_path = os.path.join(output_dir, f'masked_{video_file}')
        
        if not os.path.exists(masked_video_path):
            print(f"\n  Creating masked video...")
            mask_result = apply_mask_to_video(
                video_path,
                masked_video_path,
                boundary_data['master_left'],
                boundary_data['master_right'],
                boundary_data['median_foul_params']
            )
            
            if not mask_result:
                print(f"  ✗ Failed to create masked video. Skipping {video_name}")
                continue
        else:
            print(f"\n  Using existing masked video: {masked_video_path}")
        
        # Create preprocessed video (HSV filtered with gap filling)
        preprocessed_video_path = os.path.join(output_dir, f'preprocessed_{video_file}')
        
        if not os.path.exists(preprocessed_video_path):
            print(f"\n  Creating preprocessed video (HSV + gap filling)...")
            preprocess_result = create_preprocessed_video(
                masked_video_path, 
                preprocessed_video_path,
                config.MAX_PATCH_SIZE_ROW,
                config.MAX_PATCH_SIZE_COL
            )
            
            if not preprocess_result:
                print(f"  ✗ Failed to create preprocessed video. Skipping {video_name}")
                continue
        else:
            print(f"\n  Using existing preprocessed video: {preprocessed_video_path}")
        
        # Run top boundary detection on preprocessed video
        print(f"\n  Running Sobel-based top boundary detection...")
        detections = detect_top_boundary_all_frames(preprocessed_video_path, config)
        
        if not detections:
            print(f"  ✗ Detection failed for {video_name}")
            continue
        
        # Calculate average Y position
        avg_y = sum(d['y_position'] for d in detections) / len(detections)
        print(f"\n  Detection complete:")
        print(f"    Average Y position: {avg_y:.1f}")
        print(f"    Frames processed: {len(detections)}")
        
        # Fit MSAC line from all detections
        print(f"\n  Fitting MSAC line from all frames...")
        cap_temp = cv2.VideoCapture(video_path)
        video_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_temp.release()
        
        msac_line = fit_msac_line(detections, video_width, output_dir, video_name)
        
        # Generate 3 visualization videos (now using MSAC line)
        print(f"\n  Generating 3 visualization videos with MSAC line...")
        create_visualization_videos(
            video_path,  # Original video
            preprocessed_video_path,  # Preprocessed video
            detections,
            boundary_data,
            output_dir,
            video_name,
            config,
            msac_line  # Pass MSAC line
        )
        
        # Plot intersection Y coordinates to analyze stability
        print(f"\n  Plotting intersection Y coordinates...")
        plot_intersection_y_coordinates(detections, boundary_data, video_name, output_dir)
        
        print(f"\n  ✓ Completed: {video_name}")
        print(f"    Videos saved to: {output_dir}")
        print(f"      1. top_vis_sobel_{video_name}.mp4 - Sobel filter (red heatmap)")
        print(f"      2. top_vis_masked_{video_name}.mp4 - Preprocessed with MSAC line")
        print(f"      3. final_all_boundaries_{video_name}.mp4 - All 4 boundaries (MSAC)")
        print(f"      4. top_line_intersection_y_{video_name}.png - Intersection Y plot")
        print(f"      5. msac_fitting_{video_name}.png - MSAC fitting analysis")
    
    print(f"\n{'='*70}")
    print(" ALL VIDEOS PROCESSED")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_top_detection_all_videos()
