"""
Main Pipeline for Pin Detection

Orchestrates the complete Phase 4 pin detection pipeline:
1. Video preprocessing (extended masking)
2. Frame selection (before/after)
3. Frame differencing and contour detection
4. Pin counting
5. Visualization and export

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

import cv2
import numpy as np
import json
import os
import time
from pathlib import Path

from . import config
from .video_preprocessing import create_pin_area_masked_video
from .frame_selector import select_and_extract_frames
from .pin_counter import PinCounter
from .visualization import PinDetectionVisualizer


def detect_pins_in_video(video_name):
    """
    Run complete pin detection pipeline for a single video.
    
    Parameters:
    -----------
    video_name : str
        Name of the video file (without extension)
        
    Returns:
    --------
    dict : Pin detection results
    """
    start_time = time.time()
    
    print("\n" + "="*80)
    print(f"PIN DETECTION PIPELINE - {video_name}")
    print("="*80)
    
    # Setup paths
    video_path = config.get_video_input_path(video_name)
    boundary_data_path = config.get_boundary_data_path(video_name)
    trajectory_data_path = config.get_trajectory_data_path(video_name)
    output_dir = config.get_pin_detection_output_dir(video_name)
    intermediate_dir = config.get_intermediate_output_dir(video_name)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Verify input files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(boundary_data_path):
        raise FileNotFoundError(
            f"Boundary data not found: {boundary_data_path}\n"
            f"Please run Phase 1 (Lane Detection) first!"
        )
    
    # ========================================================================
    # STEP 1: VIDEO PREPROCESSING (Extended Masking)
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 1: VIDEO PREPROCESSING")
    print(f"{'='*80}")
    
    step_start = time.time()
    
    masked_video_path = os.path.join(output_dir, f'{video_name}_pin_area_masked.mp4')
    
    # Check if masked video already exists
    if os.path.exists(masked_video_path) and not config.DEBUG_MODE:
        print(f"✅ Using existing masked video: {masked_video_path}")
    else:
        print(f"🎬 Creating pin area masked video...")
        create_pin_area_masked_video(
            video_path,
            boundary_data_path,
            masked_video_path
        )
    
    if config.PRINT_TIMING:
        print(f"⏱️  Step 1 completed in {time.time() - step_start:.2f}s")
    
    # ========================================================================
    # STEP 2: FRAME SELECTION
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 2: FRAME SELECTION")
    print(f"{'='*80}")
    
    step_start = time.time()
    
    before_frame, after_frame, before_idx, after_idx = select_and_extract_frames(
        masked_video_path,
        boundary_data_path if os.path.exists(boundary_data_path) else None,
        trajectory_data_path if os.path.exists(trajectory_data_path) else None,
        intermediate_dir if config.SAVE_INTERMEDIATE_FRAMES else None
    )
    
    if config.PRINT_TIMING:
        print(f"⏱️  Step 2 completed in {time.time() - step_start:.2f}s")
    
    # ========================================================================
    # STEP 3: PIN COUNTING
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: PIN COUNTING")
    print(f"{'='*80}")
    
    step_start = time.time()
    
    # Load boundary data for pin area masking
    with open(boundary_data_path, 'r') as f:
        boundary_data = json.load(f)
    
    # Initialize counter
    counter = PinCounter()
    
    # Run detection with boundary data for pin area focusing
    results = counter.count_pins(before_frame, after_frame, boundary_data)
    
    # Add metadata
    results['video_name'] = video_name
    results['before_frame_index'] = before_idx
    results['after_frame_index'] = after_idx
    results['detection_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    if config.PRINT_TIMING:
        print(f"⏱️  Step 3 completed in {time.time() - step_start:.2f}s")
    
    # ========================================================================
    # STEP 4: VISUALIZATION
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 4: VISUALIZATION")
    print(f"{'='*80}")
    
    step_start = time.time()
    
    visualizer = PinDetectionVisualizer(output_dir)
    
    # Save intermediate frames if enabled
    if config.SAVE_INTERMEDIATE_FRAMES:
        # Difference pipeline
        if config.VISUALIZE_DIFFERENCE:
            visualizer.visualize_difference_pipeline(
                before_frame, after_frame,
                counter.difference, counter.binary_diff, counter.cleaned_diff,
                os.path.join(intermediate_dir, 'difference_pipeline.png')
            )
        
        # Contour detection
        if config.VISUALIZE_CONTOURS:
            visualizer.visualize_contours(
                after_frame, counter.valid_contours, counter.all_contours,
                os.path.join(intermediate_dir, 'contour_detection.png')
            )
    
    # Final result visualization
    visualizer.visualize_final_result(
        after_frame, results, before_idx, after_idx,
        os.path.join(output_dir, f'{video_name}_pin_detection_result.png')
    )
    
    # Complete comparison panel
    if config.VISUALIZE_COMPARISON:
        visualizer.create_comparison_panel(
            before_frame, after_frame,
            counter.difference, counter.binary_diff, counter.cleaned_diff,
            results, before_idx, after_idx,
            os.path.join(output_dir, f'{video_name}_complete_comparison.png')
        )
    
    # Statistical plots
    if config.SAVE_DEBUG_PLOTS and len(results['valid_contours']) > 0:
        visualizer.plot_detection_statistics(
            results,
            os.path.join(intermediate_dir, 'detection_statistics.png')
        )
    
    if config.PRINT_TIMING:
        print(f"⏱️  Step 4 completed in {time.time() - step_start:.2f}s")
    
    # ========================================================================
    # STEP 5: EXPORT RESULTS
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 5: EXPORT RESULTS")
    print(f"{'='*80}")
    
    step_start = time.time()
    
    # Prepare export data (remove non-serializable objects)
    export_results = results.copy()
    
    # Convert contour data to serializable format
    export_results['valid_contours'] = [
        {
            'pin_id': i + 1,
            'area': pin['area'],
            'bbox': pin['bbox'],
            'aspect_ratio': pin['aspect_ratio'],
            'solidity': pin['solidity'],
            'center': pin['center']
        }
        for i, pin in enumerate(results['valid_contours'])
    ]
    
    # Add detection parameters
    export_results['detection_parameters'] = {
        'unmask_extension': config.PIN_AREA_UNMASK_EXTENSION,
        'difference_threshold': config.DIFFERENCE_THRESHOLD,
        'morph_kernel_size': config.MORPH_KERNEL_SIZE,
        'min_pin_area': config.MIN_PIN_AREA,
        'max_pin_area': config.MAX_PIN_AREA,
        'min_aspect_ratio': config.MIN_PIN_ASPECT_RATIO,
        'max_aspect_ratio': config.MAX_PIN_ASPECT_RATIO,
        'min_solidity': config.MIN_PIN_SOLIDITY
    }
    
    # Export to JSON
    if config.EXPORT_JSON:
        json_path = os.path.join(output_dir, f'{video_name}_pin_detection.json')
        with open(json_path, 'w') as f:
            json.dump(export_results, f, indent=2)
        print(f"   ✅ Results exported to JSON: {json_path}")
    
    # Export to CSV (optional)
    if config.EXPORT_CSV and len(results['valid_contours']) > 0:
        import csv
        csv_path = os.path.join(output_dir, f'{video_name}_pin_contours.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'pin_id', 'area', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                'aspect_ratio', 'solidity', 'center_x', 'center_y'
            ])
            writer.writeheader()
            for i, pin in enumerate(results['valid_contours'], 1):
                bbox = pin['bbox']
                center = pin['center']
                writer.writerow({
                    'pin_id': i,
                    'area': pin['area'],
                    'bbox_x': bbox[0],
                    'bbox_y': bbox[1],
                    'bbox_w': bbox[2],
                    'bbox_h': bbox[3],
                    'aspect_ratio': pin['aspect_ratio'],
                    'solidity': pin['solidity'],
                    'center_x': center[0],
                    'center_y': center[1]
                })
        print(f"   ✅ Contour data exported to CSV: {csv_path}")
    
    if config.PRINT_TIMING:
        print(f"⏱️  Step 5 completed in {time.time() - step_start:.2f}s")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("🎳 PIN DETECTION COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Results:")
    print(f"   Video:           {video_name}")
    print(f"   Result:          {results['result']}")
    print(f"   Remaining Pins:  {results['remaining_pins']}")
    print(f"   Toppled Pins:    {results['toppled_pins']}")
    print(f"   Confidence:      {results['detection_confidence']:.1%}")
    print(f"\n⏱️  Total Time:      {total_time:.2f}s")
    print(f"\n📁 Output Directory: {output_dir}")
    print(f"{'='*80}\n")
    
    return results


def main():
    """
    Main entry point for pin detection pipeline.
    Processes all configured videos or specific video from command line.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pin Detection Pipeline')
    parser.add_argument('--video', type=str, help='Video file name (with or without .mp4 extension)')
    args = parser.parse_args()
    
    # Determine which videos to process
    if args.video:
        # Process specific video from command line
        video_name = args.video.replace('.mp4', '')  # Remove .mp4 if present
        video_names = [video_name]
    else:
        # Process all videos from config
        video_names = config.VIDEO_NAMES
    
    # Print configuration summary
    if config.VERBOSE:
        config.print_config_summary()
    
    # Process each video
    all_results = {}
    
    for video_name in video_names:
        try:
            results = detect_pins_in_video(video_name)
            all_results[video_name] = results
        except Exception as e:
            print(f"\n❌ Error processing {video_name}: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    # Print summary for all videos
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("SUMMARY: ALL VIDEOS")
        print("="*80)
        for video_name, results in all_results.items():
            print(f"\n{video_name}:")
            print(f"   Result: {results['result']}")
            print(f"   Pins:   {results['toppled_pins']}/{config.TOTAL_PINS} toppled")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
