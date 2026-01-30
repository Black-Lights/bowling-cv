"""
Debug script to understand why ball detection is failing

This will show us:
- Motion mask candidates
- Color mask candidates
- Combined mask candidates
- After morphology
- After lane masking
- After shape filtering

Author: Debug Assistant
Created: January 30, 2026
"""

import cv2
import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.ball_tracking import config
from src.ball_tracking.detection_functions import (
    detect_ball_by_motion,
    detect_ball_by_color,
    combine_masks,
    apply_morphological_operations,
    apply_lane_mask,
    filter_ball_contours
)
from src.lane_detection.mask_lane_area import create_lane_mask
import json


def count_contours(mask, min_area=10):
    """Count contours in mask above minimum area"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    return len(valid), valid


def analyze_frame(frame, prev_frame, lane_mask=None):
    """Analyze a single frame and report detection at each step"""
    print("\n" + "="*60)
    print("FRAME ANALYSIS")
    print("="*60)
    
    # Step 1: Motion detection
    motion_mask = detect_ball_by_motion(
        frame,
        prev_frame,
        config.FRAME_DIFF_THRESHOLD
    ) if prev_frame is not None else None
    
    if motion_mask is not None:
        n_motion, motion_contours = count_contours(motion_mask, config.MIN_MOTION_AREA)
        print(f"\n1. MOTION MASK: {n_motion} candidates (>= {config.MIN_MOTION_AREA} px)")
        if n_motion > 0:
            areas = [cv2.contourArea(c) for c in motion_contours]
            print(f"   Areas: min={min(areas):.0f}, max={max(areas):.0f}, avg={np.mean(areas):.0f}")
    else:
        print("\n1. MOTION MASK: N/A (first frame)")
        n_motion = 0
    
    # Step 2: Color detection
    color_mask = detect_ball_by_color(
        frame,
        config.BALL_COLOR_LOWER,
        config.BALL_COLOR_UPPER
    )
    
    n_color, color_contours = count_contours(color_mask, 10)
    print(f"\n2. COLOR MASK: {n_color} candidates (>= 10 px)")
    if n_color > 0:
        areas = [cv2.contourArea(c) for c in color_contours]
        print(f"   Areas: min={min(areas):.0f}, max={max(areas):.0f}, avg={np.mean(areas):.0f}")
    
    # Step 3: Combined mask
    combined = combine_masks(motion_mask, color_mask, use_both=False)
    if combined is None:
        print("\n3. COMBINED MASK: NONE - both masks empty!")
        return None, 0
    
    n_combined, combined_contours = count_contours(combined, 10)
    print(f"\n3. COMBINED MASK (OR): {n_combined} candidates")
    if n_combined > 0:
        areas = [cv2.contourArea(c) for c in combined_contours]
        print(f"   Areas: min={min(areas):.0f}, max={max(areas):.0f}, avg={np.mean(areas):.0f}")
    
    # Step 4: Morphological operations
    morphed = apply_morphological_operations(
        combined,
        kernel_size=config.MORPH_KERNEL_SIZE,
        open_iterations=config.MORPH_OPEN_ITERATIONS,
        close_iterations=config.MORPH_CLOSE_ITERATIONS
    )
    
    n_morphed, morphed_contours = count_contours(morphed, 10)
    print(f"\n4. AFTER MORPHOLOGY: {n_morphed} candidates")
    if n_morphed > 0:
        areas = [cv2.contourArea(c) for c in morphed_contours]
        print(f"   Areas: min={min(areas):.0f}, max={max(areas):.0f}, avg={np.mean(areas):.0f}")
    
    # Step 5: Lane masking
    if lane_mask is not None:
        lane_masked = apply_lane_mask(morphed, lane_mask)
        n_lane, lane_contours = count_contours(lane_masked, 10)
        print(f"\n5. AFTER LANE MASK: {n_lane} candidates")
        if n_lane > 0:
            areas = [cv2.contourArea(c) for c in lane_contours]
            print(f"   Areas: min={min(areas):.0f}, max={max(areas):.0f}, avg={np.mean(areas):.0f}")
    else:
        lane_masked = morphed
        n_lane = n_morphed
        print(f"\n5. AFTER LANE MASK: N/A (no lane mask)")
    
    # Step 6: Shape filtering
    candidates = filter_ball_contours(
        lane_masked,
        min_radius=config.MIN_BALL_RADIUS,
        max_radius=config.MAX_BALL_RADIUS,
        min_area=config.MIN_BALL_AREA,
        max_area=config.MAX_BALL_AREA,
        min_circularity=config.MIN_CIRCULARITY,
        min_solidity=config.MIN_SOLIDITY
    )
    
    print(f"\n6. AFTER SHAPE FILTERING: {len(candidates)} candidates")
    print(f"   Constraints:")
    print(f"   - Radius: {config.MIN_BALL_RADIUS} to {config.MAX_BALL_RADIUS}")
    print(f"   - Area: {config.MIN_BALL_AREA} to {config.MAX_BALL_AREA}")
    print(f"   - Circularity: >= {config.MIN_CIRCULARITY}")
    print(f"   - Solidity: >= {config.MIN_SOLIDITY}")
    
    if len(candidates) > 0:
        print(f"\n   Final candidates:")
        for i, cand in enumerate(candidates[:5]):  # Show first 5
            x, y = cand['center']
            print(f"   {i+1}. x={x:.0f}, y={y:.0f}, r={cand['radius']:.1f}, "
                  f"circ={cand['circularity']:.2f}, sol={cand['solidity']:.2f}, conf={cand['confidence']:.2f}")
    
    return lane_masked, len(candidates)


def main():
    """Main debug function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug ball detection')
    parser.add_argument('--video', default='assets/input/cropped_test3.mp4', help='Video path')
    parser.add_argument('--masked', action='store_true', help='Use masked video')
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("# BALL DETECTION DEBUG")
    print("#"*60)
    
    # Load video
    if args.masked:
        video_path = "output/cropped_test3/masked_video.mp4"
    else:
        video_path = args.video
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    
    # Load lane boundaries
    boundary_path = "output/cropped_test3/boundary_data.json"
    with open(boundary_path, 'r') as f:
        boundaries = json.load(f)
    
    print(f"\nLane boundaries loaded:")
    print(f"  Left: X={boundaries['master_left']['x_intersect']}, angle={boundaries['master_left']['median_angle']:.1f}°")
    print(f"  Right: X={boundaries['master_right']['x_intersect']}, angle={boundaries['master_right']['median_angle']:.1f}°")
    print(f"  Top: Y={boundaries['top_boundary']['y_position']}")
    print(f"  Foul: Y={boundaries['median_foul_params']['center_y']}")
    
    # Create lane mask
    lane_mask = create_lane_mask(
        (height, width),
        boundaries['master_left'],
        boundaries['master_right'],
        boundaries['median_foul_params']
    )
    
    # Analyze first few frames where ball should be visible
    test_frames = [30, 50, 70, 90, 110]  # Frames to analyze
    
    prev_frame = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in test_frames:
            print(f"\n{'*'*60}")
            print(f"* ANALYZING FRAME {frame_idx}")
            print(f"{'*'*60}")
            
            analyze_frame(frame, prev_frame, lane_mask)
        
        prev_frame = frame.copy()
        frame_idx += 1
    
    cap.release()
    
    print("\n" + "#"*60)
    print("# DEBUG COMPLETE")
    print("#"*60)


if __name__ == "__main__":
    main()
