"""
Stage B: Motion Detection using Background Subtraction

This module implements ball detection using MOG2 background subtraction:
1. Apply MOG2 to get foreground mask
2. Remove shadows (grey pixels, value 127)
3. Apply morphological opening to remove noise

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Date: February 1, 2026
"""

import cv2
import numpy as np
import os
from pathlib import Path
import subprocess
import shutil
from tqdm import tqdm


def create_morphological_kernel(size, shape='ellipse'):
    """
    Create morphological kernel for noise removal
    
    Args:
        size: Kernel size (e.g., 3, 5)
        shape: 'ellipse' for circular or 'rect' for rectangular
        
    Returns:
        numpy array: Morphological kernel
    """
    if shape == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    else:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def apply_background_subtraction(masked_frames_generator, config, save_videos=True):
    """
    Apply MOG2 background subtraction to detect moving ball
    
    Pipeline:
    1. MOG2 background subtraction → foreground mask
    2. Shadow removal → threshold to keep only white pixels (255)
    3. Morphological opening → remove salt-and-pepper noise
    
    Args:
        masked_frames_generator: Generator yielding (frame_idx, frame, metadata)
        config: Configuration module with MOG2 parameters
        save_videos: Whether to save intermediate videos
        
    Returns:
        Generator yielding (frame_idx, denoised_mask, metadata, intermediate_masks)
        where intermediate_masks = {
            'foreground': raw MOG2 output,
            'shadow_removed': after shadow thresholding,
            'denoised': final clean mask
        }
    """
    # Initialize MOG2 background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=config.MOG2_HISTORY,
        varThreshold=config.MOG2_VAR_THRESHOLD,
        detectShadows=config.MOG2_DETECT_SHADOWS
    )
    
    # Create morphological kernel for noise removal
    kernel = create_morphological_kernel(
        config.MORPH_KERNEL_SIZE, 
        config.MORPH_KERNEL_SHAPE
    )
    
    if config.VERBOSE:
        print("\n" + "="*60)
        print("STAGE B: MOTION DETECTION (Background Subtraction)")
        print("="*60)
        print(f"MOG2 History: {config.MOG2_HISTORY} frames")
        print(f"Variance Threshold: {config.MOG2_VAR_THRESHOLD}")
        print(f"Detect Shadows: {config.MOG2_DETECT_SHADOWS}")
        print(f"Shadow Threshold: {config.SHADOW_THRESHOLD}")
        print(f"Morphological Kernel: {config.MORPH_KERNEL_SHAPE} {config.MORPH_KERNEL_SIZE}x{config.MORPH_KERNEL_SIZE}")
        if config.USE_SHADOW_SEPARATION:
            print(f"Shadow Separation: {config.SHADOW_SEPARATION_ITERATIONS} erosion iterations")
        print("="*60 + "\n")
    
    # Process each frame
    for frame_idx, masked_frame, metadata in masked_frames_generator:
        # Stage B.1: Apply MOG2 background subtraction
        foreground_mask = bg_subtractor.apply(masked_frame)
        
        # Stage B.2: Shadow Removal
        # MOG2 marks shadows as 127 (grey), foreground as 255 (white), background as 0 (black)
        # We threshold to keep only pure white (foreground), removing shadows and background
        _, shadow_removed = cv2.threshold(
            foreground_mask, 
            config.SHADOW_THRESHOLD,  # Typically 200 to remove 127 but keep 255
            255, 
            cv2.THRESH_BINARY
        )
        
        # Stage B.3: Shadow Separation (CRITICAL: separates ball from attached shadow)
        # Apply extra erosion to disconnect ball from shadow before opening
        if config.USE_SHADOW_SEPARATION:
            separated = cv2.erode(shadow_removed, kernel, iterations=config.SHADOW_SEPARATION_ITERATIONS)
        else:
            separated = shadow_removed
        
        # Stage B.4: Noise Removal (Morphological Opening)
        # Opening = Erosion followed by Dilation
        # Removes small white noise (salt) while preserving larger structures (ball)
        denoised_mask = cv2.morphologyEx(
            separated, 
            cv2.MORPH_OPEN, 
            kernel
        )
        
        # Collect intermediate masks for visualization
        intermediate_masks = {
            'foreground': foreground_mask,
            'shadow_removed': shadow_removed,
            'separated': separated if config.USE_SHADOW_SEPARATION else shadow_removed,
            'denoised': denoised_mask,
            'original_masked': masked_frame  # Include original for comparison
        }
        
        yield frame_idx, denoised_mask, metadata, intermediate_masks


def save_motion_detection_videos(video_path, config, output_base_dir):
    """
    Process video through motion detection and save intermediate videos
    
    Args:
        video_path: Path to input video file
        config: Configuration module
        output_base_dir: Base output directory for this video
        
    Returns:
        dict: Paths to saved videos
    """
    from .mask_video import create_masked_lane_video
    
    video_name = Path(video_path).stem
    intermediate_dir = os.path.join(output_base_dir, 'ball_detection', 'intermediate')
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Get masked frames generator (from Stage A)
    if config.VERBOSE:
        print(f"\nProcessing: {video_name}")
        print(f"Output: {intermediate_dir}")
    
    masked_frames_gen = create_masked_lane_video(
        video_path, 
        config, 
        save_video=False  # We'll handle video saving here
    )
    
    # Apply motion detection
    motion_gen = apply_background_subtraction(
        masked_frames_gen, 
        config, 
        save_videos=True
    )
    
    # Collect all frames and intermediate results
    all_frames = []
    foreground_frames = []
    shadow_removed_frames = []
    denoised_frames = []
    
    for frame_idx, denoised_mask, metadata, intermediate_masks in tqdm(
        motion_gen, 
        desc="Stage B: Motion Detection",
        disable=not config.VERBOSE
    ):
        all_frames.append(intermediate_masks['original_masked'])  # Original masked frame
        foreground_frames.append(intermediate_masks['foreground'])
        shadow_removed_frames.append(intermediate_masks['shadow_removed'])
        denoised_frames.append(intermediate_masks['denoised'])
    
    if len(all_frames) == 0:
        print("ERROR: No frames processed!")
        return {}
    
    # Get video properties
    first_frame = all_frames[0]
    height, width = first_frame.shape[:2]
    
    # Determine FPS from original video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 30
    cap.release()
    
    if config.VERBOSE:
        print(f"\nGenerating intermediate videos...")
        print(f"Total frames: {len(all_frames)}")
        print(f"Frame size: {width}x{height}")
        print(f"FPS: {fps}")
    
    saved_videos = {}
    
    # Helper function to save video using ffmpeg
    def save_video_ffmpeg(frames, output_path, is_color=True):
        """Save frames as video using ffmpeg (PNG intermediate)"""
        temp_dir = os.path.join(intermediate_dir, 'temp_frames')
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save frames as PNG
            for i, frame in enumerate(tqdm(frames, desc=f"Saving {Path(output_path).name}", disable=not config.VERBOSE)):
                frame_path = os.path.join(temp_dir, f'frame_{i:06d}.png')
                cv2.imwrite(frame_path, frame)
            
            # Combine with ffmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            
        finally:
            # Clean up temp frames
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    # Save videos based on config flags
    if config.SAVE_FOREGROUND_MASK_VIDEO:
        output_path = os.path.join(intermediate_dir, f'{video_name}_foreground_mask.mp4')
        # Convert grayscale to BGR for video
        foreground_bgr = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in foreground_frames]
        save_video_ffmpeg(foreground_bgr, output_path)
        saved_videos['foreground'] = output_path
        if config.VERBOSE:
            print(f"✓ Saved: {output_path}")
    
    if config.SAVE_SHADOW_REMOVED_VIDEO:
        output_path = os.path.join(intermediate_dir, f'{video_name}_shadow_removed.mp4')
        shadow_bgr = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in shadow_removed_frames]
        save_video_ffmpeg(shadow_bgr, output_path)
        saved_videos['shadow_removed'] = output_path
        if config.VERBOSE:
            print(f"✓ Saved: {output_path}")
    
    if config.SAVE_DENOISED_VIDEO:
        output_path = os.path.join(intermediate_dir, f'{video_name}_denoised.mp4')
        denoised_bgr = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in denoised_frames]
        save_video_ffmpeg(denoised_bgr, output_path)
        saved_videos['denoised'] = output_path
        if config.VERBOSE:
            print(f"✓ Saved: {output_path}")
    
    # Also save a side-by-side comparison video
    if config.VERBOSE:
        print(f"\nCreating comparison video...")
    
    comparison_frames = []
    for i in range(len(all_frames)):
        # Create 2x2 grid: Original | Foreground
        #                   Shadow Removed | Denoised
        
        # Convert masks to BGR for visualization
        fg_vis = cv2.cvtColor(foreground_frames[i], cv2.COLOR_GRAY2BGR)
        sr_vis = cv2.cvtColor(shadow_removed_frames[i], cv2.COLOR_GRAY2BGR)
        dn_vis = cv2.cvtColor(denoised_frames[i], cv2.COLOR_GRAY2BGR)
        orig = all_frames[i]
        
        # Add labels
        def add_label(img, text):
            labeled = img.copy()
            cv2.putText(labeled, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return labeled
        
        orig_labeled = add_label(orig, "Original Masked")
        fg_labeled = add_label(fg_vis, "Foreground (MOG2)")
        sr_labeled = add_label(sr_vis, "Shadow Removed")
        dn_labeled = add_label(dn_vis, "Denoised (Final)")
        
        # Stack: top row (orig, foreground), bottom row (shadow removed, denoised)
        top_row = np.hstack([orig_labeled, fg_labeled])
        bottom_row = np.hstack([sr_labeled, dn_labeled])
        comparison = np.vstack([top_row, bottom_row])
        
        comparison_frames.append(comparison)
    
    comparison_path = os.path.join(intermediate_dir, f'{video_name}_motion_comparison.mp4')
    save_video_ffmpeg(comparison_frames, comparison_path)
    saved_videos['comparison'] = comparison_path
    
    if config.VERBOSE:
        print(f"✓ Saved: {comparison_path}")
        print(f"\nStage B Complete! Generated {len(saved_videos)} videos.")
    
    return saved_videos


if __name__ == '__main__':
    # Test motion detection
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.ball_detection import config
    
    # Test on one video
    video_name = 'cropped_test3.mp4'
    video_path = os.path.join(config.ASSETS_DIR, video_name)
    output_dir = os.path.join(config.OUTPUT_DIR, Path(video_name).stem)
    
    print(f"Testing motion detection on {video_name}")
    saved_videos = save_motion_detection_videos(video_path, config, output_dir)
    
    print("\nGenerated videos:")
    for name, path in saved_videos.items():
        print(f"  {name}: {path}")
