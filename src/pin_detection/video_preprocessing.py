"""
Video Preprocessing for Pin Detection

Creates an extended mask that reveals the pin area while keeping lane boundaries masked.
Unlike the 3-side mask from Phase 1, this extends the left/right boundaries inward
in the pin area (above top boundary) to reveal all pins without showing adjacent lanes.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm

from . import config


class PinAreaMasker:
    """
    Creates extended mask for pin area detection.
    
    The mask strategy:
    1. Below top boundary: Use standard 3-side mask (from Phase 1)
    2. Above top boundary: Extend left/right boundaries OUTWARD by N pixels
       This reveals our pins while keeping adjacent lane pins hidden
    """
    
    def __init__(self, boundary_data, video_shape, extension_pixels=None):
        """
        Initialize pin area masker.
        
        Parameters:
        -----------
        boundary_data : dict
            Boundary data from Phase 1 (boundary_data.json)
        video_shape : tuple
            (height, width) of video frames
        extension_pixels : int, optional
            Number of pixels to extend boundaries OUTWARD in pin area
            If None, uses config.PIN_AREA_UNMASK_EXTENSION
        """
        self.boundary_data = boundary_data
        self.frame_height, self.frame_width = video_shape[:2]
        self.extension = extension_pixels or config.PIN_AREA_UNMASK_EXTENSION
        
        # Extract boundary parameters
        self._parse_boundaries()
        
        # Create the extended mask
        self.mask = self._create_extended_mask()
        
        if config.VERBOSE:
            print(f"✅ PinAreaMasker initialized:")
            print(f"   Frame size: {self.frame_width}x{self.frame_height}")
            print(f"   Top boundary: y={self.top_y}")
            print(f"   Extension: {self.extension} pixels")
    
    def _parse_boundaries(self):
        """Extract boundary positions from boundary data."""
        # Top boundary (pin area)
        self.top_y = int(self.boundary_data['top_boundary']['y_position'])
        
        # Foul line (bottom boundary)
        self.foul_y = self.boundary_data['median_foul_params']['center_y']
        
        # Left boundary (master line)
        self.left_x_intersect = self.boundary_data['master_left']['x_intersect']
        self.left_slope = self.boundary_data['master_left']['slope']
        
        # Right boundary (master line)
        self.right_x_intersect = self.boundary_data['master_right']['x_intersect']
        self.right_slope = self.boundary_data['master_right']['slope']
    
    def _get_left_x_at_y(self, y):
        """Calculate left boundary x-coordinate at given y."""
        if self.left_slope == 0:
            return self.left_x_intersect
        # x = x_intersect + (y - y_intersect) / slope
        return int(self.left_x_intersect + (y - self.foul_y) / self.left_slope)
    
    def _get_right_x_at_y(self, y):
        """Calculate right boundary x-coordinate at given y."""
        if self.right_slope == 0:
            return self.right_x_intersect
        # x = x_intersect + (y - y_intersect) / slope
        return int(self.right_x_intersect + (y - self.foul_y) / self.right_slope)
    
    def _create_extended_mask(self):
        """
        Create extended mask for pin area.
        
        Returns:
        --------
        numpy.ndarray : Binary mask (255 = keep, 0 = remove)
        """
        # Start with all-white mask (keep everything)
        mask = np.ones((self.frame_height, self.frame_width), dtype=np.uint8) * 255
        
        # Create polygon points for masking
        # We'll create two regions: one for below top boundary, one for above
        
        # Region 1: Below top boundary (standard lane mask)
        # Left side - mask everything to the left of left boundary
        for y in range(self.top_y, self.frame_height):
            left_x = self._get_left_x_at_y(y)
            mask[y, :max(0, left_x)] = 0
        
        # Right side - mask everything to the right of right boundary
        for y in range(self.top_y, self.frame_height):
            right_x = self._get_right_x_at_y(y)
            mask[y, min(self.frame_width, right_x):] = 0
        
        # Region 2: Above top boundary (extended for pin area)
        # Calculate extended boundaries at top
        left_x_top = self._get_left_x_at_y(self.top_y)
        right_x_top = self._get_right_x_at_y(self.top_y)
        
        # Extend OUTWARD by extension_pixels (outside the lane to reveal pins)
        left_x_extended = left_x_top - self.extension  # Move LEFT (outward)
        right_x_extended = right_x_top + self.extension  # Move RIGHT (outward)
        
        # Calculate extended boundaries at frame top (y=0)
        left_x_frame_top = self._get_left_x_at_y(0)
        right_x_frame_top = self._get_right_x_at_y(0)
        
        # Extend OUTWARD at frame top too
        left_x_extended_top = left_x_frame_top - self.extension  # Move LEFT (outward)
        right_x_extended_top = right_x_frame_top + self.extension  # Move RIGHT (outward)
        
        # Mask left side (pin area)
        for y in range(0, self.top_y):
            # Linear interpolation between top and frame top
            t = y / max(1, self.top_y)  # 0 at frame top, 1 at top boundary
            left_x = int(left_x_extended_top + t * (left_x_extended - left_x_extended_top))
            mask[y, :max(0, left_x)] = 0
        
        # Mask right side (pin area)
        for y in range(0, self.top_y):
            t = y / max(1, self.top_y)
            right_x = int(right_x_extended_top + t * (right_x_extended - right_x_extended_top))
            mask[y, min(self.frame_width, right_x):] = 0
        
        # Mask everything below foul line
        mask[self.foul_y:, :] = 0
        
        return mask
    
    def apply_mask(self, frame):
        """
        Apply extended mask to a frame.
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input frame (BGR)
            
        Returns:
        --------
        numpy.ndarray : Masked frame
        """
        return cv2.bitwise_and(frame, frame, mask=self.mask)
    
    def visualize_mask(self, sample_frame):
        """
        Visualize the extended mask on a sample frame.
        
        Parameters:
        -----------
        sample_frame : numpy.ndarray
            Sample frame to overlay mask visualization
            
        Returns:
        --------
        numpy.ndarray : Visualization frame showing mask boundaries
        """
        vis_frame = sample_frame.copy()
        
        # Draw the masked regions in semi-transparent red
        masked_overlay = vis_frame.copy()
        masked_overlay[self.mask == 0] = [0, 0, 128]  # Dark red
        vis_frame = cv2.addWeighted(vis_frame, 0.7, masked_overlay, 0.3, 0)
        
        # Draw boundary lines
        # Top boundary (pin area)
        cv2.line(vis_frame, (0, self.top_y), (self.frame_width, self.top_y),
                config.COLOR_TOP_BOUNDARY, 2)
        
        # Foul line
        cv2.line(vis_frame, (0, self.foul_y), (self.frame_width, self.foul_y),
                config.COLOR_FOUL_LINE, 2)
        
        # Left boundary (original and extended)
        for y in range(0, self.frame_height, 5):
            left_x_orig = self._get_left_x_at_y(y)
            
            if y < self.top_y:
                # Extended boundary in pin area
                left_x_frame_top = self._get_left_x_at_y(0)
                left_x_extended_top = left_x_frame_top - self.extension  # Move LEFT (outward)
                left_x_top = self._get_left_x_at_y(self.top_y)
                left_x_extended = left_x_top - self.extension  # Move LEFT (outward)
                
                t = y / max(1, self.top_y)
                left_x = int(left_x_extended_top + t * (left_x_extended - left_x_extended_top))
                
                cv2.circle(vis_frame, (left_x, y), 2, config.COLOR_EXTENDED_AREA, -1)
            else:
                # Original boundary below top
                cv2.circle(vis_frame, (left_x_orig, y), 2, config.COLOR_LEFT_BOUNDARY, -1)
        
        # Right boundary (original and extended)
        for y in range(0, self.frame_height, 5):
            right_x_orig = self._get_right_x_at_y(y)
            
            if y < self.top_y:
                # Extended boundary in pin area
                right_x_frame_top = self._get_right_x_at_y(0)
                right_x_extended_top = right_x_frame_top + self.extension  # Move RIGHT (outward)
                right_x_top = self._get_right_x_at_y(self.top_y)
                right_x_extended = right_x_top + self.extension  # Move RIGHT (outward)
                
                t = y / max(1, self.top_y)
                right_x = int(right_x_extended_top + t * (right_x_extended - right_x_extended_top))
                
                cv2.circle(vis_frame, (right_x, y), 2, config.COLOR_EXTENDED_AREA, -1)
            else:
                # Original boundary below top
                cv2.circle(vis_frame, (right_x_orig, y), 2, config.COLOR_RIGHT_BOUNDARY, -1)
        
        # Add legend
        legend_y = 30
        cv2.putText(vis_frame, "Mask Visualization:", (10, legend_y),
                   config.FONT_FACE, config.FONT_SCALE, config.COLOR_TEXT, config.FONT_THICKNESS)
        cv2.putText(vis_frame, "Yellow: Extended boundaries (pin area)", (10, legend_y + 30),
                   config.FONT_FACE, 0.5, config.COLOR_EXTENDED_AREA, 1)
        cv2.putText(vis_frame, "Blue: Original boundaries (lane)", (10, legend_y + 55),
                   config.FONT_FACE, 0.5, config.COLOR_LEFT_BOUNDARY, 1)
        cv2.putText(vis_frame, "Dark Red: Masked regions", (10, legend_y + 80),
                   config.FONT_FACE, 0.5, (0, 0, 128), 1)
        
        return vis_frame


def create_pin_area_masked_video(video_path, boundary_data_path, output_path, 
                                  extension_pixels=None):
    """
    Create a video with extended masking for pin area detection.
    
    Parameters:
    -----------
    video_path : str
        Path to input video
    boundary_data_path : str
        Path to boundary_data.json from Phase 1
    output_path : str
        Path to save masked video
    extension_pixels : int, optional
        Pixels to extend boundaries. If None, uses config value
        
    Returns:
    --------
    str : Path to created masked video
    """
    # Load boundary data
    with open(boundary_data_path, 'r') as f:
        boundary_data = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize masker
    masker = PinAreaMasker(boundary_data, (frame_height, frame_width), extension_pixels)
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if config.VERBOSE:
        print(f"\n🎬 Creating pin area masked video...")
        print(f"   Input:  {video_path}")
        print(f"   Output: {output_path}")
        print(f"   Frames: {total_frames}")
    
    # Process video
    pbar = tqdm(total=total_frames, desc="Masking video", disable=not config.SHOW_PROGRESS_BAR)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply extended mask
        masked_frame = masker.apply_mask(frame)
        
        # Write frame
        out.write(masked_frame)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    if config.VERBOSE:
        print(f"✅ Pin area masked video created: {output_path}")
    
    # Save mask visualization if enabled
    if config.VISUALIZE_EXTENDED_MASK:
        cap = cv2.VideoCapture(video_path)
        ret, sample_frame = cap.read()
        cap.release()
        
        if ret:
            mask_vis = masker.visualize_mask(sample_frame)
            vis_path = output_path.replace('.mp4', '_mask_visualization.png')
            cv2.imwrite(vis_path, mask_vis)
            
            if config.VERBOSE:
                print(f"✅ Mask visualization saved: {vis_path}")
    
    return output_path


if __name__ == "__main__":
    # Test with cropped_test3
    test_video = 'cropped_test3'
    video_path = config.get_video_input_path(test_video)
    boundary_data_path = config.get_boundary_data_path(test_video)
    output_dir = config.get_pin_detection_output_dir(test_video)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'{test_video}_pin_area_masked.mp4')
    
    print("\n" + "="*80)
    print("TESTING PIN AREA MASKING")
    print("="*80)
    
    try:
        masked_video_path = create_pin_area_masked_video(
            video_path, 
            boundary_data_path, 
            output_path
        )
        print(f"\n✅ Success! Masked video created at:")
        print(f"   {masked_video_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
