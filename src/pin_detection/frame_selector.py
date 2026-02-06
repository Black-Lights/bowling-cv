"""
Frame Selector for Pin Detection

Selects appropriate "before" and "after" frames for pin state comparison.
Uses either fixed offsets or trajectory data to determine optimal frames.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

from . import config


class FrameSelector:
    """
    Selects before and after frames for pin detection.
    
    Two modes:
    1. Fixed offset mode: Use configured frame offsets
    2. Trajectory mode: Use ball trajectory to determine impact frame
    """
    
    def __init__(self, video_path, boundary_data=None, trajectory_data=None):
        """
        Initialize frame selector.
        
        Parameters:
        -----------
        video_path : str
            Path to masked video
        boundary_data : dict, optional
            Boundary data from Phase 1
        trajectory_data : dict, optional
            Trajectory data from Phase 2
        """
        self.video_path = video_path
        self.boundary_data = boundary_data
        self.trajectory_data = trajectory_data
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if config.VERBOSE:
            print(f"\n📹 FrameSelector initialized:")
            print(f"   Video: {os.path.basename(video_path)}")
            print(f"   Total frames: {self.total_frames}")
            print(f"   FPS: {self.fps}")
    
    def _find_impact_frame_from_trajectory(self):
        """
        Find the frame where ball impacts pins using trajectory data.
        
        Returns:
        --------
        int or None : Impact frame number, or None if not found
        """
        if self.trajectory_data is None or self.boundary_data is None:
            return None
        
        # Get top boundary y-position
        top_y = self.boundary_data['top_boundary']['y_position']
        
        # Find when ball crosses top boundary
        trajectory = self.trajectory_data.get('trajectory_original', [])
        
        for entry in trajectory:
            if entry['y'] <= top_y:
                impact_frame = entry['frame']
                if config.VERBOSE:
                    print(f"   📍 Impact detected at frame {impact_frame} (y={entry['y']:.1f})")
                return impact_frame
        
        if config.VERBOSE:
            print(f"   ⚠️  Ball did not reach pin area in trajectory data")
        
        return None
    
    def select_frames(self):
        """
        Select before and after frames for comparison.
        
        Returns:
        --------
        tuple : (before_frame_idx, after_frame_idx)
        """
        # Try trajectory-based selection if enabled
        if config.USE_TRAJECTORY_FOR_TIMING and self.trajectory_data and self.boundary_data:
            impact_frame = self._find_impact_frame_from_trajectory()
            
            if impact_frame is not None:
                # Before frame: Use configured offset before impact
                before_frame_idx = max(0, impact_frame - abs(config.BEFORE_FRAME_OFFSET))
                
                # After frame: Impact + settle time
                after_frame_idx = min(
                    self.total_frames - 1,
                    impact_frame + config.SETTLE_TIME_FRAMES
                )
                
                if config.VERBOSE:
                    print(f"\n✅ Trajectory-based frame selection:")
                    print(f"   Before frame: {before_frame_idx} (impact - {abs(config.BEFORE_FRAME_OFFSET)})")
                    print(f"   After frame:  {after_frame_idx} (impact + {config.SETTLE_TIME_FRAMES})")
                
                return before_frame_idx, after_frame_idx
        
        # Fall back to fixed offset mode
        before_frame_idx = config.BEFORE_FRAME_OFFSET
        
        if config.AFTER_FRAME_OFFSET < 0:
            after_frame_idx = self.total_frames + config.AFTER_FRAME_OFFSET
        else:
            after_frame_idx = config.AFTER_FRAME_OFFSET
        
        # Validate indices
        before_frame_idx = max(0, min(before_frame_idx, self.total_frames - 1))
        after_frame_idx = max(0, min(after_frame_idx, self.total_frames - 1))
        
        if config.VERBOSE:
            print(f"\n✅ Fixed offset frame selection:")
            print(f"   Before frame: {before_frame_idx}")
            print(f"   After frame:  {after_frame_idx}")
        
        return before_frame_idx, after_frame_idx
    
    def extract_frame(self, frame_idx):
        """
        Extract a specific frame from video.
        
        Parameters:
        -----------
        frame_idx : int
            Frame index to extract
            
        Returns:
        --------
        numpy.ndarray : Extracted frame (BGR)
        """
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to extract frame {frame_idx}")
        
        return frame
    
    def extract_frames(self):
        """
        Extract before and after frames.
        
        Returns:
        --------
        tuple : (before_frame, after_frame, before_idx, after_idx)
        """
        # Select frame indices
        before_idx, after_idx = self.select_frames()
        
        # Extract frames
        if config.VERBOSE:
            print(f"\n📸 Extracting frames...")
        
        before_frame = self.extract_frame(before_idx)
        after_frame = self.extract_frame(after_idx)
        
        if config.VERBOSE:
            print(f"   ✅ Before frame: {before_frame.shape}")
            print(f"   ✅ After frame:  {after_frame.shape}")
        
        return before_frame, after_frame, before_idx, after_idx
    
    def apply_focus_roi(self, frame):
        """
        Apply top-focused ROI to frame (only analyze top portion).
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input frame
            
        Returns:
        --------
        numpy.ndarray : ROI frame (top portion only)
        """
        if config.FOCUS_TOP_FRACTION >= 1.0:
            return frame
        
        roi_height = int(self.frame_height * config.FOCUS_TOP_FRACTION)
        return frame[:roi_height, :]
    
    def visualize_frame_selection(self, before_frame, after_frame, 
                                  before_idx, after_idx, output_path):
        """
        Create visualization showing before/after frame selection.
        
        Parameters:
        -----------
        before_frame : numpy.ndarray
            Before frame
        after_frame : numpy.ndarray
            After frame
        before_idx : int
            Before frame index
        after_idx : int
            After frame index
        output_path : str
            Path to save visualization
        """
        # Create side-by-side comparison
        h, w = before_frame.shape[:2]
        
        # Add focus ROI visualization if enabled
        before_vis = before_frame.copy()
        after_vis = after_frame.copy()
        
        if config.FOCUS_TOP_FRACTION < 1.0:
            roi_height = int(h * config.FOCUS_TOP_FRACTION)
            
            # Draw ROI boundary
            cv2.line(before_vis, (0, roi_height), (w, roi_height), 
                    (0, 255, 255), 2)  # Cyan line
            cv2.line(after_vis, (0, roi_height), (w, roi_height), 
                    (0, 255, 255), 2)
            
            # Add semi-transparent overlay below ROI
            overlay_before = before_vis.copy()
            overlay_after = after_vis.copy()
            cv2.rectangle(overlay_before, (0, roi_height), (w, h), (0, 0, 0), -1)
            cv2.rectangle(overlay_after, (0, roi_height), (w, h), (0, 0, 0), -1)
            before_vis = cv2.addWeighted(before_vis, 0.7, overlay_before, 0.3, 0)
            after_vis = cv2.addWeighted(after_vis, 0.7, overlay_after, 0.3, 0)
        
        # Add labels
        cv2.putText(before_vis, f"BEFORE (Frame {before_idx})", (10, 30),
                   config.FONT_FACE, 0.8, config.COLOR_TEXT, 2)
        cv2.putText(after_vis, f"AFTER (Frame {after_idx})", (10, 30),
                   config.FONT_FACE, 0.8, config.COLOR_TEXT, 2)
        
        if config.FOCUS_TOP_FRACTION < 1.0:
            cv2.putText(before_vis, f"Analysis ROI: Top {config.FOCUS_TOP_FRACTION*100:.0f}%", 
                       (10, 60), config.FONT_FACE, 0.6, (0, 255, 255), 2)
            cv2.putText(after_vis, f"Analysis ROI: Top {config.FOCUS_TOP_FRACTION*100:.0f}%", 
                       (10, 60), config.FONT_FACE, 0.6, (0, 255, 255), 2)
        
        # Create side-by-side comparison
        comparison = np.hstack([before_vis, after_vis])
        
        # Add title banner
        banner_height = 50
        banner = np.zeros((banner_height, comparison.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, "Frame Selection for Pin Detection", 
                   (comparison.shape[1]//2 - 250, 35),
                   config.FONT_FACE, 0.9, config.COLOR_TEXT, 2)
        
        final_vis = np.vstack([banner, comparison])
        
        # Save
        cv2.imwrite(output_path, final_vis)
        
        if config.VERBOSE:
            print(f"   ✅ Frame selection visualization saved: {output_path}")


def select_and_extract_frames(video_path, boundary_data_path=None, 
                              trajectory_data_path=None, output_dir=None):
    """
    Select and extract before/after frames for pin detection.
    
    Parameters:
    -----------
    video_path : str
        Path to masked video
    boundary_data_path : str, optional
        Path to boundary_data.json
    trajectory_data_path : str, optional
        Path to trajectory_data.json
    output_dir : str, optional
        Directory to save frames and visualizations
        
    Returns:
    --------
    tuple : (before_frame, after_frame, before_idx, after_idx)
    """
    # Load data if provided
    boundary_data = None
    trajectory_data = None
    
    if boundary_data_path and os.path.exists(boundary_data_path):
        with open(boundary_data_path, 'r') as f:
            boundary_data = json.load(f)
    
    if trajectory_data_path and os.path.exists(trajectory_data_path):
        with open(trajectory_data_path, 'r') as f:
            trajectory_data = json.load(f)
    
    # Initialize selector
    selector = FrameSelector(video_path, boundary_data, trajectory_data)
    
    # Extract frames
    before_frame, after_frame, before_idx, after_idx = selector.extract_frames()
    
    # Apply focus ROI
    before_roi = selector.apply_focus_roi(before_frame)
    after_roi = selector.apply_focus_roi(after_frame)
    
    # Save frames if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual frames
        before_path = os.path.join(output_dir, f'before_frame_{before_idx}.png')
        after_path = os.path.join(output_dir, f'after_frame_{after_idx}.png')
        cv2.imwrite(before_path, before_frame)
        cv2.imwrite(after_path, after_frame)
        
        # Save ROI frames
        before_roi_path = os.path.join(output_dir, f'before_frame_{before_idx}_roi.png')
        after_roi_path = os.path.join(output_dir, f'after_frame_{after_idx}_roi.png')
        cv2.imwrite(before_roi_path, before_roi)
        cv2.imwrite(after_roi_path, after_roi)
        
        if config.VERBOSE:
            print(f"\n💾 Frames saved:")
            print(f"   Before: {before_path}")
            print(f"   After:  {after_path}")
            print(f"   Before ROI: {before_roi_path}")
            print(f"   After ROI:  {after_roi_path}")
        
        # Create visualization if enabled
        if config.VISUALIZE_FRAME_SELECTION:
            vis_path = os.path.join(output_dir, 'frame_selection_visualization.png')
            selector.visualize_frame_selection(
                before_frame, after_frame, before_idx, after_idx, vis_path
            )
    
    return before_roi, after_roi, before_idx, after_idx


if __name__ == "__main__":
    # Test with cropped_test3
    test_video = 'cropped_test3'
    masked_video_path = os.path.join(
        config.get_pin_detection_output_dir(test_video),
        f'{test_video}_pin_area_masked.mp4'
    )
    boundary_data_path = config.get_boundary_data_path(test_video)
    trajectory_data_path = config.get_trajectory_data_path(test_video)
    output_dir = config.get_intermediate_output_dir(test_video)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("TESTING FRAME SELECTION")
    print("="*80)
    
    try:
        before, after, before_idx, after_idx = select_and_extract_frames(
            masked_video_path,
            boundary_data_path,
            trajectory_data_path,
            output_dir
        )
        
        print(f"\n✅ Success!")
        print(f"   Before frame shape: {before.shape}")
        print(f"   After frame shape:  {after.shape}")
        print(f"   Before index: {before_idx}")
        print(f"   After index:  {after_idx}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
