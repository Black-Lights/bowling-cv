"""
Lane Detection Module - Complete Pipeline for Phase 1

This module provides the LaneDetector class for detecting all 4 boundaries
of a bowling lane: bottom (foul line), left/right (master lines), and top (pin area).

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: January 29, 2026
Last Updated: January 30, 2026
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Import detection modules
from .detection_functions import detect_horizontal_line, detect_vertical_boundaries_approach1
from .master_line_computation import compute_master_line_from_collection, visualize_bin_analysis
from .top_boundary_detection import (
    detect_top_boundary_all_frames,
    fit_msac_line,
    create_visualization_videos,
    plot_intersection_y_coordinates
)
from .mask_lane_area import apply_mask_to_video
from .preprocess_frames import create_preprocessed_video
from .tracking_analysis import analyze_master_line_tracking, plot_master_line_tracking


class LaneDetector:
    """
    Complete lane detection pipeline for bowling videos.
    
    Detects all 4 boundaries of the bowling lane:
    - Bottom boundary: Foul line (horizontal)
    - Left boundary: Left master line (vertical)
    - Right boundary: Right master line (vertical)
    - Top boundary: Pin area boundary (horizontal)
    
    Also calculates all 4 intersection points where horizontal lines
    cross the vertical master lines.
    
    Attributes:
        video_path (str): Path to input video file
        config: Configuration module with detection parameters
        output_dir (str): Directory for output files
        video_name (str): Name of video (without extension)
        
        # Detected boundaries
        boundaries (dict): All detected boundary parameters
        intersections (dict): All 4 intersection points
        
        # Processing state
        _bottom_detected (bool): Whether bottom boundary is detected
        _sides_detected (bool): Whether left/right boundaries are detected
        _top_detected (bool): Whether top boundary is detected
        
    Example:
        >>> from lane_detection import LaneDetector
        >>> import config
        >>> 
        >>> detector = LaneDetector('video.mp4', config)
        >>> detector.detect_all()  # Run complete pipeline
        >>> detector.save()  # Save all outputs
        >>> 
        >>> # Access results
        >>> print(detector.boundaries)
        >>> print(detector.intersections)
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, video_path: str, config):
        """
        Initialize LaneDetector with video and configuration.
        
        Args:
            video_path (str): Path to input bowling video
            config: Configuration module with detection parameters
            
        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        # Validate video path
        if not os.path.isabs(video_path):
            # Try ASSETS_DIR if it exists, otherwise use current directory
            assets_dir = getattr(config, 'ASSETS_DIR', os.getcwd())
            video_path = os.path.join(assets_dir, video_path)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.video_path = video_path
        self.config = config
        
        # Setup output directory
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = getattr(config, 'OUTPUT_DIR', 'output')
        self.output_dir = os.path.join(output_dir, self.video_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize boundary storage
        self.boundaries = {
            'bottom': None,  # Foul line parameters
            'left': None,    # Left master line
            'right': None,   # Right master line
            'top': None      # Top boundary (MSAC line)
        }
        
        # Initialize intersection storage
        self.intersections = {
            'top_left': None,     # Top line ∩ Left master
            'top_right': None,    # Top line ∩ Right master
            'bottom_left': None,  # Bottom line ∩ Left master
            'bottom_right': None  # Bottom line ∩ Right master
        }
        
        # Detection state flags
        self._bottom_detected = False
        self._sides_detected = False
        self._top_detected = False
        
        # Storage for intermediate files (for cleanup)
        self._temp_files = []
        
        print(f"\n{'='*70}")
        print(f"LaneDetector v{self.VERSION} initialized")
        print(f"Video: {self.video_name}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")
    
    
    def detect_all(self) -> Tuple[Dict, Dict]:
        """
        Run complete lane detection pipeline.
        
        Detects all 4 boundaries and calculates all intersection points.
        This is the main entry point for the complete pipeline.
        
        Returns:
            tuple: (boundaries dict, intersections dict)
            
        Example:
            >>> detector = LaneDetector('video.mp4', config)
            >>> boundaries, intersections = detector.detect_all()
        """
        print(f"\n{'#'*70}")
        print(f"# PHASE 1: COMPLETE LANE DETECTION PIPELINE")
        print(f"# Video: {self.video_name}")
        print(f"{'#'*70}\n")
        
        # Step 1: Detect bottom boundary (foul line)
        self.detect_bottom_boundary()
        
        # Step 2: Detect side boundaries (left/right master lines)
        self.detect_side_boundaries()
        
        # Step 3: Detect top boundary (pin area)
        self.detect_top_boundary()
        
        # Step 4: Calculate all intersection points
        self.calculate_intersections()
        
        print(f"\n{'='*70}")
        print(f"✅ PHASE 1 COMPLETE - All 4 boundaries detected")
        print(f"{'='*70}\n")
        
        return self.boundaries, self.intersections
    
    
    def detect_bottom_boundary(self):
        """
        Detect bottom boundary (foul line) from first N frames.
        
        Uses horizontal line detection with voting system to find
        the most consistent foul line position across frames.
        
        Updates:
            self.boundaries['bottom']: Foul line parameters
            self._bottom_detected: Set to True
        """
        if self._bottom_detected:
            print("  Bottom boundary already detected. Skipping.")
            return
        
        print(f"{'='*70}")
        print(f"STEP 1: Detecting Bottom Boundary (Foul Line)")
        print(f"{'='*70}")
        
        # Read first frame to get foul line
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Could not read video frame")
        
        # Detect horizontal foul line
        _, _, foul_params = detect_horizontal_line(frame)
        
        if foul_params is None:
            raise RuntimeError("Could not detect foul line")
        
        self.boundaries['bottom'] = foul_params
        self._bottom_detected = True
        
        print(f"✅ Bottom boundary detected at Y={foul_params['center_y']}")
        print()
    
    
    def detect_side_boundaries(self):
        """
        Detect side boundaries (left & right master lines).
        
        Collects vertical lines from first N frames and uses voting system
        to compute robust master lines for left and right lane boundaries.
        
        Dependencies:
            - Requires bottom boundary (auto-runs if not detected)
        
        Updates:
            self.boundaries['left']: Left master line parameters
            self.boundaries['right']: Right master line parameters
            self._sides_detected: Set to True
        """
        # Check dependencies
        if not self._bottom_detected:
            print("  Bottom boundary required. Running detect_bottom_boundary()...")
            self.detect_bottom_boundary()
        
        if self._sides_detected:
            print("  Side boundaries already detected. Skipping.")
            return
        
        print(f"{'='*70}")
        print(f"STEP 2: Detecting Side Boundaries (Master Lines)")
        print(f"{'='*70}")
        
        # Collect lines from first N frames
        collection_data = self._collect_side_lines()
        
        # Compute master lines from collected lines
        angle_mode = 'from_vertical' if self.config.USE_ABSOLUTE_ANGLES else 'from_horizontal'
        
        master_left, debug_left = compute_master_line_from_collection(
            collection_data['left_lines'],
            collection_data['median_foul_params'],
            bin_width=self.config.BIN_WIDTH,
            vote_threshold=self.config.VOTE_THRESHOLD,
            angle_tolerance=self.config.ANGLE_TOLERANCE,
            side='left',
            angle_mode=angle_mode
        )
        
        master_right, debug_right = compute_master_line_from_collection(
            collection_data['right_lines'],
            collection_data['median_foul_params'],
            bin_width=self.config.BIN_WIDTH,
            vote_threshold=self.config.VOTE_THRESHOLD,
            angle_tolerance=self.config.ANGLE_TOLERANCE,
            side='right',
            angle_mode=angle_mode
        )
        
        if master_left is None or master_right is None:
            raise RuntimeError("Could not compute master lines")
        
        self.boundaries['left'] = master_left
        self.boundaries['right'] = master_right
        self._sides_detected = True
        
        # Save bin analysis plots if configured
        if self.config.SAVE_BIN_ANALYSIS_PLOTS:
            visualize_bin_analysis(debug_left, 'left', 
                                  os.path.join(self.output_dir, 'bin_analysis_left.png'),
                                  angle_mode)
            visualize_bin_analysis(debug_right, 'right',
                                  os.path.join(self.output_dir, 'bin_analysis_right.png'),
                                  angle_mode)
        
        print(f"✅ Side boundaries detected:")
        print(f"   Left: X={master_left['x_intersect']}, Angle={master_left['median_angle']:.1f}°")
        print(f"   Right: X={master_right['x_intersect']}, Angle={master_right['median_angle']:.1f}°")
        print()
    
    
    def detect_top_boundary(self):
        """
        Detect top boundary (pin area) using Sobel edge detection and MSAC.
        
        Pipeline:
        1. Create masked video (lane area only)
        2. Preprocess with HSV filtering + gap filling
        3. Detect top boundary in all frames using Sobel
        4. Fit MSAC line from all detections
        5. Generate visualization videos
        
        Dependencies:
            - Requires bottom boundary (auto-runs if not detected)
            - Requires side boundaries (auto-runs if not detected)
        
        Updates:
            self.boundaries['top']: Top boundary MSAC line parameters
            self._top_detected: Set to True
        """
        # Check dependencies
        if not self._bottom_detected:
            print("  Bottom boundary required. Running detect_bottom_boundary()...")
            self.detect_bottom_boundary()
        
        if not self._sides_detected:
            print("  Side boundaries required. Running detect_side_boundaries()...")
            self.detect_side_boundaries()
        
        if self._top_detected:
            print("  Top boundary already detected. Skipping.")
            return
        
        print(f"{'='*70}")
        print(f"STEP 3: Detecting Top Boundary (Pin Area)")
        print(f"{'='*70}")
        
        # Create intermediate folder if needed
        intermediate_dir = os.path.join(self.output_dir, self.config.INTERMEDIATE_FOLDER)
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # Step 3.1: Process frames directly (no video creation unless saving)
        print("\n  Processing frames...")
        
        # Load and process all frames
        masked_frames = self._process_masked_frames()
        preprocessed_frames = self._preprocess_frames(masked_frames)
        
        # Save videos only if configured
        if self.config.SAVE_MASKED_VIDEO:
            masked_video_path = os.path.join(self.output_dir, f'masked_{self.video_name}.mp4')
            print(f"  Saving masked video to {os.path.basename(masked_video_path)}...")
            self._save_frames_as_video(masked_frames, masked_video_path)
        
        if self.config.SAVE_PREPROCESSED_VIDEO:
            preprocessed_video_path = os.path.join(self.output_dir, f'preprocessed_{self.video_name}.mp4')
            print(f"  Saving preprocessed video to {os.path.basename(preprocessed_video_path)}...")
            self._save_frames_as_video(preprocessed_frames, preprocessed_video_path)
        
        # Step 3.2: Detect top boundary from frames
        print("  Detecting top boundary using Sobel edge detection...")
        detections = self._detect_top_boundary_from_frames(preprocessed_frames)
        
        # Step 3.3: Fit MSAC line from all detections
        cap = cv2.VideoCapture(self.video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        
        msac_line = fit_msac_line(detections, video_width, self.output_dir, self.video_name)
        
        self.boundaries['top'] = msac_line
        self._top_detected = True
        
        # Step 3.4: Generate visualization videos
        print("\n  Generating visualization videos...")
        boundary_data = self._get_boundary_data_dict()
        
        self._create_visualization_videos_from_frames(
            preprocessed_frames,
            detections,
            boundary_data,
            msac_line
        )
        
        # Step 3.5: Generate intersection plot if configured
        if self.config.SAVE_INTERSECTION_PLOTS:
            plot_intersection_y_coordinates(detections, boundary_data, self.video_name, self.output_dir)
        
        print(f"✅ Top boundary detected:")
        print(f"   MSAC line: Y={msac_line['y_position']:.1f}")
        print(f"   Inliers: {msac_line['n_inliers']}/{msac_line['n_inliers']+msac_line['n_outliers']}")
        print()
    
    
    def calculate_intersections(self):
        """
        Calculate all 4 intersection points.
        
        Calculates where horizontal boundaries (top/bottom) intersect
        with vertical master lines (left/right).
        
        Dependencies:
            - Requires all boundaries detected (auto-runs if needed)
        
        Updates:
            self.intersections: Dict with all 4 intersection points
            
        Intersection Points:
            - top_left: Top line ∩ Left master line
            - top_right: Top line ∩ Right master line
            - bottom_left: Bottom line ∩ Left master line
            - bottom_right: Bottom line ∩ Right master line
        """
        # Check dependencies
        if not self._bottom_detected or not self._sides_detected or not self._top_detected:
            print("  All boundaries required for intersection calculation.")
            if not self._bottom_detected or not self._sides_detected:
                print("  Running detect_all()...")
                self.detect_all()
                return
        
        print(f"{'='*70}")
        print(f"STEP 4: Calculating Intersection Points")
        print(f"{'='*70}\n")
        
        # Get line parameters
        left_x_intersect = self.boundaries['left']['x_intersect']
        left_slope = self.boundaries['left']['slope']
        
        right_x_intersect = self.boundaries['right']['x_intersect']
        right_slope = self.boundaries['right']['slope']
        
        top_y = self.boundaries['top']['y_position']
        bottom_y = self.boundaries['bottom']['center_y']
        
        # Calculate top intersections
        # Master line equation: x = x_intersect + y * slope
        # Solve for y when line is at specific Y
        
        # Top-Left: Where does left master line cross top boundary?
        top_left_y = (self.boundaries['top']['line_left'][0] - left_x_intersect) / left_slope if left_slope != 0 else top_y
        self.intersections['top_left'] = {
            'x': self.boundaries['top']['line_left'][0],
            'y': int(top_left_y),
            'description': 'Top boundary ∩ Left master line'
        }
        
        # Top-Right: Where does right master line cross top boundary?
        top_right_y = (self.boundaries['top']['line_right'][0] - right_x_intersect) / right_slope if right_slope != 0 else top_y
        self.intersections['top_right'] = {
            'x': self.boundaries['top']['line_right'][0],
            'y': int(top_right_y),
            'description': 'Top boundary ∩ Right master line'
        }
        
        # Bottom intersections (foul line ∩ master lines)
        # Calculate X where master line crosses foul line Y
        bottom_left_x = int(left_x_intersect + bottom_y * left_slope)
        self.intersections['bottom_left'] = {
            'x': bottom_left_x,
            'y': bottom_y,
            'description': 'Bottom boundary (foul) ∩ Left master line'
        }
        
        bottom_right_x = int(right_x_intersect + bottom_y * right_slope)
        self.intersections['bottom_right'] = {
            'x': bottom_right_x,
            'y': bottom_y,
            'description': 'Bottom boundary (foul) ∩ Right master line'
        }
        
        print("  Intersection Points:")
        print(f"    Top-Left:     ({self.intersections['top_left']['x']}, {self.intersections['top_left']['y']})")
        print(f"    Top-Right:    ({self.intersections['top_right']['x']}, {self.intersections['top_right']['y']})")
        print(f"    Bottom-Left:  ({self.intersections['bottom_left']['x']}, {self.intersections['bottom_left']['y']})")
        print(f"    Bottom-Right: ({self.intersections['bottom_right']['x']}, {self.intersections['bottom_right']['y']})")
        print()
    
    
    def save(self):
        """
        Save all detection results and clean up temporary files.
        
        Saves:
            - boundary_data.json: All boundary parameters and intersections
            - Configured plots (bin analysis, MSAC, tracking)
            - Final videos
        
        Cleans up:
            - Temporary files (masked, preprocessed videos) if configured
        """
        print(f"{'='*70}")
        print(f"SAVING RESULTS")
        print(f"{'='*70}\n")
        
        # Save boundary data JSON
        boundary_data = self._get_boundary_data_dict()
        boundary_file = os.path.join(self.output_dir, 'boundary_data.json')
        
        with open(boundary_file, 'w') as f:
            json.dump(boundary_data, f, indent=2)
        
        print(f"  ✅ Saved boundary_data.json")
        
        # Save tracking analysis if configured
        if self.config.SAVE_TRACKING_PLOTS:
            tracking_data = analyze_master_line_tracking(
                self.video_path,
                self.boundaries['left'],
                self.boundaries['right'],
                self.boundaries['bottom']
            )
            plot_master_line_tracking(
                tracking_data,
                self.video_name,
                self.output_dir,
                self.boundaries['left'],
                self.boundaries['right']
            )
            print(f"  ✅ Saved tracking analysis plot")
        
        # Generate intermediate videos if configured
        if self.config.SAVE_INTERMEDIATE_VIDEOS:
            self._generate_intermediate_videos()
        
        # Cleanup temporary files
        self._cleanup_temp_files()
        
        print(f"\n{'='*70}")
        print(f"✅ All results saved to: {self.output_dir}")
        print(f"{'='*70}\n")
    
    
    def _collect_side_lines(self) -> Dict:
        """
        Internal: Collect vertical lines from first N frames.
        
        Returns:
            dict: Collection data with left/right lines and foul params
        """
        from tqdm import tqdm
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = min(self.config.NUM_COLLECTION_FRAMES, total_frames)
        
        left_lines = []
        right_lines = []
        foul_params_list = []
        
        angle_mode = 'from_vertical' if self.config.USE_ABSOLUTE_ANGLES else 'from_horizontal'
        
        for frame_idx in tqdm(range(frames_to_process), desc="  Collecting lines"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect lines in this frame
            left_boundary, right_boundary, _, left_lines_frame, right_lines_frame = \
                detect_vertical_boundaries_approach1(frame, self.boundaries['bottom'], angle_mode)
            
            if left_lines_frame:
                left_lines.extend(left_lines_frame)
            if right_lines_frame:
                right_lines.extend(right_lines_frame)
            
            _, _, foul_params = detect_horizontal_line(frame)
            if foul_params:
                foul_params_list.append(foul_params)
        
        cap.release()
        
        # Calculate median foul parameters
        if foul_params_list:
            median_foul_params = {
                'center_y': int(np.median([fp['center_y'] for fp in foul_params_list])),
                'slope': np.median([fp['slope'] for fp in foul_params_list]),
                'center_x': foul_params_list[0]['center_x'],
                'width': foul_params_list[0]['width'],
                'height': foul_params_list[0]['height']
            }
        else:
            median_foul_params = self.boundaries['bottom']
        
        print(f"  Collected {len(left_lines)} left lines, {len(right_lines)} right lines")
        
        return {
            'left_lines': left_lines,
            'right_lines': right_lines,
            'foul_params_list': foul_params_list,
            'median_foul_params': median_foul_params
        }
    
    
    def _get_boundary_data_dict(self) -> Dict:
        """
        Internal: Get boundary data in JSON-serializable format.
        
        Returns:
            dict: All boundary parameters and intersections
        """
        return {
            'video_name': self.video_name,
            'version': self.VERSION,
            'master_left': {
                'x_top': int(self.boundaries['left']['x_top']),
                'y_top': int(self.boundaries['left']['y_top']),
                'x_bottom': int(self.boundaries['left']['x_bottom']),
                'y_bottom': int(self.boundaries['left']['y_bottom']),
                'x_intersect': int(self.boundaries['left']['x_intersect']),
                'slope': float(self.boundaries['left']['slope']),
                'median_angle': float(self.boundaries['left'].get('median_angle', 0))
            },
            'master_right': {
                'x_top': int(self.boundaries['right']['x_top']),
                'y_top': int(self.boundaries['right']['y_top']),
                'x_bottom': int(self.boundaries['right']['x_bottom']),
                'y_bottom': int(self.boundaries['right']['y_bottom']),
                'x_intersect': int(self.boundaries['right']['x_intersect']),
                'slope': float(self.boundaries['right']['slope']),
                'median_angle': float(self.boundaries['right'].get('median_angle', 0))
            },
            'median_foul_params': {
                'center_y': int(self.boundaries['bottom']['center_y']),
                'slope': float(self.boundaries['bottom']['slope']),
                'y_left': int(self.boundaries['bottom'].get('y_left', 0)),
                'y_right': int(self.boundaries['bottom'].get('y_right', 0))
            },
            'top_boundary': {
                'line_left': list(self.boundaries['top']['line_left']),
                'line_right': list(self.boundaries['top']['line_right']),
                'y_position': float(self.boundaries['top']['y_position']),
                'n_inliers': int(self.boundaries['top']['n_inliers']),
                'n_outliers': int(self.boundaries['top']['n_outliers'])
            },
            'intersections': self.intersections
        }
    
    
    def _cleanup_temp_files(self):
        """
        Internal: Delete temporary files based on configuration.
        """
        if not self._temp_files:
            return
        
        print(f"\n  Cleaning up temporary files...")
        for temp_file in self._temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"    🗑️  Deleted: {os.path.basename(temp_file)}")
                except Exception as e:
                    print(f"    ⚠️  Could not delete {os.path.basename(temp_file)}: {e}")
        
        self._temp_files.clear()
    
    
    def _process_masked_frames(self):
        """
        Internal: Apply lane mask to all frames and return frame array.
        Much faster than creating video when we don't need to save it.
        """
        from tqdm import tqdm
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create mask polygon
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        left_x_top = self.boundaries['left']['x_top']
        left_y_top = self.boundaries['left']['y_top']
        left_x_bottom = self.boundaries['left']['x_bottom']
        left_y_bottom = self.boundaries['left']['y_bottom']
        
        right_x_top = self.boundaries['right']['x_top']
        right_y_top = self.boundaries['right']['y_top']
        right_x_bottom = self.boundaries['right']['x_bottom']
        right_y_bottom = self.boundaries['right']['y_bottom']
        
        polygon = np.array([
            [left_x_bottom, left_y_bottom],
            [left_x_top, left_y_top],
            [right_x_top, right_y_top],
            [right_x_bottom, right_y_bottom]
        ], dtype=np.int32)
        
        masked_frames = []
        with tqdm(total=total_frames, desc="  Masking frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply mask
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 255)
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                
                masked_frames.append(masked_frame)
                pbar.update(1)
        
        cap.release()
        return masked_frames
    
    
    def _preprocess_frames(self, frames):
        """
        Internal: Preprocess frames with HSV filtering and gap filling.
        
        Supports caching: if SAVE_PREPROCESSED_FRAMES=True and cache exists, loads from cache.
        Otherwise processes frames and saves cache if flag is True.
        
        Args:
            frames: List of frames to preprocess
            
        Returns:
            list: List of preprocessed frames
        """
        from tqdm import tqdm
        from .preprocess_frames import preprocess_frame_hsv
        
        # Check if cached frames exist
        cache_path = os.path.join(self.output_dir, 'preprocessed_frames.npz')
        if self.config.SAVE_PREPROCESSED_FRAMES and os.path.exists(cache_path):
            print(f"  Loading preprocessed frames from cache: {os.path.basename(cache_path)}")
            data = np.load(cache_path)
            preprocessed_frames = [data[f'frame_{i}'] for i in range(len(data.files))]
            print(f"  ✓ Loaded {len(preprocessed_frames)} frames from cache (skipped ~4 min processing!)")
            return preprocessed_frames
        
        preprocessed = []
        with tqdm(total=len(frames), desc="  Preprocessing frames") as pbar:
            for frame in frames:
                processed = preprocess_frame_hsv(
                    frame,
                    self.config.MAX_PATCH_SIZE_ROW,
                    self.config.MAX_PATCH_SIZE_COL,
                    self.config.TOP_REGION_RATIO,
                    self.config.MAX_TOP_PATCH_AREA
                )
                preprocessed.append(processed)
                pbar.update(1)
        
        # Save cache if flag is True
        if self.config.SAVE_PREPROCESSED_FRAMES:
            print(f"  Saving preprocessed frames to cache: {os.path.basename(cache_path)}")
            # Save as compressed npz with individual frames
            save_dict = {f'frame_{i}': frame for i, frame in enumerate(preprocessed)}
            np.savez_compressed(cache_path, **save_dict)
            file_size_mb = os.path.getsize(cache_path) / 1024 / 1024
            print(f"  ✓ Cache saved ({file_size_mb:.1f} MB)")
        
        return preprocessed
    
    
    def _detect_top_boundary_from_frames(self, frames):
        """
        Internal: Detect top boundary from frame array.
        """
        from tqdm import tqdm
        from .top_boundary_detection import detect_top_boundary_sobel
        
        detections = []
        with tqdm(total=len(frames), desc="  Detecting boundaries") as pbar:
            for frame in frames:
                detection = detect_top_boundary_sobel(frame, self.config)
                detections.append(detection)
                pbar.update(1)
        
        return detections
    
    
    def _save_frames_as_video(self, frames, output_path):
        """
        Internal: Save frame array as video file.
        """
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    
    def _create_visualization_videos_from_frames(self, preprocessed_frames, detections, boundary_data, msac_line):
        """
        Internal: Create visualization videos from frame arrays.
        Only creates videos that are configured to be saved.
        """
        from tqdm import tqdm
        
        # Get original video FPS
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Load original frames for final video (always created)
        cap = cv2.VideoCapture(self.video_path)
        original_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            original_frames.append(frame)
        cap.release()
        
        # Prepare output paths
        videos_to_create = []
        
        # Sobel video (optional)
        if self.config.SAVE_SOBEL_VIDEO:
            videos_to_create.append(('sobel', os.path.join(self.output_dir, f'top_vis_sobel_{self.video_name}.mp4')))
        
        # Masked video (optional)
        if self.config.SAVE_TOP_MASKED_VIDEO:
            videos_to_create.append(('masked', os.path.join(self.output_dir, f'top_vis_masked_{self.video_name}.mp4')))
        
        # Final video (always)
        videos_to_create.append(('final', os.path.join(self.output_dir, f'final_all_boundaries_{self.video_name}.mp4')))
        
        # Generate frames for each video type
        for video_type, output_path in videos_to_create:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            with tqdm(total=len(preprocessed_frames), desc=f"  Creating {video_type} video") as pbar:
                for i, (preprocessed_frame, detection) in enumerate(zip(preprocessed_frames, detections)):
                    if video_type == 'sobel':
                        # Sobel visualization with red heatmap (COLORMAP_HOT)
                        gray = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)
                        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.config.SOBEL_KERNEL_SIZE)
                        sobel_y = np.abs(sobel_y)
                        sobel_normalized = (sobel_y / sobel_y.max() * 255).astype(np.uint8) if sobel_y.max() > 0 else sobel_y.astype(np.uint8)
                        vis_frame = cv2.applyColorMap(sobel_normalized, cv2.COLORMAP_HOT)
                        
                        # Draw detected line in cyan (per-frame detection)
                        if detection:
                            cv2.line(vis_frame, detection['line_left'], detection['line_right'], (255, 255, 0), 2)
                        
                        # Draw MSAC line in bright green
                        top_y = int(msac_line['y_position'])
                        cv2.line(vis_frame, (0, top_y), (width-1, top_y), (0, 255, 0), 3)
                        
                    elif video_type == 'masked':
                        vis_frame = preprocessed_frame.copy()
                        # Draw MSAC line in green
                        top_y = int(msac_line['y_position'])
                        cv2.line(vis_frame, (0, top_y), (width-1, top_y), (0, 255, 0), 3)
                    
                    else:  # final
                        vis_frame = original_frames[i].copy()
                        
                        # Draw all 4 boundaries
                        # Bottom (foul line)
                        foul_y = boundary_data['median_foul_params']['center_y']
                        cv2.line(vis_frame, (0, foul_y), (width-1, foul_y), (0, 0, 255), 2)
                        
                        # Left master line
                        left = boundary_data['master_left']
                        cv2.line(vis_frame, (left['x_bottom'], left['y_bottom']), 
                                (left['x_top'], left['y_top']), (255, 0, 0), 2)
                        
                        # Right master line
                        right = boundary_data['master_right']
                        cv2.line(vis_frame, (right['x_bottom'], right['y_bottom']),
                                (right['x_top'], right['y_top']), (255, 0, 0), 2)
                        
                        # Top boundary (MSAC line)
                        top_y = int(msac_line['y_position'])
                        cv2.line(vis_frame, (0, top_y), (width-1, top_y), (0, 255, 0), 2)
                    
                    out.write(vis_frame)
                    pbar.update(1)
            
            out.release()
            print(f"    ✓ Saved: {os.path.basename(output_path)}")
    
    
    def _generate_intermediate_videos(self):
        """
        Internal: Generate intermediate visualization videos for debugging.
        
        Creates videos showing different processing stages (edges, gaussian, etc.)
        based on INTERMEDIATE_MODES configuration.
        """
        from tqdm import tqdm
        from .intermediate_visualization import get_horizontal_intermediates, get_vertical_intermediates
        
        print(f"\n  Generating intermediate visualization videos...")
        
        intermediate_dir = os.path.join(self.output_dir, self.config.INTERMEDIATE_FOLDER)
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # Get video properties
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process each mode
        for mode in self.config.INTERMEDIATE_MODES:
            output_path = os.path.join(intermediate_dir, f"{mode}_{self.video_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            
            with tqdm(total=total_frames, desc=f"    Creating {mode} video") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Get intermediate visualization based on mode
                    if 'horizontal' in mode:
                        intermediates = get_horizontal_intermediates(frame)
                        mode_key = mode.replace('_horizontal', '')
                        vis_frame = intermediates.get(mode_key, frame)
                    elif 'vertical' in mode:
                        intermediates = get_vertical_intermediates(frame, self.boundaries.get('bottom'))
                        mode_key = mode.replace('_vertical', '')
                        vis_frame = intermediates.get(mode_key, frame)
                    else:
                        vis_frame = frame
                    
                    out.write(vis_frame)
                    pbar.update(1)
            
            out.release()
            print(f"    ✓ Saved: {os.path.basename(output_path)}")
        
        cap.release()
        print(f"  ✅ Intermediate videos saved to: {intermediate_dir}\n")
    
    
    def __repr__(self):
        """String representation of LaneDetector."""
        status = []
        if self._bottom_detected:
            status.append("bottom")
        if self._sides_detected:
            status.append("left/right")
        if self._top_detected:
            status.append("top")
        
        status_str = ", ".join(status) if status else "none"
        return f"LaneDetector(video='{self.video_name}', detected=[{status_str}])"
