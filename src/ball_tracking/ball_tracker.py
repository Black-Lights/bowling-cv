"""
Ball Tracker Module - Complete Pipeline for Phase 2

This module provides the BallTracker class for detecting and tracking the bowling ball
throughout its trajectory from release to pin impact.

Version: 2.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: January 30, 2026
Last Updated: January 30, 2026
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

# Import detection functions
from .detection_functions import (
    detect_ball_by_motion,
    detect_ball_by_color,
    combine_masks,
    apply_morphological_operations,
    filter_ball_contours,
    create_lane_mask,
    apply_lane_mask
)

# Import analysis and visualization modules
from . import tracking_analysis
from . import visualization


class BallTracker:
    """
    Complete ball tracking pipeline for bowling videos.
    
    Detects and tracks the bowling ball using frame differencing, color segmentation,
    and Kalman filtering. Integrates with Phase 1 (Lane Detection) to use lane 
    boundaries as region of interest.
    
    Attributes:
        video_path (str): Path to input video file
        boundary_data (dict): Lane boundaries from Phase 1
        config: Configuration module with tracking parameters
        output_dir (str): Directory for output files
        video_name (str): Name of video (without extension)
        
        # Tracking data
        trajectory (list): List of tracked ball positions over time
        release_point (dict): Detected release point data
        impact_point (dict): Detected impact point data
        
        # Processing state
        _initialized (bool): Whether tracker is initialized
        _tracking_active (bool): Whether actively tracking ball
        
    Example:
        >>> from ball_tracking import BallTracker
        >>> import ball_tracking.config as config
        >>> 
        >>> # Load boundary data from Phase 1
        >>> boundary_data = json.load(open('boundary_data.json'))
        >>> 
        >>> tracker = BallTracker('video.mp4', boundary_data, config)
        >>> tracker.track_all()  # Run complete tracking
        >>> tracker.save_results()  # Save all outputs
        >>> 
        >>> # Access results
        >>> print(tracker.trajectory)
        >>> print(tracker.release_point)
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, video_path: str, boundary_data: Dict, config, output_dir: Optional[str] = None):
        """
        Initialize BallTracker with video, boundary data, and configuration.
        
        Args:
            video_path (str): Path to input bowling video
            boundary_data (dict): Lane boundary data from Phase 1
            config: Configuration module with tracking parameters
            output_dir (str, optional): Custom output directory path
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If boundary_data is invalid
        """
        # Validate video path
        if not os.path.isabs(video_path):
            # Check if path exists relative to current directory first
            if not os.path.exists(video_path):
                assets_dir = getattr(config, 'ASSETS_DIR', os.getcwd())
                video_path = os.path.join(assets_dir, video_path)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Validate boundary data
        if not boundary_data or 'master_left' not in boundary_data:
            raise ValueError("Invalid boundary data. Run Phase 1 (Lane Detection) first.")
        
        self.video_path = video_path
        self.boundary_data = boundary_data
        self.config = config
        
        # Setup output directory (use same as Phase 1)
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        if output_dir:
            self.output_dir = output_dir
        else:
            base_output_dir = getattr(config, 'OUTPUT_DIR', 'output')
            self.output_dir = os.path.join(base_output_dir, self.video_name, 'tracking')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tracking storage
        self.trajectory = []
        self.release_point = None
        self.impact_point = None
        
        # Initialize Kalman filter
        self.kalman = None
        self._init_kalman_filter()
        
        # Tracking state
        self._initialized = False
        self._tracking_active = False
        self._missing_frames = 0
        self._last_detection = None
        self._prev_frame = None
        
        # Lane mask (created from boundary data)
        self.lane_mask = None
        
        print(f"\n{'='*70}")
        print(f"BallTracker v{self.VERSION} initialized")
        print(f"Video: {self.video_name}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")
    
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter for ball tracking."""
        # State: [x, y, vx, vy] (position and velocity)
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Transition matrix (constant velocity model)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we measure position only)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * self.config.KALMAN_PROCESS_NOISE
        
        # Measurement noise covariance
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.config.KALMAN_MEASUREMENT_NOISE
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
    
    
    def track_all(self) -> Dict:
        """
        Run complete ball tracking pipeline.
        
        This is the main entry point for the complete tracking pipeline.
        Processes entire video and extracts full trajectory.
        
        Returns:
            dict: Tracking results with keys:
                - trajectory: list of trajectory points
                - release_point: release point data
                - impact_point: impact point data
                - statistics: tracking statistics
                
        Example:
            >>> tracker = BallTracker('video.mp4', boundary_data, config)
            >>> results = tracker.track_all()
        """
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: BALL TRACKING PIPELINE")
        print(f"# Video: {self.video_name}")
        print(f"{'#'*70}\n")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}\n")
        
        # Create lane mask from boundary data
        self.lane_mask = create_lane_mask(
            self.boundary_data, 
            (height, width),
            margin=self.config.ROI_MARGIN
        )
        
        # Setup video writer if saving output
        out = None
        if self.config.SAVE_TRACKED_VIDEO:
            output_path = os.path.join(self.output_dir, f"tracked_{self.video_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video frames
        print("Processing frames...")
        frame_number = 0
        
        with tqdm(total=total_frames, desc="  Tracking ball") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect ball
                detection = self._detect_ball(frame)
                
                # Update tracker
                track_data = self._update_tracker(detection, frame_number)
                
                # Draw visualization if saving output
                if out is not None:
                    vis_frame = self._draw_visualization(frame.copy(), track_data)
                    out.write(vis_frame)
                
                # Store previous frame for next iteration
                self._prev_frame = frame.copy()
                
                frame_number += 1
                pbar.update(1)
        
        cap.release()
        if out is not None:
            out.release()
            print(f"  ✅ Saved tracked video")
        
        # Analyze trajectory
        self._analyze_trajectory()
        
        # Print summary
        self._print_summary()
        
        return {
            'trajectory': self.trajectory,
            'release_point': self.release_point,
            'impact_point': self.impact_point,
            'statistics': self._get_statistics()
        }
    
    
    def _detect_ball(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect ball in current frame.
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            Ball detection data or None if not detected
        """
        # Motion-based detection (frame differencing)
        motion_mask = None
        if self.config.USE_FRAME_DIFFERENCING and self._prev_frame is not None:
            motion_mask = detect_ball_by_motion(
                frame, 
                self._prev_frame,
                threshold=self.config.FRAME_DIFF_THRESHOLD
            )
        
        # Color-based detection
        color_mask = None
        if self.config.USE_COLOR_DETECTION:
            color_mask = detect_ball_by_color(
                frame,
                self.config.BALL_COLOR_LOWER,
                self.config.BALL_COLOR_UPPER
            )
        
        # Combine masks (use OR - accept either motion OR color for maximum sensitivity)
        combined_mask = combine_masks(motion_mask, color_mask, use_both=False)
        
        if combined_mask is None:
            return None
        
        # Apply morphological operations
        combined_mask = apply_morphological_operations(
            combined_mask,
            kernel_size=self.config.MORPH_KERNEL_SIZE,
            open_iterations=self.config.MORPH_OPEN_ITERATIONS,
            close_iterations=self.config.MORPH_CLOSE_ITERATIONS
        )
        
        # Apply lane mask
        if self.config.USE_LANE_MASK and self.lane_mask is not None:
            combined_mask = apply_lane_mask(combined_mask, self.lane_mask)
        
        # Find ball candidates
        candidates = filter_ball_contours(
            combined_mask,
            min_radius=self.config.MIN_BALL_RADIUS,
            max_radius=self.config.MAX_BALL_RADIUS,
            min_area=self.config.MIN_BALL_AREA,
            max_area=self.config.MAX_BALL_AREA,
            min_circularity=self.config.MIN_CIRCULARITY,
            min_solidity=self.config.MIN_SOLIDITY
        )
        
        if not candidates:
            return None
        
        # Select best candidate
        best_candidate = self._select_best_candidate(candidates)
        
        return best_candidate
    
    
    def _select_best_candidate(self, candidates: List[Dict]) -> Optional[Dict]:
        """
        Select best ball candidate from multiple detections.
        
        Args:
            candidates: List of candidate detections
            
        Returns:
            Best candidate or None
        """
        if not candidates:
            return None
        
        # If tracking active, select candidate closest to predicted position
        if self._tracking_active and self._last_detection is not None:
            # Predict next position using Kalman filter
            predicted = self.kalman.predict()
            pred_x, pred_y = int(predicted[0, 0]), int(predicted[1, 0])
            
            # Find candidate closest to prediction
            best_candidate = None
            min_distance = float('inf')
            
            for candidate in candidates:
                cx, cy = candidate['center']
                distance = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
                
                if distance < min_distance and distance < self.config.MAX_DISPLACEMENT:
                    min_distance = distance
                    best_candidate = candidate
            
            return best_candidate
        
        # Otherwise, select highest confidence candidate
        return candidates[0]
    
    
    def _update_tracker(self, detection: Optional[Dict], frame_number: int) -> Optional[Dict]:
        """
        Update Kalman filter tracker with detection.
        
        Args:
            detection: Ball detection data or None
            frame_number: Current frame number
            
        Returns:
            Track data or None
        """
        if detection is not None:
            return self._update_with_detection(detection, frame_number)
        else:
            return self._update_without_detection(frame_number)
    
    
    def _update_with_detection(self, detection: Dict, frame_number: int) -> Dict:
        """Update tracker when ball is detected."""
        center = detection['center']
        confidence = detection.get('confidence', 1.0)
        
        if not self._tracking_active:
            # Start new track - ensure float32 type
            self.kalman.statePost = np.array([
                [center[0]],
                [center[1]],
                [0.0],
                [0.0]
            ], dtype=np.float32)
            self._tracking_active = True
        else:
            # Update Kalman filter
            measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
            self.kalman.correct(measurement)
        
        # Reset missing frames counter
        self._missing_frames = 0
        
        # Calculate velocity
        velocity = self._calculate_velocity(center)
        
        # Create track data
        track_data = {
            'frame': frame_number,
            'center': center,
            'radius': detection['radius'],
            'velocity': velocity,
            'confidence': detection['confidence'],
            'detected': True
        }
        
        self.trajectory.append(track_data)
        self._last_detection = detection
        
        return track_data
    
    
    def _update_without_detection(self, frame_number: int) -> Optional[Dict]:
        """Update tracker when ball is not detected."""
        if not self._tracking_active:
            return None
        
        self._missing_frames += 1
        
        # Stop tracking if too many frames missed
        if self._missing_frames > self.config.MAX_MISSING_FRAMES:
            self._tracking_active = False
            return None
        
        # Predict position using Kalman filter
        predicted = self.kalman.predict()
        pred_pos = (int(predicted[0, 0]), int(predicted[1, 0]))
        pred_vel = (float(predicted[2, 0]), float(predicted[3, 0]))
        
        track_data = {
            'frame': frame_number,
            'center': pred_pos,
            'radius': self._last_detection['radius'] if self._last_detection else 0,
            'velocity': pred_vel,
            'confidence': max(0.1, 1.0 - 0.15 * self._missing_frames),
            'detected': False
        }
        
        self.trajectory.append(track_data)
        
        return track_data
    
    
    def _calculate_velocity(self, current_pos: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculate velocity from trajectory history.
        
        Args:
            current_pos: Current ball position
            
        Returns:
            (vx, vy) velocity vector in pixels/frame
        """
        if len(self.trajectory) == 0:
            return (0.0, 0.0)
        
        prev_pos = self.trajectory[-1]['center']
        vx = float(current_pos[0] - prev_pos[0])
        vy = float(current_pos[1] - prev_pos[1])
        
        # Apply smoothing if enabled
        if self.config.VELOCITY_SMOOTHING and len(self.trajectory) >= self.config.VELOCITY_WINDOW:
            window = self.config.VELOCITY_WINDOW
            recent = self.trajectory[-window:]
            vels = [p['velocity'] for p in recent if 'velocity' in p]
            if vels:
                avg_vx = np.mean([v[0] for v in vels] + [vx])
                avg_vy = np.mean([v[1] for v in vels] + [vy])
                return (avg_vx, avg_vy)
        
        return (vx, vy)
    
    
    def _analyze_trajectory(self):
        """Analyze trajectory to find release and impact points."""
        if len(self.trajectory) < self.config.MIN_TRACK_LENGTH:
            return
        
        # Apply smoothing if configured
        if self.config.SMOOTH_TRAJECTORY:
            self.trajectory = tracking_analysis.smooth_trajectory(
                self.trajectory,
                window_size=self.config.SMOOTHING_WINDOW
            )
        
        # Detect release point
        if self.config.DETECT_RELEASE_POINT:
            self.release_point = tracking_analysis.detect_release_point(
                self.trajectory,
                velocity_threshold=self.config.RELEASE_VELOCITY_THRESHOLD,
                zone_y=self.config.RELEASE_ZONE_Y,
                boundary_data=self.boundary_data
            )
        
        # Detect impact point
        if self.config.DETECT_IMPACT_POINT:
            self.impact_point = tracking_analysis.detect_impact_point(
                self.trajectory,
                velocity_threshold=self.config.IMPACT_VELOCITY_THRESHOLD,
                zone_y=self.config.IMPACT_ZONE_Y,
                boundary_data=self.boundary_data
            )
    
    
    def _draw_visualization(self, frame: np.ndarray, track_data: Optional[Dict]) -> np.ndarray:
        """Draw tracking visualization on frame."""
        return visualization.draw_tracking(
            frame,
            track_data,
            self.trajectory,
            self.config,
            release_point=self.release_point,
            impact_point=self.impact_point
        )
    
    
    def _get_statistics(self) -> Dict:
        """Calculate tracking statistics."""
        if not self.trajectory:
            return {}
        
        detected_count = sum(1 for p in self.trajectory if p.get('detected', False))
        avg_confidence = np.mean([p.get('confidence', 0) for p in self.trajectory])
        
        velocities = [p.get('velocity', (0, 0)) for p in self.trajectory]
        speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in velocities]
        
        return {
            'total_points': len(self.trajectory),
            'detected_points': detected_count,
            'predicted_points': len(self.trajectory) - detected_count,
            'detection_rate': detected_count / len(self.trajectory) if self.trajectory else 0,
            'avg_confidence': float(avg_confidence),
            'max_speed': float(max(speeds)) if speeds else 0,
            'avg_speed': float(np.mean(speeds)) if speeds else 0,
            'has_release_point': self.release_point is not None,
            'has_impact_point': self.impact_point is not None
        }
    
    
    def _print_summary(self):
        """Print tracking summary."""
        stats = self._get_statistics()
        
        print(f"\n{'='*70}")
        print(f"📊 TRACKING SUMMARY")
        print(f"{'='*70}")
        print(f"  Total points: {stats.get('total_points', 0)}")
        print(f"  Detected: {stats.get('detected_points', 0)}/{stats.get('total_points', 0)} "
              f"({100*stats.get('detection_rate', 0):.1f}%)")
        print(f"  Avg confidence: {stats.get('avg_confidence', 0):.3f}")
        print(f"  Max speed: {stats.get('max_speed', 0):.2f} px/frame")
        print(f"  Avg speed: {stats.get('avg_speed', 0):.2f} px/frame")
        
        if self.release_point:
            print(f"\n  ✅ Release point detected:")
            print(f"     Frame: {self.release_point['frame']}")
            print(f"     Position: {self.release_point['position']}")
            print(f"     Speed: {self.release_point['speed']:.2f} px/frame")
        
        if self.impact_point:
            print(f"\n  ✅ Impact point detected:")
            print(f"     Frame: {self.impact_point['frame']}")
            print(f"     Position: {self.impact_point['position']}")
        
        print(f"\n{'='*70}\n")
    
    
    def save_results(self):
        """Save all tracking results to output directory."""
        print(f"{'='*70}")
        print(f"SAVING RESULTS")
        print(f"{'='*70}\n")
        
        # Save trajectory data
        if self.config.SAVE_TRAJECTORY_DATA:
            tracking_analysis.save_trajectory_json(
                self.trajectory,
                self.release_point,
                self.impact_point,
                self._get_statistics(),
                os.path.join(self.output_dir, 'trajectory_data.json')
            )
            print(f"  ✅ Saved trajectory data")
        
        # Save trajectory plot
        if self.config.SAVE_TRAJECTORY_PLOT and self.trajectory:
            visualization.plot_trajectory(
                self.trajectory,
                self.boundary_data,
                os.path.join(self.output_dir, 'trajectory_plot.png'),
                release_point=self.release_point,
                impact_point=self.impact_point
            )
            print(f"  ✅ Saved trajectory plot")
        
        # Save velocity plot
        if self.config.SAVE_VELOCITY_PLOT and self.trajectory:
            visualization.plot_velocity(
                self.trajectory,
                os.path.join(self.output_dir, 'velocity_plot.png')
            )
            print(f"  ✅ Saved velocity plot")
        
        # Save analysis report
        if self.config.SAVE_ANALYSIS_REPORT:
            tracking_analysis.save_analysis_report(
                self.trajectory,
                self.release_point,
                self.impact_point,
                self._get_statistics(),
                os.path.join(self.output_dir, 'tracking_report.txt')
            )
            print(f"  ✅ Saved analysis report")
        
        print(f"\n{'='*70}")
        print(f"✅ All results saved to: {self.output_dir}")
        print(f"{'='*70}\n")
    
    
    def __repr__(self):
        """String representation of BallTracker."""
        status = "Active" if self._tracking_active else "Inactive"
        points = len(self.trajectory)
        return f"BallTracker(video='{self.video_name}', status='{status}', points={points})"
