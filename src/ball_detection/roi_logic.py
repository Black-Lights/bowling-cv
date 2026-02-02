"""
Stage C: ROI Logic (Region of Interest Search Strategy)

This module implements a dual-mode search strategy for ball detection:
1. Global Search (tracking_active=False): Wide-area discovery mode
2. Local Tracking (tracking_active=True): Precise ROI-based tracking

Uses Kalman Filter for prediction and perspective-aware ROI scaling.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Date: February 1, 2026
"""

import cv2
import numpy as np
from collections import deque
import os
from pathlib import Path
import subprocess
import shutil
from tqdm import tqdm


class BallKalmanFilter:
    """
    Kalman Filter for ball position and velocity prediction
    
    State vector: [x, y, vx, vy]
    - x, y: Ball position in image coordinates
    - vx, vy: Ball velocity (pixels/frame)
    """
    
    def __init__(self, process_noise=1.0, measurement_noise=10.0):
        """
        Initialize Kalman Filter
        
        Args:
            process_noise: Process noise covariance (motion model uncertainty)
            measurement_noise: Measurement noise covariance (detection uncertainty)
        """
        # Create Kalman filter: 4 state variables, 2 measurements
        self.kf = cv2.KalmanFilter(4, 2)
        
        # State transition matrix (constant velocity model)
        # x_new = x + vx, y_new = y + vy
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix (we measure x, y directly)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],  # Measure x
            [0, 1, 0, 0]   # Measure y
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Error covariance matrix (initial uncertainty)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1000
        
        self.initialized = False
    
    def initialize(self, x, y):
        """Initialize filter with first detection"""
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True
    
    def predict(self):
        """Predict next state"""
        if not self.initialized:
            return None
        
        prediction = self.kf.predict()
        return {
            'x': float(prediction[0, 0]),
            'y': float(prediction[1, 0]),
            'vx': float(prediction[2, 0]),
            'vy': float(prediction[3, 0])
        }
    
    def update(self, x, y):
        """Update filter with new measurement"""
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kf.correct(measurement)
        
        state = self.kf.statePost
        return {
            'x': float(state[0, 0]),
            'y': float(state[1, 0]),
            'vx': float(state[2, 0]),
            'vy': float(state[3, 0])
        }
    
    def get_state(self):
        """Get current state"""
        if not self.initialized:
            return None
        
        state = self.kf.statePost
        return {
            'x': float(state[0, 0]),
            'y': float(state[1, 0]),
            'vx': float(state[2, 0]),
            'vy': float(state[3, 0])
        }


def calculate_roi_size(y_ball, config):
    """
    Calculate ROI buffer size based on ball's Y position (perspective-aware)
    
    Formula: B_t = max(B_min, k * y_ball)
    - Near foul line (large y): Large ROI
    - Near pins (small y): Small ROI (but never < B_min)
    
    Args:
        y_ball: Ball's vertical position
        config: Configuration module
        
    Returns:
        int: ROI buffer size (half-width/height of square ROI)
    """
    dynamic_buffer = int(config.K_SCALE * y_ball)
    return max(config.B_MIN, dynamic_buffer)


def create_roi_box(center_x, center_y, buffer, frame_width, frame_height):
    """
    Create ROI bounding box with bounds checking
    
    Args:
        center_x, center_y: Predicted center position
        buffer: ROI half-size
        frame_width, frame_height: Frame dimensions
        
    Returns:
        tuple: (x1, y1, x2, y2) ROI coordinates
    """
    x1 = max(0, int(center_x - buffer))
    y1 = max(0, int(center_y - buffer))
    x2 = min(frame_width, int(center_x + buffer))
    y2 = min(frame_height, int(center_y + buffer))
    
    return (x1, y1, x2, y2)


def find_ball_contours(mask, config):
    """
    Find ball candidates from binary mask using contour analysis
    
    Args:
        mask: Binary mask (ball=white, background=black)
        config: Configuration module
        
    Returns:
        list: List of candidate dicts with keys:
            - center: (x, y) tuple
            - radius: float
            - contour: numpy array
            - area: float
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for contour in contours:
        # Calculate minimum enclosing circle
        ((cx, cy), radius) = cv2.minEnclosingCircle(contour)
        area = cv2.contourArea(contour)
        
        # Filter by radius constraints
        if config.MIN_BALL_RADIUS <= radius <= config.MAX_BALL_RADIUS:
            candidates.append({
                'center': (float(cx), float(cy)),
                'radius': float(radius),
                'contour': contour,
                'area': float(area)
            })
    
    return candidates


def global_search(mask, config, foul_line_y, prev_positions=None, max_y_boundary=None):
    """
    Global Search Mode: Find ball in entire masked lane (or restricted region)
    
    Prioritizes:
    1. Objects near foul line (bottom boundary)
    2. Objects with negative Y velocity (moving toward pins)
    
    Args:
        mask: Denoised binary mask
        config: Configuration module
        foul_line_y: Y-coordinate of foul line
        prev_positions: Previous ball positions (for velocity calculation)
        max_y_boundary: Maximum Y value for search (None = full lane, int = restrict to y < max_y_boundary)
                       Used to prevent re-detecting ball behind where it was lost
        
    Returns:
        dict or None: {
            'center': (x, y),
            'radius': float,
            'velocity': (vx, vy),
            'confidence': float
        }
    """
    candidates = find_ball_contours(mask, config)
    
    if not candidates:
        return None
    
    # Filter candidates by Y-position boundary if specified (Problem 2 solution)
    if max_y_boundary is not None:
        candidates = [c for c in candidates if c['center'][1] < max_y_boundary]
        
        if not candidates:
            return None
    
    # Filter out candidates in upper part of frame (bowler area)
    # Only consider detections in lower 70% of frame (where ball should be)
    # This prevents tracking the bowler's head/body
    min_y_threshold = foul_line_y * 0.3  # Upper 30% is bowler area
    candidates = [c for c in candidates if c['center'][1] > min_y_threshold]
    
    if not candidates:
        return None
    
    # Score candidates
    scored = []
    priority_zone_start = foul_line_y - config.FOUL_LINE_PRIORITY_ZONE
    
    for candidate in candidates:
        cx, cy = candidate['center']
        score = 0.0
        
        # Priority 1: Near foul line (higher score = closer to foul line)
        if cy > priority_zone_start:
            proximity_score = (cy - priority_zone_start) / config.FOUL_LINE_PRIORITY_ZONE
            score += proximity_score * 100  # Weight: 0-100 points (increased from 50)
        else:
            # Penalize detections far from foul line
            distance_penalty = (priority_zone_start - cy) / priority_zone_start
            score -= distance_penalty * 50  # Penalty for being too far up
        
        # Priority 2: Negative Y velocity (if we have history)
        if prev_positions and len(prev_positions) >= 2:
            # Calculate velocity from last 2 positions
            p1 = prev_positions[-1]
            p2 = prev_positions[-2]
            vy = p1[1] - p2[1]  # Negative = moving up (toward pins)
            
            if vy < config.MIN_VELOCITY_Y:
                score += abs(vy) * 10  # More negative = higher score
        
        # Priority 3: Larger objects (bowling ball is prominent)
        size_score = candidate['radius'] / config.MAX_BALL_RADIUS
        score += size_score * 20  # Weight: 0-20 points
        
        scored.append((score, candidate))
    
    # Select best candidate
    if scored:
        best_score, best_candidate = max(scored, key=lambda x: x[0])
        
        # Calculate velocity if possible
        vx, vy = 0.0, 0.0
        if prev_positions:
            vx = best_candidate['center'][0] - prev_positions[-1][0]
            vy = best_candidate['center'][1] - prev_positions[-1][1]
        
        return {
            'center': best_candidate['center'],
            'radius': best_candidate['radius'],
            'velocity': (vx, vy),
            'confidence': best_score / 70.0  # Normalize to 0-1
        }
    
    return None


def local_tracking(mask, roi_box, config):
    """
    Local Tracking Mode: Find ball in small ROI around predicted position
    
    Args:
        mask: Denoised binary mask (full frame)
        roi_box: (x1, y1, x2, y2) ROI coordinates
        config: Configuration module
        
    Returns:
        dict or None: {
            'center': (x, y) in full frame coordinates,
            'radius': float,
            'roi_box': (x1, y1, x2, y2)
        }
    """
    x1, y1, x2, y2 = roi_box
    
    # Crop mask to ROI
    roi_mask = mask[y1:y2, x1:x2]
    
    if roi_mask.size == 0:
        return None
    
    # Find contours in ROI
    candidates = find_ball_contours(roi_mask, config)
    
    if not candidates:
        return None
    
    # Select largest candidate (most likely the ball)
    best_candidate = max(candidates, key=lambda c: c['area'])
    
    # Convert ROI coordinates to full frame coordinates
    cx_roi, cy_roi = best_candidate['center']
    cx_full = cx_roi + x1
    cy_full = cy_roi + y1
    
    return {
        'center': (cx_full, cy_full),
        'radius': best_candidate['radius'],
        'roi_box': roi_box
    }


class BallTracker:
    """
    Complete ball tracking system with dual-mode search strategy
    REFACTORED: Implements Tracking-by-Detection architecture
    Filter ALL candidates (Stage D) -> Select based on state -> Update Kalman
    """
    
    def __init__(self, config, frame_width, frame_height, foul_line_y, blob_analyzer=None):
        """
        Initialize ball tracker
        
        Args:
            config: Configuration module
            frame_width, frame_height: Frame dimensions
            foul_line_y: Y-coordinate of foul line
            blob_analyzer: BlobAnalyzer instance for Stage D filtering (optional)
        """
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.foul_line_y = foul_line_y
        self.blob_analyzer = blob_analyzer
        
        # Kalman filter
        self.kalman = BallKalmanFilter(
            config.KALMAN_PROCESS_NOISE,
            config.KALMAN_MEASUREMENT_NOISE
        )
        
        # State management
        self.tracking_active = False
        self.lost_frames = 0
        self.trajectory = []  # List of (x, y) positions
        
        # NEW: Global search type tracking
        self.search_type = 'initial'  # 'initial' or 'reactivation'
        self.last_known_y = None  # Y position when ball last seen
        self.reactivation_lost_frames = 0  # Frames without detection in reactivation mode
        
        # Confirmation tracking (Problem 2 solution)
        self.confirmation_counter = 0  # Consecutive successful tracking frames
        self.is_confirmed = False  # True if CONFIRMATION_THRESHOLD reached
        self.total_distance_traveled = 0.0  # Cumulative pixel distance
        self.last_position = None  # Previous (x, y) for distance calculation
        
        # Stage F: Stop condition (pin impact)
        self.trajectory_complete = False  # True when ball reaches pin area
        self.stop_frame = None  # Frame number where tracking stopped
        self.interpolated_points = []  # List of extrapolated (x, y) points
        self.stop_threshold_y = None  # Calculated in set_boundaries()
        self.top_boundary_y = None  # Loaded from Phase 1 data
        
        # For global search velocity calculation
        self.recent_detections = deque(maxlen=5)
    
    def set_boundaries(self, top_boundary_y):
        """
        Set top boundary from Phase 1 and calculate stop threshold
        
        Args:
            top_boundary_y: Y-coordinate of top boundary (green line) from Phase 1
        """
        self.top_boundary_y = top_boundary_y
        
        if self.config.ENABLE_STOP_CONDITION:
            # Calculate stop threshold: top_boundary + 3% of frame height
            threshold_offset = int(self.config.STOP_THRESHOLD_PCT * self.frame_height)
            self.stop_threshold_y = top_boundary_y + threshold_offset
            
            if self.config.VERBOSE:
                print(f"\n{'='*60}")
                print(f"STAGE F: STOP CONDITION INITIALIZED")
                print(f"{'='*60}")
                print(f"  Top Boundary (pins): Y = {top_boundary_y}")
                print(f"  Frame Height: {self.frame_height}px")
                print(f"  Threshold Offset: {threshold_offset}px ({self.config.STOP_THRESHOLD_PCT*100:.1f}% of height)")
                print(f"  Stop Threshold: Y <= {self.stop_threshold_y}")
                print(f"  -> Tracking will stop when ball reaches Y <= {self.stop_threshold_y}")
                print(f"{'='*60}\n")
    
    def _interpolate_trajectory(self, last_x, last_y, frame_idx):
        """
        Collect N Kalman filter predictions after trajectory completion
        Simulates future ball positions using Kalman state propagation
        
        Args:
            last_x: X coordinate of last detection
            last_y: Y coordinate of last detection
            frame_idx: Frame number of last detection
        """
        if not self.kalman.initialized:
            return
        
        # Get current velocity from Kalman filter
        state = self.kalman.kf.statePost
        vx = state[2, 0]
        vy = state[3, 0]
        
        # Check if ball is moving toward pins (negative Y velocity)
        if vy >= 0:
            if self.config.VERBOSE:
                print(f"  Warning: Ball not moving toward pins (vy={vy:.2f}). Skipping Kalman predictions.")
            return
        
        # Get number of predictions from config
        num_predictions = getattr(self.config, 'NUM_KALMAN_PREDICTIONS_AFTER_STOP', 5)
        
        if self.config.VERBOSE:
            print(f"  Collecting {num_predictions} Kalman predictions after stop:")
            print(f"    Last detection: ({last_x:.0f}, {last_y:.0f})")
            print(f"    Initial velocity: vx={vx:.2f}, vy={vy:.2f} px/frame")
        
        # Collect N predictions by running Kalman filter forward without measurements
        predictions = []
        for i in range(num_predictions):
            # Predict next frame
            prediction = self.kalman.predict()
            
            # Store prediction
            pred_x = int(prediction['x'])
            pred_y = int(prediction['y'])
            predictions.append((pred_x, pred_y))
            
            if self.config.VERBOSE:
                print(f"    Prediction {i+1}: ({pred_x}, {pred_y})")
        
        # Store all predictions
        self.interpolated_points = predictions
        
        if self.config.VERBOSE:
            print(f"  Stored {len(predictions)} Kalman predictions")
    
    def _filter_all_candidates(self, denoised_mask, frame=None):
        """
        Stage D: Filter ALL contours in full frame
        Independent of tracking state - always runs on complete frame
        
        Args:
            denoised_mask: Binary mask from Stage B
            frame: Original BGR frame (optional, for color filter)
            
        Returns:
            List of validated candidates: [{'center': (x, y), 'radius': r, 'area': a, ...}, ...]
        """
        # Find ALL contours in denoised mask
        contours, _ = cv2.findContours(denoised_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return []
        
        validated_candidates = []
        
        # If blob_analyzer is available, use it for filtering
        if self.blob_analyzer and self.blob_analyzer.is_calibrated:
            for contour in contours:
                # Get blob metrics and filter results
                metrics = self.blob_analyzer.analyze_blob(frame, contour, self.frame_height)
                
                # Only keep candidates passing ALL filters
                if metrics.passes_all_filters:
                    # Calculate bounding circle for tracking
                    (cx, cy), radius = cv2.minEnclosingCircle(contour)
                    
                    validated_candidates.append({
                        'center': (int(cx), int(cy)),
                        'radius': int(radius),
                        'area': metrics.area,
                        'circularity': metrics.circularity,
                        'aspect_ratio': metrics.aspect_ratio,
                        'contour': contour
                    })
        else:
            # Fallback: Simple area + circularity filtering
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip tiny contours
                if area < self.config.MIN_BALL_RADIUS ** 2:
                    continue
                
                # Get centroid
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                
                # Simple circularity check
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                if circularity >= 0.5:  # Loose threshold for fallback
                    (cx, cy), radius = cv2.minEnclosingCircle(contour)
                    validated_candidates.append({
                        'center': (int(cx), int(cy)),
                        'radius': int(radius),
                        'area': area,
                        'circularity': circularity,
                        'contour': contour
                    })
        
        return validated_candidates
    
    def _global_search_selection(self, validated_candidates):
        """
        Select best candidate from validated list using global search strategy
        Two modes: Initial (near foul line) vs Reactivation (above last position)
        
        Args:
            validated_candidates: List of candidates passing Stage D filters
            
        Returns:
            Selected candidate dict or None
        """
        if len(validated_candidates) == 0:
            return None
        
        if self.search_type == 'initial':
            # TYPE 1: Initial detection - focus near foul line, exclude upper frame
            min_y_threshold = self.foul_line_y * self.config.FOUL_LINE_EXCLUSION_FACTOR
            
            # Filter to candidates in valid zone (exclude upper 30%)
            valid_candidates = [
                c for c in validated_candidates 
                if c['center'][1] > min_y_threshold
            ]
            
            if len(valid_candidates) == 0:
                return None
            
            # Select candidate closest to foul line (highest Y value)
            selected = max(valid_candidates, key=lambda c: c['center'][1])
            
            return selected
            
        else:  # self.search_type == 'reactivation'
            # TYPE 2: Reactivation - search only above last known position (toward pins)
            if self.last_known_y is None:
                # Fallback to initial mode if no last position
                self.search_type = 'initial'
                return self._global_search_selection(validated_candidates)
            
            # Search above (lower Y) last position with safety margin
            # In image coordinates: smaller Y = higher up toward pins
            max_y_search = self.last_known_y + self.config.REACTIVATION_SEARCH_MARGIN
            
            # Filter to candidates in valid zone (above last position + margin)
            # Keep only candidates with Y < max_y_search (closer to pins/frame top)
            valid_candidates = [
                c for c in validated_candidates 
                if c['center'][1] < max_y_search
            ]
            
            if len(valid_candidates) == 0:
                return None
            
            # Select candidate closest to last known position
            def distance_to_last(candidate):
                cx, cy = candidate['center']
                if self.last_position:
                    lx, ly = self.last_position
                    return np.sqrt((cx - lx)**2 + (cy - ly)**2)
                else:
                    return abs(cy - self.last_known_y)
            
            selected = min(valid_candidates, key=distance_to_last)
            
            return selected
    
    def _local_tracking_selection(self, validated_candidates, prediction):
        """
        Select best candidate from validated list using local tracking strategy
        Filters to ROI around Kalman prediction, selects closest
        
        Args:
            validated_candidates: List of candidates passing Stage D filters
            prediction: Kalman prediction dict {'x', 'y', 'vx', 'vy'}
            
        Returns:
            (selected_candidate, roi_box, roi_size) tuple or (None, None, None)
        """
        if len(validated_candidates) == 0 or prediction is None:
            return None, None, None
        
        # Calculate ROI size (perspective-aware)
        roi_buffer = calculate_roi_size(prediction['y'], self.config)
        
        # Create ROI box
        roi_box = create_roi_box(
            prediction['x'],
            prediction['y'],
            roi_buffer,
            self.frame_width,
            self.frame_height
        )
        
        x1, y1, x2, y2 = roi_box
        
        # Filter validated candidates to those within ROI
        candidates_in_roi = [
            c for c in validated_candidates
            if (x1 <= c['center'][0] <= x2 and y1 <= c['center'][1] <= y2)
        ]
        
        if len(candidates_in_roi) == 0:
            return None, roi_box, roi_buffer
        
        # Select candidate closest to prediction
        def distance_to_prediction(candidate):
            cx, cy = candidate['center']
            return np.sqrt((cx - prediction['x'])**2 + (cy - prediction['y'])**2)
        
        selected = min(candidates_in_roi, key=distance_to_prediction)
        
        return selected, roi_box, roi_buffer
    
    def _select_candidate(self, validated_candidates, prediction=None):
        """
        Dispatcher: Select candidate based on tracking state
        
        Args:
            validated_candidates: List of candidates passing Stage D filters
            prediction: Kalman prediction (if tracking active)
            
        Returns:
            (selected_candidate, roi_box, roi_size, mode) tuple
        """
        if not self.tracking_active:
            # GLOBAL SEARCH MODE
            selected = self._global_search_selection(validated_candidates)
            return selected, None, None, 'global'
        else:
            # LOCAL TRACKING MODE
            selected, roi_box, roi_size = self._local_tracking_selection(
                validated_candidates, prediction
            )
            return selected, roi_box, roi_size, 'local'
    
    def _update_state(self, selected_candidate, frame_idx):
        """
        Update Kalman filter and state management
        CRITICAL: Only called with VALIDATED candidates (passed Stage D)
        
        Args:
            selected_candidate: Candidate dict or None
            frame_idx: Current frame number
        """
        if selected_candidate:
            cx, cy = selected_candidate['center']
            
            # Initialize or update Kalman filter
            if not self.kalman.initialized:
                self.kalman.initialize(cx, cy)
            else:
                self.kalman.update(cx, cy)
            
            # Activate tracking
            if not self.tracking_active:
                self.tracking_active = True
                if self.config.VERBOSE and frame_idx % 30 == 0:
                    print(f"  Frame {frame_idx}: Ball detected! Activating local tracking mode")
            
            # Update trajectory and state
            self.trajectory.append((cx, cy))
            self.recent_detections.append((cx, cy))
            self.lost_frames = 0
            self.last_known_y = cy  # Update last known position
            self.reactivation_lost_frames = 0  # Reset reactivation timeout counter
            
            # Update confirmation tracking
            self.confirmation_counter += 1
            
            # Calculate distance traveled (for spatial confirmation)
            if self.last_position is not None:
                prev_x, prev_y = self.last_position
                distance = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                self.total_distance_traveled += distance
            
            self.last_position = (cx, cy)
            
            # Check if confirmed
            if self.confirmation_counter >= self.config.CONFIRMATION_THRESHOLD:
                if not self.is_confirmed:
                    self.is_confirmed = True
                    if self.config.VERBOSE:
                        print(f"  Frame {frame_idx}: Ball CONFIRMED (tracked {self.confirmation_counter} frames, traveled {self.total_distance_traveled:.1f}px)")
            
            # STAGE F: Check stop condition (pin impact)
            if (self.config.ENABLE_STOP_CONDITION and 
                self.is_confirmed and 
                not self.trajectory_complete and
                self.stop_threshold_y is not None):
                
                # Check if ball reached pin area (Y <= stop_threshold)
                if cy <= self.stop_threshold_y:
                    self.trajectory_complete = True
                    self.stop_frame = frame_idx
                    
                    if self.config.VERBOSE:
                        print(f"\n{'='*60}")
                        print(f"  Frame {frame_idx}: TRAJECTORY COMPLETE!")
                        print(f"  Ball reached pin area: Y={cy} <= {self.stop_threshold_y}")
                        print(f"  Total trajectory points: {len(self.trajectory)}")
                        print(f"{'='*60}\n")
                    
                    # Optional: Interpolate trajectory to/beyond top boundary
                    if self.config.INTERPOLATE_TO_BOUNDARY:
                        self._interpolate_trajectory(cx, cy, frame_idx)
        
        else:
            # No detection
            self.lost_frames += 1
            
            # Increment reactivation timeout counter if in reactivation mode
            if self.search_type == 'reactivation' and not self.tracking_active:
                self.reactivation_lost_frames += 1
                
                # Check if reactivation search has timed out
                if self.reactivation_lost_frames >= self.config.REACTIVATION_TIMEOUT:
                    # Reset to initial search (full frame)
                    self.search_type = 'initial'
                    self.last_known_y = None
                    self.reactivation_lost_frames = 0
                    if self.config.VERBOSE:
                        print(f"  Frame {frame_idx}: Reactivation timeout ({self.config.REACTIVATION_TIMEOUT} frames). Resetting to full frame search")
            
            if self.tracking_active and self.lost_frames >= self.config.MAX_LOST_FRAMES:
                # Determine search strategy based on confirmation status
                use_restricted_search = (
                    self.is_confirmed and 
                    self.total_distance_traveled >= self.config.SPATIAL_CONFIRMATION_DISTANCE and
                    self.last_position is not None
                )
                
                if use_restricted_search:
                    # CONFIRMED BALL: Switch to reactivation mode
                    self.search_type = 'reactivation'
                    if self.config.VERBOSE:
                        print(f"  Frame {frame_idx}: Confirmed ball lost. Reactivation search (y < {self.last_known_y + self.config.REACTIVATION_SEARCH_MARGIN:.0f})")
                else:
                    # UNCONFIRMED OBJECT: Reset to initial mode
                    self.search_type = 'initial'
                    if self.config.VERBOSE:
                        reason = "unconfirmed tracking" if not self.is_confirmed else f"short distance ({self.total_distance_traveled:.1f}px)"
                        print(f"  Frame {frame_idx}: Ball lost ({reason}). Initial global search activated")
                
                # Revert to global search
                self.tracking_active = False
                self.kalman = BallKalmanFilter(
                    self.config.KALMAN_PROCESS_NOISE,
                    self.config.KALMAN_MEASUREMENT_NOISE
                )
                
                # Reset confirmation tracking if not confirmed
                # NOTE: Do NOT reset last_known_y here - keep it for potential reactivation
                if not use_restricted_search:
                    self.confirmation_counter = 0
                    self.is_confirmed = False
                    self.total_distance_traveled = 0.0
    
    def process_frame(self, denoised_mask, frame_idx, frame=None):
        """
        Process single frame with new architecture
        Filter ALL → Select based on state → Update Kalman
        
        Args:
            denoised_mask: Binary mask from Stage B
            frame_idx: Frame number
            frame: Original BGR frame (optional, for Stage D color filter)
            
        Returns:
            dict: {
                'detection': {center, radius, ...} or None,
                'prediction': {x, y, vx, vy} or None,
                'mode': 'global' or 'local',
                'search_type': 'initial' or 'reactivation' (if global),
                'roi_box': (x1, y1, x2, y2) or None,
                'roi_size': buffer size or None,
                'all_candidates': List of all validated candidates,
                'candidates_count': Number of validated candidates
            }
        """
        # STEP 1: Filter ALL candidates (Stage D - full frame, independent)
        validated_candidates = self._filter_all_candidates(denoised_mask, frame)
        
        # STEP 2: Predict next position (if tracking)
        prediction = None
        if self.tracking_active:
            prediction = self.kalman.predict()
        
        # STEP 3: Select best candidate based on tracking state
        selected, roi_box, roi_size, mode = self._select_candidate(
            validated_candidates, prediction
        )
        
        # STEP 4: Update Kalman filter and state (ONLY with validated candidate)
        self._update_state(selected, frame_idx)
        
        # Return complete result
        result = {
            'detection': selected,
            'prediction': prediction,
            'mode': mode,
            'search_type': self.search_type if mode == 'global' else None,
            'roi_box': roi_box,
            'roi_size': roi_size,
            'all_candidates': validated_candidates,
            'candidates_count': len(validated_candidates),
            'last_known_y': self.last_known_y,  # For reactivation zone visualization
            # Stage F: Stop condition fields
            'trajectory_complete': self.trajectory_complete,
            'stop_frame': self.stop_frame,
            'interpolated_points': self.interpolated_points.copy() if self.interpolated_points else [],
            'stop_threshold_y': self.stop_threshold_y
        }
        
        return result


if __name__ == '__main__':
    # Test Kalman filter
    kf = BallKalmanFilter()
    kf.initialize(100, 200)
    
    # Simulate ball movement
    for i in range(10):
        pred = kf.predict()
        print(f"Prediction: {pred}")
        
        # Simulate noisy measurement
        measured_x = 100 + i * 5 + np.random.randn() * 2
        measured_y = 200 - i * 10 + np.random.randn() * 2
        
        updated = kf.update(measured_x, measured_y)
        print(f"Updated: {updated}\n")
