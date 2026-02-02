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
    """
    
    def __init__(self, config, frame_width, frame_height, foul_line_y):
        """
        Initialize ball tracker
        
        Args:
            config: Configuration module
            frame_width, frame_height: Frame dimensions
            foul_line_y: Y-coordinate of foul line
        """
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.foul_line_y = foul_line_y
        
        # Kalman filter
        self.kalman = BallKalmanFilter(
            config.KALMAN_PROCESS_NOISE,
            config.KALMAN_MEASUREMENT_NOISE
        )
        
        # State management
        self.tracking_active = False
        self.lost_frames = 0
        self.trajectory = []  # List of (x, y) positions
        
        # Confirmation tracking (Problem 2 solution)
        self.confirmation_counter = 0  # Consecutive successful tracking frames
        self.is_confirmed = False  # True if CONFIRMATION_THRESHOLD reached
        self.total_distance_traveled = 0.0  # Cumulative pixel distance
        self.last_confirmed_y = None  # Y position where confirmed tracking was lost
        self.last_position = None  # Previous (x, y) for distance calculation
        
        # For global search velocity calculation
        self.recent_detections = deque(maxlen=5)
    
    def process_frame(self, denoised_mask, frame_idx):
        """
        Process single frame: global search or local tracking
        
        Args:
            denoised_mask: Binary mask from Stage B
            frame_idx: Frame number
            
        Returns:
            dict: {
                'detection': {center, radius, ...} or None,
                'prediction': {x, y, vx, vy} or None,
                'mode': 'global' or 'local',
                'roi_box': (x1, y1, x2, y2) or None,
                'roi_size': buffer size or None
            }
        """
        result = {
            'detection': None,
            'prediction': None,
            'mode': None,
            'roi_box': None,
            'roi_size': None
        }
        
        if not self.tracking_active:
            # GLOBAL SEARCH MODE
            result['mode'] = 'global'
            
            # Determine search boundary (Problem 2 solution)
            max_y = self.last_confirmed_y if self.last_confirmed_y is not None else None
            
            detection = global_search(
                denoised_mask,
                self.config,
                self.foul_line_y,
                list(self.recent_detections),
                max_y_boundary=max_y
            )
            
            if detection:
                result['detection'] = detection
                cx, cy = detection['center']
                
                # Initialize Kalman filter
                if not self.kalman.initialized:
                    self.kalman.initialize(cx, cy)
                else:
                    self.kalman.update(cx, cy)
                
                # Activate tracking
                self.tracking_active = True
                self.lost_frames = 0
                self.last_position = (cx, cy)  # Initialize for distance calculation
                self.trajectory.append((cx, cy))
                self.recent_detections.append((cx, cy))
                
                if self.config.VERBOSE and frame_idx % 30 == 0:
                    print(f"  Frame {frame_idx}: Ball detected! Activating local tracking mode")
        
        else:
            # LOCAL TRACKING MODE
            result['mode'] = 'local'
            
            # Predict next position
            prediction = self.kalman.predict()
            result['prediction'] = prediction
            
            if prediction:
                # Calculate ROI size (perspective-aware)
                roi_buffer = calculate_roi_size(prediction['y'], self.config)
                result['roi_size'] = roi_buffer
                
                # Create ROI box
                roi_box = create_roi_box(
                    prediction['x'],
                    prediction['y'],
                    roi_buffer,
                    self.frame_width,
                    self.frame_height
                )
                result['roi_box'] = roi_box
                
                # Search in ROI
                detection = local_tracking(denoised_mask, roi_box, self.config)
                
                if detection:
                    result['detection'] = detection
                    cx, cy = detection['center']
                    
                    # Update Kalman filter
                    self.kalman.update(cx, cy)
                    self.trajectory.append((cx, cy))
                    self.recent_detections.append((cx, cy))
                    self.lost_frames = 0
                    
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
                else:
                    # Ball lost
                    self.lost_frames += 1
                    
                    if self.lost_frames >= self.config.MAX_LOST_FRAMES:
                        # Determine search strategy based on confirmation status
                        use_restricted_search = (
                            self.is_confirmed and 
                            self.total_distance_traveled >= self.config.SPATIAL_CONFIRMATION_DISTANCE and
                            self.last_position is not None
                        )
                        
                        if use_restricted_search:
                            # CONFIRMED BALL: Restrict search to prevent re-detecting behind last position
                            self.last_confirmed_y = self.last_position[1] - self.config.SEARCH_BUFFER
                            if self.config.VERBOSE:
                                print(f"  Frame {frame_idx}: Confirmed ball lost. Restricting search to y < {self.last_confirmed_y:.0f}")
                        else:
                            # UNCONFIRMED OBJECT: Full reset (might have been hand)
                            if self.config.VERBOSE:
                                reason = "unconfirmed tracking" if not self.is_confirmed else f"short distance ({self.total_distance_traveled:.1f}px)"
                                print(f"  Frame {frame_idx}: Ball lost ({reason}). Full lane search activated")
                        
                        # Revert to global search
                        self.tracking_active = False
                        self.kalman = BallKalmanFilter(
                            self.config.KALMAN_PROCESS_NOISE,
                            self.config.KALMAN_MEASUREMENT_NOISE
                        )
                        
                        # Reset confirmation tracking if not confirmed
                        if not use_restricted_search:
                            self.confirmation_counter = 0
                            self.is_confirmed = False
                            self.total_distance_traveled = 0.0
                            self.last_confirmed_y = None
        
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
