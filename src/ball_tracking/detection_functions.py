"""
Ball detection functions for bowling video analysis

Functions for detecting bowling ball using frame differencing and color segmentation.

Version: 2.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: January 30, 2026
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict


def detect_ball_by_motion(current_frame: np.ndarray, 
                           previous_frame: np.ndarray,
                           threshold: int = 30) -> np.ndarray:
    """
    Detect ball using frame differencing (motion detection).
    
    Simple and effective for static camera with minimal background motion.
    
    Args:
        current_frame: Current frame (BGR)
        previous_frame: Previous frame (BGR)
        threshold: Difference threshold for motion detection
        
    Returns:
        Binary mask with motion regions
    """
    # Convert to grayscale
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray_current = cv2.GaussianBlur(gray_current, (5, 5), 0)
    gray_previous = cv2.GaussianBlur(gray_previous, (5, 5), 0)
    
    # Compute absolute difference
    frame_diff = cv2.absdiff(gray_current, gray_previous)
    
    # Threshold to get binary mask
    _, motion_mask = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    
    return motion_mask


def detect_ball_by_color(frame: np.ndarray,
                          lower_hsv: Tuple[int, int, int],
                          upper_hsv: Tuple[int, int, int]) -> np.ndarray:
    """
    Detect ball using color segmentation in HSV space.
    
    Args:
        frame: Input frame (BGR)
        lower_hsv: Lower HSV threshold (H, S, V)
        upper_hsv: Upper HSV threshold (H, S, V)
        
    Returns:
        Binary mask with regions matching color
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create color mask
    lower = np.array(lower_hsv)
    upper = np.array(upper_hsv)
    color_mask = cv2.inRange(hsv, lower, upper)
    
    return color_mask


def combine_masks(motion_mask: np.ndarray, 
                  color_mask: np.ndarray,
                  use_both: bool = True) -> np.ndarray:
    """
    Combine motion and color masks.
    
    Args:
        motion_mask: Binary mask from motion detection
        color_mask: Binary mask from color detection
        use_both: If True, use AND operation; if False, use OR
        
    Returns:
        Combined binary mask
    """
    if motion_mask is None and color_mask is None:
        return None
    
    if motion_mask is None:
        return color_mask
    
    if color_mask is None:
        return motion_mask
    
    # AND operation for stricter detection (motion AND color)
    if use_both:
        combined = cv2.bitwise_and(motion_mask, color_mask)
    else:
        # OR operation for more lenient detection (motion OR color)
        combined = cv2.bitwise_or(motion_mask, color_mask)
    
    return combined


def apply_morphological_operations(mask: np.ndarray,
                                    kernel_size: int = 5,
                                    open_iterations: int = 2,
                                    close_iterations: int = 2) -> np.ndarray:
    """
    Apply morphological operations to clean up mask.
    
    Opening: Remove small noise
    Closing: Fill small holes
    
    Args:
        mask: Binary mask
        kernel_size: Size of morphological kernel
        open_iterations: Number of opening iterations
        close_iterations: Number of closing iterations
        
    Returns:
        Cleaned binary mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Opening: erosion followed by dilation (removes noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    
    # Closing: dilation followed by erosion (fills holes)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    
    return mask


def filter_ball_contours(mask: np.ndarray,
                         min_radius: int = 8,
                         max_radius: int = 50,
                         min_area: int = 200,
                         max_area: int = 8000,
                         min_circularity: float = 0.65,
                         min_solidity: float = 0.7) -> List[Dict]:
    """
    Find and filter ball candidates from binary mask.
    
    Filters contours based on size, shape (circularity), and solidity.
    
    Args:
        mask: Binary mask
        min_radius: Minimum ball radius (pixels)
        max_radius: Maximum ball radius (pixels)
        min_area: Minimum contour area (pixels²)
        max_area: Maximum contour area (pixels²)
        min_circularity: Minimum circularity (0-1, 1=perfect circle)
        min_solidity: Minimum solidity (area/convex_hull_area)
        
    Returns:
        List of ball candidate dictionaries with keys:
            - center: (x, y) tuple
            - radius: int
            - area: float
            - circularity: float
            - solidity: float
            - confidence: float (0-1)
            - bbox: (x, y, w, h) tuple
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Get minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Filter by radius
        if radius < min_radius or radius > max_radius:
            continue
        
        # Calculate circularity: 4π*area / perimeter²
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Filter by circularity
        if circularity < min_circularity:
            continue
        
        # Calculate solidity: area / convex_hull_area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Filter by solidity
        if solidity < min_solidity:
            continue
        
        # Get bounding box
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
        
        # Calculate confidence (weighted average of shape metrics)
        confidence = (circularity * 0.7 + solidity * 0.3)
        
        candidates.append({
            'center': (int(x), int(y)),
            'radius': int(radius),
            'area': area,
            'circularity': circularity,
            'solidity': solidity,
            'confidence': confidence,
            'bbox': (bbox_x, bbox_y, bbox_w, bbox_h),
            'contour': contour
        })
    
    # Sort by confidence (highest first)
    candidates.sort(key=lambda c: c['confidence'], reverse=True)
    
    return candidates


def create_lane_mask(boundary_data: Dict, frame_shape: Tuple[int, int], margin: int = 10) -> np.ndarray:
    """
    Create binary mask of lane area from Phase 1 boundary data.
    
    Args:
        boundary_data: Boundary data from Phase 1 (boundary_data.json)
        frame_shape: (height, width) of frame
        margin: Margin to extend lane boundaries (pixels)
        
    Returns:
        Binary mask where lane area is white (255)
    """
    height, width = frame_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Get boundary parameters
    left = boundary_data['master_left']
    right = boundary_data['master_right']
    top_y = int(boundary_data['top_boundary']['y_position'])
    bottom_y = boundary_data['median_foul_params']['center_y']
    
    # Create polygon points for lane area (with margin)
    pts = np.array([
        [max(0, left['x_top'] - margin), max(0, top_y - margin)],
        [min(width-1, right['x_top'] + margin), max(0, top_y - margin)],
        [min(width-1, right['x_bottom'] + margin), min(height-1, bottom_y + margin)],
        [max(0, left['x_bottom'] - margin), min(height-1, bottom_y + margin)]
    ], dtype=np.int32)
    
    # Fill polygon
    cv2.fillPoly(mask, [pts], 255)
    
    return mask


def apply_lane_mask(mask: np.ndarray, lane_mask: np.ndarray) -> np.ndarray:
    """
    Apply lane mask to detection mask to filter out detections outside lane.
    
    Args:
        mask: Detection mask
        lane_mask: Lane region mask
        
    Returns:
        Masked result
    """
    return cv2.bitwise_and(mask, lane_mask)
