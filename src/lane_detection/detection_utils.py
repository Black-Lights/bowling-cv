"""
Core detection utilities for bowling lane boundary detection
"""

import numpy as np
import cv2


def calculate_line_length(x1, y1, x2, y2):
    """Calculate Euclidean length of a line."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_line_angle(x1, y1, x2, y2, angle_mode='from_horizontal'):
    """
    Calculate angle of a line.
    
    Parameters:
    -----------
    angle_mode : str
        'from_horizontal' - Standard arctan angle from horizontal axis
        'from_vertical' - Angle from vertical axis (better for nearly vertical lines)
    
    Returns:
    --------
    angle : float
        Angle in degrees
    """
    dx = x2 - x1
    dy = y2 - y1
    
    if angle_mode == 'from_horizontal':
        if dx == 0:
            return 90.0 if dy < 0 else -90.0
        slope = dy / dx
        angle = np.degrees(np.arctan(slope))
    
    elif angle_mode == 'from_vertical':
        # Calculate angle from vertical axis
        # Vertical line = 0°, horizontal line = 90°
        # Positive angle = leans right, Negative angle = leans left
        
        if dy == 0:
            # Horizontal line
            return 90.0
        
        if dx == 0:
            # Perfectly vertical line
            return 0.0
        
        # For a line segment, we need to know its orientation
        # We'll assume y1 > y2 (bottom to top) from normalize_line_direction
        # Then: dx > 0 means lean right, dx < 0 means lean left
        
        # Calculate angle using atan2
        # atan2(dx, dy) gives angle from vertical when dy is the main component
        angle = np.degrees(np.arctan2(dx, abs(dy)))
    
    else:
        raise ValueError(f"Unknown angle_mode: {angle_mode}")
    
    return angle


def normalize_line_direction(x1, y1, x2, y2):
    """
    Normalize line direction so it always goes from bottom to top.
    This ensures consistent slope calculations.
    
    Returns:
    --------
    (x1, y1, x2, y2) with y1 > y2 (point 1 is lower)
    """
    if y1 < y2:
        # Swap points
        return x2, y2, x1, y1
    return x1, y1, x2, y2


def line_crosses_midline(x1, y1, x2, y2, mid_x):
    """
    Check if a line crosses the vertical mid_line.
    Returns True if the line crosses from left to right or right to left.
    """
    left_of_mid_1 = x1 < mid_x
    left_of_mid_2 = x2 < mid_x
    return left_of_mid_1 != left_of_mid_2


def get_line_foul_intersection(x1, y1, x2, y2, foul_params):
    """
    Calculate where a vertical line intersects the foul line.
    Returns x-coordinate of intersection.
    """
    if y2 - y1 == 0:
        return None
    
    v_slope = (x2 - x1) / (y2 - y1)
    foul_y = foul_params['center_y']
    x_intersect = int(x1 + v_slope * (foul_y - y1))
    
    return x_intersect


def extrapolate_line_to_full_height(line_data, median_x, foul_y, height):
    """
    Extrapolate a line to full frame height.
    
    Parameters:
    -----------
    line_data : dict
        Dictionary with 'line', 'slope', etc.
    median_x : int
        X-position where line should pass through foul line
    foul_y : int
        Y-position of foul line
    height : int
        Frame height
    
    Returns:
    --------
    dict with 'x_top', 'y_top', 'x_bottom', 'y_bottom'
    """
    x1, y1, x2, y2 = line_data['line']
    
    # Normalize to go bottom to top
    x1, y1, x2, y2 = normalize_line_direction(x1, y1, x2, y2)
    
    # Calculate slope for extrapolation
    if y2 - y1 == 0:
        # Horizontal line (shouldn't happen for vertical boundaries)
        v_slope = 0
    else:
        v_slope = (x2 - x1) / (y2 - y1)
    
    # Extrapolate to full height, passing through median_x at foul_y
    y_top = 0
    y_bottom = height
    
    if v_slope != float('inf'):
        x_top = int(median_x + v_slope * (y_top - foul_y))
        x_bottom = int(median_x + v_slope * (y_bottom - foul_y))
    else:
        x_top = median_x
        x_bottom = median_x
    
    return {
        'x_top': x_top,
        'y_top': y_top,
        'x_bottom': x_bottom,
        'y_bottom': y_bottom
    }


def calculate_slope_from_angle(angle, angle_mode='from_horizontal'):
    """
    Calculate slope from angle.
    
    Parameters:
    -----------
    angle : float
        Angle in degrees
    angle_mode : str
        'from_horizontal' or 'from_vertical'
    
    Returns:
    --------
    slope : float
        dy/dx slope value
    """
    if angle_mode == 'from_horizontal':
        if abs(angle) >= 89.9:
            return float('inf')
        return np.tan(np.radians(angle))
    
    elif angle_mode == 'from_vertical':
        if abs(angle) <= 0.1:
            return float('inf')
        # Convert from vertical to horizontal
        horizontal_angle = 90 - abs(angle)
        slope = np.tan(np.radians(horizontal_angle))
        # Apply sign
        if angle < 0:
            slope = -slope
        return slope
    
    return float('inf')
