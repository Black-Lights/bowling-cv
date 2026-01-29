"""
Line detection functions for bowling lane analysis
"""

import numpy as np
import cv2
from detection_utils import *


def detect_horizontal_line(img):
    """
    Detect horizontal foul line from bottom half of image.
    
    Returns:
    --------
    tuple: (horizontal_lines, edges, foul_line_params)
    """
    height, width = img.shape[:2]
    bottom_half = img[height//2:height, 0:width]
    
    blurred = cv2.GaussianBlur(bottom_half, (5, 5), 0)
    grayscale = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    otsu_value, otsu_thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower_thresh = int(0.5 * otsu_value)
    upper_thresh = int(1.5 * otsu_value)
    
    edges = cv2.Canny(grayscale, lower_thresh, upper_thresh)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:
                    horizontal_lines.append(line[0])
    
    foul_line_params = None
    if horizontal_lines:
        fx1, fy1, fx2, fy2 = horizontal_lines[0]
        fy1_full = fy1 + height // 2
        fy2_full = fy2 + height // 2
        
        foul_slope = (fy2 - fy1) / (fx2 - fx1) if fx2 != fx1 else 0
        foul_y_left = int(fy1_full + foul_slope * (0 - fx1))
        foul_y_right = int(fy1_full + foul_slope * (width - fx1))
        
        foul_center_x = width // 2
        foul_center_y = int((foul_y_left + foul_y_right) / 2)
        
        foul_line_params = {
            'slope': foul_slope,
            'y_left': foul_y_left,
            'y_right': foul_y_right,
            'center_x': foul_center_x,
            'center_y': foul_center_y,
            'width': width,
            'height': height
        }
    
    return horizontal_lines, edges, foul_line_params


def detect_vertical_boundaries_approach1(img, foul_line_params, angle_mode='from_horizontal'):
    """
    Approach 1: Simple closest-to-center method.
    
    Returns:
    --------
    left_boundary, right_boundary, edges, left_lines_all, right_lines_all
    """
    if foul_line_params is None:
        return None, None, None, [], []
    
    height, width = img.shape[:2]
    
    gaussian = cv2.GaussianBlur(img, (11, 11), 0)
    greyscale = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    
    otsu_value, otsu_thresh = cv2.threshold(greyscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower_thresh = int(0.5 * otsu_value)
    upper_thresh = int(1.5 * otsu_value)
    
    contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_center_x = width // 2
    img_center_y = height // 2
    
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                if abs(cX - img_center_x) < width * 0.25 and abs(cY - img_center_y) < height * 0.25:
                    filtered_contours.append(contour)
    
    mask = np.zeros_like(greyscale)
    cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
    masked_greyscale = cv2.bitwise_and(greyscale, greyscale, mask=mask)
    
    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(masked_greyscale, kernel, iterations=1)
    
    kernel_er = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(dilated, kernel_er, iterations=1)
    
    edges = cv2.Canny(eroded, lower_thresh, upper_thresh)
    hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if hough_lines is None:
        return None, None, edges, [], []
    
    mid_x = foul_line_params['center_x']
    foul_center_x = foul_line_params['center_x']
    
    y_top_crop = height // 4
    y_bottom_crop = 3 * height // 4
    
    left_lines = []
    right_lines = []
    
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        
        mid_y = (y1 + y2) // 2
        if mid_y < y_top_crop or mid_y > y_bottom_crop:
            continue
        
        # Normalize line direction
        x1_norm, y1_norm, x2_norm, y2_norm = normalize_line_direction(x1, y1, x2, y2)
        
        # Check slope for vertical lines
        if x2_norm - x1_norm == 0:
            slope = float('inf')
        else:
            slope = (y2_norm - y1_norm) / (x2_norm - x1_norm)
            if abs(slope) <= 1:
                continue
        
        if line_crosses_midline(x1, y1, x2, y2, mid_x):
            continue
        
        x_intersect = get_line_foul_intersection(x1_norm, y1_norm, x2_norm, y2_norm, foul_line_params)
        
        if x_intersect is None:
            continue
        
        line_length = calculate_line_length(x1, y1, x2, y2)
        angle = calculate_line_angle(x1_norm, y1_norm, x2_norm, y2_norm, angle_mode)
        
        line_data = {
            'line': line[0],
            'x_intersect': x_intersect,
            'slope': slope,
            'angle': angle,
            'length': line_length
        }
        
        if x_intersect < foul_center_x:
            left_lines.append(line_data)
        elif x_intersect > foul_center_x:
            right_lines.append(line_data)
    
    left_boundary = None
    right_boundary = None
    
    if left_lines:
        best_left = max(left_lines, key=lambda x: x['x_intersect'])
        left_boundary = (best_left['line'], best_left['x_intersect'], best_left['slope'])
    
    if right_lines:
        best_right = min(right_lines, key=lambda x: x['x_intersect'])
        right_boundary = (best_right['line'], best_right['x_intersect'], best_right['slope'])
    
    return left_boundary, right_boundary, edges, left_lines, right_lines
