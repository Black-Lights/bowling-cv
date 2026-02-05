"""
Stage D: Blob Analysis & Filtering
Identity verification layer - ensures we're tracking a bowling ball, not hands/arms/reflections.

Implements four filters:
1. Adaptive Area Filter (Perspective Awareness)
2. Circularity Filter (C > 0.65)
3. Aspect Ratio Filter (Ratio < 2.0)
4. Color Verification (Optional HSV check)

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Last Updated: February 1, 2026
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BlobMetrics:
    """Stores all calculated metrics for a blob"""
    contour: np.ndarray
    area: float
    circularity: float
    aspect_ratio: float
    centroid: Tuple[float, float]
    color_match: Optional[float] = None  # Percentage of pixels matching color
    
    # Filter results
    passes_area: bool = False
    passes_circularity: bool = False
    passes_aspect_ratio: bool = False
    passes_color: bool = True  # Default True if color filter disabled
    
    @property
    def passes_all_filters(self) -> bool:
        """Check if blob passes all enabled filters"""
        return (self.passes_area and 
                self.passes_circularity and 
                self.passes_aspect_ratio and 
                self.passes_color)


class BlobAnalyzer:
    """Stage D: Blob Analysis & Filtering"""
    
    def __init__(self, config):
        self.config = config
        
        # Area thresholds (will be set by calibration or config)
        self.area_max_at_foul = config.AREA_MAX_AT_FOUL
        self.area_min_at_foul = config.AREA_MIN_AT_FOUL
        self.area_max_at_pins = config.AREA_MAX_AT_PINS
        self.area_min_at_pins = config.AREA_MIN_AT_PINS
        
        # Calibration state
        self.is_calibrated = not config.AUTO_CALIBRATE_AREA
        self.calibration_samples = []
        
    def auto_calibrate(self, frame: np.ndarray, contours: List[np.ndarray], 
                      frame_height: int, foul_line_y: float) -> bool:
        """
        Auto-calibrate area thresholds by detecting ball in first N frames
        
        Args:
            frame: Current frame (BGR)
            contours: All detected contours
            frame_height: Total frame height
            foul_line_y: Y position of foul line (bottom boundary)
            
        Returns:
            True if calibration complete, False otherwise
        """
        if self.is_calibrated:
            return True
            
        # Look for circular blobs near foul line
        foul_zone_min_y = foul_line_y - 100  # 100px above foul line
        
        for contour in contours:
            # Skip tiny contours
            area = cv2.contourArea(contour)
            if area < 10:
                continue
                
            # Get centroid
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            
            # Check if in foul zone
            if cy < foul_zone_min_y or cy > foul_line_y:
                continue
                
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity >= self.config.CALIBRATION_MIN_CIRCULARITY:
                self.calibration_samples.append(area)
                
        # Check if we have enough samples
        if len(self.calibration_samples) >= self.config.CALIBRATION_FRAMES:
            # Use median to avoid outliers
            median_area = np.median(self.calibration_samples)
            
            # Set thresholds based on detected ball size
            self.area_max_at_foul = median_area * 1.5  # 50% tolerance
            self.area_min_at_foul = median_area * 0.5
            
            # Maintain perspective ratio (assume 6-10x shrinkage at pins)
            perspective_ratio = 8.0
            self.area_max_at_pins = self.area_max_at_foul / perspective_ratio
            self.area_min_at_pins = self.area_min_at_foul / perspective_ratio
            
            self.is_calibrated = True
            
            print(f"\n{'='*60}")
            print("AUTO-CALIBRATION COMPLETE")
            print(f"{'='*60}")
            print(f"Samples collected: {len(self.calibration_samples)}")
            print(f"Detected ball area at foul line: {median_area:.1f} px²")
            print(f"Area thresholds at foul: [{self.area_min_at_foul:.1f}, {self.area_max_at_foul:.1f}]")
            print(f"Area thresholds at pins: [{self.area_min_at_pins:.1f}, {self.area_max_at_pins:.1f}]")
            print(f"{'='*60}\n")
            
            return True
            
        return False
        
    def calculate_adaptive_area_thresholds(self, y_pos: float, 
                                          frame_height: int) -> Tuple[float, float]:
        """
        Calculate perspective-aware area thresholds based on Y position
        
        Args:
            y_pos: Vertical position of blob centroid (pixels)
            frame_height: Total frame height (pixels)
            
        Returns:
            (area_min, area_max) tuple
        """
        # Linear interpolation based on y position
        # y=0 (pins) -> use pins thresholds
        # y=frame_height (foul line) -> use foul thresholds
        
        t = y_pos / frame_height  # Normalized position [0, 1]
        
        area_min = self.area_min_at_pins + t * (self.area_min_at_foul - self.area_min_at_pins)
        area_max = self.area_max_at_pins + t * (self.area_max_at_foul - self.area_max_at_pins)
        
        return area_min, area_max
        
    def area_filter(self, contour: np.ndarray, y_pos: float, 
                   frame_height: int) -> Tuple[bool, float]:
        """
        Adaptive Area Filter - perspective-aware size validation
        
        Args:
            contour: OpenCV contour
            y_pos: Y position of centroid
            frame_height: Total frame height
            
        Returns:
            (passes_filter, area) tuple
        """
        area = cv2.contourArea(contour)
        area_min, area_max = self.calculate_adaptive_area_thresholds(y_pos, frame_height)
        
        passes = area_min <= area <= area_max
        return passes, area
        
    def circularity_filter(self, contour: np.ndarray) -> Tuple[bool, float]:
        """
        Circularity Filter - primary defense against hand tracking
        
        Formula: C = 4π·Area / Perimeter²
        Perfect circle: C = 1.0
        Threshold: C > 0.65 (accommodates motion blur)
        
        Args:
            contour: OpenCV contour
            
        Returns:
            (passes_filter, circularity) tuple
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return False, 0.0
            
        circularity = 4 * np.pi * area / (perimeter ** 2)
        passes = circularity >= self.config.CIRCULARITY_THRESHOLD
        
        return passes, circularity
        
    def aspect_ratio_filter(self, contour: np.ndarray) -> Tuple[bool, float, Optional[Tuple]]:
        """
        Aspect Ratio Filter - distinguishes ball from elongated objects
        
        Fits ellipse and checks Major/Minor axis ratio < 2.0
        Hands/arms are more elongated and will be rejected
        
        Args:
            contour: OpenCV contour
            
        Returns:
            (passes_filter, aspect_ratio, ellipse) tuple
            ellipse is ((cx, cy), (MA, ma), angle) or None if fit failed
        """
        # Need at least 5 points to fit ellipse
        if len(contour) < 5:
            return False, 999.0, None
            
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (MA, ma), angle = ellipse
            
            # Avoid division by zero
            if ma == 0:
                return False, 999.0, ellipse
                
            # Major axis is always >= minor axis in OpenCV
            aspect_ratio = MA / ma
            passes = aspect_ratio <= self.config.ASPECT_RATIO_MAX
            
            return passes, aspect_ratio, ellipse
            
        except cv2.error:
            return False, 999.0, None
            
    def color_filter(self, frame: np.ndarray, contour: np.ndarray) -> Tuple[bool, float]:
        """
        Color Verification - HSV-based color matching (optional)
        
        Prevents white/yellow reflections from being tracked as ball
        
        Args:
            frame: BGR frame
            contour: OpenCV contour
            
        Returns:
            (passes_filter, match_percentage) tuple
        """
        if not self.config.ENABLE_COLOR_FILTER:
            return True, 1.0  # Skip if disabled
            
        # Create mask from contour
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create color range mask
        color_mask = cv2.inRange(hsv, self.config.BALL_HSV_MIN, self.config.BALL_HSV_MAX)
        
        # Calculate percentage of pixels matching color
        roi_pixels = cv2.countNonZero(mask)
        if roi_pixels == 0:
            return False, 0.0
            
        matching_pixels = cv2.countNonZero(cv2.bitwise_and(color_mask, mask))
        match_percentage = matching_pixels / roi_pixels
        
        passes = match_percentage >= self.config.COLOR_MATCH_THRESHOLD
        return passes, match_percentage
        
    def analyze_blob(self, frame: np.ndarray, contour: np.ndarray, 
                    frame_height: int) -> BlobMetrics:
        """
        Analyze a single blob against all filters
        
        Args:
            frame: BGR frame (for color filter)
            contour: OpenCV contour
            frame_height: Total frame height (for adaptive area)
            
        Returns:
            BlobMetrics object with all results
        """
        # Calculate centroid
        M = cv2.moments(contour)
        if M['m00'] == 0:
            cx, cy = 0, 0
        else:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            
        # Run all filters
        passes_area, area = self.area_filter(contour, cy, frame_height)
        passes_circ, circularity = self.circularity_filter(contour)
        passes_aspect, aspect_ratio, ellipse = self.aspect_ratio_filter(contour)
        passes_color, color_match = self.color_filter(frame, contour)
        
        # Create metrics object
        metrics = BlobMetrics(
            contour=contour,
            area=area,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            centroid=(cx, cy),
            color_match=color_match,
            passes_area=passes_area,
            passes_circularity=passes_circ,
            passes_aspect_ratio=passes_aspect,
            passes_color=passes_color
        )
        
        return metrics
        
    def filter_blobs(self, frame: np.ndarray, mask: np.ndarray, 
                    frame_idx: int) -> List[BlobMetrics]:
        """
        Main entry point: Find and filter all blobs in current frame
        
        Args:
            frame: BGR frame
            mask: Binary motion mask from Stage B
            frame_idx: Current frame index (for calibration)
            
        Returns:
            List of BlobMetrics for all analyzed blobs
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Auto-calibrate if needed (first N frames)
        if not self.is_calibrated and frame_idx < self.config.CALIBRATION_FRAMES:
            frame_height = frame.shape[0]
            # Assume foul line is near bottom (will get actual value from boundary data)
            foul_line_y = frame_height * 0.9
            self.auto_calibrate(frame, contours, frame_height, foul_line_y)
            
        # Analyze all blobs
        frame_height = frame.shape[0]
        blob_metrics = []
        
        for contour in contours:
            # Skip tiny contours (noise)
            if cv2.contourArea(contour) < 3:
                continue
                
            metrics = self.analyze_blob(frame, contour, frame_height)
            blob_metrics.append(metrics)
            
        return blob_metrics
        
    def get_filtered_blobs(self, blob_metrics: List[BlobMetrics]) -> List[BlobMetrics]:
        """
        Get only blobs that pass all filters
        
        Args:
            blob_metrics: List of all analyzed blobs
            
        Returns:
            List of blobs passing all filters
        """
        return [blob for blob in blob_metrics if blob.passes_all_filters]
