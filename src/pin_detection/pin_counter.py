"""
Pin Counter for Pin Detection

Performs frame differencing and contour detection to count remaining pins.
Uses morphological operations and geometric filtering for robust detection.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

import cv2
import numpy as np
import os

from . import config


class PinCounter:
    """
    Counts remaining bowling pins using frame differencing and contour detection.
    
    Algorithm:
    1. Compute absolute difference between before/after frames
    2. Apply threshold to create binary mask
    3. Morphological operations to clean noise
    4. Find contours
    5. Filter contours by geometric properties
    6. Count valid pins
    """
    
    def __init__(self):
        """Initialize pin counter with configuration parameters."""
        self.difference_threshold = config.DIFFERENCE_THRESHOLD
        self.kernel_size = config.MORPH_KERNEL_SIZE
        self.morph_iterations = config.MORPH_ITERATIONS
        
        # Create morphological kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.kernel_size, self.kernel_size)
        )
        
        # Contour filtering parameters
        self.min_area = config.MIN_PIN_AREA
        self.max_area = config.MAX_PIN_AREA
        self.min_aspect = config.MIN_PIN_ASPECT_RATIO
        self.max_aspect = config.MAX_PIN_ASPECT_RATIO
        self.min_solidity = config.MIN_PIN_SOLIDITY
        
        # Results
        self.difference = None
        self.binary_diff = None
        self.cleaned_diff = None
        self.valid_contours = []
        self.all_contours = []
        
        if config.VERBOSE:
            print(f"\n🎯 PinCounter initialized:")
            print(f"   Difference threshold: {self.difference_threshold}")
            print(f"   Morphology kernel: {self.kernel_size}x{self.kernel_size}")
            print(f"   Pin area range: {self.min_area}-{self.max_area} px²")
    
    def compute_difference(self, before_frame, after_frame):
        """
        Compute absolute difference between before and after frames.
        
        Parameters:
        -----------
        before_frame : numpy.ndarray
            Before frame (BGR or grayscale)
        after_frame : numpy.ndarray
            After frame (BGR or grayscale)
            
        Returns:
        --------
        numpy.ndarray : Absolute difference (grayscale)
        """
        # Convert to grayscale if needed
        if len(before_frame.shape) == 3:
            before_gray = cv2.cvtColor(before_frame, cv2.COLOR_BGR2GRAY)
        else:
            before_gray = before_frame
        
        if len(after_frame.shape) == 3:
            after_gray = cv2.cvtColor(after_frame, cv2.COLOR_BGR2GRAY)
        else:
            after_gray = after_frame
        
        # Compute absolute difference
        self.difference = cv2.absdiff(before_gray, after_gray)
        
        if config.VERBOSE:
            diff_stats = {
                'min': np.min(self.difference),
                'max': np.max(self.difference),
                'mean': np.mean(self.difference),
                'std': np.std(self.difference)
            }
            print(f"\n📊 Frame difference computed:")
            print(f"   Range: {diff_stats['min']:.1f} - {diff_stats['max']:.1f}")
            print(f"   Mean: {diff_stats['mean']:.1f} ± {diff_stats['std']:.1f}")
        
        return self.difference
    
    def apply_threshold(self):
        """
        Apply binary threshold to difference image.
        
        Returns:
        --------
        numpy.ndarray : Binary mask (255 = changed, 0 = unchanged)
        """
        _, self.binary_diff = cv2.threshold(
            self.difference,
            self.difference_threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        if config.VERBOSE:
            changed_pixels = np.sum(self.binary_diff == 255)
            total_pixels = self.binary_diff.size
            changed_percent = (changed_pixels / total_pixels) * 100
            print(f"\n🎨 Binary threshold applied:")
            print(f"   Changed pixels: {changed_pixels:,} ({changed_percent:.1f}%)")
        
        return self.binary_diff
    
    def apply_morphology(self):
        """
        Apply morphological operations to clean binary mask.
        
        Operations:
        1. Opening - Remove small noise
        2. Closing - Fill small holes
        
        Returns:
        --------
        numpy.ndarray : Cleaned binary mask
        """
        # Opening: Remove small noise (erosion then dilation)
        opened = cv2.morphologyEx(
            self.binary_diff,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=self.morph_iterations
        )
        
        # Closing: Fill small holes (dilation then erosion)
        self.cleaned_diff = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=self.morph_iterations
        )
        
        if config.VERBOSE:
            print(f"\n🧹 Morphological operations applied:")
            print(f"   Opening iterations: {self.morph_iterations}")
            print(f"   Closing iterations: {self.morph_iterations}")
        
        return self.cleaned_diff
    
    def find_contours(self):
        """
        Find contours in cleaned binary mask.
        
        Returns:
        --------
        list : All contours found
        """
        # Find contours
        contours, _ = cv2.findContours(
            self.cleaned_diff,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        self.all_contours = list(contours)
        
        if config.VERBOSE:
            print(f"\n🔍 Contours detected: {len(self.all_contours)}")
        
        return self.all_contours
    
    def filter_contours(self, top_y_max=None):
        """
        Filter contours by geometric properties to identify valid pins.
        
        Filters:
        1. Y-position: Contour must be in pin area (y < top_y_max)
        2. Area: MIN_PIN_AREA < area < MAX_PIN_AREA
        3. Aspect ratio: MIN_PIN_ASPECT_RATIO < w/h < MAX_PIN_ASPECT_RATIO
        4. Solidity: contour_area / convex_hull_area > MIN_PIN_SOLIDITY
        
        Parameters:
        -----------
        top_y_max : int, optional
            Maximum Y coordinate for pins (above this is pin area)
            
        Returns:
        --------
        list : Valid pin contours
        """
        self.valid_contours = []
        rejected_contours = []
        
        for idx, contour in enumerate(self.all_contours):
            # Calculate bounding rectangle first for Y-position check
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by Y-position if specified (pins should be in upper area)
            if top_y_max is not None:
                contour_center_y = y + h // 2
                if contour_center_y > top_y_max:
                    rejected_contours.append((idx, 0, "Below pin area", f"y={contour_center_y} > {top_y_max}"))
                    continue
            
            # Calculate geometric properties
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area:
                rejected_contours.append((idx, area, "Area too small", f"{area:.0f} < {self.min_area}"))
                continue
            if area > self.max_area:
                rejected_contours.append((idx, area, "Area too large", f"{area:.0f} > {self.max_area}"))
                continue
            
            # Calculate aspect ratio
            if h == 0:
                rejected_contours.append((idx, area, "Invalid height", "h=0"))
                continue
            aspect_ratio = w / h
            
            # Filter by aspect ratio
            if aspect_ratio < self.min_aspect:
                rejected_contours.append((idx, area, "Aspect too low", f"{aspect_ratio:.2f} < {self.min_aspect}"))
                continue
            if aspect_ratio > self.max_aspect:
                rejected_contours.append((idx, area, "Aspect too high", f"{aspect_ratio:.2f} > {self.max_aspect}"))
                continue
            
            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                rejected_contours.append((idx, area, "Invalid hull", "hull_area=0"))
                continue
            solidity = area / hull_area
            
            # Filter by solidity
            if solidity < self.min_solidity:
                rejected_contours.append((idx, area, "Solidity too low", f"{solidity:.2f} < {self.min_solidity}"))
                continue
            
            # Valid pin contour
            self.valid_contours.append({
                'contour': contour,
                'area': area,
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'center': (x + w//2, y + h//2)
            })
        
        if config.VERBOSE:
            print(f"   ✅ Valid pin contours: {len(self.valid_contours)}")
            if len(self.valid_contours) > 0:
                print(f"\n   Valid contour details:")
                for i, pin in enumerate(self.valid_contours, 1):
                    print(f"      Pin {i}: area={pin['area']:.0f}, "
                          f"aspect={pin['aspect_ratio']:.2f}, "
                          f"solidity={pin['solidity']:.2f}, "
                          f"center=({pin['center'][0]}, {pin['center'][1]})")
            
            # Show rejection reasons
            if len(rejected_contours) > 0:
                print(f"\n   ⚠️  Rejected {len(rejected_contours)} contours:")
                for idx, area, reason, detail in rejected_contours[:10]:  # Show first 10
                    print(f"      Contour {idx}: {reason} ({detail})")
                if len(rejected_contours) > 10:
                    print(f"      ... and {len(rejected_contours) - 10} more")
        
        return self.valid_contours
    
    def count_pins(self, before_frame, after_frame, boundary_data=None):
        """
        Complete pin counting pipeline.
        
        Parameters:
        -----------
        before_frame : numpy.ndarray
            Before frame (all pins standing)
        after_frame : numpy.ndarray
            After frame (some pins remaining)
        boundary_data : dict, optional
            Boundary data from Phase 1 to focus on pin area only
            
        Returns:
        --------
        dict : Pin counting results
        """
        # Step 1: Compute difference (for visualization only)
        self.compute_difference(before_frame, after_frame)
        
        # Step 2: Apply threshold to difference (for visualization)
        self.apply_threshold()
        
        # Step 3: Morphological operations on difference (for visualization)
        self.apply_morphology()
        
        # ===== NEW APPROACH: Detect pins directly in AFTER frame =====
        # Convert after frame to grayscale
        if len(after_frame.shape) == 3:
            after_gray = cv2.cvtColor(after_frame, cv2.COLOR_BGR2GRAY)
        else:
            after_gray = after_frame
        
        # Apply ROI: Focus only on top portion where pins are located
        frame_height = after_gray.shape[0]
        roi_height = int(frame_height * config.FOCUS_TOP_FRACTION)
        after_roi = after_gray[:roi_height, :]
        
        # If boundary data provided, mask out everything below top boundary
        if boundary_data is not None:
            top_y = int(boundary_data['top_boundary']['y_position'])
            # Create mask for pin area only (above top boundary)
            pin_area_mask = np.zeros_like(after_roi, dtype=np.uint8)
            pin_area_mask[:top_y, :] = 255
        else:
            # Use entire ROI
            pin_area_mask = np.ones_like(after_roi, dtype=np.uint8) * 255
        
        # Apply threshold to isolate WHITE pins from dark background
        # Pins are very bright white, increase threshold to 150-180 for better isolation
        _, pins_binary = cv2.threshold(
            after_roi,
            config.PIN_DETECTION_THRESHOLD,  # Only very white pins (not gray lane surface)
            255,
            cv2.THRESH_BINARY
        )
        
        # Apply pin area mask to focus only on region above top boundary
        pins_binary = cv2.bitwise_and(pins_binary, pins_binary, mask=pin_area_mask)
        
        # Apply morphology to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Smaller kernel
        pins_cleaned = cv2.morphologyEx(pins_binary, cv2.MORPH_OPEN, kernel, iterations=1)
        pins_cleaned = cv2.morphologyEx(pins_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Store for visualization (expand to full frame size for visualization)
        self.cleaned_diff = np.zeros_like(after_gray)
        self.cleaned_diff[:roi_height, :] = pins_cleaned
        
        # Step 4: Find contours in AFTER frame (not difference)
        contours, _ = cv2.findContours(
            pins_cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        self.all_contours = list(contours)
        
        if config.VERBOSE:
            print(f"\n🔍 Contours detected in AFTER frame (ROI): {len(self.all_contours)}")
            if boundary_data:
                print(f"   Pin area height: 0 - {top_y} pixels")
        
        # Step 5: Filter contours (now with Y-position filtering)
        self.filter_contours(top_y_max=roi_height if boundary_data is None else top_y)
        
        # Step 6: Count pins
        remaining_pins = len(self.valid_contours)
        toppled_pins = config.TOTAL_PINS - remaining_pins
        
        # Validate counts
        if remaining_pins > config.TOTAL_PINS and not config.ALLOW_OVER_COUNT:
            print(f"\n⚠️  Warning: Detected {remaining_pins} pins (> {config.TOTAL_PINS})")
            print(f"   This may indicate detection errors. Review parameters.")
        
        if remaining_pins < 0 and not config.ALLOW_UNDER_COUNT:
            print(f"\n⚠️  Warning: Invalid pin count: {remaining_pins}")
        
        # Determine result
        if toppled_pins == config.TOTAL_PINS:
            result = f"STRIKE! ({toppled_pins}/{config.TOTAL_PINS})"
            is_strike = True
            is_spare = False
        elif remaining_pins == 0:
            result = f"All Pins Down ({toppled_pins}/{config.TOTAL_PINS})"
            is_strike = False
            is_spare = True  # Could be spare on second ball
        elif remaining_pins == config.TOTAL_PINS:
            result = f"Gutter Ball (0/{config.TOTAL_PINS})"
            is_strike = False
            is_spare = False
        else:
            result = f"{toppled_pins} Pin{'s' if toppled_pins != 1 else ''} Down ({toppled_pins}/{config.TOTAL_PINS})"
            is_strike = False
            is_spare = False
        
        # Compile results
        results = {
            'remaining_pins': remaining_pins,
            'toppled_pins': toppled_pins,
            'total_pins': config.TOTAL_PINS,
            'result': result,
            'is_strike': is_strike,
            'is_spare': is_spare,
            'valid_contours': self.valid_contours,
            'total_contours_found': len(self.all_contours),
            'detection_confidence': self._calculate_confidence()
        }
        
        if config.VERBOSE:
            print(f"\n🎳 Pin Counting Results:")
            print(f"   Remaining pins: {remaining_pins}")
            print(f"   Toppled pins:   {toppled_pins}")
            print(f"   Result:         {result}")
            print(f"   Confidence:     {results['detection_confidence']:.1%}")
        
        return results
    
    def _calculate_confidence(self):
        """
        Calculate detection confidence based on various factors.
        
        Returns:
        --------
        float : Confidence score (0.0 - 1.0)
        """
        confidence = 1.0
        
        # Reduce confidence if count is unrealistic
        remaining_pins = len(self.valid_contours)
        if remaining_pins > config.TOTAL_PINS:
            confidence *= 0.5  # Low confidence on over-count
        
        # Reduce confidence if very few contours were found
        if len(self.all_contours) < 3 and remaining_pins > 0:
            confidence *= 0.7  # Might be missing pins
        
        # Reduce confidence if too many contours were filtered out
        if len(self.all_contours) > 30:
            confidence *= 0.8  # Lots of noise
        
        return max(0.0, min(1.0, confidence))


if __name__ == "__main__":
    # Test with sample frames (requires frames from frame_selector)
    test_video = 'cropped_test3'
    intermediate_dir = config.get_intermediate_output_dir(test_video)
    
    print("\n" + "="*80)
    print("TESTING PIN COUNTING")
    print("="*80)
    
    # Find before/after frames
    import glob
    before_files = glob.glob(os.path.join(intermediate_dir, 'before_frame_*_roi.png'))
    after_files = glob.glob(os.path.join(intermediate_dir, 'after_frame_*_roi.png'))
    
    if not before_files or not after_files:
        print("❌ No frames found. Run frame_selector.py first!")
    else:
        before_frame = cv2.imread(before_files[0])
        after_frame = cv2.imread(after_files[0])
        
        print(f"\n📸 Loaded frames:")
        print(f"   Before: {before_files[0]}")
        print(f"   After:  {after_files[0]}")
        
        # Initialize counter
        counter = PinCounter()
        
        # Count pins
        results = counter.count_pins(before_frame, after_frame)
        
        print(f"\n✅ Testing complete!")
        print(f"   Result: {results['result']}")
