import cv2
import numpy as np
from config.config import Config

class BallDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=Config.BG_HISTORY,
            varThreshold=Config.BG_VAR_THRESHOLD,
            detectShadows=Config.BG_DETECT_SHADOWS
        )
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Config.MORPH_OPEN_KERNEL_SIZE)
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Config.MORPH_CLOSE_KERNEL_SIZE)

    def detect(self, frame, lane_mask, roi_offset=(0,0)):
        """
        roi_offset: If we are processing a cropped frame, we need to know the offset
                    to return global coordinates.
        """
        # 1. Apply Mask (Focus only on lane pixels)
        masked_frame = cv2.bitwise_and(frame, frame, mask=lane_mask)
        
        # 2. Background Subtraction
        fg_mask = self.bg_subtractor.apply(masked_frame)
        
        # 3. Shadow Removal (Shadows are gray/127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # 4. Morphological Cleanup
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        
        # 5. Contour Extraction
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        max_circularity = 0
        
        for cnt in contours:
            # --- GEOMETRIC FILTERING ---
            
            # Filter A: Area
            area = cv2.contourArea(cnt)
            if not (Config.BALL_MIN_AREA < area < Config.BALL_MAX_AREA):
                continue
                
            # Filter B: Circularity (The Bowler Killer)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < Config.MIN_CIRCULARITY:
                continue
                
            # Filter C: Aspect Ratio
            x, y, w, h = cv2.boundingRect(cnt)
            ar = float(w) / h
            if not (Config.MIN_ASPECT_RATIO < ar < Config.MAX_ASPECT_RATIO):
                continue
                
            # If we pass all filters, pick the most circular one (most likely the ball)
            if circularity > max_circularity:
                max_circularity = circularity
                # Calculate centroid
                M = cv2.moments(cnt)
                if M["m00"]!= 0:
                    cX = int(M["m10"] / M["m00"]) + roi_offset
                    cY = int(M["m01"] / M["m00"]) + roi_offset
                    best_candidate = (cX, cY, cnt) # Return global coords
                    
        return best_candidate, fg_mask