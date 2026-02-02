"""
Configuration file for bowling ball detection - Phase 2
Edit parameters here without touching the main code

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Last Updated: February 1, 2026
"""

import os

# ============================================
# PROJECT PATHS
# ============================================

# Base paths (relative to project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets', 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# List of videos to process (same as Phase 1)
VIDEO_FILES = [
    'cropped_test3.mp4',
    'cropped_test6.mp4',
    'cropped_test7.mp4',
    'cropped_test9.mp4',
    'cropped_test10.mp4'
]

# ============================================
# OUTPUT CONFIGURATION
# ============================================

# Output folder structure:
# output/<video_name>/ball_detection/
#   └── intermediate/  (masked videos, debug frames)

# Save masked lane video (for debugging/visualization)
SAVE_MASKED_VIDEO = True  # True to see the masked lane video

# Masking Configuration
MASK_TOP_BOUNDARY = False  # False = 3-boundary masking (L+R+B only, allows tracking to pins)
                           # True = 4-boundary masking (L+R+B+T, stops at top boundary)
                           # Note: Stage A (homography) always uses 4 boundaries regardless

# Save perspective-corrected video (overhead view)
SAVE_TRANSFORMED_VIDEO = True  # True to create transformed video

# ============================================
# PERSPECTIVE TRANSFORMATION
# ============================================

# Pixels per inch for transformed video
# IMPORTANT: Use SAME scale for width and height to preserve shapes (circles stay circles)
# Higher = better resolution but larger file size
TRANSFORM_SCALE = 20  # Pixels per inch (same for both dimensions to avoid distortion)
# This gives: 41.5in × 20 = 830px width, 720in × 20 = 14400px height

# Auto-crop transformed frames to remove black borders
AUTO_CROP_TRANSFORMED = True  # Crop to actual lane content
                              # This gives ~1:1.7 aspect ratio instead of 1:17

# Alternative: Use same scale for both (true proportions but very tall)
# TRANSFORM_SCALE = 10  # Would give 415x7200 (too thin to view)

# ============================================
# STAGE B: MOTION DETECTION (BACKGROUND SUBTRACTION)
# ============================================

# Background Subtraction Method
USE_MOG2 = True  # Use MOG2 (Mixture of Gaussians) for background subtraction

# MOG2 Parameters
MOG2_HISTORY = 500  # Number of frames for background learning (higher = slower adaptation)
MOG2_VAR_THRESHOLD = 40  # Threshold for squared Mahalanobis distance (lower = more sensitive, increased to reduce noise)
MOG2_DETECT_SHADOWS = True  # Detect shadows (they appear as grey pixels, value 127)

# Shadow Removal
SHADOW_THRESHOLD = 200  # Threshold to remove shadows (keep only 255, remove 127 and below)
                        # Shadows in OpenCV MOG2 are value 127, foreground is 255

# Morphological Noise Removal
MORPH_KERNEL_SIZE = 3  # Kernel size for morphological operations (3x3 or 5x5)
                       # Smaller = preserve small details, Larger = remove more noise
MORPH_KERNEL_SHAPE = 'ellipse'  # 'ellipse' for circular, 'rect' for rectangular

# Shadow Separation (CRITICAL for ball/shadow separation)
USE_SHADOW_SEPARATION = True  # Apply extra erosion to separate ball from shadow
SHADOW_SEPARATION_ITERATIONS = 2  # Number of erosion iterations (2-3 recommended)
                                   # Separates merged ball+shadow blobs

# Stage B Intermediate Videos (for debugging)
SAVE_FOREGROUND_MASK_VIDEO = True  # Raw MOG2 output (with shadows as grey)
SAVE_SHADOW_REMOVED_VIDEO = True   # After shadow thresholding
SAVE_DENOISED_VIDEO = True         # After morphological opening (final clean mask)

# ============================================
# STAGE C: ROI LOGIC (SEARCH STRATEGY)
# ============================================

# ROI Dynamic Scaling (Perspective-Aware)
B_MIN = 30  # Minimum ROI buffer size (pixels) - prevents ROI from becoming too small near pins
K_SCALE = 0.15  # Perspective scaling factor: B_t = max(B_min, k * y_ball)
                # Larger = bigger search box, Smaller = tighter tracking

# Ball Size Constraints (for contour filtering)
MIN_BALL_RADIUS = 5   # Minimum ball radius (pixels) - ball at 60ft (near pins)
MAX_BALL_RADIUS = 150  # Maximum ball radius (pixels) - ball at foul line

# Velocity Filtering (Global Search Mode)
MIN_VELOCITY_Y = -2  # Minimum Y velocity (negative = toward pins, away from camera)
                     # Filters out bowler's body and lateral movements
FOUL_LINE_PRIORITY_ZONE = 200  # Pixels from bottom boundary to prioritize detection
                               # Focus on area near foul line for initial detection

# Kalman Filter Parameters
KALMAN_PROCESS_NOISE = 1.0      # Process noise covariance (motion model uncertainty)
KALMAN_MEASUREMENT_NOISE = 10.0  # Measurement noise covariance (detection uncertainty)
                                 # Lower = trust measurements more, Higher = trust prediction more

# State Management
MAX_LOST_FRAMES = 5  # Maximum consecutive frames without detection before reverting to global search
                      # Prevents getting stuck in local mode when ball is lost
                      # Low value (2-3) prevents Kalman drift when ball hits pins

# Confirmation Logic (Problem 2 Solution - Prevents false search space pruning)
CONFIRMATION_THRESHOLD = 20  # Consecutive frames needed to confirm it's the ball (not hand)
                              # Will integrate with Stage D filters (circularity, aspect ratio)
                              # Currently just counts frames, Stage D will add geometric validation

SPATIAL_CONFIRMATION_DISTANCE = 240  # Minimum travel distance (pixels) to confirm it's the ball
                                      # Approximately 12 feet from foul line in perspective view
                                      # Prevents early search restriction if only tracking hand

SEARCH_BUFFER = 50  # Buffer (pixels) above last known Y position for restricted search
                    # Adds safety margin to account for prediction uncertainty

# Global Search Configuration (Two Types)
REACTIVATION_SEARCH_MARGIN = 20  # Pixels above last known Y to search when ball lost mid-lane
                                  # Searches toward pins (lower Y), with safety margin below last position
REACTIVATION_TIMEOUT = 100  # Frames of failed reactivation search before resetting to full frame
                            # If ball not found after this many frames in reactivation mode,
                            # reset to initial search (useful when ball completes path or leaves frame)
FOUL_LINE_EXCLUSION_FACTOR = 0.6  # Exclude upper 30% of frame in initial global search
                                   # Prevents tracking bowler's head instead of ball

# Stage C Intermediate Videos (for debugging)
SAVE_ROI_GLOBAL_SEARCH_VIDEO = True      # Global search mode visualization
SAVE_ROI_LOCAL_TRACKING_VIDEO = True     # Local tracking mode with ROI boxes
SAVE_KALMAN_PREDICTION_VIDEO = True      # Kalman predictions vs actual detections
SAVE_ROI_MODE_COMPARISON_VIDEO = True    # Side-by-side global vs local
SAVE_ROI_SCALING_DEMO_VIDEO = True       # Perspective scaling demonstration
SAVE_FULL_ROI_PIPELINE_VIDEO = True      # Complete 2x3 grid pipeline view

# ============================================================
# STAGE D: BLOB ANALYSIS & FILTERING
# ============================================================

# ----- Area Filter (Perspective Awareness) -----
# Manual thresholds (used if AUTO_CALIBRATE_AREA is False)
AREA_MAX_AT_FOUL = 400  # Maximum ball area near foul line (pixels²) - increased for cropped videos
AREA_MIN_AT_FOUL = 80   # Minimum ball area near foul line (pixels²) - increased for cropped videos
AREA_MAX_AT_PINS = 50   # Maximum ball area at pins (pixels²) - increased proportionally
AREA_MIN_AT_PINS = 10   # Minimum ball area at pins (pixels²) - increased proportionally

# Auto-calibration (Hybrid approach - adapts to video zoom/crop)
AUTO_CALIBRATE_AREA = True  # If True, detect ball in first frames and adapt thresholds
CALIBRATION_FRAMES = 30     # Number of frames to use for auto-calibration
CALIBRATION_MIN_CIRCULARITY = 0.7  # Minimum circularity to accept blob for calibration

# ----- Circularity Filter -----
CIRCULARITY_THRESHOLD = 0.70  # C = 4π·Area / Perimeter² (1.0 = perfect circle)
                               # 0.70 rejects shadows and irregular shapes
                               # Ball with slight motion blur still passes (~0.72-0.85)

# ----- Aspect Ratio Filter -----
ASPECT_RATIO_MAX = 1.8  # Major axis / Minor axis (allows motion blur elongation)
                        # Shadows are elongated (>2.0), ball even with blur is ~1.2-1.6

# ----- Color Verification (Optional) -----
ENABLE_COLOR_FILTER = False  # Set to True if you know the ball color
BALL_HSV_MIN = (100, 50, 50)   # Lower HSV bound (example: blue ball)
BALL_HSV_MAX = (130, 255, 255) # Upper HSV bound (example: blue ball)
COLOR_MATCH_THRESHOLD = 0.7    # Percentage of pixels that must match color range

# Stage D Intermediate Videos (for debugging)
SAVE_BLOB_ALL_CONTOURS_VIDEO = True          # All detected contours before filtering
SAVE_BLOB_AREA_FILTER_VIDEO = True           # Color-coded area filter results
SAVE_BLOB_CIRCULARITY_FILTER_VIDEO = True    # Circularity filter with values
SAVE_BLOB_ASPECT_RATIO_FILTER_VIDEO = True   # Aspect ratio filter with ellipses
SAVE_BLOB_COLOR_FILTER_VIDEO = True          # HSV color filter (if enabled)
SAVE_BLOB_FINAL_FILTERED_VIDEO = True        # Only blobs passing all filters
SAVE_BLOB_FILTER_COMPARISON_VIDEO = True     # Side-by-side raw vs filtered
SAVE_FULL_BLOB_PIPELINE_VIDEO = True         # 2x3 grid showing all stages

# ============================================
# STAGE F: STOP CONDITION (PIN IMPACT)
# ============================================

# Stop tracking when ball reaches pin area
ENABLE_STOP_CONDITION = True                 # Enable automatic stop when near pins
STOP_THRESHOLD_PCT = -0.03                   # Stop when Y ≤ (top_boundary + STOP_THRESHOLD_PCT * frame_height)
                                             # Negative = stop ABOVE top boundary (toward frame top)
                                             # Example: If top=130, height=954, stop at Y≤111 (130 - 19)

# Trajectory Interpolation using Kalman Filter predictions
INTERPOLATE_TO_BOUNDARY = True               # Extend trajectory beyond last detection using Kalman
NUM_KALMAN_PREDICTIONS_AFTER_STOP = 5        # Number of Kalman predictions to collect after stopping
                                             # These predictions simulate future ball positions
MARK_INTERPOLATED_POINTS = True              # Visual distinction in plots (dashed line)

# Visualization
SHOW_STOP_THRESHOLD_LINE = True              # Draw stop threshold in debug videos
STOP_THRESHOLD_COLOR = (255, 0, 255)         # Magenta line for stop threshold
INTERPOLATION_COLOR = (0, 165, 255)          # Orange for interpolated trajectory section

# ============================================
# DEBUG & LOGGING
# ============================================

DEBUG_MODE = False
VERBOSE = True
