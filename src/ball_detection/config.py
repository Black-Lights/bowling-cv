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
MOG2_VAR_THRESHOLD = 16  # Threshold for squared Mahalanobis distance (lower = more sensitive)
MOG2_DETECT_SHADOWS = True  # Detect shadows (they appear as grey pixels, value 127)

# Shadow Removal
SHADOW_THRESHOLD = 200  # Threshold to remove shadows (keep only 255, remove 127 and below)
                        # Shadows in OpenCV MOG2 are value 127, foreground is 255

# Morphological Noise Removal (Opening = Erosion then Dilation)
MORPH_KERNEL_SIZE = 3  # Kernel size for morphological operations (3x3 or 5x5)
                       # Smaller = preserve small details, Larger = remove more noise
MORPH_KERNEL_SHAPE = 'ellipse'  # 'ellipse' for circular, 'rect' for rectangular

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
MAX_BALL_RADIUS = 50  # Maximum ball radius (pixels) - ball at foul line

# Velocity Filtering (Global Search Mode)
MIN_VELOCITY_Y = -2  # Minimum Y velocity (negative = toward pins, away from camera)
                     # Filters out bowler's body and lateral movements
FOUL_LINE_PRIORITY_ZONE = 100  # Pixels from bottom boundary to prioritize detection
                               # Focus on area near foul line for initial detection

# Kalman Filter Parameters
KALMAN_PROCESS_NOISE = 1.0      # Process noise covariance (motion model uncertainty)
KALMAN_MEASUREMENT_NOISE = 10.0  # Measurement noise covariance (detection uncertainty)
                                 # Lower = trust measurements more, Higher = trust prediction more

# State Management
MAX_LOST_FRAMES = 1  # Maximum consecutive frames without detection before reverting to global search
                      # Prevents getting stuck in local mode when ball is lost

# Confirmation Logic (Problem 2 Solution - Prevents false search space pruning)
CONFIRMATION_THRESHOLD = 20  # Consecutive frames needed to confirm it's the ball (not hand)
                              # Will integrate with Stage D filters (circularity, aspect ratio)
                              # Currently just counts frames, Stage D will add geometric validation

SPATIAL_CONFIRMATION_DISTANCE = 240  # Minimum travel distance (pixels) to confirm it's the ball
                                      # Approximately 12 feet from foul line in perspective view
                                      # Prevents early search restriction if only tracking hand

SEARCH_BUFFER = 50  # Buffer (pixels) above last known Y position for restricted search
                    # Adds safety margin to account for prediction uncertainty

# Stage C Intermediate Videos (for debugging)
SAVE_ROI_GLOBAL_SEARCH_VIDEO = True      # Global search mode visualization
SAVE_ROI_LOCAL_TRACKING_VIDEO = True     # Local tracking mode with ROI boxes
SAVE_KALMAN_PREDICTION_VIDEO = True      # Kalman predictions vs actual detections
SAVE_ROI_MODE_COMPARISON_VIDEO = True    # Side-by-side global vs local
SAVE_ROI_SCALING_DEMO_VIDEO = True       # Perspective scaling demonstration
SAVE_FULL_ROI_PIPELINE_VIDEO = True      # Complete 2x3 grid pipeline view

# ============================================
# DEBUG & LOGGING
# ============================================

DEBUG_MODE = False
VERBOSE = True
