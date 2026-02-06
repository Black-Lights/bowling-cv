"""
Configuration for Spin Analysis Module - Phase 3

All parameters for spin detection, post-processing, and analysis.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

import os
import cv2

# ============================================
# PROJECT PATHS
# ============================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets', 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# Input from Ball Detection (Phase 2)
BALL_DETECTION_SUBDIR = 'ball_detection'
TRAJECTORY_INPUT_FILE = 'trajectory_processed_original.csv'

# Output structure
SPIN_ANALYSIS_SUBDIR = 'spin_analysis'
DEBUG_SUBDIR = 'debug'

# List of videos to process
VIDEO_FILES = [
    'cropped_test3.mp4',
    'cropped_test6.mp4',
    'cropped_test7.mp4'
]

# ============================================
# OUTPUT FILES
# ============================================

RAW_SPIN_FILENAME = "ball_spin_raw.csv"
PROCESSED_SPIN_FILENAME = "ball_spin_processed.csv"
REPORT_FILENAME = "spin_analysis_report.txt"
VISUALIZATION_FILENAME = "spin_visualization.png"

# Stage A outputs
STAGE_A_TRAJECTORY_CSV = "stage_a_trajectory_prepared.csv"
STAGE_A_VALIDATION_PLOT = "stage_a_trajectory_validation.png"

# Stage B outputs
STAGE_B_OPTICAL_FLOW_TEST = "stage_b_optical_flow_test.png"
STAGE_B_FEATURE_DISTRIBUTION = "stage_b_feature_distribution.png"
STAGE_B_FEATURE_SUMMARY = "stage_b_feature_count_summary.png"  # Feature count vs frame plot

# Stage C outputs
STAGE_C_3D_PROJECTION_PLOT = "stage_c_3d_projection.png"
STAGE_C_3D_POINTS_CSV = "stage_c_3d_points.csv"

# Stage D outputs
STAGE_D_ROTATION_ANALYSIS_PLOT = "stage_d_rotation_analysis.png"
STAGE_D_ROTATION_VECTORS_CSV = "stage_d_rotation_vectors.csv"

# ============================================
# OPTICAL FLOW PARAMETERS (Stage B)
# ============================================

# Feature Detection (Shi-Tomasi Corner Detection)
FEATURE_PARAMS = dict(
    maxCorners=100,          # Maximum number of features to track
    qualityLevel=0.001,      # Quality threshold for feature detection
    minDistance=0,           # Minimum distance between features (0 = no constraint)
    blockSize=3              # Size of averaging block for corner detection
)

# Lucas-Kanade Optical Flow
LK_PARAMS = dict(
    winSize=(21, 21),        # Search window size for optical flow
    maxLevel=3,              # Number of pyramid levels (0 = no pyramid)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01)
    # Termination criteria: max 15 iterations or 0.01 epsilon
)

# ============================================
# FILTERING PARAMETERS (Stage B & C)
# ============================================

# Optical Flow Validation
FB_ERROR_THRESHOLD = 15.0           # Forward-backward error threshold (pixels)
                                     # Lower = stricter (more false negatives)
                                     # Higher = more lenient (more false positives)
                                     # Increased to 15 for better detection on small balls

LOW_THRESHOLD_FACTOR = 10           # Divisor for minimum movement threshold (legacy)
                                     # movement > radius/LOW_THRESHOLD_FACTOR
                                     # Higher = require less movement (more lenient)
                                     # Note: Deprecated in favor of exponential threshold

# Exponential Movement Threshold (replaces LOW_THRESHOLD_FACTOR)
USE_EXPONENTIAL_THRESHOLD = True    # Use exponential decay for movement threshold
BASE_MOVEMENT_THRESHOLD = 4.0       # Base minimum movement for largest ball (pixels)
                                     # Increased from 3.0 for stricter filtering
MOVEMENT_THRESHOLD_DECAY = 0.6      # Exponential decay power (0.5 = square root decay)
                                     # Increased from 0.5 to keep thresholds higher for small balls
                                     # Lower = more aggressive decay (smaller balls have much lower thresholds)
                                     # Higher = gentler decay (threshold stays higher for small balls)

# ROI and Masking
ROI_OFFSET = 2                      # Extra padding around ball ROI (pixels)
BALL_MASK_RADIUS_FACTOR = 0.8       # Use 80% of ball radius for feature mask
                                     # With HSV filtering, can be less conservative
                                     # HSV removes ground contamination by color

# HSV Color Filtering (remove background: brown lane, white pins)
USE_HSV_FILTERING = True            # Apply HSV color filtering to remove non-ball features
HSV_LOWER_BROWN = (5, 30, 50)      # Brown lane lower bound (H, S, V)
HSV_UPPER_BROWN = (25, 200, 200)   # Brown lane upper bound
HSV_LOWER_WHITE = (0, 0, 180)      # White pins lower bound (high V, low S)
HSV_UPPER_WHITE = (180, 50, 255)   # White pins upper bound
# Note: Ball is dark (low V), so we keep pixels with V < 180 and not brown

# ============================================
# POST-PROCESSING PARAMETERS (Stage E)
# ============================================

# Outlier Removal
Z_SCORE_THRESHOLD = 3.0             # Z-score threshold for angular velocity outliers
                                     # 3.0 = remove points > 3 std deviations from mean

LINEAR_FIT_STD_MULTIPLIER = 2.0     # Std multiplier for linear fit outliers
                                     # Points beyond fit ± 2*std are removed

# Smoothing
GAUSSIAN_SIGMA = 2.0                # Sigma for Gaussian smoothing
                                     # Higher = more smoothing (smoother but less responsive)
                                     # Lower = less smoothing (more responsive but noisier)

# ============================================
# ANALYSIS PARAMETERS (Stage F)
# ============================================

# Zone Division (for axis migration analysis)
RELEASE_ZONE_FRACTION = 0.33        # First 33% of trajectory
MIDLANE_ZONE_FRACTION = 0.34        # Middle 34% of trajectory  
ENTRY_ZONE_FRACTION = 0.33          # Last 33% of trajectory

# ============================================
# VISUALIZATION SETTINGS
# ============================================

# Master switches
SAVE_PLOTS = True                   # Save analysis plots
PLOT_DPI = 150                      # DPI for saved plots

# Stage-specific visualization controls
GENERATE_STAGE_A_PLOT = False       # Trajectory validation plot (x,y position + radius over time)
GENERATE_STAGE_B_PLOT = True        # Optical flow test visualization (feature tracking)
GENERATE_STAGE_B_SUMMARY = True     # Feature count summary plot (tracked vs good features)
GENERATE_STAGE_C_PLOT = True        # 3D projection scatter plots (sphere surface visualization)
GENERATE_STAGE_D_PLOT = True        # Rotation analysis (axis vectors, angles over time)

# Stage C visualization options
STAGE_C_NUM_SAMPLE_FRAMES = 5       # Number of sample frames to visualize in 3D plots
STAGE_C_SHOW_STATISTICS = True      # Include statistics text in 3D visualization

# Debug visualization colors
COLOR_GOOD_FEATURES = (0, 255, 0)   # Green for good tracked features
COLOR_BAD_FEATURES = (0, 0, 255)    # Red for rejected features
COLOR_BALL_ROI = (255, 255, 0)      # Cyan for ball ROI boundary
COLOR_FLOW_VECTORS = (255, 0, 255)  # Magenta for optical flow vectors

# ============================================
# DEBUG SETTINGS
# ============================================

VERBOSE = True                      # Print progress messages
DEBUG_MODE = True                   # Enable debug output and visualizations
SAVE_DEBUG_IMAGES = True            # Save intermediate debug images
PROGRESS_UPDATE_INTERVAL = 10       # Print progress every N frames

# Processing Mode
FULL_VIDEO_MODE = True              # Process all frames (True) vs test frames only (False)

# Test frames for Stage B validation (used only when FULL_VIDEO_MODE = False)
TEST_FRAMES = [65, 70, 75, 80]      # Use early frames where ball is larger and better detected
