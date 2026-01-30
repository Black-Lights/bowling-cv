"""
Configuration file for bowling lane detection - Phase 1
Edit parameters here without touching the main code

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Last Updated: January 30, 2026
"""

import os

# ============================================
# PROJECT PATHS
# ============================================

# Base paths (relative to project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets', 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# List of videos to process (will be loaded from assets/input/)
VIDEO_FILES = [
    'cropped_test3.mp4',
    'cropped_test6.mp4',
    'cropped_test7.mp4'
]

# Output configuration
SAVE_MASKED_VIDEO = False  # Save masked video (will be deleted after processing if False)
SAVE_PREPROCESSED_VIDEO = False  # Save preprocessed video (will be deleted after processing if False)
SAVE_SOBEL_VIDEO = False  # Save Sobel filter visualization video
SAVE_TOP_MASKED_VIDEO = False  # Save top boundary masked video
SAVE_COLLECTION_VIDEO = False  # Save collection video for all frames
SAVE_INTERMEDIATE_VIDEOS = True  # Enable legacy intermediate video debug system (see INTERMEDIATE_MODES below)

# Frame caching for faster iteration (saves ~4 mins per video during development)
SAVE_PREPROCESSED_FRAMES = True  # Save preprocessed frames as .npz file for reuse in later phases
SAVE_MASKED_FRAMES = True  # Save masked frames as .npz file for reuse in later phases
# When True: frames are saved to output/video_name/preprocessed_frames.npz (or masked_frames.npz)
# On next run: if file exists, loads from cache instead of reprocessing (huge speedup!)
# When False: always reprocess frames, don't save cache


# ============================================
# PHASE 1: DETECTION PARAMETERS
# ============================================

# --- Bottom & Side Boundaries (Foul Line & Master Lines) ---

# Number of frames to collect lines from (first N frames)
NUM_COLLECTION_FRAMES = 100

# Bin width for voting system (pixels)
BIN_WIDTH = 10

# Vote threshold (lines in bin must be > threshold * max_votes to be valid)
VOTE_THRESHOLD = 0.15  # 15%

# Angle tolerance for filtering lines (degrees)
ANGLE_TOLERANCE = 3

# Angle mode: 'from_vertical' or 'from_horizontal'
USE_ABSOLUTE_ANGLES = True  # True = from_vertical (recommended)
ANGLE_OFFSET = 0  # degrees


# --- Top Boundary (Pin Area) ---

# Use full video or limited frames for top detection
TOP_DETECTION_USE_FULL_VIDEO = True
TOP_DETECTION_FRAMES = 100  # Only used if USE_FULL_VIDEO = False

# Top scan region (avoid scoreboard at very top)
TOP_SCAN_REGION_START = 0.10  # Start at 10% from top
TOP_SCAN_REGION_END = 0.35    # End at 35% from top

# HSV preprocessing - Gap filling thresholds
MAX_PATCH_SIZE_ROW = 100  # Fill horizontal gaps ≤ 100px
MAX_PATCH_SIZE_COL = 50   # Fill vertical gaps ≤ 50px

# HSV preprocessing - Small patch removal from top (keeps only large pin deck area)
TOP_REGION_RATIO = 0.3        # Top 30% of image to scan for small patches
MAX_TOP_PATCH_AREA = 5000     # Remove patches < 5000 pixels from top (keeps only large pin deck)

# Sobel edge detection parameters
SOBEL_KERNEL_SIZE = 5      # Kernel size (must be odd: 1, 3, 5, 7)
SOBEL_THRESHOLD = 10       # Minimum edge strength
CENTER_REGION_START = 0.25 # Center region start (25% from left)
CENTER_REGION_END = 0.75   # Center region end (75% from left)
TOP_CANDIDATES_RATIO = 0.20  # Top 20% strongest rows

# MSAC (M-estimator SAmple Consensus) parameters
MSAC_RESIDUAL_THRESHOLD = 5.0  # Max distance for inliers (pixels)
MSAC_MAX_TRIALS = 1000         # Number of iterations
MSAC_MIN_SAMPLES = 2           # Minimum points to fit line
MSAC_RANDOM_STATE = 42         # For reproducibility



# ============================================
# FILE MANAGEMENT - WHAT TO SAVE/DELETE
# ============================================

# --- Analysis Plots (Always generated) ---

# Bin analysis plots (voting distribution)
SAVE_BIN_ANALYSIS_PLOTS = True

# Tracking analysis plots (stability over time)
SAVE_TRACKING_PLOTS = True

# MSAC fitting analysis plots (inliers/outliers/residuals)
SAVE_MSAC_PLOTS = True

# Top line intersection plots (Y coordinates)
SAVE_INTERSECTION_PLOTS = True

# Summary comparison plot (all videos)
SAVE_SUMMARY_PLOT = True


# --- Final Outputs (Always Saved) ---

# Final video with all 4 boundaries (ALWAYS SAVED)
# Located at: output/<video_name>/final_all_boundaries_<video_name>.mp4

# Boundary data JSON (ALWAYS SAVED)
# Located at: output/<video_name>/boundary_data.json

# Master line video with bottom/left/right (ALWAYS SAVED)
# Located at: output/<video_name>/master_final_<video_name>.mp4


# ============================================
# VISUALIZATION OPTIONS
# ============================================

# Main visualization mode for master line video
# Options: 'final', 'master_lines_only', 'with_stats'
VISUALIZATION_MODE = 'final'

# Intermediate folder for temporary files
INTERMEDIATE_FOLDER = 'intermediate'  # Subfolder name in output directory

# Legacy intermediate visualization system (for debugging bottom/side detection)
# Set to True to generate intermediate videos for debugging
# Available modes: 'edges_horizontal', 'edges_vertical', 'gaussian_vertical', 
#                  'grayscale_vertical', 'otsu_vertical', 'mask_vertical', etc.
INTERMEDIATE_MODES = [
    'edges_horizontal',
    'edges_vertical',
    'gaussian_vertical',
    'grayscale_vertical',
    'otsu_vertical',
    'mask_vertical',
]

# ============================================
# DEBUG OPTIONS
# ============================================

# Print detailed debug information
DEBUG_MODE = False

# Save intermediate frames for debugging
SAVE_DEBUG_FRAMES = False
DEBUG_FRAME_INTERVAL = 10  # Save every Nth frame
