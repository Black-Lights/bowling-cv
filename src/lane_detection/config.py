"""
Configuration file for bowling lane detection
Edit parameters here without touching the main code
"""

# ============================================
# VIDEO CONFIGURATION
# ============================================

import os

# Base paths (relative to project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets', 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# List of videos to process (will be loaded from assets/input/)
VIDEO_FILES = [
    'cropped_test3.mp4',  # Test video for initial setup
    # Add more videos as needed:
    # 'cropped_test2.mp4',
    # 'cropped_test4.mp4',
    # 'cropped_test5.mp4',
    # 'cropped_test6.mp4',
    # 'cropped_test7.mp4'
]

# ============================================
# PHASE 1: LINE COLLECTION
# ============================================

# Number of frames to collect lines from (first N frames)
NUM_COLLECTION_FRAMES = 100

# ============================================
# PHASE 2: MASTER LINE COMPUTATION
# ============================================

# Bin width for voting system (pixels)
BIN_WIDTH = 10

# Vote threshold (lines in bin must be > threshold * max_votes to be valid)
VOTE_THRESHOLD = 0.15  # 15%

# Angle tolerance for filtering lines (degrees)
ANGLE_TOLERANCE = 3

# ============================================
# VISUALIZATION OPTIONS
# ============================================

# Main visualization mode for final video
# Options: 'final', 'master_lines_only', 'with_stats'
VISUALIZATION_MODE = 'final'

# Save bin analysis plots (PNG files showing voting distribution)
SAVE_BIN_ANALYSIS = True

# Save collection phase video (first 100 frames with approach1)
SAVE_COLLECTION_VIDEO = False

# ============================================
# INTERMEDIATE VISUALIZATION MODES
# ============================================
# Create videos showing intermediate processing steps
# Set to True to generate intermediate videos for debugging

SAVE_INTERMEDIATE_VIDEOS = False

# List of intermediate modes to generate (only if SAVE_INTERMEDIATE_VIDEOS = True)
# Available modes:
#   'edges_horizontal' - Canny edges for horizontal line detection
#   'edges_vertical' - Canny edges for vertical line detection
#   'gaussian_horizontal' - Gaussian blur for horizontal line detection
#   'gaussian_vertical' - Gaussian blur for vertical line detection
#   'grayscale_horizontal' - Grayscale for horizontal line detection
#   'grayscale_vertical' - Grayscale for vertical line detection
#   'otsu_horizontal' - Otsu threshold for horizontal line detection
#   'otsu_vertical' - Otsu threshold for vertical line detection
#   'contours_vertical' - Contour detection for vertical line detection
#   'mask_vertical' - Mask after morphology for vertical line detection
#   'dilated_vertical' - Dilated mask for vertical line detection
#   'eroded_vertical' - Eroded mask for vertical line detection

INTERMEDIATE_MODES = [
    'edges_horizontal',
    'edges_vertical',
    'gaussian_vertical',
    'grayscale_vertical',
    'otsu_vertical',
    'mask_vertical',
]

# ============================================
# TRACKING ANALYSIS OPTIONS
# ============================================

# Generate tracking analysis plots
GENERATE_TRACKING_PLOTS = True

# Create summary comparison across all videos
CREATE_SUMMARY_PLOT = True

# ============================================
# SLOPE/ANGLE CORRECTION
# ============================================

# Angle offset to correct slope calculation
# If angles seem inverted, try adjusting this
ANGLE_OFFSET = 0  # degrees

# Use absolute angle values (convert negative to positive relative to vertical)
# FALSE = angles from horizontal (-90° to 90°, confusing for vertical lines)
# TRUE = angles from vertical (0° = vertical, 90° = horizontal, more intuitive)
USE_ABSOLUTE_ANGLES = True  # Recommended for vertical lane boundaries

# ============================================
# DEBUG OPTIONS
# ============================================

# Print detailed debug information
DEBUG_MODE = False

# Save intermediate frames for debugging
SAVE_DEBUG_FRAMES = False
DEBUG_FRAME_INTERVAL = 10  # Save every Nth frame
