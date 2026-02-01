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
# DEBUG & LOGGING
# ============================================

DEBUG_MODE = False
VERBOSE = True
