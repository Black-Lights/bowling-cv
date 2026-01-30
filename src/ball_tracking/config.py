"""
Configuration file for bowling ball tracking - Phase 2

Edit parameters here without touching the main code.

Version: 2.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: January 30, 2026
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

# ============================================
# BALL DETECTION PARAMETERS
# ============================================

# Motion detection method
USE_FRAME_DIFFERENCING = True       # Use frame differencing (recommended for static camera)
FRAME_DIFF_THRESHOLD = 25           # Threshold for frame differencing (less sensitive to reduce noise)
MIN_MOTION_AREA = 200               # Minimum area to consider as motion

# Color-based detection (HSV color space)
USE_COLOR_DETECTION = True          # Enable color-based detection
# Purple/Blue bowling ball detection (more targeted than wide range)
BALL_COLOR_LOWER = (90, 30, 30)     # Purple/blue hue range, some saturation, not too bright
BALL_COLOR_UPPER = (150, 255, 180)  # Purple/blue spectrum with reasonable value limit

# Alternative color presets (uncomment to use):
# Purple/Blue ball: BALL_COLOR_LOWER = (110, 40, 60), BALL_COLOR_UPPER = (160, 255, 255)
# Black ball: BALL_COLOR_LOWER = (0, 0, 0), BALL_COLOR_UPPER = (180, 30, 80)
# Red ball: BALL_COLOR_LOWER = (0, 100, 100), BALL_COLOR_UPPER = (10, 255, 255)

# Morphological operations
MORPH_KERNEL_SIZE = 3               # Kernel size for morphological operations (small to preserve ball)
MORPH_OPEN_ITERATIONS = 2           # Opening iterations (remove noise)
MORPH_CLOSE_ITERATIONS = 1          # Closing iterations (minimal - avoid merging separate blobs)

# Size constraints (pixels)
MIN_BALL_RADIUS = 8                 # Minimum ball radius
MAX_BALL_RADIUS = 60                # Maximum ball radius (accounting for motion blur)
MIN_BALL_AREA = 200                 # Minimum contour area
MAX_BALL_AREA = 12000               # Maximum contour area (realistic for bowling ball)

# Shape filtering
MIN_CIRCULARITY = 0.30              # Minimum circularity (very relaxed for motion blur)
MIN_SOLIDITY = 0.50                 # Minimum solidity (relaxed for motion blur)

# Region of Interest
USE_LANE_MASK = True                # Use lane boundaries from Phase 1
ROI_MARGIN = 50                     # Margin around lane boundaries (increased for safety)

# ============================================
# BALL TRACKING PARAMETERS
# ============================================

# Kalman filter parameters
KALMAN_PROCESS_NOISE = 0.03         # Process noise covariance
KALMAN_MEASUREMENT_NOISE = 1.0      # Measurement noise covariance

# Tracking thresholds
MIN_CONFIDENCE_TO_START = 0.55      # Minimum confidence to start new track (lowered for initial detection)
MAX_MISSING_FRAMES = 25             # Max consecutive frames without detection before lost (INCREASED)
MAX_DISPLACEMENT = 200              # Max pixel displacement between frames (increased for fast balls)
MIN_TRACK_LENGTH = 10               # Minimum frames for valid track (reduced)

# Trajectory smoothing
SMOOTH_TRAJECTORY = True            # Apply smoothing to trajectory
SMOOTHING_WINDOW = 5                # Moving average window size

# Multi-track handling
MAX_TRACKS = 3                      # Maximum simultaneous tracks
MIN_SEPARATION_DISTANCE = 50        # Minimum distance between different balls

# ============================================
# ANALYSIS PARAMETERS
# ============================================

# Velocity calculation
CALCULATE_VELOCITY = True           # Calculate ball velocity
VELOCITY_SMOOTHING = True           # Smooth velocity data
VELOCITY_WINDOW = 3                 # Smoothing window for velocity

# Release point detection
DETECT_RELEASE_POINT = True         # Detect ball release point
RELEASE_VELOCITY_THRESHOLD = 3.0    # Minimum velocity to detect release (pixels/frame)
RELEASE_ZONE_Y = 0.70               # Release zone (70% from top of lane)

# Impact point detection
DETECT_IMPACT_POINT = True          # Detect ball impact at pins
IMPACT_ZONE_Y = 0.20                # Impact zone (20% from top of lane)
IMPACT_VELOCITY_THRESHOLD = 2.0     # Minimum velocity at impact

# ============================================
# VISUALIZATION OPTIONS
# ============================================

# Drawing options
DRAW_BALL_CENTER = True             # Draw ball center point
DRAW_BALL_CIRCLE = True             # Draw circle around ball
DRAW_TRAJECTORY = True              # Draw trajectory path
TRAJECTORY_LENGTH = 40              # Number of points in trajectory trail
DRAW_VELOCITY_VECTOR = True         # Draw velocity arrow
VELOCITY_VECTOR_SCALE = 3           # Scale factor for velocity arrow

# Color scheme (BGR)
COLOR_DETECTED = (0, 255, 0)        # Green for detected ball
COLOR_PREDICTED = (0, 255, 255)     # Yellow for predicted position
COLOR_TRAJECTORY = (255, 0, 255)    # Magenta for trajectory path
COLOR_RELEASE = (0, 0, 255)         # Red for release point
COLOR_IMPACT = (255, 0, 0)          # Blue for impact point

# ============================================
# OUTPUT OPTIONS
# ============================================

# Video output
SAVE_TRACKED_VIDEO = True           # Save video with tracking visualization
SAVE_DEBUG_VIDEO = False            # Save debug video with detection steps

# Data output
SAVE_TRAJECTORY_DATA = True         # Save trajectory coordinates to JSON
SAVE_TRAJECTORY_PLOT = True         # Save trajectory plot
SAVE_VELOCITY_PLOT = True           # Save velocity over time plot
SAVE_ANALYSIS_REPORT = True         # Save text analysis report

# Frame output
SAVE_DETECTION_FRAMES = False       # Save individual detection frames
DEBUG_FRAME_INTERVAL = 10           # Save every Nth frame

# ============================================
# DEBUG OPTIONS
# ============================================

DEBUG_MODE = True                   # Enable debug visualizations (ENABLED FOR TUNING)
PRINT_DETECTION_INFO = True         # Print detection info per frame (ENABLED)
SAVE_DEBUG_FRAMES = True            # Save debug frames showing detection steps (ENABLED)
SAVE_INTERMEDIATE_VIDEOS = True     # Save intermediate detection videos (ENABLED FOR ANALYSIS)
