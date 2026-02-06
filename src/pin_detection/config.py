"""
Pin Detection Configuration

Configuration parameters for Phase 4: Pin Detection
Includes masking, frame selection, contour detection, and visualization settings.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets', 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Video configuration
VIDEO_NAMES = ['cropped_test3']  # Add more videos as needed

# ============================================================================
# MASKING PARAMETERS
# ============================================================================

# Extended masking for pin area (above top boundary)
PIN_AREA_UNMASK_EXTENSION = 20  # Pixels to extend left/right boundaries OUTWARD

# This reveals our pins fully while preventing adjacent lane pins from being visible
# Adjust this value based on your camera angle and lane width

# ============================================================================
# FRAME SELECTION PARAMETERS
# ============================================================================

# Frame offsets for before/after comparison
BEFORE_FRAME_OFFSET = 15        # Frame index for "before" state (all pins standing)
AFTER_FRAME_OFFSET = -15        # Frame offset from end for "after" state (pins settled)

# Alternative: Use trajectory data to determine frames
USE_TRAJECTORY_FOR_TIMING = True  # If True, use ball trajectory to find impact frame
SETTLE_TIME_FRAMES = 45           # Frames to wait after impact for pins to settle (1.5s @ 30fps)

# ROI focus (analyze only top portion of frame where pins are)
FOCUS_TOP_FRACTION = 0.5        # Analyze top 50% of frame only (0.0 = none, 1.0 = all)

# ============================================================================
# FRAME DIFFERENCING PARAMETERS
# ============================================================================

# Difference thresholding
DIFFERENCE_THRESHOLD = 30       # Binary threshold for frame difference (0-255)
                                # Lower = more sensitive, Higher = less sensitive

# Pin detection threshold (for direct AFTER frame analysis)
PIN_DETECTION_THRESHOLD = 120   # Threshold for isolating white pins (0-255)
                                # Higher = only very white pins, Lower = includes grayer areas
                                # 100-120: Picks up lane surface and pins
                                # 150-180: Only bright white pins

# Morphological operations
MORPH_KERNEL_SIZE = 5          # Kernel size for morphological operations (odd number)
MORPH_ITERATIONS = 2           # Number of iterations for morphology operations

# ============================================================================
# CONTOUR DETECTION PARAMETERS
# ============================================================================

# Pin contour area constraints (in pixels²)
MIN_PIN_AREA = 150             # Minimum contour area to be considered a pin
MAX_PIN_AREA = 8000            # Maximum contour area to be considered a pin

# Pin aspect ratio constraints (width/height)
MIN_PIN_ASPECT_RATIO = 0.2     # Minimum aspect ratio (tall and thin)
MAX_PIN_ASPECT_RATIO = 1.5     # Maximum aspect ratio (not too wide)

# Circularity/Solidity constraints
MIN_PIN_SOLIDITY = 0.5         # Minimum solidity (contour_area / convex_hull_area)
                                # Helps filter out irregular shapes

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Intermediate visualization flags (set to False to speed up processing)
SAVE_INTERMEDIATE_FRAMES = True          # Save individual frames at each step
CREATE_INTERMEDIATE_VIDEOS = True        # Create videos showing processing steps
SAVE_DEBUG_PLOTS = True                  # Save matplotlib plots for analysis

# Specific visualization flags
VISUALIZE_EXTENDED_MASK = True           # Show the extended masking result
VISUALIZE_FRAME_SELECTION = True         # Show before/after frame selection
VISUALIZE_DIFFERENCE = True              # Show frame differencing process
VISUALIZE_MORPHOLOGY = True              # Show morphological operations
VISUALIZE_CONTOURS = True                # Show detected contours
VISUALIZE_COMPARISON = True              # Show before/after comparison

# Video encoding settings
VIDEO_CODEC = 'mp4v'                     # Codec for intermediate videos
VIDEO_FPS = 30                           # Frame rate for output videos

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Output subdirectory structure
PIN_DETECTION_SUBDIR = 'pin_detection'
INTERMEDIATE_SUBDIR = 'intermediate'

# Output file naming
EXTENDED_MASK_VIDEO_SUFFIX = '_pin_area_masked'
BEFORE_FRAME_SUFFIX = '_before_frame'
AFTER_FRAME_SUFFIX = '_after_frame'
DIFFERENCE_SUFFIX = '_difference'
BINARY_DIFF_SUFFIX = '_binary_diff'
MORPHOLOGY_SUFFIX = '_morphology'
CONTOURS_SUFFIX = '_contours'
RESULT_SUFFIX = '_pin_detection_result'
COMPARISON_SUFFIX = '_comparison'

# Data export
EXPORT_JSON = True                       # Export results to JSON
EXPORT_CSV = False                       # Export contour data to CSV

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================

# Total number of pins in regulation bowling
TOTAL_PINS = 10

# Confidence thresholds
MIN_CONFIDENCE = 0.6                     # Minimum confidence to report result

# Edge case handling
ALLOW_OVER_COUNT = False                 # If True, allow remaining_pins > 10
ALLOW_UNDER_COUNT = False                # If True, allow remaining_pins < 0

# ============================================================================
# DEBUG AND LOGGING
# ============================================================================

# Verbosity
VERBOSE = True                           # Print detailed progress messages
DEBUG_MODE = False                       # Enable extra debugging output

# Console output formatting
SHOW_PROGRESS_BAR = True                 # Show tqdm progress bars
PRINT_TIMING = True                      # Print execution time for each step

# ============================================================================
# COLOR DEFINITIONS (BGR format for OpenCV)
# ============================================================================

# Colors for visualization
COLOR_PIN_STANDING = (0, 255, 0)        # Green
COLOR_PIN_TOPPLED = (0, 0, 255)         # Red
COLOR_CONTOUR = (255, 0, 255)           # Magenta
COLOR_BOUNDING_BOX = (0, 255, 255)      # Cyan
COLOR_TEXT = (255, 255, 255)            # White
COLOR_BACKGROUND = (0, 0, 0)            # Black

# Boundary colors (matching Phase 1)
COLOR_TOP_BOUNDARY = (0, 255, 0)        # Green
COLOR_FOUL_LINE = (0, 0, 255)           # Red
COLOR_LEFT_BOUNDARY = (255, 0, 0)       # Blue
COLOR_RIGHT_BOUNDARY = (255, 0, 0)      # Blue
COLOR_EXTENDED_AREA = (255, 255, 0)     # Yellow (for visualization)

# ============================================================================
# FONT SETTINGS
# ============================================================================

FONT_FACE = 0                            # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
FONT_LINE_TYPE = 2                       # cv2.LINE_AA

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_video_input_path(video_name):
    """Get the full path to input video file."""
    return os.path.join(ASSETS_DIR, f'{video_name}.mp4')


def get_video_output_dir(video_name):
    """Get the output directory for a specific video."""
    return os.path.join(OUTPUT_DIR, video_name)


def get_pin_detection_output_dir(video_name):
    """Get the pin detection output directory for a specific video."""
    return os.path.join(get_video_output_dir(video_name), PIN_DETECTION_SUBDIR)


def get_intermediate_output_dir(video_name):
    """Get the intermediate output directory for a specific video."""
    return os.path.join(get_pin_detection_output_dir(video_name), INTERMEDIATE_SUBDIR)


def get_boundary_data_path(video_name):
    """Get the path to boundary data JSON from Phase 1."""
    return os.path.join(get_video_output_dir(video_name), 'boundary_data.json')


def get_trajectory_data_path(video_name):
    """Get the path to trajectory data JSON from Phase 2."""
    return os.path.join(
        get_video_output_dir(video_name), 
        'ball_detection', 
        f'{video_name}_trajectory_data.json'
    )


def print_config_summary():
    """Print configuration summary for debugging."""
    print("\n" + "="*80)
    print("PIN DETECTION CONFIGURATION SUMMARY")
    print("="*80)
    print(f"\n📁 Paths:")
    print(f"   Base Directory:    {BASE_DIR}")
    print(f"   Assets Directory:  {ASSETS_DIR}")
    print(f"   Output Directory:  {OUTPUT_DIR}")
    print(f"\n🎯 Detection Parameters:")
    print(f"   Unmask Extension:  {PIN_AREA_UNMASK_EXTENSION} px")
    print(f"   Before Frame:      {BEFORE_FRAME_OFFSET}")
    print(f"   After Frame:       {AFTER_FRAME_OFFSET}")
    print(f"   Focus Top:         {FOCUS_TOP_FRACTION * 100:.0f}%")
    print(f"\n🔍 Contour Parameters:")
    print(f"   Min Pin Area:      {MIN_PIN_AREA} px²")
    print(f"   Max Pin Area:      {MAX_PIN_AREA} px²")
    print(f"   Aspect Ratio:      {MIN_PIN_ASPECT_RATIO:.1f} - {MAX_PIN_ASPECT_RATIO:.1f}")
    print(f"\n📊 Visualization:")
    print(f"   Intermediate:      {SAVE_INTERMEDIATE_FRAMES}")
    print(f"   Videos:            {CREATE_INTERMEDIATE_VIDEOS}")
    print(f"   Debug Plots:       {SAVE_DEBUG_PLOTS}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test configuration
    print_config_summary()
    
    # Test path generation
    test_video = 'cropped_test3'
    print(f"\n🧪 Testing path generation for '{test_video}':")
    print(f"   Input Video:       {get_video_input_path(test_video)}")
    print(f"   Output Dir:        {get_pin_detection_output_dir(test_video)}")
    print(f"   Boundary Data:     {get_boundary_data_path(test_video)}")
    print(f"   Trajectory Data:   {get_trajectory_data_path(test_video)}")
