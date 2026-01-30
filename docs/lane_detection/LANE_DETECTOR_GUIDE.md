# Phase 1 Complete Pipeline - LaneDetector Class

## Overview

The LaneDetector class provides a complete, professional pipeline for detecting all 4 boundaries of a bowling lane:

- **Bottom Boundary**: Foul line (horizontal)
- **Left/Right Boundaries**: Master lines (vertical)  
- **Top Boundary**: Pin area (horizontal, using MSAC line fitting)
- **Intersection Points**: All 4 points where horizontal lines cross vertical master lines

## Quick Start

### Option 1: Process All Configured Videos

```bash
python main.py
```

### Option 2: Process Single Video

```bash
python main.py --video cropped_test3.mp4
```

### Option 3: Use in Your Own Code

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lane_detection import LaneDetector
from lane_detection import config

# Initialize detector
detector = LaneDetector('cropped_test3.mp4', config)

# Run complete pipeline
boundaries, intersections = detector.detect_all()

# Save results
detector.save()

# Access results
print(boundaries)
print(intersections)
```

## What Gets Saved

### Always Saved (Final Outputs)

1. **boundary_data.json** - All 4 boundaries + intersection points
2. **final_all_boundaries_*.mp4** - Visualization with all detected lines
3. **final_top_boundary_*.mp4** - Focused view of top boundary detection
4. **bin_analysis_*.png** - Master line voting visualizations (if `SAVE_BIN_ANALYSIS_PLOTS=True`)
5. **tracking_plot_*.png** - Master line tracking stability (if `SAVE_TRACKING_PLOTS=True`)

### Intermediate Files (Configurable in src/lane_detection/config.py)

Control what gets saved via config flags:

```python
# In src/lane_detection/config.py:
SAVE_MASKED_VIDEO = False          # Delete after use
SAVE_PREPROCESSED_VIDEO = False    # Delete after use
SAVE_SOBEL_VIDEO = False           # Delete after use
SAVE_TOP_MASKED_VIDEO = False      # Delete after use
SAVE_COLLECTION_VIDEO = False      # Delete after use

# Plots (usually keep these)
SAVE_BIN_ANALYSIS_PLOTS = True
SAVE_TRACKING_PLOTS = True
SAVE_MSAC_PLOTS = True
SAVE_INTERSECTION_PLOTS = True
```

When `SAVE_X = False`, the file is created, used, and automatically deleted after processing.

## Architecture

### Automatic Dependency Resolution

Methods automatically run prerequisites:

```python
detector = LaneDetector('video.mp4', config)

# Option A: Run everything at once
detector.detect_all()

# Option B: Run steps individually (dependencies auto-run)
detector.detect_top_boundary()  # Auto-runs bottom + sides if needed
detector.calculate_intersections()  # Auto-runs all boundaries if needed
```

### Detection Pipeline

1. **detect_bottom_boundary()**
   - Detects foul line from first frame
   - Sets `boundaries['bottom']`

2. **detect_side_boundaries()**  
   - Requires: Bottom boundary (auto-runs if missing)
   - Collects lines from first N frames
   - Votes on master lines using bin analysis
   - Sets `boundaries['left']` and `boundaries['right']`

3. **detect_top_boundary()**
   - Requires: Bottom + side boundaries (auto-runs if missing)
   - Creates masked video (lane area only)
   - Preprocesses with HSV + gap filling
   - Detects edges with Sobel filter
   - Fits MSAC line from all detections
   - Generates visualization videos
   - Sets `boundaries['top']`

4. **calculate_intersections()**
   - Requires: All boundaries (auto-runs detect_all() if missing)
   - Calculates all 4 intersection points:
     * Top-Left: Top boundary ∩ Left master
     * Top-Right: Top boundary ∩ Right master
     * Bottom-Left: Foul line ∩ Left master
     * Bottom-Right: Foul line ∩ Right master

5. **save()**
   - Saves boundary_data.json
   - Saves tracking plots (if configured)
   - Cleans up temporary files

## Output Structure

```
output/
  cropped_test3/
    boundary_data.json              # All boundaries + intersections
    bin_analysis_left.png           # Left master line voting
    bin_analysis_right.png          # Right master line voting
    tracking_plot_*.png             # Master line stability analysis
    final_all_boundaries_*.mp4      # Complete visualization
    final_top_boundary_*.mp4        # Top boundary focused view
    intersection_analysis_*.png     # MSAC intersection plot
    
    intermediate/                   # Temp files (auto-deleted if configured)
      masked_*.mp4                  # (deleted if SAVE_MASKED_VIDEO=False)
      preprocessed_*.mp4            # (deleted if SAVE_PREPROCESSED_VIDEO=False)
```

## Configuration

Key parameters in `src/lane_detection/config.py`:

```python
# Detection Parameters
NUM_COLLECTION_FRAMES = 100         # Frames to collect for master lines
BIN_WIDTH = 4                       # Bin width for voting
VOTE_THRESHOLD = 0.7                # Vote threshold for consensus
MSAC_RESIDUAL_THRESHOLD = 5.0       # MSAC outlier threshold
MSAC_MAX_TRIALS = 1000              # MSAC iterations

# File Management
INTERMEDIATE_FOLDER = 'intermediate'  # Where temp files go
SAVE_MASKED_VIDEO = False             # Auto-delete after use
SAVE_PREPROCESSED_VIDEO = False       # Auto-delete after use
SAVE_SOBEL_VIDEO = False              # Auto-delete after use
SAVE_COLLECTION_VIDEO = False         # Auto-delete after use

# Visualization
SAVE_BIN_ANALYSIS_PLOTS = True        # Keep voting visualizations
SAVE_TRACKING_PLOTS = True            # Keep stability analysis
SAVE_MSAC_PLOTS = True                # Keep MSAC plots
SAVE_INTERSECTION_PLOTS = True        # Keep intersection plots
```

## Example Output

After running `python main.py`, you'll see:

```
################################################################################
# BOWLING LANE DETECTION PIPELINE v1.0.0
# Processing 3 video(s)
################################################################################

======================================================================
LaneDetector v1.0.0 initialized
Video: cropped_test3
Output: output/cropped_test3
======================================================================

======================================================================
STEP 1: Detecting Bottom Boundary (Foul Line)
======================================================================
✅ Bottom boundary detected at Y=1050

======================================================================
STEP 2: Detecting Side Boundaries (Master Lines)
======================================================================
  Collecting lines: 100%|████████████| 100/100 [00:05<00:00]
  Collected 1234 left lines, 1189 right lines
✅ Side boundaries detected:
   Left: X=215, Angle=88.5°
   Right: X=1705, Angle=91.2°

======================================================================
STEP 3: Detecting Top Boundary (Pin Area)
======================================================================
  Creating masked video (lane area only)...
  Preprocessing with HSV filtering + gap filling...
  Detecting top boundary using Sobel edge detection...
  Generating visualization videos...
✅ Top boundary detected:
   MSAC line: Y=245.3
   Inliers: 427/450

======================================================================
STEP 4: Calculating Intersection Points
======================================================================
  Intersection Points:
    Top-Left:     (213, 246)
    Top-Right:    (1707, 245)
    Bottom-Left:  (974, 1050)
    Bottom-Right: (976, 1050)

======================================================================
📊 SUMMARY: cropped_test3
======================================================================
  Boundaries Detected:
    ✅ Bottom (Foul):  Y=1050
    ✅ Left Master:    X=215, θ=88.5°
    ✅ Right Master:   X=1705, θ=91.2°
    ✅ Top (Pin Area): Y=245.3
  
  Intersection Points:
    • Top-Left:     {'x': 213, 'y': 246, 'description': ...}
    • Top-Right:    {'x': 1707, 'y': 245, 'description': ...}
    • Bottom-Left:  {'x': 974, 'y': 1050, 'description': ...}
    • Bottom-Right: {'x': 976, 'y': 1050, 'description': ...}
  
  Output: output/cropped_test3
======================================================================
```

## Version Information

- **Version**: 1.0.0
- **Authors**: Mohammad Umayr Romshoo, Mohammad Ammar Mughees  
- **Date**: January 30, 2026
- **Status**: Phase 1 Complete - All 4 boundaries detected

## Next Steps (Phase 2)

The LaneDetector class is designed to be extended for Phase 2:

```python
# Future Phase 2 usage
from lane_detection import LaneDetector

detector = LaneDetector('video.mp4', config)
boundaries, intersections = detector.detect_all()

# Phase 2: Ball tracking (future)
# tracker = BallTracker(detector.boundaries, config)
# ball_positions = tracker.track_ball()
```

All boundary data is saved in a clean, JSON-serializable format ready for Phase 2 integration.
