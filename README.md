# Bowling Analysis Project

[![Status](https://img.shields.io/badge/status-Phase%201%20Complete-green)](https://github.com/Black-Lights/bowling-cv)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-Academic-lightgrey)](LICENSE)

Computer vision system for analyzing bowling ball trajectory, spin/rotation axis, and toppled pins from video recordings using OpenCV and Python.

## Team Members

- **Mohmmad Umayr Romshoo**
- **Mohammad Ammar Mughees** ([Black-Lights](https://github.com/Black-Lights))

## Project Status

**Phase 1: Lane Detection** - ✅ **COMPLETE** (All 4 Boundaries Detected)  
**Phase 2: Ball Detection** - ✅ **COMPLETE** (Stages B+C+D+E+F+G Integrated)  
**Phase 3: 3D Trajectory Reconstruction** - Planned  
**Phase 4: Spin/Rotation Analysis** - Planned  
**Phase 5: Pin Detection** - Planned

## Current Progress

Below is a demonstration of the complete lane detection system working on bowling videos. The system successfully detects **all 4 boundaries**:
- **Top boundary** (green horizontal line - pin area)
- **Foul line** (red horizontal line at the bottom)
- **Left lane boundary** (blue vertical line)
- **Right lane boundary** (blue vertical line)

<!-- TODO: Upload video by editing this file on GitHub web interface -->
<!-- Drag and drop the video file from output/cropped_test3/master_final_cropped_test3.mp4 -->
<!-- GitHub will generate a URL like: https://github.com/user-attachments/assets/... -->
<!-- Paste the generated URL below to replace this comment -->

**Demo Video**: [Download master_final_cropped_test3.mp4](output/cropped_test3/master_final_cropped_test3.mp4)


https://github.com/user-attachments/assets/e9d9bca5-96c8-45bb-a854-e1c39bdd7f19


---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenCV 4.0+
- NumPy, SciPy
- Pandas, Matplotlib
- tqdm (for progress bars)
- scikit-learn (for MSAC/RANSAC fitting)

### Installation

```bash
# Clone the repository
git clone https://github.com/Black-Lights/bowling-cv.git
cd bowling-cv

# Install required packages
pip install opencv-python numpy scipy pandas matplotlib tqdm scikit-learn
```

### Running Complete Lane Detection (All 4 Boundaries)

```bash
# Run the complete pipeline (uses LaneDetector class)
python main.py

# Or specify a video
python main.py --video cropped_test3.mp4
```

**Output:** Complete lane box with all 4 boundaries in `output/<video_name>/final_all_boundaries_*.mp4`

**New Features:**
- ✅ Frame caching for faster iteration (saves ~4 mins per video)
- ✅ Small patch removal from top region (cleaner Sobel detection)
- ✅ Professional class-based architecture (LaneDetector)
- ✅ Automatic dependency resolution

### Running Complete Ball Detection (Stages B-G Integrated)

```bash
# Run the complete Phase 2 pipeline (all 6 steps)
python -m src.ball_detection.main --video cropped_test3.mp4

# Process all configured videos
python -m src.ball_detection.main
```

**Output:** 
- 4 diagnostic videos (candidates, selection, trajectory, debug)
- Trajectory data JSON (original + overhead coordinates)
- Trajectory plots (original + overhead views)
- Processed & reconstructed trajectory CSVs (Stage G)
- Complete ball tracking from foul line to pins

**Pipeline Steps:**
1. ✅ Lane masking (4-side boundaries)
2. ✅ Perspective transformation (overhead view)
3. ✅ Motion detection (MOG2 background subtraction)
4. ✅ ROI tracking (legacy visualization)
5. ✅ Integrated tracking (Stages C+D+E+F: filter → select → track → stop)
6. ✅ Post-processing (Stage G: cleaning + reconstruction)

---

## Project Structure

```
bowling-cv/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── assets/                        # Input assets
│   └── input/                     # Video files for processing
│       └── *.mp4                  # Place your bowling videos here
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── lane_detection/            # Lane boundary detection module (Phase 1)
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration settings
│   │   ├── lane_detector.py       # LaneDetector class (main entry)
│   │   ├── main_legacy.py         # Legacy entry point
│   │   ├── detection_functions.py # Line detection algorithms
│   │   ├── detection_utils.py     # Utility functions
│   │   ├── master_line_computation.py # Master line voting system
│   │   ├── top_boundary_detection.py  # Top boundary with MSAC
│   │   ├── mask_lane_area.py      # Lane masking utilities (shared)
│   │   ├── preprocess_frames.py   # HSV filtering + gap filling
│   │   ├── intermediate_visualization.py # Debug visualizations
│   │   └── tracking_analysis.py   # Tracking stability analysis
│   │
│   └── ball_detection/            # Ball detection module (Phase 2)
│       ├── __init__.py
│       ├── config.py              # Ball detection configuration
│       ├── main.py                # Ball detection entry point
│       ├── mask_video.py          # Video masking for ball focus
│       ├── homography.py          # 2D homography calculation (DLT)
│       ├── transform_video.py     # Perspective transformation to overhead view
│       ├── motion_detection.py    # MOG2 background subtraction (Stage B)
│       ├── roi_logic.py           # Kalman filter tracking (Stages C+E)
│       ├── blob_analysis.py       # Geometric validation (Stage D)
│       ├── integrated_visualization.py # 4 diagnostic visualization videos
│       └── roi_visualization.py   # Legacy visualization (pre-integration)
│
├── output/                        # Generated outputs
│   └── <video_name>/
│       ├── boundary_data.json     # Saved boundary parameters
│       ├── masked_*.mp4           # Lane-masked video
│       ├── preprocessed_*.mp4     # HSV filtered video
│       ├── master_final_*.mp4     # Video with bottom/left/right
│       ├── final_all_boundaries_*.mp4 # All 4 boundaries (COMPLETE)
│       ├── top_vis_sobel_*.mp4    # Sobel edge visualization
│       ├── top_vis_masked_*.mp4   # Preprocessed with top line
│       ├── msac_fitting_*.png     # MSAC analysis plot
│       ├── bin_analysis_*.png     # Voting system visualization
│       ├── tracking_*.png         # Tracking stability plots
│       └── ball_detection/        # Ball detection outputs
│           ├── intermediate/      # Intermediate stage videos
│           │   ├── *_lane_masked.mp4          # 4-side masked video
│           │   ├── *_foreground_mask.mp4      # MOG2 raw output
│           │   ├── *_shadow_removed.mp4       # After shadow threshold
│           │   ├── *_denoised.mp4             # Final clean mask
│           │   └── *_motion_comparison.mp4    # 2×2 comparison grid
│           ├── *_transformed.mp4              # Overhead perspective view
│           ├── *_integrated_candidates.mp4    # All validated candidates + ROI
│           ├── *_integrated_selection.mp4     # Search strategy visualization
│           ├── *_integrated_trajectory.mp4    # Ball trajectory trail
│           └── *_integrated_debug.mp4         # Complete debug overlay
│
└── docs/                          # Documentation
    ├── ANGLE_GUIDE.md             # Angle calculation documentation
    ├── FIXES.md                   # Bug fixes and improvements
    ├── PERSPECTIVE_GUIDE.md       # Perspective correction guide
    └── WHATS_NEW.md               # Change log
```

---

## Features

### ✅ Implemented (Phase 1 - COMPLETE)
- **Complete Lane Boundary Detection**
  - ✅ Horizontal foul line detection (bottom boundary)
  - ✅ Vertical master lines (left & right boundaries)
  - ✅ Top boundary detection (pin area) with MSAC fitting
  - ✅ Professional class-based architecture (LaneDetector)
  - ✅ Frame caching system (saves ~4 mins per video)
  - ✅ Small patch removal from top region
  - ✅ 6 intermediate visualization modes

### ✅ Implemented (Phase 2 - COMPLETE)
- **Ball Detection & Tracking**
  - ✅ **Stage A: Video Preprocessing**
    - 4-side lane masking using Phase 1 boundaries
    - 2D homography calculation (DLT)
    - Perspective transformation to overhead view (20 px/in)
    - High-quality encoding (PNG + yuv444p, no artifacts)
  
  - ✅ **Stage B: Motion Detection**
    - MOG2 background subtraction (500 frame history)
    - Shadow removal (threshold at 200)
    - Shadow separation via erosion (2 iterations)
    - Morphological opening for noise removal
    - Intermediate visualization (4 videos)
  
  - ✅ **Stage D: Blob Analysis & Geometric Validation**
    - Connected component analysis
    - Circularity filter (≥ 0.70 for ball shapes)
    - Aspect ratio filter (≤ 1.8 to reject shadows)
    - Size-based filtering (perspective-aware)
    - Auto-calibration system for blob parameters
  
  - ✅ **Stage C+E: Tracking-by-Detection Architecture**
    - **Correct Implementation**: Filter ALL (Stage D) → Select (Stage C) → Track (Stage E)
    - Kalman filter for position prediction (4-state: x, y, vx, vy)
    - Dual search modes:
      - **Global Search Type 1 (Initial)**: Exclude upper 30%, select near foul line
      - **Global Search Type 2 (Reactivation)**: Search above last position (Y < last_y + margin)
        - ⚠️ Fixed: Now correctly searches toward pins (smaller Y values)
      - **Local Tracking**: ROI around Kalman prediction
    - Perspective-aware dynamic ROI: B = max(30px, 0.15 × y_ball)
    - Confirmation logic: 20 frames + 240px travel distance
    - Reactivation timeout: 60 frames before full frame reset
    - MAX_LOST_FRAMES = 2 (prevents Kalman drift)
  
  - ✅ **Integrated Visualization System**
    - 4 diagnostic videos showing complete tracking pipeline
    - Candidates view (all validated + selected + ROI)
    - Selection strategy (search zones, exclusion areas)
    - Trajectory view (ball path with fading trail)
    - Debug overlay (complete info panel)
  
  - ✅ **Stage F: Stop Condition & Trajectory Export**
    - **Configurable stop threshold** (2% above top boundary - allows tracking to pins)
      - NEW: Stops when Y ≤ top_boundary - 2% of frame_height
      - Example: top=130, height=954 → stops at Y≤111 (closer to pins)
    - **3-Boundary Masking Mode** for tracking (L+R+B only)
      - Top boundary NOT masked during tracking (allows complete trajectory)
      - Stage A (homography) still uses 4 boundaries for correct transformation
    - **5 Kalman Predictions** after stopping (NEW)
      - Replaced single linear interpolation with 5 sequential Kalman predictions
      - Simulates ball motion for 5 future frames
      - More accurate trajectory extrapolation using filter state propagation
    - Early tracking termination when ball reaches pin area
    - Saves ~35-45% processing time (tested: 141/254 frames on cropped_test3)
    - Trajectory data export for post-processing:
      - Original (perspective) coordinates with frame numbers
      - Transformed (overhead) coordinates via homography
      - Frame-accurate timing (frame_number field for spin analysis)
      - 5 interpolated endpoints in both coordinate systems (NEW)
      - JSON format with complete metadata
    - Enhanced visualizations:
      - Stop threshold line (magenta)
      - Interpolated trajectory with 5 predictions (dashed orange)
      - Trajectory plots on original and overhead views
  
  - ✅ **Stage G: Post-Processing (Trajectory Cleaning & Reconstruction)**
    - **Stage G1: Trajectory Processing** (Updated Feb 5, 2026)
      - **Rolling Median MAD outlier detection** (window=3) - local context-aware outlier removal using Euclidean distance
      - **Median filter** (kernel=3) - scipy.medfilt smoothing on valid points
      - **Savitzky-Golay smoothing** (window=45, poly=3) - applied BEFORE interpolation
      - **Linear interpolation** - fills gaps AFTER smoothing valid data points
      - **Improved approach**: Smooths real data first, then interpolates (vs old: interpolate then smooth)
    - **Stage G2: Template Reconstruction**
      - Boundary filtering (removes points outside valid lane area)
      - Coordinate scaling from homography space to template space
      - Resolution smoothing (window=31, poly=2) - removes pixelation artifacts
      - Float precision maintained throughout pipeline for smooth trajectories
    - **Output Files**:
      - `trajectory_processed_original.csv` - cleaned trajectory in original (perspective) coordinates
      - `trajectory_processed_overhead.csv` - cleaned trajectory in overhead (homography) coordinates
      - `trajectory_reconstructed.csv` - final trajectory on lane template
    - **Validation Visualizations** (10 PNG plots + 1 MP4):
      - `trajectory_processing_original.png` - Original coordinate cleaning plots (before/after)
      - `trajectory_processing_overhead.png` - Overhead coordinate cleaning plots (before/after)
      - **NEW**: `median_filter_original.png` - Median filter effects on original coordinates
      - **NEW**: `median_filter_overhead.png` - Median filter effects on overhead coordinates
      - **NEW**: `mad_outliers_original.png` - Outlier detection visualization for original coordinates
      - **NEW**: `mad_outliers_overhead.png` - Outlier detection visualization for overhead coordinates
      - **NEW**: `interpolation_original.png` - Interpolation visualization for original coordinates
      - **NEW**: `interpolation_overhead.png` - Interpolation visualization for overhead coordinates
      - `radius_processing_visualization.png` - Radius cleaning with RANSAC fit
      - `trajectory_on_template.png` - Final trajectory overlaid on bowling lane template
      - `trajectory_animation.mp4` - Animated video showing trajectory building frame-by-frame
    - Fully integrated into main pipeline as Step 6
    - Configurable via `--skip-postprocess` CLI flag and config.py parameters
    - All visualization flags individually controllable in config.py

### 🔄 In Progress (Phase 2 - Advanced Analysis)
- **Trajectory Analysis & Physics**
  - Velocity and acceleration curves over time
  - Path curvature analysis for hook detection
  - Impact angle calculations at pin deck
  - Ball speed measurements (mph/fps)
  - Comparative analysis across multiple throws
  - Statistical trajectory metrics

### Planned (Phase 3+)
- **3D Trajectory Reconstruction**
- **Spin/Rotation Analysis and Axis Detection**
- **Pin Detection and Topple Counting**
- **Strike/Spare Classification**
- **Comprehensive Visualization Dashboard**

---

## Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Computer Vision** | OpenCV, scikit-image |
| **Numerical Computing** | NumPy, SciPy |
| **Data Analysis** | Pandas |
| **Visualization** | Matplotlib, OpenCV |
| **Progress Tracking** | tqdm |

---

## Usage Guide

### Phase 1: Complete Lane Detection

```bash
# Run the complete pipeline (uses LaneDetector class)
python main.py

# Or specify a video
python main.py --video cropped_test3.mp4
```

**Output:** Complete lane box with all 4 boundaries in `output/<video_name>/final_all_boundaries_*.mp4`

### Phase 2: Ball Detection (Stages B+C+D+E+F+G Integrated)

```bash
# Run complete ball detection pipeline (all 6 steps)
python -m src.ball_detection.main --video cropped_test3.mp4

# Or skip specific steps if you already have preprocessed files
python -m src.ball_detection.main --video cropped_test3.mp4 --skip-masking --skip-transform --skip-motion --skip-roi

# Skip post-processing if you only need raw trajectory
python -m src.ball_detection.main --video cropped_test3.mp4 --skip-postprocess
```

**Integrated Tracking Outputs (4 diagnostic videos + trajectory data + post-processing):**
- `*_integrated_candidates.mp4` - Shows all validated candidates (cyan), selected candidate (yellow), ROI box (green in local mode), candidate counts
- `*_integrated_selection.mp4` - Search strategy visualization:
  - **GLOBAL (initial)**: Red exclusion zone (upper 30%), green search zone
  - **GLOBAL (reactivation)**: Green search line above last position, orange last position marker
  - **LOCAL**: Green ROI box, red Kalman prediction crosshair
- `*_integrated_trajectory.mp4` - Ball trajectory trail with fading effect, current position highlight, interpolated section (dashed)
- `*_integrated_debug.mp4` - Complete overlay with transparent info panel showing mode, candidates, detection status
- `*_trajectory_data.json` - **NEW**: Complete trajectory export for post-processing
  - Original (perspective) coordinates: {index, frame_number, x, y, interpolated}
  - Transformed (overhead) coordinates: {index, frame_number, x, y, interpolated}
  - `frame_number`: Actual video frame number (for timing/spin analysis)
  - `index`: Sequential trajectory point number (0, 1, 2...)
  - Interpolated endpoints: {original: {x, y}, transformed: {x, y}}
  - Stop info: {stopped_at_frame, stop_threshold_y, top_boundary_y}
  - Statistics: {total_points, extrapolated_endpoints}
- `*_original_trajectory.png` - **NEW**: Trajectory plot on perspective view with stop threshold line
- `*_overhead_trajectory.png` - **NEW**: Trajectory plot on overhead (transformed) view

**Stage G Post-Processing Outputs:**
- `trajectory_processed.csv` - Cleaned trajectory in homography space (median filter + outlier removal + smoothing)
- `trajectory_reconstructed.csv` - Final trajectory scaled to lane template coordinates

**Stage A-B Intermediate Outputs:**
- `output/<video_name>/ball_detection/intermediate/cropped_<video>_lane_masked.mp4` - 4-side masked video
- `output/<video_name>/ball_detection/cropped_<video>_transformed.mp4` - Overhead perspective view
- `output/<video_name>/ball_detection/intermediate/cropped_<video>_foreground_mask.mp4` - MOG2 output
- `output/<video_name>/ball_detection/intermediate/cropped_<video>_shadow_removed.mp4` - After shadow threshold
- `output/<video_name>/ball_detection/intermediate/cropped_<video>_denoised.mp4` - Final clean mask (with shadow separation)
- `output/<video_name>/ball_detection/intermediate/cropped_<video>_motion_comparison.mp4` - 2×2 comparison

### Using as a Module

**Phase 1: Lane Detection**
```python
from src.lane_detection import LaneDetector, config

# Create detector instance
detector = LaneDetector('path/to/video.mp4', config)

# Run complete detection pipeline
boundaries, intersections = detector.detect_all()
```

**Phase 2: Integrated Ball Detection (Tracking-by-Detection Architecture)**
```python
from src.ball_detection.main import run_integrated_tracking
from src.ball_detection import config

# Run complete integrated pipeline (Stages B+C+D+E)
# Returns: list of tracking results with detections, predictions, ROI info
tracking_results = run_integrated_tracking('video.mp4', config)

# Each result contains:
# - detection: {center, radius, ...} or None
# - prediction: {x, y, vx, vy} or None (from Kalman)
# - mode: 'global' or 'local'
# - search_type: 'initial' or 'reactivation' (if global)
# - roi_box: (x1, y1, x2, y2) or None
# - all_candidates: List of all validated candidates from Stage D
# - last_known_y: Y position when ball last seen (for reactivation)
```

**Phase 2: Stage B Motion Detection (Legacy)**
```python
from src.ball_detection.mask_video import create_masked_lane_video
from src.ball_detection.motion_detection import apply_background_subtraction
from src.ball_detection import config

# Get masked frames generator
frames_gen = create_masked_lane_video('video.mp4', config, save_video=False)

# Apply motion detection (MOG2 + shadow removal + shadow separation + denoising)
motion_gen = apply_background_subtraction(frames_gen, config, save_videos=False)

# Process each denoised frame
for frame_idx, denoised_mask, metadata, intermediate_masks in motion_gen:
    # denoised_mask is the clean binary mask (ball = white, background = black)
    # Find ball contours and extract position
    ball_position = detect_ball_from_mask(denoised_mask)
    trajectory.append(ball_position)
```

---

## Architecture Deep Dive

### Tracking-by-Detection (Stage C+D+E Integration)

**The Problem We Solved:**
Original implementation violated the Tracking-by-Detection principle by filtering candidates AFTER using them for Kalman updates. This meant:
- Kalman filter received raw motion detections (including noise, shadows, bowler's hand)
- Predictions became unreliable, causing drift
- ROI box could follow false detections

**Correct Architecture:**
```
Stage B (Motion) → Stage D (Filter ALL full-frame) → Stage C (Select based on state) → Stage E (Update Kalman)
```

**Key Principles:**
1. **Filter First**: Stage D validates ALL candidates on full frame (independent of tracking state)
2. **Select Smart**: Stage C chooses best candidate based on tracking mode (global vs local)
3. **Track Last**: Stage E updates Kalman ONLY with validated candidates

**Implementation Details:**

**Stage D - Blob Analysis (Full Frame Filtering)**
- Runs on complete frame regardless of tracking state
- Geometric validation: circularity ≥ 0.70, aspect ratio ≤ 1.8
- Size filtering: perspective-aware (larger near foul, smaller near pins)
- Returns: List of ALL validated candidates

**Stage C - Selection Strategy (State-Based)**

*Global Search Type 1 (Initial):*
- When: First detection or after timeout reset
- Strategy: Exclude upper 30% (foul line exclusion factor)
- Selection: Candidate with highest Y (closest to foul line)
- Purpose: Detect ball near bowler without detecting bowler's head

*Global Search Type 2 (Reactivation):*
- When: Confirmed ball lost mid-lane
- Strategy: Search Y < last_known_y + 50px (search ABOVE toward pins)
  - ⚠️ CRITICAL FIX: Changed from `Y > last_y - 50` to `Y < last_y + 50`
  - Correctly searches toward pins (smaller Y values in image coordinates)
  - Prevents false detections in bowler's hand area below last position
- Selection: Closest to last known position
- Purpose: Prevent re-detecting bowler when ball already at pins
- Timeout: Reset to initial search after 100 frames without detection

*Local Tracking:*
- When: Ball actively tracked
- Strategy: Filter candidates to ROI around Kalman prediction
- ROI Size: B = max(30px, 0.15 × y_ball) - perspective-aware scaling
- Selection: Closest to Kalman prediction
- Fallback: Switch to global reactivation after MAX_LOST_FRAMES (2 frames)

**Stage E - Kalman Filter Update**
- Only updates with validated candidates from Stage D
- 4-state filter: x, y, vx, vy
- Prediction used for ROI calculation in local mode
- Never uses raw motion detections

### Bug Fixes Implemented

**1. Reactivation Search Direction (CRITICAL)**
- **Problem**: Searched Y < last_y + margin (toward foul line/bowler)
- **Fix**: Changed to Y > last_y - margin (toward pins)
- **Impact**: System no longer re-detects bowler after ball passes mid-lane

**2. Shadow Interference**
- **Problem**: Ball+shadow merged into irregular blob, or shadow detected as separate candidate
- **Fix**: 
  - Stage B: Added erosion-based separation (2 iterations before morphological opening)
  - Stage D: Strengthened filters (circularity 0.65→0.70, aspect ratio 2.0→1.8)
- **Impact**: Multi-layer defense prevents shadow false positives

**3. Visualization Inaccuracy**
- **Problem**: Reactivation zone shown at current detection position (middle of frame) instead of actual last_known_y
- **Fix**: Added last_known_y to result dict, visualization uses actual tracker state
- **Impact**: Debugging videos now show correct search boundaries

**4. Kalman Drift Prevention**
- **Problem**: MAX_LOST_FRAMES=5 allowed prediction to drift backward when ball disappeared at pins
- **Fix**: Reduced to MAX_LOST_FRAMES=2
- **Impact**: Quick transition from LOCAL→GLOBAL prevents ROI from following drifted predictions

**5. Last Known Position Reset**
- **Problem**: last_known_y reset to None when unconfirmed tracking failed, causing stale visualization values
- **Fix**: Only reset last_known_y during initialization and timeout, preserve for reactivation
- **Impact**: Reactivation search always uses most recent ball position

### Configuration Parameters

**Key parameters in [config.py](src/ball_detection/config.py):**

```python
# Reactivation Search
REACTIVATION_SEARCH_MARGIN = 50      # Search 50px above last Y
FOUL_LINE_EXCLUSION_FACTOR = 0.3    # Exclude upper 30% in initial search
REACTIVATION_TIMEOUT = 60            # Reset to full frame after 60 frames
MAX_LOST_FRAMES = 2                  # Quick switch to prevent Kalman drift

# Stage D Filters
CIRCULARITY_THRESHOLD = 0.70         # Reject irregular shapes (shadows)
ASPECT_RATIO_MAX = 1.8               # Reject elongated shadows
USE_SHADOW_SEPARATION = True         # Enable erosion-based separation
SHADOW_SEPARATION_ITERATIONS = 2     # Erosion iterations

# Stage B Motion Detection
MOG2_HISTORY = 500                   # Background learning frames
MOG2_VAR_THRESHOLD = 40              # Variance threshold
SHADOW_THRESHOLD = 200               # Binary threshold for shadow removal

# Stage C+E Tracking
CONFIRMATION_THRESHOLD = 20          # Frames to confirm ball
SPATIAL_CONFIRMATION_DISTANCE = 240  # Pixels traveled for confirmation
K_SCALE = 0.15                       # ROI scale factor: B = k × y_ball
B_MIN = 30                           # Minimum ROI size at pins
```

### Output Files

After processing, check the `output/<video_name>/` directory for:
- `master_final_*.mp4` - Video with detected lane boundaries
- `bin_analysis_left.png` - Voting distribution for left boundary
- `bin_analysis_right.png` - Voting distribution for right boundary
- `tracking_*.png` - Tracking stability over time

---

## Documentation

Detailed documentation is available in the [`docs/`](docs/) directory:

- **[ANGLE_GUIDE.md](docs/ANGLE_GUIDE.md)** - Understanding angle calculations and perspective
- **[FIXES.md](docs/FIXES.md)** - Bug fixes and implementation improvements
- **[PERSPECTIVE_GUIDE.md](docs/PERSPECTIVE_GUIDE.md)** - Perspective correction techniques
- **[WHATS_NEW.md](docs/WHATS_NEW.md)** - Version history and changes

### Phase 1: Lane Detection

- **[LANE_DETECTOR_GUIDE.md](docs/lane_detection/LANE_DETECTOR_GUIDE.md)** - LaneDetector class usage and architecture
- **[REFACTOR_SUMMARY.md](docs/lane_detection/REFACTOR_SUMMARY.md)** - Phase 1 refactoring details

---

## Development Roadmap

### Phase 1: Lane Detection ✅ COMPLETE
- [x] Horizontal foul line detection (bottom boundary)
- [x] Vertical boundary detection (left & right sides)
- [x] Master line voting system
- [x] Perspective correction
- [x] Tracking analysis
- [x] **Top boundary detection with MSAC fitting**
- [x] Frame caching system
- [x] Small patch removal
- [x] Professional class-based architecture (LaneDetector)
- [x] 6 intermediate visualization modes

### Phase 2: Ball Detection & Tracking ✅ COMPLETE
- [x] **Stage A: Video Preprocessing**
  - [x] 4-side lane masking using Phase 1 boundaries
  - [x] 2D homography calculation (DLT)
  - [x] Perspective transformation to overhead view
  - [x] High-quality encoding (PNG + yuv444p)
- [x] **Stage B: Motion Detection**
  - [x] MOG2 background subtraction
  - [x] Shadow removal and separation
  - [x] Morphological noise removal
- [x] **Stage D: Blob Analysis**
  - [x] Geometric validation (circularity, aspect ratio)
  - [x] Auto-calibration system
- [x] **Stage C+E: Tracking-by-Detection**
  - [x] Kalman filter tracking
  - [x] Dual-mode search (global + local)
  - [x] Confirmation logic
  - [x] Reactivation search
- [x] **Stage F: Stop Condition & Export**
  - [x] Configurable stop threshold
  - [x] 5 Kalman predictions for extrapolation
  - [x] Trajectory JSON export (original + overhead)
  - [x] Trajectory plots
- [x] **Stage G: Post-Processing**
  - [x] Trajectory cleaning (median filter, outlier detection)
  - [x] Template reconstruction with scaling
  - [x] CSV export for analysis
- [x] **Integrated Visualization**
  - [x] 4 diagnostic videos (candidates, selection, trajectory, debug)

### Phase 3: Advanced Trajectory Analysis (In Progress)
- [ ] Velocity and acceleration curves
- [ ] Path curvature analysis for hook detection
- [ ] Impact angle calculations
- [ ] Ball speed measurements (mph/fps)
- [ ] Multi-throw comparative analysis
- [ ] Statistical trajectory metrics

### Phase 4: 3D Reconstruction (Planned)
- [ ] Camera calibration
- [ ] Perspective transformation
- [ ] 3D trajectory mapping
- [ ] Height estimation

### Phase 4: Spin Analysis (Planned)
- [ ] Rotation detection
- [ ] Axis calculation
- [ ] Angular velocity measurement
- [ ] Spin visualization

### Phase 5: Pin Detection (Planned)
- [ ] Pin position detection
- [ ] Topple counting
- [ ] Strike/spare classification
- [ ] Score calculation

---

## Contributing

This is an academic project for the **Image Analysis and Computer Vision** course. While we're not accepting external contributions at this time, feedback and suggestions are welcome!

### Development Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run tests (when available)
pytest tests/
```

---

## Academic Context

This project is being developed from scratch as part of our coursework in Image Analysis and Computer Vision. The implementation applies computer vision principles and techniques learned throughout the course, including:

- Edge detection and line detection (Hough Transform)
- Image preprocessing and filtering
- Perspective geometry
- Object tracking algorithms
- Feature extraction
- 3D reconstruction techniques

---

## License

Academic project for educational purposes. All rights reserved by the team members.

---

## Acknowledgments

- Course instructors and teaching assistants
- OpenCV community for excellent documentation
- Fellow students for valuable discussions and feedback

---

## Contact

For questions or collaboration inquiries:

- **Mohammad Ammar Mughees**: [GitHub](https://github.com/Black-Lights)
- **Mohmmad Umayr Romshoo**: Contact via course portal

---

## Project Status & Updates

**Phase 1 (Lane Detection)** - ✅ **COMPLETE** (February 2026)
- All 4 boundaries successfully detected (top, bottom, left, right)
- Professional class-based architecture (LaneDetector)
- Frame caching for performance optimization
- MSAC-based top boundary detection

**Phase 2 (Ball Detection & Tracking)** - ✅ **COMPLETE** (February 2026)
- Complete pipeline: Stages A through G integrated
- Tracking-by-Detection architecture (filter → select → track)
- Stop condition with Kalman predictions
- Post-processing with trajectory cleaning and reconstruction
- 4 diagnostic visualization videos
- JSON and CSV trajectory export

**Current Focus**: Advanced trajectory analysis (velocity curves, hook detection, impact angles)

**Next Phase**: 3D trajectory reconstruction and spin/rotation analysis

**Last Updated**: February 5, 2026

---

## Recent Achievements

### Stage G Post-Processing (February 5, 2026)
- Integrated trajectory cleaning pipeline
- MAD outlier detection with Modified Z-score
- Template reconstruction with coordinate scaling
- CSV export for external analysis tools

### Complete Phase 2 Pipeline (February 2026)
- Tracking-by-Detection architecture implemented
- All 7 stages integrated (A through G)
- Tested on multiple videos with excellent results
- Zero outliers detected in test6 (indicates robust tracking)

### Bug Fixes & Improvements
- Fixed reactivation search direction (critical fix)
- Implemented shadow separation via erosion
- Strengthened geometric filters
- Prevented Kalman drift with quick fallback
