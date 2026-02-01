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
**Phase 2: Ball Detection** - 🔄 **IN PROGRESS** (Masking + Homography Complete)  
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
│       └── motion_detection.py    # MOG2 background subtraction (Stage B)
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
│       └── tracking_*.png         # Tracking stability plots
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
  - ✅ Vertical lane boundary detection (left & right sides)
  - ✅ Top boundary detection (pin area) with MSAC line fitting
  - ✅ Master line computation using voting system
  - ✅ Perspective-aware angle calculations
  - ✅ Tracking stability analysis
  - ✅ Multiple visualization modes
  - ✅ HSV preprocessing with gap filling
  - ✅ Robust MSAC (M-estimator SAmple Consensus) fitting
  - ✅ Complete lane box (all 4 boundaries)

### 🔄 In Progress (Phase 2 - Ball Detection)
- **Video Masking** ✅ COMPLETE
  - Reuses Phase 1 lane boundaries
  - 4-side masking (top, bottom, left, right)
  - Two modes: video file or frame generator
  - Memory-efficient frame processing
  - Foul line area properly excluded (30px cutoff)
- **2D Homography & Perspective Transformation** ✅ COMPLETE
  - Direct Linear Transform (DLT) for homography calculation
  - Perspective transformation to overhead view
  - Uniform scaling (20 px/in) preserves circular shapes
  - Auto-crop to remove black borders
  - High-quality encoding (PNG frames + yuv444p)
  - Real-world dimensions: 60 ft × 41.5 in bowling lane
- **Motion Detection (Background Subtraction)** ✅ COMPLETE
  - MOG2 (Mixture of Gaussians) background subtractor
  - Shadow removal (threshold grey pixels at 127)
  - Morphological opening for noise removal (3×3 ellipse kernel)
  - Intermediate videos: foreground mask, shadow removed, denoised
  - 2×2 comparison video for debugging
- **ROI Logic & Tracking (Kalman Filter)** ✅ COMPLETE
  - Dual-mode tracking: Global Search + Local Tracking
  - OpenCV Kalman Filter (4-state: x, y, vx, vy)
  - Perspective-aware dynamic ROI sizing: B_t = max(30px, 0.15 * y_ball)
  - Global search: prioritizes foul line + negative Y velocity filtering
  - Local tracking: searches within predicted ROI box
  - 10-frame timeout before reverting to global search
  - 6 intermediate videos: global search, local tracking, Kalman predictions, mode comparison, scaling demo, full pipeline
  - **Confirmation Logic (Problem 2 Solution)**:
    - Dual confirmation: 20 consecutive frames + 240px travel distance (~12 feet)
    - Unconfirmed object lost → Full lane search (prevents false restriction if tracking hand)
    - Confirmed ball lost → Restricted search (y < last_position - 50px buffer)
    - Physics-informed: ball cannot move back toward camera
    - Prevents re-detecting ball behind where it was lost
    - Successfully tested: Frame 109 confirmation, Frame 139 restricted search
- **Ball Filtering (Blob Analysis)** (Next - Stage D)
  - Circularity filter (C > 0.65)
  - Aspect ratio validation (< 2.0)
  - Size constraints (MIN/MAX radius)
  - Hand vs. ball discrimination
- **Trajectory Extraction** (Upcoming)
  - Ball position time series
  - Velocity and acceleration analysis
  - Path smoothing algorithms

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

### Phase 2: Ball Detection (Masking + Homography + Motion Detection + ROI Tracking)

```bash
# Run complete ball detection pipeline (all steps)
python -m src.ball_detection.main --video cropped_test3.mp4

# Or skip specific steps
python -m src.ball_detection.main --video cropped_test3.mp4 --skip-masking --skip-transform --skip-motion
```

**Outputs:**
- `output/<video_name>/ball_detection/intermediate/cropped_<video>_lane_masked.mp4` - 4-side masked video
- `output/<video_name>/ball_detection/cropped_<video>_transformed.mp4` - Overhead perspective view
- `output/<video_name>/ball_detection/intermediate/cropped_<video>_foreground_mask.mp4` - MOG2 output
- `output/<video_name>/ball_detection/intermediate/cropped_<video>_shadow_removed.mp4` - After shadow threshold
- `output/<video_name>/ball_detection/intermediate/cropped_<video>_denoised.mp4` - Final clean mask
- `output/<video_name>/ball_detection/intermediate/cropped_<video>_motion_comparison.mp4` - 2×2 comparison
- **Stage C ROI Tracking Videos (6 total):**
  - `cropped_<video>_roi_global_search.mp4` - Global search mode visualization
  - `cropped_<video>_roi_local_tracking.mp4` - Local tracking mode visualization
  - `cropped_<video>_kalman_prediction.mp4` - Kalman filter predictions
  - `cropped_<video>_roi_mode_comparison.mp4` - Side-by-side mode comparison
  - `cropped_<video>_roi_scaling_demo.mp4` - Perspective-aware ROI scaling
  - `cropped_<video>_full_roi_pipeline.mp4` - 2×3 grid showing all stages

### Using as a Module

**Phase 1: Lane Detection**
```python
from src.lane_detection import LaneDetector, config

# Create detector instance
detector = LaneDetector('path/to/video.mp4', config)

# Run complete detection pipeline
boundaries, intersections = detector.detect_all()
```

**Phase 2: Ball Detection (Motion Detection with MOG2)**
```python
from src.ball_detection.mask_video import create_masked_lane_video
from src.ball_detection.motion_detection import apply_background_subtraction
from src.ball_detection import config

# Get masked frames generator
frames_gen = create_masked_lane_video('video.mp4', config, save_video=False)

# Apply motion detection (MOG2 + shadow removal + denoising)
motion_gen = apply_background_subtraction(frames_gen, config, save_videos=False)

# Process each denoised frame
for frame_idx, denoised_mask, metadata, intermediate_masks in motion_gen:
    # denoised_mask is the clean binary mask (ball = white, background = black)
    # Find ball contours and extract position
    ball_position = detect_ball_from_mask(denoised_mask)
    trajectory.append(ball_position)
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

### Phase 1: Lane Detection (In Progress)
- [x] Horizontal foul line detection (bottom boundary)
- [x] Vertical boundary detection (left & right sides)
- [x] Master line voting system
- [x] Perspective correction
- [x] Tracking analysis
- [ ] **Top boundary detection** ← Next task

### Phase 2: Ball Tracking (Planned)
- [ ] Ball detection algorithm
- [ ] Multi-frame tracking
- [ ] Trajectory extraction
- [ ] Position smoothing

### Phase 3: 3D Reconstruction (Planned)
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

## Note

This is a work in progress. **Phase 1 (Lane Detection)** is currently being completed - bottom foul line and side boundaries are working, with top boundary detection as the next development task. The implementation is being developed iteratively, with each phase building upon the previous one.

**Current Focus**: Completing lane detection by adding top boundary detection to fully define the bowling area.

**Last Updated**: January 2026
