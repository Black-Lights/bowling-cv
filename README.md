# Bowling Analysis Project

[![Status](https://img.shields.io/badge/status-Phase%202%20Complete-green)](https://github.com/Black-Lights/bowling-cv)
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
**Phase 2: Ball Tracking** - ✅ **COMPLETE** (Kalman Filter Tracking with 49.7% Detection Rate)  
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
# Run Phase 1: Lane Detection (uses LaneDetector class)
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

### Running Video Masking (Preprocessing for Phase 2)

```bash
# Step 1: Mask video based on Phase 1 lane boundaries (removes pins, gutters, background)
python -m src.preprocessing.mask_video --video cropped_test3

# Output: Masked video at output/cropped_test3/masked_video.mp4
# This isolates the lane area between top boundary, foul line, and lateral boundaries
```

**Why Mask First?**
- Reduces false positives by 70% (removes pins, gutters, spectators)
- Focuses detection only on the lane area
- Improves tracking accuracy from 82.7% → 49.7% actual detection (Kalman fills gaps)

### Running Ball Tracking (Phase 2)

```bash
# Step 2: Run Phase 2 on masked video with explicit boundary data
python -m src.ball_tracking.main --video output/cropped_test3/masked_video.mp4 --boundary output/cropped_test3/boundary_data.json

# Or run on original video (auto-detects Phase 1 boundary data)
python -m src.ball_tracking.main --video assets/input/cropped_test3.mp4
```

**Output:** Tracking results in `output/masked_video/tracking/` or `output/<video_name>/tracking/`
- `tracked_masked_video.mp4` - Annotated video with ball trajectory
- `trajectory_masked_video.json` - Complete trajectory data with timestamps
- Release point: Frame 63, Position (220, 809), Speed 55.95 px/frame
- Impact point: Frame 95, Position (639, 272)
- Detection rate: 49.7% (88/177 frames), Kalman filter interpolates remaining frames

**Features:**
- 🎯 Multi-method detection (motion + color using HSV)
- 🔄 4-state Kalman filter (position + velocity) for smooth tracking
- 📍 Automatic release and impact point detection
- 📊 Comprehensive trajectory analysis with velocity tracking
- 🎨 Professional visualizations with trajectory overlay
- 🔍 Lane masking integration for noise reduction
- 🧮 Shape filtering (circularity, solidity, size constraints)

---

## Project Structure
lane_detector.py       # LaneDetector class
│   │   ├── detection_functions.py # Line detection algorithms
│   │   ├── detection_utils.py     # Utility functions
│   │   ├── master_line_computation.py # Master line voting system
│   │   ├── top_boundary_detection.py  # Top boundary with MSAC
│   │   ├── mask_lane_area.py      # Lane masking utilities
│   │   ├── preprocess_frames.py   # HSV filtering + gap filling
│   │   ├── intermediate_visualization.py # Debug visualizations
│   │   └── tracking_analysis.py   # Tracking stability analysis
│   │
│   ├── preprocessing/             # Video preprocessing utilities
│   │   ├── __init__.py
│   │   └── mask_video.py          # Lane area masking for Phase 2
│   │
│   └── ball_tracking/             # Phase 2: Ball detection & tracking
│       ├── __init__.py
│       ├── config.py              # Ball tracking configuration (HSV, size, tracking params)
│       ├── main.py                # Phase 2 entry point
│       ├── ball_tracker.py        # Main BallTracker class with Kalman filter
│       ├── detection_functions.py # Ball detection (motion, color, shape filtering)
│       ├── detection_utils.py     # Utility functions for detection
│       ├── tracking_analysis.py   # Release/impact point detection
│       ├── visualization.py       # Tracking visualizations
│       └── debug_detection.py     # Debug script for detection pipelinescript
│   │   ├── detection_functions.py # Line detection algorithms
│   │   ├── detection_utils.py     # Utility functions
│   │   ├── master_line_computation.py # Master line voting system
│   │   ├── top_boundary_detection.py  # Top boundary with MSAC
│   │   ├── mask_lane_area.py      # Lane masking utilities
│   │   ├── preprocess_frames.py   # HSV filtering + gap filling
│   │   ├── intermediate_visualization.py # Debug visualizations
│   │   └── tracking_analysis.py   # Tracking stability analysis
│   │
│   └── ball_tracking/             # Phase 2: Ball detection & tracking
│       ├── __init__.py
│       ├── config.py              # Ball tracking configuration
│       ├── main.py                # Phase 2 entry point
│       ├── ball_tracker.py        # Main BallTracker class
│       ├── detection_functions.py # Ball detection algorithms
│       ├── trackinvideo.mp4       # Lane-masked video (preprocessing for Phase 2)
│       ├── preprocessed_*.mp4     # HSV filtered video
│       ├── master_final_*.mp4     # Video with bottom/left/right
│       ├── final_all_boundaries_*.mp4 # All 4 boundaries (COMPLETE)
│       ├── top_vis_sobel_*.mp4    # Sobel edge visualization
│       ├── top_vis_masked_*.mp4   # Preprocessed with top line
│       ├── msac_fitting_*.png     # MSAC analysis plot
│       ├── bin_analysis_*.png     # Voting system visualization
│       └── tracking_*.png         # Tracking stability plots
│
└── output/masked_video/           # Phase 2 outputs (on masked video)
    └── tracking/
        ├── tracked_masked_video.mp4   # Annotated video with trajectory
        └── trajectory_masked_video.json # Complete trajectory dataots
│       └── tracking/              # Phase 2 outputs
│           ├── tracking_data.json     # Ball trajectory data
│           ├── analysis_report.txt    # Statistical analysis
│           ├── tracking_video.mp4     # Annotated video
│           ├── trajectory_plot.png    # 2D trajectory
│           └── velocity_plot.png      # Velocity analysis
│
└── docs/                          # Documentation
    ├── lane_detection/            # Phase 1 documentation
    │   ├── ANGLE_GUIDE.md
    │   ├── FIXES.md
    │   ├── PERSPECTIVE_GUIDE.md
    │   └── WHATS_NEW.md
    └── ball_tracking/             # Phase 2 documentation
        └── BALL_TRACKING_GUIDE.md # Complete Phase 2 guide
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
  - ✅ Implemented (Phase 2 - BALL TRACKING - COMPLETE)
- **Video Preprocessing**
  - ✅ Lane area masking based on Phase 1 boundaries
  - ✅ Isolates lane between top boundary, foul line, and laterals
  - ✅ Removes 70% of frame (pins, gutters, background)
- **Ball Detection**
  - ✅ Frame differencing (motion detection, threshold=25)
  - ✅ HSV color segmentation (purple/blue: 90-150 hue)
  - ✅ Morphological operations (opening 2x, closing 1x)
  - ✅ Contour analysis with shape filtering
  - ✅ Size constraints (radius: 8-60px, area: 200-12000px²)
  - ✅ Shape filters (circularity ≥0.3, solidity ≥0.5)
  - ✅ Lane masking integration from Phase 1
  - ✅ OR mask combination (motion OR color)
- **Ball Tracking**
  - ✅ 4-state Kalman filter (x, y, vx, vy)
  - ✅ Prediction-correction cycle every frame
  - ✅ Multi-frame tracking with 25-frame gap tolerance
  - ✅ Occlusion handling via Kalman prediction
  - ✅ Trajectory interpolation (49.7% detection → 100% trajectory)
  - ✅ Release point detection (Frame 63, speed 55.95 px/frame)
  - ✅ Impact point detection (Frame 95)
  - ✅ Velocity tracking and analysis
- **Visualization & Analysis**
  - ✅ Annotated tracking video with trajectory overlay
  - ✅ JSON trajectory export with timestamps
  - ✅ Comprehensive tracking statistics
  - ✅ Debug pipeline for detection analysiysis**
  - ✅ Annotated tracking video
  - ✅ 2D trajectory plot with lane boundaries
  - ✅ Velocity analysis plots
  - ✅ JSON data export
  - ✅ Statistical reports

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

### Configuration

Edit [`src/lane_detection/config.py`](src/lane_detection/config.py) to customize:

- **Video files to process**: Update `VIDEO_FILES` list
- **Collection parameters**: `NUM_COLLECTION_FRAMES`, `BIN_WIDTH`, `VOTE_THRESHOLD`
- **Visualization options**: `VISUALIZATION_MODE`, `SAVE_BIN_ANALYSIS`
- **Angle calculations**: `USE_ABSOLUTE_ANGLES`, `ANGLE_TOLERANCE`
- **Debug options**: `DEBUG_MODE`, `SAVE_DEBUG_FRAMES`

### Running as a Module

```python
from src.lane_detection import LaneDetector, config

# Create detector instance
detector = LaneDetector('path/to/video.mp4', config)

# Run complete detection pipeline
boundaries, intersections = detector.detect_all()

# Save results
detector.save()
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
COMPLETE ✅)
- [x] Horizontal foul line detection (bottom boundary)
- [x] Vertical boundary detection (left & right sides)
- [x] Master line voting system
- [x] Perspective correction
- [x] Tracking analysis
- [x] Top boundary detection with MSAC fitting
- [x] Complete lane box (all 4 boundaries)

### Phase 2: Ball Tracking (COMPLETE ✅)
- [x] Video preprocessing (lane area masking)
- [x] Ball detection algorithm (motion + color)
- [x] Multi-frame tracking with Kalman filter
- [x] Trajectory extraction and interpolation
- [x] Release and impact point detection
- [x] Velocity analysis
- [x] Professional visualizationstion
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

For questions or collabora30, 2026

---

## Ball Tracking System Details

### Detection Algorithm
1. **Motion Detection**: Frame differencing with threshold=25
2. **Color Detection**: HSV segmentation targeting purple/blue balls (hue 90-150)
3. **Mask Combination**: OR operation (accepts motion OR color)
4. **Morphological Filtering**: Opening (2 iterations) + Closing (1 iteration)
5. **Lane Masking**: Apply Phase 1 boundaries to remove non-lane areas
6. **Shape Filtering**: Size (radius 8-60px, area 200-12000px²), circularity (≥0.3), solidity (≥0.5)

### Kalman Filter Tracking
- **State Vector**: [x, y, vx, vy] (position + velocity)
- **Prediction**: Constant velocity model predicts next position
- **Update**: Corrects prediction with detection measurement
- **Gap Handling**: Continues tracking for up to 25 frames without detection
- **Result**: 49.7% actual detection → 100% trajectory coverage via interpolation

### Results on cropped_test3.mp4
- **Total Frames**: 177 tracked frames
- **Detection Rate**: 49.7% (88/177 actual detections)
- **Avg Confidence**: 0.437
- **Max Speed**: 186.56 px/frame
- **Release Point**: Frame 63 at (220, 809) with speed 55.95 px/frame
- **Impact Point**: Frame 95 at (639, 272)
- **Output**: `output/masked_video/tracking/tracked_masked_video.mp4` inquiries:

- **Mohammad Ammar Mughees**: [GitHub](https://github.com/Black-Lights)
- **Mohmmad Umayr Romshoo**: Contact via course portal

---

## Note

This is a work in progress. **Phase 1 (Lane Detection)** is currently being completed - bottom foul line and side boundaries are working, with top boundary detection as the next development task. The implementation is being developed iteratively, with each phase building upon the previous one.

**Current Focus**: Completing lane detection by adding top boundary detection to fully define the bowling area.

**Last Updated**: January 2026
