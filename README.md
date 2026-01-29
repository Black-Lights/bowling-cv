# Bowling Analysis Project

[![Status](https://img.shields.io/badge/status-in%20development-yellow)](https://github.com/Black-Lights/bowling-cv)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-Academic-lightgrey)](LICENSE)

Computer vision system for analyzing bowling ball trajectory, spin/rotation axis, and toppled pins from video recordings using OpenCV and Python.

## Team Members

- **Mohmmad Umayr Romshoo**
- **Mohammad Ammar Mughees** ([Black-Lights](https://github.com/Black-Lights))

## Project Status

**Phase 1: Lane Detection** - In Progress (Bottom & Side Boundaries Complete, Top Boundary Pending)  
**Phase 2: Ball Tracking** - Planned  
**Phase 3: 3D Trajectory Reconstruction** - Planned  
**Phase 4: Spin/Rotation Analysis** - Planned  
**Phase 5: Pin Detection** - Planned

## Current Progress

Below is a demonstration of the lane detection system working on a sample bowling video. The system successfully detects:
- **Foul line** (magenta horizontal line at the bottom)
- **Left lane boundary** (green vertical line)
- **Right lane boundary** (red vertical line)

![Lane Detection Demo](output/cropped_test3/master_final_cropped_test3.mp4)

*Sample output showing detected lane boundaries on cropped_test3.mp4*

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenCV 4.0+
- NumPy, SciPy
- Pandas, Matplotlib
- tqdm (for progress bars)

### Installation

```bash
# Clone the repository
git clone https://github.com/Black-Lights/bowling-cv.git
cd bowling-cv

# Install required packages
pip install -r requirements.txt
```

### Running Lane Detection

```bash
# Place your bowling video files in assets/input/
# Then run the lane detection module
cd src/lane_detection
python main.py
```

Results will be saved in the `output/` directory.

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
│   └── lane_detection/            # Lane boundary detection module
│       ├── __init__.py
│       ├── config.py              # Configuration settings
│       ├── main.py                # Main entry point
│       ├── detection_functions.py # Line detection algorithms
│       ├── detection_utils.py     # Utility functions
│       ├── master_line_computation.py # Master line voting system
│       ├── intermediate_visualization.py # Debug visualizations
│       └── tracking_analysis.py   # Tracking stability analysis
│
├── output/                        # Generated outputs
│   └── <video_name>/
│       ├── master_final_*.mp4     # Processed video with lane boundaries
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

### Implemented (Phase 1 - Partial)
- **Lane Boundary Detection**
  - Horizontal foul line detection (bottom boundary)
  - Vertical lane boundary detection (left & right sides)
  - Master line computation using voting system
  - Perspective-aware angle calculations
  - Tracking stability analysis
  - Multiple visualization modes
  - **Top boundary detection (pending)**

### In Progress
- **Complete Lane Detection (Phase 1)**
  - Top boundary of bowling area detection
  - Full lane box detection

### Planned
- **Ball Detection and Tracking (Phase 2)**
  - Ball detection using color/motion
  - Frame-to-frame tracking
  - Trajectory smoothing
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
from src.lane_detection import (
    detect_horizontal_line,
    detect_vertical_boundaries_approach1,
    compute_master_line_from_collection
)

# Your code here...
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
