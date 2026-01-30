# Ball Tracking - Phase 2

Automated ball detection and tracking system for bowling lane analysis.

## Overview

Phase 2 builds upon Phase 1 (Lane Detection) to track the bowling ball's trajectory from release to pin impact. It uses frame differencing and color segmentation with Kalman filtering for robust tracking.

## Features

- **Multi-Method Detection**: Combines motion detection (frame differencing) and color segmentation
- **Kalman Filtering**: Smooth trajectory prediction and handling of temporary occlusions
- **Lane Masking**: Uses Phase 1 boundaries to focus detection on the lane area
- **Key Point Detection**: Automatically identifies ball release and pin impact points
- **Comprehensive Visualization**: Generates tracking videos, trajectory plots, and velocity analysis
- **Detailed Analytics**: Exports trajectory data, statistics, and analysis reports

## Architecture

```
src/ball_tracking/
├── __init__.py              # Module initialization
├── config.py                # Configuration parameters
├── detection_functions.py   # Ball detection algorithms
├── ball_tracker.py          # Main BallTracker class
├── tracking_analysis.py     # Trajectory analysis and exports
├── visualization.py         # Drawing and plotting functions
└── main.py                  # Command-line entry point
```

## Detection Approach

### 1. Motion Detection (Frame Differencing)
- Computes absolute difference between consecutive frames
- Applies Gaussian blur to reduce noise
- Thresholds to binary mask highlighting moving regions

### 2. Color Segmentation
- Converts frame to HSV color space
- Applies color range filter for ball color (default: black/dark bowling ball)
- Uses morphological operations to clean up mask

### 3. Mask Combination
- Combines motion and color masks with logical AND or OR
- Applies additional morphological operations
- Masks non-lane regions using Phase 1 boundaries

### 4. Contour Analysis
- Extracts contours from combined mask
- Filters by size, circularity, and solidity
- Ranks candidates by confidence score

### 5. Kalman Tracking
- 4-state Kalman filter (x, y, vx, vy)
- Predicts ball position when not detected
- Associates detections with predicted positions
- Handles temporary occlusions (up to 8 frames)

## Usage

### Command Line

```bash
# Run with auto-detection of Phase 1 boundary data
python -m src.ball_tracking.main --video assets/input/cropped_test8.mp4

# Run with custom boundary data
python -m src.ball_tracking.main --video video.mp4 --boundary path/to/boundary_data.json

# Specify custom output directory
python -m src.ball_tracking.main --video video.mp4 --output custom_output/
```

### Python API

```python
from src.ball_tracking.ball_tracker import BallTracker
from src.ball_tracking import config
import json

# Load boundary data from Phase 1
with open('output/cropped_test8/boundary_data.json', 'r') as f:
    boundary_data = json.load(f)

# Initialize tracker
tracker = BallTracker(
    video_path='assets/input/cropped_test8.mp4',
    boundary_data=boundary_data,
    config_module=config
)

# Run tracking
results = tracker.track_all()

# Access results
trajectory = results['trajectory']
release_point = results['release_point']
impact_point = results['impact_point']
stats = results['statistics']
```

## Configuration

Key parameters in `config.py`:

### Detection
```python
# Motion detection
USE_FRAME_DIFFERENCING = True
FRAME_DIFF_THRESHOLD = 30
GAUSSIAN_BLUR_SIZE = (5, 5)

# Color detection
USE_COLOR_DETECTION = True
BALL_COLOR_LOWER = (0, 0, 0)      # HSV lower bound
BALL_COLOR_UPPER = (180, 255, 60)  # HSV upper bound

# Shape filtering
MIN_RADIUS = 8
MAX_RADIUS = 50
MIN_CIRCULARITY = 0.65
MIN_SOLIDITY = 0.75
```

### Tracking
```python
# Kalman filter
PROCESS_NOISE = 0.03
MEASUREMENT_NOISE = 1.0
MAX_MISSING_FRAMES = 8

# Trajectory analysis
RELEASE_VELOCITY_THRESHOLD = 3.0
IMPACT_VELOCITY_THRESHOLD = 2.0
```

### Visualization
```python
SAVE_TRACKING_VIDEO = True
SAVE_TRAJECTORY_PLOT = True
SAVE_VELOCITY_PLOT = True
DRAW_TRAJECTORY = True
TRAJECTORY_LENGTH = 40
```

## Output Structure

```
output/<video_name>/tracking/
├── tracking_data.json         # Full trajectory data
├── analysis_report.txt        # Statistical analysis
├── tracking_video.mp4         # Annotated video with tracking overlay
├── trajectory_plot.png        # 2D trajectory visualization
└── velocity_plot.png          # Velocity over time
```

### tracking_data.json Format

```json
{
  "video_info": {
    "video_name": "cropped_test8",
    "total_frames": 300,
    "fps": 30,
    "duration": 10.0
  },
  "trajectory": [
    {
      "frame": 0,
      "center": [400, 450],
      "radius": 12,
      "confidence": 0.95,
      "detected": true,
      "velocity": [2.5, -1.8],
      "timestamp": 0.0
    }
  ],
  "release_point": {
    "frame": 15,
    "position": [405, 430],
    "velocity": 5.2,
    "timestamp": 0.5
  },
  "impact_point": {
    "frame": 285,
    "position": [420, 200],
    "velocity": 3.8,
    "timestamp": 9.5
  },
  "statistics": {
    "total_points": 300,
    "detected_points": 285,
    "predicted_points": 15,
    "detection_rate": 0.95,
    "max_velocity": 6.3,
    "avg_velocity": 4.2
  }
}
```

## Key Point Detection

### Release Point
- Detected in bottom 30% of lane (near foul line)
- Requires velocity > 3.0 pixels/frame
- First point after ball starts moving significantly

### Impact Point
- Detected in top 20% of lane (near pins)
- Ball enters pin area
- Velocity typically still > 2.0 pixels/frame

## Dependencies

- OpenCV (cv2): Video processing and computer vision
- NumPy: Numerical computations
- Matplotlib: Plotting and visualization
- tqdm: Progress bars

Install via:
```bash
pip install opencv-contrib-python numpy matplotlib tqdm
```

## Integration with Phase 1

Phase 2 requires boundary data from Phase 1:

1. **Run Phase 1** (Lane Detection):
   ```bash
   python main.py --video assets/input/cropped_test8.mp4
   ```

2. **Verify boundary data** exists:
   ```
   output/cropped_test8/boundary_data.json
   ```

3. **Run Phase 2** (Ball Tracking):
   ```bash
   python -m src.ball_tracking.main --video assets/input/cropped_test8.mp4
   ```

## Troubleshooting

### Ball not detected
- **Adjust color ranges**: Modify `BALL_COLOR_LOWER` and `BALL_COLOR_UPPER` in config.py
- **Check motion threshold**: Lower `FRAME_DIFF_THRESHOLD` if ball moves slowly
- **Verify lane mask**: Ensure Phase 1 boundaries are correct

### False positives
- **Increase shape filters**: Raise `MIN_CIRCULARITY` and `MIN_SOLIDITY`
- **Adjust size constraints**: Tighten `MIN_RADIUS` and `MAX_RADIUS`
- **Enable both methods**: Set both `USE_FRAME_DIFFERENCING` and `USE_COLOR_DETECTION` to True

### Tracking interruptions
- **Increase occlusion handling**: Raise `MAX_MISSING_FRAMES`
- **Adjust Kalman noise**: Fine-tune `PROCESS_NOISE` and `MEASUREMENT_NOISE`

### Release/Impact not detected
- **Adjust velocity thresholds**: Lower `RELEASE_VELOCITY_THRESHOLD` or `IMPACT_VELOCITY_THRESHOLD`
- **Check zone boundaries**: Modify `RELEASE_ZONE_Y` and `IMPACT_ZONE_Y`

## Future Enhancements

- Multi-ball tracking for split shots
- Ball spin detection using template matching
- Speed in real-world units (mph/kph) using calibration
- Pin impact analysis and scoring
- Statistical comparison across multiple throws

## Version History

- **2.0.0** (January 30, 2026): Initial Phase 2 implementation
  - Frame differencing and color segmentation
  - Kalman filtering for robust tracking
  - Release and impact point detection
  - Comprehensive visualization and analytics

## Authors

- Mohammad Umayr Romshoo
- Mohammad Ammar Mughees

## References

- Research based on: "Sviluppo di un sistema di tracciamento di una palla da basket mediante tecniche di visione artificiale" (Luca Pirotta, 2013)
- Adapted for bowling with simplified detection approach suitable for static camera and controlled environment
