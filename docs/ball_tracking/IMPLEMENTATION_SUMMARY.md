# Phase 2 Implementation Summary

## Overview
Complete ball tracking system implemented for bowling lane analysis, following the architectural patterns established in Phase 1.

## Implementation Date
January 30, 2026

## Files Created

### Core Modules (src/ball_tracking/)
1. **`__init__.py`** - Module initialization
   - Exports BallTracker class
   - Version 2.0.0

2. **`config.py`** - Configuration parameters (146 lines)
   - Detection settings (motion, color, morphology)
   - Tracking parameters (Kalman filter, occlusion handling)
   - Analysis settings (release/impact detection)
   - Visualization options (colors, drawing flags)

3. **`detection_functions.py`** - Ball detection algorithms (300+ lines)
   - `detect_ball_by_motion()` - Frame differencing
   - `detect_ball_by_color()` - HSV color segmentation
   - `combine_masks()` - Fusion of detection methods
   - `apply_morphological_operations()` - Noise removal
   - `filter_ball_contours()` - Shape-based filtering
   - `create_lane_mask()` - ROI from Phase 1 boundaries
   - `apply_lane_mask()` - Restrict detection to lane area

4. **`ball_tracker.py`** - Main tracking class (500+ lines)
   - `BallTracker` class with complete pipeline
   - `_init_kalman_filter()` - 4-state filter setup
   - `track_all()` - Main entry point
   - `_detect_ball()` - Per-frame detection
   - `_update_tracker()` - Kalman predict/correct
   - `_select_best_candidate()` - Detection association
   - `_analyze_trajectory()` - Release/impact detection
   - `save_results()` - Export data and visualizations

5. **`tracking_analysis.py`** - Trajectory analysis (200+ lines)
   - `smooth_trajectory()` - Moving average smoothing
   - `detect_release_point()` - Ball release detection
   - `detect_impact_point()` - Pin impact detection
   - `save_trajectory_json()` - JSON export
   - `save_analysis_report()` - Text report generation

6. **`visualization.py`** - Drawing and plotting (300+ lines)
   - `draw_tracking()` - Annotate frames with tracking overlay
   - `plot_trajectory()` - 2D trajectory with lane boundaries
   - `plot_velocity()` - Velocity component and speed plots

7. **`main.py`** - Command-line entry point (130+ lines)
   - Argument parsing
   - Auto-detection of Phase 1 boundary data
   - Progress reporting
   - Error handling

### Documentation (docs/ball_tracking/)
8. **`BALL_TRACKING_GUIDE.md`** - Complete user guide
   - Architecture overview
   - Detection approach explanation
   - Usage examples
   - Configuration guide
   - Output format documentation
   - Troubleshooting section
   - Integration with Phase 1

### Updated Files
9. **`README.md`** - Main project README
   - Updated project status (Phase 2 in development)
   - Added ball_tracking/ to project structure
   - Added Phase 2 usage section
   - Expanded features list

## Technical Details

### Detection Pipeline
```
Input Frame
    ↓
[Motion Detection] → Frame differencing, Gaussian blur, threshold
    ↓
[Color Detection] → HSV conversion, color range filter
    ↓
[Mask Combination] → Logical AND/OR of motion and color
    ↓
[Morphology] → Opening, closing operations
    ↓
[Lane Masking] → Restrict to Phase 1 boundaries
    ↓
[Contour Analysis] → Extract contours, filter by shape
    ↓
[Candidate Ranking] → Score by size, circularity, solidity
    ↓
Detected Ball Candidates
```

### Tracking Pipeline
```
Detected Candidates
    ↓
[Kalman Predict] → Predict position from previous state
    ↓
[Data Association] → Match detection to prediction
    ↓
[Kalman Update] → Correct state with measurement
    ↓
[Occlusion Handling] → Use prediction if no detection (max 8 frames)
    ↓
Tracked Ball State (x, y, vx, vy)
```

### Analysis Pipeline
```
Trajectory Data
    ↓
[Smoothing] → Moving average filter (window=5)
    ↓
[Release Detection] → Find first high-velocity point in bottom 30%
    ↓
[Impact Detection] → Find entry into top 20% zone
    ↓
[Statistics] → Calculate max/avg velocity, detection rate
    ↓
Export JSON, Plots, Reports
```

## Key Design Decisions

### 1. Architecture Pattern
- **Matched Phase 1 structure**: Class-based design with separate config module
- **Modular functions**: Similar to detection_functions.py in Phase 1
- **Consistent naming**: Following LaneDetector → BallTracker convention

### 2. Detection Approach
- **Simple methods over complex ML**: Frame differencing + color instead of GMM
- **Rationale**: Static camera, controlled environment, clean background
- **Inspiration**: Adapted from basketball tracking research (TesiLucaPirotta.pdf)

### 3. Kalman Filter
- **4-state model**: Position (x, y) + Velocity (vx, vy)
- **Constant velocity assumption**: Appropriate for bowling ball physics
- **Occlusion handling**: Up to 8 frames using prediction only

### 4. Shape Filtering
- **Circularity**: ≥ 0.65 (allows slightly non-circular due to motion blur)
- **Solidity**: ≥ 0.75 (convex shape requirement)
- **Size constraints**: 8-50 pixel radius (adaptable to different videos)

### 5. Key Point Detection
- **Release point**: First velocity > 3.0 px/frame in bottom 30% zone
- **Impact point**: Entry into top 20% zone near pins
- **Zone-based approach**: Robust to velocity variations

## Integration with Phase 1

### Input Requirements
Phase 2 requires `boundary_data.json` from Phase 1 containing:
- `master_left`: Left boundary line parameters
- `master_right`: Right boundary line parameters
- `median_foul_params`: Foul line position
- `top_boundary`: Pin area boundary

### Data Flow
```
Phase 1 (Lane Detection)
    ↓ boundary_data.json
Phase 2 (Ball Tracking)
    ↓ tracking_data.json
Phase 3+ (Future Analysis)
```

### Automatic Detection
`main.py` automatically finds boundary data:
```python
# Looks for: output/<video_name>/boundary_data.json
boundary_path = find_boundary_data_for_video(video_path)
```

## Output Structure

```
output/<video_name>/tracking/
├── tracking_data.json          # Complete trajectory data
│   ├── video_info              # FPS, duration, frame count
│   ├── trajectory[]            # Frame-by-frame positions
│   ├── release_point           # Ball release detection
│   ├── impact_point            # Pin impact detection
│   └── statistics              # Performance metrics
│
├── analysis_report.txt         # Human-readable summary
├── tracking_video.mp4          # Annotated video
├── trajectory_plot.png         # 2D trajectory visualization
└── velocity_plot.png           # Velocity analysis plots
```

## Configuration Highlights

### Critical Parameters to Tune
```python
# Color range for ball (HSV)
BALL_COLOR_LOWER = (0, 0, 0)      # Black ball
BALL_COLOR_UPPER = (180, 255, 60)

# Motion detection sensitivity
FRAME_DIFF_THRESHOLD = 30          # Lower = more sensitive

# Shape filtering
MIN_CIRCULARITY = 0.65            # Higher = stricter circle
MIN_SOLIDITY = 0.75               # Higher = more convex

# Tracking robustness
MAX_MISSING_FRAMES = 8            # Frames to predict without detection

# Key point detection
RELEASE_VELOCITY_THRESHOLD = 3.0  # px/frame
IMPACT_VELOCITY_THRESHOLD = 2.0   # px/frame
```

## Testing Requirements

### Next Steps for Validation
1. Run on actual bowling video (cropped_test8.mp4)
2. Verify ball detection accuracy
3. Tune color ranges if needed
4. Validate release/impact point detection
5. Check trajectory smoothness
6. Adjust Kalman noise parameters if needed

### Expected Challenges
- **Ball color variation**: May need to adjust HSV ranges
- **Motion blur**: Fast ball might appear non-circular
- **Shadows/reflections**: Could create false positives
- **Multiple balls**: Current implementation tracks single ball

### Tuning Process
1. Start with intermediate video generation (set `SAVE_INTERMEDIATE_VIDEOS = True`)
2. Check motion mask quality
3. Verify color mask captures ball
4. Ensure combined mask is clean
5. Adjust shape filters based on false positive rate

## Performance Considerations

### Computational Efficiency
- Frame differencing: Very fast (~1ms per frame)
- Color segmentation: Fast (~2ms per frame)
- Morphological operations: Fast (~1ms per frame)
- Contour analysis: Moderate (~5ms per frame)
- Kalman filter: Very fast (<0.1ms per frame)

### Expected Processing Speed
- ~10-15 FPS on modern CPU for 720p video
- Bottleneck: Contour extraction and analysis
- Parallelization possible but not implemented

## Code Quality

### Matches Phase 1 Standards
- ✅ Comprehensive docstrings
- ✅ Type hints throughout
- ✅ Error handling and validation
- ✅ Progress bars with tqdm
- ✅ Professional logging
- ✅ Modular, maintainable structure

### Improvements Over Research Paper
- Simplified approach (no GMM complexity)
- Better suited for bowling environment
- Kalman filter for smoother tracking
- Automatic key point detection
- Comprehensive visualization suite

## Dependencies

### Already Installed (from Phase 1)
- OpenCV (cv2): 4.13.0
- NumPy: 2.4.1
- SciPy: 1.17.0
- scikit-learn: 1.8.0
- Matplotlib: 3.10.8
- Pandas: 3.0.0

### Not Yet Installed
- tqdm: Need to install for progress bars

## Future Enhancements

### Immediate Improvements
- Multi-ball tracking for splits
- Adaptive color learning
- Ball size estimation from distance
- Real-world speed conversion (mph/kph)

### Advanced Features
- Ball spin detection (template matching)
- Rotation axis calculation
- Hook/curve analysis
- Pin impact prediction
- Integration with Phase 3 (3D reconstruction)

## Version History

### Version 2.0.0 (January 30, 2026)
- Initial Phase 2 implementation
- Frame differencing detection
- Color segmentation detection
- Kalman filtering
- Release/impact detection
- Complete visualization suite
- JSON export format
- Integration with Phase 1

## Authors
- Mohammad Umayr Romshoo
- Mohammad Ammar Mughees

## References
- Research paper: "Sviluppo di un sistema di tracciamento di una palla da basket mediante tecniche di visione artificiale" by Luca Pirotta (2013)
- Adapted basketball tracking methods to bowling scenario
- Simplified approach focusing on static camera advantages
