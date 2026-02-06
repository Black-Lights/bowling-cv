# Phase 4: Pin Detection - Quick Start Guide

## ✅ Prerequisites Check

Before running Phase 4, ensure you have:

- [x] **Phase 1 Complete**: `boundary_data.json` exists in output directory
- [x] **Phase 2 Complete** (optional): `trajectory_data.json` for smart timing
- [x] **Original Video**: Located in `assets/input/`
- [x] **Dependencies Installed**: OpenCV, NumPy, Matplotlib

## 🚀 Running the Pipeline

### Method 1: Run Everything (Recommended)

```bash
# From project root
cd /mnt/project
python -m src.pin_detection.main
```

This will:
1. Create masked video with visible pins
2. Select before/after frames
3. Count pins using frame differencing
4. Generate all visualizations
5. Export results to JSON

### Method 2: Step-by-Step Testing

```bash
# Test individual modules

# Step 1: Test masking
python -m src.pin_detection.video_preprocessing

# Step 2: Test frame selection  
python -m src.pin_detection.frame_selector

# Step 3: Test pin counting
python -m src.pin_detection.pin_counter

# Step 4: Full pipeline
python -m src.pin_detection.main
```

### Method 3: Python API

```python
from src.pin_detection import detect_pins_in_video

# Detect pins
results = detect_pins_in_video('cropped_test3')

# Print results
print(f"Result: {results['result']}")
print(f"Remaining: {results['remaining_pins']}")
print(f"Toppled: {results['toppled_pins']}")
print(f"Confidence: {results['detection_confidence']:.1%}")
```

## 📁 Expected Output Structure

```
output/cropped_test3/pin_detection/
├── cropped_test3_pin_area_masked.mp4              # Masked video
├── cropped_test3_pin_area_masked_mask_visualization.png
├── cropped_test3_pin_detection_result.png         # Final result
├── cropped_test3_complete_comparison.png          # Full pipeline view
├── cropped_test3_pin_detection.json               # Results JSON
└── intermediate/
    ├── before_frame_15.png
    ├── after_frame_235.png
    ├── before_frame_15_roi.png
    ├── after_frame_235_roi.png
    ├── frame_selection_visualization.png
    ├── difference_pipeline.png
    ├── contour_detection.png
    └── detection_statistics.png
```

## 🎯 First Run Checklist

### 1. Verify Input Files

```bash
# Check video exists
ls assets/input/cropped_test3.mp4

# Check boundary data exists (from Phase 1)
ls output/cropped_test3/boundary_data.json

# Check trajectory data exists (from Phase 2, optional)
ls output/cropped_test3/ball_detection/cropped_test3_trajectory_data.json
```

### 2. Configure Parameters (Optional)

Edit `src/pin_detection/config.py`:

```python
# Frame selection
BEFORE_FRAME_OFFSET = 15         # Adjust if ball visible in "before" frame
AFTER_FRAME_OFFSET = -15         # Adjust if pins still flying

# Detection sensitivity
DIFFERENCE_THRESHOLD = 30        # Lower = more sensitive, Higher = less

# Contour filtering
MIN_PIN_AREA = 300              # Adjust based on pin size in your video
MAX_PIN_AREA = 8000

# Visualization (turn off for speed)
SAVE_INTERMEDIATE_FRAMES = True  # Set False to skip intermediate outputs
```

### 3. Run Pipeline

```bash
python -m src.pin_detection.main
```

### 4. Check Results

```bash
# View final result
xdg-open output/cropped_test3/pin_detection/cropped_test3_pin_detection_result.png

# View complete comparison
xdg-open output/cropped_test3/pin_detection/cropped_test3_complete_comparison.png

# View JSON results
cat output/cropped_test3/pin_detection/cropped_test3_pin_detection.json
```

## 🔧 Troubleshooting

### Issue: "Boundary data not found"

**Solution:** Run Phase 1 first
```bash
python main.py  # Run Phase 1 lane detection
```

### Issue: Detected 0 pins (but pins are visible)

**Solutions:**
1. Check masked video: Are pins visible?
   ```bash
   # View mask visualization
   xdg-open output/.../pin_area_masked_mask_visualization.png
   ```
2. Lower the threshold:
   ```python
   config.DIFFERENCE_THRESHOLD = 20  # Was 30
   ```
3. Check frame selection timing:
   ```bash
   # View frame selection
   xdg-open output/.../intermediate/frame_selection_visualization.png
   ```

### Issue: Detected 15+ pins (over-counting)

**Solutions:**
1. Increase minimum area:
   ```python
   config.MIN_PIN_AREA = 500  # Was 300
   ```
2. Check contour detection:
   ```bash
   # View contour visualization
   xdg-open output/.../intermediate/contour_detection.png
   ```
3. Increase morphology iterations:
   ```python
   config.MORPH_ITERATIONS = 3  # Was 2
   ```

### Issue: Pins not visible in masked video

**Solution:** Increase extension:
```python
config.PIN_AREA_UNMASK_EXTENSION = 150  # Was 100
```

## 📊 Understanding the Output

### JSON Results Structure

```json
{
  "video_name": "cropped_test3",
  "remaining_pins": 3,
  "toppled_pins": 7,
  "total_pins": 10,
  "result": "7 Pins Down",
  "is_strike": false,
  "is_spare": false,
  "detection_confidence": 0.95,
  "before_frame_index": 15,
  "after_frame_index": 235,
  "valid_contours": [
    {
      "pin_id": 1,
      "area": 1250.5,
      "bbox": [520, 180, 45, 65],
      "aspect_ratio": 0.69,
      "solidity": 0.87,
      "center": [542, 212]
    }
  ],
  "total_contours_found": 25,
  "detection_timestamp": "2026-02-06 14:30:00"
}
```

### Visualization Guide

1. **`pin_detection_result.png`**
   - Final result with pins numbered
   - Result banner at top
   - Use this for reports/presentations

2. **`complete_comparison.png`**
   - 6-panel view of entire pipeline
   - Best for understanding algorithm

3. **`difference_pipeline.png`**
   - Detailed differencing steps
   - Use for debugging threshold issues

4. **`contour_detection.png`**
   - All contours vs filtered
   - Use for tuning geometric filters

## ⚙️ Performance Tuning

### Speed Optimization (Production Mode)

```python
# In config.py
SAVE_INTERMEDIATE_FRAMES = False
CREATE_INTERMEDIATE_VIDEOS = False  
SAVE_DEBUG_PLOTS = False
VERBOSE = False
SHOW_PROGRESS_BAR = False
```

**Expected speed:** 2-3 seconds per video (down from 8-10s)

### Quality Optimization (Development Mode)

```python
# In config.py
SAVE_INTERMEDIATE_FRAMES = True
CREATE_INTERMEDIATE_VIDEOS = True
SAVE_DEBUG_PLOTS = True
VERBOSE = True
DEBUG_MODE = True
```

**Expected speed:** 10-15 seconds per video (with all visualizations)

## 📝 Next Steps

After successful detection:

1. **Validate Results**: Check if pin count matches visual inspection
2. **Tune Parameters**: Adjust thresholds if needed
3. **Test More Videos**: Run on different bowling videos
4. **Integrate**: Use results with other phases (e.g., combine with trajectory)
5. **Analyze**: Create statistics across multiple throws

## 🆘 Getting Help

If issues persist:

1. Check intermediate visualizations in `intermediate/` folder
2. Enable debug mode: `config.DEBUG_MODE = True`
3. Review the detailed algorithm explanation in `PHASE4_PIN_DETECTION_ALGORITHM.md`
4. Check module README: `src/pin_detection/README.md`

## ✅ Success Indicators

You know it's working when:

- ✅ Masked video shows all pins clearly
- ✅ Before frame has 10 standing pins
- ✅ After frame shows remaining pins after impact
- ✅ Pin count matches visual inspection
- ✅ Confidence > 90%
- ✅ Visualizations look reasonable

---

**Ready to detect pins? Run:** `python -m src.pin_detection.main`
