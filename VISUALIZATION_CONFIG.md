# Visualization Configuration Guide

## Location
All visualization settings are in: `src/spin_analysis/config.py`

## Master Controls

```python
SAVE_PLOTS = True                   # Master switch for all plot saving
PLOT_DPI = 150                      # Resolution for saved plots (150 = high quality)
SAVE_DEBUG_IMAGES = True            # Save intermediate debug visualizations
```

## Stage-Specific Visualization Flags

### Stage A: Data Preparation
```python
GENERATE_STAGE_A_PLOT = False       # Trajectory validation plot
```
**Output:** `stage_a_trajectory_validation.png`
- X/Y position over time
- Radius evolution
- Trajectory path

### Stage B: Optical Flow
```python
GENERATE_STAGE_B_PLOT = True        # Optical flow test visualization
GENERATE_STAGE_B_SUMMARY = True     # Feature count summary plot
```
**Outputs:**
- `stage_b_optical_flow_test.png` - Sample frames with feature tracking
- `stage_b_feature_count_summary.png` - Tracked vs good features over time

### Stage C: 3D Projection
```python
GENERATE_STAGE_C_PLOT = True        # 3D projection scatter plots
STAGE_C_NUM_SAMPLE_FRAMES = 5       # Number of sample frames to visualize
STAGE_C_SHOW_STATISTICS = True      # Include statistics panel
```
**Output:** `stage_c_3d_projection.png`
- 3D scatter plots (X, Y, Z coordinates on sphere)
- Z-coordinate distributions
- Projection statistics (success rate, radius validation)

### Stage D: Rotation Analysis (Coming Soon)
```python
GENERATE_STAGE_D_PLOT = True        # Rotation analysis visualization
```
**Output:** `stage_d_rotation_analysis.png`
- Rotation axis vectors over time
- Angular velocity plots
- RPM calculation

## Performance vs Quality Trade-offs

### Minimal Output (Fastest)
```python
GENERATE_STAGE_A_PLOT = False
GENERATE_STAGE_B_PLOT = False
GENERATE_STAGE_B_SUMMARY = False
GENERATE_STAGE_C_PLOT = False
STAGE_C_NUM_SAMPLE_FRAMES = 3
STAGE_C_SHOW_STATISTICS = False
```

### Debug Mode (Full Details)
```python
GENERATE_STAGE_A_PLOT = True
GENERATE_STAGE_B_PLOT = True
GENERATE_STAGE_B_SUMMARY = True
GENERATE_STAGE_C_PLOT = True
STAGE_C_NUM_SAMPLE_FRAMES = 10
STAGE_C_SHOW_STATISTICS = True
```

### Production (Balanced)
```python
GENERATE_STAGE_A_PLOT = False       # Skip trajectory validation
GENERATE_STAGE_B_PLOT = True        # Keep optical flow debug
GENERATE_STAGE_B_SUMMARY = True     # Keep summary metrics
GENERATE_STAGE_C_PLOT = True        # Keep 3D validation
STAGE_C_NUM_SAMPLE_FRAMES = 5       # Balance detail vs size
STAGE_C_SHOW_STATISTICS = True      # Keep statistics
```

## File Locations

All visualizations are saved to:
```
output/{video_name}/spin_analysis/debug/
├── stage_a_trajectory_validation.png     (if GENERATE_STAGE_A_PLOT = True)
├── stage_b_optical_flow_test.png          (if GENERATE_STAGE_B_PLOT = True)
├── stage_b_feature_count_summary.png     (if GENERATE_STAGE_B_SUMMARY = True)
└── stage_c_3d_projection.png             (if GENERATE_STAGE_C_PLOT = True)
```

## Tips

1. **Development:** Enable all visualizations to debug pipeline
2. **Batch Processing:** Disable Stage A/B plots, keep Stage C for validation
3. **Storage:** Each full visualization ~1MB, adjust `STAGE_C_NUM_SAMPLE_FRAMES` to reduce
4. **Quality:** Increase `PLOT_DPI` to 300 for publication-quality images
