# Understanding Angles in Lane Detection

## Angle Calculation Modes

The system supports two modes for calculating line angles. The mode is controlled by `USE_ABSOLUTE_ANGLES` in `config.py`.

### Mode 1: From Horizontal (USE_ABSOLUTE_ANGLES = False)

**Traditional mathematical convention:**
- 0° = horizontal line pointing right
- 90° = vertical line pointing up
- -90° = vertical line pointing down

**For lane boundaries (nearly vertical lines):**
- Left boundary: approximately -63° to -70°
- Right boundary: approximately 80° to 82°

**Issues:**
- Confusing for vertical lines
- Negative angles are unintuitive
- Small changes in angle are hard to interpret

### Mode 2: From Vertical (USE_ABSOLUTE_ANGLES = True) ⭐ RECOMMENDED

**Intuitive convention for vertical lines:**
- 0° = perfectly vertical line
- 90° = horizontal line
- Positive angle = leans to the right
- Negative angle = leans to the left

**For lane boundaries (nearly vertical lines):**
- Left boundary: approximately -2° to -5° (leans slightly left)
- Right boundary: approximately 8° to 10° (leans slightly right)

**Advantages:**
- Easy to understand: "3° from vertical"
- Small angles mean nearly vertical
- Sign indicates lean direction

---

## Expected Angle Ranges

### Perfectly Aligned Lane
- Left boundary: -3° to -1° (slight leftward lean)
- Right boundary: 1° to 3° (slight rightward lean)

### Typical Camera Perspective
Due to camera angle and perspective:
- Left boundary: -5° to -2°
- Right boundary: 5° to 10°
- Greater right lean due to perspective distortion

### Warning Signs
If you see angles like:
- **> 15°**: Very tilted, check camera position
- **> 30°**: Likely detection error
- **Opposite signs**: Lines crossing (wrong boundaries selected)

---

## Interpreting Your Results

### Example Output (Mode 2):
```
Left master line:
  X-intersect: 180
  Angle: -3.5° (from vertical, 0°=vertical)
  
Right master line:
  X-intersect: 816
  Angle: 8.2° (from vertical, 0°=vertical)
```

**Interpretation:**
- Left boundary leans 3.5° to the left of vertical ✓
- Right boundary leans 8.2° to the right of vertical ✓
- This is normal for bowling lanes viewed from typical camera angle

---

## Troubleshooting Angles

### Problem: Angles seem backwards
**Solution:** Lines may be detected in reverse direction
- The system normalizes line direction (bottom to top)
- Angles should be consistent across frames

### Problem: Angles too large (> 20°)
**Possible causes:**
1. Camera is tilted
2. Wrong lines detected (not lane boundaries)
3. Perspective distortion is extreme

**Solutions:**
1. Check intermediate videos (`edges_vertical`, `otsu_vertical`)
2. Adjust `ANGLE_TOLERANCE` in config.py
3. Increase `NUM_COLLECTION_FRAMES` for more stable results

### Problem: Angles changing frame-to-frame
**This shouldn't happen with master lines!**
- Master lines are static (same for all frames)
- If you see changing angles, the system isn't using master lines correctly

---

## Visualization in Plots

### Bin Analysis Plots
The angle distribution histogram shows:
- X-axis: Angle values
- Y-axis: Number of lines at that angle
- Red dashed line: Median angle (selected)
- Green bars: Lines within ±3° tolerance

**Good distribution:**
- Narrow peak around median
- Most lines within ±3°
- Clear consensus

**Poor distribution:**
- Multiple peaks (conflicting lines)
- Wide spread (> 10° range)
- Solution: Increase NUM_COLLECTION_FRAMES

---

## Converting Between Modes

If you need to convert:

**From horizontal to vertical:**
```
angle_vertical = 90° - abs(angle_horizontal)
if angle_horizontal < 0:
    angle_vertical = -angle_vertical
```

**From vertical to horizontal:**
```
angle_horizontal = 90° - abs(angle_vertical)
if angle_vertical < 0:
    angle_horizontal = -angle_horizontal
```

---

## Best Practices

1. **Always use Mode 2 (from_vertical)** for lane boundaries
2. **Expected range**: -10° to +10° for typical setups
3. **Check bin analysis plots** to verify angle consistency
4. **Compare across videos** using summary plots
5. **If angles look wrong**, check intermediate videos first

---

## Quick Reference Card

```
┌─────────────────────────────────────────┐
│  ANGLE MODE: From Vertical (Recommended)│
├─────────────────────────────────────────┤
│  0° = Perfectly Vertical                │
│  Positive = Leans Right                 │
│  Negative = Leans Left                  │
├─────────────────────────────────────────┤
│  TYPICAL VALUES:                        │
│  Left:  -5° to -2°                      │
│  Right:  5° to 10°                      │
├─────────────────────────────────────────┤
│  WARNING if > 15°                       │
│  ERROR if > 30°                         │
└─────────────────────────────────────────┘
```
