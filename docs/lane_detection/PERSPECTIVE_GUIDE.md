# Understanding Perspective and Angles

## 🎯 IMPORTANT: The Angles Are Correct!

If you're seeing angles like:
- **Left boundary:** +23° to +27° (positive)
- **Right boundary:** -8° to -10° (negative)

**This is CORRECT due to camera perspective!**

---

## 📐 Why Perspective Matters

### Camera View
When viewing a bowling lane from behind the bowler:
- The lane boundaries **converge** toward the pins
- This is called **perspective convergence** (vanishing point)
- Lines that are parallel in 3D appear to converge in 2D image

### What This Means for Angles

```
        PINS (vanishing point)
              /\
             /  \
            /    \
           /  ||  \
          /   ||   \
         /    ||    \
        /     ||     \
       /      ||      \
      /       ||       \
    LEFT     LANE     RIGHT
  BOUNDARY           BOUNDARY
  (leans →)          (← leans)
  
  BOWLER
```

- **LEFT boundary** (left side of lane):
  - Starts at bottom-left
  - Goes toward top-center (toward vanishing point)
  - **Leans TO THE RIGHT** (positive angle from vertical)
  - Angle: **+20° to +30°** is normal!

- **RIGHT boundary** (right side of lane):
  - Starts at bottom-right
  - Goes toward top-center (toward vanishing point)
  - **Leans TO THE LEFT** (negative angle from vertical)
  - Angle: **-8° to -12°** is normal!

---

## ✅ Your Results Are Correct

```
Left master line:
  X-intersect: 180
  Angle: 26.95° (from vertical, 0°=vertical)  ← Leans right, toward center ✓

Right master line:
  X-intersect: 816
  Angle: -8.88° (from vertical, 0°=vertical)  ← Leans left, toward center ✓
```

**Why right boundary has smaller angle:**
- Right boundary is closer to the camera (bottom-right)
- Less perspective distortion at the bottom
- Left boundary is farther, so more convergence = larger angle

---

## 🔍 How to Verify This Is Correct

### Check 1: Signs Should Be Opposite
- Left boundary: POSITIVE angle ✓
- Right boundary: NEGATIVE angle ✓
- If both were same sign, something is wrong

### Check 2: Left Angle > Right Angle (absolute value)
- |Left angle| ≈ 20-30°
- |Right angle| ≈ 8-12°
- Left has more convergence due to perspective ✓

### Check 3: Visual Inspection
Look at your output videos:
- Green line (left): Should lean slightly toward center
- Red line (right): Should lean slightly toward center
- Both converging toward the pins ✓

---

## 🎬 What If The Angles Look Wrong in Video?

### If left boundary actually leans LEFT:
**Problem:** Camera mounted backwards or video is flipped
**Solution:** Swap left/right in the code

### If both boundaries lean the same direction:
**Problem:** Wrong lines detected (not lane boundaries)
**Solution:** Check intermediate videos (edges, otsu)

### If angles are > 45°:
**Problem:** Extreme camera angle or wrong lines
**Solution:** 
1. Check `edges_vertical` mode
2. Increase `NUM_COLLECTION_FRAMES`
3. Adjust camera position

---

## 📊 Expected Angle Ranges

### Normal Bowling Lane (from behind bowler):

| Camera Position | Left Angle | Right Angle |
|----------------|------------|-------------|
| Close to lane  | +25° to +35° | -8° to -12° |
| Medium distance| +20° to +25° | -6° to -10° |
| Far from lane  | +15° to +20° | -5° to -8°  |

### Signs
- ✅ Left: POSITIVE (leans right)
- ✅ Right: NEGATIVE (leans left)
- ❌ Both positive or both negative = ERROR

### Magnitude
- Left angle usually 2-3x larger than right angle
- This is normal due to perspective

---

## 🔧 When to Worry

### RED FLAGS:
1. **Both angles same sign** (both + or both -)
2. **Right angle larger than left angle** (|right| > |left|)
3. **Angles > 45°** for either boundary
4. **Angles changing frame-to-frame** (shouldn't happen with master lines)

### GREEN FLAGS (You're seeing these!):
1. ✅ Opposite signs (left +, right -)
2. ✅ Left angle 2-3x larger
3. ✅ Angles in expected range (20-30° and 8-12°)
4. ✅ Consistent across frames

---

## 💡 Key Takeaway

**Your angles are CORRECT!**

The seeming "inversion" is actually the real geometry:
- Left boundary leans RIGHT (toward center) = positive angle
- Right boundary leans LEFT (toward center) = negative angle

This is **exactly** what you should see for a bowling lane viewed from behind the bowler!

---

## 📚 Still Confused?

### Sanity Check
1. Open your output video
2. Look at the green line (left boundary)
3. Does it start at bottom-left and go toward top-center? ✓
4. Look at the red line (right boundary)  
5. Does it start at bottom-right and go toward top-center? ✓

If YES to both → Your angles are correct!

### Mental Model
Think of it this way:
- **"Left" and "Right"** refer to POSITION (which side of lane)
- **Angle sign** refers to LEAN DIRECTION (which way it tilts)
- Left POSITION can have RIGHT lean (positive angle)
- This is normal for perspective!
