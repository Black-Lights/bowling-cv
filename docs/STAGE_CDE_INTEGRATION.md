# Stage C+D+E Integration: Unified Tracking-by-Detection Pipeline

**Document Version:** 2.0  
**Date:** February 2, 2026  
**Purpose:** Correct architectural design for integrating ROI tracking, blob filtering, and Kalman state estimation

---

## Problem Statement

### Current Implementation Issues ❌

1. **Wrong Processing Order:**
   - Current: Motion → Select contour → Update Kalman → Filter
   - Kalman filter updates from **raw motion** (unvalidated objects)
   - ROI boxes lock onto **any moving object** (bowler's head/hand/shoe)

2. **Stage D Applied Too Late:**
   - Blob filtering happens AFTER tracking decision
   - By the time filters run, Kalman already committed to wrong object

3. **ROI Used for Filtering Instead of Selection:**
   - ROI constrains what gets filtered
   - Should constrain which **validated candidate** gets selected

### Root Cause

**Violation of "Tracking-by-Detection" principle:** The tracker must only update from objects that have been **validated as ball-like**, not from raw motion.

---

## Correct Architecture ✅

### Processing Order

```
FOR EACH FRAME:
│
├─ Stage B: Motion Detection (Background Subtraction)
│  └─ Output: Binary mask of ALL moving pixels
│
├─ Stage D: Blob Filtering (FULL FRAME, no ROI yet)
│  ├─ Find ALL contours in motion mask
│  ├─ Apply filters to ALL contours:
│  │  ├─ Area Filter (perspective-aware)
│  │  ├─ Circularity Filter (≥ 0.65)
│  │  └─ Aspect Ratio Filter (< 2.0)
│  └─ Output: List of validated ball candidates
│
├─ Stage C: Search Strategy & Selection
│  ├─ Check tracking state → Determine search mode
│  ├─ IF NOT tracking (Global Search):
│  │  ├─ Type 1: Initial Search (ball not seen yet)
│  │  │  ├─ Exclude upper 30% of frame (bowler area)
│  │  │  └─ Select candidate closest to foul line
│  │  └─ Type 2: Re-activation Search (ball lost mid-lane)
│  │     ├─ Search only above last_known_y + safety_margin
│  │     └─ Select candidate closest to last known position
│  └─ ELSE (Local Tracking):
│     ├─ Kalman.predict() → Get expected position
│     ├─ Define ROI around prediction (dynamic size)
│     ├─ Filter candidates to those within ROI
│     └─ Select candidate closest to prediction
│
└─ Stage E: State Update
   ├─ IF valid candidate selected:
   │  ├─ Update Kalman filter with candidate position
   │  ├─ Reset lost_frames = 0
   │  ├─ Increment confirmation_counter
   │  └─ tracking_active = True
   └─ ELSE (no valid candidate):
      ├─ Use Kalman prediction (don't update)
      ├─ Increment lost_frames++
      └─ IF lost_frames > threshold:
         └─ Revert to Global Search
```

---

## Implementation Details

### 1. BallTracker Class Structure

```python
class BallTracker:
    def __init__(self, config, frame_width, frame_height, foul_line_y):
        # Kalman Filter
        self.kalman = BallKalmanFilter(...)
        
        # Tracking State
        self.tracking_active = False
        self.lost_frames = 0
        self.confirmation_counter = 0
        self.is_confirmed = False
        
        # Position History
        self.trajectory = []
        self.last_position = None
        self.last_known_y = None  # NEW: For re-activation search
        self.total_distance_traveled = 0.0
        
        # Search State
        self.search_type = 'initial'  # NEW: 'initial' or 'reactivation'
        
    def process_frame(self, frame, mask, frame_idx):
        """Main processing pipeline"""
        # Step 1: Filter ALL contours (Stage D)
        ball_candidates = self._filter_all_candidates(frame, mask)
        
        # Step 2: Select best candidate (Stage C)
        selected_candidate = self._select_candidate(ball_candidates)
        
        # Step 3: Update state (Stage E)
        self._update_state(selected_candidate, frame_idx)
        
        return self._build_result(ball_candidates, selected_candidate)
```

---

### 2. Stage D: Full-Frame Blob Filtering

**Method:** `_filter_all_candidates(frame, mask)`

```python
def _filter_all_candidates(self, frame, mask):
    """
    Apply Stage D filters to ALL contours in entire frame
    Returns only validated ball-like objects
    
    This runs BEFORE any tracking or ROI logic
    """
    # Find all contours in motion mask
    contours = find_ball_contours(mask, self.config)
    
    valid_candidates = []
    
    for contour in contours:
        cx, cy = contour['center']
        area = contour['area']
        
        # Filter 1: Area (perspective-aware)
        area_min, area_max = self._calculate_area_thresholds(cy)
        if not (area_min <= area <= area_max):
            continue
        
        # Filter 2: Circularity
        perimeter = cv2.arcLength(contour['contour'], True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < self.config.CIRCULARITY_THRESHOLD:
            continue
        
        # Filter 3: Aspect Ratio
        if len(contour['contour']) >= 5:
            ellipse = cv2.fitEllipse(contour['contour'])
            (minor, major) = ellipse[1]
            if minor > 0:
                aspect_ratio = major / minor
                if aspect_ratio > self.config.ASPECT_RATIO_MAX:
                    continue
        
        # Passed all filters!
        valid_candidates.append({
            'center': (cx, cy),
            'radius': contour['radius'],
            'area': area,
            'circularity': circularity,
            'contour': contour['contour']
        })
    
    return valid_candidates
```

**Key Points:**
- ✅ Processes ENTIRE frame
- ✅ No ROI constraints
- ✅ Returns clean list of ball-like objects
- ✅ Independent of tracking state

---

### 3. Stage C: Search Strategy & Selection

**Method:** `_select_candidate(ball_candidates)`

```python
def _select_candidate(self, ball_candidates):
    """
    Select best candidate from validated list
    Strategy depends on tracking state
    """
    if not ball_candidates:
        return None
    
    if not self.tracking_active:
        # GLOBAL SEARCH MODE
        return self._global_search_selection(ball_candidates)
    else:
        # LOCAL TRACKING MODE
        return self._local_tracking_selection(ball_candidates)
```

---

#### 3A. Global Search - Initial Type

**When:** Ball has never been detected yet  
**Method:** `_global_search_selection()` with `search_type == 'initial'`

```python
def _global_search_selection(self, ball_candidates):
    """
    Global search: Select from validated candidates
    Two types: initial (first detection) or reactivation (lost mid-lane)
    """
    if self.search_type == 'initial':
        # TYPE 1: Initial Search (ball not seen yet)
        # Focus: Near foul line, exclude bowler area
        
        # Exclude upper 30% of frame (bowler area)
        min_y_threshold = self.foul_line_y * 0.3
        candidates_in_search_zone = [
            c for c in ball_candidates 
            if c['center'][1] > min_y_threshold
        ]
        
        if not candidates_in_search_zone:
            return None
        
        # Score candidates - prioritize near foul line
        priority_zone_start = self.foul_line_y - self.config.FOUL_LINE_PRIORITY_ZONE
        
        scored = []
        for candidate in candidates_in_search_zone:
            cx, cy = candidate['center']
            score = 0.0
            
            # Higher Y (closer to foul line) = higher score
            if cy > priority_zone_start:
                proximity = (cy - priority_zone_start) / self.config.FOUL_LINE_PRIORITY_ZONE
                score += proximity * 100  # 0-100 points
            else:
                # Penalty for being far from foul line
                distance = (priority_zone_start - cy) / priority_zone_start
                score -= distance * 50  # Negative score
            
            scored.append((score, candidate))
        
        # Return highest scored candidate
        best_score, best_candidate = max(scored, key=lambda x: x[0])
        return best_candidate
    
    else:
        # TYPE 2: Reactivation search (see next section)
        return self._reactivation_search_selection(ball_candidates)
```

**Configuration:**
```python
# In config.py
FOUL_LINE_PRIORITY_ZONE = 200  # Pixels from foul line to prioritize
```

---

#### 3B. Global Search - Reactivation Type

**When:** Ball was being tracked but lost (e.g., partial occlusion, crossed pins)  
**Method:** `_reactivation_search_selection(ball_candidates)`

```python
def _reactivation_search_selection(self, ball_candidates):
    """
    TYPE 2: Re-activation Search
    Ball was lost mid-lane - search above last known position
    """
    if self.last_known_y is None:
        # Fallback to initial search if no history
        return self._global_search_selection(ball_candidates)
    
    # Define search zone: Above last known Y position
    # Add safety margin to account for uncertainty
    safety_margin = self.config.REACTIVATION_SEARCH_MARGIN  # e.g., 50 pixels
    max_y_search = self.last_known_y + safety_margin
    
    # Filter candidates to search zone (y < max_y_search)
    candidates_in_zone = [
        c for c in ball_candidates
        if c['center'][1] < max_y_search
    ]
    
    if not candidates_in_zone:
        return None
    
    # Select candidate closest to last known position
    last_x, last_y = self.last_position if self.last_position else (0, 0)
    
    def distance_to_last(candidate):
        cx, cy = candidate['center']
        return np.sqrt((cx - last_x)**2 + (cy - last_y)**2)
    
    best_candidate = min(candidates_in_zone, key=distance_to_last)
    return best_candidate
```

**Configuration:**
```python
# In config.py
REACTIVATION_SEARCH_MARGIN = 50  # Pixels below last known Y to search
```

**Example Scenario:**
```
Frame 150: Ball at y=300, tracking active
Frame 160: Ball hits pin, lost
Frame 161-165: No valid candidates, lost_frames++
Frame 166: lost_frames > threshold
           → Switch to reactivation search
           → Search only y < 350 (300 + 50)
           → Ignore foul line area (y > 600)
```

---

#### 3C. Local Tracking Selection

**When:** Ball is being actively tracked  
**Method:** `_local_tracking_selection(ball_candidates)`

```python
def _local_tracking_selection(self, ball_candidates):
    """
    Local tracking: Select validated candidate closest to Kalman prediction
    """
    if not self.kalman.initialized:
        return None
    
    # Get Kalman prediction
    prediction = self.kalman.predict()
    pred_x, pred_y = prediction['x'], prediction['y']
    
    # Calculate dynamic ROI size (perspective-aware)
    roi_buffer = calculate_roi_size(pred_y, self.config)
    
    # Define ROI box
    roi_x1 = max(0, pred_x - roi_buffer)
    roi_y1 = max(0, pred_y - roi_buffer)
    roi_x2 = min(self.frame_width, pred_x + roi_buffer)
    roi_y2 = min(self.frame_height, pred_y + roi_buffer)
    
    # Filter candidates to those within ROI
    candidates_in_roi = [
        c for c in ball_candidates
        if (roi_x1 <= c['center'][0] <= roi_x2 and
            roi_y1 <= c['center'][1] <= roi_y2)
    ]
    
    if not candidates_in_roi:
        # No validated candidates in ROI - tracking failure
        return None
    
    # Pick candidate closest to prediction
    def distance_to_prediction(candidate):
        cx, cy = candidate['center']
        return np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
    
    best_candidate = min(candidates_in_roi, key=distance_to_prediction)
    
    # Attach ROI info for visualization
    best_candidate['roi_box'] = (roi_x1, roi_y1, roi_x2, roi_y2)
    best_candidate['prediction'] = (pred_x, pred_y)
    
    return best_candidate
```

**ROI Size Calculation:**
```python
def calculate_roi_size(y_ball, config):
    """
    B_t = max(B_min, k * y_ball)
    Larger ROI near foul line, smaller near pins
    """
    dynamic_buffer = int(config.K_SCALE * y_ball)
    return max(config.B_MIN, dynamic_buffer)
```

---

### 4. Stage E: State Update

**Method:** `_update_state(selected_candidate, frame_idx)`

```python
def _update_state(self, selected_candidate, frame_idx):
    """
    Update Kalman filter and tracking state
    """
    if selected_candidate:
        # VALID CANDIDATE FOUND
        cx, cy = selected_candidate['center']
        
        # Update Kalman filter
        if not self.kalman.initialized:
            self.kalman.initialize(cx, cy)
        else:
            self.kalman.update(cx, cy)
        
        # Update tracking state
        self.tracking_active = True
        self.lost_frames = 0
        self.confirmation_counter += 1
        self.search_type = 'initial'  # Reset for next time
        
        # Track distance
        if self.last_position:
            dist = np.sqrt((cx - self.last_position[0])**2 + 
                          (cy - self.last_position[1])**2)
            self.total_distance_traveled += dist
        
        # Update position history
        self.last_position = (cx, cy)
        self.last_known_y = cy  # Store for potential reactivation
        self.trajectory.append((cx, cy))
        
        # Check confirmation
        if (self.confirmation_counter >= self.config.CONFIRMATION_THRESHOLD and
            self.total_distance_traveled >= self.config.SPATIAL_CONFIRMATION_DISTANCE):
            self.is_confirmed = True
            
    else:
        # NO VALID CANDIDATE FOUND
        self.lost_frames += 1
        
        # Use Kalman prediction as estimated position
        if self.kalman.initialized:
            prediction = self.kalman.predict()
            # Don't call update() - just use prediction
            self.trajectory.append((prediction['x'], prediction['y']))
        
        # Check if lost too long
        if self.lost_frames > self.config.MAX_LOST_FRAMES:
            # Ball lost - revert to global search
            self.tracking_active = False
            
            # Determine search type for next global search
            if self.is_confirmed and self.last_known_y is not None:
                # Ball was confirmed and we know where it was
                # Use reactivation search (above last position)
                self.search_type = 'reactivation'
            else:
                # Short track or never confirmed
                # Use initial search (near foul line)
                self.search_type = 'initial'
                self.confirmation_counter = 0
                self.total_distance_traveled = 0.0
```

---

## Configuration Parameters

### New Parameters to Add

```python
# In config.py

# ============================================
# STAGE C: SEARCH STRATEGY
# ============================================

# Initial Global Search (first detection)
FOUL_LINE_PRIORITY_ZONE = 200  # Pixels from foul line to prioritize

# Reactivation Global Search (lost mid-lane)
REACTIVATION_SEARCH_MARGIN = 50  # Pixels below last known Y to search
                                  # Prevents searching impossible zones

# Local Tracking ROI
B_MIN = 30                  # Minimum ROI buffer size (pixels)
K_SCALE = 0.15              # ROI scaling: B = max(B_min, k * y)

# State Management
MAX_LOST_FRAMES = 5         # Frames without detection before reverting to global
CONFIRMATION_THRESHOLD = 20 # Consecutive detections to confirm it's the ball
SPATIAL_CONFIRMATION_DISTANCE = 240  # Min travel distance to confirm (pixels)

# ============================================
# STAGE D: BLOB FILTERING
# ============================================

# Area Filter (perspective-aware)
AREA_MIN_AT_FOUL = 80       # Min area at foul line (px²)
AREA_MAX_AT_FOUL = 400      # Max area at foul line
AREA_MIN_AT_PINS = 10       # Min area at pins
AREA_MAX_AT_PINS = 50       # Max area at pins

# Shape Filters
CIRCULARITY_THRESHOLD = 0.65  # C = 4πA/P² (1.0 = perfect circle)
ASPECT_RATIO_MAX = 2.0        # Major/Minor axis ratio

# ============================================
# STAGE B: MOTION DETECTION
# ============================================

MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 40     # Increased to reduce noise
MOG2_DETECT_SHADOWS = True
SHADOW_THRESHOLD = 200
MORPH_KERNEL_SIZE = 3
MORPH_KERNEL_SHAPE = 'ellipse'
```

---

## What to Keep from Previous Implementation ✅

### Files to Keep (Mostly Unchanged)

1. **`mask_video.py`** - Lane masking logic
   - ✅ Working correctly
   - ✅ No changes needed

2. **`motion_detection.py`** - Background subtraction
   - ✅ MOG2 implementation correct
   - ✅ Update `MOG2_VAR_THRESHOLD` from 16 → 40

3. **`intermediate_visualization.py`** - Video generation utilities
   - ✅ Keep as-is
   - ✅ Used for all intermediate videos

### Code to Keep from Current Implementation

1. **`BallKalmanFilter` class** (roi_logic.py)
   - ✅ Prediction and update logic correct
   - ✅ No changes needed

2. **`calculate_roi_size()` function** (roi_logic.py)
   - ✅ Perspective-aware scaling correct
   - ✅ Keep as-is

3. **`create_roi_box()` function** (roi_logic.py)
   - ✅ Bounds checking correct
   - ✅ Keep as-is

4. **Blob filtering logic** (blob_analysis.py)
   - ✅ Area/Circularity/Aspect ratio calculations correct
   - ✅ Move into BallTracker class

---

## What to Change/Remove ❌

### Files to Refactor

1. **`roi_logic.py`**
   - ❌ Remove: `global_search()` function (wrong order)
   - ❌ Remove: `local_tracking()` function (wrong order)
   - ✅ Keep: `BallKalmanFilter` class
   - ✅ Keep: `calculate_roi_size()`, `create_roi_box()`
   - ✅ Refactor: `BallTracker.process_frame()` with new architecture

2. **`blob_analysis.py`**
   - ❌ Current: Separate blob analyzer run after tracking
   - ✅ New: Integrate into BallTracker (run before selection)
   - ✅ Keep: Filter calculation methods
   - ✅ Change: Make methods callable on frame-wide contour list

3. **`main.py`**
   - ❌ Remove: Separate Step 4 (ROI) and Step 5 (Blob) loops
   - ✅ New: Single unified loop with integrated tracker

### Specific Code to Remove

```python
# ❌ DELETE from roi_logic.py:
def global_search(mask, config, foul_line_y, prev_positions=None, max_y_boundary=None):
    # This function selects BEFORE filtering (wrong order)
    
def local_tracking(mask, roi_box, config):
    # This function filters in ROI only (wrong approach)
```

```python
# ❌ DELETE from main.py (Step 5 integration):
# Stage D: Blob filtering within ROI
if roi_result['roi_box'] is not None:
    roi_mask = np.zeros_like(denoised_mask)
    roi_mask[y1:y2, x1:x2] = denoised_mask[y1:y2, x1:x2]
    blob_metrics = analyzer.filter_blobs(frame, roi_mask, frame_idx)
```

---

## Expected Behavior After Refactoring

### Test Case: cropped_test6.mp4

```
Frames 0-16:   Global Search (Initial Type)
               - Motion detected (bowler moving)
               - Contours found: 20+ (hand, shoe, etc.)
               - Stage D filters to: 0-2 candidates
               - Selection: None in valid zone
               - Result: No tracking yet

Frame 17:      FIRST VALID BALL DETECTED
               - Contours: 15 total
               - Filtered to: 1 candidate (the ball at y=680)
               - Global search selects: Ball (nearest foul line)
               - Initialize Kalman
               - tracking_active = True

Frames 18-37:  Local Tracking
               - ROI around Kalman prediction
               - Contours: 5-10 (various objects)
               - Filtered to: 1-2 candidates
               - ROI filter: Only ball in ROI
               - Kalman updates successfully
               - Confirmation at frame 37 (20 frames, 203px)

Frames 38-39:  Occlusion (ball blocked briefly)
               - Filtered candidates: 0
               - Use Kalman prediction
               - lost_frames = 1, 2

Frame 40:      Ball reappears
               - Filtered: 1 candidate in ROI
               - Resume tracking
               - lost_frames = 0

Frames 41-250: Continued tracking with adaptive ROI

Frame 250:     Ball crosses pins, lost
               - lost_frames > MAX_LOST_FRAMES (5)
               - tracking_active = False
               - search_type = 'reactivation'
               - last_known_y = 213

Frames 251+:   Global Search (Reactivation Type)
               - Search only y < 263 (213 + 50)
               - Ignore foul line area
               - Looking for ball in pins area
```

---

## Implementation Checklist

### Phase 1: Preparation
- [ ] Create backup branch: `git checkout -b backup/stage-cde-before-refactor`
- [ ] Document current behavior with test videos
- [ ] Review all existing code in roi_logic.py and blob_analysis.py

### Phase 2: Refactor BallTracker
- [ ] Add blob filtering methods to BallTracker class
- [ ] Implement `_filter_all_candidates()` (Stage D)
- [ ] Implement `_select_candidate()` dispatcher
- [ ] Implement `_global_search_selection()` with initial/reactivation types
- [ ] Implement `_local_tracking_selection()`
- [ ] Implement `_update_state()`
- [ ] Add search_type and last_known_y tracking

### Phase 3: Update Configuration
- [ ] Add REACTIVATION_SEARCH_MARGIN parameter
- [ ] Update MOG2_VAR_THRESHOLD to 40
- [ ] Verify all Stage C/D parameters

### Phase 4: Simplify main.py
- [ ] Combine Step 4 (ROI) and Step 5 (Blob) into single loop
- [ ] BallTracker.process_frame() returns all info needed
- [ ] Update visualization to use integrated results

### Phase 5: Testing
- [ ] Test on cropped_test6.mp4
- [ ] Verify initial global search detects ball (not bowler)
- [ ] Verify local tracking maintains lock
- [ ] Verify reactivation search after ball lost mid-lane
- [ ] Test on cropped_test3.mp4, cropped_test7.mp4

### Phase 6: Validation
- [ ] Review ROI videos - should show ball tracking
- [ ] Review blob videos - should show filtered candidates
- [ ] Verify no bowler tracking in any video
- [ ] Document frame-by-frame behavior

---

## Success Criteria

✅ **The refactoring is successful if:**

1. **No bowler tracking**: ROI boxes never lock onto person's head/hand
2. **Ball detected at foul line**: Initial global search finds ball in frames 15-20
3. **Continuous tracking**: Local tracking maintains lock on ball for 200+ frames
4. **Proper reactivation**: After ball lost at pins, search zone restricted correctly
5. **Clean filtering**: Blob videos show only circular objects passing Stage D
6. **Correct Kalman updates**: Trajectory shows smooth ball motion, no jumps to bowler

---

## Visualization Videos

### Purpose

Generate intermediate videos at each stage to:
- **Debug** filtering and selection logic
- **Verify** correct objects are being tracked
- **Analyze** failure modes (why ball was lost)
- **Validate** reactivation search behavior

---

### Stage B: Motion Detection Videos (4 videos)

**Generated during background subtraction step**

1. **`foreground_mask.mp4`**
   - Raw MOG2 output (white=motion, gray=shadow, black=background)
   - Shows ALL moving pixels before cleaning
   - Purpose: Verify MOG2 is detecting ball movement

2. **`shadow_removed.mp4`**
   - After shadow thresholding (white=motion, black=background/shadow)
   - Purpose: Verify shadows aren't being tracked

3. **`denoised.mp4`**
   - After morphological opening (final clean mask)
   - Purpose: Verify noise removal, see what goes to Stage D
   - **This is the mask used for contour detection**

4. **`motion_comparison.mp4`**
   - Side-by-side: Original frame | Denoised mask
   - Purpose: Quick visual check of motion detection quality

**Configuration:**
```python
SAVE_FOREGROUND_MASK_VIDEO = True
SAVE_SHADOW_REMOVED_VIDEO = True
SAVE_DENOISED_VIDEO = True
SAVE_MOTION_COMPARISON_VIDEO = True
```

---

### Stage D: Blob Filtering Videos (7 videos)

**Generated from full-frame contour analysis**

5. **`blob_all_contours.mp4`**
   - ALL contours found in denoised mask (cyan outlines)
   - Shows everything before filtering
   - Purpose: See total candidate pool (ball + noise + bowler)
   - Typical count: 10-50 contours per frame

6. **`blob_area_filter.mp4`**
   - Color-coded by area filter result:
     - Green outline = PASS (within perspective-aware limits)
     - Red outline = FAIL (too large/small for Y position)
   - Displays area value on each contour
   - Shows perspective thresholds as text overlay
   - Purpose: Verify area limits are correct for each Y position

7. **`blob_circularity_filter.mp4`**
   - Color-coded by circularity:
     - Green = PASS (C ≥ 0.65, ball-like)
     - Red = FAIL (C < 0.65, elongated/irregular)
   - Displays circularity value (0.0-1.0) on each contour
   - Purpose: See which objects are circular (ball) vs irregular (hand/shoe)

8. **`blob_aspect_ratio_filter.mp4`**
   - Shows fitted ellipses on contours
   - Color-coded by aspect ratio:
     - Green = PASS (R < 2.0, roughly circular)
     - Red = FAIL (R ≥ 2.0, stretched/elongated)
   - Displays ratio value on each contour
   - Purpose: Verify motion blur doesn't over-elongate ball

9. **`blob_final_validated.mp4`**
   - **ONLY contours passing ALL filters** (green outlines)
   - This is the clean candidate list for Stage C
   - Purpose: Verify filtering quality
   - Typical count: 0-3 candidates per frame
   - **Critical: Should show ball, NOT bowler**

10. **`blob_filter_progression.mp4`**
    - Side-by-side comparison:
      - Left: All contours (cyan)
      - Right: Final validated (green)
    - Purpose: Quick before/after comparison

11. **`blob_full_pipeline.mp4`**
    - 2x3 grid showing all filter stages:
      ```
      [All Contours]  [Area Filter]     [Circularity]
      [Aspect Ratio]  [Final Validated] [Original Frame]
      ```
    - Purpose: Complete overview of filtering process

**Configuration:**
```python
SAVE_BLOB_ALL_CONTOURS_VIDEO = True
SAVE_BLOB_AREA_FILTER_VIDEO = True
SAVE_BLOB_CIRCULARITY_FILTER_VIDEO = True
SAVE_BLOB_ASPECT_RATIO_FILTER_VIDEO = True
SAVE_BLOB_FINAL_VALIDATED_VIDEO = True
SAVE_BLOB_FILTER_PROGRESSION_VIDEO = True
SAVE_BLOB_FULL_PIPELINE_VIDEO = True
```

---

### Stage C: Selection Strategy Videos (6 videos)

**Generated from candidate selection logic**

12. **`selection_global_initial.mp4`**
    - Shows initial global search behavior
    - Visualizations:
      - Exclusion zone (upper 30%, red overlay)
      - Search zone (lower 70%, green overlay)
      - All validated candidates (cyan circles)
      - Scored candidates (text: score value)
      - **Selected candidate (yellow circle, larger)**
      - Foul line priority zone (dashed line)
    - Purpose: Verify ball detected near foul line (not bowler)

13. **`selection_global_reactivation.mp4`**
    - Shows reactivation search behavior (ball lost mid-lane)
    - Visualizations:
      - Last known position (red X)
      - Search zone boundary (y < last_y + margin, green line)
      - Restricted zone (y > boundary, red overlay with "NO SEARCH")
      - All validated candidates (cyan circles)
      - Candidates in zone (white circles)
      - **Selected candidate (yellow circle)**
    - Purpose: Verify search restricted to reasonable zone

14. **`selection_local_tracking.mp4`**
    - Shows local tracking with ROI
    - Visualizations:
      - Kalman prediction (red crosshair)
      - ROI box (green rectangle, size adapts to Y)
      - All validated candidates (cyan circles, faint if outside ROI)
      - Candidates in ROI (white circles)
      - **Selected candidate (yellow circle)**
      - Distance from prediction (text)
    - Purpose: Verify ROI follows ball, not distracted by noise

15. **`selection_mode_comparison.mp4`**
    - Side-by-side when switching modes:
      - Left: Global search view
      - Right: Local tracking view
    - Shows transition at tracking activation/deactivation
    - Purpose: Understand mode switching behavior

16. **`selection_search_zones.mp4`**
    - Overlay showing search zone evolution over time
    - Heat map of where system is looking
    - Purpose: Visualize search strategy over entire video

17. **`selection_candidate_scores.mp4`**
    - For each frame, show score/distance for all candidates
    - Bar chart overlay showing why one was selected
    - Purpose: Debug selection logic

**Configuration:**
```python
SAVE_SELECTION_GLOBAL_INITIAL_VIDEO = True
SAVE_SELECTION_GLOBAL_REACTIVATION_VIDEO = True
SAVE_SELECTION_LOCAL_TRACKING_VIDEO = True
SAVE_SELECTION_MODE_COMPARISON_VIDEO = True
SAVE_SELECTION_SEARCH_ZONES_VIDEO = True
SAVE_SELECTION_CANDIDATE_SCORES_VIDEO = True
```

---

### Stage E: State Update Videos (5 videos)

**Generated from Kalman filter and state management**

18. **`state_kalman_tracking.mp4`**
    - Visualization:
      - Kalman prediction (red circle)
      - Actual detection (green circle, if found)
      - Line connecting prediction to detection
      - Prediction error distance (text)
      - State: tracking_active, lost_frames (text overlay)
    - Purpose: Verify Kalman predictions are accurate

19. **`state_trajectory.mp4`**
    - Shows ball trajectory over time
    - Visualizations:
      - Complete trajectory path (colored line)
      - Last N positions (dots, fading with age)
      - Current position (large dot)
      - Direction arrow (velocity vector)
      - Distance traveled (text)
    - Purpose: Verify smooth trajectory (no jumps to bowler)

20. **`state_confirmation.mp4`**
    - Shows confirmation progress
    - Visualizations:
      - Confirmation counter progress bar
      - Distance traveled progress bar
      - Frame-by-frame tracking status:
        - Gray: Not tracking
        - Yellow: Tracking but unconfirmed
        - Green: Confirmed tracking
      - Confirmation event marker (flash when confirmed)
    - Purpose: Verify ball confirmation logic

21. **`state_failure_recovery.mp4`**
    - Highlights tracking failures and recovery
    - Visualizations:
      - Lost frames counter (red when > 0)
      - Prediction-based position (orange circle when no detection)
      - Recovery event (green flash when detection resumes)
      - Mode transition markers (global ↔ local)
    - Purpose: Analyze occlusion handling

22. **`state_full_tracking_pipeline.mp4`**
    - Complete tracking state overlay
    - Shows all state information simultaneously:
      - Search mode (text: GLOBAL/LOCAL)
      - Tracking status (text: ACTIVE/LOST)
      - Confirmation status (text: CONFIRMED/UNCONFIRMED)
      - Kalman prediction + detection
      - ROI box (if local tracking)
      - Trajectory trail
    - Purpose: Complete diagnostic view

**Configuration:**
```python
SAVE_STATE_KALMAN_TRACKING_VIDEO = True
SAVE_STATE_TRAJECTORY_VIDEO = True
SAVE_STATE_CONFIRMATION_VIDEO = True
SAVE_STATE_FAILURE_RECOVERY_VIDEO = True
SAVE_STATE_FULL_TRACKING_PIPELINE_VIDEO = True
```

---

### Complete Pipeline Videos (3 videos)

**High-level overview videos**

23. **`pipeline_stage_progression.mp4`**
    - 3x2 grid showing stages:
      ```
      [Original]        [Motion Mask]      [Validated Candidates]
      [Selection View]  [Kalman Tracking]  [Final Trajectory]
      ```
    - Purpose: See entire pipeline at a glance

24. **`pipeline_split_screen.mp4`**
    - Top half: Original video
    - Bottom half: Diagnostic overlay (selected candidate, ROI, state)
    - Purpose: Quick comparison of input vs output

25. **`pipeline_debug_overlay.mp4`**
    - Original video with all debug info overlaid:
      - All validated candidates (cyan)
      - Selected candidate (yellow)
      - ROI box (green, if tracking)
      - Trajectory (blue line)
      - State text (corner overlay)
      - Frame counter, timestamps
    - Purpose: Single comprehensive video for presentations

**Configuration:**
```python
SAVE_PIPELINE_STAGE_PROGRESSION_VIDEO = True
SAVE_PIPELINE_SPLIT_SCREEN_VIDEO = True
SAVE_PIPELINE_DEBUG_OVERLAY_VIDEO = True
```

---

## Video Generation Strategy

### Minimal Set (Fast debugging)

For quick testing, generate only essential videos:

```python
# Stage D: Filtering
SAVE_BLOB_ALL_CONTOURS_VIDEO = True
SAVE_BLOB_FINAL_VALIDATED_VIDEO = True

# Stage C: Selection
SAVE_SELECTION_LOCAL_TRACKING_VIDEO = True

# Stage E: State
SAVE_STATE_KALMAN_TRACKING_VIDEO = True

# Complete
SAVE_PIPELINE_DEBUG_OVERLAY_VIDEO = True
```

**Total: 5 videos** - Enough to verify basic functionality

---

### Complete Set (Full analysis)

For thorough analysis and documentation:

```python
# Enable all videos (25 total)
# Good for initial testing, debugging failures, presentations
```

**Total: 25 videos** - Complete diagnostic suite

---

### Custom Sets by Problem Type

**Problem: Ball not detected initially**
- Enable: `blob_all_contours`, `blob_final_validated`, `selection_global_initial`
- Analyze: Are candidates being found? Are filters too strict?

**Problem: Tracking jumps to bowler**
- Enable: `blob_final_validated`, `selection_local_tracking`, `state_kalman_tracking`
- Analyze: Is bowler passing filters? Is ROI too large?

**Problem: Ball lost mid-lane**
- Enable: `state_failure_recovery`, `selection_global_reactivation`, `state_trajectory`
- Analyze: Why did tracking fail? Is reactivation search working?

**Problem: Kalman predictions inaccurate**
- Enable: `state_kalman_tracking`, `state_trajectory`
- Analyze: Are predictions drifting? Is process noise too high?

---

## Implementation Notes

### Video Naming Convention

```
{video_name}_{stage}_{type}.mp4

Examples:
- cropped_test6_motion_denoised.mp4
- cropped_test6_blob_final_validated.mp4
- cropped_test6_selection_local_tracking.mp4
- cropped_test6_state_trajectory.mp4
- cropped_test6_pipeline_debug_overlay.mp4
```

### Output Directory Structure

```
output/
└── {video_name}/
    └── ball_detection/
        ├── boundary_data.json
        ├── preprocessed_frames.npz
        └── intermediate/
            ├── Stage B (Motion) - 4 videos
            │   ├── {video}_motion_foreground_mask.mp4
            │   ├── {video}_motion_shadow_removed.mp4
            │   ├── {video}_motion_denoised.mp4
            │   └── {video}_motion_comparison.mp4
            │
            ├── Stage D (Filtering) - 7 videos
            │   ├── {video}_blob_all_contours.mp4
            │   ├── {video}_blob_area_filter.mp4
            │   ├── {video}_blob_circularity_filter.mp4
            │   ├── {video}_blob_aspect_ratio_filter.mp4
            │   ├── {video}_blob_final_validated.mp4
            │   ├── {video}_blob_filter_progression.mp4
            │   └── {video}_blob_full_pipeline.mp4
            │
            ├── Stage C (Selection) - 6 videos
            │   ├── {video}_selection_global_initial.mp4
            │   ├── {video}_selection_global_reactivation.mp4
            │   ├── {video}_selection_local_tracking.mp4
            │   ├── {video}_selection_mode_comparison.mp4
            │   ├── {video}_selection_search_zones.mp4
            │   └── {video}_selection_candidate_scores.mp4
            │
            ├── Stage E (State) - 5 videos
            │   ├── {video}_state_kalman_tracking.mp4
            │   ├── {video}_state_trajectory.mp4
            │   ├── {video}_state_confirmation.mp4
            │   ├── {video}_state_failure_recovery.mp4
            │   └── {video}_state_full_tracking_pipeline.mp4
            │
            └── Complete Pipeline - 3 videos
                ├── {video}_pipeline_stage_progression.mp4
                ├── {video}_pipeline_split_screen.mp4
                └── {video}_pipeline_debug_overlay.mp4
```

### Performance Considerations

**Video generation is CPU-intensive!**

- Each frame needs encoding (FFmpeg)
- 25 videos × 346 frames = 8,650 encode operations
- Estimated time: 5-10 minutes for all videos

**Optimization strategies:**
1. Generate only needed videos (use custom sets)
2. Parallelize video encoding (if CPU has multiple cores)
3. Lower video quality/resolution for debugging
4. Skip frames (e.g., save every 5th frame for faster preview)

---

## Expected Video Outputs for Test Case

### cropped_test6.mp4 (346 frames)

**Frames 0-16: Initial Search Phase**
- `selection_global_initial.mp4`: Should show search zone near foul line, no candidates selected yet
- `blob_final_validated.mp4`: Should show 0-1 candidates (no ball visible yet)

**Frame 17: First Detection**
- `selection_global_initial.mp4`: Yellow circle on ball at y≈680
- `state_kalman_tracking.mp4`: Kalman initialized (green flash)
- `state_confirmation.mp4`: Counter starts incrementing

**Frames 18-37: Confirmation Phase**
- `selection_local_tracking.mp4`: ROI box appears, follows ball
- `state_trajectory.mp4`: Trajectory line starts forming
- `state_confirmation.mp4`: Progress bars filling up
- **Frame 37**: Green flash (CONFIRMED)

**Frames 38-39: Brief Occlusion**
- `state_failure_recovery.mp4`: Lost frames counter = 1, 2
- `state_kalman_tracking.mp4`: Red prediction circle (no green detection)

**Frames 40-250: Continuous Tracking**
- `selection_local_tracking.mp4`: ROI box shrinks as ball moves toward pins
- `state_trajectory.mp4`: Long trajectory line visible
- `blob_final_validated.mp4`: Should show 1-2 candidates consistently

**Frame 250: Ball Lost**
- `state_failure_recovery.mp4`: Lost frames counter exceeds threshold
- Mode switches to GLOBAL (reactivation)

**Frames 251+: Reactivation Search**
- `selection_global_reactivation.mp4`: Search zone restricted to y < 263
- Shows red overlay on lower frame (foul line area = NO SEARCH)

---

## References

- Current Implementation: `roi_logic.py` (lines 322-544)
- Blob Analysis: `blob_analysis.py` (full file)
- Main Pipeline: `main.py` (lines 130-230)
- Configuration: `config.py` (Stage C/D sections)
