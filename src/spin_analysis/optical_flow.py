"""
Stage B: Optical Flow Detection

Tracks ball surface texture features using Lucas-Kanade optical flow.
Includes forward-backward consistency validation.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional


def roi_bounds(center: Tuple[float, float], radius: float, frame_shape: Tuple[int, int], offset: int) -> Tuple[int, int, int, int]:
    """
    Calculate Region of Interest bounds around the ball.
    
    Args:
        center: (x, y) center coordinates
        radius: Ball radius in pixels
        frame_shape: (height, width) of frame
        offset: Extra padding around ball
    
    Returns:
        tuple: (x_min, x_max, y_min, y_max) ROI bounds
    """
    x_min = max(int(center[0] - radius - offset), 0)
    x_max = min(int(center[0] + radius + offset), frame_shape[1])
    y_min = max(int(center[1] - radius - offset), 0)
    y_max = min(int(center[1] + radius + offset), frame_shape[0])
    return x_min, x_max, y_min, y_max


def enhance_texture(gray_image: np.ndarray) -> np.ndarray:
    """
    Enhance ball texture using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    This improves visibility of subtle patterns like oil rings and logos.
    
    Args:
        gray_image: Grayscale image
    
    Returns:
        Enhanced grayscale image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def compute_optical_flow(gray1: np.ndarray, gray2: np.ndarray, mask: np.ndarray, 
                         config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage B: Compute optical flow between two frames using Lucas-Kanade method.
    
    Includes forward-backward consistency check for validation.
    
    Args:
        gray1: First grayscale image
        gray2: Second grayscale image
        mask: Circular mask for feature detection
        config: Configuration module
    
    Returns:
        tuple: (p0, p1, p0r, status_forward, status_backward, fb_error)
            - p0: Initial feature points in frame1
            - p1: Tracked points in frame2 (forward)
            - p0r: Back-tracked points in frame1 (backward)
            - status_forward: Tracking status for forward flow
            - status_backward: Tracking status for backward flow
            - fb_error: Forward-backward error for each point
    
    Raises:
        ValueError: If no features are detected
    """
    # Detect corner features in first frame
    p0 = cv2.goodFeaturesToTrack(gray1, mask=mask, **config.FEATURE_PARAMS)
    
    if p0 is None or len(p0) == 0:
        raise ValueError("No features detected. Ball may be too uniform or blurry.")
    
    # Track features forward: frame1 -> frame2
    p1, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, p0, None, **config.LK_PARAMS
    )
    
    # Track features backward: frame2 -> frame1 (for validation)
    p0r, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        gray2, gray1, p1, None, **config.LK_PARAMS
    )
    
    # Calculate forward-backward error
    # Good tracks should return to original position
    fb_error = np.linalg.norm(p0 - p0r, axis=2)
    
    return p0, p1, p0r, status_forward, status_backward, fb_error


def filter_optical_flow_features(
    p0: np.ndarray, p1: np.ndarray, p0r: np.ndarray,
    status_forward: np.ndarray, status_backward: np.ndarray, fb_error: np.ndarray,
    ball_radius: float, center_roi: np.ndarray, config
) -> Tuple[List[Tuple], List[Tuple], int]:
    """
    Stage B: Filter tracked features based on quality criteria.
    
    Filtering criteria:
    1. Spatial constraint - feature must be within conservative radius from center
    2. Forward-backward consistency (fb_error < threshold)
    3. Both tracking directions succeeded
    4. Movement within reasonable range (not too small, not too large)
    
    Args:
        p0: Initial 2D feature points
        p1: Tracked 2D feature points
        p0r: Back-tracked 2D feature points
        status_forward: Forward tracking status
        status_backward: Backward tracking status
        fb_error: Forward-backward error
        ball_radius: Ball radius for movement thresholds
        center_roi: Ball center in ROI coordinates (for spatial filtering)
        config: Configuration module
    
    Returns:
        tuple: (good_features, rejected_features, num_good)
            - good_features: List of (old_pt, new_pt) tuples
            - rejected_features: List of (old_pt, new_pt, reason) tuples
            - num_good: Count of good features
    """
    # Define movement thresholds
    movement_threshold = ball_radius  # Maximum expected movement
    
    # Calculate minimum movement threshold using exponential decay
    if config.USE_EXPONENTIAL_THRESHOLD:
        # Exponential decay based on ball size
        # Normalize ball radius to [0, 1] range (assuming max radius ~60px, min ~10px)
        max_radius = 60.0
        min_radius = 10.0
        radius_normalized = (ball_radius - min_radius) / (max_radius - min_radius)
        radius_normalized = max(0.0, min(1.0, radius_normalized))  # Clamp to [0, 1]
        
        # Apply exponential decay: threshold = base * (normalized_radius ^ decay_power)
        low_movement_threshold = config.BASE_MOVEMENT_THRESHOLD * (radius_normalized ** config.MOVEMENT_THRESHOLD_DECAY)
        
        # Ensure minimum threshold of 0.5px for very small balls
        low_movement_threshold = max(0.5, low_movement_threshold)
    else:
        # Legacy linear threshold
        low_movement_threshold = ball_radius / config.LOW_THRESHOLD_FACTOR
    
    # Spatial constraint: features must be within this radius from center
    # Use mask radius to ensure we only track features well inside the ball
    max_distance_from_center = ball_radius * config.BALL_MASK_RADIUS_FACTOR
    
    good_features = []
    rejected_features = []
    
    for old_pt, new_pt, p0r_pt, s1, s2, err in zip(
        p0.reshape(-1, 2),
        p1.reshape(-1, 2),
        p0r.reshape(-1, 2),
        status_forward.ravel(),
        status_backward.ravel(),
        fb_error.ravel()
    ):
        # Filter 0: Check if feature is within safe distance from ball center
        # This prevents ground features from being tracked
        distance_from_center = np.linalg.norm(old_pt - center_roi)
        if distance_from_center > max_distance_from_center:
            rejected_features.append((old_pt, new_pt, f'outside_ball:{distance_from_center:.1f}'))
            continue
        
        # Filter 1: Check tracking quality
        if not (s1 and s2):
            rejected_features.append((old_pt, new_pt, 'tracking_failed'))
            continue
        
        if err >= config.FB_ERROR_THRESHOLD:
            rejected_features.append((old_pt, new_pt, f'fb_error:{err:.1f}'))
            continue
        
        # Filter 2: Check movement magnitude
        displacement = np.linalg.norm(new_pt - old_pt)
        
        if displacement <= low_movement_threshold:
            rejected_features.append((old_pt, new_pt, f'too_small:{displacement:.1f}'))
            continue
        
        if displacement >= movement_threshold:
            rejected_features.append((old_pt, new_pt, f'too_large:{displacement:.1f}'))
            continue
        
        # Feature passed all checks
        good_features.append((old_pt, new_pt))
    
    return good_features, rejected_features, len(good_features)


def test_optical_flow_on_frame(
    video_path: str,
    trajectory_df: pd.DataFrame,
    test_frame: int,
    config
) -> Optional[dict]:
    """
    Stage B: Test optical flow detection on a specific frame pair.
    
    Args:
        video_path: Path to video file
        trajectory_df: Trajectory DataFrame
        test_frame: Frame number to test (uses frame and frame+1)
        config: Configuration module
    
    Returns:
        dict: Results with images and statistics, or None if frame not available
    """
    # Check if both frames exist in trajectory
    frame_data = trajectory_df[trajectory_df['frame'] == test_frame]
    next_frame_data = trajectory_df[trajectory_df['frame'] == test_frame + 1]
    
    if len(frame_data) == 0 or len(next_frame_data) == 0:
        if config.VERBOSE:
            print(f"  ⚠ Frame {test_frame} or {test_frame + 1} not in trajectory, skipping")
        return None
    
    # Get ball positions
    ball_center1 = np.array([frame_data['x'].iloc[0], frame_data['y'].iloc[0]])
    ball_center2 = np.array([next_frame_data['x'].iloc[0], next_frame_data['y'].iloc[0]])
    ball_radius = frame_data['radius'].iloc[0]
    
    # Open video and read frames
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    cap.release()
    
    if not (ret1 and ret2):
        if config.VERBOSE:
            print(f"  ⚠ Could not read frame {test_frame}, skipping")
        return None
    
    # Extract ROI around ball
    x_min1, x_max1, y_min1, y_max1 = roi_bounds(
        ball_center1, ball_radius, frame1.shape[:2], config.ROI_OFFSET
    )
    x_min2, x_max2, y_min2, y_max2 = roi_bounds(
        ball_center2, ball_radius, frame2.shape[:2], config.ROI_OFFSET
    )
    
    roi1 = frame1[y_min1:y_max1, x_min1:x_max1].copy()
    roi2 = frame2[y_min2:y_max2, x_min2:x_max2].copy()
    
    # Convert to grayscale and enhance
    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    
    gray1_enhanced = enhance_texture(gray1)
    gray2_enhanced = enhance_texture(gray2)
    
    # Calculate ball center in ROI coordinates
    center_roi1 = ball_center1 - np.array([x_min1, y_min1])
    center_roi2 = ball_center2 - np.array([x_min2, y_min2])
    
    # Create circular mask (50% of ball radius)
    mask = np.zeros_like(gray1)
    cv2.circle(
        mask, 
        tuple(center_roi1.astype(int)), 
        int(ball_radius * config.BALL_MASK_RADIUS_FACTOR), 
        255, 
        -1
    )
    
    # Apply HSV color filtering if enabled (remove brown lane and white pins)
    if config.USE_HSV_FILTERING:
        # Convert ROI to HSV
        hsv1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
        
        # Create mask for brown lane
        mask_brown = cv2.inRange(hsv1, 
                                 np.array(config.HSV_LOWER_BROWN), 
                                 np.array(config.HSV_UPPER_BROWN))
        
        # Create mask for white pins
        mask_white = cv2.inRange(hsv1, 
                                 np.array(config.HSV_LOWER_WHITE), 
                                 np.array(config.HSV_UPPER_WHITE))
        
        # Combine masks: exclude brown and white areas
        # Invert: 255 where we DON'T want features (brown or white)
        mask_background = cv2.bitwise_or(mask_brown, mask_white)
        
        # Keep only dark ball areas: invert background mask
        mask_ball_color = cv2.bitwise_not(mask_background)
        
        # Combine circular mask with color mask (both must be satisfied)
        mask = cv2.bitwise_and(mask, mask_ball_color)
    
    # Compute optical flow
    try:
        p0, p1, p0r, status_f, status_b, fb_error = compute_optical_flow(
            gray1_enhanced, gray2_enhanced, mask, config
        )
        
        # Filter features
        good_features, rejected_features, num_good = filter_optical_flow_features(
            p0, p1, p0r, status_f, status_b, fb_error, ball_radius, center_roi1, config
        )
        
        # Prepare result
        result = {
            'frame': test_frame,
            'ball_center1': ball_center1,
            'ball_center2': ball_center2,
            'ball_radius': ball_radius,
            'roi1': roi1,
            'roi2': roi2,
            'gray1': gray1,
            'gray2': gray2,
            'gray1_enhanced': gray1_enhanced,
            'gray2_enhanced': gray2_enhanced,
            'center_roi1': center_roi1,
            'center_roi2': center_roi2,
            'roi_offset1': np.array([x_min1, y_min1]),  # Store offsets for coordinate conversion
            'roi_offset2': np.array([x_min2, y_min2]),
            'mask': mask,
            'total_features': len(p0),
            'good_features': good_features,
            'rejected_features': rejected_features,
            'num_good': num_good,
            'fb_errors': fb_error.ravel()
        }
        
        if config.VERBOSE:
            print(f"  [OK] Frame {test_frame}: {num_good}/{len(p0)} features passed filtering")
        
        return result
        
    except ValueError as e:
        if config.VERBOSE:
            print(f"  ✗ Frame {test_frame}: {e}")
        return None


def visualize_optical_flow_test(results: List[dict], output_dir: Path, config):
    """
    Stage B: Create visualization for optical flow test results.
    
    Shows for each test frame:
    - Original ROI with ball boundary
    - Enhanced ROI with feature mask
    - Optical flow vectors
    - Forward-backward error distribution
    
    Args:
        results: List of test results from test_optical_flow_on_frame
        output_dir: Output directory for plot
        config: Configuration module
    """
    num_results = len(results)
    if num_results == 0:
        print("⚠ No successful optical flow tests to visualize")
        return
    
    fig = plt.figure(figsize=(18, 5 * num_results))
    
    for idx, result in enumerate(results):
        if result is None:
            continue
        
        base_row = idx * 2
        
        # Row 1: Original and Enhanced ROIs
        # Original frame 1 with ball boundary
        ax1 = plt.subplot(num_results * 2, 4, base_row * 4 + 1)
        roi_display = result['roi1'].copy()
        # Draw actual ball radius (yellow)
        cv2.circle(roi_display, tuple(result['center_roi1'].astype(int)), 
                  int(result['ball_radius']), (0, 255, 255), 2)
        # Draw conservative mask radius (green)
        cv2.circle(roi_display, tuple(result['center_roi1'].astype(int)), 
                  int(result['ball_radius'] * config.BALL_MASK_RADIUS_FACTOR), (0, 255, 0), 2)
        ax1.imshow(cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Frame {result['frame']}\nYellow=Ball, Green=Mask", fontsize=10)
        ax1.axis('off')
        
        # Enhanced with mask
        ax2 = plt.subplot(num_results * 2, 4, base_row * 4 + 2)
        masked_enhanced = cv2.bitwise_and(result['gray1_enhanced'], result['mask'])
        ax2.imshow(masked_enhanced, cmap='gray')
        ax2.set_title(f"Enhanced + Mask\n{result['total_features']} features detected", fontsize=10)
        ax2.axis('off')
        
        # Optical flow visualization
        ax3 = plt.subplot(num_results * 2, 4, base_row * 4 + 3)
        flow_viz = result['roi1'].copy()
        
        # Draw good features in green
        for old_pt, new_pt in result['good_features']:
            pt1 = tuple(old_pt.astype(int))
            pt2 = tuple(new_pt.astype(int))
            cv2.circle(flow_viz, pt1, 3, config.COLOR_GOOD_FEATURES, -1)
            cv2.arrowedLine(flow_viz, pt1, pt2, config.COLOR_FLOW_VECTORS, 2, tipLength=0.3)
        
        # Draw rejected features in red
        for old_pt, new_pt, reason in result['rejected_features']:
            pt1 = tuple(old_pt.astype(int))
            cv2.circle(flow_viz, pt1, 3, config.COLOR_BAD_FEATURES, -1)
        
        ax3.imshow(cv2.cvtColor(flow_viz, cv2.COLOR_BGR2RGB))
        ax3.set_title(f"Optical Flow\n{result['num_good']} good (green), "
                     f"{len(result['rejected_features'])} rejected (red)", fontsize=10)
        ax3.axis('off')
        
        # Forward-backward error distribution
        ax4 = plt.subplot(num_results * 2, 4, base_row * 4 + 4)
        ax4.hist(result['fb_errors'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.axvline(x=config.FB_ERROR_THRESHOLD, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold: {config.FB_ERROR_THRESHOLD}')
        ax4.set_xlabel('Forward-Backward Error (px)', fontsize=9)
        ax4.set_ylabel('Count', fontsize=9)
        ax4.set_title(f"FB Error Distribution\nMedian: {np.median(result['fb_errors']):.2f} px", fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Row 2: Statistics and details
        ax5 = plt.subplot(num_results * 2, 4, (base_row + 1) * 4 + 1)
        stats_text = f"Frame Pair: {result['frame']} → {result['frame'] + 1}\n\n"
        stats_text += f"Ball Center 1: ({result['ball_center1'][0]:.1f}, {result['ball_center1'][1]:.1f})\n"
        stats_text += f"Ball Center 2: ({result['ball_center2'][0]:.1f}, {result['ball_center2'][1]:.1f})\n"
        stats_text += f"Ball Radius: {result['ball_radius']:.1f} px\n\n"
        stats_text += f"Features Detected: {result['total_features']}\n"
        stats_text += f"Features Passed: {result['num_good']}\n"
        stats_text += f"Pass Rate: {result['num_good']/result['total_features']*100:.1f}%\n\n"
        stats_text += f"FB Error Range: {result['fb_errors'].min():.2f} - {result['fb_errors'].max():.2f}\n"
        stats_text += f"FB Error Mean: {result['fb_errors'].mean():.2f} px"
        
        ax5.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax5.axis('off')
        
        # Rejection reasons breakdown
        ax6 = plt.subplot(num_results * 2, 4, (base_row + 1) * 4 + 2)
        rejection_counts = {}
        for _, _, reason in result['rejected_features']:
            reason_type = reason.split(':')[0]
            rejection_counts[reason_type] = rejection_counts.get(reason_type, 0) + 1
        
        if rejection_counts:
            reasons = list(rejection_counts.keys())
            counts = list(rejection_counts.values())
            ax6.barh(reasons, counts, color='coral', edgecolor='black')
            ax6.set_xlabel('Count', fontsize=9)
            ax6.set_title(f"Rejection Reasons\n({len(result['rejected_features'])} total)", fontsize=10)
            ax6.grid(True, alpha=0.3, axis='x')
        else:
            ax6.text(0.5, 0.5, 'No rejections', ha='center', va='center', fontsize=12)
            ax6.axis('off')
        
        # Movement magnitude distribution
        ax7 = plt.subplot(num_results * 2, 4, (base_row + 1) * 4 + 3)
        movements = [np.linalg.norm(np.array(new_pt) - np.array(old_pt)) 
                    for old_pt, new_pt in result['good_features']]
        if movements:
            ax7.hist(movements, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
            ax7.axvline(x=np.mean(movements), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(movements):.2f} px')
            ax7.set_xlabel('Movement (px)', fontsize=9)
            ax7.set_ylabel('Count', fontsize=9)
            ax7.set_title(f"Feature Movement\nGood Features Only", fontsize=10)
            ax7.legend(fontsize=8)
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'No good features', ha='center', va='center', fontsize=12)
            ax7.axis('off')
        
        # Frame 2 with tracked features
        ax8 = plt.subplot(num_results * 2, 4, (base_row + 1) * 4 + 4)
        roi2_display = result['roi2'].copy()
        for old_pt, new_pt in result['good_features']:
            pt2 = tuple(new_pt.astype(int))
            cv2.circle(roi2_display, pt2, 3, config.COLOR_GOOD_FEATURES, -1)
        cv2.circle(roi2_display, tuple(result['center_roi2'].astype(int)), 
                  int(result['ball_radius']), (0, 255, 255), 2)
        ax8.imshow(cv2.cvtColor(roi2_display, cv2.COLOR_BGR2RGB))
        ax8.set_title(f"Frame {result['frame'] + 1}\nTracked Features", fontsize=10)
        ax8.axis('off')
    
    plt.suptitle('Stage B: Optical Flow Detection Test Results', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    plot_path = output_dir / config.STAGE_B_OPTICAL_FLOW_TEST
    plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"[OK] Stage B optical flow test plot saved: {plot_path.name}")
