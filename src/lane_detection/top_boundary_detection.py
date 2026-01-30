"""
Top boundary detection using Sobel edge detection
Finds the topmost strong horizontal line in the pin area

GPU-accelerated Sobel operations when available
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from .gpu_utils import get_gpu_accelerator, get_performance_tracker


def detect_top_boundary_sobel(frame, config, use_gpu=True, gpu_accelerator=None):
    """
    Detect top boundary using Sobel edge detection to find topmost strong horizontal line.
    
    GPU-accelerated Sobel operations when available.
    
    Args:
        frame: Preprocessed frame (BGR format)
        config: Configuration module with thresholds
        use_gpu: Whether to use GPU acceleration (default: True)
        gpu_accelerator: Existing GPUAccelerator instance (optional, creates new if None)
    
    Returns:
        Dictionary containing:
            - 'y_position': Y-coordinate of detected line
            - 'line_left': Left endpoint (x, y)
            - 'line_right': Right endpoint (x, y)
            - 'edge_strength': Average edge strength
            - 'points_sampled': Number of boundary points found
    """
    # Get GPU accelerator (create if not provided)
    if gpu_accelerator is None and use_gpu:
        gpu_accelerator = get_gpu_accelerator(use_gpu=True, verbose=False)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Apply Sobel for horizontal edges (GPU-accelerated if available)
    if use_gpu and gpu_accelerator is not None and gpu_accelerator.use_gpu:
        sobel_y = gpu_accelerator.Sobel(gray, 0, 1, ksize=config.SOBEL_KERNEL_SIZE)
        sobel_y = sobel_y.astype(np.float64)
    else:
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=config.SOBEL_KERNEL_SIZE)
    
    sobel_y = np.abs(sobel_y)
    
    # Define search region
    left_margin = int(width * config.CENTER_REGION_START)
    right_margin = int(width * config.CENTER_REGION_END)
    search_start = int(height * config.TOP_SCAN_REGION_START)
    search_end = int(height * config.TOP_SCAN_REGION_END)
    
    # Calculate average edge strength for each row in search region
    row_strengths = []
    for y in range(search_start, search_end):
        row_strength = np.mean(sobel_y[y, left_margin:right_margin])
        row_strengths.append((y, row_strength))
    
    # Sort by strength and get top candidates
    row_strengths.sort(key=lambda x: x[1], reverse=True)
    
    # Find the TOPMOST among strong candidates
    num_candidates = max(5, int(len(row_strengths) * config.TOP_CANDIDATES_RATIO))
    top_candidates = row_strengths[:num_candidates]
    
    # Among strong candidates, pick the topmost one
    topmost_y = min([y for y, strength in top_candidates])
    edge_strength = [s for y, s in top_candidates if y == topmost_y][0]
    
    # Sample points along this horizontal line
    boundary_points = []
    for x in range(left_margin, right_margin, 5):
        if sobel_y[topmost_y, x] > config.SOBEL_THRESHOLD:
            boundary_points.append([x, topmost_y])
    
    # If not enough points on exact row, sample nearby rows
    if len(boundary_points) < 10:
        boundary_points = []
        for x in range(left_margin, right_margin, 5):
            for dy in range(-5, 6):
                y_check = topmost_y + dy
                if 0 <= y_check < height and sobel_y[y_check, x] > config.SOBEL_THRESHOLD:
                    boundary_points.append([x, y_check])
                    break
    
    boundary_points = np.array(boundary_points)
    
    # Fit line (should be nearly horizontal)
    if len(boundary_points) > 5:
        [vx, vy, x0, y0] = cv2.fitLine(boundary_points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])
        
        lefty = int((-x0 * vy / vx) + y0)
        righty = int(((width - x0) * vy / vx) + y0)
        
        line_left = (0, lefty)
        line_right = (width - 1, righty)
    else:
        # Fallback: use the topmost_y as horizontal line
        line_left = (0, topmost_y)
        line_right = (width - 1, topmost_y)
    
    return {
        'y_position': topmost_y,
        'line_left': line_left,
        'line_right': line_right,
        'edge_strength': edge_strength,
        'points_sampled': len(boundary_points),
        'sobel_y': sobel_y  # Include for visualization
    }


def detect_top_boundary_all_frames(video_path, config):
    """
    Detect top boundary across all frames in a video.
    
    Args:
        video_path: Path to preprocessed video
        config: Configuration module
    
    Returns:
        List of detection results (one per frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Detecting top boundary in {total_frames} frames...")
    
    detections = []
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detection = detect_top_boundary_sobel(frame, config)
            detections.append(detection)
            
            pbar.update(1)
    
    cap.release()
    
    print(f"Detected top boundary in {len(detections)} frames")
    
    return detections


def fit_msac_line(detections, video_width, output_dir, video_name):
    """
    Fit a single line using MSAC (M-estimator SAmple Consensus) from all detected top lines.
    
    Args:
        detections: List of detection results from all frames
        video_width: Width of the video frame
        output_dir: Directory to save intermediate plots
        video_name: Name of video for output files
    
    Returns:
        Dictionary with:
            - 'line_left': Left endpoint (x, y) of MSAC line
            - 'line_right': Right endpoint (x, y) of MSAC line
            - 'y_position': Y coordinate of the line
            - 'inliers': Boolean array indicating inlier detections
            - 'ransac_model': Fitted RANSAC model
    """
    print("\n  Fitting MSAC line from all detections...")
    
    # Collect all points from all detected lines
    all_points = []
    frame_indices = []
    
    for frame_idx, detection in enumerate(detections):
        # Get left and right endpoints
        left_x, left_y = detection['line_left']
        right_x, right_y = detection['line_right']
        
        # Add both endpoints
        all_points.append([left_x, left_y])
        all_points.append([right_x, right_y])
        frame_indices.extend([frame_idx, frame_idx])
    
    all_points = np.array(all_points)
    frame_indices = np.array(frame_indices)
    
    print(f"    Total points collected: {len(all_points)} from {len(detections)} frames")
    
    # Prepare data for RANSAC (X coordinates -> Y coordinates)
    X = all_points[:, 0].reshape(-1, 1)
    y = all_points[:, 1]
    
    # Fit RANSAC model (similar to MSAC)
    ransac = RANSACRegressor(
        residual_threshold=5.0,  # Maximum residual for inliers
        max_trials=1000,
        min_samples=2,
        random_state=42
    )
    
    ransac.fit(X, y)
    
    # Get inlier mask
    inlier_mask = ransac.inlier_mask_
    outlier_mask = ~inlier_mask
    
    n_inliers = np.sum(inlier_mask)
    n_outliers = np.sum(outlier_mask)
    
    print(f"    MSAC fitting complete:")
    print(f"      Inliers: {n_inliers}/{len(all_points)} ({100*n_inliers/len(all_points):.1f}%)")
    print(f"      Outliers: {n_outliers}/{len(all_points)} ({100*n_outliers/len(all_points):.1f}%)")
    
    # Predict Y for left and right edges of frame
    line_left_x = 0
    line_right_x = video_width - 1
    
    line_left_y = int(ransac.predict([[line_left_x]])[0])
    line_right_y = int(ransac.predict([[line_right_x]])[0])
    
    # Calculate average Y position
    y_position = (line_left_y + line_right_y) / 2
    
    print(f"      MSAC line: ({line_left_x}, {line_left_y}) to ({line_right_x}, {line_right_y})")
    print(f"      Average Y: {y_position:.1f}")
    
    # Create intermediate visualization plots
    plot_msac_fitting(all_points, frame_indices, ransac, inlier_mask, 
                      line_left_x, line_left_y, line_right_x, line_right_y,
                      output_dir, video_name)
    
    # Determine which frames are inliers (if both endpoints are inliers)
    frame_inliers = np.zeros(len(detections), dtype=bool)
    for frame_idx in range(len(detections)):
        # Check if both points from this frame are inliers
        frame_mask = frame_indices == frame_idx
        frame_inliers[frame_idx] = np.all(inlier_mask[frame_mask])
    
    return {
        'line_left': (line_left_x, line_left_y),
        'line_right': (line_right_x, line_right_y),
        'y_position': y_position,
        'inliers': frame_inliers,
        'ransac_model': ransac,
        'n_inliers': n_inliers,
        'n_outliers': n_outliers
    }


def plot_msac_fitting(all_points, frame_indices, ransac, inlier_mask, 
                      line_left_x, line_left_y, line_right_x, line_right_y,
                      output_dir, video_name):
    """
    Create intermediate plots showing MSAC fitting process.
    
    Creates 4 plots:
    1. All detected points colored by frame
    2. Inliers vs outliers with MSAC line
    3. Residuals distribution
    4. Y values across frames
    """
    print(f"    Creating MSAC intermediate plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All detected points colored by frame
    ax1 = axes[0, 0]
    scatter = ax1.scatter(all_points[:, 0], all_points[:, 1], 
                         c=frame_indices, cmap='viridis', alpha=0.6, s=20)
    ax1.plot([line_left_x, line_right_x], [line_left_y, line_right_y], 
            'r-', linewidth=3, label='MSAC Line')
    ax1.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax1.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax1.set_title(f'All Detected Top Boundary Points\n({len(all_points)} points from {len(np.unique(frame_indices))} frames)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Invert Y axis (0 at top)
    plt.colorbar(scatter, ax=ax1, label='Frame Index')
    
    # Plot 2: Inliers vs Outliers
    ax2 = axes[0, 1]
    inliers = all_points[inlier_mask]
    outliers = all_points[~inlier_mask]
    
    ax2.scatter(outliers[:, 0], outliers[:, 1], c='orange', alpha=0.6, s=30, label=f'Outliers ({len(outliers)})')
    ax2.scatter(inliers[:, 0], inliers[:, 1], c='green', alpha=0.6, s=30, label=f'Inliers ({len(inliers)})')
    ax2.plot([line_left_x, line_right_x], [line_left_y, line_right_y], 
            'r-', linewidth=3, label='MSAC Line')
    ax2.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax2.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax2.set_title(f'MSAC Inliers vs Outliers\nInlier ratio: {100*len(inliers)/len(all_points):.1f}%', 
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # Plot 3: Residuals distribution
    ax3 = axes[1, 0]
    X = all_points[:, 0].reshape(-1, 1)
    y_true = all_points[:, 1]
    y_pred = ransac.predict(X)
    residuals = y_true - y_pred
    
    ax3.hist(residuals[inlier_mask], bins=30, alpha=0.7, color='green', label='Inliers')
    ax3.hist(residuals[~inlier_mask], bins=30, alpha=0.7, color='orange', label='Outliers')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
    ax3.set_xlabel('Residual (pixels)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Residuals\n(Distance from MSAC line)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Y values across frames
    ax4 = axes[1, 1]
    frame_y_values = []
    for frame_idx in range(len(np.unique(frame_indices))):
        frame_mask = frame_indices == frame_idx
        frame_y = np.mean(all_points[frame_mask, 1])
        frame_y_values.append(frame_y)
    
    ax4.plot(frame_y_values, 'b-', alpha=0.5, linewidth=1, label='Per-frame Y position')
    ax4.axhline(y=(line_left_y + line_right_y)/2, color='red', linestyle='--', 
               linewidth=2, label=f'MSAC Y = {(line_left_y + line_right_y)/2:.1f}')
    ax4.set_xlabel('Frame Number', fontsize=12)
    ax4.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax4.set_title('Top Boundary Y Position Across Frames', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'msac_fitting_{video_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ MSAC plots saved: msac_fitting_{video_name}.png")


def create_visualization_videos(original_video_path, preprocessed_video_path, detections, 
                                 boundary_data, output_dir, video_name, config, msac_line=None):
    """
    Create 3 output videos:
    1. Sobel filter visualization (red heatmap)
    2. Preprocessed video with detected horizontal line
    3. Final video with all 4 boundaries (top, bottom, left, right)
    
    Args:
        original_video_path: Path to original video
        preprocessed_video_path: Path to preprocessed video
        detections: List of detection results
        boundary_data: Dictionary with left, right, foul boundaries
        output_dir: Directory to save videos
        video_name: Name of video (for output filenames)
        config: Configuration module
    """
    print("\nCreating visualization videos...")
    
    # Open videos
    cap_original = cv2.VideoCapture(original_video_path)
    cap_preprocessed = cv2.VideoCapture(preprocessed_video_path)
    
    if not cap_original.isOpened() or not cap_preprocessed.isOpened():
        print("Error: Cannot open videos")
        return False
    
    # Get video properties
    width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_original.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temp directories for each video
    temp_sobel = os.path.join(output_dir, f'temp_sobel_{video_name}')
    temp_masked = os.path.join(output_dir, f'temp_masked_{video_name}')
    temp_final = os.path.join(output_dir, f'temp_final_{video_name}')
    
    os.makedirs(temp_sobel, exist_ok=True)
    os.makedirs(temp_masked, exist_ok=True)
    os.makedirs(temp_final, exist_ok=True)
    
    print(f"Generating frames for 3 videos...")
    
    with tqdm(total=total_frames, desc="Creating frames") as pbar:
        frame_idx = 0
        while True:
            ret_orig, frame_orig = cap_original.read()
            ret_prep, frame_prep = cap_preprocessed.read()
            
            if not ret_orig or not ret_prep:
                break
            
            if frame_idx >= len(detections):
                break
            
            detection = detections[frame_idx]
            
            # VIDEO 1: Sobel filter visualization (red heatmap)
            sobel_viz = detection['sobel_y'].copy()
            sobel_viz = (sobel_viz / sobel_viz.max() * 255).astype(np.uint8)
            sobel_viz = cv2.applyColorMap(sobel_viz, cv2.COLORMAP_HOT)
            
            # Draw detected line in cyan (per-frame detection)
            cv2.line(sobel_viz, detection['line_left'], detection['line_right'], (255, 255, 0), 2)
            
            # If MSAC line exists, draw it in bright green
            if msac_line:
                cv2.line(sobel_viz, msac_line['line_left'], msac_line['line_right'], (0, 255, 0), 3)
            
            # Draw search region boundaries
            search_start = int(height * config.TOP_SCAN_REGION_START)
            search_end = int(height * config.TOP_SCAN_REGION_END)
            cv2.line(sobel_viz, (0, search_start), (width-1, search_start), (0, 255, 255), 2)
            cv2.line(sobel_viz, (0, search_end), (width-1, search_end), (0, 255, 255), 2)
            
            cv2.imwrite(os.path.join(temp_sobel, f'frame_{frame_idx:06d}.png'), sobel_viz)
            
            # VIDEO 2: Preprocessed with MSAC line (or detected line if no MSAC)
            masked_viz = frame_prep.copy()
            if msac_line:
                # Use MSAC line
                cv2.line(masked_viz, msac_line['line_left'], msac_line['line_right'], (0, 255, 0), 3)
            else:
                # Use per-frame detection
                cv2.line(masked_viz, detection['line_left'], detection['line_right'], (0, 255, 0), 3)
            cv2.imwrite(os.path.join(temp_masked, f'frame_{frame_idx:06d}.png'), masked_viz)
            
            # VIDEO 3: Final with all 4 boundaries
            final_viz = frame_orig.copy()
            
            # Draw master left boundary (blue) - using exact same method as main.py
            cv2.line(final_viz,
                    (boundary_data['master_left']['x_top'], boundary_data['master_left']['y_top']),
                    (boundary_data['master_left']['x_bottom'], boundary_data['master_left']['y_bottom']),
                    (255, 0, 0), 3)
            
            # Draw master right boundary (blue) - using exact same method as main.py
            cv2.line(final_viz,
                    (boundary_data['master_right']['x_top'], boundary_data['master_right']['y_top']),
                    (boundary_data['master_right']['x_bottom'], boundary_data['master_right']['y_bottom']),
                    (255, 0, 0), 3)
            
            # Draw foul line (red)
            foul_y = int(boundary_data['median_foul_params']['center_y'])
            cv2.line(final_viz, (0, foul_y), (width-1, foul_y), (0, 0, 255), 3)
            
            # Draw top boundary (green) - use MSAC line if available
            if msac_line:
                cv2.line(final_viz, msac_line['line_left'], msac_line['line_right'], (0, 255, 0), 3)
            else:
                cv2.line(final_viz, detection['line_left'], detection['line_right'], (0, 255, 0), 3)
            
            cv2.imwrite(os.path.join(temp_final, f'frame_{frame_idx:06d}.png'), final_viz)
            
            frame_idx += 1
            pbar.update(1)
    
    cap_original.release()
    cap_preprocessed.release()
    
    # Encode videos with ffmpeg
    videos_to_create = [
        ('sobel', temp_sobel, os.path.join(output_dir, f'top_vis_sobel_{video_name}.mp4')),
        ('masked', temp_masked, os.path.join(output_dir, f'top_vis_masked_{video_name}.mp4')),
        ('final', temp_final, os.path.join(output_dir, f'final_all_boundaries_{video_name}.mp4'))
    ]
    
    print(f"\nEncoding videos with ffmpeg...")
    
    for name, temp_dir, output_path in videos_to_create:
        print(f"  Creating {name} video...")
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            output_path
        ]
        
        try:
            subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            print(f"    ✓ Saved: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Error: {e.stderr}")
    
    # Clean up temp directories
    import shutil
    for name, temp_dir, _ in videos_to_create:
        shutil.rmtree(temp_dir)
    
    print(f"All visualization videos created")
    
    return True


def calculate_intersection_y_coordinates(detections, boundary_data):
    """
    Calculate Y coordinates where the top boundary line intersects with master left and right lines.
    
    Args:
        detections: List of detection results (one per frame)
        boundary_data: Dictionary with master line parameters
    
    Returns:
        Dictionary with 'left_y' and 'right_y' lists (one per frame)
    """
    left_y_coords = []
    right_y_coords = []
    
    # Get master line parameters
    # Master line equation: x = x_intersect + y * slope
    left_x_intersect = boundary_data['master_left']['x_intersect']
    left_slope = boundary_data['master_left']['slope']
    
    right_x_intersect = boundary_data['master_right']['x_intersect']
    right_slope = boundary_data['master_right']['slope']
    
    for detection in detections:
        # Get top boundary line endpoints
        top_left = detection['line_left']   # (x1, y1)
        top_right = detection['line_right']  # (x2, y2)
        
        # Top line is nearly horizontal, so we can approximate:
        # For intersection with left master line, solve:
        # x = left_x_intersect + y * left_slope (master line)
        # y = top_left[1] + (top_right[1] - top_left[1]) / (top_right[0] - top_left[0]) * (x - top_left[0]) (top line)
        # 
        # Since top line is nearly horizontal, y ≈ constant
        # Use the Y coordinate where the master line crosses the top line's X range
        
        # For left intersection: find Y where master left line is at X = top_left[0]
        # x = left_x_intersect + y * left_slope
        # Solve for y: y = (x - left_x_intersect) / left_slope
        left_y = (top_left[0] - left_x_intersect) / left_slope if left_slope != 0 else top_left[1]
        
        # For right intersection: find Y where master right line is at X = top_right[0]
        right_y = (top_right[0] - right_x_intersect) / right_slope if right_slope != 0 else top_right[1]
        
        # Clamp to reasonable values (use top line Y as fallback)
        if not (0 <= left_y <= 10000):
            left_y = top_left[1]
        if not (0 <= right_y <= 10000):
            right_y = top_right[1]
        
        left_y_coords.append(left_y)
        right_y_coords.append(right_y)
    
    return {
        'left_y': left_y_coords,
        'right_y': right_y_coords
    }


def plot_intersection_y_coordinates(detections, boundary_data, video_name, output_dir):
    """
    Plot the Y coordinates of top line intersections with master lines across frames.
    
    Args:
        detections: List of detection results
        boundary_data: Dictionary with master line parameters
        video_name: Name of video for plot title
        output_dir: Directory to save plot
    """
    # Calculate intersection Y coordinates
    intersections = calculate_intersection_y_coordinates(detections, boundary_data)
    
    frames = list(range(len(detections)))
    left_y = intersections['left_y']
    right_y = intersections['right_y']
    
    # Calculate statistics
    left_mean = np.mean(left_y)
    left_std = np.std(left_y)
    right_mean = np.mean(right_y)
    right_std = np.std(right_y)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Y coordinates over frames
    ax1.plot(frames, left_y, 'b-', alpha=0.6, linewidth=1, label=f'Left Intersection (μ={left_mean:.1f}, σ={left_std:.2f})')
    ax1.plot(frames, right_y, 'r-', alpha=0.6, linewidth=1, label=f'Right Intersection (μ={right_mean:.1f}, σ={right_std:.2f})')
    ax1.axhline(left_mean, color='b', linestyle='--', alpha=0.5, label=f'Left Mean')
    ax1.axhline(right_mean, color='r', linestyle='--', alpha=0.5, label=f'Right Mean')
    ax1.fill_between(frames, left_mean - left_std, left_mean + left_std, color='b', alpha=0.1)
    ax1.fill_between(frames, right_mean - right_std, right_mean + right_std, color='r', alpha=0.1)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Y Coordinate (pixels)')
    ax1.set_title(f'Top Boundary Line Intersection Y Coordinates - {video_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Invert Y axis so lower Y values (top of frame) are at top
    
    # Plot 2: Histogram of Y coordinates
    ax2.hist(left_y, bins=30, alpha=0.6, color='b', label='Left Intersection', edgecolor='black')
    ax2.hist(right_y, bins=30, alpha=0.6, color='r', label='Right Intersection', edgecolor='black')
    ax2.axvline(left_mean, color='b', linestyle='--', linewidth=2, label=f'Left Mean: {left_mean:.1f}')
    ax2.axvline(right_mean, color='r', linestyle='--', linewidth=2, label=f'Right Mean: {right_mean:.1f}')
    ax2.set_xlabel('Y Coordinate (pixels)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Intersection Y Coordinates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'top_line_intersection_y_{video_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Intersection Y plot saved: {plot_path}")
    print(f"    Left Y: mean={left_mean:.1f}, std={left_std:.2f}, range=[{min(left_y):.0f}, {max(left_y):.0f}]")
    print(f"    Right Y: mean={right_mean:.1f}, std={right_std:.2f}, range=[{min(right_y):.0f}, {max(right_y):.0f}]")
    
    return intersections
