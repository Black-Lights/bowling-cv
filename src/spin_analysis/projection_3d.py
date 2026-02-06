"""
Stage C: 3D Projection & Feature Filtering

Projects 2D optical flow features onto 3D sphere surface.
Uses sphere equation: x² + y² + z² = r² to calculate z-coordinate.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Tuple, List, Optional


def project_2d_to_3d(
    points_2d: np.ndarray,
    ball_center: np.ndarray,
    ball_radius: float,
    assume_front: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 2D points to 3D sphere surface.
    
    Uses sphere equation: x² + y² + z² = r²
    Given (x, y) in 2D, calculates z to place point on sphere surface.
    
    Args:
        points_2d: 2D points (N x 2) in image coordinates
        ball_center: Ball center (x, y) in image coordinates
        ball_radius: Ball radius in pixels
        assume_front: If True, use positive z (front of sphere), else negative (back)
    
    Returns:
        tuple: (points_3d, projection_validity)
            - points_3d: 3D points (N x 3) on sphere surface
            - projection_validity: Boolean mask (N,) indicating valid projections
    """
    # Center the points (ball center becomes origin)
    centered_2d = points_2d - ball_center
    
    # Calculate x² + y²
    xy_squared = np.sum(centered_2d**2, axis=1)
    
    # Check if points are within sphere radius (valid for projection)
    r_squared = ball_radius**2
    valid = xy_squared <= r_squared
    
    # Calculate z using sphere equation: z² = r² - x² - y²
    z_squared = np.maximum(0, r_squared - xy_squared)  # Clamp negative to 0
    z = np.sqrt(z_squared)
    
    # Choose front (+z) or back (-z) of sphere
    if not assume_front:
        z = -z
    
    # Combine to 3D points
    points_3d = np.column_stack([centered_2d, z])
    
    return points_3d, valid


def filter_projection_quality(
    points_3d_1: np.ndarray,
    points_3d_2: np.ndarray,
    valid_1: np.ndarray,
    valid_2: np.ndarray,
    ball_radius: float,
    config
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Filter 3D projections based on quality criteria.
    
    Filtering:
    - Both projections must be valid
    - Points must be on front hemisphere (z > -radius*0.1, allow slightly behind)
    - Distance from origin should be close to radius (within 20% tolerance)
    
    Args:
        points_3d_1: 3D points from frame 1 (N x 3)
        points_3d_2: 3D points from frame 2 (N x 3)
        valid_1: Validity mask for frame 1 (N,)
        valid_2: Validity mask for frame 2 (N,)
        ball_radius: Ball radius in pixels
        config: Configuration module
    
    Returns:
        tuple: (good_points_3d_1, good_points_3d_2, num_good)
    """
    # Both must be valid projections
    both_valid = valid_1 & valid_2
    
    # Both should be on front hemisphere (visible to camera)
    # Allow slightly negative z (ball edges can appear slightly behind center)
    z_threshold = -ball_radius * 0.2
    front_hemisphere_1 = points_3d_1[:, 2] > z_threshold
    front_hemisphere_2 = points_3d_2[:, 2] > z_threshold
    both_front = front_hemisphere_1 & front_hemisphere_2
    
    # Check distance from origin (should be close to radius)
    dist_1 = np.linalg.norm(points_3d_1, axis=1)
    dist_2 = np.linalg.norm(points_3d_2, axis=1)
    
    # Allow 30% tolerance (projection is approximate for imperfect circles)
    radius_tolerance = 0.3 * ball_radius
    valid_dist_1 = np.abs(dist_1 - ball_radius) <= radius_tolerance
    valid_dist_2 = np.abs(dist_2 - ball_radius) <= radius_tolerance
    
    # Combine all criteria
    good_mask = both_valid & both_front & valid_dist_1 & valid_dist_2
    
    good_points_3d_1 = points_3d_1[good_mask]
    good_points_3d_2 = points_3d_2[good_mask]
    
    return good_points_3d_1, good_points_3d_2, int(np.sum(good_mask))


def process_frame_pair_3d(
    good_features_2d: List[Tuple[np.ndarray, np.ndarray]],
    ball_center_1: np.ndarray,
    ball_center_2: np.ndarray,
    ball_radius: float,
    roi_offset_1: np.ndarray,
    roi_offset_2: np.ndarray,
    config
) -> dict:
    """
    Process a frame pair: project 2D features to 3D.
    
    Args:
        good_features_2d: List of (point1, point2) tuples from optical flow (ROI coordinates)
        ball_center_1: Ball center in frame 1 (IMAGE coordinates from CSV)
        ball_center_2: Ball center in frame 2 (IMAGE coordinates from CSV)
        ball_radius: Ball radius
        roi_offset_1: ROI offset for frame 1 (to convert ROI to image coords)
        roi_offset_2: ROI offset for frame 2 (to convert ROI to image coords)
        config: Configuration module
    
    Returns:
        dict: Results with 3D projections and statistics
    """
    if len(good_features_2d) == 0:
        return {
            'points_3d_1': np.array([]),
            'points_3d_2': np.array([]),
            'num_input': 0,
            'num_output': 0,
            'projection_rate': 0.0
        }
    
    # Extract 2D points (in ROI coordinates)
    points_2d_1_roi = np.array([feat[0] for feat in good_features_2d])
    points_2d_2_roi = np.array([feat[1] for feat in good_features_2d])
    
    # Convert from ROI coordinates to IMAGE coordinates
    # This ensures we use the same reference frame (CSV coordinates) across all frames
    points_2d_1_image = points_2d_1_roi + roi_offset_1
    points_2d_2_image = points_2d_2_roi + roi_offset_2
    
    # Project to 3D using IMAGE coordinates and CSV ball centers
    # This gives consistent centering: ox, oy = feature_image - ball_center_csv
    points_3d_1, valid_1 = project_2d_to_3d(points_2d_1_image, ball_center_1, ball_radius)
    points_3d_2, valid_2 = project_2d_to_3d(points_2d_2_image, ball_center_2, ball_radius)
    
    # Filter by quality
    good_3d_1, good_3d_2, num_good = filter_projection_quality(
        points_3d_1, points_3d_2, valid_1, valid_2, ball_radius, config
    )
    
    projection_rate = (num_good / len(good_features_2d) * 100) if len(good_features_2d) > 0 else 0
    
    return {
        'points_3d_1': good_3d_1,
        'points_3d_2': good_3d_2,
        'num_input': len(good_features_2d),
        'num_output': num_good,
        'projection_rate': projection_rate,
        'ball_center_1': ball_center_1,
        'ball_center_2': ball_center_2,
        'ball_radius': ball_radius
    }


def visualize_3d_projection(results: List[dict], output_dir: Path, config):
    """
    Visualize 3D projection results for sample frames.
    
    Creates visualization showing:
    - 3D scatter plots of projected points
    - Z-coordinate distributions
    - Projection statistics
    
    Args:
        results: List of result dictionaries from process_frame_pair_3d
        output_dir: Output directory for plots
        config: Configuration module
    """
    # Select sample frames (configurable number)
    if len(results) == 0:
        return
    
    # Select evenly spaced samples based on config
    num_samples = min(config.STAGE_C_NUM_SAMPLE_FRAMES, len(results))
    if num_samples == 1:
        sample_indices = [0]
    elif num_samples == 2:
        sample_indices = [0, len(results)-1]
    else:
        # Evenly distribute samples across the trajectory
        step = len(results) / (num_samples - 1)
        sample_indices = [int(i * step) for i in range(num_samples - 1)] + [len(results) - 1]
    
    sample_results = [results[i] for i in sample_indices if i < len(results)]
    
    num_samples = len(sample_results)
    fig = plt.figure(figsize=(16, 4 * num_samples))
    
    for idx, result in enumerate(sample_results):
        if result['num_output'] == 0:
            continue
            
        points_3d_1 = result['points_3d_1']
        points_3d_2 = result['points_3d_2']
        radius = result['ball_radius']
        
        # 3D scatter plot for frame 1
        ax1 = fig.add_subplot(num_samples, 4, idx*4 + 1, projection='3d')
        ax1.scatter(points_3d_1[:, 0], points_3d_1[:, 1], points_3d_1[:, 2],
                   c='blue', marker='o', s=20, alpha=0.6)
        ax1.set_xlabel('X (px)')
        ax1.set_ylabel('Y (px)')
        ax1.set_zlabel('Z (px)')
        ax1.set_title(f'Frame 1: 3D Points\n{result["num_output"]} points', fontsize=10)
        ax1.set_xlim([-radius, radius])
        ax1.set_ylim([-radius, radius])
        ax1.set_zlim([0, radius])
        
        # 3D scatter plot for frame 2
        ax2 = fig.add_subplot(num_samples, 4, idx*4 + 2, projection='3d')
        ax2.scatter(points_3d_2[:, 0], points_3d_2[:, 1], points_3d_2[:, 2],
                   c='green', marker='o', s=20, alpha=0.6)
        ax2.set_xlabel('X (px)')
        ax2.set_ylabel('Y (px)')
        ax2.set_zlabel('Z (px)')
        ax2.set_title(f'Frame 2: 3D Points\n{result["num_output"]} points', fontsize=10)
        ax2.set_xlim([-radius, radius])
        ax2.set_ylim([-radius, radius])
        ax2.set_zlim([0, radius])
        
        # Z-coordinate distribution
        ax3 = fig.add_subplot(num_samples, 4, idx*4 + 3)
        ax3.hist(points_3d_1[:, 2], bins=20, alpha=0.6, color='blue', label='Frame 1')
        ax3.hist(points_3d_2[:, 2], bins=20, alpha=0.6, color='green', label='Frame 2')
        ax3.set_xlabel('Z coordinate (px)')
        ax3.set_ylabel('Count')
        ax3.set_title('Z Distribution', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Statistics (if enabled)
        ax4 = fig.add_subplot(num_samples, 4, idx*4 + 4)
        ax4.axis('off')
        
        if config.STAGE_C_SHOW_STATISTICS:
            # Calculate stats
            dist_1 = np.linalg.norm(points_3d_1, axis=1)
            dist_2 = np.linalg.norm(points_3d_2, axis=1)
            
            stats_text = f"""
Projection Statistics:
        
Input (2D): {result['num_input']} features
Output (3D): {result['num_output']} features
Success rate: {result['projection_rate']:.1f}%

Ball radius: {radius:.1f} px

Frame 1:
  Z range: [{points_3d_1[:, 2].min():.1f}, {points_3d_1[:, 2].max():.1f}] px
  Distance: {dist_1.mean():.2f} ± {dist_1.std():.2f} px

Frame 2:
  Z range: [{points_3d_2[:, 2].min():.1f}, {points_3d_2[:, 2].max():.1f}] px
  Distance: {dist_2.mean():.2f} ± {dist_2.std():.2f} px
        """.strip()
            
            ax4.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
                    family='monospace', transform=ax4.transAxes)
    
    plt.suptitle('Stage C: 3D Projection Results', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    plot_path = output_dir / config.STAGE_C_3D_PROJECTION_PLOT
    plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"[OK] Stage C 3D projection plot saved: {plot_path.name}")
