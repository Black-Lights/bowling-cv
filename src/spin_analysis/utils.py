"""
Utility Functions for Spin Analysis Module - Stage A

Data loading, preparation, and validation helpers.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict


def load_trajectory_from_ball_detection(video_name: str, config) -> Tuple[pd.DataFrame, Path]:
    """
    Stage A: Load trajectory data from Phase 2 ball detection output.
    
    Uses RANSAC fitted radius for most accurate ball boundary detection.
    
    Args:
        video_name: Name of video (without extension)
        config: Configuration module
    
    Returns:
        tuple: (trajectory_df, trajectory_path)
            - trajectory_df: DataFrame with frame, x, y, radius
            - trajectory_path: Path to original trajectory file
    
    Raises:
        FileNotFoundError: If ball detection output not found
    """
    # Construct path to ball detection output
    trajectory_path = Path(config.OUTPUT_DIR) / video_name / config.BALL_DETECTION_SUBDIR / config.TRAJECTORY_INPUT_FILE
    
    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"Ball detection trajectory not found: {trajectory_path}\n"
            f"Please run Phase 2 (Ball Detection) first:\n"
            f"  python -m src.ball_detection.main --video {video_name}.mp4"
        )
    
    # Load trajectory CSV
    df = pd.read_csv(trajectory_path)
    
    if config.VERBOSE:
        print(f"\n{'='*70}")
        print(f"STAGE A: DATA PREPARATION")
        print(f"{'='*70}")
        print(f"Loading trajectory: {trajectory_path.name}")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {', '.join(df.columns)}")
    
    # Validate required columns
    required_cols = ['frame', 'x', 'y', 'radius_fitted']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in trajectory CSV: {missing_cols}")
    
    # Create working DataFrame with cleaned trajectory and RANSAC radius
    trajectory_df = pd.DataFrame({
        'frame': df['frame'].astype(int),
        'x': df['x'].astype(float),
        'y': df['y'].astype(float),
        'radius': df['radius_fitted'].astype(float)  # Use RANSAC fitted radius
    })
    
    # Remove any rows with NaN values
    initial_count = len(trajectory_df)
    trajectory_df = trajectory_df.dropna()
    removed_count = initial_count - len(trajectory_df)
    
    if removed_count > 0 and config.VERBOSE:
        print(f"  ⚠ Removed {removed_count} rows with NaN values")
    
    if config.VERBOSE:
        print(f"\nPrepared trajectory data:")
        print(f"  Frames: {len(trajectory_df)}")
        print(f"  Frame range: {trajectory_df['frame'].min()} - {trajectory_df['frame'].max()}")
        print(f"  X range: {trajectory_df['x'].min():.1f} - {trajectory_df['x'].max():.1f}")
        print(f"  Y range: {trajectory_df['y'].min():.1f} - {trajectory_df['y'].max():.1f}")
        print(f"  Radius range: {trajectory_df['radius'].min():.1f} - {trajectory_df['radius'].max():.1f}")
        print(f"  Using RANSAC fitted radius [OK]")
        print(f"{'='*70}\n")
    
    return trajectory_df, trajectory_path


def validate_trajectory_data(trajectory_df: pd.DataFrame, config) -> Dict:
    """
    Stage A: Validate trajectory data quality and completeness.
    
    Args:
        trajectory_df: Trajectory DataFrame
        config: Configuration module
    
    Returns:
        dict: Validation statistics
    """
    stats = {
        'total_frames': len(trajectory_df),
        'frame_min': int(trajectory_df['frame'].min()),
        'frame_max': int(trajectory_df['frame'].max()),
        'frame_gaps': [],
        'avg_radius': float(trajectory_df['radius'].mean()),
        'radius_std': float(trajectory_df['radius'].std()),
        'trajectory_length_px': 0.0
    }
    
    # Check for frame gaps
    expected_frames = set(range(stats['frame_min'], stats['frame_max'] + 1))
    actual_frames = set(trajectory_df['frame'].values)
    missing_frames = expected_frames - actual_frames
    
    if missing_frames:
        stats['frame_gaps'] = sorted(list(missing_frames))
        if config.VERBOSE:
            print(f"⚠ Warning: {len(missing_frames)} missing frames detected")
            if len(missing_frames) <= 10:
                print(f"  Missing frames: {stats['frame_gaps']}")
    
    # Calculate trajectory length
    x_vals = trajectory_df['x'].values
    y_vals = trajectory_df['y'].values
    distances = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)
    stats['trajectory_length_px'] = float(np.sum(distances))
    
    if config.VERBOSE:
        print(f"\nTrajectory Statistics:")
        print(f"  Average radius: {stats['avg_radius']:.2f} px")
        print(f"  Radius std dev: {stats['radius_std']:.2f} px")
        print(f"  Trajectory length: {stats['trajectory_length_px']:.1f} px")
        print(f"  Detection coverage: {len(trajectory_df)}/{stats['frame_max'] - stats['frame_min'] + 1} frames")
    
    return stats


def visualize_stage_a(trajectory_df: pd.DataFrame, stats: Dict, output_dir: Path, config):
    """
    Stage A: Create validation visualization for loaded trajectory data.
    
    Creates 2x2 subplot showing:
    - Trajectory path (x vs y)
    - Radius over frames
    - X position over frames
    - Y position over frames
    
    Args:
        trajectory_df: Trajectory DataFrame
        stats: Validation statistics
        output_dir: Output directory for plot
        config: Configuration module
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stage A: Trajectory Data Validation', fontsize=16, fontweight='bold')
    
    # Plot 1: Trajectory path (x vs y)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(trajectory_df['x'], trajectory_df['y'], 
                         c=trajectory_df['frame'], cmap='viridis', 
                         s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('X Position (pixels)', fontsize=11)
    ax1.set_ylabel('Y Position (pixels)', fontsize=11)
    ax1.set_title('Ball Trajectory Path', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Invert Y axis (image coordinates)
    plt.colorbar(scatter, ax=ax1, label='Frame Number')
    
    # Add start/end markers
    ax1.plot(trajectory_df['x'].iloc[0], trajectory_df['y'].iloc[0], 
            'go', markersize=12, label='Start', markeredgecolor='black', markeredgewidth=2)
    ax1.plot(trajectory_df['x'].iloc[-1], trajectory_df['y'].iloc[-1], 
            'ro', markersize=12, label='End', markeredgecolor='black', markeredgewidth=2)
    ax1.legend(loc='best')
    
    # Plot 2: Radius over frames
    ax2 = axes[0, 1]
    ax2.plot(trajectory_df['frame'], trajectory_df['radius'], 'b-', linewidth=2, label='RANSAC Fitted Radius')
    ax2.axhline(y=stats['avg_radius'], color='r', linestyle='--', linewidth=1.5, label=f'Mean: {stats["avg_radius"]:.1f} px')
    ax2.fill_between(trajectory_df['frame'], 
                     stats['avg_radius'] - stats['radius_std'],
                     stats['avg_radius'] + stats['radius_std'],
                     alpha=0.2, color='red', label=f'±1σ: {stats["radius_std"]:.1f} px')
    ax2.set_xlabel('Frame Number', fontsize=11)
    ax2.set_ylabel('Radius (pixels)', fontsize=11)
    ax2.set_title('Ball Radius (RANSAC Fitted)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # Plot 3: X position over frames
    ax3 = axes[1, 0]
    ax3.plot(trajectory_df['frame'], trajectory_df['x'], 'g-', linewidth=2)
    ax3.set_xlabel('Frame Number', fontsize=11)
    ax3.set_ylabel('X Position (pixels)', fontsize=11)
    ax3.set_title('Horizontal Position Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Y position over frames
    ax4 = axes[1, 1]
    ax4.plot(trajectory_df['frame'], trajectory_df['y'], 'm-', linewidth=2)
    ax4.set_xlabel('Frame Number', fontsize=11)
    ax4.set_ylabel('Y Position (pixels)', fontsize=11)
    ax4.set_title('Vertical Position Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()  # Invert Y axis (higher values = lower on screen)
    
    # Add statistics text box
    stats_text = f"Statistics:\n"
    stats_text += f"Frames: {stats['total_frames']}\n"
    stats_text += f"Range: {stats['frame_min']}-{stats['frame_max']}\n"
    stats_text += f"Gaps: {len(stats['frame_gaps'])}\n"
    stats_text += f"Avg Radius: {stats['avg_radius']:.1f} px\n"
    stats_text += f"Trajectory: {stats['trajectory_length_px']:.0f} px"
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save plot
    plot_path = output_dir / config.STAGE_A_VALIDATION_PLOT
    plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"[OK] Stage A validation plot saved: {plot_path.name}")


def get_video_info(video_path: str) -> Dict:
    """
    Get video metadata (fps, frame count, dimensions).
    
    Args:
        video_path: Path to video file
    
    Returns:
        dict: Video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    cap.release()
    return info


def setup_output_directory(video_name: str, config) -> Path:
    """
    Create output directory structure for spin analysis results.
    
    Args:
        video_name: Name of video (without extension)
        config: Configuration module
    
    Returns:
        Path: spin_analysis output directory
    """
    output_dir = Path(config.OUTPUT_DIR) / video_name / config.SPIN_ANALYSIS_SUBDIR
    debug_dir = output_dir / config.DEBUG_SUBDIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir
