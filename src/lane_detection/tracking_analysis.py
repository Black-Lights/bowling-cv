"""
Tracking analysis module for analyzing master line stability
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def analyze_master_line_tracking(video_path, master_left, master_right, median_foul_params):
    """
    Analyze master line positions across all video frames.
    
    Returns:
    --------
    dict with tracking data
    """
    print("  Analyzing master line tracking...")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("    Error: Could not open video")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Storage for tracking data
    data = {
        'frames': [],
        'foul_y': [],
        'left_x': [],
        'right_x': [],
        'lane_width': []
    }
    
    frame_num = 0
    
    for _ in tqdm(range(total_frames), desc="    Tracking", leave=False):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        data['frames'].append(frame_num)
        
        # Foul line Y (constant)
        if median_foul_params:
            data['foul_y'].append(median_foul_params['center_y'])
        else:
            data['foul_y'].append(None)
        
        # Left boundary X (constant)
        if master_left:
            data['left_x'].append(master_left['x_intersect'])
        else:
            data['left_x'].append(None)
        
        # Right boundary X (constant)
        if master_right:
            data['right_x'].append(master_right['x_intersect'])
        else:
            data['right_x'].append(None)
        
        # Lane width
        if master_left and master_right:
            width = master_right['x_intersect'] - master_left['x_intersect']
            data['lane_width'].append(width)
        else:
            data['lane_width'].append(None)
        
        frame_num += 1
    
    cap.release()
    
    print(f"    ✓ Analyzed {frame_num} frames")
    
    return data


def plot_master_line_tracking(data, video_name, output_dir, master_left, master_right):
    """
    Create tracking plots for master lines.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: X coordinates over time
    ax1 = axes[0]
    ax1.plot(data['frames'], data['left_x'], 'g-', linewidth=2, label=f'Left Boundary (X={master_left["x_intersect"]})', alpha=0.7)
    ax1.plot(data['frames'], data['right_x'], 'r-', linewidth=2, label=f'Right Boundary (X={master_right["x_intersect"]})', alpha=0.7)
    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_ylabel('X Coordinate (pixels)', fontsize=12)
    ax1.set_title(f'{video_name} - Master Line Positions (Static)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Foul line Y over time
    ax2 = axes[1]
    ax2.plot(data['frames'], data['foul_y'], 'y-', linewidth=2, label=f'Foul Line Y={data["foul_y"][0]}', alpha=0.7)
    ax2.set_xlabel('Frame Number', fontsize=12)
    ax2.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax2.set_title(f'{video_name} - Foul Line Position (Static)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Lane width over time
    ax3 = axes[2]
    lane_width = data['lane_width'][0] if data['lane_width'] else 0
    ax3.plot(data['frames'], data['lane_width'], 'b-', linewidth=2, label=f'Lane Width={lane_width} pixels', alpha=0.7)
    ax3.set_xlabel('Frame Number', fontsize=12)
    ax3.set_ylabel('Lane Width (pixels)', fontsize=12)
    ax3.set_title(f'{video_name} - Lane Width (Static)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'tracking_{video_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: tracking_{video_name}.png")
    
    return data


def create_summary_plot(all_data, output_dir):
    """
    Create summary plot comparing all videos.
    """
    if not all_data:
        return
    
    print("\n  Creating summary plot...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'purple', 'brown']
    
    # Plot 1: Foul line Y for all videos
    ax1 = axes[0]
    for idx, (video_name, data) in enumerate(all_data.items()):
        color = colors[idx % len(colors)]
        ax1.plot(data['frames'], data['foul_y'], '-', linewidth=2,
                label=f"{video_name} (Y={data['foul_y'][0]})", alpha=0.7, color=color)
    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_ylabel('Foul Line Y (pixels)', fontsize=12)
    ax1.set_title('Foul Line Position - All Videos', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Left boundary X for all videos
    ax2 = axes[1]
    for idx, (video_name, data) in enumerate(all_data.items()):
        color = colors[idx % len(colors)]
        ax2.plot(data['frames'], data['left_x'], '-', linewidth=2,
                label=f"{video_name} (X={data['left_x'][0]})", alpha=0.7, color=color)
    ax2.set_xlabel('Frame Number', fontsize=12)
    ax2.set_ylabel('Left Boundary X (pixels)', fontsize=12)
    ax2.set_title('Left Boundary Position - All Videos', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Lane width for all videos
    ax3 = axes[2]
    for idx, (video_name, data) in enumerate(all_data.items()):
        color = colors[idx % len(colors)]
        ax3.plot(data['frames'], data['lane_width'], '-', linewidth=2,
                label=f"{video_name} (W={data['lane_width'][0]})", alpha=0.7, color=color)
    ax3.set_xlabel('Frame Number', fontsize=12)
    ax3.set_ylabel('Lane Width (pixels)', fontsize=12)
    ax3.set_title('Lane Width - All Videos', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, 'summary_all_videos.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: summary_all_videos.png")
