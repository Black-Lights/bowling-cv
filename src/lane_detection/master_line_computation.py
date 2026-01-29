"""
Master line computation using voting system
"""

import numpy as np
import matplotlib.pyplot as plt
from detection_utils import *


def compute_master_line_from_collection(collected_lines, median_foul_params, bin_width=10,
                                       vote_threshold=0.15, angle_tolerance=3, side='left',
                                       angle_mode='from_horizontal'):
    """
    Modified Approach2: Compute master line from collected lines.
    
    Returns:
    --------
    tuple: (master_line_data, debug_info)
    """
    if not collected_lines:
        return None, None
    
    foul_center_x = median_foul_params['center_x']
    width = median_foul_params['width']
    
    if side == 'left':
        x_min, x_max = 0, foul_center_x
    else:
        x_min, x_max = foul_center_x, width
    
    # STEP 3: Create bins
    num_bins = int((x_max - x_min) / bin_width) + 1
    bins = {}
    for i in range(num_bins):
        bin_start = x_min + i * bin_width
        bin_end = bin_start + bin_width
        bins[i] = {
            'bin_id': i,
            'x_range': (bin_start, bin_end),
            'vote_count': 0,
            'line_list': []
        }
    
    # STEP 4: Vote for each line
    for line_data in collected_lines:
        x_intersect = line_data['x_intersect']
        
        bin_id = int((x_intersect - x_min) / bin_width)
        bin_id = max(0, min(bin_id, num_bins - 1))
        
        bins[bin_id]['line_list'].append(line_data)
        bins[bin_id]['vote_count'] += 1
    
    # STEP 5: Find winning bin
    max_votes = max(bin_data['vote_count'] for bin_data in bins.values())
    
    if max_votes == 0:
        return None, None
    
    threshold = int(max_votes * vote_threshold)
    valid_bins = [bin_data for bin_data in bins.values() if bin_data['vote_count'] >= threshold]
    
    if not valid_bins:
        return None, None
    
    winning_bin = max(valid_bins, key=lambda b: b['vote_count'])
    
    # STEP 6: Select best line from winning bin
    bin_lines = winning_bin['line_list']
    
    if not bin_lines:
        return None, None
    
    print(f"Winning bin {winning_bin['bin_id']} has {len(bin_lines)} lines for {side} side")
    
    # Calculate median angle
    angles = [line_data['angle'] for line_data in bin_lines]
    median_angle = np.median(angles)
    
    # Filter lines within angle tolerance
    filtered_lines = [
        line_data for line_data in bin_lines
        if abs(line_data['angle'] - median_angle) <= angle_tolerance
    ]
    
    if not filtered_lines:
        filtered_lines = bin_lines
    
    print(f"After angle filtering: {len(filtered_lines)} lines remain")
    
    # Find longest line
    longest_line = max(filtered_lines, key=lambda ld: ld['length'])
    
    # Calculate median X-position
    x_positions = [line_data['x_intersect'] for line_data in filtered_lines]
    median_x = int(np.median(x_positions))
    
    # STEP 7: Create master line
    # Use longest line's slope, passing through median_x at foul line
    x1, y1, x2, y2 = longest_line['line']
    x1, y1, x2, y2 = normalize_line_direction(x1, y1, x2, y2)
    
    # Calculate v_slope for extrapolation
    if y2 - y1 == 0:
        v_slope = 0
    else:
        v_slope = (x2 - x1) / (y2 - y1)
    
    foul_y = median_foul_params['center_y']
    height = median_foul_params['height']
    
    y_top = 0
    y_bottom = height
    
    x_top = int(median_x + v_slope * (y_top - foul_y))
    x_bottom = int(median_x + v_slope * (y_bottom - foul_y))
    
    # Recalculate angle for master line
    master_angle = calculate_line_angle(x_bottom, y_bottom, x_top, y_top, angle_mode)
    
    master_line_data = {
        'x_top': x_top,
        'y_top': y_top,
        'x_bottom': x_bottom,
        'y_bottom': y_bottom,
        'slope': longest_line['slope'],
        'v_slope': v_slope,  # For extrapolation
        'x_intersect': median_x,
        'median_angle': master_angle,
        'original_angle': median_angle
    }
    
    debug_info = {
        'bins': bins,
        'winning_bin': winning_bin,
        'all_lines': collected_lines,
        'filtered_lines': filtered_lines,
        'longest_line': longest_line,
        'median_x': median_x,
        'median_angle': median_angle,
        'angles': angles
    }
    
    return master_line_data, debug_info


def visualize_bin_analysis(debug_info, side, output_path, angle_mode='from_horizontal'):
    """Create visualization of bin voting analysis."""
    bins = debug_info['bins']
    winning_bin = debug_info['winning_bin']
    
    bin_ids = list(bins.keys())
    vote_counts = [bins[i]['vote_count'] for i in bin_ids]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Bin votes
    colors = ['red' if i == winning_bin['bin_id'] else 'blue' for i in bin_ids]
    ax1.bar(bin_ids, vote_counts, color=colors, alpha=0.7)
    ax1.set_xlabel('Bin ID', fontsize=12)
    ax1.set_ylabel('Vote Count', fontsize=12)
    ax1.set_title(f'{side.capitalize()} Boundary - Bin Voting Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if max(vote_counts) > 0:
        ax1.axhline(y=max(vote_counts) * 0.15, color='green', linestyle='--', label='15% Threshold')
    ax1.legend()
    
    # Plot 2: Angle distribution in winning bin
    if debug_info['filtered_lines']:
        angles = debug_info['angles']
        ax2.hist(angles, bins=20, alpha=0.7, color='green')
        ax2.axvline(x=debug_info['median_angle'], color='red', linestyle='--', linewidth=2, label='Median Angle')
        ax2.set_xlabel(f'Angle ({angle_mode}) (degrees)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Angle Distribution in Winning Bin (Bin {winning_bin["bin_id"]})', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved bin analysis: {output_path}")
