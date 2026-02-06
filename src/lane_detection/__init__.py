"""
Lane Detection Module - Phase 1 Complete Pipeline

Detects all 4 bowling lane boundaries:
- Bottom: Foul line (horizontal)
- Left/Right: Master lines (vertical)
- Top: Pin area boundary (horizontal)

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Last Updated: February 6, 2026
"""

# Main API - LaneDetector class and pipeline entry point
from .lane_detector import LaneDetector

# Legacy imports for backward compatibility
from .detection_functions import detect_horizontal_line, detect_vertical_boundaries_approach1
from .master_line_computation import compute_master_line_from_collection, visualize_bin_analysis
from .intermediate_visualization import create_intermediate_video
from .tracking_analysis import analyze_master_line_tracking, plot_master_line_tracking, create_summary_plot
from .top_boundary_detection import (
    detect_top_boundary_all_frames,
    fit_msac_line, 
    create_visualization_videos,
    plot_intersection_y_coordinates
)

__version__ = '1.0.0'

__all__ = [
    # Main API
    'LaneDetector',
    
    # Detection functions
    'detect_horizontal_line',
    'detect_vertical_boundaries_approach1',
    
    # Master line computation
    'compute_master_line_from_collection',
    'visualize_bin_analysis',
    
    # Visualization
    'create_intermediate_video',
    
    # Tracking analysis
    'analyze_master_line_tracking',
    'plot_master_line_tracking',
    'create_summary_plot',
    
    # Top boundary detection
    'detect_top_boundary_all_frames',
    'fit_msac_line',
    'create_visualization_videos',
    'plot_intersection_y_coordinates',
]
