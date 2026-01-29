"""
Lane Detection Module
Detects bowling lane boundaries, foul lines, and master lines
"""

from .config import *
from .detection_utils import *
from .detection_functions import detect_horizontal_line, detect_vertical_boundaries_approach1
from .master_line_computation import compute_master_line_from_collection, visualize_bin_analysis
from .intermediate_visualization import create_intermediate_video
from .tracking_analysis import analyze_master_line_tracking, plot_master_line_tracking, create_summary_plot
from .main import save_boundary_data, load_boundary_data
from .top_boundary_detection import (
    collect_top_boundary_detections, 
    plot_detection_results,
    create_detection_visualization_videos,
    detect_color_stop,
    detect_intensity_transition,
    detect_edge_boundary
)

__all__ = [
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
    
    # Boundary data management
    'save_boundary_data',
    'load_boundary_data',
    
    # Top boundary detection
    'collect_top_boundary_detections',
    'plot_detection_results',
    'create_detection_visualization_videos',
    'detect_color_stop',
    'detect_intensity_transition',
    'detect_edge_boundary',
]
