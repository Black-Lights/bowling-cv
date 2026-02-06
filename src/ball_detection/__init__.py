"""
Ball Detection Module - Phase 2

This module provides ball detection functionality for bowling videos.
Uses boundary data from Phase 1 (lane detection) to focus on the lane area.

Current Implementation:
- Stage A: Video masking and perspective transformation
- Stage B: Motion detection with background subtraction (MOG2)
- Stage C: ROI logic (search strategy with Kalman tracking)
- Stage D: Blob analysis and filtering
- Stage E: Kalman filter tracking (integrated with Stage C)
- Stage G: Post-processing (trajectory cleaning and reconstruction)
- Stage H: Overlay video generation (RANSAC fitted radius visualization)

Version: 1.3.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 1, 2026
Last Updated: February 6, 2026
"""

from . import config
from . import roi_logic
from . import blob_analysis
from . import post_processing

# Expose main function for external orchestration
def run_ball_detection_pipeline(videos=None, **kwargs):
    """
    Run ball detection pipeline programmatically.
    
    Parameters:
    -----------
    videos : list, optional
        List of video files to process. If None, uses config.VIDEO_FILES
    **kwargs : dict
        Optional flags: skip_masking, skip_transform, skip_motion, 
        skip_roi, skip_blob, skip_postprocess
        
    Returns:
    --------
    dict : Results for each processed video
    """
    from .main import main
    import sys
    
    # Build command line arguments
    args = []
    if videos:
        for video in videos:
            args.extend(['--video', video])
    
    for flag, value in kwargs.items():
        if value:
            args.append(f'--{flag.replace("_", "-")}')
    
    # Save original argv and replace
    original_argv = sys.argv
    sys.argv = ['main.py'] + args
    
    try:
        main()
    finally:
        # Restore original argv
        sys.argv = original_argv

__all__ = ['config', 'roi_logic', 'blob_analysis', 'post_processing', 'run_ball_detection_pipeline']
__version__ = '1.3.0'
