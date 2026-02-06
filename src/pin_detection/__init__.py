"""
Pin Detection Module

Phase 4: Toppled Pin Detection for bowling video analysis.
Uses frame differencing and contour detection to count remaining pins.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

from .video_preprocessing import PinAreaMasker, create_pin_area_masked_video
from .frame_selector import FrameSelector, select_and_extract_frames
from .pin_counter import PinCounter
from .visualization import PinDetectionVisualizer
from . import config

# Expose main function for external orchestration
def run_pin_detection_pipeline(videos=None):
    """
    Run pin detection pipeline programmatically.
    
    Parameters:
    -----------
    videos : list, optional
        List of video files to process. If None, uses config.VIDEO_FILES
        
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
    
    # Save original argv and replace
    original_argv = sys.argv
    sys.argv = ['main.py'] + args
    
    try:
        main()
    finally:
        # Restore original argv
        sys.argv = original_argv

__version__ = '1.0.0'
__all__ = [
    'PinAreaMasker',
    'create_pin_area_masked_video',
    'FrameSelector',
    'select_and_extract_frames',
    'PinCounter',
    'PinDetectionVisualizer',
    'config',
    'run_pin_detection_pipeline'
]
