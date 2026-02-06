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

__version__ = '1.0.0'
__all__ = [
    'PinAreaMasker',
    'create_pin_area_masked_video',
    'FrameSelector',
    'select_and_extract_frames',
    'PinCounter',
    'PinDetectionVisualizer',
    'config'
]
