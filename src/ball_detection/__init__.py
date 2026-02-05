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

Version: 1.2.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 1, 2026
Last Updated: February 3, 2026
"""

from . import config
from . import roi_logic
from . import blob_analysis
from . import post_processing

__all__ = ['config', 'roi_logic', 'blob_analysis', 'post_processing']
__version__ = '1.2.0'
