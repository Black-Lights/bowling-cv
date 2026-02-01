"""
Ball Detection Module - Phase 2

This module provides ball detection functionality for bowling videos.
Uses boundary data from Phase 1 (lane detection) to focus on the lane area.

Current Implementation:
- Stage A: Video masking and perspective transformation
- Stage B: Motion detection with background subtraction (MOG2)

Version: 1.1.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 1, 2026
Last Updated: February 1, 2026
"""

from . import config

__all__ = ['config']
__version__ = '1.1.0'
