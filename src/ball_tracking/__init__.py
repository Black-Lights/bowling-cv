"""
Ball Tracking Module - Phase 2

This module provides ball detection and tracking functionality for bowling videos.
Builds on Phase 1 (Lane Detection) to track the ball throughout its trajectory.

Version: 2.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: January 30, 2026
Last Updated: January 30, 2026
"""

from .ball_tracker import BallTracker
from . import config

__all__ = [
    'BallTracker',
    'config',
]

__version__ = '2.0.0'
