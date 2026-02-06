"""
Spin Analysis Module - Phase 3

Detects bowling ball spin using optical flow and 3D rotation analysis.

Stages:
- Stage A: Data Preparation & Trajectory Loading
- Stage B: Optical Flow Detection
- Stage C: 3D Projection & Feature Filtering
- Stage D: Rotation Calculation
- Stage E: Post-Processing & Outlier Removal
- Stage F: Metrics Calculation
- Stage G: Visualization & Reporting

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

from . import config
from . import utils
from . import projection_3d

__all__ = ['config', 'utils', 'projection_3d']
__version__ = '1.0.0'
