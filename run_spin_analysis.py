"""
Quick script to run Phase 3 (spin analysis) on a video

Currently implements:
- Stage A: Data Preparation & Trajectory Loading
- Stage B: Optical Flow Detection (test phase)

Usage:
    python run_spin_analysis.py <video_file>

Example:
    python run_spin_analysis.py cropped_test3.mp4
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from spin_analysis.main import main as spin_main


def main():
    """Wrapper for spin analysis main function."""
    
    # If video argument provided, inject it into sys.argv for argparse
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        sys.argv = ['run_spin_analysis.py', '--video', video_file]
    
    # Run spin analysis
    spin_main()


if __name__ == '__main__':
    main()
