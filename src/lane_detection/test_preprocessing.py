"""
Test script for HSV preprocessing
Generates preprocessed videos from masked videos
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import config
from preprocess_frames import preprocess_all_masked_videos


def main():
    """
    Preprocess all masked videos with HSV filtering and gap filling
    """
    print("="*70)
    print(" HSV PREPROCESSING - GENERATE PREPROCESSED VIDEOS")
    print("="*70)
    print(f"Max patch size: Row={config.MAX_PATCH_SIZE_ROW}, Col={config.MAX_PATCH_SIZE_COL} pixels")
    print("="*70)
    
    # Preprocess all masked videos with gap filling
    preprocessed_videos = preprocess_all_masked_videos(
        config.VIDEO_FILES,
        config.OUTPUT_DIR,
        config.MAX_PATCH_SIZE_ROW,
        config.MAX_PATCH_SIZE_COL
    )
    
    print("\n" + "="*70)
    print(" PREPROCESSING COMPLETE")
    print("="*70)
    
    if preprocessed_videos:
        print("\nPreprocessed videos created:")
        for video_name, video_path in preprocessed_videos.items():
            print(f"  {video_name}: {video_path}")
    else:
        print("\nNo videos were preprocessed")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
