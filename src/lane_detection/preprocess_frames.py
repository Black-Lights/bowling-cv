"""
Preprocessing module for lane detection
Applies HSV color filtering to extract brown and red/orange regions
Fills small black gaps with original image pixels
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import subprocess


def fill_small_black_patches(masked_img, original_img, max_patch_size_row, max_patch_size_col):
    """
    Scan row by row and column by column to replace small black patches with original image pixels.
    
    Args:
        masked_img: The masked image (with black regions)
        original_img: The original image
        max_patch_size_row: Maximum size of black patch to fill in rows (in pixels)
        max_patch_size_col: Maximum size of black patch to fill in columns (in pixels)
    
    Returns:
        Image with small black patches filled
    """
    result = masked_img.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Scan rows (horizontal)
    for i in range(height):
        j = 0
        while j < width:
            if gray[i, j] == 0:
                black_start = j
                while j < width and gray[i, j] == 0:
                    j += 1
                black_length = j - black_start
                
                if black_length <= max_patch_size_row:
                    result[i, black_start:j] = original_img[i, black_start:j]
                    # Update grayscale version - convert the BGR slice to grayscale
                    gray[i, black_start:j] = cv2.cvtColor(original_img[i:i+1, black_start:j], cv2.COLOR_BGR2GRAY)[0]
            else:
                j += 1
    
    # Scan columns (vertical)
    for j in range(width):
        i = 0
        while i < height:
            if gray[i, j] == 0:
                black_start = i
                while i < height and gray[i, j] == 0:
                    i += 1
                black_length = i - black_start
                
                if black_length <= max_patch_size_col:
                    result[black_start:i, j] = original_img[black_start:i, j]
                    # Update grayscale version - convert the BGR slice to grayscale
                    gray[black_start:i, j] = cv2.cvtColor(original_img[black_start:i, j:j+1], cv2.COLOR_BGR2GRAY)[:, 0]
            else:
                i += 1
    
    return result


def preprocess_frame_hsv(frame, max_patch_size_row=100, max_patch_size_col=50):
    """
    Apply HSV color filtering to extract brown and red/orange regions,
    then fill small black gaps with original pixels (both rows and columns).
    
    Args:
        frame: Input frame (BGR format)
        max_patch_size_row: Maximum size of black patch to fill in rows (in pixels)
        max_patch_size_col: Maximum size of black patch to fill in columns (in pixels)
    
    Returns:
        Preprocessed frame with brown/red regions and small gaps filled
    """
    # Convert to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Brown mask (H: 0-20, S: 0-255, V: 50-255)
    mask_brown = cv2.inRange(hsv_frame, (0, 0, 50), (20, 255, 255))
    
    # Red/Orange mask (H: 150-180, S: 30-200, V: 200-255)
    mask_red_orange = cv2.inRange(hsv_frame, (150, 30, 200), (180, 200, 255))
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask_brown, mask_red_orange)
    
    # Apply mask to original frame
    masked_img = cv2.bitwise_and(frame, frame, mask=combined_mask)
    
    # Fill small black patches with original image pixels (rows and columns)
    final_img = fill_small_black_patches(masked_img, frame, max_patch_size_row, max_patch_size_col)
    
    return final_img


def create_preprocessed_video(input_video_path, output_video_path, max_patch_size_row=100, max_patch_size_col=50):
    """
    Create a preprocessed video by applying HSV filtering to each frame
    
    Args:
        input_video_path: Path to input video (usually masked video)
        output_video_path: Path to save preprocessed video
        max_patch_size_row: Maximum size of black patch to fill in rows (in pixels)
        max_patch_size_col: Maximum size of black patch to fill in columns (in pixels)
    
    Returns:
        True if successful, False otherwise
    """
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Create temporary directory for frames
    temp_dir = output_video_path.replace('.mp4', '_temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Preprocessing frames...")
    frame_idx = 0
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame (rows and columns)
            preprocessed = preprocess_frame_hsv(frame, max_patch_size_row, max_patch_size_col)
            
            # Save frame
            frame_path = os.path.join(temp_dir, f'frame_{frame_idx:06d}.png')
            cv2.imwrite(frame_path, preprocessed)
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    print(f"Processed {frame_idx} frames")
    print(f"Encoding video with ffmpeg...")
    
    # Use ffmpeg to create video from frames
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_video_path
    ]
    
    try:
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        print(f"✓ Video saved: {output_video_path}")
        
        # Clean up temp frames
        import shutil
        shutil.rmtree(temp_dir)
        print(f"✓ Cleaned up temporary frames")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error encoding video: {e.stderr}")
        return False


def preprocess_all_masked_videos(video_files, output_dir, max_patch_size_row=100, max_patch_size_col=50):
    """
    Preprocess all masked videos with HSV filtering and gap filling
    
    Args:
        video_files: List of video filenames (e.g., ['cropped_test3.mp4'])
        output_dir: Base output directory
        max_patch_size_row: Maximum size of black patch to fill in rows (in pixels)
        max_patch_size_col: Maximum size of black patch to fill in columns (in pixels)
    
    Returns:
        Dictionary mapping video name to preprocessed video path
    """
    preprocessed_videos = {}
    
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, video_name)
        
        # Input: masked video
        masked_video_path = os.path.join(video_output_dir, f'masked_{video_file}')
        
        # Output: preprocessed video
        preprocessed_video_path = os.path.join(video_output_dir, f'preprocessed_{video_file}')
        
        if not os.path.exists(masked_video_path):
            print(f"Warning: Masked video not found: {masked_video_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Preprocessing: {video_file}")
        print(f"{'='*70}")
        print(f"Input:  {masked_video_path}")
        print(f"Output: {preprocessed_video_path}")
        print(f"Max patch size: Row={max_patch_size_row}, Col={max_patch_size_col} pixels")
        
        # Create preprocessed video with gap filling
        success = create_preprocessed_video(masked_video_path, preprocessed_video_path, max_patch_size_row, max_patch_size_col)
        
        if success:
            preprocessed_videos[video_name] = preprocessed_video_path
            print(f"✓ Completed: {video_name}")
        else:
            print(f"✗ Failed: {video_name}")
    
    return preprocessed_videos


if __name__ == '__main__':
    # Test preprocessing on a single frame
    import sys
    
    if len(sys.argv) > 1:
        # Test on provided image
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        
        if img is not None:
            preprocessed = preprocess_frame_hsv(img)
            
            # Show side-by-side comparison
            combined = np.hstack([img, preprocessed])
            cv2.imshow('Original | Preprocessed (HSV Filtered)', combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Error: Cannot load image {img_path}")
    else:
        print("Usage: python preprocess_frames.py <image_path>")
        print("Or import this module to use preprocessing functions")