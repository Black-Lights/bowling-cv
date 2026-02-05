"""
Stage D: Blob Analysis Visualization
Generates 8 intermediate videos showing blob filtering pipeline

Videos generated:
1. blob_all_contours.mp4 - All detected contours before filtering
2. blob_area_filter.mp4 - Color-coded area filter results
3. blob_circularity_filter.mp4 - Circularity filter with values
4. blob_aspect_ratio_filter.mp4 - Aspect ratio filter with ellipses
5. blob_color_filter.mp4 - HSV color filter (if enabled)
6. blob_final_filtered.mp4 - Only blobs passing all filters
7. blob_filter_comparison.mp4 - Side-by-side raw vs filtered
8. full_blob_pipeline.mp4 - 2x3 grid showing all stages

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Last Updated: February 1, 2026
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
from typing import List, Tuple
import subprocess

from .blob_analysis import BlobMetrics


def save_video_ffmpeg(frames: List[np.ndarray], output_path: str, fps: int = 30, 
                     desc: str = "Saving video") -> None:
    """
    Save frames as video using FFmpeg (handles odd dimensions)
    
    Args:
        frames: List of frames (BGR)
        output_path: Path to save video
        fps: Frames per second
        desc: Progress bar description
    """
    if len(frames) == 0:
        print(f"Warning: No frames to save for {output_path}")
        return
        
    h, w = frames[0].shape[:2]
    
    # Ensure dimensions are even (required for yuv420p)
    if h % 2 != 0 or w % 2 != 0:
        new_h = h + (h % 2)
        new_w = w + (w % 2)
        padded_frames = []
        for frame in frames:
            padded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            padded[:h, :w] = frame
            padded_frames.append(padded)
        frames = padded_frames
        h, w = new_h, new_w
        
    # Create temporary directory for frames
    temp_dir = os.path.join(os.path.dirname(output_path), 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save frames as PNG
    for i, frame in enumerate(tqdm(frames, desc=desc)):
        frame_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
        cv2.imwrite(frame_path, frame)
        
    # Use FFmpeg to create video
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path
    ]
    
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cleanup temp frames
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    

def draw_blob_contours(frame: np.ndarray, blob_metrics: List[BlobMetrics], 
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw blob contours on frame"""
    vis = frame.copy()
    contours = [blob.contour for blob in blob_metrics]
    cv2.drawContours(vis, contours, -1, color, 2)
    return vis
    

def draw_area_filter_results(frame: np.ndarray, blob_metrics: List[BlobMetrics],
                             frame_height: int, analyzer) -> np.ndarray:
    """
    Color-coded area filter visualization
    Green = pass, Red = fail
    """
    vis = frame.copy()
    
    for blob in blob_metrics:
        color = (0, 255, 0) if blob.passes_area else (0, 0, 255)
        cv2.drawContours(vis, [blob.contour], -1, color, 2)
        
        # Get adaptive thresholds for this blob
        cx, cy = blob.centroid
        area_min, area_max = analyzer.calculate_adaptive_area_thresholds(cy, frame_height)
        
        # Draw area value and thresholds
        text = f"A:{blob.area:.0f} [{area_min:.0f}-{area_max:.0f}]"
        cv2.putText(vis, text, (int(cx) - 50, int(cy) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                   
    return vis
    

def draw_circularity_filter_results(frame: np.ndarray, 
                                    blob_metrics: List[BlobMetrics]) -> np.ndarray:
    """
    Color-coded circularity filter visualization
    Green = pass (C >= 0.65), Red = fail
    """
    vis = frame.copy()
    
    for blob in blob_metrics:
        color = (0, 255, 0) if blob.passes_circularity else (0, 0, 255)
        cv2.drawContours(vis, [blob.contour], -1, color, 2)
        
        # Draw circularity value
        cx, cy = blob.centroid
        text = f"C:{blob.circularity:.2f}"
        cv2.putText(vis, text, (int(cx) - 30, int(cy) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                   
    return vis
    

def draw_aspect_ratio_filter_results(frame: np.ndarray, 
                                     blob_metrics: List[BlobMetrics]) -> np.ndarray:
    """
    Color-coded aspect ratio filter visualization
    Green = pass (R < 2.0), Red = fail
    Shows fitted ellipses
    """
    vis = frame.copy()
    
    for blob in blob_metrics:
        color = (0, 255, 0) if blob.passes_aspect_ratio else (0, 0, 255)
        
        # Draw contour
        cv2.drawContours(vis, [blob.contour], -1, color, 2)
        
        # Fit and draw ellipse
        if len(blob.contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(blob.contour)
                cv2.ellipse(vis, ellipse, color, 1)
                
                # Draw aspect ratio value
                cx, cy = blob.centroid
                text = f"R:{blob.aspect_ratio:.2f}"
                cv2.putText(vis, text, (int(cx) - 30, int(cy) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            except cv2.error:
                pass
                
    return vis
    

def draw_color_filter_results(frame: np.ndarray, 
                              blob_metrics: List[BlobMetrics]) -> np.ndarray:
    """
    Color-coded HSV filter visualization
    Green = pass, Red = fail
    """
    vis = frame.copy()
    
    for blob in blob_metrics:
        color = (0, 255, 0) if blob.passes_color else (0, 0, 255)
        cv2.drawContours(vis, [blob.contour], -1, color, 2)
        
        # Draw color match percentage
        if blob.color_match is not None:
            cx, cy = blob.centroid
            text = f"M:{blob.color_match*100:.0f}%"
            cv2.putText(vis, text, (int(cx) - 30, int(cy) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                       
    return vis
    

def draw_final_filtered(frame: np.ndarray, 
                       blob_metrics: List[BlobMetrics]) -> np.ndarray:
    """Draw only blobs that pass all filters"""
    vis = frame.copy()
    
    filtered = [blob for blob in blob_metrics if blob.passes_all_filters]
    
    for blob in filtered:
        # Draw contour in bright green
        cv2.drawContours(vis, [blob.contour], -1, (0, 255, 0), 2)
        
        # Draw centroid
        cx, cy = blob.centroid
        cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 255), -1)
        
        # Draw "BALL" label
        cv2.putText(vis, "BALL", (int(cx) - 20, int(cy) - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                   
    return vis
    

def create_side_by_side(frame1: np.ndarray, frame2: np.ndarray, 
                       label1: str, label2: str) -> np.ndarray:
    """Create side-by-side comparison"""
    h, w = frame1.shape[:2]
    
    # Create canvas
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = frame1
    canvas[:, w:] = frame2
    
    # Add labels
    cv2.putText(canvas, label1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(canvas, label2, (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return canvas
    

def create_grid(frames: List[np.ndarray], labels: List[str], 
               rows: int = 2, cols: int = 3) -> np.ndarray:
    """Create grid layout of frames"""
    h, w = frames[0].shape[:2]
    
    # Create canvas
    canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    for i, (frame, label) in enumerate(zip(frames, labels)):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        y1, y2 = row * h, (row + 1) * h
        x1, x2 = col * w, (col + 1) * w
        
        canvas[y1:y2, x1:x2] = frame
        
        # Add label
        cv2.putText(canvas, label, (x1 + 10, y1 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
    return canvas


def generate_blob_videos(video_name: str, frames: List[np.ndarray], 
                        masks: List[np.ndarray], blob_metrics_list: List[List[BlobMetrics]],
                        analyzer, output_dir: str, config) -> None:
    """
    Generate all 8 blob analysis intermediate videos
    
    Args:
        video_name: Base name of video (e.g., "cropped_test3")
        frames: List of BGR frames
        masks: List of binary masks from Stage B
        blob_metrics_list: List of blob metrics for each frame
        analyzer: BlobAnalyzer instance (for threshold calculations)
        output_dir: Output directory path
        config: Configuration object
    """
    print(f"\n{'='*80}")
    print("STAGE D: BLOB ANALYSIS VISUALIZATION")
    print(f"Video: {video_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    num_frames = len(frames)
    frame_height = frames[0].shape[0]
    
    # Prepare frame lists for each video
    all_contours_frames = []
    area_filter_frames = []
    circularity_filter_frames = []
    aspect_ratio_filter_frames = []
    color_filter_frames = []
    final_filtered_frames = []
    comparison_frames = []
    pipeline_frames = []
    
    print("Generating visualization frames...")
    for i in tqdm(range(num_frames), desc="Processing frames"):
        frame = frames[i]
        blob_metrics = blob_metrics_list[i]
        
        # 1. All contours
        all_contours = draw_blob_contours(frame, blob_metrics, (255, 255, 0))
        all_contours_frames.append(all_contours)
        
        # 2. Area filter
        area_vis = draw_area_filter_results(frame, blob_metrics, frame_height, analyzer)
        area_filter_frames.append(area_vis)
        
        # 3. Circularity filter
        circ_vis = draw_circularity_filter_results(frame, blob_metrics)
        circularity_filter_frames.append(circ_vis)
        
        # 4. Aspect ratio filter
        aspect_vis = draw_aspect_ratio_filter_results(frame, blob_metrics)
        aspect_ratio_filter_frames.append(aspect_vis)
        
        # 5. Color filter (if enabled)
        if config.ENABLE_COLOR_FILTER:
            color_vis = draw_color_filter_results(frame, blob_metrics)
            color_filter_frames.append(color_vis)
            
        # 6. Final filtered
        final_vis = draw_final_filtered(frame, blob_metrics)
        final_filtered_frames.append(final_vis)
        
        # 7. Side-by-side comparison
        comparison = create_side_by_side(all_contours, final_vis, 
                                        "All Contours", "Final Filtered")
        comparison_frames.append(comparison)
        
        # 8. Full pipeline grid (2x3)
        grid_frames = [all_contours, area_vis, circ_vis, 
                      aspect_vis, final_vis, frame.copy()]
        grid_labels = ["All Contours", "Area Filter", "Circularity", 
                      "Aspect Ratio", "Final Filtered", "Original"]
        grid = create_grid(grid_frames, grid_labels, rows=2, cols=3)
        pipeline_frames.append(grid)
        
    # Save videos
    if config.SAVE_BLOB_ALL_CONTOURS_VIDEO:
        path = os.path.join(output_dir, f"{video_name}_blob_all_contours.mp4")
        save_video_ffmpeg(all_contours_frames, path, desc="Saving all contours")
        print(f"✓ Saved: {path}")
        
    if config.SAVE_BLOB_AREA_FILTER_VIDEO:
        path = os.path.join(output_dir, f"{video_name}_blob_area_filter.mp4")
        save_video_ffmpeg(area_filter_frames, path, desc="Saving area filter")
        print(f"✓ Saved: {path}")
        
    if config.SAVE_BLOB_CIRCULARITY_FILTER_VIDEO:
        path = os.path.join(output_dir, f"{video_name}_blob_circularity_filter.mp4")
        save_video_ffmpeg(circularity_filter_frames, path, desc="Saving circularity filter")
        print(f"✓ Saved: {path}")
        
    if config.SAVE_BLOB_ASPECT_RATIO_FILTER_VIDEO:
        path = os.path.join(output_dir, f"{video_name}_blob_aspect_ratio_filter.mp4")
        save_video_ffmpeg(aspect_ratio_filter_frames, path, desc="Saving aspect ratio filter")
        print(f"✓ Saved: {path}")
        
    if config.SAVE_BLOB_COLOR_FILTER_VIDEO and config.ENABLE_COLOR_FILTER:
        path = os.path.join(output_dir, f"{video_name}_blob_color_filter.mp4")
        save_video_ffmpeg(color_filter_frames, path, desc="Saving color filter")
        print(f"✓ Saved: {path}")
        
    if config.SAVE_BLOB_FINAL_FILTERED_VIDEO:
        path = os.path.join(output_dir, f"{video_name}_blob_final_filtered.mp4")
        save_video_ffmpeg(final_filtered_frames, path, desc="Saving final filtered")
        print(f"✓ Saved: {path}")
        
    if config.SAVE_BLOB_FILTER_COMPARISON_VIDEO:
        path = os.path.join(output_dir, f"{video_name}_blob_filter_comparison.mp4")
        save_video_ffmpeg(comparison_frames, path, desc="Saving comparison")
        print(f"✓ Saved: {path}")
        
    if config.SAVE_FULL_BLOB_PIPELINE_VIDEO:
        path = os.path.join(output_dir, f"{video_name}_full_blob_pipeline.mp4")
        save_video_ffmpeg(pipeline_frames, path, desc="Saving full pipeline")
        print(f"✓ Saved: {path}")
        
    print(f"\nStage D Complete! Generated {sum([config.SAVE_BLOB_ALL_CONTOURS_VIDEO, config.SAVE_BLOB_AREA_FILTER_VIDEO, config.SAVE_BLOB_CIRCULARITY_FILTER_VIDEO, config.SAVE_BLOB_ASPECT_RATIO_FILTER_VIDEO, config.SAVE_BLOB_COLOR_FILTER_VIDEO and config.ENABLE_COLOR_FILTER, config.SAVE_BLOB_FINAL_FILTERED_VIDEO, config.SAVE_BLOB_FILTER_COMPARISON_VIDEO, config.SAVE_FULL_BLOB_PIPELINE_VIDEO])} videos.\n")
