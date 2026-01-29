"""
Intermediate visualization module for debugging processing pipeline
"""

import numpy as np
import cv2
from tqdm import tqdm
import os


def get_horizontal_intermediates(img):
    """
    Get intermediate results for horizontal line detection.
    
    Returns dict with all intermediate images.
    """
    height, width = img.shape[:2]
    bottom_half = img[height//2:height, 0:width]
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(bottom_half, (5, 5), 0)
    
    # Grayscale
    grayscale = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Otsu threshold
    otsu_value, otsu_thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Canny edges
    lower_thresh = int(0.5 * otsu_value)
    upper_thresh = int(1.5 * otsu_value)
    edges = cv2.Canny(grayscale, lower_thresh, upper_thresh)
    
    # Create full-size images (pad top half with zeros)
    full_blurred = img.copy()
    full_blurred[height//2:height, 0:width] = blurred
    
    full_gray = np.zeros((height, width), dtype=np.uint8)
    full_gray[height//2:height, 0:width] = grayscale
    
    full_otsu = np.zeros((height, width), dtype=np.uint8)
    full_otsu[height//2:height, 0:width] = otsu_thresh
    
    full_edges = np.zeros((height, width), dtype=np.uint8)
    full_edges[height//2:height, 0:width] = edges
    
    return {
        'gaussian': full_blurred,
        'grayscale': cv2.cvtColor(full_gray, cv2.COLOR_GRAY2BGR),
        'otsu': cv2.cvtColor(full_otsu, cv2.COLOR_GRAY2BGR),
        'edges': cv2.cvtColor(full_edges, cv2.COLOR_GRAY2BGR)
    }


def get_vertical_intermediates(img, foul_line_params):
    """
    Get intermediate results for vertical line detection.
    
    Returns dict with all intermediate images.
    """
    if foul_line_params is None:
        height, width = img.shape[:2]
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        return {
            'gaussian': blank,
            'grayscale': blank,
            'otsu': blank,
            'contours': blank,
            'mask': blank,
            'dilated': blank,
            'eroded': blank,
            'edges': blank
        }
    
    height, width = img.shape[:2]
    
    # Gaussian blur
    gaussian = cv2.GaussianBlur(img, (11, 11), 0)
    
    # Grayscale
    greyscale = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    
    # Otsu threshold
    otsu_value, otsu_thresh = cv2.threshold(greyscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Contours
    contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_center_x = width // 2
    img_center_y = height // 2
    
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                if abs(cX - img_center_x) < width * 0.25 and abs(cY - img_center_y) < height * 0.25:
                    filtered_contours.append(contour)
    
    # Visualize contours
    contour_vis = img.copy()
    cv2.drawContours(contour_vis, filtered_contours, -1, (0, 255, 0), 2)
    
    # Mask
    mask = np.zeros_like(greyscale)
    cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
    masked_greyscale = cv2.bitwise_and(greyscale, greyscale, mask=mask)
    
    # Dilation
    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(masked_greyscale, kernel, iterations=1)
    
    # Erosion
    kernel_er = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(dilated, kernel_er, iterations=1)
    
    # Canny edges
    lower_thresh = int(0.5 * otsu_value)
    upper_thresh = int(1.5 * otsu_value)
    edges = cv2.Canny(eroded, lower_thresh, upper_thresh)
    
    return {
        'gaussian': gaussian,
        'grayscale': cv2.cvtColor(greyscale, cv2.COLOR_GRAY2BGR),
        'otsu': cv2.cvtColor(otsu_thresh, cv2.COLOR_GRAY2BGR),
        'contours': contour_vis,
        'mask': cv2.cvtColor(masked_greyscale, cv2.COLOR_GRAY2BGR),
        'dilated': cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR),
        'eroded': cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR),
        'edges': cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    }


def process_frame_intermediate(frame, mode, foul_line_params=None):
    """
    Process frame for intermediate visualization.
    
    Parameters:
    -----------
    frame : numpy.ndarray
        Input frame
    mode : str
        Intermediate mode name
    foul_line_params : dict
        Foul line parameters (needed for vertical modes)
    
    Returns:
    --------
    numpy.ndarray : Processed frame
    """
    if 'horizontal' in mode:
        intermediates = get_horizontal_intermediates(frame)
        key = mode.replace('_horizontal', '')
        return intermediates.get(key, frame)
    
    elif 'vertical' in mode:
        intermediates = get_vertical_intermediates(frame, foul_line_params)
        key = mode.replace('_vertical', '')
        return intermediates.get(key, frame)
    
    return frame


def create_intermediate_video(video_path, output_path, mode, detect_foul_func):
    """
    Create a video showing intermediate processing step.
    
    Parameters:
    -----------
    video_path : str
        Input video path
    output_path : str
        Output video path
    mode : str
        Intermediate visualization mode
    detect_foul_func : function
        Function to detect foul line (needed for vertical modes)
    """
    print(f"  Creating intermediate video: {mode}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"    Error: Could not open video")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    codecs_to_try = [('avc1', '.mp4'), ('XVID', '.avi'), ('MJPG', '.avi')]
    
    out = None
    final_output_path = output_path
    
    for codec, extension in codecs_to_try:
        try:
            if not output_path.endswith(extension):
                final_output_path = os.path.splitext(output_path)[0] + extension
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
            
            if out.isOpened():
                break
            else:
                out.release()
                out = None
        except:
            continue
    
    if out is None or not out.isOpened():
        print(f"    Error: Could not initialize video writer")
        cap.release()
        return
    
    frame_count = 0
    
    for _ in tqdm(range(total_frames), desc=f"    {mode}", leave=False):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Get foul line params for vertical modes
        foul_line_params = None
        if 'vertical' in mode:
            _, _, foul_line_params = detect_foul_func(frame)
        
        # Process frame
        processed_frame = process_frame_intermediate(frame, mode, foul_line_params)
        
        out.write(processed_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"    ✓ Saved: {os.path.basename(final_output_path)}")
