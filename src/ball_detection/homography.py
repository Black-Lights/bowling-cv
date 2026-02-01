"""
Homography Calculation and Perspective Transformation

Calculates 2D homography matrix using DLT (Direct Linear Transform) to map
image coordinates to real-world bowling lane coordinates.

Real-world bowling lane dimensions:
- Length: 60 feet = 720 inches
- Width: 41.5 inches

Coordinate system (top-left origin):
- Top-left:     (0, 0)
- Top-right:    (41.5, 0)
- Bottom-right: (41.5, 720)
- Bottom-left:  (0, 720)

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 1, 2026
"""

import cv2
import numpy as np
import json
import os


# Bowling lane real-world dimensions (in inches)
LANE_WIDTH_INCHES = 41.5
LANE_LENGTH_FEET = 60
LANE_LENGTH_INCHES = LANE_LENGTH_FEET * 12  # 720 inches


def calculate_lane_corners_image(boundary_data):
    """
    Calculate the 4 corner points of the bowling lane in image coordinates.
    
    Uses the same calculation as the masking polygon to ensure consistency.
    
    Parameters:
    -----------
    boundary_data : dict
        Boundary data from Phase 1 containing master lines and boundaries
        
    Returns:
    --------
    numpy.ndarray : 4x2 array of corner points in image coordinates
        [[top_left_x, top_left_y],
         [top_right_x, top_right_y],
         [bottom_right_x, bottom_right_y],
         [bottom_left_x, bottom_left_y]]
    """
    master_left = boundary_data['master_left']
    master_right = boundary_data['master_right']
    top_boundary = boundary_data['top_boundary']
    median_foul_params = boundary_data['median_foul_params']
    
    # Get boundary y positions
    top_y = int(top_boundary['y_position'])
    foul_y = median_foul_params['center_y']
    
    # Get x positions at foul line (bottom boundary)
    left_x_intersect = master_left['x_intersect']
    right_x_intersect = master_right['x_intersect']
    left_slope = master_left['slope']
    right_slope = master_right['slope']
    
    # Calculate x positions at top boundary using line equation
    # x = x_intersect + (y - y_intersect) / slope
    left_x_top = int(left_x_intersect + (top_y - foul_y) / left_slope) if left_slope != 0 else left_x_intersect
    right_x_top = int(right_x_intersect + (top_y - foul_y) / right_slope) if right_slope != 0 else right_x_intersect
    
    # Bottom positions (at foul line)
    left_x_foul = left_x_intersect
    right_x_foul = right_x_intersect
    
    # Create corner points array (clockwise from top-left)
    corners_image = np.array([
        [left_x_top, top_y],        # Top-left
        [right_x_top, top_y],       # Top-right
        [right_x_foul, foul_y],     # Bottom-right
        [left_x_foul, foul_y]       # Bottom-left
    ], dtype=np.float32)
    
    return corners_image


def calculate_lane_corners_realworld():
    """
    Calculate the 4 corner points of the bowling lane in real-world coordinates.
    
    Uses standard bowling lane dimensions with top-left as origin.
    
    Returns:
    --------
    numpy.ndarray : 4x2 array of corner points in real-world coordinates (inches)
        [[top_left_x, top_left_y],
         [top_right_x, top_right_y],
         [bottom_right_x, bottom_right_y],
         [bottom_left_x, bottom_left_y]]
    """
    corners_real = np.array([
        [0, 0],                                    # Top-left
        [LANE_WIDTH_INCHES, 0],                    # Top-right
        [LANE_WIDTH_INCHES, LANE_LENGTH_INCHES],   # Bottom-right
        [0, LANE_LENGTH_INCHES]                    # Bottom-left
    ], dtype=np.float32)
    
    return corners_real


def calculate_homography(boundary_data):
    """
    Calculate homography matrix H that maps image coordinates to real-world coordinates.
    
    Uses cv2.getPerspectiveTransform which implements DLT (Direct Linear Transform).
    
    Parameters:
    -----------
    boundary_data : dict
        Boundary data from Phase 1
        
    Returns:
    --------
    tuple : (H, corners_image, corners_real)
        H : numpy.ndarray - 3x3 homography matrix
        corners_image : numpy.ndarray - 4x2 image corner points
        corners_real : numpy.ndarray - 4x2 real-world corner points
    """
    # Get lane corners in image and real-world coordinates
    corners_image = calculate_lane_corners_image(boundary_data)
    corners_real = calculate_lane_corners_realworld()
    
    # Calculate homography matrix using DLT
    # H maps: corners_image -> corners_real
    H = cv2.getPerspectiveTransform(corners_image, corners_real)
    
    return H, corners_image, corners_real


def apply_perspective_transform(frame, H, output_width=None, output_height=None):
    """
    Apply perspective transformation to a frame using homography matrix.
    
    Parameters:
    -----------
    frame : numpy.ndarray
        Input frame to transform
    H : numpy.ndarray
        3x3 homography matrix
    output_width : int, optional
        Width of output frame. If None, uses lane width in pixels (scaled)
    output_height : int, optional
        Height of output frame. If None, uses lane length in pixels (scaled)
        
    Returns:
    --------
    numpy.ndarray : Transformed frame
    """
    # Default output size: scale real-world dimensions to reasonable pixel size
    # Use 10 pixels per inch for good resolution
    if output_width is None:
        output_width = int(LANE_WIDTH_INCHES * 10)  # ~415 pixels
    if output_height is None:
        output_height = int(LANE_LENGTH_INCHES * 10)  # ~7200 pixels
    
    # Apply perspective warp
    transformed = cv2.warpPerspective(frame, H, (output_width, output_height))
    
    return transformed


def save_homography_data(output_dir, H, corners_image, corners_real):
    """
    Save homography matrix and corner points to JSON file.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save homography data
    H : numpy.ndarray
        3x3 homography matrix
    corners_image : numpy.ndarray
        4x2 image corner points
    corners_real : numpy.ndarray
        4x2 real-world corner points
    """
    os.makedirs(output_dir, exist_ok=True)
    
    homography_data = {
        'homography_matrix': H.tolist(),
        'corners_image': corners_image.tolist(),
        'corners_real': corners_real.tolist(),
        'lane_dimensions': {
            'width_inches': LANE_WIDTH_INCHES,
            'length_inches': LANE_LENGTH_INCHES,
            'length_feet': LANE_LENGTH_FEET
        },
        'coordinate_system': 'Top-left origin, X=width, Y=length'
    }
    
    output_path = os.path.join(output_dir, 'homography_data.json')
    with open(output_path, 'w') as f:
        json.dump(homography_data, f, indent=2)
    
    print(f"Homography data saved: {output_path}")


def load_homography_data(output_dir):
    """
    Load homography matrix and corner points from JSON file.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing homography data
        
    Returns:
    --------
    tuple : (H, corners_image, corners_real)
        H : numpy.ndarray - 3x3 homography matrix
        corners_image : numpy.ndarray - 4x2 image corner points
        corners_real : numpy.ndarray - 4x2 real-world corner points
    """
    homography_path = os.path.join(output_dir, 'homography_data.json')
    
    with open(homography_path, 'r') as f:
        data = json.load(f)
    
    H = np.array(data['homography_matrix'], dtype=np.float32)
    corners_image = np.array(data['corners_image'], dtype=np.float32)
    corners_real = np.array(data['corners_real'], dtype=np.float32)
    
    return H, corners_image, corners_real


if __name__ == "__main__":
    # Test homography calculation
    import sys
    from pathlib import Path
    
    # Load boundary data
    boundary_file = Path(__file__).parent.parent.parent / 'output' / 'cropped_test3' / 'boundary_data.json'
    
    if boundary_file.exists():
        with open(boundary_file, 'r') as f:
            boundary_data = json.load(f)
        
        # Calculate homography
        H, corners_img, corners_real = calculate_homography(boundary_data)
        
        print("\n" + "="*80)
        print("HOMOGRAPHY CALCULATION TEST")
        print("="*80)
        
        print("\nImage Corners (pixels):")
        for i, corner in enumerate(corners_img):
            print(f"  Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f})")
        
        print("\nReal-World Corners (inches):")
        for i, corner in enumerate(corners_real):
            print(f"  Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f})")
        
        print("\nHomography Matrix H:")
        print(H)
        
        print("\nLane Dimensions:")
        print(f"  Width:  {LANE_WIDTH_INCHES} inches")
        print(f"  Length: {LANE_LENGTH_INCHES} inches ({LANE_LENGTH_FEET} feet)")
        
        print("\n" + "="*80)
    else:
        print(f"Boundary data not found: {boundary_file}")
        print("Please run Phase 1 lane detection first!")
