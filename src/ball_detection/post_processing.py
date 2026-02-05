"""
Post-Processing Module - Stage G

This module provides complete trajectory post-processing functionality:
- Stage 1: Trajectory cleaning (median filter, outlier detection, interpolation, smoothing)
- Stage 2: Template reconstruction (scaling, boundary filtering, resolution smoothing)

Combines the functionality from:
- ball_trajectory_processing_clean.ipynb
- trajectory_reconstruction.ipynb

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 3, 2026
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy.signal import savgol_filter
import json
from typing import Tuple, Optional, Dict, Any


class TrajectoryProcessor:
    """
    Stage 1: Process and clean raw trajectory data.
    
    Pipeline:
    1. Moving median filter - Remove spikes and noise
    2. MAD outlier detection - Detect and remove outliers
    3. Cubic interpolation - Fill gaps from outlier removal
    4. Savitzky-Golay smoothing - Final smoothing
    """
    
    def __init__(
        self,
        median_window: int = 5,
        mad_threshold: float = 3.5,
        savgol_window: int = 45,
        savgol_polyorder: int = 2
    ):
        """
        Initialize trajectory processor with processing parameters.
        
        Args:
            median_window: Window size for moving median filter (must be odd)
            mad_threshold: Modified Z-score threshold for outlier detection
            savgol_window: Window length for Savitzky-Golay filter (must be odd)
            savgol_polyorder: Polynomial order for Savitzky-Golay filter
        """
        self.median_window = median_window if median_window % 2 == 1 else median_window + 1
        self.mad_threshold = mad_threshold
        self.savgol_window = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        self.savgol_polyorder = savgol_polyorder
    
    @staticmethod
    def moving_median_filter(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Apply moving median filter to 1D data.
        
        Args:
            data: Input 1D array
            window_size: Size of the median window (must be odd)
        
        Returns:
            Filtered data
        """
        if window_size % 2 == 0:
            window_size += 1
        
        filtered = np.copy(data)
        half_window = window_size // 2
        
        for i in range(len(data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(data), i + half_window + 1)
            filtered[i] = np.nanmedian(data[start_idx:end_idx])
        
        return filtered
    
    @staticmethod
    def modified_zscore_mad(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        """
        Detect outliers using Modified Z-score based on Median Absolute Deviation (MAD).
        
        Args:
            data: Input data
            threshold: Modified Z-score threshold for outlier detection
        
        Returns:
            Boolean mask (True = outlier, False = valid)
        """
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - median))
        
        if mad == 0:
            return np.zeros(len(data), dtype=bool)
        
        # Modified Z-score: 0.6745 makes MAD consistent with std for normal distribution
        modified_z_scores = 0.6745 * (data - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        
        return outliers
    
    def process_trajectory(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete trajectory processing pipeline.
        
        Args:
            df: DataFrame with 'frame', 'x', 'y' columns
            verbose: Print processing information
        
        Returns:
            Tuple of (cleaned_df, final_df) where:
            - cleaned_df: Data after outlier removal (before interpolation)
            - final_df: Final processed and smoothed trajectory
        """
        result_df = df.copy()
        
        if verbose:
            print(f"  Original points: {len(result_df)}")
        
        # Step 1: Moving median filter
        result_df['x'] = self.moving_median_filter(result_df['x'].values, self.median_window)
        result_df['y'] = self.moving_median_filter(result_df['y'].values, self.median_window)
        if verbose:
            print(f"  ✓ Applied moving median filter (window={self.median_window})")
        
        # Step 2: MAD outlier detection
        outliers_x = self.modified_zscore_mad(result_df['x'].values, self.mad_threshold)
        outliers_y = self.modified_zscore_mad(result_df['y'].values, self.mad_threshold)
        outliers = outliers_x | outliers_y
        num_outliers = np.sum(outliers)
        if verbose:
            print(f"  ✓ Detected outliers: {num_outliers} ({100*num_outliers/len(result_df):.2f}%)")
        
        # Step 3: Remove outliers
        result_df.loc[outliers, 'x'] = np.nan
        result_df.loc[outliers, 'y'] = np.nan
        cleaned_df = result_df.copy()
        
        # Step 4: Interpolate missing values
        try:
            result_df['x'] = result_df['x'].interpolate(method='cubic', limit_direction='both').ffill().bfill()
            result_df['y'] = result_df['y'].interpolate(method='cubic', limit_direction='both').ffill().bfill()
            if verbose:
                print(f"  ✓ Interpolated missing values (cubic)")
        except:
            result_df['x'] = result_df['x'].interpolate(method='linear', limit_direction='both').ffill().bfill()
            result_df['y'] = result_df['y'].interpolate(method='linear', limit_direction='both').ffill().bfill()
            if verbose:
                print(f"  ✓ Interpolated missing values (linear - cubic failed)")
        
        # Step 5: Savitzky-Golay smoothing
        result_df['x'] = savgol_filter(
            result_df['x'],
            window_length=self.savgol_window,
            polyorder=self.savgol_polyorder
        )
        result_df['y'] = savgol_filter(
            result_df['y'],
            window_length=self.savgol_window,
            polyorder=self.savgol_polyorder
        )
        if verbose:
            print(f"  ✓ Applied Savitzky-Golay smoothing (window={self.savgol_window}, poly={self.savgol_polyorder})")
        
        # Round to integers
        result_df['x'] = result_df['x'].round().astype(int)
        result_df['y'] = result_df['y'].round().astype(int)
        
        return cleaned_df, result_df


class TrajectoryReconstructor:
    """
    Stage 2: Reconstruct trajectory on template with proper scaling.
    
    Pipeline:
    1. Load template and calculate dimensions
    2. Filter coordinates by boundary
    3. Calculate scaling factors
    4. Scale coordinates from homography space to template space
    5. Apply resolution smoothing
    """
    
    def __init__(
        self,
        template_path: str,
        boundary_y: int = 135,
        homography_width: float = 41.5,
        homography_height: float = 720,
        resolution_smooth_window: int = 15,
        resolution_smooth_polyorder: int = 3
    ):
        """
        Initialize trajectory reconstructor.
        
        Args:
            template_path: Path to template image
            boundary_y: Y-coordinate of boundary line on template
            homography_width: Width of homography coordinate space
            homography_height: Height of homography coordinate space
            resolution_smooth_window: Window for resolution smoothing
            resolution_smooth_polyorder: Polynomial order for resolution smoothing
        """
        self.template_path = template_path
        self.boundary_y = boundary_y
        self.homography_width = homography_width
        self.homography_height = homography_height
        self.resolution_smooth_window = resolution_smooth_window
        self.resolution_smooth_polyorder = resolution_smooth_polyorder
        
        # Load template
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise FileNotFoundError(f"Template image not found: {template_path}")
        
        self.template_height, self.template_width = self.template.shape[:2]
        
        # Calculate scaling factors
        self.usable_template_height = self.template_height - self.boundary_y
        self.scale_x = self.template_width / self.homography_width
        self.scale_y = self.usable_template_height / self.homography_height
    
    def filter_by_boundary(
        self,
        df: pd.DataFrame,
        boundary_threshold: float = 0.0,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Filter trajectory points by boundary (remove points with Y < threshold).
        
        Args:
            df: DataFrame with 'frame', 'x', 'y' columns
            boundary_threshold: Y threshold in homography space
            verbose: Print filtering information
        
        Returns:
            Filtered DataFrame
        """
        if verbose:
            print(f"  Before filtering: {len(df)} points")
            print(f"    Y range: [{df['y'].min():.1f}, {df['y'].max():.1f}]")
        
        filtered_df = df[df['y'] >= boundary_threshold].reset_index(drop=True)
        
        if verbose:
            print(f"  After filtering (Y >= {boundary_threshold}): {len(filtered_df)} points")
            print(f"    Removed: {len(df) - len(filtered_df)} points")
        
        return filtered_df
    
    def scale_to_template(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Scale coordinates from homography space to template space.
        
        Args:
            df: DataFrame with 'frame', 'x', 'y' columns in homography space
            verbose: Print scaling information
        
        Returns:
            DataFrame with scaled coordinates in template space
        """
        scaled_df = df.copy()
        
        # Apply scaling and offset
        scaled_df['x'] = (scaled_df['x'] * self.scale_x).round().astype(int)
        scaled_df['y'] = ((scaled_df['y'] * self.scale_y) + self.boundary_y).round().astype(int)
        
        if verbose:
            print(f"  ✓ Scaled to template")
            print(f"    X range: [{scaled_df['x'].min()}, {scaled_df['x'].max()}]")
            print(f"    Y range: [{scaled_df['y'].min()}, {scaled_df['y'].max()}]")
        
        return scaled_df
    
    def apply_resolution_smoothing(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Apply Savitzky-Golay smoothing to remove pixelation artifacts from scaling.
        
        Args:
            df: DataFrame with 'frame', 'x', 'y' columns
            verbose: Print smoothing information
        
        Returns:
            DataFrame with smoothed coordinates
        """
        smoothed_df = df.copy()
        
        window_length = self.resolution_smooth_window
        if window_length % 2 == 0:
            window_length += 1
        
        # Apply Savitzky-Golay filter
        smoothed_df['x'] = savgol_filter(
            smoothed_df['x'],
            window_length=window_length,
            polyorder=self.resolution_smooth_polyorder
        )
        smoothed_df['y'] = savgol_filter(
            smoothed_df['y'],
            window_length=window_length,
            polyorder=self.resolution_smooth_polyorder
        )
        
        # Round to integers
        smoothed_df['x'] = smoothed_df['x'].round().astype(int)
        smoothed_df['y'] = smoothed_df['y'].round().astype(int)
        
        if verbose:
            print(f"  ✓ Applied resolution smoothing")
            print(f"    Window: {window_length}, Polyorder: {self.resolution_smooth_polyorder}")
        
        return smoothed_df
    
    def reconstruct_trajectory(
        self,
        df: pd.DataFrame,
        filter_boundary: bool = True,
        apply_smoothing: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Complete trajectory reconstruction pipeline.
        
        Args:
            df: DataFrame with 'frame', 'x', 'y' columns in homography space
            filter_boundary: Whether to filter by boundary
            apply_smoothing: Whether to apply resolution smoothing
            verbose: Print processing information
        
        Returns:
            Final reconstructed trajectory in template space
        """
        result_df = df.copy()
        
        if verbose:
            print("=" * 60)
            print("Trajectory Reconstruction")
            print("=" * 60)
        
        # Step 1: Filter by boundary
        if filter_boundary:
            result_df = self.filter_by_boundary(result_df, verbose=verbose)
        
        # Step 2: Scale to template
        result_df = self.scale_to_template(result_df, verbose=verbose)
        
        # Step 3: Apply resolution smoothing
        if apply_smoothing:
            result_df = self.apply_resolution_smoothing(result_df, verbose=verbose)
        
        if verbose:
            print("=" * 60)
        
        return result_df
    
    def get_template_info(self) -> Dict[str, Any]:
        """
        Get template and scaling information.
        
        Returns:
            Dictionary with template and scaling information
        """
        return {
            'template_path': self.template_path,
            'template_width': self.template_width,
            'template_height': self.template_height,
            'boundary_y': self.boundary_y,
            'usable_height': self.usable_template_height,
            'homography_width': self.homography_width,
            'homography_height': self.homography_height,
            'scale_x': self.scale_x,
            'scale_y': self.scale_y
        }


def process_and_reconstruct(
    trajectory_json_path: str,
    template_path: str,
    output_dir: Optional[str] = None,
    processor_params: Optional[Dict] = None,
    reconstructor_params: Optional[Dict] = None,
    save_outputs: bool = True,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Complete post-processing pipeline: process trajectory and reconstruct on template.
    
    Args:
        trajectory_json_path: Path to input JSON with trajectory data
        template_path: Path to template image
        output_dir: Directory for output files (default: same as input)
        processor_params: Parameters for TrajectoryProcessor
        reconstructor_params: Parameters for TrajectoryReconstructor
        save_outputs: Whether to save output CSVs
        verbose: Print processing information
    
    Returns:
        Dictionary with processed DataFrames:
        - 'raw_trajectory': Original trajectory
        - 'processed_trajectory': After Stage 1 processing
        - 'reconstructed_trajectory': Final trajectory on template
    """
    # Load trajectory data
    if verbose:
        print("=" * 60)
        print("Stage G: Post-Processing")
        print("=" * 60)
        print(f"\nLoading trajectory from: {trajectory_json_path}")
    
    with open(trajectory_json_path, 'r') as f:
        trajectory_data = json.load(f)
    
    # Extract homography trajectory
    transformed_points = trajectory_data['trajectory_points']['transformed']
    raw_df = pd.DataFrame([
        {
            'frame': point['frame_number'],
            'x': point['x'],
            'y': point['y']
        }
        for point in transformed_points
    ])
    
    if verbose:
        print(f"✓ Loaded {len(raw_df)} trajectory points")
    
    # Stage 1: Process trajectory
    if verbose:
        print("\n--- Stage 1: Trajectory Processing ---")
    
    processor = TrajectoryProcessor(**(processor_params or {}))
    cleaned_df, processed_df = processor.process_trajectory(raw_df, verbose=verbose)
    
    # Stage 2: Reconstruct on template
    if verbose:
        print("\n--- Stage 2: Trajectory Reconstruction ---")
    
    reconstructor = TrajectoryReconstructor(
        template_path=template_path,
        **(reconstructor_params or {})
    )
    reconstructed_df = reconstructor.reconstruct_trajectory(
        processed_df,
        verbose=verbose
    )
    
    # Save outputs
    if save_outputs:
        output_path = Path(output_dir) if output_dir else Path(trajectory_json_path).parent
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_csv = output_path / "trajectory_processed.csv"
        reconstructed_csv = output_path / "trajectory_reconstructed.csv"
        
        processed_df.to_csv(processed_csv, index=False)
        reconstructed_df.to_csv(reconstructed_csv, index=False)
        
        if verbose:
            print(f"\n✓ Saved processed trajectory: {processed_csv}")
            print(f"✓ Saved reconstructed trajectory: {reconstructed_csv}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("Post-Processing Complete!")
        print("=" * 60)
    
    return {
        'raw_trajectory': raw_df,
        'processed_trajectory': processed_df,
        'reconstructed_trajectory': reconstructed_df
    }
