"""
Post-Processing Module - Stage G

This module provides complete trajectory post-processing functionality:
- Stage 1: Trajectory cleaning (median filter, outlier detection, interpolation, smoothing)
- Stage 1.5: Radius cleaning (exponential decay model, RANSAC fitting, outlier removal)
- Stage 2: Template reconstruction (scaling, boundary filtering, resolution smoothing)

Combines the functionality from:
- ball_trajectory_processing_clean.ipynb
- trajectory_reconstruction.ipynb
- Post_processing_outliers.py (radius processing)

Version: 1.1.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 3, 2026
Updated: February 5, 2026 - Added radius processing
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy.signal import savgol_filter
import json
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any

# Import configuration
try:
    from . import config
except ImportError:
    import config


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
        median_window: int = None,
        mad_threshold: float = None,
        savgol_window: int = None,
        savgol_polyorder: int = None
    ):
        """
        Initialize trajectory processor with processing parameters.
        
        Args:
            median_window: Window size for moving median filter (must be odd). Uses config if None.
            mad_threshold: Modified Z-score threshold for outlier detection. Uses config if None.
            savgol_window: Window length for Savitzky-Golay filter (must be odd). Uses config if None.
            savgol_polyorder: Polynomial order for Savitzky-Golay filter. Uses config if None.
        """
        # Use config defaults if not provided
        self.median_window = median_window if median_window is not None else config.MEDIAN_WINDOW
        self.mad_threshold = mad_threshold if mad_threshold is not None else config.MAD_THRESHOLD
        self.savgol_window = savgol_window if savgol_window is not None else config.SAVGOL_WINDOW
        self.savgol_polyorder = savgol_polyorder if savgol_polyorder is not None else config.SAVGOL_POLYORDER
        
        # Ensure odd windows
        self.median_window = self.median_window if self.median_window % 2 == 1 else self.median_window + 1
        self.savgol_window = self.savgol_window if self.savgol_window % 2 == 1 else self.savgol_window + 1
    
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
        verbose: bool = True,
        return_intermediate: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Complete trajectory processing pipeline.
        
        Args:
            df: DataFrame with 'frame', 'x', 'y' columns
            verbose: Print processing information
            return_intermediate: If True, return intermediate processing steps for visualization
        
        Returns:
            If return_intermediate=False:
                Tuple of (cleaned_df, final_df) where:
                - cleaned_df: Data after outlier removal (before interpolation)
                - final_df: Final processed and smoothed trajectory
            
            If return_intermediate=True:
                Tuple of (cleaned_df, final_df, intermediate_dict) where intermediate_dict contains:
                - 'raw': Original data
                - 'after_median': After median filter applied
                - 'outlier_mask': Boolean mask of detected outliers
                - 'after_interpolation': After interpolation, before Savitzky-Golay
        """
        intermediate = {} if return_intermediate else None
        result_df = df.copy()
        
        if return_intermediate:
            intermediate['raw'] = df.copy()
        
        if verbose:
            print(f"  Original points: {len(result_df)}")
        
        # Step 1: Moving median filter
        result_df['x'] = self.moving_median_filter(result_df['x'].values, self.median_window)
        result_df['y'] = self.moving_median_filter(result_df['y'].values, self.median_window)
        if verbose:
            print(f"  ✓ Applied moving median filter (window={self.median_window})")
        
        if return_intermediate:
            intermediate['after_median'] = result_df.copy()
        
        # Step 2: MAD outlier detection
        outliers_x = self.modified_zscore_mad(result_df['x'].values, self.mad_threshold)
        outliers_y = self.modified_zscore_mad(result_df['y'].values, self.mad_threshold)
        outliers = outliers_x | outliers_y
        num_outliers = np.sum(outliers)
        if verbose:
            print(f"  ✓ Detected outliers: {num_outliers} ({100*num_outliers/len(result_df):.2f}%)")
        
        if return_intermediate:
            intermediate['outlier_mask'] = outliers
        
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
        
        if return_intermediate:
            intermediate['after_interpolation'] = result_df.copy()
        
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
        
        if return_intermediate:
            return cleaned_df, result_df, intermediate
        return cleaned_df, result_df


class RadiusProcessor:
    """
    Stage 1.5: Process and clean radius data using perspective-aware exponential decay model.
    
    Pipeline:
    1. Fill missing frames - Create continuous frame sequence
    2. Moving median filter - Quick baseline smoothing
    3. RANSAC exponential fitting - Model radius perspective decay
    4. Outlier removal - Flag measurements >10px from model
    
    Physical Model:
        radius(frame) = a × exp(-b × frame) + c
    
    Where:
        - a: Initial radius amplitude
        - b: Decay rate (perspective effect)
        - c: Asymptotic baseline (minimum radius at pins)
    
    This accounts for the apparent radius shrinkage as the ball moves away from camera.
    """
    
    def __init__(
        self,
        median_window: int = 21,
        quantile_baseline: float = 0.05,
        ransac_threshold: float = 0.4
    ):
        """
        Initialize radius processor with processing parameters.
        
        Args:
            median_window: Window size for moving median filter (must be odd)
            quantile_baseline: Quantile for baseline estimation (c parameter)
            ransac_threshold: Residual threshold for RANSAC fitting
        """
        self.median_window = median_window if median_window % 2 == 1 else median_window + 1
        self.quantile_baseline = quantile_baseline
        self.ransac_threshold = ransac_threshold
    
    @staticmethod
    def _decreasing_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Exponential decay function to model the radius over time.
        
        Args:
            x: Frame numbers (independent variable)
            a: Amplitude coefficient
            b: Decay rate
            c: Asymptotic baseline
        
        Returns:
            Computed radius values
        """
        return a * np.exp(-b * x) + c
    
    def process_radius(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Process the radius data using RANSAC-based exponential decay fitting.
        
        Args:
            df: DataFrame with 'frame' and 'radius' columns
            verbose: Print processing information
        
        Returns:
            DataFrame with fitted radius values
        """
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input df must be a pandas DataFrame.")
        
        if 'frame' not in df.columns or 'radius' not in df.columns:
            raise ValueError("DataFrame must have 'frame' and 'radius' columns")
        
        # Fill missing frames
        all_frames = pd.DataFrame({
            "frame": np.arange(int(df["frame"].min()), int(df["frame"].max()) + 1)
        })
        df = pd.merge(all_frames, df, on="frame", how="left")
        
        # Moving median filter for baseline
        df["median"] = (
            df["radius"].rolling(window=self.median_window, center=True, min_periods=1).median()
        )
        
        # Estimate baseline (c parameter) using quantile
        c_est = df["radius"].quantile(self.quantile_baseline)
        
        # Filter valid points for fitting (above baseline + epsilon)
        eps = 1e-6
        valid = df["radius"] > (c_est + eps)
        
        if valid.sum() < 10:
            raise ValueError(
                f"Too few valid points for robust fitting ({valid.sum()}/10 minimum). "
                "Consider adjusting parameters."
            )
        
        # Linearize exponential: log(radius - c) = log(a) - b × frame
        x = df.loc[valid, "frame"].values.reshape(-1, 1)
        y = np.log(df.loc[valid, "radius"].values - c_est)
        
        # RANSAC fitting on linearized data
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=self.ransac_threshold,
            max_trials=1000,
            random_state=42
        )
        ransac.fit(x, y)
        
        # Recover exponential parameters
        slope = -ransac.estimator_.coef_[0]  # -b in linearized form
        intercept = ransac.estimator_.intercept_
        a_est = np.exp(intercept)
        
        if verbose:
            inliers = ransac.inlier_mask_.sum()
            total = len(x)
            print(f"  Radius RANSAC fitting:")
            print(f"    Parameters: a={a_est:.2f}, b={slope:.6f}, c={c_est:.2f}")
            print(f"    Inliers: {inliers}/{total} ({100*inliers/total:.1f}%)")
        
        # Generate fitted radius for all frames
        full_x = df["frame"].values
        fitted = self._decreasing_func(full_x, a_est, slope, c_est)
        df["radius"] = np.round(fitted).astype(int)
        
        return df
    
    def delete_outliers(
        self,
        df_original: pd.DataFrame,
        df_fitted: pd.DataFrame,
        threshold: float = 10.0,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Delete outliers based on the processed radius values.
        
        Args:
            df_original: Original DataFrame with 'frame', 'radius', 'x', 'y' columns
            df_fitted: DataFrame with fitted radius values
            threshold: Maximum deviation (pixels) from fitted radius
            verbose: Print filtering information
        
        Returns:
            DataFrame with outliers marked as NaN
        """
        if verbose:
            print(f"  Removing radius outliers (threshold={threshold}px):")
        
        # Merge fitted radius into original DataFrame to ensure alignment
        merged = df_original.merge(
            df_fitted[['frame', 'radius']].rename(columns={'radius': 'radius_fitted'}),
            on='frame',
            how='left'
        )
        
        # Detect outliers using aligned values
        outlier_mask = (
            (merged["radius"] > merged["radius_fitted"] + threshold) |
            (merged["radius"] < merged["radius_fitted"] - threshold)
        )
        
        cleaned_df = df_original.copy()
        cleaned_df.loc[outlier_mask, ["radius", "x", "y"]] = np.nan
        
        if verbose:
            outlier_count = outlier_mask.sum()
            print(f"    Marked {outlier_count} outliers as NaN ({100*outlier_count/len(df_original):.1f}%)")
        
        return cleaned_df


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
        template_path: str = None,
        boundary_y: int = None,
        homography_width: float = None,
        homography_height: float = None,
        resolution_smooth_window: int = None,
        resolution_smooth_polyorder: int = None
    ):
        """
        Initialize trajectory reconstructor.
        
        Args:
            template_path: Path to template image. Uses config if None.
            boundary_y: Y-coordinate of boundary line on template. Uses config if None.
            homography_width: Width of homography coordinate space. Uses config if None.
            homography_height: Height of homography coordinate space. Uses config if None.
            resolution_smooth_window: Window for resolution smoothing. Uses config if None.
            resolution_smooth_polyorder: Polynomial order for resolution smoothing. Uses config if None.
        """
        # Use config defaults if not provided
        self.template_path = template_path if template_path is not None else config.TEMPLATE_PATH
        self.boundary_y = boundary_y if boundary_y is not None else config.BOUNDARY_Y
        self.homography_width = homography_width if homography_width is not None else config.HOMOGRAPHY_WIDTH
        self.homography_height = homography_height if homography_height is not None else config.HOMOGRAPHY_HEIGHT
        self.resolution_smooth_window = resolution_smooth_window if resolution_smooth_window is not None else config.RESOLUTION_SMOOTH_WINDOW
        self.resolution_smooth_polyorder = resolution_smooth_polyorder if resolution_smooth_polyorder is not None else config.RESOLUTION_SMOOTH_POLYORDER
        
        # Load template
        self.template = cv2.imread(str(self.template_path))
        if self.template is None:
            raise FileNotFoundError(f"Template image not found: {self.template_path}")
        
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
        
        # Apply scaling and offset (keep as floats for smooth visualization)
        scaled_df['x'] = scaled_df['x'] * self.scale_x
        scaled_df['y'] = (scaled_df['y'] * self.scale_y) + self.boundary_y
        
        if verbose:
            print(f"  ✓ Scaled to template")
            print(f"    X range: [{scaled_df['x'].min():.1f}, {scaled_df['x'].max():.1f}]")
            print(f"    Y range: [{scaled_df['y'].min():.1f}, {scaled_df['y'].max():.1f}]")
        
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
        
        # Keep as floats for smooth visualization (no rounding)
        
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
    generate_visualizations: bool = True,
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
        generate_visualizations: Whether to generate validation plots (default: True)
        verbose: Print processing information
    
    Returns:
        Dictionary with processed DataFrames:
        - 'raw_trajectory_original': Raw trajectory in original (perspective) coordinates
        - 'raw_trajectory_overhead': Raw trajectory in overhead (homography) coordinates
        - 'processed_trajectory_original': Cleaned trajectory in original coordinates (frame, x, y, radius)
        - 'processed_trajectory_overhead': Cleaned trajectory in overhead coordinates (frame, x, y, radius)
        - 'reconstructed_trajectory': Final trajectory scaled to template (for visualization only)
    """
    # Load trajectory data
    if verbose:
        print("=" * 60)
        print("Stage G: Post-Processing")
        print("=" * 60)
        print(f"\nLoading trajectory from: {trajectory_json_path}")
    
    with open(trajectory_json_path, 'r') as f:
        trajectory_data = json.load(f)
    
    # Extract BOTH original and transformed trajectories with radius
    original_points = trajectory_data['trajectory_points']['original']
    transformed_points = trajectory_data['trajectory_points']['transformed']
    
    # Create DataFrames for both coordinate systems
    raw_df_original = pd.DataFrame([
        {
            'frame': point['frame_number'],
            'x': point['x'],
            'y': point['y'],
            'radius': point.get('radius', 20.0)
        }
        for point in original_points
    ])
    
    raw_df_overhead = pd.DataFrame([
        {
            'frame': point['frame_number'],
            'x': point['x'],
            'y': point['y'],
            'radius': point.get('radius', 20.0)  # Default radius if not present
        }
        for point in transformed_points
    ])
    
    if verbose:
        print(f"✓ Loaded {len(raw_df_original)} original trajectory points (perspective view)")
        print(f"✓ Loaded {len(raw_df_overhead)} overhead trajectory points (homography view)")
    
    # Stage 1: Process ORIGINAL trajectory
    if verbose:
        print("\n--- Stage 1a: Original Trajectory Processing ---")
    
    processor_original = TrajectoryProcessor(**(processor_params or {}))
    
    # Check if we need intermediate results for visualization
    need_intermediate = (
        config.SAVE_MEDIAN_FILTER_PLOT or 
        config.SAVE_MAD_OUTLIER_PLOT or 
        config.SAVE_INTERPOLATION_PLOT
    )
    
    if need_intermediate:
        cleaned_df_original, processed_df_original, intermediate_original = processor_original.process_trajectory(
            raw_df_original, verbose=verbose, return_intermediate=True
        )
    else:
        cleaned_df_original, processed_df_original = processor_original.process_trajectory(
            raw_df_original, verbose=verbose
        )
        intermediate_original = None
    
    # Stage 1b: Process OVERHEAD trajectory
    if verbose:
        print("\n--- Stage 1b: Overhead Trajectory Processing ---")
    
    processor_overhead = TrajectoryProcessor(**(processor_params or {}))
    
    if need_intermediate:
        cleaned_df_overhead, processed_df_overhead, intermediate_overhead = processor_overhead.process_trajectory(
            raw_df_overhead, verbose=verbose, return_intermediate=True
        )
    else:
        cleaned_df_overhead, processed_df_overhead = processor_overhead.process_trajectory(
            raw_df_overhead, verbose=verbose
        )
        intermediate_overhead = None
    
    # Stage 1.5: Process radius (applies to both coordinate systems)
    # Radius is the same in original and overhead (measured once, used in both)
    if 'radius' in raw_df_original.columns:
        if verbose:
            print("\n--- Stage 1.5: Radius Processing ---")
        
        # Create radius processor
        radius_processor = RadiusProcessor(
            median_window=21,
            quantile_baseline=0.05,
            ransac_threshold=0.4
        )
        
        # Process radius using exponential decay model (use original frame data)
        radius_fitted_df = None
        cleaned_radius_df = None
        try:
            radius_fitted_df = radius_processor.process_radius(
                raw_df_original[['frame', 'radius']].copy(),
                verbose=verbose
            )
            
            # Remove outliers
            cleaned_radius_df = radius_processor.delete_outliers(
                raw_df_original[['frame', 'radius', 'x', 'y']].copy(),
                radius_fitted_df,
                threshold=10.0,
                verbose=verbose
            )
            
            # Merge cleaned radius into BOTH processed trajectories
            processed_df_original = pd.merge(
                processed_df_original,
                cleaned_radius_df[['frame', 'radius']],
                on='frame',
                how='left'
            )
            
            processed_df_overhead = pd.merge(
                processed_df_overhead,
                cleaned_radius_df[['frame', 'radius']],
                on='frame',
                how='left'
            )
            
            if verbose:
                print(f"  ✓ Radius cleaning complete (applied to both coordinate systems)")
        
        except Exception as e:
            if verbose:
                print(f"  ⚠ Warning: Radius processing failed: {e}")
                print(f"  Continuing without radius cleaning...")
    
    # Stage 2: Reconstruct on template (uses overhead coordinates)
    if verbose:
        print("\n--- Stage 2: Trajectory Reconstruction ---")
    
    reconstructor = TrajectoryReconstructor(
        template_path=template_path,
        **(reconstructor_params or {})
    )
    reconstructed_df = reconstructor.reconstruct_trajectory(
        processed_df_overhead,
        verbose=verbose
    )
    
    # Save outputs
    if save_outputs:
        output_path = Path(output_dir) if output_dir else Path(trajectory_json_path).parent
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save THREE CSV files
        processed_original_csv = output_path / "trajectory_processed_original.csv"
        processed_overhead_csv = output_path / "trajectory_processed_overhead.csv"
        reconstructed_csv = output_path / "trajectory_reconstructed.csv"
        
        processed_df_original.to_csv(processed_original_csv, index=False)
        processed_df_overhead.to_csv(processed_overhead_csv, index=False)
        reconstructed_df.to_csv(reconstructed_csv, index=False)
        
        if verbose:
            print(f"\n✓ Saved processed original trajectory: {processed_original_csv}")
            print(f"✓ Saved processed overhead trajectory: {processed_overhead_csv}")
            print(f"✓ Saved reconstructed trajectory: {reconstructed_csv}")
    
    # Generate visualizations
    if generate_visualizations and save_outputs:
        output_path = Path(output_dir) if output_dir else Path(trajectory_json_path).parent
        
        if verbose:
            print("\n" + "=" * 60)
            print("Generating Validation Visualizations")
            print("=" * 60)
        
        # Visualize trajectory processing for both coordinate systems
        if config.SAVE_TRAJECTORY_PROCESSING_ORIGINAL_PLOT or config.SAVE_TRAJECTORY_PROCESSING_OVERHEAD_PLOT:
            if verbose:
                print("\n--- Trajectory Processing Visualizations ---")
            
            if config.SAVE_TRAJECTORY_PROCESSING_ORIGINAL_PLOT:
                visualize_trajectory_processing(
                    raw_df_original[['frame', 'x', 'y']],
                    processed_df_original[['frame', 'x', 'y']],
                    output_path,
                    coordinate_system="original",
                    verbose=verbose
                )
            
            if config.SAVE_TRAJECTORY_PROCESSING_OVERHEAD_PLOT:
                visualize_trajectory_processing(
                    raw_df_overhead[['frame', 'x', 'y']],
                    processed_df_overhead[['frame', 'x', 'y']],
                    output_path,
                    coordinate_system="overhead",
                    verbose=verbose
                )
        
        # Visualize intermediate processing steps (median filter, MAD outliers, interpolation)
        if intermediate_original is not None or intermediate_overhead is not None:
            if verbose:
                print("\n--- Intermediate Processing Visualizations ---")
            
            # Median filter visualization
            if config.SAVE_MEDIAN_FILTER_PLOT:
                if intermediate_original is not None:
                    visualize_median_filter_output(
                        intermediate_original['raw'][['frame', 'x', 'y']],
                        intermediate_original['after_median'][['frame', 'x', 'y']],
                        output_path,
                        coordinate_system="original",
                        verbose=verbose
                    )
                
                if intermediate_overhead is not None:
                    visualize_median_filter_output(
                        intermediate_overhead['raw'][['frame', 'x', 'y']],
                        intermediate_overhead['after_median'][['frame', 'x', 'y']],
                        output_path,
                        coordinate_system="overhead",
                        verbose=verbose
                    )
            
            # MAD outlier detection visualization
            if config.SAVE_MAD_OUTLIER_PLOT:
                if intermediate_original is not None:
                    visualize_mad_outlier_output(
                        intermediate_original['after_median'][['frame', 'x', 'y']],
                        intermediate_original['outlier_mask'],
                        output_path,
                        coordinate_system="original",
                        verbose=verbose
                    )
                
                if intermediate_overhead is not None:
                    visualize_mad_outlier_output(
                        intermediate_overhead['after_median'][['frame', 'x', 'y']],
                        intermediate_overhead['outlier_mask'],
                        output_path,
                        coordinate_system="overhead",
                        verbose=verbose
                    )
            
            # Interpolation visualization
            if config.SAVE_INTERPOLATION_PLOT:
                if intermediate_original is not None:
                    visualize_interpolation_output(
                        cleaned_df_original[['frame', 'x', 'y']],  # After outlier removal (has NaN)
                        intermediate_original['after_interpolation'][['frame', 'x', 'y']],
                        intermediate_original['outlier_mask'],
                        output_path,
                        coordinate_system="original",
                        verbose=verbose
                    )
                
                if intermediate_overhead is not None:
                    visualize_interpolation_output(
                        cleaned_df_overhead[['frame', 'x', 'y']],  # After outlier removal (has NaN)
                        intermediate_overhead['after_interpolation'][['frame', 'x', 'y']],
                        intermediate_overhead['outlier_mask'],
                        output_path,
                        coordinate_system="overhead",
                        verbose=verbose
                    )
        
        # Visualize radius processing if available
        if config.SAVE_RADIUS_PROCESSING_PLOT and radius_fitted_df is not None and cleaned_radius_df is not None:
            if verbose:
                print("\n--- Radius Processing Visualization ---")
            
            visualize_radius_processing(
                raw_df_original[['frame', 'radius']],
                radius_fitted_df,
                cleaned_radius_df,
                output_path,
                title_suffix="",
                verbose=verbose
            )
        
        # Visualize trajectory on template (static image)
        if config.SAVE_TRAJECTORY_ON_TEMPLATE_PLOT:
            if verbose:
                print("\n--- Trajectory on Template ---")
            
            visualize_trajectory_on_template(
                reconstructed_df,
                template_path,
                output_path,
                verbose=verbose
            )
        
        # Create trajectory animation video
        if config.SAVE_TRAJECTORY_ANIMATION_VIDEO:
            if verbose:
                print("\n--- Trajectory Animation Video ---")
            
            create_trajectory_animation_video(
                reconstructed_df,
                template_path,
                output_path,
                fps=config.ANIMATION_FPS,
                point_size=config.ANIMATION_POINT_SIZE,
                line_width=config.ANIMATION_LINE_WIDTH,
                trail_length=config.ANIMATION_TRAIL_LENGTH,
                verbose=verbose
            )
    
    if verbose:
        print("\n" + "=" * 60)
        print("Post-Processing Complete!")
        print("=" * 60)
    
    return {
        'raw_trajectory_original': raw_df_original,
        'raw_trajectory_overhead': raw_df_overhead,
        'processed_trajectory_original': processed_df_original,
        'processed_trajectory_overhead': processed_df_overhead,
        'reconstructed_trajectory': reconstructed_df
    }


def visualize_radius_processing(
    raw_radius_df: pd.DataFrame,
    fitted_radius_df: pd.DataFrame,
    cleaned_radius_df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "",
    verbose: bool = True
) -> None:
    """
    Visualize radius processing: raw, fitted, and cleaned radius.
    
    Args:
        raw_radius_df: Original radius data (frame, radius)
        fitted_radius_df: RANSAC fitted radius (frame, radius)
        cleaned_radius_df: After outlier removal (frame, radius, x, y)
        output_path: Directory to save visualization
        title_suffix: Optional suffix for plot title
        verbose: Print save information
    """
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Raw vs Fitted Radius
    plt.subplot(2, 1, 1)
    plt.plot(raw_radius_df['frame'], raw_radius_df['radius'], 
             'o', alpha=0.5, markersize=4, label='Measured Radius', color='lightblue')
    plt.plot(fitted_radius_df['frame'], fitted_radius_df['radius'], 
             '-', linewidth=2, label='Fitted Model (Exponential Decay)', color='red')
    
    # Mark outliers
    outlier_mask = cleaned_radius_df['radius'].isna()
    outlier_frames = cleaned_radius_df.loc[outlier_mask, 'frame']
    outlier_radii = raw_radius_df.loc[raw_radius_df['frame'].isin(outlier_frames), 'radius']
    if len(outlier_frames) > 0:
        plt.scatter(outlier_frames, outlier_radii, 
                   marker='x', s=100, color='red', label='Outliers Flagged', zorder=5)
    
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Radius (pixels)', fontsize=12)
    plt.title(f'Raw Measurements vs RANSAC Fitted Model{title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cleaned Radius (After Outlier Removal)
    plt.subplot(2, 1, 2)
    valid_mask = ~cleaned_radius_df['radius'].isna()
    plt.plot(cleaned_radius_df.loc[valid_mask, 'frame'], 
             cleaned_radius_df.loc[valid_mask, 'radius'],
             'o-', markersize=4, linewidth=1.5, label='Measured (outliers removed)', color='green')
    plt.plot(fitted_radius_df['frame'], fitted_radius_df['radius'], 
             '--', linewidth=2, alpha=0.7, label='Fitted Model (Exponential Decay)', color='orange')
    
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Radius (pixels)', fontsize=12)
    plt.title(f'After Outlier Removal vs Fitted Model{title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / f"radius_processing_visualization{title_suffix.replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved radius visualization: {output_file}")


def visualize_trajectory_processing(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    output_path: Path,
    coordinate_system: str = "overhead",
    verbose: bool = True
) -> None:
    """
    Visualize trajectory processing: raw vs cleaned coordinates.
    
    Args:
        raw_df: Original trajectory (frame, x, y)
        processed_df: Cleaned trajectory (frame, x, y)
        output_path: Directory to save visualization
        coordinate_system: Name of coordinate system ("original" or "overhead")
        verbose: Print save information
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: X coordinate over time
    axes[0, 0].plot(raw_df['frame'], raw_df['x'], 
                    'o', alpha=0.4, markersize=3, label='Raw', color='lightcoral')
    axes[0, 0].plot(processed_df['frame'], processed_df['x'], 
                    '-', linewidth=2, label='Cleaned', color='darkblue')
    axes[0, 0].set_xlabel('Frame Number', fontsize=11)
    axes[0, 0].set_ylabel('X Coordinate (pixels)', fontsize=11)
    axes[0, 0].set_title(f'X Coordinate - {coordinate_system.capitalize()}', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Y coordinate over time
    axes[0, 1].plot(raw_df['frame'], raw_df['y'], 
                    'o', alpha=0.4, markersize=3, label='Raw', color='lightcoral')
    axes[0, 1].plot(processed_df['frame'], processed_df['y'], 
                    '-', linewidth=2, label='Cleaned', color='darkblue')
    axes[0, 1].set_xlabel('Frame Number', fontsize=11)
    axes[0, 1].set_ylabel('Y Coordinate (pixels)', fontsize=11)
    axes[0, 1].set_title(f'Y Coordinate - {coordinate_system.capitalize()}', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: 2D Trajectory (Raw)
    axes[1, 0].plot(raw_df['x'], raw_df['y'], 
                    'o-', alpha=0.6, markersize=4, linewidth=1, color='red', label='Raw')
    axes[1, 0].scatter(raw_df['x'].iloc[0], raw_df['y'].iloc[0], 
                      s=200, marker='o', color='green', label='Start', zorder=5)
    axes[1, 0].scatter(raw_df['x'].iloc[-1], raw_df['y'].iloc[-1], 
                      s=200, marker='s', color='blue', label='End', zorder=5)
    axes[1, 0].set_xlabel('X (pixels)', fontsize=11)
    axes[1, 0].set_ylabel('Y (pixels)', fontsize=11)
    axes[1, 0].set_title(f'Raw Trajectory - {coordinate_system.capitalize()}', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].invert_yaxis()  # Invert Y axis (image coordinates)
    axes[1, 0].set_aspect('equal', adjustable='box')
    
    # Plot 4: 2D Trajectory (Cleaned)
    axes[1, 1].plot(processed_df['x'], processed_df['y'], 
                    'o-', alpha=0.8, markersize=4, linewidth=1.5, color='darkgreen', label='Cleaned')
    axes[1, 1].scatter(processed_df['x'].iloc[0], processed_df['y'].iloc[0], 
                      s=200, marker='o', color='green', label='Start', zorder=5)
    axes[1, 1].scatter(processed_df['x'].iloc[-1], processed_df['y'].iloc[-1], 
                      s=200, marker='s', color='blue', label='End', zorder=5)
    axes[1, 1].set_xlabel('X (pixels)', fontsize=11)
    axes[1, 1].set_ylabel('Y (pixels)', fontsize=11)
    axes[1, 1].set_title(f'Cleaned Trajectory - {coordinate_system.capitalize()}', fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='best', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / f"trajectory_processing_{coordinate_system}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved trajectory visualization: {output_file}")


def visualize_median_filter_output(
    raw_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    output_path: Path,
    coordinate_system: str = "overhead",
    verbose: bool = True
) -> None:
    """
    Visualize the effect of moving median filter on trajectory.
    
    Args:
        raw_df: Original trajectory before median filter (frame, x, y)
        filtered_df: Trajectory after median filter (frame, x, y)
        output_path: Directory to save visualization
        coordinate_system: Name of coordinate system ("original" or "overhead")
        verbose: Print save information
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: X coordinate - Before vs After
    axes[0, 0].plot(raw_df['frame'], raw_df['x'], 
                    'o', alpha=0.3, markersize=4, label='Before Filter', color='lightcoral')
    axes[0, 0].plot(filtered_df['frame'], filtered_df['x'], 
                    '-', linewidth=2, label='After Median Filter', color='darkgreen')
    axes[0, 0].set_xlabel('Frame Number', fontsize=11)
    axes[0, 0].set_ylabel('X Coordinate (pixels)', fontsize=11)
    axes[0, 0].set_title(f'X Coordinate - Median Filter Effect', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Y coordinate - Before vs After
    axes[0, 1].plot(raw_df['frame'], raw_df['y'], 
                    'o', alpha=0.3, markersize=4, label='Before Filter', color='lightcoral')
    axes[0, 1].plot(filtered_df['frame'], filtered_df['y'], 
                    '-', linewidth=2, label='After Median Filter', color='darkgreen')
    axes[0, 1].set_xlabel('Frame Number', fontsize=11)
    axes[0, 1].set_ylabel('Y Coordinate (pixels)', fontsize=11)
    axes[0, 1].set_title(f'Y Coordinate - Median Filter Effect', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Difference in X coordinate
    diff_x = filtered_df['x'].values - raw_df['x'].values
    axes[1, 0].plot(raw_df['frame'], diff_x, 'o-', color='purple', alpha=0.6, markersize=3)
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Frame Number', fontsize=11)
    axes[1, 0].set_ylabel('Δ X (filtered - raw)', fontsize=11)
    axes[1, 0].set_title('Change in X After Filtering', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Difference in Y coordinate
    diff_y = filtered_df['y'].values - raw_df['y'].values
    axes[1, 1].plot(raw_df['frame'], diff_y, 'o-', color='purple', alpha=0.6, markersize=3)
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlabel('Frame Number', fontsize=11)
    axes[1, 1].set_ylabel('Δ Y (filtered - raw)', fontsize=11)
    axes[1, 1].set_title('Change in Y After Filtering', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / f"median_filter_{coordinate_system}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved median filter visualization: {output_file}")


def visualize_mad_outlier_output(
    filtered_df: pd.DataFrame,
    outlier_mask: np.ndarray,
    output_path: Path,
    coordinate_system: str = "overhead",
    verbose: bool = True
) -> None:
    """
    Visualize outliers detected by MAD (Median Absolute Deviation) method.
    
    Args:
        filtered_df: Trajectory after median filter (frame, x, y)
        outlier_mask: Boolean array where True indicates outlier
        output_path: Directory to save visualization
        coordinate_system: Name of coordinate system ("original" or "overhead")
        verbose: Print save information
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Separate inliers and outliers
    inliers = ~outlier_mask
    outliers = outlier_mask
    
    num_outliers = np.sum(outliers)
    total_points = len(outlier_mask)
    outlier_percent = 100 * num_outliers / total_points
    
    # Plot 1: X coordinate with outliers highlighted
    axes[0, 0].plot(filtered_df['frame'][inliers], filtered_df['x'][inliers], 
                    'o', markersize=4, label='Inliers', color='darkgreen', alpha=0.7)
    axes[0, 0].plot(filtered_df['frame'][outliers], filtered_df['x'][outliers], 
                    'x', markersize=8, label=f'Outliers ({num_outliers})', color='red', markeredgewidth=2)
    axes[0, 0].set_xlabel('Frame Number', fontsize=11)
    axes[0, 0].set_ylabel('X Coordinate (pixels)', fontsize=11)
    axes[0, 0].set_title(f'X Coordinate - MAD Outlier Detection', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Y coordinate with outliers highlighted
    axes[0, 1].plot(filtered_df['frame'][inliers], filtered_df['y'][inliers], 
                    'o', markersize=4, label='Inliers', color='darkgreen', alpha=0.7)
    axes[0, 1].plot(filtered_df['frame'][outliers], filtered_df['y'][outliers], 
                    'x', markersize=8, label=f'Outliers ({num_outliers})', color='red', markeredgewidth=2)
    axes[0, 1].set_xlabel('Frame Number', fontsize=11)
    axes[0, 1].set_ylabel('Y Coordinate (pixels)', fontsize=11)
    axes[0, 1].set_title(f'Y Coordinate - MAD Outlier Detection', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: 2D trajectory with outliers highlighted
    axes[1, 0].plot(filtered_df['x'][inliers], filtered_df['y'][inliers], 
                    'o-', markersize=4, linewidth=1, alpha=0.7, color='darkgreen', label='Inliers')
    axes[1, 0].scatter(filtered_df['x'][outliers], filtered_df['y'][outliers], 
                      s=100, marker='x', color='red', linewidths=2, label='Outliers', zorder=5)
    axes[1, 0].set_xlabel('X (pixels)', fontsize=11)
    axes[1, 0].set_ylabel('Y (pixels)', fontsize=11)
    axes[1, 0].set_title(f'2D Trajectory - Outliers Marked', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_aspect('equal', adjustable='box')
    
    # Plot 4: Outlier statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    MAD Outlier Detection Results
    ═══════════════════════════════
    
    Coordinate System: {coordinate_system.capitalize()}
    
    Total Points: {total_points}
    Inliers: {total_points - num_outliers} ({100 - outlier_percent:.2f}%)
    Outliers: {num_outliers} ({outlier_percent:.2f}%)
    
    Status: {'✓ Clean data' if outlier_percent < 5 else '⚠ High outlier rate' if outlier_percent < 15 else '✗ Very noisy data'}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / f"mad_outliers_{coordinate_system}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved MAD outlier visualization: {output_file}")


def visualize_interpolation_output(
    before_interp_df: pd.DataFrame,
    after_interp_df: pd.DataFrame,
    outlier_mask: np.ndarray,
    output_path: Path,
    coordinate_system: str = "overhead",
    verbose: bool = True
) -> None:
    """
    Visualize the effect of interpolation on missing values (removed outliers).
    
    Args:
        before_interp_df: Trajectory with NaN values where outliers were removed (frame, x, y)
        after_interp_df: Trajectory after interpolation (frame, x, y)
        outlier_mask: Boolean array indicating which points were outliers
        output_path: Directory to save visualization
        coordinate_system: Name of coordinate system ("original" or "overhead")
        verbose: Print save information
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Identify interpolated points (where outliers were)
    interpolated_points = outlier_mask
    original_points = ~outlier_mask
    
    num_interpolated = np.sum(interpolated_points)
    
    # Plot 1: X coordinate with interpolated values highlighted
    axes[0, 0].plot(before_interp_df['frame'][original_points], 
                    before_interp_df['x'][original_points], 
                    'o', markersize=4, label='Original Data', color='darkblue', alpha=0.7)
    axes[0, 0].plot(after_interp_df['frame'][interpolated_points], 
                    after_interp_df['x'][interpolated_points], 
                    's', markersize=6, label=f'Interpolated ({num_interpolated})', 
                    color='orange', markeredgecolor='red', markeredgewidth=1.5)
    axes[0, 0].plot(after_interp_df['frame'], after_interp_df['x'], 
                    '-', linewidth=1, alpha=0.4, color='gray', label='Full Curve')
    axes[0, 0].set_xlabel('Frame Number', fontsize=11)
    axes[0, 0].set_ylabel('X Coordinate (pixels)', fontsize=11)
    axes[0, 0].set_title(f'X Coordinate - Interpolation Effect', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Y coordinate with interpolated values highlighted
    axes[0, 1].plot(before_interp_df['frame'][original_points], 
                    before_interp_df['y'][original_points], 
                    'o', markersize=4, label='Original Data', color='darkblue', alpha=0.7)
    axes[0, 1].plot(after_interp_df['frame'][interpolated_points], 
                    after_interp_df['y'][interpolated_points], 
                    's', markersize=6, label=f'Interpolated ({num_interpolated})', 
                    color='orange', markeredgecolor='red', markeredgewidth=1.5)
    axes[0, 1].plot(after_interp_df['frame'], after_interp_df['y'], 
                    '-', linewidth=1, alpha=0.4, color='gray', label='Full Curve')
    axes[0, 1].set_xlabel('Frame Number', fontsize=11)
    axes[0, 1].set_ylabel('Y Coordinate (pixels)', fontsize=11)
    axes[0, 1].set_title(f'Y Coordinate - Interpolation Effect', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: 2D trajectory showing interpolated points
    axes[1, 0].plot(after_interp_df['x'][original_points], 
                    after_interp_df['y'][original_points], 
                    'o', markersize=4, alpha=0.7, color='darkblue', label='Original Data')
    axes[1, 0].scatter(after_interp_df['x'][interpolated_points], 
                      after_interp_df['y'][interpolated_points], 
                      s=80, marker='s', color='orange', edgecolors='red', linewidths=1.5,
                      label='Interpolated Points', zorder=5)
    axes[1, 0].plot(after_interp_df['x'], after_interp_df['y'], 
                    '-', linewidth=1, alpha=0.3, color='gray')
    axes[1, 0].set_xlabel('X (pixels)', fontsize=11)
    axes[1, 0].set_ylabel('Y (pixels)', fontsize=11)
    axes[1, 0].set_title(f'2D Trajectory - Interpolated Points', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_aspect('equal', adjustable='box')
    
    # Plot 4: Interpolation statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    Cubic Interpolation Results
    ════════════════════════════════
    
    Coordinate System: {coordinate_system.capitalize()}
    
    Total Points: {len(outlier_mask)}
    Original Points: {np.sum(original_points)} ({100*np.sum(original_points)/len(outlier_mask):.2f}%)
    Interpolated: {num_interpolated} ({100*num_interpolated/len(outlier_mask):.2f}%)
    
    Method: Cubic spline interpolation
    
    Status: {'✓ Smooth trajectory' if num_interpolated < len(outlier_mask)*0.15 else '⚠ Many interpolated points'}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / f"interpolation_{coordinate_system}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved interpolation visualization: {output_file}")


def visualize_trajectory_on_template(
    reconstructed_df: pd.DataFrame,
    template_path: str,
    output_path: Path,
    verbose: bool = True
) -> None:
    """
    Visualize the reconstructed trajectory overlaid on the bowling lane template.
    
    Args:
        reconstructed_df: Reconstructed trajectory DataFrame (x, y in template coordinates)
        template_path: Path to the template image
        output_path: Directory to save visualization
        verbose: Print save information
    """
    # Load template image
    template = cv2.imread(str(template_path))
    if template is None:
        if verbose:
            print(f"  ⚠ Warning: Could not load template from {template_path}")
        return
    
    # Convert to RGB for matplotlib
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 24))
    
    # Display template
    ax.imshow(template_rgb, aspect='auto')
    
    # Plot trajectory
    trajectory_x = reconstructed_df['x'].values
    trajectory_y = reconstructed_df['y'].values
    
    # Draw trajectory line
    ax.plot(trajectory_x, trajectory_y, 
            color='black', linewidth=3, alpha=0.8, label='Ball Path')
    
    # Mark start point (green)
    ax.scatter(trajectory_x[0], trajectory_y[0], 
              s=200, c='lime', marker='o', edgecolors='darkgreen', 
              linewidth=2, label='Start', zorder=10)
    
    # Mark end point (red)
    ax.scatter(trajectory_x[-1], trajectory_y[-1], 
              s=200, c='red', marker='s', edgecolors='darkred', 
              linewidth=2, label='End', zorder=10)
    
    # Add title and legend
    ax.set_title('Ball Trajectory on Bowling Lane Template', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / "trajectory_on_template.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved trajectory on template: {output_file}")


def create_trajectory_animation_video(
    reconstructed_df: pd.DataFrame,
    template_path: str,
    output_path: Path,
    fps: int = 30,
    point_size: int = 40,
    line_width: int = 3,
    trail_length: int = 0,
    verbose: bool = True
) -> None:
    """
    Create animated video showing trajectory building frame-by-frame on template.
    Shows complete trajectory line from start to current frame (cumulative view).
    
    Args:
        reconstructed_df: Reconstructed trajectory DataFrame (x, y in template coordinates)
        template_path: Path to the template image
        output_path: Directory to save visualization
        fps: Frames per second for output video
        point_size: Size of current point marker
        line_width: Width of trajectory line
        trail_length: Unused (kept for compatibility, trajectory always grows from start)
        verbose: Print save information
    """
    # Load template image
    template = cv2.imread(str(template_path))
    if template is None:
        if verbose:
            print(f"  ⚠ Warning: Could not load template from {template_path}")
        return
    
    h, w = template.shape[:2]
    
    # Prepare output video
    output_file = output_path / "trajectory_animation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_file), fourcc, fps, (w, h))
    
    if not out.isOpened():
        if verbose:
            print(f"  ⚠ Warning: Could not create video writer")
        return
    
    # Get trajectory coordinates
    trajectory_x = reconstructed_df['x'].values
    trajectory_y = reconstructed_df['y'].values
    total_frames = len(trajectory_x)
    
    # Get frame repeat count from config
    frame_repeat = config.ANIMATION_FRAME_REPEAT
    
    if verbose:
        print(f"  Generating trajectory animation: {total_frames} frames at {fps} FPS")
        print(f"  Frame repeat: {frame_repeat}x (slower animation)")
    
    # Generate video frames - show cumulative trajectory
    for i in range(total_frames):
        # Start with fresh template
        frame = template.copy()
        
        # Draw complete trajectory line from start to current point (no points, just line)
        for j in range(i):
            pt1 = (int(trajectory_x[j]), int(trajectory_y[j]))
            pt2 = (int(trajectory_x[j + 1]), int(trajectory_y[j + 1]))
            # Black line for trajectory
            cv2.line(frame, pt1, pt2, (0, 0, 0), line_width)
        
        # Mark start point (green) - always visible
        start_pt = (int(trajectory_x[0]), int(trajectory_y[0]))
        cv2.circle(frame, start_pt, int(point_size * 0.8), (0, 255, 0), -1)
        cv2.circle(frame, start_pt, int(point_size * 0.8), (0, 150, 0), 2)
        cv2.putText(frame, "START", (start_pt[0] + 15, start_pt[1] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw current point (highlighted - larger red circle)
        current_pt = (int(trajectory_x[i]), int(trajectory_y[i]))
        cv2.circle(frame, current_pt, int(point_size * 0.7), (0, 0, 255), -1)
        cv2.circle(frame, current_pt, int(point_size * 0.7), (0, 0, 150), 3)
        
        # Mark end point on final frames
        if i == total_frames - 1:
            cv2.putText(frame, "END", (current_pt[0] + 15, current_pt[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add frame counter and progress
        frame_text = f"Frame: {i + 1}/{total_frames}"
        progress_pct = (i + 1) / total_frames * 100
        progress_text = f"Progress: {progress_pct:.1f}%"
        
        # Frame counter (top-left)
        cv2.putText(frame, frame_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.putText(frame, frame_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        # Progress bar (top-left, below frame counter)
        cv2.putText(frame, progress_text, (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
        cv2.putText(frame, progress_text, (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # Write frame multiple times to slow down animation
        for _ in range(frame_repeat):
            out.write(frame)
    
    # Add a few seconds of hold on final frame
    for _ in range(fps * 2):  # Hold for 2 seconds
        out.write(frame)
    
    out.release()
    
    if verbose:
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        actual_frames = total_frames * frame_repeat + fps * 2
        duration_sec = actual_frames / fps
        print(f"  ✓ Saved trajectory animation: {output_file}")
        print(f"    Size: {file_size_mb:.1f} MB, Duration: {duration_sec:.1f}s")



def visualize_all_processing(
    trajectory_json_path: str,
    output_dir: Optional[str] = None,
    save_visualizations: bool = True,
    verbose: bool = True
) -> None:
    """
    Generate all post-processing visualizations from trajectory JSON.
    
    Args:
        trajectory_json_path: Path to trajectory JSON file
        output_dir: Directory for output files (default: same as input)
        save_visualizations: Whether to save visualization plots
        verbose: Print processing information
    """
    if not save_visualizations:
        return
    
    output_path = Path(output_dir) if output_dir else Path(trajectory_json_path).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Generating Post-Processing Visualizations")
        print("=" * 60)
    
    # Load trajectory JSON
    with open(trajectory_json_path, 'r') as f:
        trajectory_data = json.load(f)
    
    # Extract data
    original_points = trajectory_data['trajectory_points']['original']
    transformed_points = trajectory_data['trajectory_points']['transformed']
    
    # Create raw DataFrames
    raw_df_original = pd.DataFrame([
        {'frame': p['frame_number'], 'x': p['x'], 'y': p['y'], 'radius': p.get('radius', 20.0)}
        for p in original_points
    ])
    
    raw_df_overhead = pd.DataFrame([
        {'frame': p['frame_number'], 'x': p['x'], 'y': p['y'], 'radius': p.get('radius', 20.0)}
        for p in transformed_points
    ])
    
    # Load processed CSVs if they exist
    processed_original_csv = output_path / "trajectory_processed_original.csv"
    processed_overhead_csv = output_path / "trajectory_processed_overhead.csv"
    
    if processed_original_csv.exists() and processed_overhead_csv.exists():
        processed_df_original = pd.read_csv(processed_original_csv)
        processed_df_overhead = pd.read_csv(processed_overhead_csv)
        
        # Visualize trajectories
        if verbose:
            print("\n--- Trajectory Visualizations ---")
        
        visualize_trajectory_processing(
            raw_df_original, processed_df_original,
            output_path, coordinate_system="original", verbose=verbose
        )
        
        visualize_trajectory_processing(
            raw_df_overhead, processed_df_overhead,
            output_path, coordinate_system="overhead", verbose=verbose
        )
        
        # Visualize radius if available
        if 'radius' in processed_df_original.columns and 'radius' in raw_df_original.columns:
            if verbose:
                print("\n--- Radius Visualizations ---")
            
            # We need to recreate fitted radius for visualization
            # Simple approach: use the cleaned radius as proxy for fitted
            # (In actual pipeline, radius_fitted_df is available during processing)
            
            # Check if there are any cleaned radius values
            if processed_df_original['radius'].notna().any():
                # Create a simple fitted model visualization
                # Using processed radius as the "fitted" curve
                visualize_radius_processing(
                    raw_df_original[['frame', 'radius']],
                    processed_df_original[['frame', 'radius']].copy(),  # Use processed as fitted
                    processed_df_original,
                    output_path,
                    title_suffix="",
                    verbose=verbose
                )
        
        if verbose:
            print("\n" + "=" * 60)
            print("Visualization Generation Complete!")
            print("=" * 60)
    else:
        if verbose:
            print("  ⚠ Warning: Processed CSV files not found. Run post-processing first.")
            print(f"    Expected: {processed_original_csv}")
            print(f"    Expected: {processed_overhead_csv}")
