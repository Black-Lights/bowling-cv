"""
GPU Utilities for Lane Detection
==================================

Provides GPU-accelerated image processing operations with automatic CPU fallback.
Detects CUDA availability and provides optimized implementations for:
- HSV color space conversion
- Sobel edge detection
- Other intensive operations

Author: Mohmmad Umayr Romshoo, Mohammad Ammar Mughees
Date: January 30, 2026
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional


class GPUAccelerator:
    """
    GPU acceleration manager with automatic CPU fallback.
    
    Detects CUDA availability on initialization and provides optimized
    image processing operations. Falls back to CPU implementations if
    GPU is unavailable.
    
    Attributes:
        cuda_available (bool): Whether CUDA is available
        gpu_count (int): Number of CUDA-enabled devices
        use_gpu (bool): Whether to use GPU (based on availability and config)
        device_info (dict): GPU device information
    """
    
    def __init__(self, use_gpu: bool = True, verbose: bool = True):
        """
        Initialize GPU accelerator.
        
        Args:
            use_gpu: Whether to attempt GPU usage (default: True)
            verbose: Whether to print GPU status information (default: True)
        """
        self.cuda_available = False
        self.gpu_count = 0
        self.use_gpu = False
        self.device_info = {}
        self.verbose = verbose
        
        # Check CUDA availability
        if use_gpu and hasattr(cv2, 'cuda'):
            try:
                self.gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
                self.cuda_available = self.gpu_count > 0
                
                if self.cuda_available:
                    self.use_gpu = True
                    # Get device info
                    cv2.cuda.setDevice(0)
                    self.device_info = {
                        'device_id': 0,
                        'device_count': self.gpu_count,
                    }
                    
                    if verbose:
                        print(f"[+] GPU Acceleration ENABLED")
                        print(f"  CUDA devices: {self.gpu_count}")
                        print(f"  Using device: 0")
                else:
                    if verbose:
                        print("[-] GPU Acceleration DISABLED: No CUDA-enabled devices found")
                        
            except Exception as e:
                if verbose:
                    print(f"[-] GPU Acceleration DISABLED: Error checking CUDA - {e}")
        else:
            if verbose:
                if not use_gpu:
                    print("[-] GPU Acceleration DISABLED: Disabled in config")
                else:
                    print("[-] GPU Acceleration DISABLED: OpenCV not built with CUDA support")
    
    def cvtColor_BGR2HSV(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR to HSV color space (GPU-accelerated if available).
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            HSV image as numpy array
        """
        if self.use_gpu:
            try:
                # Upload to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # Convert on GPU
                gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
                
                # Download from GPU
                return gpu_hsv.download()
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: GPU cvtColor failed, falling back to CPU - {e}")
                return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    def Sobel(self, frame: np.ndarray, dx: int, dy: int, 
              ksize: int = 3) -> np.ndarray:
        """
        Apply Sobel edge detection (GPU-accelerated if available).
        
        Args:
            frame: Input image (grayscale or single channel)
            dx: Order of derivative in x
            dy: Order of derivative in y
            ksize: Kernel size (3, 5, or 7)
            
        Returns:
            Sobel filtered image as numpy array
        """
        if self.use_gpu:
            try:
                # Upload to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # Create Sobel filter
                sobel_filter = cv2.cuda.createSobelFilter(
                    cv2.CV_8U, cv2.CV_8U, dx, dy, ksize
                )
                
                # Apply filter on GPU
                gpu_sobel = sobel_filter.apply(gpu_frame)
                
                # Download from GPU
                return gpu_sobel.download()
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: GPU Sobel failed, falling back to CPU - {e}")
                return cv2.Sobel(frame, cv2.CV_8U, dx, dy, ksize=ksize)
        else:
            return cv2.Sobel(frame, cv2.CV_8U, dx, dy, ksize=ksize)
    
    def threshold(self, frame: np.ndarray, thresh: float, maxval: float,
                  thresh_type: int) -> Tuple[float, np.ndarray]:
        """
        Apply threshold operation (GPU-accelerated if available).
        
        Args:
            frame: Input image
            thresh: Threshold value
            maxval: Maximum value to use with THRESH_BINARY types
            thresh_type: Thresholding type (cv2.THRESH_*)
            
        Returns:
            Tuple of (threshold_value, thresholded_image)
        """
        if self.use_gpu:
            try:
                # Upload to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # Apply threshold on GPU
                gpu_thresh = cv2.cuda.threshold(
                    gpu_frame, thresh, maxval, thresh_type
                )[1]
                
                # Download from GPU
                return thresh, gpu_thresh.download()
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: GPU threshold failed, falling back to CPU - {e}")
                return cv2.threshold(frame, thresh, maxval, thresh_type)
        else:
            return cv2.threshold(frame, thresh, maxval, thresh_type)
    
    def inRange(self, frame_hsv: np.ndarray, lower: np.ndarray, 
                upper: np.ndarray) -> np.ndarray:
        """
        Check if array elements lie between two bounds (GPU-accelerated if available).
        
        Args:
            frame_hsv: HSV image
            lower: Lower boundary array
            upper: Upper boundary array
            
        Returns:
            Binary mask
        """
        # Use CPU - inRange is very fast on CPU and GPU overhead isn't worth it
        return cv2.inRange(frame_hsv, lower, upper)
    
    def bitwise_or(self, src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
        """
        Bitwise OR operation (GPU-accelerated if available).
        
        Args:
            src1: First input array
            src2: Second input array
            
        Returns:
            Result array
        """
        if self.use_gpu:
            try:
                # Upload to GPU
                gpu_src1 = cv2.cuda_GpuMat()
                gpu_src2 = cv2.cuda_GpuMat()
                gpu_src1.upload(src1)
                gpu_src2.upload(src2)
                
                # Bitwise OR on GPU
                gpu_result = cv2.cuda.bitwise_or(gpu_src1, gpu_src2)
                
                # Download from GPU
                return gpu_result.download()
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: GPU bitwise_or failed, falling back to CPU - {e}")
                return cv2.bitwise_or(src1, src2)
        else:
            return cv2.bitwise_or(src1, src2)
    
    def bitwise_and(self, src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
        """
        Bitwise AND operation (GPU-accelerated if available).
        
        Args:
            src1: First input array
            src2: Second input array
            
        Returns:
            Result array
        """
        if self.use_gpu:
            try:
                # Upload to GPU
                gpu_src1 = cv2.cuda_GpuMat()
                gpu_src2 = cv2.cuda_GpuMat()
                gpu_src1.upload(src1)
                gpu_src2.upload(src2)
                
                # Bitwise AND on GPU
                gpu_result = cv2.cuda.bitwise_and(gpu_src1, gpu_src2)
                
                # Download from GPU
                return gpu_result.download()
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: GPU bitwise_and failed, falling back to CPU - {e}")
                return cv2.bitwise_and(src1, src2)
        else:
            return cv2.bitwise_and(src1, src2)


class PerformanceTracker:
    """
    Track performance metrics for GPU vs CPU operations.
    
    Provides timing comparisons and speedup calculations to quantify
    the benefits of GPU acceleration.
    """
    
    def __init__(self):
        """Initialize performance tracker."""
        self.timings = {
            'gpu': {},
            'cpu': {}
        }
        self.current_operation = None
        self.current_device = None
        self.start_time = None
    
    def start(self, operation: str, device: str):
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation (e.g., 'cvtColor', 'Sobel')
            device: Device type ('gpu' or 'cpu')
        """
        self.current_operation = operation
        self.current_device = device
        self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop timing and record the result."""
        if self.start_time is None:
            return
        
        elapsed = time.perf_counter() - self.start_time
        
        if self.current_operation not in self.timings[self.current_device]:
            self.timings[self.current_device][self.current_operation] = []
        
        self.timings[self.current_device][self.current_operation].append(elapsed)
        
        self.start_time = None
        self.current_operation = None
        self.current_device = None
    
    def get_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with timing statistics and speedup factors
        """
        stats = {}
        
        for device in ['gpu', 'cpu']:
            stats[device] = {}
            for op, times in self.timings[device].items():
                if times:
                    stats[device][op] = {
                        'count': len(times),
                        'total': sum(times),
                        'mean': np.mean(times),
                        'std': np.std(times),
                        'min': min(times),
                        'max': max(times)
                    }
        
        # Calculate speedup
        stats['speedup'] = {}
        for op in set(list(self.timings['gpu'].keys()) + list(self.timings['cpu'].keys())):
            if op in stats['gpu'] and op in stats['cpu']:
                cpu_mean = stats['cpu'][op]['mean']
                gpu_mean = stats['gpu'][op]['mean']
                stats['speedup'][op] = cpu_mean / gpu_mean if gpu_mean > 0 else 0
        
        return stats
    
    def print_summary(self):
        """Print performance summary."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("GPU PERFORMANCE SUMMARY")
        print("="*60)
        
        if stats.get('speedup'):
            print("\nSpeedup (CPU time / GPU time):")
            for op, speedup in stats['speedup'].items():
                print(f"  {op:20s}: {speedup:6.2f}x")
            
            avg_speedup = np.mean(list(stats['speedup'].values()))
            print(f"\n  Average Speedup: {avg_speedup:6.2f}x")
        
        print("\nDetailed Timings:")
        for device in ['gpu', 'cpu']:
            if stats.get(device):
                print(f"\n{device.upper()}:")
                for op, timing in stats[device].items():
                    print(f"  {op:20s}: {timing['mean']*1000:8.2f} ms "
                          f"(±{timing['std']*1000:.2f} ms, n={timing['count']})")
        
        print("="*60 + "\n")


# Global GPU accelerator instance (initialized on first import)
_gpu_accelerator: Optional[GPUAccelerator] = None
_performance_tracker = PerformanceTracker()


def get_gpu_accelerator(use_gpu: bool = True, verbose: bool = True) -> GPUAccelerator:
    """
    Get or create the global GPU accelerator instance.
    
    Args:
        use_gpu: Whether to attempt GPU usage
        verbose: Whether to print status information
        
    Returns:
        GPUAccelerator instance
    """
    global _gpu_accelerator
    
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator(use_gpu=use_gpu, verbose=verbose)
    
    return _gpu_accelerator


def get_performance_tracker() -> PerformanceTracker:
    """
    Get the global performance tracker instance.
    
    Returns:
        PerformanceTracker instance
    """
    return _performance_tracker
