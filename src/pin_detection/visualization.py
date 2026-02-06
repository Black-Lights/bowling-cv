"""
Visualization for Pin Detection

Creates comprehensive visualizations of the pin detection pipeline including
intermediate steps and final results.

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Created: February 6, 2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from . import config


class PinDetectionVisualizer:
    """
    Creates visualizations for pin detection pipeline.
    """
    
    def __init__(self, output_dir):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if config.VERBOSE:
            print(f"\n🎨 Visualizer initialized:")
            print(f"   Output: {output_dir}")
    
    def visualize_difference_pipeline(self, before_frame, after_frame, 
                                     difference, binary_diff, cleaned_diff,
                                     output_path=None):
        """
        Visualize the complete frame differencing pipeline.
        
        Parameters:
        -----------
        before_frame : numpy.ndarray
            Before frame
        after_frame : numpy.ndarray
            After frame
        difference : numpy.ndarray
            Absolute difference
        binary_diff : numpy.ndarray
            Binary thresholded difference
        cleaned_diff : numpy.ndarray
            Morphologically cleaned difference
        output_path : str, optional
            Path to save visualization
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Pin Detection: Frame Differencing Pipeline', 
                    fontsize=16, fontweight='bold')
        
        # Convert BGR to RGB for matplotlib
        if len(before_frame.shape) == 3:
            before_rgb = cv2.cvtColor(before_frame, cv2.COLOR_BGR2RGB)
            after_rgb = cv2.cvtColor(after_frame, cv2.COLOR_BGR2RGB)
        else:
            before_rgb = before_frame
            after_rgb = after_frame
        
        # Row 1: Input frames and difference
        axes[0, 0].imshow(before_rgb)
        axes[0, 0].set_title('Before Frame\n(All Pins Standing)', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(after_rgb)
        axes[0, 1].set_title('After Frame\n(Pins After Impact)', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(difference, cmap='gray')
        axes[0, 2].set_title(f'Absolute Difference\n(Threshold: {config.DIFFERENCE_THRESHOLD})', 
                            fontsize=12)
        axes[0, 2].axis('off')
        
        # Row 2: Processing steps
        axes[1, 0].imshow(binary_diff, cmap='gray')
        axes[1, 0].set_title('Binary Threshold\n(White = Changed)', fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cleaned_diff, cmap='gray')
        axes[1, 1].set_title(f'Morphology Cleaned\n(Kernel: {config.MORPH_KERNEL_SIZE}x{config.MORPH_KERNEL_SIZE})', 
                            fontsize=12)
        axes[1, 1].axis('off')
        
        # Difference histogram
        axes[1, 2].hist(difference.ravel(), bins=50, color='blue', alpha=0.7)
        axes[1, 2].axvline(x=config.DIFFERENCE_THRESHOLD, color='red', 
                          linestyle='--', label='Threshold')
        axes[1, 2].set_title('Difference Histogram', fontsize=12)
        axes[1, 2].set_xlabel('Pixel Difference Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'difference_pipeline.png')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if config.VERBOSE:
            print(f"   ✅ Difference pipeline visualization: {output_path}")
    
    def visualize_contours(self, after_frame, valid_contours, all_contours, 
                          output_path=None):
        """
        Visualize detected contours and filtering.
        
        Parameters:
        -----------
        after_frame : numpy.ndarray
            After frame to draw on
        valid_contours : list
            List of valid pin contours (filtered)
        all_contours : list
            List of all detected contours
        output_path : str, optional
            Path to save visualization
        """
        # Create two versions: all contours vs valid only
        frame_all = after_frame.copy()
        frame_valid = after_frame.copy()
        
        # Draw all contours in red
        for contour in all_contours:
            cv2.drawContours(frame_all, [contour], -1, (0, 0, 255), 2)
        
        # Draw valid contours in green with labels
        for i, pin in enumerate(valid_contours, 1):
            contour = pin['contour']
            center = pin['center']
            bbox = pin['bbox']
            
            # Draw contour
            cv2.drawContours(frame_valid, [contour], -1, config.COLOR_PIN_STANDING, 3)
            
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(frame_valid, (x, y), (x+w, y+h), 
                         config.COLOR_BOUNDING_BOX, 2)
            
            # Draw center point
            cv2.circle(frame_valid, center, 5, (255, 0, 255), -1)
            
            # Label
            label = f"Pin {i}"
            cv2.putText(frame_valid, label, (center[0] - 30, center[1] - 15),
                       config.FONT_FACE, 0.5, config.COLOR_TEXT, 2)
            
            # Area info
            area_label = f"{pin['area']:.0f}px²"
            cv2.putText(frame_valid, area_label, (center[0] - 30, center[1] + 20),
                       config.FONT_FACE, 0.4, (255, 255, 0), 1)
        
        # Create side-by-side comparison
        comparison = np.hstack([frame_all, frame_valid])
        
        # Add labels
        h, w = after_frame.shape[:2]
        cv2.putText(comparison, f"All Contours ({len(all_contours)})", 
                   (10, 30), config.FONT_FACE, 0.8, config.COLOR_TEXT, 2)
        cv2.putText(comparison, f"Valid Pins ({len(valid_contours)})", 
                   (w + 10, 30), config.FONT_FACE, 0.8, config.COLOR_TEXT, 2)
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'contour_detection.png')
        
        cv2.imwrite(output_path, comparison)
        
        if config.VERBOSE:
            print(f"   ✅ Contour visualization: {output_path}")
    
    def visualize_final_result(self, after_frame, results, before_idx, after_idx,
                              output_path=None):
        """
        Create final result visualization with pin annotations.
        
        Parameters:
        -----------
        after_frame : numpy.ndarray
            After frame
        results : dict
            Pin counting results
        before_idx : int
            Before frame index
        after_idx : int
            After frame index
        output_path : str, optional
            Path to save visualization
        """
        result_frame = after_frame.copy()
        h, w = result_frame.shape[:2]
        
        # Draw valid pin contours
        for i, pin in enumerate(results['valid_contours'], 1):
            contour = pin['contour']
            center = pin['center']
            
            # Draw contour
            cv2.drawContours(result_frame, [contour], -1, 
                           config.COLOR_PIN_STANDING, 3)
            
            # Draw pin number
            cv2.circle(result_frame, center, 20, (0, 255, 0), -1)
            cv2.putText(result_frame, str(i), (center[0] - 7, center[1] + 7),
                       config.FONT_FACE, 0.7, (0, 0, 0), 2)
        
        # Create result banner
        banner_height = 120
        banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(banner, "PIN DETECTION RESULTS", (w//2 - 180, 30),
                   config.FONT_FACE, 1.0, config.COLOR_TEXT, 2)
        
        # Result text with color coding
        result_text = results['result']
        if results['is_strike']:
            result_color = (0, 255, 0)  # Green for strike
        elif results['toppled_pins'] == 0:
            result_color = (0, 0, 255)  # Red for gutter
        else:
            result_color = (255, 255, 0)  # Yellow for partial
        
        cv2.putText(banner, result_text, (w//2 - 150, 70),
                   config.FONT_FACE, 1.2, result_color, 3)
        
        # Stats
        stats_text = f"Remaining: {results['remaining_pins']} | Toppled: {results['toppled_pins']}"
        cv2.putText(banner, stats_text, (w//2 - 180, 105),
                   config.FONT_FACE, 0.7, config.COLOR_TEXT, 2)
        
        # Combine banner and frame
        final_vis = np.vstack([banner, result_frame])
        
        # Add frame info
        cv2.putText(final_vis, f"Frames: {before_idx} -> {after_idx}", 
                   (10, banner_height + 30),
                   config.FONT_FACE, 0.5, config.COLOR_TEXT, 1)
        
        # Add confidence
        conf_text = f"Confidence: {results['detection_confidence']:.1%}"
        cv2.putText(final_vis, conf_text, (10, banner_height + 55),
                   config.FONT_FACE, 0.5, config.COLOR_TEXT, 1)
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'pin_detection_result.png')
        
        cv2.imwrite(output_path, final_vis)
        
        if config.VERBOSE:
            print(f"   ✅ Final result visualization: {output_path}")
    
    def create_comparison_panel(self, before_frame, after_frame, difference,
                               binary_diff, cleaned_diff, results,
                               before_idx, after_idx, output_path=None):
        """
        Create comprehensive comparison panel showing all steps.
        
        Parameters:
        -----------
        before_frame : numpy.ndarray
            Before frame
        after_frame : numpy.ndarray
            After frame
        difference : numpy.ndarray
            Difference image
        binary_diff : numpy.ndarray
            Binary difference
        cleaned_diff : numpy.ndarray
            Cleaned difference
        results : dict
            Detection results
        before_idx : int
            Before frame index
        after_idx : int
            After frame index
        output_path : str, optional
            Path to save visualization
        """
        h, w = before_frame.shape[:2]
        
        # Annotate after frame with pins
        after_annotated = after_frame.copy()
        for i, pin in enumerate(results['valid_contours'], 1):
            contour = pin['contour']
            center = pin['center']
            cv2.drawContours(after_annotated, [contour], -1, 
                           config.COLOR_PIN_STANDING, 2)
            cv2.circle(after_annotated, center, 15, (0, 255, 0), -1)
            cv2.putText(after_annotated, str(i), (center[0] - 5, center[1] + 5),
                       config.FONT_FACE, 0.5, (0, 0, 0), 2)
        
        # Convert grayscale to BGR for stacking
        diff_bgr = cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR)
        binary_bgr = cv2.cvtColor(binary_diff, cv2.COLOR_GRAY2BGR)
        cleaned_bgr = cv2.cvtColor(cleaned_diff, cv2.COLOR_GRAY2BGR)
        
        # Add labels to each image
        def add_label(img, text):
            img = img.copy()
            cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(img, text, (10, 28), 
                       config.FONT_FACE, 0.7, config.COLOR_TEXT, 2)
            return img
        
        before_labeled = add_label(before_frame, f"Before (Frame {before_idx})")
        after_labeled = add_label(after_frame, f"After (Frame {after_idx})")
        after_ann_labeled = add_label(after_annotated, 
                                     f"Detected: {len(results['valid_contours'])} pins")
        diff_labeled = add_label(diff_bgr, "Difference")
        binary_labeled = add_label(binary_bgr, "Binary Threshold")
        cleaned_labeled = add_label(cleaned_bgr, "Morphology")
        
        # Create grid: 2 rows x 3 columns
        row1 = np.hstack([before_labeled, after_labeled, after_ann_labeled])
        row2 = np.hstack([diff_labeled, binary_labeled, cleaned_labeled])
        comparison = np.vstack([row1, row2])
        
        # Add title banner
        banner_height = 80
        banner = np.zeros((banner_height, comparison.shape[1], 3), dtype=np.uint8)
        
        cv2.putText(banner, "PIN DETECTION: Complete Pipeline", 
                   (comparison.shape[1]//2 - 280, 35),
                   config.FONT_FACE, 1.2, config.COLOR_TEXT, 2)
        
        result_text = f"{results['result']} - {results['toppled_pins']}/{config.TOTAL_PINS} pins toppled"
        cv2.putText(banner, result_text, 
                   (comparison.shape[1]//2 - 250, 65),
                   config.FONT_FACE, 0.8, (0, 255, 255), 2)
        
        final_comparison = np.vstack([banner, comparison])
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'complete_comparison.png')
        
        cv2.imwrite(output_path, final_comparison)
        
        if config.VERBOSE:
            print(f"   ✅ Complete comparison panel: {output_path}")
    
    def plot_detection_statistics(self, results, output_path=None):
        """
        Create statistical plots for detection analysis.
        
        Parameters:
        -----------
        results : dict
            Detection results
        output_path : str, optional
            Path to save plot
        """
        if len(results['valid_contours']) == 0:
            if config.VERBOSE:
                print(f"   ⚠️  No valid contours to plot statistics")
            return
        
        # Extract contour properties
        areas = [pin['area'] for pin in results['valid_contours']]
        aspects = [pin['aspect_ratio'] for pin in results['valid_contours']]
        solidities = [pin['solidity'] for pin in results['valid_contours']]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Pin Detection: Contour Statistics', 
                    fontsize=14, fontweight='bold')
        
        # Area distribution
        axes[0, 0].hist(areas, bins=10, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=config.MIN_PIN_AREA, color='red', linestyle='--', 
                          label='Min Threshold')
        axes[0, 0].axvline(x=config.MAX_PIN_AREA, color='red', linestyle='--', 
                          label='Max Threshold')
        axes[0, 0].set_title('Pin Area Distribution')
        axes[0, 0].set_xlabel('Area (pixels²)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Aspect ratio distribution
        axes[0, 1].hist(aspects, bins=10, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=config.MIN_PIN_ASPECT_RATIO, color='red', linestyle='--')
        axes[0, 1].axvline(x=config.MAX_PIN_ASPECT_RATIO, color='red', linestyle='--')
        axes[0, 1].set_title('Aspect Ratio Distribution')
        axes[0, 1].set_xlabel('Aspect Ratio (width/height)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Solidity distribution
        axes[1, 0].hist(solidities, bins=10, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=config.MIN_PIN_SOLIDITY, color='red', linestyle='--', 
                          label='Min Threshold')
        axes[1, 0].set_title('Solidity Distribution')
        axes[1, 0].set_xlabel('Solidity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Detection Summary
        ═════════════════════════
        
        Total Pins:        {config.TOTAL_PINS}
        Remaining Pins:    {results['remaining_pins']}
        Toppled Pins:      {results['toppled_pins']}
        
        Result: {results['result']}
        
        Contour Statistics:
        ───────────────────────
        Total Found:       {results['total_contours_found']}
        Valid Pins:        {len(results['valid_contours'])}
        Filtered Out:      {results['total_contours_found'] - len(results['valid_contours'])}
        
        Confidence:        {results['detection_confidence']:.1%}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                       verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'detection_statistics.png')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if config.VERBOSE:
            print(f"   ✅ Detection statistics plot: {output_path}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VISUALIZATION MODULE")
    print("="*80)
    print("\nThis module provides visualization utilities for pin detection.")
    print("Run the main pipeline to see visualizations in action.")
