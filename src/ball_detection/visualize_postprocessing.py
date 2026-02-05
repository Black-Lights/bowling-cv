"""
Visualization script for post-processing validation.

This script helps visualize the results of Stage G post-processing to verify
that trajectory cleaning and radius processing are working correctly.

Usage:
    python visualize_postprocessing.py <trajectory_json_path> <template_path>

Example:
    python visualize_postprocessing.py output/trajectory_data.json assets/bowling_template.png
"""

import sys
from pathlib import Path
from post_processing import process_and_reconstruct, visualize_all_processing


def main():
    """Run post-processing with visualization."""
    if len(sys.argv) < 3:
        print("Usage: python visualize_postprocessing.py <trajectory_json_path> <template_path>")
        print("\nExample:")
        print("  python visualize_postprocessing.py output/trajectory_data.json assets/bowling_template.png")
        sys.exit(1)
    
    trajectory_json_path = sys.argv[1]
    template_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Verify input files exist
    if not Path(trajectory_json_path).exists():
        print(f"Error: Trajectory JSON not found: {trajectory_json_path}")
        sys.exit(1)
    
    if not Path(template_path).exists():
        print(f"Error: Template image not found: {template_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("Post-Processing with Visualization")
    print("=" * 70)
    print(f"\nInput trajectory: {trajectory_json_path}")
    print(f"Template image: {template_path}")
    
    if output_dir:
        print(f"Output directory: {output_dir}")
    else:
        output_dir = Path(trajectory_json_path).parent
        print(f"Output directory: {output_dir} (same as input)")
    
    # Run post-processing with visualizations enabled
    results = process_and_reconstruct(
        trajectory_json_path=trajectory_json_path,
        template_path=template_path,
        output_dir=output_dir,
        save_outputs=True,
        generate_visualizations=True,  # Enable visualizations
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("✓ Post-Processing Complete!")
    print("=" * 70)
    print("\nOutput Files:")
    print("-" * 70)
    
    output_path = Path(output_dir)
    csv_files = [
        output_path / "trajectory_processed_original.csv",
        output_path / "trajectory_processed_overhead.csv",
        output_path / "trajectory_reconstructed.csv"
    ]
    
    png_files = [
        output_path / "trajectory_processing_original.png",
        output_path / "trajectory_processing_overhead.png",
        output_path / "radius_processing_visualization.png"
    ]
    
    print("\nCSV Files (Analysis Data):")
    for csv_file in csv_files:
        if csv_file.exists():
            print(f"  ✓ {csv_file.name}")
        else:
            print(f"  ✗ {csv_file.name} (not found)")
    
    print("\nVisualization Files (PNG):")
    for png_file in png_files:
        if png_file.exists():
            print(f"  ✓ {png_file.name}")
        else:
            print(f"  ✗ {png_file.name} (not found)")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Review visualization PNG files to verify processing quality")
    print("2. Check CSV files for cleaned trajectory data")
    print("3. Use processed_original.csv or processed_overhead.csv for spin analysis")
    print("4. Use reconstructed.csv only for visualization on bowling template")
    print()


if __name__ == "__main__":
    main()
