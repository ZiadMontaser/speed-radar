"""
Segmentation Iteration Visualization Test

Takes a binary foreground mask image as input and visualizes the iterative 
segmentation process showing:
- (a) Horizontal scan results
- (b) Vertical scan results  
- (c) Combined results (both)

For the first 3 iterations, matching the document figures 6 and 7.

Usage:
    python segmentation_iteration_test.py mask_image.png
    python segmentation_iteration_test.py mask_image.png --iterations 3 --output results/
"""

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
import sys

from segmentation import horizontal_scan, vertical_scan


def create_visualization(mask: np.ndarray, bboxes: List[tuple]) -> np.ndarray:
    """
    Create colored visualization matching document style (red background, white/green regions).
    
    Args:
        mask: Binary foreground mask
        bboxes: List of bounding boxes
        
    Returns:
        Colored visualization image (BGR)
    """
    h, w = mask.shape
    
    # Red background (like document figures)
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:] = [0, 0, 200]  # Red background
    
    # Draw each region alternating white and green
    for i, bbox in enumerate(bboxes):
        x, y, box_w, box_h = bbox
        
        # Alternate colors: white and green
        if i % 2 == 0:
            color = [255, 255, 255]  # White
        else:
            color = [0, 255, 0]  # Green
        
        # Extract region mask
        if y + box_h <= h and x + box_w <= w and y >= 0 and x >= 0:
            region_mask = mask[y:y+box_h, x:x+box_w]
            
            # Color the region where mask is white
            if region_mask.size > 0:
                vis[y:y+box_h, x:x+box_w][region_mask > 0] = color
        
        # Draw black bounding box
        cv2.rectangle(vis, (x, y), (x + box_w, y + box_h), (0, 0, 0), 2)
    
    return vis


def perform_iteration_step(
    mask: np.ndarray,
    current_bboxes: List[tuple],
    config: dict
) -> Tuple[List[tuple], List[tuple], List[tuple]]:
    """
    Perform one complete iteration step: horizontal scan, vertical scan, and combine.
    
    Args:
        mask: Binary foreground mask
        current_bboxes: Current list of bounding boxes
        config: Configuration dictionary
        
    Returns:
        Tuple of (horizontal_results, vertical_results, combined_results)
    """
    # Step 1: Horizontal scan on all current boxes
    horizontal_results = []
    for bbox in current_bboxes:
        h_splits = horizontal_scan(mask, bbox)
        horizontal_results.extend(h_splits)
    
    # Step 2: Vertical scan on horizontal results
    vertical_results = []
    for bbox in horizontal_results:
        v_splits = vertical_scan(mask, bbox)
        vertical_results.extend(v_splits)
    
    # Combined results are the same as vertical results (both applied)
    combined_results = vertical_results
    
    return horizontal_results, vertical_results, combined_results


def visualize_iterations(
    mask: np.ndarray,
    config: dict,
    num_iterations: int = 3,
    output_dir: str = 'segmentation_iterations'
) -> None:
    """
    Visualize the first N iterations of the segmentation process.
    
    Args:
        mask: Binary foreground mask image
        config: Configuration dictionary
        num_iterations: Number of iterations to visualize (default: 3)
        output_dir: Directory to save output images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find initial bounding box (entire foreground)
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        print("ERROR: No foreground pixels found in mask!")
        print("Make sure the input is a binary mask with white (255) foreground.")
        return
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    initial_bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    
    current_bboxes = [initial_bbox]
    
    print("=" * 80)
    print("SEGMENTATION ITERATION VISUALIZATION")
    print("=" * 80)
    print(f"Mask shape: {mask.shape}")
    print(f"Foreground pixels: {np.sum(mask > 0)}")
    print(f"Initial bbox: {initial_bbox}")
    print(f"Iterations to visualize: {num_iterations}")
    print("=" * 80)
    
    # Perform and visualize each iteration
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}:")
        print("-" * 40)
        
        # Perform iteration step
        h_results, v_results, combined_results = perform_iteration_step(
            mask, current_bboxes, config
        )
        
        print(f"  Input regions: {len(current_bboxes)}")
        print(f"  After horizontal scan: {len(h_results)} regions")
        print(f"  After vertical scan: {len(v_results)} regions")
        print(f"  Combined (both): {len(combined_results)} regions")
        
        # Create visualizations
        vis_horizontal = create_visualization(mask, h_results)
        vis_vertical = create_visualization(mask, v_results)
        vis_combined = create_visualization(mask, combined_results)
        
        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # (a) Horizontal scan
        axes[0].imshow(cv2.cvtColor(vis_horizontal, cv2.COLOR_BGR2RGB))
        axes[0].set_title('(a)', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # (b) Vertical scan
        axes[1].imshow(cv2.cvtColor(vis_vertical, cv2.COLOR_BGR2RGB))
        axes[1].set_title('(b)', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        # (c) Combined (both)
        axes[2].imshow(cv2.cvtColor(vis_combined, cv2.COLOR_BGR2RGB))
        axes[2].set_title('(c)', fontsize=16, fontweight='bold')
        axes[2].axis('off')
        
        # Overall title
        ordinal = {1: '1st', 2: '2nd', 3: '3rd'}.get(iteration + 1, f'{iteration + 1}th')
        fig.suptitle(f'Fig. ({iteration + 6}). Object segmentation {ordinal} iteration',
                    fontsize=14, fontweight='normal')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'iteration_{iteration + 1}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")
        
        plt.show()
        
        # Save individual images
        cv2.imwrite(os.path.join(output_dir, f'iter_{iteration + 1}_a_horizontal.png'), vis_horizontal)
        cv2.imwrite(os.path.join(output_dir, f'iter_{iteration + 1}_b_vertical.png'), vis_vertical)
        cv2.imwrite(os.path.join(output_dir, f'iter_{iteration + 1}_c_combined.png'), vis_combined)
        
        # Update for next iteration
        current_bboxes = combined_results
        
        # Check for convergence
        if iteration > 0 and set(combined_results) == set(current_bboxes):
            print(f"\n✓ Convergence detected at iteration {iteration + 1}")
            break
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Final number of regions: {len(combined_results)}")
    print(f"Output saved to: {output_dir}/")
    print("=" * 80)


def load_mask_image(image_path: str) -> np.ndarray:
    """
    Load a binary mask image from file.
    
    Args:
        image_path: Path to mask image file
        
    Returns:
        Binary mask (grayscale, 0 or 255)
    """
    print(f"Loading mask image: {image_path}")
    
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"ERROR: Could not load image from {image_path}")
        sys.exit(1)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        mask = img
    
    # Ensure binary (threshold at 127)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    print(f"  Mask shape: {mask.shape}")
    print(f"  Foreground pixels: {np.sum(mask > 0)}")
    
    return mask


def main():
    """Main entry point - takes mask image as input"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize Segmentation Iterations (like document Figures 6 & 7)',
        epilog='Example: python segmentation_iteration_test.py foreground_mask.png'
    )
    parser.add_argument(
        'mask_image',
        help='Path to binary foreground mask image (white=foreground, black=background)'
    )
    parser.add_argument(
        '--output', '-o',
        default='segmentation_iterations',
        help='Output directory for visualizations (default: segmentation_iterations)'
    )
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=3,
        help='Number of iterations to visualize (default: 3)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SEGMENTATION ITERATION TEST")
    print("=" * 80)
    print(f"Input mask: {args.mask_image}")
    print(f"Output directory: {args.output}")
    print(f"Iterations: {args.iterations}")
    print("=" * 80)
    
    try:
        # Load configuration
        print("\nLoading configuration...")
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Load mask image
        mask = load_mask_image(args.mask_image)
        
        # Save copy of input mask
        os.makedirs(args.output, exist_ok=True)
        cv2.imwrite(os.path.join(args.output, 'input_mask.png'), mask)
        
        # Visualize iterations
        visualize_iterations(mask, config, args.iterations, args.output)
        
        print("\n✅ Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
