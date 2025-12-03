"""
Segmentation Test - Visualize detected cars as colored points

This script tests the segmentation module by processing a video,
detecting moving objects, and displaying each car's centroid as a colored point.
"""

import cv2
import numpy as np
import yaml
from typing import List, Tuple

from data_structures import Region
from motion_detection import MotionDetector, MotionDetectionConfig
from segmentation import segment_foreground


# ============ Visualization ============

def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n distinct colors for visualization."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))
    return colors


def draw_detections(frame: np.ndarray, regions: List[Region], colors: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Draw colored points at each car's centroid.
    
    Args:
        frame: Original BGR frame
        regions: List of detected regions
        colors: List of colors to use
        
    Returns:
        Frame with colored points drawn
    """
    output = frame.copy()
    
    for i, region in enumerate(regions):
        color = colors[i % len(colors)]
        cx, cy = int(region.centroid[0]), int(region.centroid[1])
        
        # Draw a filled circle at centroid
        cv2.circle(output, (cx, cy), 10, color, -1)
        cv2.circle(output, (cx, cy), 10, (255, 255, 255), 2)  # White border
        
        # Optional: Draw bounding box
        x, y, w, h = region.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        
        # Label with area
        label = f"Area: {region.area}"
        cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output


def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize motion detector
    motion_config = MotionDetectionConfig.from_dict(config)
    detector = MotionDetector(motion_config)
    
    # Open video
    video_path = 'real_traffic/input-001.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    print(f"Processing video: {video_path}")
    print(f"Min area threshold: {config['segmentation']['min_area']}")
    print("Press 'q' to quit, 'p' to pause/resume")
    
    # Generate colors for visualization
    colors = generate_colors(20)
    
    prev_frame = None
    prev2_frame = None
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            
            frame_count += 1
            
            # Compute foreground mask
            mask = detector.compute_foreground_mask(frame, prev_frame, prev2_frame)
            
            # Segment foreground into regions
            regions = segment_foreground(mask, config)
            
            # Draw detections
            output = draw_detections(frame, regions, colors)
            
            # Create mask visualization (colorized)
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Add info text
            info_text = f"Frame: {frame_count} | Detections: {len(regions)}"
            cv2.putText(output, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Stack original with detections and mask
            h, w = frame.shape[:2]
            mask_resized = cv2.resize(mask_colored, (w // 2, h // 2))
            
            # Display
            cv2.imshow('Segmentation Test - Detections', output)
            cv2.imshow('Foreground Mask', mask)
            
            # Update previous frames
            prev2_frame = prev_frame
            prev_frame = frame.copy()
        
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
