"""
Motion Detection Module - Usage Examples and Demonstrations

Shows how to use the motion detection system with real video files
and synthetic test scenarios.
"""

import numpy as np
import cv2
import yaml
from motion_detection import MotionDetector, MotionDetectionConfig
import time


def load_config_yaml(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def fix_orientation(frame, orientation):
    if orientation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if orientation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def example_video_processing():
    """
    Example: Process a video file and show a real-time window:
    [ ORIGINAL ] | [ MODIFIED (mask overlay + text) ]
    Quit with 'q' or ESC.
    """
    print("=" * 60)
    print("Example: Video Motion Detection (real-time preview)")
    print("=" * 60)
    
    # Load config / init objects (your existing helpers)
    config_dict = load_config_yaml()
    motion_config = MotionDetectionConfig.from_dict(config_dict)
    detector = MotionDetector(motion_config)
    
    # Open video (change to 0 for webcam)
    video_path = "datasets/YTDown.com_YouTube_Relaxing-highway-traffic_Media_nt3D26lrkho_001_720p.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)

    
    # Video properties
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    
    print(f"Video: {video_path}  Resolution: {width}x{height}  FPS: {frame_rate:.2f}  Frames: {frame_count}")
    
    
    frame_idx = 0
    prev_frame = None
    prev2_frame = None
    timestamp = 0.0
    motion_frames = []
    
    # Window name and wait time
    win_name = "Original | Modified"
    # Keep aspect ratio; allow manual resize while preserving proportions
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    
    print("Processing frames... (press 'q' or ESC to quit)")
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of file / stream closed
            
            frame = fix_orientation(frame, orientation)
            # optional: you may want to resize frames here if extremely large for real-time
            # frame = cv2.resize(frame, (width, height))

            # h, w = frame.shape[:2]
            # half_h = h // 2
            # half_w = w // 2
            # frame = frame[half_h:h, half_w:w]
            # frame = cv2.resize(frame, (960, 540))  # ~720p
            # height, width = frame.shape[:2]
            
            # Compute foreground mask (assumed to be single-channel uint8 mask)
            three_frame_motion, bg_subtraction_mask, combined_mask, shadow_mask = detector.compute_foreground_mask_withSteps(frame, prev_frame, prev2_frame)
            
            # Create a colored visualization of mask and blend with frame
            three_frame_motion_color = cv2.applyColorMap(three_frame_motion, cv2.COLORMAP_JET)
            bg_subtraction_mask_color = cv2.applyColorMap(bg_subtraction_mask, cv2.COLORMAP_JET)
            combined_mask_color = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
            shadow_mask_color = cv2.cvtColor(shadow_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            background_rgb = cv2.cvtColor(detector.background.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            # Add overlay text (frame index, motion pixels, fps estimate)

            
            # Combine in a 2x2 grid with separators to preserve aspect
            separator_v = np.ones((height, 5, 3), dtype=np.uint8) * 255
            row_top = cv2.hconcat([frame, separator_v, background_rgb])
            row_middle = cv2.hconcat([three_frame_motion_color, separator_v, bg_subtraction_mask_color])
            row_bottom = cv2.hconcat([combined_mask_color, separator_v, shadow_mask_color])
            separator_h = np.ones((5, row_top.shape[1], 3), dtype=np.uint8) * 255
            grid = cv2.vconcat([row_top, separator_h, row_middle, separator_h, row_bottom])

            # Show at native resolution; keep window aspect ratio
            if frame_idx == 0:
                cv2.setWindowProperty(win_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(win_name, grid.shape[1], grid.shape[0])

            cv2.imshow(win_name, grid)

            
            # Update frame history
            prev2_frame = prev_frame
            prev_frame = frame.copy()
            frame_idx += 1
            timestamp += 1.0 / frame_rate
            
            wait_ms = max(1, int(round(1000.0 / frame_rate)))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("User requested exit.")
                break
            
            # Optional progress printing (keep minimal to not slow down)
            if frame_idx % 100 == 0 and frame_count > 0:
                print(f"  Processed {frame_idx}/{frame_count} frames...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Summary
    total_elapsed = time.time() - start_time
    avg_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0.0
    print(f"\nResults:")
    print(f"  Frames processed: {frame_idx}")
    print(f"  Avg processing FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    # Run examples
    
    # Example 1: Video processing (if video file exists)
    try:
        example_video_processing()
    except Exception as e:
        print(f"Example 1 skipped: {e}")

    print("\n" + "=" * 60)
    print("Examples completed")
    print("=" * 60)
