"""
Speed Capture Module v2 - Paper-Based Approach

This implementation follows the paper's methodology more closely:
1. Uses actual trajectory path length (sum of distances between consecutive points)
2. Implements Fr0 to FrN time calculation (entry to exit frames)
3. Enhanced multi-frame capture with averaging for better quality
4. Optional super-resolution enhancement for violation images
5. More robust speed calculation with minimum trajectory validation

Key differences from v1:
- v1: Uses straight-line distance from start to end point
- v2: Uses cumulative path distance along the full trajectory
- v1: Simple single-frame capture
- v2: Multi-frame averaging for image enhancement
"""

import cv2
import numpy as np
import os
import json
import time
from typing import List, Tuple, Optional
from data_structures import Frame, TrackedObject, Calibration


def _apply_homography(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """Transform a point using homography matrix."""
    pt_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst_array = cv2.perspectiveTransform(pt_array, H)
    return (dst_array[0][0][0], dst_array[0][0][1])


def compute_speed_paper_method(
    tracked_obj: TrackedObject,
    calibration: Calibration,
    frame_rate: float,
    config: dict,
) -> float:
    """
    Compute speed using paper's methodology:
    speed = distance / (N * frame_duration)

    Where:
    - distance: cumulative path length along trajectory (not straight-line)
    - N: number of frames from Fr0 to FrN
    - frame_duration: 1 / frame_rate

    This method is more accurate for non-linear trajectories.
    """
    min_trajectory_length = config.get("min_trajectory_length", 3)

    if (
        not tracked_obj.trajectory
        or len(tracked_obj.trajectory) < min_trajectory_length
    ):
        return 0.0

    # Use Fr0 and FrN if available, otherwise use first and last trajectory points
    start_frame = (
        tracked_obj.Fr0 if tracked_obj.Fr0 is not None else tracked_obj.trajectory[0][0]
    )
    end_frame = (
        tracked_obj.FrN
        if tracked_obj.FrN is not None
        else tracked_obj.trajectory[-1][0]
    )

    # Calculate time elapsed in seconds
    N = end_frame - start_frame
    if N == 0:
        return 0.0

    frame_duration = 1.0 / frame_rate
    time_elapsed = N * frame_duration

    # Calculate cumulative distance along trajectory path
    total_distance_meters = 0.0

    for i in range(1, len(tracked_obj.trajectory)):
        frame_idx_prev, pos_prev = tracked_obj.trajectory[i - 1]
        frame_idx_curr, pos_curr = tracked_obj.trajectory[i]

        # Only count segments within Fr0 to FrN range
        if frame_idx_prev < start_frame or frame_idx_curr > end_frame:
            continue

        if calibration.homography is not None:
            # Transform both points to real-world coordinates
            real_prev = _apply_homography(pos_prev, calibration.homography)
            real_curr = _apply_homography(pos_curr, calibration.homography)
            segment_distance = np.sqrt(
                (real_curr[0] - real_prev[0]) ** 2 + (real_curr[1] - real_prev[1]) ** 2
            )
        elif calibration.scale_m_per_pixel is not None:
            # Calculate pixel distance and convert to meters
            px_dist = np.sqrt(
                (pos_curr[0] - pos_prev[0]) ** 2 + (pos_curr[1] - pos_prev[1]) ** 2
            )
            segment_distance = px_dist * calibration.scale_m_per_pixel
        else:
            raise RuntimeError("Calibration missing! Must provide scale or homography.")

        total_distance_meters += segment_distance

    # Speed = distance / time
    speed_m_s = total_distance_meters / time_elapsed if time_elapsed > 0 else 0.0

    return speed_m_s


def enhance_capture_image(
    frames: List[np.ndarray], bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Enhance captured image using multi-frame averaging to reduce blur.

    This implements the paper's suggestion for better image quality:
    - Average multiple frames to reduce motion blur
    - Optionally apply sharpening

    Args:
        frames: List of image frames to average
        bbox: Bounding box (x, y, w, h) to crop

    Returns:
        Enhanced image
    """
    if not frames:
        return None

    x, y, w, h = bbox

    # Crop all frames to the bounding box
    cropped = []
    for frame in frames:
        h_frame, w_frame = frame.shape[:2]
        # Ensure bbox is within image bounds
        x_safe = max(0, min(x, w_frame - 1))
        y_safe = max(0, min(y, h_frame - 1))
        x2 = max(0, min(x + w, w_frame))
        y2 = max(0, min(y + h, h_frame))

        if x2 > x_safe and y2 > y_safe:
            cropped.append(frame[y_safe:y2, x_safe:x2])

    if not cropped:
        return None

    # Average frames to reduce noise and blur
    averaged = np.mean(cropped, axis=0).astype(np.uint8)

    # Optional: Apply unsharp masking for enhancement
    blurred = cv2.GaussianBlur(averaged, (3, 3), 1.0)
    sharpened = cv2.addWeighted(averaged, 1.5, blurred, -0.5, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def capture_violation_paper_method(
    tracked_obj: TrackedObject, frame_buffer: List[Frame], config: dict
) -> str:
    """
    Capture violation image using paper's multi-frame enhancement approach.

    Key improvements over v1:
    1. Captures multiple frames around the object's center position
    2. Applies multi-frame averaging for better image quality
    3. Uses Fr0 and FrN information for better metadata
    4. Includes more detailed trajectory information
    """
    save_folder = config.get("violation_save_folder", "./violations")
    os.makedirs(save_folder, exist_ok=True)

    if not tracked_obj.trajectory:
        return None

    # Determine start and end frames
    start_f = (
        tracked_obj.Fr0 if tracked_obj.Fr0 is not None else tracked_obj.trajectory[0][0]
    )
    end_f = (
        tracked_obj.FrN
        if tracked_obj.FrN is not None
        else tracked_obj.trajectory[-1][0]
    )

    # Find the middle frame for best representation
    mid_f = int((start_f + end_f) / 2)

    # Collect frames around the middle frame for multi-frame enhancement
    num_frames_to_average = config.get("capture_frame_window", 3)
    half_window = num_frames_to_average // 2

    capture_frames = []
    target_frame = None

    for fr in frame_buffer:
        if abs(fr.index - mid_f) <= half_window:
            capture_frames.append(fr.image)
            if (
                abs(fr.index - mid_f) < abs(target_frame.index - mid_f)
                if target_frame
                else True
            ):
                target_frame = fr

    if not target_frame:
        target_frame = frame_buffer[-1]
        capture_frames = [target_frame.image]

    # Get bbox and centroid for the target frame
    x_curr, y_curr, w_curr, h_curr = tracked_obj.bbox

    # Find centroid at target frame
    target_centroid = None
    for frame_idx, centroid in tracked_obj.trajectory:
        if frame_idx == target_frame.index:
            target_centroid = centroid
            break

    if target_centroid is None:
        # Use last known centroid
        target_centroid = tracked_obj.trajectory[-1][1]

    # Calculate bbox position from centroid
    draw_x = int(target_centroid[0] - w_curr / 2)
    draw_y = int(target_centroid[1] - h_curr / 2)

    # Create main visualization image
    img = target_frame.image.copy()

    # Draw bounding box
    cv2.rectangle(
        img, (draw_x, draw_y), (draw_x + w_curr, draw_y + h_curr), (0, 0, 255), 3
    )

    # Draw trajectory path
    trajectory_points = [
        (int(centroid[0]), int(centroid[1]))
        for frame_idx, centroid in tracked_obj.trajectory
        if start_f <= frame_idx <= end_f
    ]

    if len(trajectory_points) > 1:
        for i in range(1, len(trajectory_points)):
            cv2.line(
                img, trajectory_points[i - 1], trajectory_points[i], (0, 255, 255), 2
            )

    # Add speed and ID text
    speed_kmph = (tracked_obj.speed_m_s * 3.6) if tracked_obj.speed_m_s else 0.0
    text_lines = [
        f"ID: {tracked_obj.id}",
        f"Speed: {speed_kmph:.1f} km/h",
        f"Frames: {start_f}-{end_f}",
    ]

    y_offset = draw_y - 15
    for line in reversed(text_lines):
        cv2.putText(
            img, line, (draw_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        y_offset -= 25

    # Save main image
    ts = int(time.time())
    img_name = f"violation_{tracked_obj.id}_{ts}.jpg"
    img_path = os.path.join(save_folder, img_name)
    cv2.imwrite(img_path, img)

    # Create enhanced close-up using multi-frame averaging
    if len(capture_frames) > 1:
        enhanced_bbox = (draw_x, draw_y, w_curr, h_curr)
        enhanced_img = enhance_capture_image(capture_frames, enhanced_bbox)

        if enhanced_img is not None and enhanced_img.size > 0:
            enhanced_name = f"violation_{tracked_obj.id}_{ts}_enhanced.jpg"
            enhanced_path = os.path.join(save_folder, enhanced_name)
            cv2.imwrite(enhanced_path, enhanced_img)
        else:
            enhanced_path = None
    else:
        enhanced_path = None

    # Create metadata
    meta_data = {
        "id": tracked_obj.id,
        "speed_kmph": round(speed_kmph, 2),
        "speed_m_s": round(tracked_obj.speed_m_s, 2) if tracked_obj.speed_m_s else 0.0,
        "timestamp": target_frame.timestamp,
        "capture_frame_index": target_frame.index,
        "Fr0": start_f,
        "FrN": end_f,
        "frames_tracked": end_f - start_f,
        "trajectory_length": len(tracked_obj.trajectory),
        "bbox_in_image": [draw_x, draw_y, w_curr, h_curr],
        "centroid": [float(target_centroid[0]), float(target_centroid[1])],
        "image_path": img_path,
        "enhanced_image_path": enhanced_path,
        "method": "paper_based_v2",
    }

    json_path = os.path.join(save_folder, f"violation_{tracked_obj.id}_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(meta_data, f, indent=4)

    return json_path


def calculate_trajectory_metrics(
    tracked_obj: TrackedObject, calibration: Calibration
) -> dict:
    """
    Calculate detailed trajectory metrics for analysis.

    Returns:
        Dictionary with path_length, straight_line_distance, tortuosity, etc.
    """
    if not tracked_obj.trajectory or len(tracked_obj.trajectory) < 2:
        return {
            "path_length_m": 0.0,
            "straight_line_distance_m": 0.0,
            "tortuosity": 0.0,
            "num_points": 0,
        }

    # Calculate path length (cumulative)
    path_length_m = 0.0
    for i in range(1, len(tracked_obj.trajectory)):
        _, pos_prev = tracked_obj.trajectory[i - 1]
        _, pos_curr = tracked_obj.trajectory[i]

        if calibration.homography is not None:
            real_prev = _apply_homography(pos_prev, calibration.homography)
            real_curr = _apply_homography(pos_curr, calibration.homography)
            segment_distance = np.sqrt(
                (real_curr[0] - real_prev[0]) ** 2 + (real_curr[1] - real_prev[1]) ** 2
            )
        elif calibration.scale_m_per_pixel is not None:
            px_dist = np.sqrt(
                (pos_curr[0] - pos_prev[0]) ** 2 + (pos_curr[1] - pos_prev[1]) ** 2
            )
            segment_distance = px_dist * calibration.scale_m_per_pixel
        else:
            segment_distance = 0.0

        path_length_m += segment_distance

    # Calculate straight-line distance
    start_pos = tracked_obj.trajectory[0][1]
    end_pos = tracked_obj.trajectory[-1][1]

    if calibration.homography is not None:
        real_start = _apply_homography(start_pos, calibration.homography)
        real_end = _apply_homography(end_pos, calibration.homography)
        straight_distance_m = np.sqrt(
            (real_end[0] - real_start[0]) ** 2 + (real_end[1] - real_start[1]) ** 2
        )
    elif calibration.scale_m_per_pixel is not None:
        px_dist = np.sqrt(
            (end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2
        )
        straight_distance_m = px_dist * calibration.scale_m_per_pixel
    else:
        straight_distance_m = 0.0

    # Tortuosity: ratio of path length to straight-line distance
    # tortuosity = 1.0 means straight path, >1.0 means curved
    tortuosity = path_length_m / straight_distance_m if straight_distance_m > 0 else 1.0

    return {
        "path_length_m": path_length_m,
        "straight_line_distance_m": straight_distance_m,
        "tortuosity": tortuosity,
        "num_points": len(tracked_obj.trajectory),
    }
