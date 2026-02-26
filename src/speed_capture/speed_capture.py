import cv2
import numpy as np
import os
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional
from common.data_structures import Frame, TrackedObject, Calibration
from common import image_processing as ip


def _apply_homography(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    pt_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
    # dst_array = ip.perspective_transform(pt_array, H)
    dst_array = cv2.perspectiveTransform(pt_array, H)
    return (dst_array[0][0][0], dst_array[0][0][1])


def compute_speed_paper_method(
    tracked_obj: TrackedObject,
    calibration: Calibration,
    frame_rate: float,
    config: dict,
) -> float:
    min_trajectory_length = config.get("tracking", {}).get("min_trajectory_length", 5)
    
    if not tracked_obj.trajectory or len(tracked_obj.trajectory) < min_trajectory_length:
        return 0.0
    
    first_frame_idx, first_pos = tracked_obj.trajectory[0]
    last_frame_idx, last_pos = tracked_obj.trajectory[-1]
    
    frame_span = last_frame_idx - first_frame_idx
    if frame_span <= 0:
        return 0.0
    
    time_elapsed = frame_span / frame_rate
    
    if calibration.homography is not None:
        real_first = _apply_homography(first_pos, calibration.homography)
        real_last = _apply_homography(last_pos, calibration.homography)
        distance_meters = np.sqrt(
            (real_last[0] - real_first[0]) ** 2 + (real_last[1] - real_first[1]) ** 2
        )
    elif calibration.scale_m_per_pixel is not None:
        px_dist = np.sqrt(
            (last_pos[0] - first_pos[0]) ** 2 + (last_pos[1] - first_pos[1]) ** 2
        )
        distance_meters = px_dist * calibration.scale_m_per_pixel
    else:
        raise RuntimeError("Calibration missing! Must provide scale or homography.")

    speed_m_s = distance_meters / time_elapsed if time_elapsed > 0 else 0.0
    
    max_speed_m_s = 200 / 3.6
    if speed_m_s > max_speed_m_s:
        return 0.0
    
    return speed_m_s


def enhance_capture_image(
    frames: List[np.ndarray], bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    if not frames:
        return None

    x, y, w, h = bbox

    cropped = []
    for frame in frames:
        h_frame, w_frame = frame.shape[:2]
        x_safe = max(0, min(x, w_frame - 1))
        y_safe = max(0, min(y, h_frame - 1))
        x2 = max(0, min(x + w, w_frame))
        y2 = max(0, min(y + h, h_frame))

        if x2 > x_safe and y2 > y_safe:
            cropped.append(frame[y_safe:y2, x_safe:x2])

    if not cropped:
        return None

    averaged = np.mean(cropped, axis=0).astype(np.uint8)

    blurred = cv2.GaussianBlur(averaged, (3, 3), 1.0)
    sharpened = cv2.addWeighted(averaged, 1.5, blurred, -0.5, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def capture_violation_paper_method(
    tracked_obj: TrackedObject, frame_buffer: List[Frame], config: dict
) -> str:
    save_folder = config.get("violation_save_folder", "./violations")

    if not os.path.isabs(save_folder):
        project_root = Path(__file__).parent.parent.parent
        save_folder = str(project_root / save_folder)
    os.makedirs(save_folder, exist_ok=True)

    if not tracked_obj.trajectory:
        return None

    start_f = (
        tracked_obj.Fr0 if tracked_obj.Fr0 is not None else tracked_obj.trajectory[0][0]
    )
    end_f = (
        tracked_obj.FrN
        if tracked_obj.FrN is not None
        else tracked_obj.trajectory[-1][0]
    )

    mid_f = int((start_f + end_f) / 2)

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

    frame_height, frame_width = target_frame.image.shape[:2]

    x_curr, y_curr, w_curr, h_curr = tracked_obj.bbox

    target_centroid = None
    for frame_idx, centroid in tracked_obj.trajectory:
        if frame_idx == target_frame.index:
            target_centroid = centroid
            break

    if target_centroid is None:
        target_centroid = tracked_obj.trajectory[-1][1]

    draw_x = int(target_centroid[0] - w_curr / 2)
    draw_y = int(target_centroid[1] - h_curr / 2)

    speed_kmph = (tracked_obj.speed_m_s * 3.6) if tracked_obj.speed_m_s else 0.0

    padding = config.get("capture_padding", 50)
    
    crop_x1 = max(0, draw_x - padding)
    crop_y1 = max(0, draw_y - padding)
    crop_x2 = min(frame_width, draw_x + w_curr + padding)
    crop_y2 = min(frame_height, draw_y + h_curr + padding)
    
    car_crop = target_frame.image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    
    bbox_in_crop_x = draw_x - crop_x1
    bbox_in_crop_y = draw_y - crop_y1
    
    cv2.rectangle(
        car_crop,
        (bbox_in_crop_x, bbox_in_crop_y),
        (bbox_in_crop_x + w_curr, bbox_in_crop_y + h_curr),
        (0, 0, 255), 2
    )
    
    trajectory_points = []
    for frame_idx, centroid in tracked_obj.trajectory:
        if start_f <= frame_idx <= end_f:
            tx = int(centroid[0]) - crop_x1
            ty = int(centroid[1]) - crop_y1

            if 0 <= tx < (crop_x2 - crop_x1) and 0 <= ty < (crop_y2 - crop_y1):
                trajectory_points.append((tx, ty))
    
    if len(trajectory_points) > 1:
        for i in range(1, len(trajectory_points)):
            cv2.line(car_crop, trajectory_points[i - 1], trajectory_points[i], (0, 255, 255), 2)
    
    text_y = max(15, bbox_in_crop_y - 10)
    cv2.putText(
        car_crop,
        f"ID:{tracked_obj.id} {speed_kmph:.1f}km/h",
        (max(0, bbox_in_crop_x), text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )
    
    ts = int(time.time())
    img_name = f"violation_{tracked_obj.id}_{ts}.jpg"
    img_path = os.path.join(save_folder, img_name)
    cv2.imwrite(img_path, car_crop)

    enhanced_path = None
    if len(capture_frames) > 1:
        enhance_padding = 10
        enhance_x1 = max(0, draw_x - enhance_padding)
        enhance_y1 = max(0, draw_y - enhance_padding)
        enhance_bbox = (enhance_x1, enhance_y1, w_curr + 2*enhance_padding, h_curr + 2*enhance_padding)
        enhanced_img = enhance_capture_image(capture_frames, enhance_bbox)

        if enhanced_img is not None and enhanced_img.size > 0:
            enhanced_name = f"violation_{tracked_obj.id}_{ts}_enhanced.jpg"
            enhanced_path = os.path.join(save_folder, enhanced_name)
            cv2.imwrite(enhanced_path, enhanced_img)

    context_path = None
    if config.get("save_full_context", False):
        context_img = target_frame.image.copy()
        
        cv2.rectangle(
            context_img,
            (draw_x, draw_y),
            (draw_x + w_curr, draw_y + h_curr),
            (0, 0, 255), 3
        )
        
        full_traj_points = [
            (int(centroid[0]), int(centroid[1]))
            for frame_idx, centroid in tracked_obj.trajectory
            if start_f <= frame_idx <= end_f
        ]
        if len(full_traj_points) > 1:
            for i in range(1, len(full_traj_points)):
                cv2.line(context_img, full_traj_points[i - 1], full_traj_points[i], (0, 255, 255), 2)
        
        text_lines = [
            f"ID: {tracked_obj.id}",
            f"Speed: {speed_kmph:.1f} km/h",
            f"Frames: {start_f}-{end_f}",
        ]
        y_offset = draw_y - 15
        for line in reversed(text_lines):
            cv2.putText(
                context_img, line, (draw_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            y_offset -= 25
        
        context_name = f"violation_{tracked_obj.id}_{ts}_context.jpg"
        context_path = os.path.join(save_folder, context_name)
        cv2.imwrite(context_path, context_img)

    meta_data = {
        "id": int(tracked_obj.id),
        "speed_kmph": round(float(speed_kmph), 2),
        "speed_m_s": round(float(tracked_obj.speed_m_s), 2) if tracked_obj.speed_m_s else 0.0,
        "timestamp": float(target_frame.timestamp),
        "capture_frame_index": int(target_frame.index),
        "Fr0": int(start_f),
        "FrN": int(end_f),
        "frames_tracked": int(end_f - start_f),
        "trajectory_length": int(len(tracked_obj.trajectory)),
        "bbox_in_frame": [int(draw_x), int(draw_y), int(w_curr), int(h_curr)],
        "crop_region": [int(crop_x1), int(crop_y1), int(crop_x2 - crop_x1), int(crop_y2 - crop_y1)],
        "centroid": [float(target_centroid[0]), float(target_centroid[1])],
        "image_path": img_path,
        "enhanced_image_path": enhanced_path,
        "context_image_path": context_path,
        "method": "paper_based_v3",
    }

    json_path = os.path.join(save_folder, f"violation_{tracked_obj.id}_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(meta_data, f, indent=4)

    return json_path


def calculate_trajectory_metrics(
    tracked_obj: TrackedObject, calibration: Calibration
) -> dict:
    if not tracked_obj.trajectory or len(tracked_obj.trajectory) < 2:
        return {
            "path_length_m": 0.0,
            "straight_line_distance_m": 0.0,
            "tortuosity": 0.0,
            "num_points": 0,
        }

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

    tortuosity = path_length_m / straight_distance_m if straight_distance_m > 0 else 1.0

    return {
        "path_length_m": path_length_m,
        "straight_line_distance_m": straight_distance_m,
        "tortuosity": tortuosity,
        "num_points": len(tracked_obj.trajectory),
    }
