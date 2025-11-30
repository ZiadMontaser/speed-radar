import cv2
import numpy as np
import os
import json
import time
from typing import List, Tuple, Optional
from data_structures import Frame, TrackedObject, Calibration


def _apply_homography(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:

    pt_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst_array = cv2.perspectiveTransform(pt_array, H)
    return (dst_array[0][0][0], dst_array[0][0][1])


def compute_speed(
    tracked_obj: TrackedObject, calibration: Calibration, frame_rate: float
) -> float:

    if not tracked_obj.trajectory or len(tracked_obj.trajectory) < 2:
        return 0.0

    start_frame, start_pos = tracked_obj.trajectory[0]
    end_frame, end_pos = tracked_obj.trajectory[-1]

    time_elapsed = (end_frame - start_frame) / frame_rate
    if time_elapsed == 0:
        return 0.0

    distance_meters = 0.0

    if calibration.homography is not None:
        real_start = _apply_homography(start_pos, calibration.homography)
        real_end = _apply_homography(end_pos, calibration.homography)
        distance_meters = np.sqrt(
            (real_start[0] - real_end[0]) ** 2 + (real_start[1] - real_end[1]) ** 2
        )

    elif calibration.scale_m_per_pixel is not None:
        px_dist = np.sqrt(
            (start_pos[0] - end_pos[0]) ** 2 + (start_pos[1] - end_pos[1]) ** 2
        )
        distance_meters = px_dist * calibration.scale_m_per_pixel

    else:
        raise RuntimeError("Calibration missing! Must provide scale or homography.")

    return distance_meters / time_elapsed


def capture_violation(
    tracked_obj: TrackedObject, frame_buffer: List[Frame], config: dict
) -> str:
    save_folder = config.get("violation_save_folder", "./violations")
    os.makedirs(save_folder, exist_ok=True)

    start_f = tracked_obj.trajectory[0][0]
    end_f = tracked_obj.trajectory[-1][0]
    mid_f = int((start_f + end_f) / 2)

    best_frame = None
    min_diff = float("inf")
    for fr in frame_buffer:
        diff = abs(fr.index - mid_f)
        if diff < min_diff:
            min_diff = diff
            best_frame = fr

    if best_frame is None:
        best_frame = frame_buffer[-1]

    _, _, w_curr, h_curr = tracked_obj.bbox

    target_centroid = None

    for frame_idx, centroid in tracked_obj.trajectory:
        if frame_idx == best_frame.index:
            target_centroid = centroid
            break

    if target_centroid is None:
        target_centroid = tracked_obj.trajectory[-1][1]

    draw_x = int(target_centroid[0] - w_curr / 2)
    draw_y = int(target_centroid[1] - h_curr / 2)

    img = best_frame.image.copy()

    cv2.rectangle(
        img, (draw_x, draw_y), (draw_x + w_curr, draw_y + h_curr), (0, 0, 255), 2
    )

    speed_kmph = (tracked_obj.speed_m_s * 3.6) if tracked_obj.speed_m_s else 0.0
    text = f"ID:{tracked_obj.id} {speed_kmph:.1f} km/h"
    cv2.putText(
        img, text, (draw_x, draw_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )

    ts = int(time.time())
    img_name = f"violation_{tracked_obj.id}_{ts}.jpg"
    img_path = os.path.join(save_folder, img_name)
    cv2.imwrite(img_path, img)

    meta_data = {
        "id": tracked_obj.id,
        "speed_kmph": round(speed_kmph, 2),
        "timestamp": best_frame.timestamp,
        "capture_frame_index": best_frame.index,
        "bbox_in_image": [
            draw_x,
            draw_y,
            w_curr,
            h_curr,
        ],
        "image_path": img_path,
    }
    json_path = os.path.join(save_folder, f"violation_{tracked_obj.id}_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(meta_data, f, indent=4)

    return json_path
