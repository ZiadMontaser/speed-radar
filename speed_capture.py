import cv2
import numpy as np
import os
import json
import time
from typing import List, Tuple, Optional
from data_structures import Frame, TrackedObject, Calibration


# =========================
# Geometry helpers
# =========================
def _apply_homography(
    point: Tuple[float, float], H: np.ndarray
) -> Tuple[float, float]:
    pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pt, H)
    return float(dst[0][0][0]), float(dst[0][0][1])


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n


# =========================
# SPEED COMPUTATION (OPTION B)
# =========================
def compute_speed(
    tracked_obj: TrackedObject,
    calibration: Calibration,
    frame_rate: float,
) -> float:
    """
    SDCS+ speed computation (Option B)

    Uses:
    - Fr0 (entry frame)
    - FrN (exit frame)
    - Projected centroid displacement along motion direction
    """

    # ---- Must have exited scene
    if tracked_obj.FrN is None:
        return 0.0

    fr0 = tracked_obj.Fr0
    frn = tracked_obj.FrN

    if frn <= fr0:
        return 0.0

    time_elapsed = (frn - fr0) / frame_rate
    if time_elapsed <= 0:
        return 0.0

    # ---- Entry & exit centroids
    start_pos = tracked_obj.trajectory[0][1]
    end_pos = tracked_obj.trajectory[-1][1]

    # ---- Apply homography if exists
    if calibration.homography is not None:
        start_pos = _apply_homography(start_pos, calibration.homography)
        end_pos = _apply_homography(end_pos, calibration.homography)

    # ---- Motion direction (unit vector)
    delta = np.array(end_pos) - np.array(start_pos)
    norm = np.linalg.norm(delta)
    if norm == 0:
        return 0.0

    direction = delta / norm

    # ---- Project displacement
    distance = np.dot(delta, direction)

    # ---- Convert to meters if needed
    if calibration.homography is None:
        if calibration.scale_m_per_pixel is None:
            raise RuntimeError("Calibration missing scale or homography")
        distance *= calibration.scale_m_per_pixel

    speed_m_s = abs(distance) / time_elapsed
    return speed_m_s

# =========================
# VIOLATION CAPTURE (unchanged, correct)
# =========================
def capture_violation(
    tracked_obj: TrackedObject,
    frame_buffer: List[Frame],
    config: dict,
) -> str:

    save_folder = config.get("violation_save_folder", "./violations")
    os.makedirs(save_folder, exist_ok=True)

    # --- Prefer center-crossing frame if available
    if hasattr(tracked_obj, "center_cross_frame"):
        target_frame_idx = tracked_obj.center_cross_frame
    else:
        target_frame_idx = int(
            (tracked_obj.entry_frame + tracked_obj.exit_frame) / 2
        )

    best_frame = min(
        frame_buffer,
        key=lambda f: abs(f.index - target_frame_idx),
        default=frame_buffer[-1],
    )

    cx, cy = tracked_obj.exit_centroid
    _, _, w, h = tracked_obj.bbox

    draw_x = int(cx - w / 2)
    draw_y = int(cy - h / 2)

    img = best_frame.image.copy()

    cv2.rectangle(
        img,
        (draw_x, draw_y),
        (draw_x + w, draw_y + h),
        (0, 0, 255),
        2,
    )

    speed_kmph = tracked_obj.speed_m_s * 3.6
    text = f"ID:{tracked_obj.id} {speed_kmph:.1f} km/h"

    cv2.putText(
        img,
        text,
        (draw_x, draw_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    ts = int(time.time())
    img_path = os.path.join(
        save_folder, f"violation_{tracked_obj.id}_{ts}.jpg"
    )
    cv2.imwrite(img_path, img)

    meta = {
        "id": tracked_obj.id,
        "speed_kmph": round(speed_kmph, 2),
        "entry_frame": tracked_obj.entry_frame,
        "exit_frame": tracked_obj.exit_frame,
        "timestamp": best_frame.timestamp,
        "image_path": img_path,
    }

    json_path = os.path.join(
        save_folder, f"violation_{tracked_obj.id}_{ts}.json"
    )
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=4)

    return json_path
