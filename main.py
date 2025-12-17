import cv2
import yaml
import time
from collections import deque

from motion_detection import MotionDetector, MotionDetectionConfig
from segmentation import segment_foreground
from tracking import Tracker
from speed_capture import compute_speed, capture_violation
from data_structures import Frame, Calibration

# =========================
# Load config
# =========================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

VIDEO_PATH = "real_traffic/test2.mp4"

# =========================
# Init modules
# =========================
motion_cfg = MotionDetectionConfig.from_dict(config)
motion_detector = MotionDetector(motion_cfg)

tracker = Tracker(config)

calibration = Calibration(
    scale_m_per_pixel=config.get("scale_m_per_pixel"),
    homography=None
)

frame_rate = config["frame_rate"]
speed_limit = config["speed_limit_kmph"]

# Buffer for violation capture
frame_buffer = deque(maxlen=50)

# =========================
# Video
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Failed to open video"

paused = False
frame_idx = 0

print("Press 'q' to quit | 'p' to pause")

# =========================
# Main loop
# =========================
prev_frame = None
prev2_frame = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.time()

        # ---- Motion detection
        fg_mask = motion_detector.compute_foreground_mask(
            frame, prev_frame, prev2_frame
        )

        prev2_frame = prev_frame
        prev_frame = frame.copy()

        # ---- Segmentation
        regions = segment_foreground(fg_mask)

        # ---- Tracking
        tracked_objects = tracker.update(
            regions, frame_idx, timestamp
        )

        # ---- Speed computation on exited tracks
        for obj in tracker.get_exited():
            if obj.speed_m_s is None:
                obj.speed_m_s = compute_speed(
                    obj, calibration, frame_rate
                )
                speed_kmph = obj.speed_m_s * 3.6

                if speed_kmph > speed_limit:
                    capture_violation(
                        obj, list(frame_buffer), config
                    )

        # ---- Draw GUI
        vis = frame.copy()

        for obj in tracked_objects:
            x, y, w, h = obj.bbox
            cx, cy = map(int, obj.centroid)

            cv2.rectangle(
                vis, (x, y), (x + w, y + h), (0, 255, 0), 2
            )
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

            label = f"ID {obj.id}"
            print(obj.speed_m_s)
            if obj.speed_m_s:
                label += f" {obj.speed_m_s*3.6:.1f} km/h"

            cv2.putText(
                vis, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # ---- HUD
        cv2.putText(
            vis,
            f"Frame: {frame_idx} | Active: {len(tracked_objects)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # ---- Show
        cv2.imshow("Traffic Monitoring", vis)
        cv2.imshow("Foreground Mask", fg_mask)

        # ---- Frame buffer for capture
        frame_buffer.append(
            Frame(
                index=frame_idx,
                image=frame.copy(),
                timestamp=timestamp
            )
        )

        frame_idx += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("p"):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
