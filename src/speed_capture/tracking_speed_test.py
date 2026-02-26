import numpy as np
import cv2
import yaml
import time
from typing import List

from motion_detection import MotionDetector, MotionDetectionConfig
from src.segmentation.segmentation import segment_foreground
from src.tracking.tracking import Tracker
from speed_capture import (
    compute_speed_paper_method,
    capture_violation_paper_method,
    calculate_trajectory_metrics,
)
from common.data_structures import Frame, TrackedObject, Calibration, Region


def load_config_yaml(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def fix_orientation(frame, orientation):
    """Fix video orientation based on metadata."""
    if orientation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if orientation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def draw_tracking_visualization(
    frame: np.ndarray,
    tracked_objects: List[TrackedObject],
    show_trajectory: bool = True,
    show_speed: bool = True,
) -> np.ndarray:
    """
    Draw tracking information on the frame.

    Args:
        frame: Image to draw on
        tracked_objects: List of tracked objects
        show_trajectory: Whether to draw trajectory paths
        show_speed: Whether to show speed information

    Returns:
        Frame with visualizations
    """
    vis = frame.copy()

    for obj in tracked_objects:
        # Draw bounding box
        x, y, w, h = obj.bbox
        color = (0, 255, 0)  # Green for active tracks
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        # Draw centroid
        cx, cy = int(obj.centroid[0]), int(obj.centroid[1])
        cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)

        # Draw trajectory
        if show_trajectory and len(obj.trajectory) > 1:
            points = [
                (int(c[0]), int(c[1])) for _, c in obj.trajectory[-20:]
            ]  # Last 20 points
            for i in range(1, len(points)):
                cv2.line(vis, points[i - 1], points[i], (255, 255, 0), 2)

        # Draw ID and info
        info_lines = [f"ID: {obj.id}"]

        if show_speed and obj.speed_m_s is not None:
            speed_kmph = obj.speed_m_s * 3.6
            info_lines.append(f"Speed: {speed_kmph:.1f} km/h")

        # Trajectory length
        if len(obj.trajectory) >= 2:
            info_lines.append(f"Frames: {len(obj.trajectory)}")

        # Draw text
        y_offset = y - 10
        for line in reversed(info_lines):
            cv2.putText(
                vis, line, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            y_offset -= 20

    return vis


def draw_statistics_panel(
    frame: np.ndarray, stats: dict, frame_idx: int, fps: float
) -> np.ndarray:
    """Draw statistics panel on the frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Create semi-transparent panel
    panel_height = 150
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

    # Draw statistics
    text_lines = [
        f"Frame: {frame_idx}  FPS: {fps:.1f}",
        f"Active Tracks: {stats.get('active_tracks', 0)}",
        f"Total Tracked: {stats.get('total_objects_tracked', 0)}",
        f"Exited: {stats.get('exited_tracks', 0)}",
        f"Lost: {stats.get('lost_tracks', 0)}",
    ]

    y_offset = 25
    for line in text_lines:
        cv2.putText(
            vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        y_offset += 25

    return vis


def integrated_tracking_speed_test(
    video_path: str = "datasets/test.mp4",
    config_path: str = "config.yaml",
    display: bool = True,
    save_output: bool = True,
):
    """
    Run integrated tracking and speed capture test.

    Args:
        video_path: Path to test video
        config_path: Path to configuration file
        display: Whether to display real-time visualization
        save_output: Whether to save annotated output video
    """
    print("=" * 80)
    print("Integrated Tracking & Speed Capture Test")
    print("=" * 80)

    # Load configuration
    config_dict = load_config_yaml(config_path)
    motion_config = MotionDetectionConfig.from_dict(config_dict)

    # Initialize modules
    detector = MotionDetector(motion_config)
    tracker = Tracker(config_dict)

    # Setup calibration (using scale from config)
    calibration = Calibration(
        scale_m_per_pixel=config_dict.get("scale_m_per_pixel", 0.02)
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        else -1
    )

    print(f"\nVideo: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {frame_rate:.2f}")
    print(f"Total Frames: {frame_count}")
    print(f"Calibration: {calibration.scale_m_per_pixel} m/pixel")
    print(f"Speed Limit: {config_dict.get('speed_limit_kmph', 60)} km/h")

    # Setup output video writer
    output_writer = None
    if save_output:
        output_path = "output_tracking_speed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_writer = cv2.VideoWriter(
            output_path, fourcc, frame_rate, (width, height)
        )
        print(f"Output will be saved to: {output_path}")

    # Setup display window
    if display:
        win_name = "Tracking & Speed Capture"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Processing state
    frame_idx = 0
    prev_frame = None
    prev2_frame = None
    timestamp = 0.0
    frame_buffer: List[Frame] = []
    max_buffer_size = 30  # Keep last 30 frames for capture

    # Speed limit from config
    speed_limit_kmph = config_dict.get("speed_limit_kmph", 60)
    speed_limit_m_s = speed_limit_kmph / 3.6

    # Statistics
    violation_count = 0
    start_time = time.time()
    fps_calc_interval = 30
    fps_frames = []

    print("\nProcessing... (press 'q' or ESC to quit)")
    print("-" * 80)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = fix_orientation(frame, orientation)
            timestamp = frame_idx / frame_rate

            # Store frame in buffer
            frame_obj = Frame(index=frame_idx, timestamp=timestamp, image=frame.copy())
            frame_buffer.append(frame_obj)
            if len(frame_buffer) > max_buffer_size:
                frame_buffer.pop(0)

            # Motion detection
            if prev_frame is None or prev2_frame is None:
                prev2_frame = prev_frame
                prev_frame = frame.copy()
                frame_idx += 1
                continue

            # Compute foreground mask
            mask = detector.compute_foreground_mask(frame, prev_frame, prev2_frame)

            # Update background model
            detector.update_background_and_threshold(frame, mask)

            # Segmentation
            regions: List[Region] = segment_foreground(mask, config_dict)

            # Tracking
            tracked_objects = tracker.update(regions, frame_idx, timestamp)

            # Speed calculation and violation detection
            for obj in tracked_objects:
                # Only calculate speed if trajectory is long enough
                if len(obj.trajectory) >= config_dict.get("tracking", {}).get(
                    "min_trajectory_length", 5
                ):
                    try:
                        speed = compute_speed_paper_method(
                            obj, calibration, frame_rate, config_dict
                        )
                        obj.speed_m_s = speed

                        # Check for violation
                        if speed > speed_limit_m_s and obj.captured_img_ref is None:
                            print(f"\n🚨 VIOLATION DETECTED!")
                            print(f"   Object ID: {obj.id}")
                            print(
                                f"   Speed: {speed * 3.6:.2f} km/h (Limit: {speed_limit_kmph} km/h)"
                            )
                            print(f"   Frame: {frame_idx}")

                            # Capture violation
                            json_path = capture_violation_paper_method(
                                obj, frame_buffer, config_dict
                            )
                            obj.captured_img_ref = json_path
                            violation_count += 1

                            print(f"   Saved to: {json_path}")

                            # Calculate trajectory metrics
                            metrics = calculate_trajectory_metrics(obj, calibration)
                            print(
                                f"   Trajectory: {metrics['path_length_m']:.2f}m over {len(obj.trajectory)} frames"
                            )
                            print(f"   Tortuosity: {metrics['tortuosity']:.3f}")
                    except Exception as e:
                        print(f"Error calculating speed for object {obj.id}: {e}")

            # Check exited tracks for speed violations
            for obj in tracker.get_exited():
                if obj.speed_m_s is None and len(obj.trajectory) >= 3:
                    try:
                        speed = compute_speed_paper_method(
                            obj, calibration, frame_rate, config_dict
                        )
                        obj.speed_m_s = speed

                        if speed > speed_limit_m_s and obj.captured_img_ref is None:
                            print(f"\n🚨 VIOLATION DETECTED (Exited Object)!")
                            print(f"   Object ID: {obj.id}")
                            print(
                                f"   Speed: {speed * 3.6:.2f} km/h (Limit: {speed_limit_kmph} km/h)"
                            )

                            json_path = capture_violation_paper_method(
                                obj, frame_buffer, config_dict
                            )
                            obj.captured_img_ref = json_path
                            violation_count += 1

                            print(f"   Saved to: {json_path}")
                    except Exception as e:
                        print(
                            f"Error calculating speed for exited object {obj.id}: {e}"
                        )

            # Visualization
            vis_frame = draw_tracking_visualization(
                frame, tracked_objects, show_trajectory=True, show_speed=True
            )

            # Add statistics panel
            stats = tracker.get_statistics()

            # Calculate FPS
            fps_frames.append(time.time())
            if len(fps_frames) > fps_calc_interval:
                fps_frames.pop(0)
            current_fps = (
                len(fps_frames) / (fps_frames[-1] - fps_frames[0])
                if len(fps_frames) > 1
                else 0
            )

            vis_frame = draw_statistics_panel(vis_frame, stats, frame_idx, current_fps)

            # Display
            if display:
                cv2.imshow(win_name, vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    print("\nUser requested quit.")
                    break

            # Save output
            if output_writer:
                output_writer.write(vis_frame)

            # Update for next iteration
            prev2_frame = prev_frame
            prev_frame = frame.copy()
            frame_idx += 1

            # Progress indicator
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Processed {frame_idx} frames in {elapsed:.1f}s "
                    f"({frame_idx/elapsed:.1f} fps) - "
                    f"Active: {stats['active_tracks']}, "
                    f"Total: {stats['total_objects_tracked']}, "
                    f"Violations: {violation_count}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Cleanup
        cap.release()
        if output_writer:
            output_writer.release()
        if display:
            cv2.destroyAllWindows()

        # Final statistics
        elapsed = time.time() - start_time
        stats = tracker.get_statistics()

        print("\n" + "=" * 80)
        print("FINAL STATISTICS")
        print("=" * 80)
        print(f"Total Frames Processed: {frame_idx}")
        print(f"Processing Time: {elapsed:.2f}s")
        print(f"Average FPS: {frame_idx/elapsed:.2f}")
        print(f"\nTracking:")
        print(f"  Total Objects Tracked: {stats['total_objects_tracked']}")
        print(f"  Active Tracks: {stats['active_tracks']}")
        print(f"  Exited Tracks: {stats['exited_tracks']}")
        print(f"  Lost Tracks: {stats['lost_tracks']}")
        print(f"\nSpeed Violations: {violation_count}")
        print("=" * 80)

        # Show all exited tracks with speeds
        print("\nAll Tracked Objects Summary:")
        print("-" * 80)
        print(
            f"{'ID':<5} {'Frames':<8} {'Speed (km/h)':<12} {'Violation':<10} {'Status':<10}"
        )
        print("-" * 80)

        for obj in tracker.get_exited():
            speed_kmph = (obj.speed_m_s * 3.6) if obj.speed_m_s else 0.0
            is_violation = "YES" if speed_kmph > speed_limit_kmph else "NO"
            frames_tracked = len(obj.trajectory)
            status = "Exited"
            print(
                f"{obj.id:<5} {frames_tracked:<8} {speed_kmph:<12.2f} {is_violation:<10} {status:<10}"
            )

        for obj in tracker.get_active():
            speed_kmph = (obj.speed_m_s * 3.6) if obj.speed_m_s else 0.0
            is_violation = "YES" if speed_kmph > speed_limit_kmph else "NO"
            frames_tracked = len(obj.trajectory)
            status = "Active"
            print(
                f"{obj.id:<5} {frames_tracked:<8} {speed_kmph:<12.2f} {is_violation:<10} {status:<10}"
            )

        print("-" * 80)


if __name__ == "__main__":
    # Run the integrated test
    integrated_tracking_speed_test(
        video_path="datasets/test.mp4",
        config_path="config.yaml",
        display=True,
        save_output=True,
    )
