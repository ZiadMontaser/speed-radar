"""
Quick Violation Viewer

View captured violation images and their metadata.
"""

import cv2
import json
import os
from glob import glob


def view_violations(violations_folder="./violations"):
    """View all violation images with their metadata."""

    # Find all JSON files
    json_files = sorted(glob(os.path.join(violations_folder, "*.json")))

    if not json_files:
        print(f"No violations found in {violations_folder}")
        return

    print(f"\nFound {len(json_files)} violations")
    print("=" * 80)

    for json_file in json_files:
        # Load metadata
        with open(json_file, "r") as f:
            metadata = json.load(f)

        # Print metadata
        print(f"\nViolation ID: {metadata['id']}")
        print(
            f"  Speed: {metadata['speed_kmph']:.2f} km/h ({metadata['speed_m_s']:.2f} m/s)"
        )
        print(
            f"  Frames: {metadata['Fr0']} → {metadata['FrN']} ({metadata['frames_tracked']} frames)"
        )
        print(f"  Trajectory: {metadata['trajectory_length']} points")
        print(f"  Timestamp: {metadata['timestamp']:.2f}s")
        print(f"  Image: {metadata['image_path']}")

        # Load and display image
        img_path = metadata["image_path"].replace(
            "./violations\\\\", violations_folder + "/"
        )
        if os.path.exists(img_path):
            img = cv2.imread(img_path)

            # Also try to load enhanced version
            enhanced_path = metadata.get("enhanced_image_path", "").replace(
                "./violations\\\\", violations_folder + "/"
            )
            if enhanced_path and os.path.exists(enhanced_path):
                enhanced = cv2.imread(enhanced_path)

                # Show both side by side
                h1, w1 = img.shape[:2]
                h2, w2 = enhanced.shape[:2]

                # Resize enhanced to match height of main image
                scale = h1 / h2
                enhanced_resized = cv2.resize(enhanced, (int(w2 * scale), h1))

                combined = cv2.hconcat([img, enhanced_resized])

                window_name = f"Violation {metadata['id']} - {metadata['speed_kmph']:.1f} km/h | Main | Enhanced"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.imshow(window_name, combined)
            else:
                window_name = (
                    f"Violation {metadata['id']} - {metadata['speed_kmph']:.1f} km/h"
                )
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.imshow(window_name, img)

            print(f"\n  Press any key to view next violation, 'q' to quit")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord("q") or key == 27:
                print("\nViewing stopped.")
                break
        else:
            print(f"  Image not found: {img_path}")

    print("\n" + "=" * 80)
    print(f"Viewed violations from {violations_folder}")


if __name__ == "__main__":
    view_violations("./violations")
