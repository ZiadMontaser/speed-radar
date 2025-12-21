"""
Real-time Motion Detection Video Stream

Displays live motion detection results with multiple visualization windows.
Shows original frame, foreground mask, background model, and overlay.

Controls:
- SPACE: Pause/Resume
- Q/ESC: Quit
- R: Reset detector
- S: Save current frame
"""

import cv2
import numpy as np
import yaml
from motion_detection import MotionDetector, MotionDetectionConfig
import argparse
from pathlib import Path
import time


class MotionDetectionStream:
    def __init__(self, video_path: str, config_path: str = "config.yaml", use_mog2: bool = False):
        """Initialize the motion detection stream."""
        self.video_path = video_path
        self.config_path = config_path
        self.use_mog2 = use_mog2
        
        # Load configuration
        self.load_config()
        
        # Initialize detector
        if use_mog2:
            # Use OpenCV's MOG2 background subtractor
            self.detector = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=False
            )
            self.mog2_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            print("✓ Using MOG2 Background Subtractor")
        else:
            self.detector = MotionDetector(config=self.motion_config)
            print("✓ Using Custom Hybrid SDCS Detector")
        
        # Frame history for three-frame differencing (only for custom detector)
        self.prev_frame = None
        self.prev2_frame = None
        
        # Video capture
        self.cap = None
        
        # State
        self.paused = False
        self.frame_count = 0
        self.fps_display = 0.0
        
        # Background model placeholder for MOG2
        self.background = None
        
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            motion_dict = config_dict.get('motion_detection', {})
            self.motion_config = MotionDetectionConfig.from_dict(motion_dict)
            print(f"✓ Configuration loaded from {self.config_path}")
            
        except FileNotFoundError:
            print(f"⚠ Config file not found, using defaults")
            self.motion_config = MotionDetectionConfig()
    
    def open_video(self):
        """Open video file and print information."""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"❌ Could not open video: {self.video_path}")
        
        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{'='*60}")
        print(f"Video Information:")
        print(f"  Path: {self.video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total Frames: {frame_count}")
        print(f"  Duration: {frame_count/fps:.2f} seconds")
        print(f"{'='*60}\n")
        
        return fps
    
    def create_windows(self):
        """Create display windows."""
        cv2.namedWindow('Original Frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Foreground Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Background Model', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)
        
        # Arrange windows (approximate positioning)
        cv2.moveWindow('Original Frame', 0, 0)
        cv2.moveWindow('Foreground Mask', 640, 0)
        cv2.moveWindow('Background Model', 0, 400)
        cv2.moveWindow('Overlay', 640, 400)
    
    def add_info_overlay(self, frame, text_lines):
        """Add information overlay to frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent background for text
        cv2.rectangle(overlay, (5, 5), (400, 25 + 25*len(text_lines)), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Add text
        y_offset = 25
        for text in text_lines:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return frame
    
    def create_overlay_visualization(self, frame, mask):
        """Create overlay visualization with foreground in green."""
        overlay = frame.copy()
        
        # Apply green color to foreground pixels
        overlay[mask > 0] = [0, 255, 0]
        
        # Blend with original
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return result
    
    def process_frame(self, frame):
        """Process a single frame through motion detection pipeline."""
        if self.use_mog2:
            # Use MOG2 background subtractor
            fg_mask = self.detector.apply(frame)
            
            # Remove shadows (MOG2 marks shadows as 127, foreground as 255)
            # Keep only foreground pixels (255), remove shadows (127) and background (0)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            
            # Clean up the mask
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.mog2_kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.mog2_kernel)
            
            # Get background model from MOG2
            background_display = self.detector.getBackgroundImage()
            if background_display is not None:
                if background_display.ndim == 3:
                    background_display = cv2.cvtColor(background_display, cv2.COLOR_BGR2GRAY)
            else:
                background_display = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Store for display
            self.background = background_display
        else:
            # Use custom detector
            fg_mask = self.detector.compute_foreground_mask(
                frame, self.prev_frame, self.prev2_frame
            )
            
            # Get background model (convert to uint8 for display)
            if self.detector.background is not None:
                background_display = self.detector.background.astype(np.uint8)
            else:
                background_display = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Update frame history
            self.prev2_frame = self.prev_frame
            self.prev_frame = frame.copy()
        
        return fg_mask, background_display
    
    def display_frames(self, frame, fg_mask, background, fps):
        """Display all visualization windows."""
        # Create info text
        info_text = [
            f"Frame: {self.frame_count}",
            f"FPS: {fps:.1f}",
            f"Status: {'PAUSED' if self.paused else 'PLAYING'}"
        ]
        
        # Original with info
        frame_display = self.add_info_overlay(frame.copy(), info_text)
        
        # Foreground mask with info
        fg_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        fg_display = self.add_info_overlay(fg_colored, [
            f"Foreground Pixels: {np.count_nonzero(fg_mask)}",
            f"Coverage: {np.count_nonzero(fg_mask) / fg_mask.size * 100:.2f}%"
        ])
        
        # Background model with info
        bg_colored = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        if self.use_mog2:
            threshold_info = [
                f"Method: MOG2",
                f"Background Updated"
            ]
        else:
            if self.detector.threshold_matrix is not None:
                threshold_info = [
                    f"Method: Custom SDCS",
                    f"Threshold Mean: {self.detector.threshold_matrix.mean():.2f}",
                    f"Threshold Range: [{self.detector.threshold_matrix.min():.1f}, {self.detector.threshold_matrix.max():.1f}]"
                ]
            else:
                threshold_info = ["Threshold: Initializing..."]
        bg_display = self.add_info_overlay(bg_colored, threshold_info)
        
        # Overlay visualization
        overlay_display = self.create_overlay_visualization(frame, fg_mask)
        overlay_display = self.add_info_overlay(overlay_display, [
            "Green = Moving Objects"
        ])
        
        # Show all windows
        cv2.imshow('Original Frame', frame_display)
        cv2.imshow('Foreground Mask', fg_display)
        cv2.imshow('Background Model', bg_display)
        cv2.imshow('Overlay', overlay_display)
    
    def save_frame(self, frame, fg_mask, background):
        """Save current frame and results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        # Save original
        cv2.imwrite(str(output_dir / f"frame_{timestamp}_original.jpg"), frame)
        
        # Save mask
        cv2.imwrite(str(output_dir / f"frame_{timestamp}_mask.jpg"), fg_mask)
        
        # Save background
        cv2.imwrite(str(output_dir / f"frame_{timestamp}_background.jpg"), background)
        
        # Save overlay
        overlay = self.create_overlay_visualization(frame, fg_mask)
        cv2.imwrite(str(output_dir / f"frame_{timestamp}_overlay.jpg"), overlay)
        
        print(f"✓ Saved frame {self.frame_count} to {output_dir}/")
    
    def handle_keys(self, key, frame, fg_mask, background):
        """Handle keyboard input."""
        if key == ord(' '):  # Space - pause/resume
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RESUMED"
            print(f"  {status}")
            
        elif key == ord('r') or key == ord('R'):  # Reset detector
            if self.use_mog2:
                # Recreate MOG2 detector
                self.detector = cv2.createBackgroundSubtractorMOG2(
                    history=500,
                    varThreshold=16,
                    detectShadows=False
                )

            else:
                self.detector.reset()
                self.prev_frame = None
                self.prev2_frame = None
            print("  Detector RESET")
            
        elif key == ord('s') or key == ord('S'):  # Save frame
            self.save_frame(frame, fg_mask, background)
            
        elif key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC - quit
            return False
        
        return True
    
    def run(self):
        """Run the motion detection stream."""
        try:
            # Open video
            video_fps = self.open_video()
            frame_delay = int(1000 / video_fps) if video_fps > 0 else 30
            
            # Create windows
            self.create_windows()
            
            print("Controls:")
            print("  SPACE: Pause/Resume")
            print("  R: Reset detector")
            print("  S: Save current frame")
            print("  Q/ESC: Quit")
            print("\nStarting stream...\n")
            
            # FPS calculation
            fps_start_time = time.time()
            fps_frame_count = 0
            
            while True:
                # Handle pause
                if self.paused:
                    key = cv2.waitKey(100) & 0xFF
                    if not self.handle_keys(key, frame, fg_mask, background):
                        break
                    continue
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("\n✓ End of video reached")
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if self.use_mog2:
                        self.detector = cv2.createBackgroundSubtractorMOG2(
                            history=500,
                            varThreshold=16,
                            detectShadows=False
                        )
                    else:
                        self.detector.reset()
                        self.prev_frame = None
                        self.prev2_frame = None
                    self.frame_count = 0
                    continue
                
                self.frame_count += 1
                
                # Process frame
                fg_mask, background = self.process_frame(frame)
                
                
                # Calculate FPS
                fps_frame_count += 1
                if fps_frame_count >= 10:
                    fps_end_time = time.time()
                    self.fps_display = fps_frame_count / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    fps_frame_count = 0
                
                # Display results
                self.display_frames(frame, fg_mask, background, self.fps_display)
                
                # Handle keyboard input
                key = cv2.waitKey(frame_delay) & 0xFF
                if not self.handle_keys(key, frame, fg_mask, background):
                    break
            
        finally:
            # Cleanup
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("\n✓ Stream ended")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time Motion Detection Video Stream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python motion_detection_stream.py video.mp4
  python motion_detection_stream.py datasets/real_traffic/traffic.mp4 --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        'video',
        type=str,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--mog2',
        action='store_true',
        help='Use MOG2 background subtractor instead of custom detector'
    )
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        return
    
    # Create and run stream
    stream = MotionDetectionStream(args.video, args.config, use_mog2=args.mog2)
    stream.run()


if __name__ == "__main__":
    main()
