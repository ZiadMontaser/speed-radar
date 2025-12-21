"""
Speed Detection Camera System - Main Processing Script

This script integrates all four modules: Motion Detection, Segmentation, 
Tracking, and Speed Capture to process video and detect speed violations.

Usage:
    python main.py

Configure MAX_FRAMES to limit processing (set to None for full video).
"""

import cv2
import yaml
import numpy as np
from collections import deque
from tqdm import tqdm
import os

from motion_detection import MotionDetector, MotionDetectionConfig
from segmentation import segment_foreground
from tracking import Tracker
from speed_capture import capture_violation_paper_method, compute_speed_paper_method
from data_structures import Frame, Calibration, TrackedObject


class FrameBuffer:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, frame: Frame):
        self.buffer.append(frame)
        
    def get(self, index: int) -> Frame:
        for f in self.buffer:
            if f.index == index:
                return f
        return None
    
    def last_n(self, n: int) -> list:
        return list(self.buffer)[-n:]
    
    def latest(self) -> Frame:
        return self.buffer[-1] if self.buffer else None


def main():
    # ============ CONFIGURATION ============
    # Set to a number to limit frames, or None to process entire video
    MAX_FRAMES = 2000
    
    VIDEO_PATH = 'datasets/test.mp4'
    OUTPUT_PATH = 'final_output.mp4'
    
    # Load Config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded.")
    
    # Initialize Modules
    md_config = MotionDetectionConfig.from_dict(config)
    motion_detector = MotionDetector(config=md_config)
    
    tracker = Tracker(config)
    
    # Calibration
    calibration = Calibration(
        scale_m_per_pixel=config.get('scale_m_per_pixel'),
        homography=None
    )
    
    print("Modules initialized.")
    
    # Video Setup
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error opening video file {VIDEO_PATH}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frames to process
    frames_to_process = min(MAX_FRAMES, total_frames) if MAX_FRAMES else total_frames
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    print(f"Processing {VIDEO_PATH}")
    print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}, Processing: {frames_to_process}")
    print(f"  Output: {OUTPUT_PATH}")
    
    # Main Processing Loop
    frame_buffer = FrameBuffer(capacity=30)
    frame_idx = 0
    
    prev_frame = None
    prev2_frame = None
    
    pbar = tqdm(total=frames_to_process)
    
    while cap.isOpened():
        # Stop if we've reached the frame limit
        if MAX_FRAMES and frame_idx >= MAX_FRAMES:
            break
        
        ret, frame_img = cap.read()
        if not ret:
            break
        
        # Calculate timestamp
        timestamp = frame_idx / fps
        current_frame_obj = Frame(index=frame_idx, timestamp=timestamp, image=frame_img)
        frame_buffer.push(current_frame_obj)
        
        # 1. Motion Detection
        mask = motion_detector.compute_foreground_mask(frame_img, prev_frame, prev2_frame)
        
        # Handle first few frames for temporal referencing
        if frame_idx < 2:
            prev2_frame = prev_frame
            prev_frame = frame_img.copy()
            frame_idx += 1
            pbar.update(1)
            out.write(frame_img)
            continue
    
        # 2. Segmentation
        regions = segment_foreground(mask, config)
        
        # 3. Tracking
        active_tracks = tracker.update(regions, frame_idx, timestamp)
        
        # 4. Speed & Capture
        for obj in active_tracks:
            if len(obj.trajectory) >= config['tracking']['min_trajectory_length']:
                speed = compute_speed_paper_method(obj, calibration, fps, config)
                obj.speed_m_s = speed
                 
                speed_kmph = speed * 3.6
                if speed_kmph > config['speed_limit_kmph'] and not obj.captured_img_ref:
                    # Capture violation
                    try:
                        json_path = capture_violation_paper_method(obj, list(frame_buffer.buffer), config)
                        obj.captured_img_ref = json_path
                        print(f"\n🚨 Violation captured: ID {obj.id}, {speed_kmph:.1f} km/h")
                    except Exception as e:
                        print(f"Error capturing violation: {e}")
    
        # Visualization
        vis_frame = frame_img.copy()
        
        for obj in active_tracks:
            x, y, w, h = obj.bbox
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            id_text = f"ID: {obj.id}"
            cv2.putText(vis_frame, id_text, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if obj.speed_m_s is not None:
                speed_kmph = obj.speed_m_s * 3.6
                speed_text = f"{speed_kmph:.1f} km/h"
                color = (0, 255, 0)
                if speed_kmph > config['speed_limit_kmph']:
                    color = (0, 0, 255)
                    cv2.putText(vis_frame, "VIOLATION", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
                cv2.putText(vis_frame, speed_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            # Draw trajectory
            if len(obj.trajectory) > 1:
                pts = [np.array(pt[1], dtype=np.int32) for pt in obj.trajectory]
                cv2.polylines(vis_frame, [np.array(pts)], False, (255, 0, 0), 1)
    
        out.write(vis_frame)
        
        prev2_frame = prev_frame
        prev_frame = frame_img.copy()
        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    out.release()
    pbar.close()
    
    print(f"\nProcessing complete!")
    print(f"  Frames processed: {frame_idx}")
    print(f"  Video saved to: {OUTPUT_PATH}")
    print(f"  Violations saved to: {config.get('violation_save_folder', './violations')}")


if __name__ == "__main__":
    main()
