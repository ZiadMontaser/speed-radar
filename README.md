# SDCS Backend API Specification

This document defines the data structures, function signatures, configuration, and example JSON for the image-processing backend pipeline for the Speed Detection Camera System (SDCS). All four backend modules must implement and use these interfaces to ensure smooth integration.

---

## Table of Contents
1. Overview
2. Data Classes (Python `dataclasses` style)
3. Core Functions / Module APIs
4. Config (`config.yaml`) keys
5. Frame Buffer API
6. Violation Record JSON
7. Error Handling & Return Conventions
8. Unit Test Contracts
9. Example Pipeline Flow

---

## 1. Overview

Pipeline stages (order):
1. **Motion Detection** (Member A) -> produces foreground mask
2. **Segmentation** (Member B) -> produces Regions from mask
3. **Tracking** (Member C) -> maintains `TrackedObject`s
4. **Speed & Capture** (Member D) -> computes speed, records violations

All image arrays are `numpy.ndarray` in BGR or grayscale as documented per function. Timestamps are seconds (float). Frame indices are integers starting from 0.

---

## 2. Data Classes

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

# Basic frame container
@dataclass
class Frame:
    index: int
    timestamp: float      # seconds
    image: np.ndarray     # HxWx3 BGR (uint8)

# Region (result of segmentation)
@dataclass
class Region:
    mask: np.ndarray             # HxW boolean or uint8 mask
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: int
    centroid: Tuple[float, float]
    contour: Optional[List[Tuple[int,int]]] = None

# Tracked object
@dataclass
class TrackedObject:
    id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    trajectory: List[Tuple[int, Tuple[float,float]]] = field(default_factory=list)
    Fr0: Optional[int] = None
    FrN: Optional[int] = None
    captured_img_ref: Optional[str] = None
    speed_m_s: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

# Calibration parameters
@dataclass
class Calibration:
    # Either `scale_m_per_pixel` (float) OR homography matrix
    scale_m_per_pixel: Optional[float] = None
    homography: Optional[np.ndarray] = None  # 3x3
    reference_points: Optional[List[Tuple[Tuple[float,float], Tuple[float,float]]]] = None

# API result wrapper
@dataclass
class APIResult:
    success: bool
    data: Optional[object] = None
    error: Optional[str] = None
```

---

## 3. Core Functions / Module APIs

### Member A — Motion Detection

```python
# Input: current frame, prev frame, prev2 frame
def compute_foreground_mask(frame: Frame, prev_frame: Frame, prev2_frame: Frame, config: dict) -> np.ndarray:
    """Returns binary mask HxW (uint8 0/255 or bool)."""

# Update background and threshold
def update_background_and_threshold(frame: Frame, mask: np.ndarray, background: np.ndarray, threshold: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (new_background, new_threshold)."""
```

- Behavior: must support adaptive `alpha` parameter, masked subtraction, shadow removal using ratio check `0.23 < I/B < 0.95`.
- Outputs must be reproducible given same random seed and config.


### Member B — Segmentation

```python
def segment_foreground(mask: np.ndarray, config: dict) -> List[Region]:
    """Multi-iteration horizontal+vertical segmentation. Returns list of Region instances."""
```

- Must perform iterative scans until segmentation stable.
- Must filter small blobs using `min_area` from config.


### Member C — Tracking

```python
class Tracker:
    def __init__(self, config: dict):
        ...

    def update(self, regions: List[Region], frame_index: int, timestamp: float) -> List[TrackedObject]:
        """Update tracker and return the list of active TrackedObject objects after this frame."""

    def get_active(self) -> List[TrackedObject]:
        ...

    def get_all(self) -> List[TrackedObject]:
        ...
```

- Tracker must:
  - assign stable IDs
  - set `Fr0` when object enters
  - set `FrN` when object leaves (and mark as inactive)
  - support round-robin ID reuse once object marked exited
  - optional smoothing (Kalman) if `config['use_kalman'] == True`


### Member D — Speed Calculation & Capture

```python
def compute_speed(tracked_obj: TrackedObject, calibration: Calibration, frame_rate: float) -> float:
    """Return speed in meters/second. Raises ValueError for insufficient data."""

# Capture violation (frame_buffer is a small ring buffer of Frame objects)
def capture_violation(tracked_obj: TrackedObject, frame_buffer: List[Frame], config: dict) -> Tuple[str, dict]:
    """Saves image file, returns (filepath, metadata_dict)."""
```

- `compute_speed` should support two modes:
  - scale-based: uses `scale_m_per_pixel` and centroid displacement over frames
  - homography-ground-plane: maps pixel coords to ground plane, then measures distance

- `capture_violation` must pick the frame where centroid nearest scene center (or use `tracked_obj.Fr0+N/2`) and optionally combine neighbor frames for enhancement.

---

## 4. Config (`config.yaml`) keys (recommended)

```yaml
frame_rate: 25.0
alpha_background: 0.1
alpha_threshold: 0.05
initial_threshold: 15
min_area: 200
use_kalman: false
scale_m_per_pixel: 0.02   # optional
homography_enabled: false
speed_limit_kmph: 60
violation_save_folder: ./violations
max_track_age: 10  # frames to keep 'lost' tracks before marking exit
shadow_ratio_min: 0.23
shadow_ratio_max: 0.95
```

---

## 5. Frame Buffer API

Provide a simple ring buffer module `framebuf.py`:

```python
class FrameBuffer:
    def __init__(self, capacity:int=16):
        ...
    def push(self, frame: Frame) -> None:
        ...
    def get(self, index:int) -> Frame:  # absolute frame index
        ...
    def last_n(self, n:int) -> List[Frame]:
        ...
```

`capture_violation` should accept `frame_buffer.last_n(k)` to access neighbors.

---

## 6. Violation Record JSON

Example saved JSON for each violation (file `violation_<id>_<time>.json`):

```json
{
  "id": 3,
  "speed_kmph": 82.5,
  "speed_m_s": 22.92,
  "capture_frame": 153,
  "capture_time": 12.12,
  "bbox": [320, 200, 120, 60],
  "image_path": "./violations/violation_3_153.png",
  "trajectory": [[140, [300,210]], [141, [305,212]], ...],
  "calibration": {
    "method": "scale",
    "scale_m_per_pixel": 0.02
  }
}
```

---

## 7. Error Handling & Return Conventions

- Functions return either expected object or raise `ValueError`/`RuntimeError` with descriptive message.
- Where helpful, wrap outputs in `APIResult(success=True/False, data=..., error=...)` for cross-language boundaries.

---

## 8. Unit Test Contracts

Each module must include automated unit tests using the shared `synthetic_test_videos/` dataset.

Minimum tests per module:
- Motion detection: moving rectangle, moving rectangle + shadow, sudden lighting change.
- Segmentation: fragmented object -> one region, two close objects -> distinct regions.
- Tracker: crossing two objects -> preserved IDs; enter/leave -> Fr0/FrN set.
- Speed: known-speed synthetic video -> measured speed ±5%.

Tests should be runnable with `pytest` and include small visualization outputs saved to `tests/output/` for manual inspection.

---

## 9. Example Pipeline Flow (pseudo)

```python
# inside main loop
frame = frame_provider.next()
mask = compute_foreground_mask(frame, prev, prev2, config)
regions = segment_foreground(mask, config)
tracked = tracker.update(regions, frame.index, frame.timestamp)
for obj in tracked:
    if obj.FrN is not None and obj.speed_m_s is None:
        obj.speed_m_s = compute_speed(obj, calibration, config['frame_rate'])
        if obj.speed_m_s * 3.6 > config['speed_limit_kmph']:
            capture_violation(obj, frame_buffer.latest(), config)
```

---

### Notes & Conventions
- Use `numpy`, `opencv-python` (cv2) for image operations.
- Keep functions pure where possible (stateless) and put state in `Tracker`, `FrameBuffer`, and background models.
- Document assumptions and units clearly (pixels vs meters, frames vs seconds).

---

End of API spec.

