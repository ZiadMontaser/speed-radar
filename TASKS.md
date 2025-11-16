# Motion Detection & Speed Monitoring System - Task Distribution

## Member A — Robust Motion Detection & Masking

**Focus:** Build the hybrid motion detector (three-frame differencing + adaptive background subtraction + masked subtraction + shadow removal).

### Tasks

1. Implement three-frame differencing and the motion mask generator.
2. Implement adaptive background model `B_n(x,y)` and threshold matrix `T_n(x,y)` updates per the paper formulas (with tunable α).
3. Implement masked subtraction: combine motion mask + background subtraction.
4. Implement shadow detection using `0.23 < I/B < 0.95` and remove shadows.
5. Provide parameter config (α, initial T0, min/max thresholds).
6. Unit tests on synthetic frames (moving object + shadow + sudden lighting change).

### Deliverables

- **Function:** `compute_foreground_mask(frame, prev_frame, prev2_frame) -> mask`
- **Function:** `update_background_and_threshold(frame, mask) -> updates`
- Saved test cases and detection visualizations (masks overlay).

---

## Member B — Segmentation & Region Extraction

**Focus:** Convert foreground masks into reliable object regions (multi-iteration segmentation, bounding boxes) and provide cleaned blobs.

### Tasks

1. Implement multi-iteration horizontal + vertical scanning segmentation until stable.
2. Merge fragmented components and compute tight bounding boxes.
3. Filter small noise blobs (configurable min area).
4. Extract per-object pixel mask, bounding box, contour, centroid candidate(s).
5. Provide spatial heuristics to separate vehicles from pedestrians by shape/area (basic).
6. Unit tests to ensure connectedness and stable bounding boxes when objects split/merge.

### Deliverables

- **Function:** `segment_foreground(mask) -> List[Region {mask, bbox, area, centroid}]`
- Visual test outputs (bounding boxes on frames).

---

## Member C — Labeling, Tracking & Trajectory Management

**Focus:** Maintain persistent object IDs, centers, handle crossings/enter/leave cases and produce time-indexed trajectories.

### Tasks

1. Implement label assignment and history-based label correction (round-robin reuse as paper).
2. Track centroids across frames, maintain trajectory list for each object.
3. Handle special cases: object enters, leaves, cross-overs, leave+enter same frame.
4. Output entry frame `Fr0`, current frame index, and detect exit `FrN`.
5. Provide smoothing / Kalman filter optional for noisy centroids (config flag).
6. Produce per-object state machine (Active, Exited, Reused).

### Deliverables

- **Class:** `Tracker` with methods:
  - `update(regions, frame_index) -> List[TrackedObject]`
  - `get_active_objects()`
- **TrackedObject** contains: `id`, `bbox`, `centroid`, `trajectory[(frame,centroid)]`, `Fr0`, `FrN`, `captured_image_ref`
- Tests for crossing and reuse scenarios.

---

## Member D — Speed Calculation, Calibration & Capture

**Focus:** Distance→speed conversion, violation detection, capture image enhancement & metadata logging.

### Tasks

1. Provide calibration utilities: map pixel displacement → real distance (meters) for the monitored scene. Implement simple linear scaling using user-provided reference distances (or homography/ground-plane mapping if available).
2. Compute speed: `speed = distance / (N * frame_duration)`. Use tracked trajectory/centroid movement to compute d (either full scene width or centroid path length).
3. Decide `Fr0`/`FrN` logic and fallback if object missing frames (partial tracks).
4. When `speed > threshold`, capture object image (frame where centroid near center). Optionally combine neighboring frames for a deblur/enhance step (multi-frame average or simple super-res approach).
5. Save violation record: image, id, speed, timestamp, frames used, bbox.
6. Unit tests: known-dimension synthetic video where speed is set.

### Deliverables

- **Function:** `compute_speed(tracked_object, calibration_params, frame_rate) -> speed_mps`
- **Function:** `capture_violation(tracked_object, frame_buffer) -> (image_path, metadata)`
- **Calibration tool:** `estimate_scale_from_ground_points(points_pixels, points_meters) -> homography/scale`