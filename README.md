# Speed Radar Detection System - User Guide

## 🎯 Overview
This Speed Radar Detection System is a computer vision application that detects and tracks vehicles in video footage, calculates their speeds, and captures violations when vehicles exceed the configured speed limit.

## 📋 Prerequisites

### System Requirements
- **Python**: Version 3.11 or higher
- **Operating System**: Windows, Linux, or macOS
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 500MB free space for dependencies and output files

### Required Python Libraries

Install all dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### Core Libraries:
- **OpenCV** (`opencv-python`): Computer vision and video processing
- **NumPy**: Numerical computations
- **Streamlit**: Web-based user interface
- **PyYAML**: Configuration file handling
- **Pandas**: Data manipulation and CSV generation
- **Plotly**: Interactive charts and visualizations
- **tqdm**: Progress bars

## 🚀 Getting Started

### 1. Installation

```bash
# Clone or download the project
cd speed-radar

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.yaml` in the project root to customize settings:

```yaml
# Key Configuration Parameters

# Frame rate of your video (fps)
frame_rate: 25.0

# Speed limit threshold (km/h)
speed_limit_kmph: 60

# Motion detection sensitivity
alpha_background: 0.01      # Background learning rate (0.0-1.0)
motion_threshold: 30        # Pixel difference threshold

# Tracking parameters
tracking:
  max_distance_threshold: 100
  min_trajectory_length: 5

# Calibration (adjust for your camera setup)
scale_m_per_pixel: 0.05     # Meters per pixel conversion
```

### 3. Running the Application

#### Option A: Web Interface (Recommended)

```bash
# From the project root directory
streamlit run .\src\app.py
```

This will:
1. Start a local web server
2. Open your default browser automatically
3. Display the Speed Detection interface at `http://localhost:8501`

#### Option B: Command Line Processing

```bash
# Navigate to the src directory
cd src

# Run the main script
python main.py
```

## 📖 How to Use

### Web Interface Workflow

#### Step 1: Upload Video
1. Click **"Browse files"** or drag-and-drop your video file
2. Supported formats: MP4, AVI, MOV
3. Wait for the video to upload

#### Step 2: Configure Settings (Optional)
Adjust the sidebar parameters:
- **Max Frames**: Limit processing to first N frames (0 = process all)
- **Alpha Background**: Background adaptation speed
- **Motion Threshold**: Sensitivity for motion detection

#### Step 3: Start Detection
1. Click the **"🚀 Start Detection"** button
2. Watch real-time processing progress
3. View the current frame being processed
4. Monitor the progress bar

#### Step 4: Review Results
After processing completes, you'll see:

**Violation Report:**
- Total violations detected
- Number of objects tracked
- Average speed of violators
- Current speed limit

**Violations Table:**
- Detailed list of all violations
- Vehicle ID, speed, timestamp, frames

**Violation Images:**
- Visual snapshots of each violation
- Vehicle crop with speed annotation

**Statistics & Charts:**
- Speed distribution histogram
- Processing summary

#### Step 5: Export Results

Three export options available:

1. **⬇️ Download Video**
   - Processed video with bounding boxes
   - Speed annotations on each vehicle
   - Trajectory visualization

2. **📦 Download Violations (ZIP)**
   - All violation images (original, enhanced, context)
   - JSON metadata files
   - Complete violation records

3. **📊 Download Report (CSV)**
   - Spreadsheet-compatible format
   - All violation details
   - Easy to analyze in Excel/Sheets

## 🎛️ Configuration Guide

### Camera Calibration

For accurate speed measurement, calibrate your camera:

1. **Measure Real-World Distance**
   - Place markers at known distance (e.g., 10 meters)
   - Measure the pixel distance in a reference frame

2. **Calculate Scale**
   ```
   scale_m_per_pixel = real_distance_meters / pixel_distance
   ```

3. **Update config.yaml**
   ```yaml
   scale_m_per_pixel: 0.05  # Replace with your calculated value
   ```

### Advanced Settings

#### Motion Detection
- `alpha_background`: How fast the system adapts to scene changes
  - Lower (0.001): Stable, but slow adaptation
  - Higher (0.1): Fast adaptation, may miss stationary vehicles

- `motion_threshold`: Sensitivity to movement
  - Lower (10-20): More sensitive, may detect noise
  - Higher (40-50): Less sensitive, may miss slow vehicles

#### Segmentation
```yaml
segmentation:
  min_area: 600              # Minimum blob size (pixels)
  min_run_width: 3          # Gap detection parameter
  max_iterations: 3         # Splitting iterations
  padding: 2                # Bounding box padding
```

#### Tracking
```yaml
tracking:
  max_distance_threshold: 100    # Maximum movement between frames
  min_trajectory_length: 5       # Minimum points for valid track
  trajectory_smoothing: 0.3      # Smoothing factor (0-1)
```

#### Violation Capture
```yaml
violation_save_folder: ./violations    # Output directory
capture_padding: 70                    # Crop padding (pixels)
capture_frame_window: 3                # Frames to average
save_full_context: false               # Save full frame
```

## 📁 Project Structure

```
speed-radar/
├── config.yaml              # Main configuration file
├── requirements.txt         # Python dependencies
├── README.md               # API documentation
├── USER_GUIDE.md           # This file
├── src/
│   ├── app.py              # Streamlit web interface
│   ├── main.py             # CLI interface
│   ├── motion_detection/   # Motion detection module
│   ├── segmentation/       # Foreground segmentation
│   ├── tracking/           # Object tracking
│   ├── speed_capture/      # Speed calculation & capture
│   └── common/             # Shared data structures
├── violations/             # Output folder (auto-created)
└── datasets/              # Sample videos (optional)
```

## 🐛 Troubleshooting

### Issue: "config.yaml not found"
**Solution:** Ensure you're running the app from the project root directory:
```bash
# Correct
streamlit run .\src\app.py

# From root, not from src/
```

### Issue: Slow Processing
**Solutions:**
- Reduce video resolution before processing
- Limit frames with "Max Frames" setting
- Adjust `alpha_background` to higher value (faster)
- Use a video with lower frame rate

### Issue: No Violations Detected
**Check:**
1. Speed limit is set correctly in `config.yaml`
2. Camera calibration (`scale_m_per_pixel`) is accurate
3. Vehicles are actually speeding
4. `min_trajectory_length` isn't too high

### Issue: Too Many False Detections
**Solutions:**
- Increase `min_area` in segmentation settings
- Increase `motion_threshold`
- Adjust `max_distance_threshold` in tracking
- Ensure proper lighting in video

### Issue: Inaccurate Speeds
**Solutions:**
1. **Recalibrate camera** - most common issue
2. Verify `frame_rate` matches video
3. Ensure camera is perpendicular to traffic flow
4. Check for perspective distortion

## 💡 Tips for Best Results

1. **Camera Placement**
   - Mount perpendicular to road
   - Avoid steep angles
   - Minimize perspective distortion
   - Ensure good lighting conditions

2. **Video Quality**
   - Use high-resolution video (720p or higher)
   - Maintain consistent frame rate
   - Avoid motion blur
   - Ensure clear vehicle visibility

3. **Calibration**
   - Use road markings for reference
   - Measure multiple distances
   - Verify with known speeds
   - Recalibrate for different locations

4. **Processing**
   - Test with short clips first
   - Adjust parameters incrementally
   - Monitor real-time display
   - Review violation images for accuracy

## 📊 Output Files

### Violations Folder Structure
```
violations/
├── violation_<id>_<timestamp>.jpg           # Main image
├── violation_<id>_<timestamp>_enhanced.jpg  # Enhanced image
├── violation_<id>_<timestamp>_context.jpg   # Full frame (if enabled)
└── violation_<id>_<timestamp>.json          # Metadata
```

### JSON Metadata Format
```json
{
  "id": 1,
  "speed_kmph": 75.5,
  "speed_m_s": 20.97,
  "timestamp": 10.5,
  "capture_frame_index": 262,
  "Fr0": 250,
  "FrN": 275,
  "frames_tracked": 25,
  "trajectory_length": 25,
  "bbox_in_frame": [450, 200, 120, 80],
  "centroid": [510.5, 240.2],
  "image_path": "violations/violation_1_1234567890.jpg",
  "method": "paper_based_v3"
}
```

## 🔧 Advanced Usage

### Running Tests

```bash
# Run individual module tests
python src/motion_detection/motion_detection_test.py
python src/segmentation/segmentation_test.py
python src/tracking/tracking_test.py
python src/speed_capture/tracking_speed_test.py
```

### Viewing Saved Violations

```bash
# Use the violation viewer
python src/view_violations.py
```

This will display all saved violations with their metadata and images.

## 📞 Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review `README.md` for API documentation
3. Verify all dependencies are installed
4. Check console output for error messages
5. Ensure config.yaml has valid values

## 📝 License & Credits

This project implements a speed detection system based on computer vision techniques for vehicle monitoring and traffic enforcement applications.

---

**Version:** 2.0  
**Last Updated:** December 2025
