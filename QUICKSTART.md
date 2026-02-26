# Speed Radar System - Quick Start Guide

## Installation & Setup (3 Steps)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Application
```bash
streamlit run .\src\app.py
```

### 3️⃣ Use the Web Interface
1. Open browser at `http://localhost:8501`
2. Upload a video file (MP4, AVI, MOV)
3. Click "🚀 Start Detection"
4. Download results (Video, ZIP, or CSV)

---

## Required Libraries

All libraries are listed in `requirements.txt`. Key dependencies:
- **streamlit**: Web interface
- **opencv-python**: Video processing
- **numpy**: Numerical operations
- **pandas**: Data handling
- **plotly**: Charts
- **pyyaml**: Configuration
- **tqdm**: Progress bars

---

## Configuration

Edit `config.yaml` for your needs:

```yaml
# Most important settings:
speed_limit_kmph: 60          # Speed threshold
frame_rate: 25.0              # Video frame rate
scale_m_per_pixel: 0.05       # Camera calibration
```

**Camera Calibration:**
- Measure a known distance in your video (e.g., 10 meters)
- Count pixels for that distance
- Calculate: `scale_m_per_pixel = real_meters / pixels`

---

## Troubleshooting

**"Config not found"**: Run from project root, not from src/  
**Slow processing**: Reduce video resolution or use "Max Frames" limit  
**No violations**: Check speed limit and camera calibration  
**Wrong speeds**: Recalibrate `scale_m_per_pixel`

---

## Output

Violations are saved to `./violations/` folder:
- `violation_*.jpg`: Vehicle images
- `violation_*.json`: Metadata (speed, time, location)
- Processed video with annotations

---

## Full Documentation

See **[USER_GUIDE.md](USER_GUIDE.md)** for complete documentation including:
- Detailed configuration guide
- Advanced settings
- Troubleshooting
- Tips for best results
- Output file formats

---

**Need Help?** Check the USER_GUIDE.md for detailed instructions!
