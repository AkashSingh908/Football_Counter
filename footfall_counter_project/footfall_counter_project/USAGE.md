# Footfall Counter Usage Guide

## Quick Start

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Test installation
python3 test_installation.py
```

### 2. Basic Usage

#### Process a video file:
```bash
python3 footfall_counter.py --input your_video.mp4
```

#### Create and test with sample video:
```bash
python3 demo.py --create-test
```

#### Real-time webcam demo:
```bash
python3 webcam_demo.py
```

#### Start API server:
```bash
python3 api_server.py
# Then visit http://localhost:5000
```

## Command Line Options

### Main Script (`footfall_counter.py`)
```bash
python3 footfall_counter.py [OPTIONS]

Options:
  -i, --input PATH        Input video file (required)
  -o, --output PATH       Output video file (optional)
  -m, --model PATH        YOLO model path (default: yolov8n.pt)
  -c, --confidence FLOAT  Confidence threshold (default: 0.5)
  --roi X1 Y1 X2 Y2       ROI line coordinates (optional)
```

### Examples

1. **Basic video processing:**
```bash
python3 footfall_counter.py --input sample.mp4
```

2. **Save output video:**
```bash
python3 footfall_counter.py --input sample.mp4 --output result.mp4
```

3. **Custom ROI line:**
```bash
python3 footfall_counter.py --input sample.mp4 --roi 200 0 200 600
```

4. **Adjust confidence threshold:**
```bash
python3 footfall_counter.py --input sample.mp4 --confidence 0.3
```

## API Usage

### Start the server:
```bash
python3 api_server.py
```

### Web Interface:
- Open http://localhost:5000 in your browser
- Upload a video file
- Configure parameters
- Click "Process Video"

### API Endpoints:

#### Process Video:
```bash
curl -X POST -F "video=@your_video.mp4" \
     -F "confidence=0.5" \
     -F "roi_x1=320" -F "roi_y1=0" \
     -F "roi_x2=320" -F "roi_y2=480" \
     http://localhost:5000/api/process
```

#### Health Check:
```bash
curl http://localhost:5000/api/health
```

## Configuration

### ROI Line Setup
The Region of Interest (ROI) line determines where people are counted:
- **Vertical line**: Good for doorways, entrances
- **Horizontal line**: Good for corridors, walkways
- **Custom line**: Any angle for specific scenarios

### Confidence Threshold
- **0.1-0.3**: More detections, may include false positives
- **0.5**: Balanced (default)
- **0.7-0.9**: Fewer detections, higher accuracy

### Model Selection
- **yolov8n.pt**: Fastest, least accurate
- **yolov8s.pt**: Balanced speed/accuracy
- **yolov8m.pt**: Slower, more accurate
- **yolov8l.pt**: Slowest, most accurate

## Troubleshooting

### Common Issues:

1. **"No module named 'ultralytics'"**
   ```bash
   pip install ultralytics
   ```

2. **"Could not open video file"**
   - Check file path and format
   - Try converting to MP4

3. **Poor detection accuracy**
   - Lower confidence threshold
   - Use larger YOLO model
   - Ensure good lighting in video

4. **Slow processing**
   - Use smaller YOLO model
   - Reduce video resolution
   - Use GPU if available

### Performance Tips:

1. **For real-time processing:**
   - Use webcam_demo.py
   - Lower confidence threshold
   - Use yolov8n.pt model

2. **For accuracy:**
   - Use higher confidence threshold
   - Use larger YOLO model
   - Ensure good video quality

3. **For batch processing:**
   - Use footfall_counter.py with output video
   - Process multiple videos in sequence

## Output Interpretation

### Console Output:
```
Processing video: sample.mp4
Video properties: 640x480 @ 30 FPS
ROI line: (320, 0) to (320, 480)
Processed 900 frames. Current counts - Entries: 3, Exits: 2
Processing complete!
Total frames processed: 900
Final counts - Entries: 3, Exits: 2
Total footfall: 5
```

### API Response:
```json
{
  "success": true,
  "entry_count": 3,
  "exit_count": 2,
  "total_count": 5,
  "processing_time": 12.34,
  "frames_processed": 900
}
```

## Advanced Usage

### Custom ROI Positioning:
- Use video analysis tools to determine optimal line position
- Consider traffic patterns and camera angle
- Test with different line positions for best accuracy

### Batch Processing:
```bash
# Process multiple videos
for video in *.mp4; do
    python3 footfall_counter.py --input "$video" --output "result_$video"
done
```

### Integration with Other Systems:
- Use API endpoints for web applications
- Process video streams in real-time
- Integrate with monitoring systems

## Support

For issues and questions:
1. Check this usage guide
2. Review README.md for detailed information
3. Test with sample video first
4. Check system requirements and dependencies

