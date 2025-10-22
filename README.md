# Footfall Counter using Computer Vision

A computer vision-based system that counts the number of people entering and exiting through a specific area using YOLO detection and custom tracking algorithms.

## Overview

This project implements a footfall counter that:
- Detects people in video streams using YOLO (You Only Look Once) object detection
- Tracks individuals across frames using centroid-based tracking
- Defines a virtual line or region of interest (ROI) for counting
- Counts entries and exits by detecting when people cross the ROI line
- Provides real-time visualization with bounding boxes, trajectories, and counts

## Features

- **Person Detection**: Uses YOLOv8 for accurate person detection
- **Multi-Object Tracking**: Custom centroid-based tracking algorithm
- **ROI Line Detection**: Configurable region of interest for counting
- **Entry/Exit Counting**: Accurate counting based on trajectory analysis
- **Real-time Visualization**: Live display with bounding boxes and trajectories
- **Video Processing**: Process pre-recorded videos or live streams
- **Configurable Parameters**: Adjustable confidence thresholds and tracking parameters

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)

### Setup

1. Clone or download this repository:
```bash
git clone <repository-url>
cd footfall_counter_project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. The YOLO model will be automatically downloaded on first run.

## Usage

### Basic Usage

```bash
python footfall_counter.py --input path/to/your/video.mp4
```

### Advanced Usage

```bash
python footfall_counter.py \
    --input path/to/your/video.mp4 \
    --output output_video.mp4 \
    --model yolov8n.pt \
    --confidence 0.5 \
    --roi 320 0 320 480
```

### Parameters

- `--input` or `-i`: Path to input video file (required)
- `--output` or `-o`: Path to save output video (optional)
- `--model` or `-m`: YOLO model path (default: yolov8n.pt)
- `--confidence` or `-c`: Confidence threshold for detection (default: 0.5)
- `--roi`: ROI line coordinates as x1 y1 x2 y2 (optional, uses center line if not specified)

### Example Commands

1. **Process a video with default settings:**
```bash
python footfall_counter.py --input sample_video.mp4
```

2. **Process with custom ROI line:**
```bash
python footfall_counter.py --input sample_video.mp4 --roi 200 0 200 600
```

3. **Save output video:**
```bash
python footfall_counter.py --input sample_video.mp4 --output result.mp4
```

4. **Use different YOLO model:**
```bash
python footfall_counter.py --input sample_video.mp4 --model yolov8s.pt
```

## How It Works

### 1. Person Detection
- Uses YOLOv8 to detect people in each frame
- Filters detections based on confidence threshold
- Extracts bounding boxes and centroids

### 2. Object Tracking
- Implements centroid-based tracking algorithm
- Associates detections across frames using distance metrics
- Maintains trajectories for each tracked person
- Handles object appearance and disappearance

### 3. ROI Line Crossing Detection
- Defines a virtual line for counting
- Analyzes trajectories to detect line crossings
- Determines entry/exit direction based on movement
- Prevents double-counting of the same person

### 4. Counting Logic
- Tracks which objects have already crossed the line
- Increments entry/exit counters appropriately
- Provides real-time count display

## Technical Details

### Detection Model
- **YOLOv8**: State-of-the-art object detection model
- **Classes**: Focuses on person detection (class 0 in COCO dataset)
- **Confidence**: Configurable threshold for detection quality

### Tracking Algorithm
- **Centroid Tracking**: Simple but effective tracking method
- **Distance-based Association**: Uses Euclidean distance for object matching
- **Trajectory Storage**: Maintains movement history for each object
- **Disappearance Handling**: Manages objects that leave the frame

### ROI Crossing Detection
- **Line Equation**: Uses mathematical line representation
- **Side Detection**: Determines which side of the line each point is on
- **Direction Analysis**: Infers movement direction from trajectory
- **Crossing Validation**: Ensures valid line crossings

## Performance Considerations

### Optimization Tips
1. **GPU Usage**: Ensure CUDA is available for faster inference
2. **Model Selection**: Use smaller models (yolov8n) for faster processing
3. **Frame Skipping**: Can be modified to process every nth frame for speed
4. **ROI Positioning**: Place ROI line strategically to avoid false positives

### System Requirements
- **Minimum**: CPU-only processing (slower)
- **Recommended**: GPU with CUDA support
- **Memory**: 4GB+ RAM for video processing
- **Storage**: Space for model weights (~50MB for YOLOv8n)

## Troubleshooting

### Common Issues

1. **Model Download Issues**:
   - Ensure internet connection for first-time model download
   - Check firewall settings if download fails

2. **CUDA Issues**:
   - Install CUDA toolkit if using GPU
   - Fallback to CPU processing if GPU unavailable

3. **Video Format Issues**:
   - Ensure video codec is supported by OpenCV
   - Try converting video to MP4 format

4. **Performance Issues**:
   - Reduce confidence threshold for more detections
   - Use smaller YOLO model (yolov8n instead of yolov8x)
   - Process lower resolution videos

### Error Messages

- **"Could not open video file"**: Check file path and format
- **"CUDA out of memory"**: Reduce batch size or use CPU
- **"No module named 'ultralytics'"**: Install requirements.txt

## Example Output

The system provides:
- Real-time count display showing entries, exits, and total
- Visual bounding boxes around detected people
- Trajectory lines showing movement paths
- ROI line visualization
- Progress updates during processing

## Future Enhancements

Potential improvements:
- **DeepSORT Integration**: More robust tracking algorithm
- **Heatmap Visualization**: Show popular movement areas
- **API Development**: REST API for video processing
- **Real-time Streaming**: Support for RTSP/WebRTC streams
- **Multi-ROI Support**: Multiple counting lines
- **Analytics Dashboard**: Historical data visualization

## Dependencies

- `opencv-python`: Computer vision operations
- `ultralytics`: YOLO model implementation
- `torch`: Deep learning framework
- `numpy`: Numerical computations
- `scipy`: Scientific computing utilities
- `matplotlib`: Visualization (optional)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or support, please open an issue in the repository.
