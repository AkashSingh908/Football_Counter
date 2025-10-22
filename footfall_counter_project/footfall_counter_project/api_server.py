"""
Footfall Counter API Server
==========================

A Flask API server that accepts video uploads and returns footfall counts.
This provides a web interface for the footfall counter system.
"""

from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import os
import tempfile
import json
from footfall_counter import FootfallCounter
import cv2
import numpy as np
from datetime import datetime


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Footfall Counter API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .upload-area:hover { border-color: #999; }
        .result { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .error { background: #ffebee; color: #c62828; }
        .success { background: #e8f5e8; color: #2e7d32; }
        button { background: #2196f3; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #1976d2; }
        input[type="file"] { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Footfall Counter API</h1>
        <p>Upload a video file to count people entering and exiting through a specific area.</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <h3>Upload Video</h3>
                <input type="file" id="videoFile" name="video" accept="video/*" required>
                <br><br>
                <label for="confidence">Confidence Threshold:</label>
                <input type="range" id="confidence" name="confidence" min="0.1" max="0.9" step="0.1" value="0.5">
                <span id="confidenceValue">0.5</span>
                <br><br>
                <label for="roi_x1">ROI Line X1:</label>
                <input type="number" id="roi_x1" name="roi_x1" value="320">
                <label for="roi_y1">Y1:</label>
                <input type="number" id="roi_y1" name="roi_y1" value="0">
                <br>
                <label for="roi_x2">ROI Line X2:</label>
                <input type="number" id="roi_x2" name="roi_x2" value="320">
                <label for="roi_y2">Y2:</label>
                <input type="number" id="roi_y2" name="roi_y2" value="480">
                <br><br>
                <button type="submit">Process Video</button>
            </div>
        </form>
        
        <div id="result" style="display: none;"></div>
        
        <h3>API Endpoints</h3>
        <ul>
            <li><strong>POST /api/process</strong> - Process video and return counts</li>
            <li><strong>GET /api/health</strong> - Check API health</li>
        </ul>
        
        <h3>Example Usage</h3>
        <pre>
curl -X POST -F "video=@your_video.mp4" -F "confidence=0.5" \\
     -F "roi_x1=320" -F "roi_y1=0" -F "roi_x2=320" -F "roi_y2=480" \\
     http://localhost:5000/api/process
        </pre>
    </div>
    
    <script>
        document.getElementById('confidence').addEventListener('input', function(e) {
            document.getElementById('confidenceValue').textContent = e.target.value;
        });
        
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const resultDiv = document.getElementById('result');
            
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '<p>Processing video... This may take a few minutes.</p>';
            
            fetch('/api/process', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div class="result success">
                            <h3>Processing Complete!</h3>
                            <p><strong>Entries:</strong> ${data.entry_count}</p>
                            <p><strong>Exits:</strong> ${data.exit_count}</p>
                            <p><strong>Total:</strong> ${data.total_count}</p>
                            <p><strong>Processing Time:</strong> ${data.processing_time}s</p>
                            <p><strong>Frames Processed:</strong> ${data.frames_processed}</p>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <h3>Error</h3>
                            <p>${data.error}</p>
                        </div>
                    `;
                }
            })
            .catch(error => {
                resultDiv.innerHTML = `
                    <div class="result error">
                        <h3>Error</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            });
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/process', methods=['POST'])
def process_video():
    """Process uploaded video and return footfall counts."""
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No video file selected'})
        
        # Get parameters
        confidence = float(request.form.get('confidence', 0.5))
        roi_x1 = int(request.form.get('roi_x1', 320))
        roi_y1 = int(request.form.get('roi_y1', 0))
        roi_x2 = int(request.form.get('roi_x2', 320))
        roi_y2 = int(request.form.get('roi_y2', 480))
        
        # Save uploaded file temporarily
        filename = secure_filename(video_file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        video_file.save(temp_path)
        
        # Initialize footfall counter
        counter = FootfallCounter(confidence_threshold=confidence)
        counter.set_roi_line((roi_x1, roi_y1), (roi_x2, roi_y2))
        
        # Process video
        start_time = datetime.now()
        
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Could not open video file'})
        
        frames_processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            counter.process_frame(frame)
            frames_processed += 1
        
        cap.release()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Clean up temporary file
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        # Return results
        return jsonify({
            'success': True,
            'entry_count': counter.entry_count,
            'exit_count': counter.exit_count,
            'total_count': counter.entry_count + counter.exit_count,
            'processing_time': round(processing_time, 2),
            'frames_processed': frames_processed,
            'parameters': {
                'confidence': confidence,
                'roi_line': [(roi_x1, roi_y1), (roi_x2, roi_y2)]
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Processing error: {str(e)}'
        })


@app.route('/api/info')
def info():
    """Get API information."""
    return jsonify({
        'name': 'Footfall Counter API',
        'version': '1.0.0',
        'description': 'Computer vision-based footfall counting system',
        'endpoints': {
            'POST /api/process': 'Process video and return counts',
            'GET /api/health': 'Check API health',
            'GET /api/info': 'Get API information'
        },
        'supported_formats': ['mp4', 'avi', 'mov', 'mkv'],
        'max_file_size': '100MB'
    })


if __name__ == '__main__':
    print("Starting Footfall Counter API Server...")
    print("Web interface: http://localhost:5000")
    print("API endpoints: http://localhost:5000/api/")
    app.run(debug=True, host='0.0.0.0', port=5000)
