"""
Setup script for Footfall Counter
================================

This script helps set up the footfall counter environment and download required models.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False


def download_yolo_model():
    """Download YOLO model if not present."""
    model_path = "yolov8n.pt"
    if os.path.exists(model_path):
        print(f"✓ YOLO model already exists: {model_path}")
        return True
    
    print("Downloading YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)  # This will download the model
        print(f"✓ YOLO model downloaded: {model_path}")
        return True
    except Exception as e:
        print(f"Error downloading YOLO model: {e}")
        return False


def test_installation():
    """Test if the installation works."""
    print("Testing installation...")
    try:
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        from footfall_counter import FootfallCounter
        
        print("✓ All imports successful")
        
        # Test basic functionality
        counter = FootfallCounter()
        print("✓ FootfallCounter initialized successfully")
        
        return True
    except Exception as e:
        print(f"Error testing installation: {e}")
        return False


def create_sample_video():
    """Create a sample video for testing."""
    print("Creating sample video...")
    try:
        from demo import create_test_video
        create_test_video("sample_video.mp4", duration=5)
        print("✓ Sample video created: sample_video.mp4")
        return True
    except Exception as e:
        print(f"Error creating sample video: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("FOOTFALL COUNTER SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("Setup failed: Could not install requirements")
        sys.exit(1)
    
    # Download YOLO model
    if not download_yolo_model():
        print("Setup failed: Could not download YOLO model")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("Setup failed: Installation test failed")
        sys.exit(1)
    
    # Create sample video
    create_sample_video()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the demo: python demo.py --create-test")
    print("2. Try webcam demo: python webcam_demo.py")
    print("3. Process your own video: python footfall_counter.py --input your_video.mp4")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()

