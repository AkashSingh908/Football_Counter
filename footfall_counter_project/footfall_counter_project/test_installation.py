#!/usr/bin/env python3
"""
Simple test script to verify footfall counter installation
"""

def test_imports():
    """Test if all required modules can be imported."""
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics imported successfully")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False
    
    try:
        from scipy.spatial import distance
        print("✓ SciPy imported successfully")
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        return False
    
    return True


def test_footfall_counter():
    """Test if FootfallCounter can be initialized."""
    try:
        from footfall_counter import FootfallCounter
        counter = FootfallCounter()
        print("✓ FootfallCounter initialized successfully")
        return True
    except Exception as e:
        print(f"✗ FootfallCounter initialization failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Footfall Counter Installation")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Installation test failed - missing dependencies")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Test FootfallCounter
    if not test_footfall_counter():
        print("\n❌ FootfallCounter test failed")
        return False
    
    print("\n✅ All tests passed! Installation is working correctly.")
    print("\nYou can now run:")
    print("  python3 demo.py --create-test")
    print("  python3 webcam_demo.py")
    print("  python3 footfall_counter.py --input your_video.mp4")
    
    return True


if __name__ == "__main__":
    main()

