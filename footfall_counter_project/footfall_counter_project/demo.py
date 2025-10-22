"""
Demo script for Footfall Counter
===============================

This script demonstrates the footfall counter with a simple example.
It can generate a test video or process an existing one.
"""

import cv2
import numpy as np
import os
import argparse
from footfall_counter import FootfallCounter


def create_test_video(output_path: str, duration: int = 10, fps: int = 30):
    """
    Create a simple test video with moving rectangles to simulate people.
    
    Args:
        output_path: Path to save the test video
        duration: Duration in seconds
        fps: Frames per second
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    print(f"Creating test video: {output_path}")
    print(f"Duration: {duration}s, FPS: {fps}, Total frames: {total_frames}")
    
    for frame_num in range(total_frames):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some background elements
        cv2.rectangle(frame, (0, 0), (width, height), (50, 50, 50), -1)
        
        # Add a "doorway" in the center
        cv2.rectangle(frame, (width//2 - 20, 0), (width//2 + 20, height), (100, 100, 100), -1)
        
        # Simulate people moving across the screen
        # Person 1: Moving from left to right
        if frame_num < total_frames * 0.3:
            x = int((frame_num / (total_frames * 0.3)) * (width - 100))
            y = height // 2
            cv2.rectangle(frame, (x, y-20), (x+40, y+20), (0, 255, 0), -1)
            cv2.putText(frame, "Person 1", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Person 2: Moving from right to left
        if frame_num > total_frames * 0.2 and frame_num < total_frames * 0.6:
            x = int(width - ((frame_num - total_frames * 0.2) / (total_frames * 0.4)) * (width - 100))
            y = height // 2 + 50
            cv2.rectangle(frame, (x, y-20), (x+40, y+20), (255, 0, 0), -1)
            cv2.putText(frame, "Person 2", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Person 3: Moving from left to right (later)
        if frame_num > total_frames * 0.5 and frame_num < total_frames * 0.8:
            x = int(((frame_num - total_frames * 0.5) / (total_frames * 0.3)) * (width - 100))
            y = height // 2 - 50
            cv2.rectangle(frame, (x, y-20), (x+40, y+20), (0, 0, 255), -1)
            cv2.putText(frame, "Person 3", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")


def run_demo(input_video: str = None, create_test: bool = False):
    """
    Run the footfall counter demo.
    
    Args:
        input_video: Path to input video (optional)
        create_test: Whether to create a test video
    """
    if create_test:
        test_video_path = "test_video.mp4"
        if not os.path.exists(test_video_path):
            create_test_video(test_video_path)
        input_video = test_video_path
    
    if not input_video or not os.path.exists(input_video):
        print("No input video provided or file doesn't exist.")
        print("Use --create-test to generate a test video or provide --input path/to/video.mp4")
        return
    
    print("=" * 50)
    print("FOOTFALL COUNTER DEMO")
    print("=" * 50)
    
    # Initialize the footfall counter
    print("Initializing footfall counter...")
    counter = FootfallCounter(confidence_threshold=0.3)  # Lower threshold for demo
    
    # Set ROI line (vertical line in the center)
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    roi_start = (width // 2, 0)
    roi_end = (width // 2, height)
    counter.set_roi_line(roi_start, roi_end)
    
    print(f"ROI line set from {roi_start} to {roi_end}")
    print(f"Processing video: {input_video}")
    print("Press 'q' to quit during processing")
    print("-" * 50)
    
    # Process the video
    try:
        counter.process_video(input_video, "demo_output.mp4")
        print("-" * 50)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print(f"Final Results:")
        print(f"  Entries: {counter.entry_count}")
        print(f"  Exits: {counter.exit_count}")
        print(f"  Total: {counter.entry_count + counter.exit_count}")
        print(f"Output video saved as: demo_output.mp4")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Make sure you have all required dependencies installed:")
        print("pip install -r requirements.txt")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Footfall Counter Demo')
    parser.add_argument('--input', '-i', help='Input video path')
    parser.add_argument('--create-test', action='store_true', help='Create a test video')
    
    args = parser.parse_args()
    
    run_demo(input_video=args.input, create_test=args.create_test)


if __name__ == "__main__":
    main()

