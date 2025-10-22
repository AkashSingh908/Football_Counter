"""
Real-time Footfall Counter using Webcam
=======================================

This script demonstrates real-time footfall counting using a webcam.
Perfect for testing the system with live video feed.
"""

import cv2
import numpy as np
from footfall_counter import FootfallCounter
import argparse


def webcam_demo():
    """Run footfall counter with webcam input."""
    print("=" * 50)
    print("REAL-TIME FOOTFALL COUNTER DEMO")
    print("=" * 50)
    print("Instructions:")
    print("1. Position yourself in front of the camera")
    print("2. Move across the green ROI line to test counting")
    print("3. Press 'q' to quit")
    print("4. Press 'r' to reset counters")
    print("5. Press 's' to set new ROI line (click two points)")
    print("-" * 50)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam resolution: {width}x{height}")
    
    # Initialize footfall counter
    counter = FootfallCounter(confidence_threshold=0.5)
    
    # Set default ROI line (vertical center)
    roi_start = (width // 2, 0)
    roi_end = (width // 2, height)
    counter.set_roi_line(roi_start, roi_end)
    
    print(f"Default ROI line: {roi_start} to {roi_end}")
    
    # Mouse callback for setting ROI
    roi_points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
            if len(roi_points) == 2:
                counter.set_roi_line(roi_points[0], roi_points[1])
                roi_points.clear()
                print(f"New ROI line set: {roi_points[0]} to {roi_points[1]}")
    
    cv2.namedWindow('Footfall Counter - Webcam')
    cv2.setMouseCallback('Footfall Counter - Webcam', mouse_callback)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from webcam")
                break
            
            # Process frame
            processed_frame = counter.process_frame(frame)
            
            # Add instructions overlay
            cv2.putText(processed_frame, "Press 'q' to quit, 'r' to reset, 's' for new ROI", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Footfall Counter - Webcam', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.entry_count = 0
                counter.exit_count = 0
                counter.crossed_objects.clear()
                print("Counters reset!")
            elif key == ord('s'):
                roi_points.clear()
                print("Click two points to set new ROI line...")
            
            frame_count += 1
            
            # Print status every 100 frames
            if frame_count % 100 == 0:
                print(f"Frames processed: {frame_count}, Current counts - Entries: {counter.entry_count}, Exits: {counter.exit_count}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 50)
        print("DEMO COMPLETED!")
        print(f"Total frames processed: {frame_count}")
        print(f"Final Results:")
        print(f"  Entries: {counter.entry_count}")
        print(f"  Exits: {counter.exit_count}")
        print(f"  Total: {counter.entry_count + counter.exit_count}")
        print("=" * 50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-time Footfall Counter with Webcam')
    args = parser.parse_args()
    
    webcam_demo()


if __name__ == "__main__":
    main()

