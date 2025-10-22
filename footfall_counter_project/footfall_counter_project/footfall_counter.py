"""
Footfall Counter using Computer Vision
=====================================

A computer vision-based system that counts people entering and exiting through
a specific area using YOLO detection and DeepSORT tracking.

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.spatial import distance
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import argparse
import os
from typing import List, Tuple, Dict, Optional


class PersonTracker:
    """Tracks individual persons across frames using centroid tracking."""
    
    def __init__(self, max_disappeared: int = 30, max_distance: int = 50):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trajectories = defaultdict(list)
    
    def register(self, centroid):
        """Register a new object."""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.trajectories[self.next_id] = [centroid]
        self.next_id += 1
    
    def deregister(self, object_id):
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.trajectories:
            del self.trajectories[object_id]
    
    def update(self, detections):
        """Update tracker with new detections."""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Compute centroids for new detections
        input_centroids = np.array([det['centroid'] for det in detections])
        
        if len(self.objects) == 0:
            # No existing objects, register all detections
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Match existing objects to new detections
            object_centroids = np.array(list(self.objects.values()))
            object_ids = list(self.objects.keys())
            
            # Compute distance matrix
            D = distance.cdist(object_centroids, input_centroids)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(input_centroids[col])
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            # If we have more objects than detections, mark unmatched objects as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                self.register(input_centroids[col])
        
        return self.objects


class FootfallCounter:
    """Main class for footfall counting system."""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the footfall counter.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for person detection
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracker = PersonTracker()
        self.roi_line = None
        self.entry_count = 0
        self.exit_count = 0
        self.crossed_objects = set()
        self.frame_count = 0
        
    def set_roi_line(self, start_point: Tuple[int, int], end_point: Tuple[int, int]):
        """Set the region of interest line for counting."""
        self.roi_line = (start_point, end_point)
        print(f"ROI line set from {start_point} to {end_point}")
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in the frame using YOLO.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection dictionaries with bbox, confidence, and centroid
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if it's a person (class 0 in COCO dataset)
                    if int(box.cls) == 0 and float(box.conf) >= self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf)
                        
                        # Calculate centroid
                        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'centroid': centroid
                        })
        
        return detections
    
    def has_crossed_line(self, trajectory: List[Tuple[int, int]], 
                        line_start: Tuple[int, int], 
                        line_end: Tuple[int, int]) -> Optional[str]:
        """
        Check if a trajectory has crossed the ROI line.
        
        Args:
            trajectory: List of (x, y) points representing the object's path
            line_start: Start point of the ROI line
            line_end: End point of the ROI line
            
        Returns:
            'entry' if crossed in one direction, 'exit' if crossed in opposite direction, None if no crossing
        """
        if len(trajectory) < 2:
            return None
        
        # Calculate line equation: ax + by + c = 0
        x1, y1 = line_start
        x2, y2 = line_end
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        crossings = []
        
        for i in range(len(trajectory) - 1):
            x, y = trajectory[i]
            x_next, y_next = trajectory[i + 1]
            
            # Calculate which side of the line each point is on
            side_current = a * x + b * y + c
            side_next = a * x_next + b * y_next + c
            
            # Check if line was crossed
            if (side_current > 0 and side_next < 0) or (side_current < 0 and side_next > 0):
                # Determine direction based on the line orientation
                # For a vertical line, check horizontal movement
                if abs(x2 - x1) < abs(y2 - y1):  # More vertical line
                    direction = 'entry' if x_next > x else 'exit'
                else:  # More horizontal line
                    direction = 'entry' if y_next < y else 'exit'
                crossings.append(direction)
        
        if crossings:
            # Return the most recent crossing direction
            return crossings[-1]
        
        return None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for person detection, tracking, and counting.
        
        Args:
            frame: Input frame
            
        Returns:
            Annotated frame with bounding boxes, trajectories, and counts
        """
        self.frame_count += 1
        
        # Detect persons
        detections = self.detect_persons(frame)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Draw detections and tracking info
        annotated_frame = frame.copy()
        
        # Draw ROI line if set
        if self.roi_line:
            cv2.line(annotated_frame, self.roi_line[0], self.roi_line[1], (0, 255, 0), 2)
            cv2.putText(annotated_frame, "ROI Line", 
                       (self.roi_line[0][0], self.roi_line[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw bounding boxes and track trajectories
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"Person {confidence:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Check for line crossings and update counts
        for object_id, centroid in tracked_objects.items():
            if object_id in self.tracker.trajectories:
                trajectory = self.tracker.trajectories[object_id]
                
                if self.roi_line and len(trajectory) > 1:
                    crossing = self.has_crossed_line(trajectory, self.roi_line[0], self.roi_line[1])
                    
                    if crossing and object_id not in self.crossed_objects:
                        self.crossed_objects.add(object_id)
                        if crossing == 'entry':
                            self.entry_count += 1
                        elif crossing == 'exit':
                            self.exit_count += 1
                
                # Draw trajectory
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        cv2.line(annotated_frame, trajectory[i-1], trajectory[i], (0, 0, 255), 2)
        
        # Draw count information
        cv2.putText(annotated_frame, f"Entries: {self.entry_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Exits: {self.exit_count}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Total: {self.entry_count + self.exit_count}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame
    
    def process_video(self, input_path: str, output_path: str = None, 
                     roi_points: Tuple[Tuple[int, int], Tuple[int, int]] = None):
        """
        Process a video file for footfall counting.
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video (optional)
            roi_points: Tuple of (start_point, end_point) for ROI line
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set ROI line if provided
        if roi_points:
            self.set_roi_line(roi_points[0], roi_points[1])
        else:
            # Default ROI line (middle of frame)
            self.set_roi_line((width//2, 0), (width//2, height))
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {input_path}")
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        print(f"ROI line: {self.roi_line}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Write frame if output specified
            if out:
                out.write(processed_frame)
            
            # Display frame (optional)
            cv2.imshow('Footfall Counter', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processed {frame_count} frames. Current counts - Entries: {self.entry_count}, Exits: {self.exit_count}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Final counts - Entries: {self.entry_count}, Exits: {self.exit_count}")
        print(f"Total footfall: {self.entry_count + self.exit_count}")


def main():
    """Main function to run the footfall counter."""
    parser = argparse.ArgumentParser(description='Footfall Counter using Computer Vision')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path (optional)')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--roi', nargs=4, type=int, help='ROI line coordinates (x1 y1 x2 y2)')
    
    args = parser.parse_args()
    
    # Initialize footfall counter
    counter = FootfallCounter(model_path=args.model, confidence_threshold=args.confidence)
    
    # Set ROI if provided
    roi_points = None
    if args.roi:
        roi_points = ((args.roi[0], args.roi[1]), (args.roi[2], args.roi[3]))
    
    # Process video
    try:
        counter.process_video(args.input, args.output, roi_points)
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    main()

