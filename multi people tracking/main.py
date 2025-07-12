import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
import time
import math
from sklearn.cluster import DBSCAN

class PersonTracker:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5, iou_threshold=0.7):
        """
        Initialize the person tracking system
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for tracking
        """
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Tracking parameters
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Track history for each person
        self.track_history = defaultdict(list)
        self.track_colors = {}
        self.next_color_idx = 0
        
        # Color palette for different tracks
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 0),
            (255, 192, 203), (64, 224, 208), (255, 20, 147), (0, 191, 255)
        ]
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def get_color_for_track(self, track_id):
        """Get consistent color for each track ID"""
        if track_id not in self.track_colors:
            self.track_colors[track_id] = self.colors[self.next_color_idx % len(self.colors)]
            self.next_color_idx += 1
        return self.track_colors[track_id]
    
    def calculate_distance(self, box1, box2):
        """Calculate distance between two bounding box centers"""
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def cluster_groups(self, frame, boxes, track_ids):
        """Cluster people into groups based on proximity using DBSCAN"""
        if len(boxes) == 0:
            return []
        # Use the center points for clustering
        centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes])
        # DBSCAN parameters: eps is the max distance for a group, min_samples is min people in a group
        db = DBSCAN(eps=100, min_samples=2).fit(centers)
        labels = db.labels_  # -1 means noise (not in any group)
        groups = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:
                groups[label].append(idx)
        return groups
    
    def draw_tracking_info(self, frame, results):
        """Draw bounding boxes, track IDs, trails, and group bounding boxes"""
        annotated_frame = frame.copy()
        group_boxes = []
        group_ids = []
        # Process tracking results
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            # Filter for person class (class 0 in COCO dataset)
            person_mask = classes == 0
            boxes = boxes[person_mask]
            track_ids = track_ids[person_mask]
            confidences = confidences[person_mask]
            # --- GROUP TRACKING ---
            groups = self.cluster_groups(annotated_frame, boxes, track_ids)
            for group_id, indices in groups.items():
                # Get all box corners in the group
                group_points = []
                for idx in indices:
                    x1, y1, x2, y2 = boxes[idx]
                    group_points.extend([(x1, y1), (x2, y2)])
                group_points = np.array(group_points, dtype=np.int32)
                # Draw convex hull around group
                if len(group_points) >= 3:
                    hull = cv2.convexHull(group_points)
                    cv2.polylines(annotated_frame, [hull], isClosed=True, color=(0, 255, 255), thickness=3)
                    # Label group
                    M = cv2.moments(hull)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(annotated_frame, f"Group {group_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
            # --- END GROUP TRACKING ---
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                if conf < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box)
                color = self.get_color_for_track(track_id)
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # Update track history
                self.track_history[track_id].append((center_x, center_y))
                if len(self.track_history[track_id]) > 30:  # Keep last 30 points
                    self.track_history[track_id].pop(0)
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                # Draw track ID and confidence
                label = f"Person {track_id}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # Draw center point
                cv2.circle(annotated_frame, (center_x, center_y), 4, color, -1)
                # Draw tracking trail
                if len(self.track_history[track_id]) > 1:
                    trail_points = np.array(self.track_history[track_id], np.int32)
                    for i in range(1, len(trail_points)):
                        # Fade the trail
                        alpha = i / len(trail_points)
                        trail_color = tuple(int(c * alpha) for c in color)
                        cv2.line(annotated_frame, tuple(trail_points[i-1]), 
                               tuple(trail_points[i]), trail_color, 2)
        return annotated_frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_info_panel(self, frame):
        """Draw information panel with statistics"""
        height, width = frame.shape[:2]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text information
        info_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Active Tracks: {len(self.track_history)}",
            f"Device: {self.device.upper()}",
            f"Confidence Threshold: {self.confidence_threshold}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def process_video(self, video_path):
        """
        Process video file for real-time tracking
        
        Args:
            video_path: Path to video file or camera index (0 for webcam)
        """
        # Open video capture
        if isinstance(video_path, int) or video_path.isdigit():
            cap = cv2.VideoCapture(int(video_path))
            print(f"Opening camera {video_path}")
        else:
            cap = cv2.VideoCapture(video_path)
            print(f"Opening video file: {video_path}")
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Set video properties for better performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Starting real-time tracking...")
        print("Press 'q' to quit, 'r' to reset tracking history")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_path, str) and not video_path.isdigit():
                        # If it's a video file, loop it
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Failed to read frame")
                        break
                
                frame_count += 1
                
                # Run YOLO tracking
                results = self.model.track(
                    frame, 
                    persist=True, 
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    classes=[0],  # Only detect persons
                    device=self.device,
                    verbose=False
                )
                
                # Draw tracking information
                annotated_frame = self.draw_tracking_info(frame, results)
                
                # Update and draw FPS
                self.update_fps()
                self.draw_info_panel(annotated_frame)
                
                # Display frame
                cv2.imshow('Real-time Multi-Person Tracking', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset tracking history
                    self.track_history.clear()
                    self.track_colors.clear()
                    self.next_color_idx = 0
                    print("Tracking history reset")
                
                # Optional: Process every nth frame for better performance
                # if frame_count % 2 == 0:  # Process every 2nd frame
                #     continue
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Tracking stopped")

def main():
    """Main function to run the tracking system"""
    # Initialize tracker
    tracker = PersonTracker(
        model_path='yolov8n.pt',  # Use yolov8s.pt or yolov8m.pt for better accuracy
        confidence_threshold=0.5,
        iou_threshold=0.7
    )
    
    # Video source options:
    # For webcam: use 0, 1, 2, etc.
    # For video file: use file path
    video_source = "853889-hd_1920_1080_25fps.mp4"  # Change this to your video path
    # video_source = 0  # Uncomment for webcam
    
    # Start tracking
    tracker.process_video(video_source)

if __name__ == "__main__":
    # Check requirements
    print("Checking system requirements...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    main()