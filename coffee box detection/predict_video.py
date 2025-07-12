from ultralytics import YOLO
import os
import cv2
import numpy as np
import math

# Path to the trained model (update if needed)
MODEL_PATH = 'coffee_box_detection_model.pt'  # or 'teeth_detection_model.pt' if that's your latest
OUTPUT_DIR = 'video_results'
VIDEO_PATH = '6444196-uhd_3840_2160_24fps.mp4'  # Set your video path here

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained model
model = YOLO(MODEL_PATH)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

# Output video writer (optional, comment out if not needed)
output_path = os.path.join(OUTPUT_DIR, 'annotated', 'annotated_video.mp4')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ROI Definition - Set your ROI here (x1, y1, x2, y2)
# Example: ROI covers the middle 40% of the frame
ROI_RATIO = 0.3  # 30% margin on each side
roi_x1 = int(width * ROI_RATIO)
roi_y1 = int(height * ROI_RATIO)
roi_x2 = int(width * (1 - ROI_RATIO))
roi_y2 = int(height * (1 - ROI_RATIO))

# --- ROI Helper Functions ---
def box_iou(boxA, boxB):
    """Compute IoU between two boxes: boxA and boxB = (x1, y1, x2, y2)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def draw_roi(frame, x1, y1, x2, y2):
    """Draw ROI rectangle on frame"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(frame, "ROI - Counting Zone", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def is_in_roi(cx, cy, roi_x1, roi_y1, roi_x2, roi_y2):
    return roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2

# --- Main Loop ---
frame_count = 0
package_count = 0
next_object_id = 0
tracked_objects = {}  # object_id: {'centroid': (x, y), 'in_roi': bool, 'missed': int}
MAX_MISSED = 5  # Remove object if not seen for this many frames
DIST_THRESHOLD = 50  # Pixel distance threshold for matching centroids

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # Run inference on the frame
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    dets = []
    if boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else np.ones(len(xyxy))
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            score = conf[i]
            dets.append([x1, y1, x2, y2, score])
    dets = np.array(dets)
    # Get annotated frame
    annotated_frame = results[0].plot()
    # Draw ROI
    draw_roi(annotated_frame, roi_x1, roi_y1, roi_x2, roi_y2)
    # Prepare for new frame
    updated_ids = set()
    detections_centroids = []
    for det in dets:
        x1, y1, x2, y2, score = det
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        detections_centroids.append({'centroid': (cx, cy), 'bbox': (int(x1), int(y1), int(x2), int(y2))})
    # Match detections to tracked objects
    for det in detections_centroids:
        cx, cy = det['centroid']
        min_dist = float('inf')
        matched_id = None
        for obj_id, obj in tracked_objects.items():
            dist = euclidean_distance((cx, cy), obj['centroid'])
            if dist < DIST_THRESHOLD and dist < min_dist:
                min_dist = dist
                matched_id = obj_id
        in_roi_now = is_in_roi(cx, cy, roi_x1, roi_y1, roi_x2, roi_y2)
        if matched_id is not None:
            # Update tracked object
            obj = tracked_objects[matched_id]
            # Entry event: outside -> inside
            if not obj['in_roi'] and in_roi_now:
                package_count += 1
                print(f"Frame {frame_count}: Object {matched_id} entered ROI at ({cx},{cy})! Total: {package_count}")
            tracked_objects[matched_id] = {'centroid': (cx, cy), 'in_roi': in_roi_now, 'missed': 0}
            updated_ids.add(matched_id)
            obj_id_to_draw = matched_id
        else:
            # New object
            tracked_objects[next_object_id] = {'centroid': (cx, cy), 'in_roi': in_roi_now, 'missed': 0}
            obj_id_to_draw = next_object_id
            next_object_id += 1
        # Draw annotation
        x1, y1, x2, y2 = det['bbox']
        if in_roi_now:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated_frame, f"IN ROI ID:{obj_id_to_draw}", (x1, y2+25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        print(f"Frame {frame_count}: Object {obj_id_to_draw} Centroid=({cx},{cy}), In ROI={in_roi_now}")
    # Increment missed count for objects not updated
    for obj_id in list(tracked_objects.keys()):
        if obj_id not in updated_ids:
            tracked_objects[obj_id]['missed'] += 1
            if tracked_objects[obj_id]['missed'] > MAX_MISSED:
                del tracked_objects[obj_id]
    # Draw bold title on the frame
    title = "COFFEE PACKAGE DETECTION"
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (0, 0)]:
        cv2.putText(annotated_frame, title, (30+dx, 80+dy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(annotated_frame, title, (30, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6, cv2.LINE_AA)
    # Display statistics on the left side
    stats_y = 150
    line_height = 40
    cv2.putText(annotated_frame, f"Total packages counted: {package_count}", 
               (30, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"ROI: ({roi_x1},{roi_y1}) to ({roi_x2},{roi_y2})", 
               (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    # Show preview
    preview_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow('Live Preview', preview_frame)
    # Write to output video
    out.write(annotated_frame)
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video: {VIDEO_PATH}")
print(f"Annotated video saved at: {output_path}")
print(f"Total packages counted (passed through ROI): {package_count}")
print(f"ROI coordinates: ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2})")