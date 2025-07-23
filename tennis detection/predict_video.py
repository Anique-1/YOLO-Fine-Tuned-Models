from ultralytics import YOLO
import os
import cv2
import numpy as np

# Paths to the trained models
MODEL_PATH_1 = 'tennis_ball&player_detection_model.pt'  # Model for ball and player
MODEL_PATH_2 = 'tenniscourt_detection_model.pt'        # Model for tennis court
OUTPUT_DIR = 'video_results'
VIDEO_PATH = 'videoplayback (1).mp4'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained models
model1 = YOLO(MODEL_PATH_1)  # Ball and player detection
model2 = YOLO(MODEL_PATH_2)  # Tennis court detection

# Define colors for each class
CLASS_COLORS = {
    'ball': (0, 255, 255),       # Yellow for ball
    'player': (0, 0, 255),       # Red for player
    'tennis-court': (255, 0, 0)  # Blue for tennis court
}

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
output_path = os.path.join(OUTPUT_DIR, 'annotated', 'annotated_video.mp4')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

BOX_THICKNESS = 1  # Reduced box thickness

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame with both models
    results1 = model1(frame, verbose=False)  # Ball and player
    results2 = model2(frame, verbose=False)  # Tennis court
    annotated_frame = frame.copy()

    # Process results from model1 (ball and player)
    for box in results1[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results1[0].names[cls] if hasattr(results1[0], 'names') else str(cls)
        color = CLASS_COLORS.get(label, (0, 255, 0))  # Default to green if class not found
        # Draw rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        # Draw label
        label_text = f"{label} {conf:.2f}"
        cv2.putText(annotated_frame, label_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Process results from model2 (tennis court)
    for box in results2[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results2[0].names[cls] if hasattr(results2[0], 'names') else str(cls)
        color = CLASS_COLORS.get(label, (0, 255, 0))  # Default to green if class not found
        # Draw rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        # Draw label
        label_text = f"{label} {conf:.2f}"
        cv2.putText(annotated_frame, label_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Resize for live preview
    preview_frame = cv2.resize(annotated_frame, (1280, 720))
    # Show live preview
    cv2.imshow('Live Preview', preview_frame)
    # Write frame to output video
    out.write(annotated_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video: {VIDEO_PATH}\nAnnotated video saved at: {output_path}")