from ultralytics import YOLO
import os
import cv2
import numpy as np

# Path to the trained model (update if needed)
MODEL_PATH = 'pothole_detection_model.pt'  # or 'teeth_detection_model.pt' if that's your latest
OUTPUT_DIR = 'video_results'
VIDEO_PATH = 'video_pothole.mp4'  # Set your video path here

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained model
model = YOLO(MODEL_PATH)

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Run inference on the frame
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    annotated_frame = frame.copy()
    num_boxes = len(boxes) if boxes is not None else 0
    thickness = 1  # Set your desired thickness
    # Use a single color for all boxes (e.g., green)
    color = (0, 255, 0)  # Green in BGR
    if boxes is not None:
        for i, box in enumerate(boxes):
            b = box.xyxy[0].cpu().numpy().astype(int)
            # Draw rectangle
            cv2.rectangle(annotated_frame, (b[0], b[1]), (b[2], b[3]), color, thickness)
            # Optionally, draw label and confidence
            if hasattr(box, 'cls') and hasattr(box, 'conf'):
                label = str(int(box.cls[0].cpu().numpy()))
                conf = box.conf[0].cpu().numpy()
                text = f"{label}: {conf:.2f}"
                cv2.putText(annotated_frame, text, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # Resize for live preview
    preview_frame = cv2.resize(annotated_frame, (1280, 720))
    # Show live preview
    cv2.imshow('Live Preview', preview_frame)
    # Write frame to output video
    out.write(annotated_frame)
    # Press 'q' to quit early
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video: {VIDEO_PATH}\nAnnotated video saved at: {output_path}") 