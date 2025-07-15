from ultralytics import YOLO
import os
import cv2

# Path to the trained model (update if needed)
MODEL_PATH = 'parking_detection_model.pt'  # or 'teeth_detection_model.pt' if that's your latest
OUTPUT_DIR = 'video_results'
VIDEO_PATH = 'videoblocks-631eb67b17bed31caaa1af84_by-7cr2eo_1080__D.mp4'  # Set your video path here

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

# Class names from dataset/data.yaml
CLASS_NAMES = ['empty', 'occupied']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Run inference on the frame
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    cls = boxes.cls.cpu().numpy().astype(int) if boxes is not None and boxes.cls is not None else []
    xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and boxes.xyxy is not None else []
    
    # Count each class
    occupied_count = (cls == 1).sum() if len(cls) > 0 else 0
    empty_count = (cls == 0).sum() if len(cls) > 0 else 0

    # Draw thick bounding boxes and labels
    annotated_frame = frame.copy()
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box[:4].astype(int)
        class_id = cls[i]
        color = (0, 0, 255) if class_id == 1 else (0, 255, 0)  # green for occupied, red for empty
        label = CLASS_NAMES[class_id]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness=1)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

    # Overlay counts
    info_text = f"Occupied: {occupied_count}  Empty: {empty_count}"
    cv2.putText(annotated_frame, info_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4, cv2.LINE_AA)

    # Resize for live preview
    preview_frame = cv2.resize(annotated_frame, (1280, 720))
    # Show live preview
    cv2.imshow('Live Preview', preview_frame)
    # Write frame to output video
    out.write(annotated_frame)
    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video: {VIDEO_PATH}\nAnnotated video saved at: {output_path}") 