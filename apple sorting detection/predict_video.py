from ultralytics import YOLO
import os
import cv2

# Path to the trained model (update if needed)
MODEL_PATH = 'apple_sorting_detection_model.pt'
OUTPUT_DIR = 'video_results'
VIDEO_PATH = 'istockphoto-1497112092-640-adpp-is_PxOZ76nq.mp4'

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

BOX_THICKNESS = 1 # Set your desired box thickness

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Run inference on the frame
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results[0].names[cls] if hasattr(results[0], 'names') else str(cls)
        color = (0, 255, 0)  # Green box, you can change this
        # Draw rectangle with thickness
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        # Draw label
        label_text = f"{label} {conf:.2f}"
        cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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