from ultralytics import YOLO
import os
import cv2

# Path to the trained model (update if needed)
MODEL_PATH = 'weapon_detection.pt'  # or 'teeth_detection_model.pt' if that's your latest
OUTPUT_DIR = 'video_results'
VIDEO_PATH = '3805926-uhd_4096_2160_25fps.mp4'  # Set your video path here

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
    annotated_frame = results[0].plot()
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