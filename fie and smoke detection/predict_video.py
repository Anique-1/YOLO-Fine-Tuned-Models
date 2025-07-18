from ultralytics import YOLO
import os
import cv2
import numpy as np

# Path to the trained segmentation model
MODEL_PATH = 'smoke_fire_detection_model.pt'  # segmentation-only model
OUTPUT_DIR = 'video_results'
VIDEO_PATH = 'istock video.mp4'  # Set your video path here

# Class names and colors (from dataset/data.yaml)
CLASS_NAMES = ['fire', 'smoke']
CLASS_COLORS = [(0, 0, 255), (255, 0, 0)]  # fire: red, smoke: blue (BGR for OpenCV)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained segmentation model
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
    # Run segmentation inference on the frame
    results = model(frame, verbose=False)
    r = results[0]
    annotated_frame = frame.copy()
    if hasattr(r, 'masks') and r.masks is not None:
        masks = r.masks.data.cpu().numpy()  # shape: (num_instances, H, W)
        classes = r.boxes.cls.cpu().numpy().astype(int)  # class indices for each mask
        for mask, cls_idx in zip(masks, classes):
            color = CLASS_COLORS[cls_idx]
            mask = (mask > 0.5).astype(np.uint8)
            # Resize mask to match frame size
            mask_resized = cv2.resize(mask, (annotated_frame.shape[1], annotated_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            colored_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = mask_resized * color[c]
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0)
            # Find centroid of the mask for label placement
            moments = cv2.moments(mask_resized)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                label = CLASS_NAMES[cls_idx]
                cv2.putText(
                    annotated_frame,
                    label,
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),  # White text for contrast
                    2,
                    cv2.LINE_AA
                )
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